// Unit tests for the OFlash engine pieces that need no GPU:
//   - capture ring format + SPSC semantics (wrap, PAD, drop-on-full)
//   - golden-bytes contract shared with the Python reader
//   - acceptance guard state machine (promote / rollback / backoff / disable)
//   - adapter safetensors parsing + refusal rules
//   - expected LoRA tensor specs
//
// Ring tests are POSIX-only (shm_open); they self-skip on Windows.

#include "CppUnitTestFramework.hpp"

#include "common/oflash/oflash_adapter.h"
#include "common/oflash/oflash_config.h"
#include "common/oflash/oflash_format.h"
#include "common/oflash/oflash_guard.h"
#include "common/oflash/oflash_ring.h"
#include "common/oflash/oflash_runtime.h"
#include "common/oflash/oflash_supervisor.h"
#include "internal.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using namespace dflash::common;
using namespace dflash::common::oflash;

struct OFlashUnitFixture {};

enum class AdapterPayloadPoison { None, NaN, Inf };

// Minimal DraftWeights dims for spec/adapter tests (no tensors needed).
DraftWeights tiny_drafter() {
    DraftWeights dw;
    dw.n_layer = 2;
    dw.n_head = 4;
    dw.n_head_kv = 2;
    dw.head_dim = 8;       // q_dim=32, kv_dim=16
    dw.n_embd = 16;
    dw.n_ff = 24;
    dw.n_target_layers = 3;  // fc_in=48
    return dw;
}

// Build a tiny valid adapter file for tiny_drafter() at the given path.
void write_adapter_file(const std::string & path,
                        const DraftWeights & dw,
                        int rank,
                        const std::string & sha,
                        uint64_t generation,
                        bool corrupt_shape = false,
                        const std::string & alpha = "32",
                        AdapterPayloadPoison poison = AdapterPayloadPoison::None) {
    const auto specs = oflash_lora_expected_tensors(dw, rank);
    nlohmann::json header;
    header["__metadata__"] = {
        {"oflash.format", "1"},
        {"oflash.drafter_sha256", sha},
        {"oflash.rank", std::to_string(rank)},
        {"oflash.alpha", alpha},
        {"oflash.generation", std::to_string(generation)},
        {"oflash.profile", "default"},
    };
    uint64_t offset = 0;
    std::vector<float> data;
    for (const auto & s : specs) {
        int64_t rows = s.out_dim, cols = s.in_dim;
        if (corrupt_shape && s.name == "blk.0.attn_q.lora_a") rows += 1;
        const uint64_t bytes = (uint64_t)rows * (uint64_t)cols * 4;
        header[s.name] = {
            {"dtype", "F32"},
            {"shape", {rows, cols}},
            {"data_offsets", {offset, offset + bytes}},
        };
        for (int64_t i = 0; i < rows * cols; i++) {
            if (data.empty() && poison == AdapterPayloadPoison::NaN) {
                data.push_back(std::numeric_limits<float>::quiet_NaN());
            } else if (data.empty() && poison == AdapterPayloadPoison::Inf) {
                data.push_back(std::numeric_limits<float>::infinity());
            } else {
                data.push_back(0.001f * (float)(data.size() % 97));
            }
        }
        offset += bytes;
    }
    const std::string hjson = header.dump();
    const uint64_t hlen = hjson.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char *)&hlen, 8);
    f.write(hjson.data(), (std::streamsize)hjson.size());
    f.write((const char *)data.data(), (std::streamsize)(data.size() * 4));
}

std::string test_sha() {
    return std::string(64, 'a');  // 64 hex chars
}

}  // namespace

// ── Format invariants ───────────────────────────────────────────────

TEST_CASE(OFlashUnitFixture, format_header_layout_is_stable) {
    CHECK_EQUAL(sizeof(OFlashRingHeader), (size_t)256);
    CHECK_EQUAL(offsetof(OFlashRingHeader, head), (size_t)64);
    CHECK_EQUAL(offsetof(OFlashRingHeader, tail), (size_t)128);
    CHECK_EQUAL(offsetof(OFlashRingHeader, dropped_records), (size_t)192);
    CHECK_EQUAL(offsetof(OFlashRingHeader, drafter_hash), (size_t)208);
    CHECK_EQUAL(sizeof(OFlashRecordHeader), (size_t)32);
}

TEST_CASE(OFlashUnitFixture, hash_from_hex_truncates_big_endian) {
    CHECK_EQUAL(oflash_hash_from_hex("0123456789abcdef0000"),
                (uint64_t)0x0123456789abcdefull);
    CHECK_EQUAL(oflash_hash_from_hex("ABCDEF0102030405zzz"),
                (uint64_t)0xabcdef0102030405ull);
    CHECK_EQUAL(oflash_hash_from_hex("shorty"), (uint64_t)0);
    CHECK_EQUAL(oflash_hash_from_hex(nullptr), (uint64_t)0);
}

TEST_CASE(OFlashUnitFixture, conservative_runtime_defaults) {
    OFlashConfig cfg;
    CHECK_EQUAL(cfg.ring_mb, 512);
    CHECK_EQUAL(cfg.topk, 8);
    CHECK_EQUAL(cfg.backfill_rows, 128);
    CHECK_EQUAL(cfg.dtype, std::string("auto"));
}

TEST_CASE(OFlashUnitFixture, adapter_generations_must_advance_resident_and_pending) {
    CHECK(oflash_generation_is_newer(8, 7));
    CHECK(!oflash_generation_is_newer(7, 7));
    CHECK(!oflash_generation_is_newer(6, 7));
    CHECK(oflash_generation_is_newer(9, 7, 8));
    CHECK(!oflash_generation_is_newer(8, 7, 8));
    // A rollback may lower the resident generation, but must not make a
    // previously seen/rejected generation eligible again.
    CHECK(!oflash_generation_is_newer(8, 7, 0, 8));
    CHECK(oflash_generation_is_newer(9, 7, 0, 8));
}

TEST_CASE(OFlashUnitFixture, runtime_refuses_incompatible_feature_width_before_gpu_init) {
    OFlashRuntime runtime;
    OFlashConfig cfg;
    DraftWeights dw = tiny_drafter();
    CHECK(!runtime.init(cfg, "/tmp/target.gguf", "/tmp/draft.gguf",
                        dw, /*draft_backend=*/nullptr,
                        /*target_capture_layers=*/dw.n_target_layers + 1,
                        /*target_hidden=*/dw.n_embd,
                        /*vocab=*/64));
    CHECK(dw.oflash == nullptr);
}

// ── Ring semantics ──────────────────────────────────────────────────

#if !defined(_WIN32)

namespace {

// Reader-side view of the shared segment for test assertions.
struct RingView {
    OFlashRingHeader * hdr;
    uint8_t * data;
    explicit RingView(OFlashRing & r) {
        // The producer maps the segment; re-derive pointers via /dev/shm.
        path = "/dev/shm" + r.name();
        f.open(path, std::ios::binary | std::ios::in | std::ios::out);
        buf.resize((size_t)sizeof(OFlashRingHeader));
        f.read((char *)buf.data(), (std::streamsize)buf.size());
        hdr = (OFlashRingHeader *)buf.data();
        data = nullptr;
    }
    std::string path;
    std::fstream f;
    std::vector<uint8_t> buf;
    // Re-read the header from disk (shm file) after producer activity.
    void refresh() {
        f.seekg(0);
        f.read((char *)buf.data(), (std::streamsize)buf.size());
    }
    // Read `n` payload bytes at logical offset L.
    std::vector<uint8_t> read_at(uint64_t logical, size_t n) {
        std::vector<uint8_t> out(n);
        const uint64_t off = hdr->data_offset + (logical % hdr->capacity);
        f.seekg((std::streamoff)off);
        f.read((char *)out.data(), (std::streamsize)n);
        return out;
    }
    void set_tail(uint64_t v) {
        f.seekp((std::streamoff)offsetof(OFlashRingHeader, tail));
        f.write((const char *)&v, 8);
        f.flush();
    }
};

}  // namespace

TEST_CASE(OFlashUnitFixture, ring_create_publishes_stream_facts) {
    OFlashRing ring;
    const std::string name =
        "/oflash-test-" + std::to_string((long)getpid()) + "-a";
    REQUIRE(ring.create(name, 1 << 20, 0x1122334455667788ull, 5, 5120, 16,
                        32, 248320));
    RingView v(ring);
    CHECK_EQUAL(v.hdr->magic, OFLASH_RING_MAGIC);
    CHECK_EQUAL(v.hdr->version, OFLASH_RING_VERSION);
    CHECK_EQUAL(v.hdr->capacity, (uint64_t)(1 << 20));
    CHECK_EQUAL(v.hdr->data_offset, (uint64_t)256);
    CHECK_EQUAL(v.hdr->drafter_hash, (uint64_t)0x1122334455667788ull);
    CHECK_EQUAL(v.hdr->n_capture_layers, (uint32_t)5);
    CHECK_EQUAL(v.hdr->hidden, (uint32_t)5120);
    CHECK_EQUAL(v.hdr->block_size, (uint32_t)16);
    CHECK_EQUAL(v.hdr->topk, (uint32_t)32);
    CHECK_EQUAL(v.hdr->vocab, (uint32_t)248320);
    ring.close();
    // close() unlinks the segment.
    CHECK(::access(v.path.c_str(), F_OK) != 0);
}

TEST_CASE(OFlashUnitFixture, ring_create_removes_dead_engine_segment) {
    std::string stale_name;
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        stale_name = "/lucebox-oflash-" +
                     std::to_string((long)::getpid());
        const int fd = ::shm_open(stale_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
                                  0600);
        if (fd < 0) ::_exit(2);
        ::close(fd);
        ::_exit(0);  // deliberately leave the segment linked
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    stale_name = "/lucebox-oflash-" + std::to_string((long)child);
    const std::string stale_path = "/dev/shm" + stale_name;
    REQUIRE(::access(stale_path.c_str(), F_OK) == 0);

    OFlashRing ring;
    const std::string name =
        "/oflash-test-" + std::to_string((long)getpid()) + "-cleanup";
    REQUIRE(ring.create(name, 1 << 20, 1, 1, 4, 2, 0, 100));
    CHECK(::access(stale_path.c_str(), F_OK) != 0);
    ring.close();
}

TEST_CASE(OFlashUnitFixture, ring_push_roundtrips_records) {
    OFlashRing ring;
    const std::string name =
        "/oflash-test-" + std::to_string((long)getpid()) + "-b";
    REQUIRE(ring.create(name, 1 << 20, 1, 1, 4, 2, 0, 100));

    OFlashRecordHeader h{};
    h.type = OFLASH_REC_STEP;
    h.seq_id = 7;
    h.pos = 42;
    h.n_rows = 2;
    const uint8_t payload[13] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
    REQUIRE(ring.push(h, payload, sizeof(payload)));

    RingView v(ring);
    v.refresh();
    // header(32) + 13 bytes payload → padded to 48.
    CHECK_EQUAL(v.hdr->head, (uint64_t)48);
    auto rec = v.read_at(0, 48);
    OFlashRecordHeader got{};
    std::memcpy(&got, rec.data(), sizeof(got));
    CHECK_EQUAL(got.type, (uint32_t)OFLASH_REC_STEP);
    CHECK_EQUAL(got.size_bytes, (uint32_t)48);
    CHECK_EQUAL(got.seq_id, (uint64_t)7);
    CHECK_EQUAL(got.pos, 42);
    CHECK_EQUAL(got.n_rows, 2);
    CHECK(std::memcmp(rec.data() + 32, payload, sizeof(payload)) == 0);
    // Tail padding zeroed.
    for (size_t i = 32 + sizeof(payload); i < 48; i++) CHECK_EQUAL(rec[i], 0);
    ring.close();
}

TEST_CASE(OFlashUnitFixture, ring_drops_when_full_and_pads_at_wrap) {
    OFlashRing ring;
    const std::string name =
        "/oflash-test-" + std::to_string((long)getpid()) + "-c";
    // Tiny ring: 1 MiB min is enforced, so use 1 MiB and a big record.
    REQUIRE(ring.create(name, 1 << 20, 1, 1, 4, 2, 0, 100));

    std::vector<uint8_t> big((1 << 18) - 32);  // record = 256 KiB exactly
    OFlashRecordHeader h{};
    h.type = OFLASH_REC_CONTEXT;
    // 3 records fill 768 KiB.
    REQUIRE(ring.push(h, big.data(), big.size()));
    REQUIRE(ring.push(h, big.data(), big.size()));
    REQUIRE(ring.push(h, big.data(), big.size()));
    // A fourth fits exactly (1 MiB total).
    REQUIRE(ring.push(h, big.data(), big.size()));
    // Fifth must drop: consumer never advanced.
    CHECK(!ring.push(h, big.data(), big.size()));
    CHECK_EQUAL(ring.dropped(), (uint64_t)1);
    CHECK_EQUAL(ring.written(), (uint64_t)4);

    // Advance the consumer past 1.5 records; a 256 KiB record now needs a
    // wrap... head is at 1 MiB (buffer end) so it starts at offset 0 again —
    // no PAD needed. Free space = tail = 384 KiB ≥ 256 KiB.
    RingView v(ring);
    v.set_tail(3 * (1 << 17));  // 384 KiB
    REQUIRE(ring.push(h, big.data(), big.size()));
    v.refresh();
    CHECK_EQUAL(v.hdr->head, (uint64_t)(1 << 20) + (1 << 18));

    // Now force a mid-buffer wrap: push a half-size record (128 KiB) then
    // advance tail generously and push a 256 KiB record — it cannot fit in
    // the 128 KiB to the buffer end, so a PAD record covers the tail.
    std::vector<uint8_t> half((1 << 17) - 32);
    REQUIRE(ring.push(h, half.data(), half.size()));      // head: 1.375 MiB
    v.set_tail((1 << 20));                                 // plenty free
    REQUIRE(ring.push(h, big.data(), big.size()));
    v.refresh();
    // 0.375 MiB → wrap: PAD covers 0.625 MiB? No — PAD spans to buffer end:
    // head%cap = 0.375 MiB, to_end = 0.625 MiB > record 0.25 MiB → fits, no
    // pad. Verify head advanced by exactly one record.
    CHECK_EQUAL(v.hdr->head, (uint64_t)(1 << 20) + (1 << 18) + (1 << 17) +
                             (1 << 18));
    ring.close();
}

TEST_CASE(OFlashUnitFixture, ring_pad_record_written_at_wrap) {
    OFlashRing ring;
    const std::string name =
        "/oflash-test-" + std::to_string((long)getpid()) + "-d";
    REQUIRE(ring.create(name, 1 << 22, 1, 1, 4, 2, 0, 100));  // 4 MiB
    OFlashRecordHeader h{};
    h.type = OFLASH_REC_CONTEXT;
    // head → 2.5 MiB, fully drained.
    std::vector<uint8_t> r15((3 << 19) - 32);  // 1.5 MiB record
    std::vector<uint8_t> r10((1 << 20) - 32);  // 1.0 MiB record
    REQUIRE(ring.push(h, r15.data(), r15.size()));
    REQUIRE(ring.push(h, r10.data(), r10.size()));
    RingView v(ring);
    v.set_tail((3 << 19) + (1 << 20));  // 2.5 MiB consumed
    // 1.75 MiB record: only 1.5 MiB to the buffer end → PAD(1.5 MiB) + wrap.
    std::vector<uint8_t> r175((7 << 18) - 32);
    REQUIRE(ring.push(h, r175.data(), r175.size()));
    v.refresh();
    // head = 2.5 + 1.5 (PAD) + 1.75 = 5.75 MiB logical.
    CHECK_EQUAL(v.hdr->head,
                (uint64_t)(3 << 19) + (1 << 20) + (3 << 19) + (7 << 18));
    // The PAD record sits at logical 2.5 MiB with size 1.5 MiB.
    auto pad = v.read_at((3 << 19) + (1 << 20), sizeof(OFlashRecordHeader));
    OFlashRecordHeader ph{};
    std::memcpy(&ph, pad.data(), sizeof(ph));
    CHECK_EQUAL(ph.type, (uint32_t)OFLASH_REC_PAD);
    CHECK_EQUAL(ph.size_bytes, (uint32_t)(3 << 19));
    ring.close();
}

// Golden bytes shared with optimizations/oflash tests: one STEP record with
// known contents must serialize to exactly these bytes (little-endian).
TEST_CASE(OFlashUnitFixture, ring_step_record_golden_bytes) {
    OFlashRing ring;
    const std::string name =
        "/oflash-test-" + std::to_string((long)getpid()) + "-e";
    REQUIRE(ring.create(name, 1 << 20, 0x0102030405060708ull, 1, 2, 2, 0, 10));
    OFlashRecordHeader h{};
    h.type = OFLASH_REC_STEP;
    h.seq_id = 1;
    h.pos = 5;
    h.n_rows = 1;
    h.t_mono_ns = 0;  // pinned for the golden comparison
    const uint16_t feat[2] = {0x3f80, 0x4000};        // bf16 bits
    const uint8_t labels[24] = {
        2,0,0,0,  0,0,0,0,                            // n_labels=2, topk_k=0
        7,0,0,0,  9,0,0,0,                            // draft_tok
        11,0,0,0, 13,0,0,0,                           // target_tok
    };
    REQUIRE(ring.push2(h, feat, sizeof(feat), labels, sizeof(labels)));
    RingView v(ring);
    v.refresh();
    // 32 header + 4 feat + 24 labels = 60 → 64 padded.
    CHECK_EQUAL(v.hdr->head, (uint64_t)64);
    auto rec = v.read_at(0, 64);
    static const uint8_t kGolden[64] = {
        0x02,0,0,0, 64,0,0,0,                          // type, size
        1,0,0,0,0,0,0,0,                               // seq_id
        5,0,0,0, 1,0,0,0,                              // pos, n_rows
        0,0,0,0,0,0,0,0,                               // t_mono_ns
        0x80,0x3f, 0x00,0x40,                          // feat
        2,0,0,0, 0,0,0,0,                              // n_labels, topk_k
        7,0,0,0, 9,0,0,0,                              // draft
        11,0,0,0, 13,0,0,0,                            // target
        0,0,0,0                                        // pad
    };
    CHECK(std::memcmp(rec.data(), kGolden, sizeof(kGolden)) == 0);
    ring.close();
}

#endif  // !_WIN32

// ── Guard state machine ─────────────────────────────────────────────

TEST_CASE(OFlashUnitFixture, guard_promotes_on_equal_al) {
    OFlashGuardConfig cfg;
    cfg.probation_steps = 4;
    cfg.min_baseline_steps = 2;
    cfg.min_steps_between_swaps = 8;
    OFlashGuard g(cfg);
    for (int i = 0; i < 8; i++) {
        CHECK(g.record_step(5.0f) == OFlashGuardAction::None);
    }
    CHECK(g.can_swap());
    g.on_swap(1);
    CHECK(g.state() == OFlashGuard::State::Probation);
    CHECK(!g.can_swap());
    OFlashGuardAction a = OFlashGuardAction::None;
    for (int i = 0; i < 4; i++) a = g.record_step(5.0f);
    CHECK(a == OFlashGuardAction::Promote);
    CHECK(g.state() == OFlashGuard::State::Serving);
    CHECK_EQUAL(g.promotes(), (uint64_t)1);
    CHECK_EQUAL(g.current_generation(), (uint64_t)1);
}

TEST_CASE(OFlashUnitFixture, guard_rolls_back_on_regression_and_backs_off) {
    OFlashGuardConfig cfg;
    cfg.probation_steps = 4;
    cfg.min_baseline_steps = 2;
    cfg.min_steps_between_swaps = 4;
    OFlashGuard g(cfg);
    for (int i = 0; i < 8; i++) g.record_step(6.0f);
    const int backoff0 = g.swap_backoff();
    g.on_swap(1);
    OFlashGuardAction a = OFlashGuardAction::None;
    for (int i = 0; i < 4; i++) a = g.record_step(3.0f);  // clearly worse
    CHECK(a == OFlashGuardAction::Rollback);
    CHECK_EQUAL(g.rollbacks(), (uint64_t)1);
    CHECK_EQUAL(g.swap_backoff(), backoff0 * 2);
    // Baseline survives the failed probation.
    CHECK(g.baseline_al() > 5.0f);
}

TEST_CASE(OFlashUnitFixture, guard_disables_after_consecutive_rollbacks) {
    OFlashGuardConfig cfg;
    cfg.probation_steps = 2;
    cfg.min_baseline_steps = 1;
    cfg.min_steps_between_swaps = 1;
    cfg.max_consecutive_rollbacks = 3;
    OFlashGuard g(cfg);
    for (int i = 0; i < 4; i++) g.record_step(8.0f);
    OFlashGuardAction last = OFlashGuardAction::None;
    for (int r = 0; r < 3; r++) {
        // Backoff doubles each rollback; serve enough steps to allow a swap.
        while (!g.can_swap()) g.record_step(8.0f);
        g.on_swap((uint64_t)r + 1);
        for (int i = 0; i < 2; i++) last = g.record_step(1.0f);
    }
    CHECK(last == OFlashGuardAction::Disable);
    CHECK(g.state() == OFlashGuard::State::Disabled);
    // Disabled guard ignores further samples and refuses swaps.
    CHECK(g.record_step(9.0f) == OFlashGuardAction::None);
    CHECK(!g.can_swap());
}

TEST_CASE(OFlashUnitFixture, guard_auto_promotes_without_baseline) {
    OFlashGuardConfig cfg;
    cfg.probation_steps = 2;
    cfg.min_baseline_steps = 100;      // more evidence than we provide
    cfg.min_steps_between_swaps = 1;   // swap allowed almost immediately
    OFlashGuard g(cfg);
    g.record_step(2.0f);
    while (!g.can_swap()) g.record_step(2.0f);
    g.on_swap(1);
    OFlashGuardAction a = OFlashGuardAction::None;
    a = g.record_step(0.5f);
    a = g.record_step(0.5f);
    // No measurable baseline → the guard cannot veto; promotes.
    CHECK(a == OFlashGuardAction::Promote);
}

// ── Adapter files ───────────────────────────────────────────────────

TEST_CASE(OFlashUnitFixture, adapter_expected_tensor_specs) {
    DraftWeights dw = tiny_drafter();
    const auto specs = oflash_lora_expected_tensors(dw, 4);
    // fc pair + 6 pairs per layer × 2 layers = 13 pairs = 26 tensors.
    CHECK_EQUAL(specs.size(), (size_t)26);
    CHECK_EQUAL(specs[0].name, std::string("dflash.fc.lora_a"));
    CHECK_EQUAL(specs[0].in_dim, (int64_t)48);   // 3 * 16
    CHECK_EQUAL(specs[0].out_dim, (int64_t)4);
    CHECK_EQUAL(specs[1].name, std::string("dflash.fc.lora_b"));
    CHECK_EQUAL(specs[1].in_dim, (int64_t)4);
    CHECK_EQUAL(specs[1].out_dim, (int64_t)16);
    // Spot-check attn_output dims (in=q_dim=32, out=hidden=16).
    for (const auto & s : specs) {
        if (s.name == "blk.1.attn_output.lora_a") {
            CHECK_EQUAL(s.in_dim, (int64_t)32);
            CHECK_EQUAL(s.out_dim, (int64_t)4);
        }
        if (s.name == "blk.1.ffn_down.lora_b") {
            CHECK_EQUAL(s.in_dim, (int64_t)4);
            CHECK_EQUAL(s.out_dim, (int64_t)16);
        }
    }
}

TEST_CASE(OFlashUnitFixture, adapter_load_accepts_valid_file) {
    DraftWeights dw = tiny_drafter();
    const std::string path =
        "/tmp/oflash-test-adapter-" + std::to_string((long)getpid()) + ".st";
    write_adapter_file(path, dw, 4, test_sha(), 3);
    OFlashAdapterHost host;
    std::string err;
    REQUIRE(oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK_EQUAL(host.generation, (uint64_t)3);
    CHECK_EQUAL(host.tensors.size(), (size_t)26);
    // F32 → F16 conversion preserves element count.
    CHECK_EQUAL(host.tensors.at("dflash.fc.lora_a").size(), (size_t)(48 * 4));
    std::remove(path.c_str());
}

TEST_CASE(OFlashUnitFixture, adapter_load_refuses_non_finite_tensor_values) {
    DraftWeights dw = tiny_drafter();
    const std::string path =
        "/tmp/oflash-test-adapter-" + std::to_string((long)getpid()) + "n.st";
    OFlashAdapterHost host;
    std::string err;

    write_adapter_file(path, dw, 4, test_sha(), 1,
                       /*corrupt_shape=*/false, /*alpha=*/"32",
                       AdapterPayloadPoison::NaN);
    CHECK(!oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK(err.find("NaN/Inf") != std::string::npos);

    write_adapter_file(path, dw, 4, test_sha(), 1,
                       /*corrupt_shape=*/false, /*alpha=*/"32",
                       AdapterPayloadPoison::Inf);
    err.clear();
    CHECK(!oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK(err.find("NaN/Inf") != std::string::npos);
    std::remove(path.c_str());
}

TEST_CASE(OFlashUnitFixture, adapter_load_refuses_mismatches) {
    DraftWeights dw = tiny_drafter();
    const std::string path =
        "/tmp/oflash-test-adapter-" + std::to_string((long)getpid()) + "r.st";
    OFlashAdapterHost host;
    std::string err;

    // Wrong drafter hash.
    write_adapter_file(path, dw, 4, std::string(64, 'b'), 1);
    CHECK(!oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK(err.find("hash") != std::string::npos);

    // Wrong rank.
    write_adapter_file(path, dw, 4, test_sha(), 1);
    CHECK(!oflash_adapter_load(path, dw, 8, 32.0f, test_sha(), host, err));
    CHECK(err.find("rank") != std::string::npos);

    // Same rank but incompatible LoRA scale.
    write_adapter_file(path, dw, 4, test_sha(), 1,
                       /*corrupt_shape=*/false, /*alpha=*/"16");
    CHECK(!oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK(err.find("alpha") != std::string::npos);

    // Corrupt tensor shape.
    write_adapter_file(path, dw, 4, test_sha(), 1, /*corrupt_shape=*/true);
    CHECK(!oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK(err.find("shape") != std::string::npos);

    std::remove(path.c_str());
}

TEST_CASE(OFlashUnitFixture, adapter_load_refuses_malformed_field_types) {
    const std::string path =
        "/tmp/oflash-test-adapter-" + std::to_string((long)getpid()) + "j.st";
    nlohmann::json header;
    header["bad"] = {
        {"dtype", 7},
        {"shape", "not-an-array"},
        {"data_offsets", {0, 0}},
    };
    const std::string hjson = header.dump();
    const uint64_t hlen = hjson.size();
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char *)&hlen, 8);
        f.write(hjson.data(), (std::streamsize)hjson.size());
    }
    DraftWeights dw = tiny_drafter();
    OFlashAdapterHost host;
    std::string err;
    CHECK(!oflash_adapter_load(path, dw, 4, 32.0f, test_sha(), host, err));
    CHECK(err.find("invalid field types") != std::string::npos);
    std::remove(path.c_str());
}

// ── Supervisor process contract ─────────────────────────────────────

#if !defined(_WIN32)
TEST_CASE(OFlashUnitFixture, supervisor_spawns_and_relays_swap_events) {
    // Fake trainer: sends the int32 ready handshake on --stream-fd, then a
    // swap_ready event, echoes nothing else, exits on stdin EOF/quit.
    const std::string script =
        "/tmp/oflash-test-trainer-" + std::to_string((long)getpid()) + ".sh";
    {
        std::ofstream f(script);
        f << "#!/bin/bash\n"  // bash: variable-fd redirection >&$fd
             "fd=2\n"
             "for a in \"$@\"; do case \"$a\" in --stream-fd=*) "
             "fd=${a#--stream-fd=};; esac; done\n"
             // int32 0, little-endian.
             "printf '\\000\\000\\000\\000' >&$fd\n"
             "printf '%s\\n' "
             "'{\"event\":\"swap_ready\",\"path\":\"/tmp/a.st\","
             "\"generation\":7}' >&$fd\n"
             "while read line; do case \"$line\" in\n"
             "emit) printf '%s\\n' "
             "'{\"event\":\"swap_ready\",\"path\":\"/tmp/b.st\","
             "\"generation\":8}' >&$fd;;\n"
             "quit) exit 0;; esac; done\n";
    }
    ::chmod(script.c_str(), 0755);

    OFlashSupervisor sup;
    OFlashSupervisorConfig cfg;
    cfg.trainer_bin = script;
    cfg.drafter_path = "/nonexistent.gguf";
    cfg.args = {"--ring-name=/x"};
    cfg.ready_timeout_ms = 5000;
    sup.stop();  // stop-before-start must not poison the first launch
    REQUIRE(sup.start(cfg));
    CHECK(!sup.start(cfg));  // a live supervisor cannot be double-started

    OFlashPendingSwap swap;
    bool got = false;
    for (int i = 0; i < 100 && !(got = sup.take_pending_swap(swap)); i++) {
        ::usleep(50 * 1000);
    }
    REQUIRE(got);
    CHECK_EQUAL(swap.path, std::string("/tmp/a.st"));
    CHECK_EQUAL(swap.generation, (uint64_t)7);
    CHECK(sup.trainer_alive());
    sup.send_line("promote 7");
    sup.send_line("emit");
    bool cleared = false;
    for (int i = 0; i < 100 && !cleared; i++) {
        cleared = sup.clear_pending_swaps();
        if (!cleared) ::usleep(20 * 1000);
    }
    REQUIRE(cleared);
    CHECK(!sup.take_pending_swap(swap));
    sup.stop();  // graceful: quit → clean child exit → reap
    CHECK(!sup.trainer_alive());
    std::remove(script.c_str());
}

TEST_CASE(OFlashUnitFixture, supervisor_rollback_barrier_and_disabled_state) {
    // Events use one ordered stream. The candidate emitted after receipt of
    // rollback but before rollback_ack is stale and must be discarded; the
    // first post-ack candidate is eligible.
    const std::string script =
        "/tmp/oflash-barrier-trainer-" +
        std::to_string((long)getpid()) + ".sh";
    {
        std::ofstream f(script);
        f << "#!/bin/bash\n"
             "fd=2\n"
             "for a in \"$@\"; do case \"$a\" in --stream-fd=*) "
             "fd=${a#--stream-fd=};; esac; done\n"
             "printf '\\000\\000\\000\\000' >&$fd\n"
             "while read line; do case \"$line\" in\n"
             "rollback\\ *) gen=${line#rollback }; "
             "printf '%s\\n' "
             "'{\"event\":\"rollback_ack\",\"generation\":6}' >&$fd; "
             "printf '%s\\n' "
             "'{\"event\":\"swap_ready\",\"path\":\"/tmp/stale.st\","
             "\"generation\":8}' >&$fd; "
             "printf '{\"event\":\"rollback_ack\",\"generation\":%s}\\n' "
             "\"$gen\" >&$fd; "
             "printf '%s\\n' "
             "'{\"event\":\"swap_ready\",\"path\":\"/tmp/fresh.st\","
             "\"generation\":9}' >&$fd;;\n"
             "self-disable) printf '%s\\n' "
             "'{\"event\":\"training_disabled\","
             "\"reason\":\"synthetic OOM\"}' >&$fd;;\n"
             "quit) exit 0;; esac; done\n";
    }
    ::chmod(script.c_str(), 0755);

    OFlashSupervisor sup;
    OFlashSupervisorConfig cfg;
    cfg.trainer_bin = script;
    cfg.ready_timeout_ms = 5000;
    REQUIRE(sup.start(cfg));
    for (int i = 0; i < 100 && !sup.trainer_alive(); i++) {
        ::usleep(20 * 1000);
    }
    REQUIRE(sup.trainer_alive());

    sup.begin_rollback(7);
    for (int i = 0; i < 100 && sup.rollback_pending(); i++) {
        ::usleep(20 * 1000);
    }
    CHECK(!sup.rollback_pending());

    OFlashPendingSwap swap;
    bool got = false;
    for (int i = 0; i < 100 && !(got = sup.take_pending_swap(swap)); i++) {
        ::usleep(20 * 1000);
    }
    REQUIRE(got);
    CHECK_EQUAL(swap.path, std::string("/tmp/fresh.st"));
    CHECK_EQUAL(swap.generation, (uint64_t)9);
    CHECK(!sup.take_pending_swap(swap));

    sup.send_line("self-disable");
    for (int i = 0; i < 100 && !sup.trainer_disabled(); i++) {
        ::usleep(20 * 1000);
    }
    CHECK(sup.trainer_disabled());

    sup.stop();
    std::remove(script.c_str());
}

TEST_CASE(OFlashUnitFixture, supervisor_survives_missing_binary) {
    OFlashSupervisor sup;
    OFlashSupervisorConfig cfg;
    cfg.trainer_bin = "/nonexistent/oflash-trainer";
    cfg.ready_timeout_ms = 500;
    cfg.max_respawns = 0;
    REQUIRE(sup.start(cfg));  // thread starts; spawn fails inside
    // Let fork/exec + failed ready handshake actually run; checking alive
    // immediately lets stop() win before the path under test starts.
    ::usleep(500 * 1000);
    CHECK(!sup.trainer_alive());
    sup.stop();  // must not hang or crash
}

TEST_CASE(OFlashUnitFixture, supervisor_closes_unrelated_fds_before_exec) {
    const std::string probe_path =
        "/tmp/oflash-fd-probe-" + std::to_string((long)getpid());
    const int probe_fd = ::open(probe_path.c_str(), O_CREAT | O_RDWR, 0600);
    REQUIRE(probe_fd >= 3);

    const std::string script =
        "/tmp/oflash-fd-trainer-" + std::to_string((long)getpid()) + ".sh";
    {
        std::ofstream f(script);
        f << "#!/bin/bash\n"
             "stream=2\nprobe=-1\n"
             "for a in \"$@\"; do case \"$a\" in "
             "--stream-fd=*) stream=${a#--stream-fd=};; "
             "--probe-fd=*) probe=${a#--probe-fd=};; esac; done\n"
             "if [ -e /proc/$$/fd/$probe ]; then "
             "printf '\\001\\000\\000\\000' >&$stream; exit 1; fi\n"
             "printf '\\000\\000\\000\\000' >&$stream\n"
             "while read line; do [ \"$line\" = quit ] && exit 0; done\n";
    }
    ::chmod(script.c_str(), 0755);

    OFlashSupervisor sup;
    OFlashSupervisorConfig cfg;
    cfg.trainer_bin = script;
    cfg.args = {"--probe-fd=" + std::to_string(probe_fd)};
    cfg.ready_timeout_ms = 5000;
    REQUIRE(sup.start(cfg));
    for (int i = 0; i < 100 && !sup.trainer_alive(); i++) {
        ::usleep(20 * 1000);
    }
    CHECK(sup.trainer_alive());
    sup.stop();
    ::close(probe_fd);
    std::remove(probe_path.c_str());
    std::remove(script.c_str());
}
#endif

// ── Profile store ───────────────────────────────────────────────────

#if !defined(_WIN32)
TEST_CASE(OFlashUnitFixture, store_promoted_roundtrip) {
    const std::string dir =
        "/tmp/oflash-test-store-" + std::to_string((long)getpid());
    ::mkdir(dir.c_str(), 0700);
    CHECK(oflash_store_write_promoted(dir, dir + "/gen3.safetensors", 3));
    std::string path;
    uint64_t gen = 0;
    REQUIRE(oflash_store_read_promoted(dir, path, gen));
    CHECK_EQUAL(gen, (uint64_t)3);
    CHECK_EQUAL(path, dir + "/gen3.safetensors");  // relative made absolute
    std::remove((dir + "/promoted.json").c_str());
    ::rmdir(dir.c_str());
}


TEST_CASE(OFlashUnitFixture, store_promoted_rejects_malformed_types) {
    const std::string dir =
        "/tmp/oflash-test-store-bad-" + std::to_string((long)getpid());
    ::mkdir(dir.c_str(), 0700);
    {
        std::ofstream f(dir + "/promoted.json");
        f << "{\"adapter\":7,\"generation\":\"bad\"}\n";
    }
    std::string path;
    uint64_t gen = 0;
    CHECK(!oflash_store_read_promoted(dir, path, gen));
    std::remove((dir + "/promoted.json").c_str());
    ::rmdir(dir.c_str());
}
#endif
