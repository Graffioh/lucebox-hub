// oflash_runtime.cpp — OFlash orchestrator implementation.

#include "oflash_runtime.h"
#include "oflash_adapter.h"
#include "oflash_format.h"

#include "internal.h"               // DraftWeights
#include "common/gguf_inspect.h"    // read_gguf_metadata (drafter sha256)

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common::oflash {

struct OFlashAdapterHostHolder {
    OFlashAdapterHost current;
    OFlashAdapterHost previous;
    std::string current_path;
};

namespace {

uint64_t mono_ns() {
#if defined(_WIN32)
    return 0;
#else
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

bool make_dirs(const std::string & path) {
#if defined(_WIN32)
    (void)path;
    return false;
#else
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (cur == "/" || cur.empty()) continue;
            if (::mkdir(cur.c_str(), 0700) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }
    return true;
#endif
}

}  // namespace

bool OFlashRuntime::init(const OFlashConfig & cfg,
                         const std::string & drafter_path,
                         DraftWeights & dw,
                         ggml_backend_t draft_backend,
                         int vocab) {
    cfg_ = cfg;
    dw_  = &dw;
    staging_ = new OFlashAdapterHostHolder();

    // 1. LoRA slots — the only hard requirement. Without them there is no
    //    adaptation, so a failure here disables OFlash entirely.
    std::string error;
    lora_ = oflash_lora_create(dw, draft_backend, cfg.lora_rank,
                               cfg.lora_alpha, error);
    if (!lora_) {
        std::fprintf(stderr, "[oflash] init failed: %s\n", error.c_str());
        return false;
    }
    dw.oflash = lora_;

    // 2. Drafter identity (sidecar-cached SHA-256 of the GGUF).
    const GgufMetadata meta = read_gguf_metadata(drafter_path, true);
    if (!meta.ok || meta.sha256.empty()) {
        std::fprintf(stderr,
            "[oflash] drafter hash failed for %s; adapters disabled, "
            "capture continues unlabeled\n", drafter_path.c_str());
    } else {
        drafter_sha256_ = meta.sha256;
    }

    // 3. Profile store + warm start (OFLASH.md §8).
    if (!drafter_sha256_.empty()) {
        profile_dir_ = oflash_profile_dir(cfg.dir, drafter_sha256_,
                                          cfg.profile);
        if (!make_dirs(profile_dir_)) {
            std::fprintf(stderr, "[oflash] cannot create %s; persistence off\n",
                         profile_dir_.c_str());
            profile_dir_.clear();
        }
        std::string warm_path;
        uint64_t warm_gen = 0;
        if (!profile_dir_.empty() &&
            oflash_store_read_promoted(profile_dir_, warm_path, warm_gen)) {
            if (load_and_upload(warm_path, warm_gen, /*warm_start=*/true)) {
                std::fprintf(stderr,
                    "[oflash] warm start: adapter gen %llu (%s)\n",
                    (unsigned long long)warm_gen, warm_path.c_str());
            }
        }
    }

    // 4. Capture ring.
#if !defined(_WIN32)
    ring_name_ = "/lucebox-oflash-" + std::to_string((long)::getpid());
#endif
    const uint64_t ring_bytes = (uint64_t)cfg.ring_mb << 20;
    if (!ring_.create(ring_name_, ring_bytes,
                      oflash_hash_from_hex(drafter_sha256_.c_str()),
                      (uint32_t)dw.n_target_layers,
                      (uint32_t)dw.n_embd,
                      (uint32_t)dw.block_size,
                      (uint32_t)cfg.topk,
                      (uint32_t)vocab)) {
        std::fprintf(stderr, "[oflash] ring create failed; capture off\n");
    } else {
        std::fprintf(stderr, "[oflash] capture ring %s (%d MiB)\n",
                     ring_name_.c_str(), cfg.ring_mb);
    }

    // 5. Trainer sidecar (optional — empty bin = M0 capture-only).
    if (!cfg.trainer_bin.empty() && ring_.active() &&
        !drafter_sha256_.empty() && !profile_dir_.empty()) {
        OFlashSupervisorConfig scfg;
        scfg.trainer_bin  = cfg.trainer_bin;
        scfg.drafter_path = drafter_path;
        scfg.args = {
            "--ring-name=" + ring_name_,
            "--out-dir=" + profile_dir_,
            "--profile=" + cfg.profile,
            "--rank=" + std::to_string(cfg.lora_rank),
            "--alpha=" + std::to_string(cfg.lora_alpha),
            "--device=" + cfg.device,
            "--drafter-sha256=" + drafter_sha256_,
            "--start-generation=" + std::to_string(lora_->generation),
        };
        if (!supervisor_.start(scfg)) {
            std::fprintf(stderr,
                "[oflash] trainer not started; capture-only mode\n");
        }
    } else if (cfg.trainer_bin.empty()) {
        std::fprintf(stderr, "[oflash] no --oflash-trainer-bin; "
                             "capture-only (M0) mode\n");
    }
    return true;
}

void OFlashRuntime::shutdown() {
    supervisor_.stop();
    ring_.close();
    if (dw_) dw_->oflash = nullptr;
    if (lora_) { oflash_lora_free(lora_); lora_ = nullptr; }
    delete staging_;
    staging_ = nullptr;
    dw_ = nullptr;
}

void OFlashRuntime::on_request_begin() {
    seq_id_++;
    want_context_ = ring_.active();
}

void OFlashRuntime::on_request_end(int committed) {
    if (!ring_.active()) return;
    OFlashRecordHeader h{};
    h.type = OFLASH_REC_SEQ_END;
    h.seq_id = seq_id_;
    h.pos = committed;
    h.n_rows = 0;
    h.t_mono_ns = mono_ns();
    ring_.push(h, nullptr, 0);
}

void OFlashRuntime::emit_context(int pos0, int n_rows,
                                 const FeatReader & read) {
    want_context_ = false;
    if (!ring_.active() || n_rows <= 0 || !dw_) return;
    const size_t row_elems = (size_t)dw_->n_target_layers * dw_->n_embd;
    std::vector<uint16_t> feat((size_t)n_rows * row_elems);
    for (int i = 0; i < n_rows; i++) {
        if (!read(pos0 + i, feat.data() + (size_t)i * row_elems)) return;
    }
    OFlashRecordHeader h{};
    h.type = OFLASH_REC_CONTEXT;
    h.seq_id = seq_id_;
    h.pos = pos0;
    h.n_rows = n_rows;
    h.t_mono_ns = mono_ns();
    ring_.push(h, feat.data(), feat.size() * sizeof(uint16_t));
}

void OFlashRuntime::emit_step(const OFlashStepCapture & s,
                              const FeatReader & read) {
    steps_++;
    // Guard first — AL bookkeeping must run even when the ring is off.
    apply_guard_action(guard_.record_step((float)s.accept_len));
    refresh_props_cache();

    if (!ring_.active() || !dw_) return;

    const size_t row_elems = (size_t)dw_->n_target_layers * dw_->n_embd;
    std::vector<uint16_t> feat((size_t)s.n_rows * row_elems);
    for (int i = 0; i < s.n_rows; i++) {
        if (!read(s.pos + i, feat.data() + (size_t)i * row_elems)) return;
    }

    // Label section, mirroring the STEP payload layout in oflash_format.h.
    const int32_t B = s.block;
    const int32_t K = (s.topk_ids && s.topk_lp) ? cfg_.topk : 0;
    const size_t flags_padded = ((size_t)B + 7) & ~(size_t)7;
    const size_t label_bytes = 8                 // n_labels + topk_k
                       + (size_t)B * 4 * 2       // draft_tok + target_tok
                       + flags_padded            // accept_flags (+pad)
                       + 8                       // accept_len + bonus_tok
                       + (size_t)B * (size_t)K * 8;
    std::vector<uint8_t> labels(label_bytes, 0);
    uint8_t * p = labels.data();
    std::memcpy(p, &B, 4);                       p += 4;
    std::memcpy(p, &K, 4);                       p += 4;
    if (B > 0) {
        std::memcpy(p, s.draft_tok, (size_t)B * 4);  p += (size_t)B * 4;
        std::memcpy(p, s.target_tok, (size_t)B * 4); p += (size_t)B * 4;
        std::memcpy(p, s.accept_flags, (size_t)B);   p += flags_padded;
    }
    std::memcpy(p, &s.accept_len, 4);            p += 4;
    std::memcpy(p, &s.bonus_tok, 4);             p += 4;
    if (K > 0) {
        std::memcpy(p, s.topk_ids, (size_t)B * K * 4); p += (size_t)B * K * 4;
        std::memcpy(p, s.topk_lp,  (size_t)B * K * 4); p += (size_t)B * K * 4;
    }

    OFlashRecordHeader h{};
    h.type = OFLASH_REC_STEP;
    h.seq_id = seq_id_;
    h.pos = s.pos;
    h.n_rows = s.n_rows;
    h.t_mono_ns = mono_ns();
    ring_.push2(h, feat.data(), feat.size() * sizeof(uint16_t),
                labels.data(), labels.size());
}

void OFlashRuntime::maybe_apply_swap() {
    if (training_disabled_ || !lora_) return;
    OFlashPendingSwap swap;
    if (supervisor_.take_pending_swap(swap)) {
        // A newer trainer export supersedes any deferred one.
        pending_local_ = swap;
        has_pending_local_ = true;
    }
    if (!has_pending_local_ || !guard_.can_swap()) return;
    has_pending_local_ = false;
    if (load_and_upload(pending_local_.path, pending_local_.generation,
                        /*warm_start=*/false)) {
        guard_.on_swap(pending_local_.generation);
    }
    refresh_props_cache();
}

void OFlashRuntime::apply_guard_action(OFlashGuardAction action) {
    switch (action) {
        case OFlashGuardAction::None:
            break;
        case OFlashGuardAction::Promote: {
            std::fprintf(stderr,
                "[oflash] promote gen %llu (AL %.2f -> %.2f)\n",
                (unsigned long long)lora_->generation,
                guard_.baseline_al(), guard_.probation_al());
            if (!profile_dir_.empty() && staging_ &&
                !staging_->current.path.empty()) {
                oflash_store_write_promoted(profile_dir_,
                                            staging_->current.path,
                                            lora_->generation);
            }
            supervisor_.send_line("promote " +
                                  std::to_string(lora_->generation));
            break;
        }
        case OFlashGuardAction::Rollback:
        case OFlashGuardAction::Disable: {
            const uint64_t bad_gen = lora_ ? lora_->generation : 0;
            std::fprintf(stderr,
                "[oflash] %s gen %llu (probation AL %.2f < baseline %.2f)\n",
                action == OFlashGuardAction::Disable ? "DISABLE after rollback of"
                                                     : "rollback",
                (unsigned long long)bad_gen,
                guard_.probation_al(), guard_.baseline_al());
            // Restore the previous generation's tensors from host staging.
            if (lora_ && staging_) {
                std::string err;
                if (!oflash_lora_upload(*lora_, staging_->previous, err)) {
                    // Previous was the zero adapter (empty tensors) or a
                    // partial staging failure — fall back to zeros.
                    OFlashAdapterHost zeros;
                    if (!oflash_lora_upload(*lora_, zeros, err)) {
                        std::fprintf(stderr,
                            "[oflash] rollback upload failed: %s\n",
                            err.c_str());
                    }
                }
                staging_->current = staging_->previous;
                adapter_dirty_ = true;
            }
            supervisor_.send_line(
                (action == OFlashGuardAction::Disable ? "disable"
                 : "rollback " + std::to_string(bad_gen)));
            if (action == OFlashGuardAction::Disable) {
                training_disabled_ = true;
                has_pending_local_ = false;
            }
            break;
        }
    }
}

bool OFlashRuntime::load_and_upload(const std::string & path,
                                    uint64_t generation,
                                    bool warm_start) {
    if (!lora_ || !dw_ || !staging_) return false;
    OFlashAdapterHost host;
    std::string error;
    if (!oflash_adapter_load(path, *dw_, cfg_.lora_rank, drafter_sha256_,
                             host, error)) {
        std::fprintf(stderr, "[oflash] adapter refused (%s): %s\n",
                     path.c_str(), error.c_str());
        return false;
    }
    if (generation != 0 && host.generation != generation) {
        std::fprintf(stderr,
            "[oflash] adapter %s generation %llu != announced %llu\n",
            path.c_str(), (unsigned long long)host.generation,
            (unsigned long long)generation);
        return false;
    }
    if (!oflash_lora_upload(*lora_, host, error)) {
        std::fprintf(stderr, "[oflash] adapter upload failed: %s\n",
                     error.c_str());
        return false;
    }
    if (!warm_start) staging_->previous = staging_->current;
    staging_->current = host;
    adapter_dirty_ = true;
    return true;
}

// Decode thread only; rebuilds the HTTP-visible snapshot after any guard or
// adapter mutation.
void OFlashRuntime::refresh_props_cache() {
    OFlashPropsSnapshot s;
    s.enabled = true;
    s.profile = cfg_.profile;
    s.adapter_generation = lora_ ? lora_->generation : 0;
    s.swaps = guard_.swaps();
    s.promotes = guard_.promotes();
    s.rollbacks = guard_.rollbacks();
    s.rolling_al = guard_.baseline_al();
    s.probation_al =
        guard_.state() == OFlashGuard::State::Probation ? guard_.probation_al()
                                                        : 0.0;
    s.training_disabled =
        training_disabled_ || guard_.state() == OFlashGuard::State::Disabled;
    s.records_written = ring_.written();
    s.steps = steps_;
    std::lock_guard<std::mutex> lock(stats_mu_);
    props_cache_ = s;
}

OFlashPropsSnapshot OFlashRuntime::props() const {
    OFlashPropsSnapshot s;
    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        s = props_cache_;
    }
    s.enabled = true;
    s.profile = cfg_.profile;
    // Cross-thread-safe live values (shm atomics / supervisor mutex).
    s.trainer_alive = supervisor_.trainer_alive();
    s.records_dropped = ring_.dropped();
    s.ring_backlog_bytes = ring_.backlog();
    return s;
}

}  // namespace dflash::common::oflash
