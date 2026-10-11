// Drafter-session host copies (pflash_session_offload.h): the byte-exact round trip of every owned tensor, the
// checks that refuse a copy into a different layout (nothing written), and the identity rule. CPU backend, model-free.

#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_session_offload.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace luce::pflash;

namespace {

struct PFlashSessionOffloadFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// A session-like layout: a quantized KV block, an F16 tensor, F32 recurrent state, I32, and a view (not owned).
struct Layout {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * view = nullptr;

    explicit Layout(int tokens, ggml_type kv_type = GGML_TYPE_Q8_0, bool rename = false) {
        backend = ggml_backend_cpu_init();
        ggml_init_params p{};
        p.mem_size = 16 * ggml_tensor_overhead();
        p.no_alloc = true;
        ctx = ggml_init(p);
        ggml_set_name(ggml_new_tensor_3d(ctx, kv_type, 64, tokens, 2), "attn_k_0");
        ggml_set_name(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, tokens, 2), "attn_v_0");
        ggml_set_name(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 4), rename ? "ssm_state_x" : "ssm_state_0");
        ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 37), "ids");
        ggml_tensor * keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 2, tokens);
        ggml_set_name(keys, "keys");
        view = ggml_view_1d(ctx, keys, 8, 0);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Layout() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

std::vector<uint8_t> bytes_of(ggml_tensor * t) {
    std::vector<uint8_t> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return out;
}

// Random raw bytes everywhere (any bit pattern: NaN payloads, -0, denormals, arbitrary quant blocks), plus explicit
// special floats at the start of every F32 tensor.
void fill_random(const std::vector<ggml_tensor *> & tensors, uint32_t seed) {
    std::mt19937 rng(seed);
    for (ggml_tensor * t : tensors) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        for (auto & x : b) x = (uint8_t) rng();
        if (t->type == GGML_TYPE_F32 && b.size() >= 5 * sizeof(float)) {
            const uint32_t specials[5] = {0x7fc12345u, 0xff800001u, 0x80000000u, 0x00000001u, 0x7f800000u};
            std::memcpy(b.data(), specials, sizeof(specials));
        }
        ggml_backend_tensor_set(t, b.data(), 0, b.size());
    }
}

std::string temp_path(const char * tag) {
    return std::string("/tmp/pflash_session_offload_") + tag + "_" + std::to_string((long) getpid());
}

} // namespace

TEST_CASE(PFlashSessionOffloadFixture, owned_tensors_skip_views_and_keep_creation_order) {
    Layout a(256);
    REQUIRE(a.buf != nullptr);
    const auto owned = pflash_owned_tensors(a.ctx);
    REQUIRE(owned.size() == 5u);
    CHECK(std::string(ggml_get_name(owned[0])) == "attn_k_0");
    CHECK(std::string(ggml_get_name(owned[4])) == "keys");
    for (ggml_tensor * t : owned) CHECK(t != a.view);
    CHECK(pflash_owned_tensors(nullptr).empty());
}

TEST_CASE(PFlashSessionOffloadFixture, round_trip_is_byte_exact_into_fresh_tensors) {
    Layout a(512);
    Layout b(512);
    REQUIRE(a.buf != nullptr);
    REQUIRE(b.buf != nullptr);
    const auto src = pflash_owned_tensors(a.ctx);
    const auto dst = pflash_owned_tensors(b.ctx);
    fill_random(src, 7);
    fill_random(dst, 99);   // the fresh allocation holds other bytes
    std::vector<PFlashHostTensor> host;
    std::string error;
    REQUIRE(pflash_tensors_to_host(src, host, error));
    size_t expected = 0;
    for (ggml_tensor * t : src) expected += ggml_nbytes(t);
    CHECK(pflash_host_tensor_bytes(host) == expected);
    // The source may be gone before the restore (the drafter is freed): overwrite it.
    fill_random(src, 1234);
    REQUIRE(pflash_tensors_from_host(host, dst, error));
    for (size_t i = 0; i < dst.size(); ++i) {
        const auto got = bytes_of(dst[i]);
        REQUIRE(got.size() == host[i].bytes.size());
        CHECK(std::memcmp(got.data(), host[i].bytes.data(), got.size()) == 0);
    }
    // A second round trip of the restored tensors gives the same bytes again.
    std::vector<PFlashHostTensor> again;
    REQUIRE(pflash_tensors_to_host(dst, again, error));
    for (size_t i = 0; i < again.size(); ++i) CHECK(again[i].bytes == host[i].bytes);
}

TEST_CASE(PFlashSessionOffloadFixture, mismatched_layouts_are_refused_and_nothing_is_written) {
    Layout a(512);
    REQUIRE(a.buf != nullptr);
    fill_random(pflash_owned_tensors(a.ctx), 5);
    std::vector<PFlashHostTensor> host;
    std::string error;
    REQUIRE(pflash_tensors_to_host(pflash_owned_tensors(a.ctx), host, error));

    const auto refused = [&](Layout & other, std::vector<ggml_tensor *> dst, const char * expect) {
        fill_random(pflash_owned_tensors(other.ctx), 77);
        std::vector<std::vector<uint8_t>> before;
        for (ggml_tensor * t : pflash_owned_tensors(other.ctx)) before.push_back(bytes_of(t));
        std::string why;
        const bool ok = pflash_tensors_from_host(host, dst, why);
        CHECK(!ok);
        CHECK(why.find(expect) != std::string::npos);
        size_t i = 0;
        for (ggml_tensor * t : pflash_owned_tensors(other.ctx)) CHECK(bytes_of(t) == before[i++]);
    };
    Layout longer(768);           // a different capacity: shape (and size) differ
    refused(longer, pflash_owned_tensors(longer.ctx), "shape differs");
    Layout other_type(512, GGML_TYPE_F16);   // another KV type
    refused(other_type, pflash_owned_tensors(other_type.ctx), "type differs");
    Layout fewer(512);
    auto dst = pflash_owned_tensors(fewer.ctx);
    dst.pop_back();
    refused(fewer, dst, "tensor count");
    Layout with_view(512);
    auto dst_view = pflash_owned_tensors(with_view.ctx);
    dst_view.back() = with_view.view;
    refused(with_view, dst_view, "not an allocated owned tensor");

    // A truncated host copy (size mismatch) is refused too.
    Layout same(512);
    auto truncated = host;
    truncated[2].bytes.pop_back();
    std::string why;
    CHECK(!pflash_tensors_from_host(truncated, pflash_owned_tensors(same.ctx), why));
    CHECK(why.find("size") != std::string::npos);
}

// A graph renames the unnamed tensors it reads ("leaf_<n>"); a fresh allocation of the same layout does not carry that
// name, and the restore must not depend on it.
TEST_CASE(PFlashSessionOffloadFixture, graph_assigned_names_do_not_block_a_restore) {
    Layout a(256);
    Layout b(256, GGML_TYPE_Q8_0, true);
    fill_random(pflash_owned_tensors(a.ctx), 3);
    std::vector<PFlashHostTensor> host;
    std::string error;
    REQUIRE(pflash_tensors_to_host(pflash_owned_tensors(a.ctx), host, error));
    const auto dst = pflash_owned_tensors(b.ctx);
    REQUIRE(pflash_tensors_from_host(host, dst, error));
    for (size_t i = 0; i < dst.size(); ++i) CHECK(bytes_of(dst[i]) == host[i].bytes);
}

TEST_CASE(PFlashSessionOffloadFixture, unallocated_or_view_tensors_are_not_offloaded) {
    ggml_init_params p{};
    p.mem_size = 4 * ggml_tensor_overhead();
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    std::vector<PFlashHostTensor> host;
    std::string error;
    CHECK(!pflash_tensors_to_host({t}, host, error));
    CHECK(host.empty());
    CHECK(!pflash_tensors_to_host({nullptr}, host, error));
    ggml_free(ctx);

    Layout a(256);
    CHECK(!pflash_tensors_to_host({a.view}, host, error));
    CHECK(host.empty());
}

TEST_CASE(PFlashSessionOffloadFixture, identity_names_the_first_differing_field) {
    const std::string a = "format=1\ndrafter=/m/a.gguf|1:2|10|5\nsessions=2\n";
    CHECK(pflash_identity_mismatch(a, a).empty());
    CHECK(pflash_identity_mismatch(a, "format=1\ndrafter=/m/b.gguf|1:3|10|5\nsessions=2\n") == "drafter differs");
    CHECK(pflash_identity_mismatch(a, "format=1\ndrafter=/m/a.gguf|1:2|10|5\nsessions=3\n") == "sessions differs");
    CHECK(pflash_identity_mismatch(a, a + "head=x\n") == "head added");
    CHECK(pflash_identity_mismatch(a + "head=x\n", a) == "head removed");
}

TEST_CASE(PFlashSessionOffloadFixture, file_identity_changes_with_the_file) {
    CHECK(pflash_file_identity("") == "-");
    const std::string path = temp_path("ident");
    std::remove(path.c_str());
    CHECK(pflash_file_identity(path) == path + "|missing");
    { std::ofstream(path) << "drafter-v1"; }
    const std::string first = pflash_file_identity(path);
    CHECK(first.rfind(path + "|", 0) == 0);
    CHECK(first != path + "|missing");
    CHECK(pflash_file_identity(path) == first);   // stable while the file is unchanged
    { std::ofstream(path) << "drafter-v2 longer"; }  // replaced in place: size changes
    CHECK(pflash_file_identity(path) != first);
    std::remove(path.c_str());
}
