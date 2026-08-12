#include "CppUnitTestFramework.hpp"
#include "internal.h"
#include "qwen35/graph_builders.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

using namespace CppUnitTestFramework;

using dflash::common::TargetCache;
using dflash::common::restore_ssm_state;
using dflash::common::snapshot_ssm_state;

namespace {
struct RecurrentSnapshotFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

static void set_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0,
                            values.size() * sizeof(float));
}

static std::vector<float> get_tensor(const ggml_tensor * tensor) {
    std::vector<float> values((size_t)ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    return values;
}

TEST_CASE(RecurrentSnapshotFixture, validates_paged_tree_layout) {
    // The packed-tree launch length is logical. KVFlash may keep a much
    // smaller physical resident pool, provided every tree scratch slab still
    // fits within that pool.
    {
        ggml_init_params shape_params{};
        shape_params.mem_size = 8 * ggml_tensor_overhead();
        shape_params.no_alloc = true;
        ggml_context * shape_ctx = ggml_init(shape_params);
        CHECK(shape_ctx != nullptr);
        if (shape_ctx) {
            TargetCache shape_cache;
            shape_cache.n_seq_slots = 2;
            shape_cache.paged_block_table =
                ggml_new_tensor_2d(shape_ctx, GGML_TYPE_I32, 4, 2);
            shape_cache.paged_kv_seq_lens =
                ggml_new_tensor_1d(shape_ctx, GGML_TYPE_I32, 2);
            shape_cache.attn_k = {
                ggml_new_tensor_4d(shape_ctx, GGML_TYPE_F16, 4, 64, 1, 1),
            };
            CHECK(dflash::common::detail::validate_target_paged_tree_layout(
                shape_cache, 8, 2, 4096, 32, 16));
            CHECK(!dflash::common::detail::validate_target_paged_tree_layout(
                shape_cache, 8, 2, 4096, 48, 16));
            CHECK(!dflash::common::detail::validate_target_paged_tree_layout(
                shape_cache, 8, 5, 4096, 32, 16));
            ggml_free(shape_ctx);
        }
    }

}

TEST_CASE(RecurrentSnapshotFixture, snapshot_and_restore_recurrent_state) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) SKIP("CPU backend is unavailable");

    ggml_init_params params{};
    params.mem_size = 8 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (!ctx) {
        ggml_backend_free(backend);
        SKIP("could not initialize ggml context");
    }

    ggml_tensor * ssm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * ssm_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_tensor * conv_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        SKIP("could not allocate CPU backend tensors");
    }

    TargetCache cache;
    cache.ssm_state = {ssm, nullptr};
    cache.ssm_state_snap = {ssm_snap, nullptr};
    cache.conv_state = {conv, nullptr};
    cache.conv_state_snap = {conv_snap, nullptr};

    const std::vector<float> ssm_original = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    const std::vector<float> conv_original = {
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    };
    const std::vector<float> ssm_mutated(ssm_original.size(), -1.0f);
    const std::vector<float> conv_mutated(conv_original.size(), -2.0f);

    CHECK(ggml_nelements(ssm) == (int64_t) ssm_original.size());
    CHECK(ggml_nelements(conv) == (int64_t) conv_original.size());
    set_tensor(ssm, ssm_original);
    set_tensor(conv, conv_original);
    CHECK(snapshot_ssm_state(cache, backend));
    set_tensor(ssm, ssm_mutated);
    set_tensor(conv, conv_mutated);
    CHECK(restore_ssm_state(cache, backend));
    CHECK(get_tensor(ssm) == ssm_original);
    CHECK(get_tensor(conv) == conv_original);

    // A partial shard may omit a complete recurrent-state quartet, but an
    // asymmetric quartet must fail validation before any copy is queued.
    set_tensor(ssm, ssm_mutated);
    cache.conv_state_snap[0] = nullptr;
    CHECK(!snapshot_ssm_state(cache, backend));
    CHECK(get_tensor(ssm_snap) == ssm_original);
    cache.conv_state_snap[0] = conv_snap;

    CHECK(!snapshot_ssm_state(cache, nullptr));
    cache.ssm_state_snap.pop_back();
    CHECK(!restore_ssm_state(cache, backend));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
