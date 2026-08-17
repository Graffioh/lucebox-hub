#include "CppUnitTestFramework.hpp"

#include "qwen35/qwen35_tensor_parallel.h"

#include "ggml.h"

#include <cstdint>

using namespace dflash::common;
using namespace CppUnitTestFramework;

namespace {
struct Qwen35TensorParallelFixture : CommonFixture {
    using CommonFixture::CommonFixture;
    void expect_staging_cache_lifecycle(ggml_context * ctx);
};
}

static bool expect_split(const TargetWeights & weights,
                         ggml_tensor * tensor,
                         const char * name,
                         ggml_backend_meta_split_axis axis,
                         int64_t per_device,
                         uint32_t repeat) {
    ggml_set_name(tensor, name);
    const auto state = qwen35_tensor_parallel_split_state(tensor, weights, 2);
    return state.axis == axis &&
        state.n_segments == 1 &&
        state.ne[0] == per_device &&
        state.ne[1] == per_device &&
        state.nr[0] == repeat &&
        (state.ne[0] + state.ne[1]) * state.nr[0] == tensor->ne[axis];
}

void Qwen35TensorParallelFixture::expect_staging_cache_lifecycle(
        ggml_context * ctx) {
    ggml_tensor * primary =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * extra =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    TargetCache cache;
    cache.prefill_staging.resize(2);
    cache.prefill_staging[0] = {{primary}, {primary}, {primary}, {primary}};
    cache.prefill_staging[1] = {{extra}, {extra}, {extra}, {extra}};

    CHECK(staging_k_for(cache, 0).at(0) == primary);
    CHECK(staging_k_for(cache, 1).at(0) == extra);
    CHECK(staging_k_for(cache, -1).empty());
    CHECK(staging_k_for(cache, 2).empty());
    CHECK(staging_v_for(cache, -1).empty());
    CHECK(staging_ssm_for(cache, 2).empty());
    CHECK(staging_conv_for(cache, 2).empty());

    free_target_cache(cache);
    CHECK(cache.prefill_staging.empty());
}

TEST_CASE(Qwen35TensorParallelFixture, tensor_parallel_split_state) {
    ggml_init_params params{};
    params.mem_size = 32 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    expect_staging_cache_lifecycle(ctx);

    TargetWeights weights;

    ggml_tensor * zero_device_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 10240);
    ggml_set_name(zero_device_tensor, "blk.0.attn_qkv.weight");
    const auto zero_device_state =
        qwen35_tensor_parallel_split_state(zero_device_tensor, weights, 0);
    CHECK(zero_device_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    CHECK(zero_device_state.n_segments == 1);

    REQUIRE(expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 10240),
        "blk.0.attn_qkv.weight", GGML_BACKEND_SPLIT_AXIS_1, 1024, 5));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10240),
        "blk.0.ssm_conv1d.weight", GGML_BACKEND_SPLIT_AXIS_1, 1024, 5));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, 5120),
        "blk.0.ssm_out.weight", GGML_BACKEND_SPLIT_AXIS_0, 1024, 3));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 12288),
        "blk.3.attn_q.weight", GGML_BACKEND_SPLIT_AXIS_1, 6144, 1));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, 5120),
        "blk.3.attn_output.weight", GGML_BACKEND_SPLIT_AXIS_0, 3072, 1));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4096, 4),
        "cache_k_3", GGML_BACKEND_SPLIT_AXIS_2, 2, 1));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 128, 48),
        "ssm_state_0", GGML_BACKEND_SPLIT_AXIS_2, 8, 3));
    REQUIRE(expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 10240),
        "conv_state_0", GGML_BACKEND_SPLIT_AXIS_1, 1024, 5));

    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    ggml_set_name(norm, "blk.0.attn_norm.weight");
    const auto mirrored = qwen35_tensor_parallel_split_state(norm, weights, 2);
    CHECK(mirrored.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 248320);
    ggml_set_name(output, "output.weight");
    const auto output_state =
        qwen35_tensor_parallel_split_state(output, weights, 2);
    CHECK(output_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    DevicePlacement placement;
    placement.layer_split_gpus = {1, 2};
    placement.layer_split_backends = {PlacementBackend::Cuda, PlacementBackend::Cuda};
    CHECK(placement.is_layer_split());
    placement.split_mode = TargetSplitMode::Tensor;
    CHECK(placement.is_tensor_parallel());
    CHECK(validate_device_placement(placement, 3).empty());
    placement.layer_split_weights = {1.0, 1.0};
    CHECK(!validate_device_placement(placement, 3).empty());

    DevicePlacement missing_tensor_devices;
    missing_tensor_devices.split_mode = TargetSplitMode::Tensor;
    CHECK(!validate_device_placement(missing_tensor_devices, 3).empty());
    missing_tensor_devices.layer_split_gpus = {0};
    missing_tensor_devices.layer_split_backends = {PlacementBackend::Cuda};
    CHECK(!validate_device_placement(missing_tensor_devices, 3).empty());

    DevicePlacement too_many_devices;
    too_many_devices.split_mode = TargetSplitMode::Tensor;
    too_many_devices.layer_split_gpus.resize(
        GGML_BACKEND_META_MAX_DEVICES + 1);
    too_many_devices.layer_split_backends.resize(
        GGML_BACKEND_META_MAX_DEVICES + 1, PlacementBackend::Cuda);
    for (size_t i = 0; i < too_many_devices.layer_split_gpus.size(); ++i) {
        too_many_devices.layer_split_gpus[i] = (int) i;
    }
    CHECK(!Qwen35TensorParallelContext::create(too_many_devices, weights));

    ggml_free(ctx);
}
