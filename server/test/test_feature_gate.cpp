// Unit tests for the backend feature/architecture gate.
//
// check_feature_compatibility(), collect_feature_warnings() and the
// model_capabilities.h table are pure functions over resolved facts, so this
// binary needs no model file, no GPU, and none of the backend stack — it
// compiles against feature_gate.cpp and placement_config.cpp alone. Keeping
// it separate from test_server_unit keeps that true: a gate rule stays
// testable in seconds rather than behind a full CUDA build.
//
// Build: cmake --build . --target test_feature_gate
// Run:   ./test_feature_gate

#include "common/feature_gate.h"
#include "common/model_capabilities.h"
#include "common/paged_attention_config.h"
#include "placement/placement_config.h"

#include <climits>
#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

static int test_failures = 0;
static int test_count = 0;

#define TEST_ASSERT(expr) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

// ── Backend compatibility gate ──────────────────────────────────────────
// One case per rule cluster in check_feature_compatibility(). All resolved
// facts are parameters, so none of this needs a model file or GPU.

static BackendArgs gate_args_hip_deepseek4() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.device.backend = PlacementBackend::Hip;
    args.device.gpu = 0;
    return args;
}

static bool gate_accepts(
    const BackendArgs & args,
    const std::string & arch,
    PlacementBackend backend,
    const BackendFeatureConfig & features = {}) {
    return check_feature_compatibility(
        args, features, arch, backend, backend).empty();
}

static std::string gate_result_for_binary(
    const BackendArgs & args,
    const std::string & arch,
    PlacementBackend target_backend,
    PlacementBackend compiled_backend,
    const BackendFeatureConfig & features = {}) {
    return check_feature_compatibility(
        args, features, arch, target_backend, compiled_backend);
}

static void test_feature_gate_accepts_plain_launch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(gate_accepts(
        args, "qwen35", PlacementBackend::Cuda));
}

static void test_feature_gate_rejects_undetected_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(!gate_accepts(
        args, "", PlacementBackend::Cuda));
}

static void test_feature_gate_requires_compiled_target_backend() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.device.backend = PlacementBackend::Hip;
    TEST_ASSERT(!gate_result_for_binary(
        args, "qwen35", PlacementBackend::Hip,
        PlacementBackend::Cuda).empty());
}

static void test_feature_gate_ipc_options_require_ipc_binary() {
    BackendArgs draft;
    draft.model_path = "/nonexistent/model.gguf";
    draft.remote_draft.work_dir = "/tmp/draft";
    TEST_ASSERT(!gate_accepts(
        draft, "qwen35", PlacementBackend::Cuda));

    BackendArgs target;
    target.model_path = "/nonexistent/model.gguf";
    target.remote_target_shard.work_dir = "/tmp/target";
    TEST_ASSERT(!gate_accepts(
        target, "qwen35", PlacementBackend::Cuda));
}

static void test_feature_gate_mixed_draft_placement_requires_ipc() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";
    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;

    TEST_ASSERT(!gate_accepts(
        args, "qwen35", PlacementBackend::Cuda));

    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";
    TEST_ASSERT(gate_accepts(
        args, "qwen35", PlacementBackend::Cuda));

    args.draft_device.backend = PlacementBackend::Cuda;
    TEST_ASSERT(!gate_accepts(
        args, "qwen35", PlacementBackend::Cuda));
}

static void test_feature_gate_pflash_requires_drafter_and_supported_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";

    BackendFeatureConfig features;
    features.pflash_enabled = true;
    TEST_ASSERT(!gate_accepts(
        args, "qwen35", PlacementBackend::Cuda, features));

    features.pflash_drafter_configured = true;
    TEST_ASSERT(gate_accepts(
        args, "gemma4", PlacementBackend::Cuda, features));

    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;
    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";
    TEST_ASSERT(!gate_accepts(
        args, "gemma4", PlacementBackend::Cuda, features));
    TEST_ASSERT(gate_accepts(
        args, "qwen35", PlacementBackend::Cuda, features));
}

static void test_feature_gate_validates_target_split_topology() {
    BackendArgs weights;
    weights.model_path = "/nonexistent/model.gguf";
    weights.device.layer_split_weights = {1.0, 1.0};
    TEST_ASSERT(!gate_accepts(
        weights, "qwen35", PlacementBackend::Cuda));

    BackendArgs mixed;
    mixed.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(parse_placement_device_list(
        "cuda:0,hip:0", mixed.device));
    TEST_ASSERT(!gate_accepts(
        mixed, "qwen35", PlacementBackend::Cuda));

    mixed.remote_target_shard.ipc_bin = "/usr/bin/target-shard";
    TEST_ASSERT(gate_accepts(
        mixed, "qwen35", PlacementBackend::Cuda));

    BackendArgs two_boundaries;
    two_boundaries.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(parse_placement_device_list(
        "cuda:0,hip:0,cuda:1", two_boundaries.device));
    two_boundaries.remote_target_shard.ipc_bin =
        "/usr/bin/target-shard";
    TEST_ASSERT(!gate_accepts(
        two_boundaries, "qwen35", PlacementBackend::Cuda));
}

static void test_feature_gate_tensor_parallel_requirements() {
    BackendArgs valid;
    valid.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(parse_placement_device_list(
        "cuda:0,cuda:1", valid.device));
    valid.device.split_mode = TargetSplitMode::Tensor;
    TEST_ASSERT(gate_accepts(
        valid, "qwen35", PlacementBackend::Cuda));

    BackendArgs missing_devices;
    missing_devices.model_path = "/nonexistent/model.gguf";
    missing_devices.device.split_mode = TargetSplitMode::Tensor;
    TEST_ASSERT(!gate_accepts(
        missing_devices, "qwen35", PlacementBackend::Cuda));

    TEST_ASSERT(!gate_accepts(
        valid, "laguna", PlacementBackend::Cuda));

    BackendArgs hip;
    hip.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(parse_placement_device_list("hip:0,hip:1", hip.device));
    hip.device.split_mode = TargetSplitMode::Tensor;
    TEST_ASSERT(!gate_accepts(
        hip, "qwen35", PlacementBackend::Hip));

    BackendArgs mixed = valid;
    TEST_ASSERT(parse_placement_device_list(
        "cuda:0,hip:0", mixed.device));
    mixed.device.split_mode = TargetSplitMode::Tensor;
    TEST_ASSERT(!gate_accepts(
        mixed, "qwen35", PlacementBackend::Cuda));

    BackendArgs weighted = valid;
    weighted.device.layer_split_weights = {1.0, 1.0};
    TEST_ASSERT(!gate_accepts(
        weighted, "qwen35", PlacementBackend::Cuda));

    BackendArgs remote = valid;
    remote.remote_target_shard.ipc_bin = "/usr/bin/target-shard";
    TEST_ASSERT(!gate_accepts(
        remote, "qwen35", PlacementBackend::Cuda));

    BackendFeatureConfig pflash;
    pflash.pflash_enabled = true;
    pflash.pflash_drafter_configured = true;
    TEST_ASSERT(!gate_accepts(
        valid, "qwen35", PlacementBackend::Cuda, pflash));

    BackendArgs draft = valid;
    draft.draft_path = "/nonexistent/draft.gguf";
    TEST_ASSERT(!gate_accepts(
        draft, "qwen35", PlacementBackend::Cuda));
}

static void test_feature_gate_ds4_prefill_requires_deepseek4() {
    BackendArgs args = gate_args_hip_deepseek4();
    args.ds4_prefill_mode_set = true;
    args.ds4_prefill_mode = PrefillAttentionMode::Dense;

    TEST_ASSERT(!gate_accepts(
        args, "qwen35", PlacementBackend::Hip));
    TEST_ASSERT(gate_accepts(
        args, "deepseek4", PlacementBackend::Hip));
}

static void test_feature_gate_approximate_ds4_prefill_requires_local_hip() {
    BackendArgs args = gate_args_hip_deepseek4();
    args.ds4_prefill_mode_set = true;
    args.ds4_prefill_mode = PrefillAttentionMode::Sparse;

    // CUDA has no approximate prefill path.
    TEST_ASSERT(!gate_accepts(
        args, "deepseek4", PlacementBackend::Cuda));

    // Neither does the layer-split adapter, even on HIP.
    BackendArgs split = args;
    TEST_ASSERT(parse_placement_device_list("hip:0,hip:1", split.device));
    TEST_ASSERT(!gate_accepts(
        split, "deepseek4", PlacementBackend::Hip));

    // Nor a remote target shard.
    BackendArgs remote = args;
    remote.remote_target_shard.ipc_bin = "/usr/bin/shard";
    TEST_ASSERT(!gate_accepts(
        remote, "deepseek4", PlacementBackend::Hip));

    // Single local HIP device is the supported placement.
    TEST_ASSERT(gate_accepts(
        args, "deepseek4", PlacementBackend::Hip));

    // Exact prefill is unrestricted.
    BackendArgs exact = gate_args_hip_deepseek4();
    exact.ds4_prefill_mode_set = true;
    exact.ds4_prefill_mode = PrefillAttentionMode::Exact;
    TEST_ASSERT(gate_accepts(
        exact, "deepseek4", PlacementBackend::Cuda));
}

static void test_feature_gate_ds4_decode_options_require_monolithic_hip() {
    BackendArgs fused = gate_args_hip_deepseek4();
    fused.ds4_fused_decode = true;
    TEST_ASSERT(!gate_accepts(
        fused, "deepseek4", PlacementBackend::Cuda));
    TEST_ASSERT(gate_accepts(
        fused, "deepseek4", PlacementBackend::Hip));

    BackendArgs topk = gate_args_hip_deepseek4();
    topk.ds4_expert_top_k = 4;
    TEST_ASSERT(!gate_accepts(
        topk, "qwen35", PlacementBackend::Hip));
    TEST_ASSERT(gate_accepts(
        topk, "deepseek4", PlacementBackend::Hip));

    // Top-k is a model policy in the monolithic backend and is independent of
    // the GPU vendor. Unlike fused decode, mixed CUDA-primary expert
    // placement can therefore use it.
    BackendArgs cuda_topk = topk;
    cuda_topk.device.backend = PlacementBackend::Cuda;
    TEST_ASSERT(gate_accepts(
        cuda_topk, "deepseek4", PlacementBackend::Cuda));

    BackendArgs split_topk = topk;
    TEST_ASSERT(parse_placement_device_list("hip:0,hip:1", split_topk.device));
    TEST_ASSERT(!gate_accepts(
        split_topk, "deepseek4", PlacementBackend::Hip));
}

static void test_feature_gate_remote_draft_requires_supported_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";
    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;
    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";

    TEST_ASSERT(!gate_accepts(
        args, "gemma4", PlacementBackend::Cuda));
    TEST_ASSERT(gate_accepts(
        args, "qwen35", PlacementBackend::Cuda));

    // Without a draft model or PFlash, remote draft IPC is unnecessary.
    BackendArgs no_draft = args;
    no_draft.draft_path = nullptr;
    TEST_ASSERT(!gate_accepts(
        no_draft, "gemma4", PlacementBackend::Cuda));
}

static void test_feature_gate_layer_split_requires_supported_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", args.device));

    // These four have a layer-split adapter.
    for (const char * arch : {"qwen35", "laguna", "gemma4", "deepseek4"}) {
        TEST_ASSERT(gate_accepts(args, arch, PlacementBackend::Cuda));
    }
    // These two do not: the factory would hand the split placement to a
    // monolithic backend, which reads only the primary GPU.
    for (const char * arch : {"qwen35moe", "qwen3"}) {
        TEST_ASSERT(!gate_accepts(args, arch, PlacementBackend::Cuda));
    }

    // Single-device placement is unaffected for the same architectures.
    BackendArgs single;
    single.model_path = "/nonexistent/model.gguf";
    TEST_ASSERT(gate_accepts(single, "qwen35moe", PlacementBackend::Cuda));
    TEST_ASSERT(gate_accepts(single, "qwen3", PlacementBackend::Cuda));
}

static void test_feature_gate_paged_attention_requires_supported_monolithic() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.paged_attention = true;
    TEST_ASSERT(gate_accepts(args, "qwen35", PlacementBackend::Cuda));
    TEST_ASSERT(gate_accepts(args, "qwen35", PlacementBackend::Hip));
    TEST_ASSERT(gate_accepts(args, "deepseek4", PlacementBackend::Cuda));

    // Only qwen35 has a paged decode path. qwen35moe shares Qwen35Config, so
    // its rejection is this gate's job — the factory's field-presence
    // cross-check cannot tell the two apart.
    for (const char * arch : {"qwen35moe", "laguna", "qwen3", "gemma4"}) {
        TEST_ASSERT(!gate_accepts(args, arch, PlacementBackend::Cuda));
    }

    // Only the monolithic qwen35 backend owns a paged K/V pool. Both
    // placements are supported qwen35 launches without the flag, so the
    // rejection has to come from the paged rule.
    BackendArgs split = args;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", split.device));
    TEST_ASSERT(!gate_accepts(split, "qwen35", PlacementBackend::Cuda));
    TEST_ASSERT(!gate_accepts(split, "deepseek4", PlacementBackend::Cuda));

    BackendArgs remote_shard = args;
    remote_shard.remote_target_shard.ipc_bin = "/usr/bin/target-shard";
    TEST_ASSERT(!gate_accepts(
        remote_shard, "qwen35", PlacementBackend::Cuda));

    for (BackendArgs * relaxed : {&split, &remote_shard}) {
        relaxed->paged_attention = false;
        TEST_ASSERT(gate_accepts(
            *relaxed, "qwen35", PlacementBackend::Cuda));
    }
}

static void test_feature_gate_deepseek4_paged_reference_constraints() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.paged_attention = true;
    args.max_concurrency = 4;
    TEST_ASSERT(gate_accepts(args, "deepseek4", PlacementBackend::Cuda));

    args.max_concurrency = 5;
    TEST_ASSERT(!gate_accepts(args, "deepseek4", PlacementBackend::Cuda));
    args.max_concurrency = 4;
    args.ds4_fused_decode = true;
    TEST_ASSERT(!gate_accepts(args, "deepseek4", PlacementBackend::Cuda));
    args.ds4_fused_decode = false;
    args.ds4_prefill_mode = PrefillAttentionMode::Dense;
    TEST_ASSERT(!gate_accepts(args, "deepseek4", PlacementBackend::Cuda));
}

static void test_feature_gate_paged_attention_requires_plain_ar_decode() {
    BackendArgs base;
    base.model_path = "/nonexistent/model.gguf";
    base.paged_attention = true;

    BackendArgs draft = base;
    draft.draft_path = "/nonexistent/draft.gguf";
    TEST_ASSERT(!gate_accepts(draft, "qwen35", PlacementBackend::Cuda));

    BackendArgs ddtree = base;
    ddtree.ddtree_mode = true;
    TEST_ASSERT(!gate_accepts(ddtree, "qwen35", PlacementBackend::Cuda));

    BackendArgs windowed = base;
    windowed.fa_window = 4096;
    TEST_ASSERT(!gate_accepts(
        windowed, "qwen35", PlacementBackend::Cuda));

    BackendFeatureConfig pflash;
    pflash.pflash_enabled = true;
    pflash.pflash_drafter_configured = true;
    TEST_ASSERT(!gate_accepts(
        base, "qwen35", PlacementBackend::Cuda, pflash));

    BackendFeatureConfig kvflash;
    kvflash.kvflash_enabled = true;
    TEST_ASSERT(!gate_accepts(
        base, "qwen35", PlacementBackend::Cuda, kvflash));

    // The pool rounds max_ctx up to whole blocks, so both ends of the range
    // are rejected: nothing to allocate, and rounding that overflows int.
    BackendArgs empty_ctx = base;
    empty_ctx.device.max_ctx = 0;
    TEST_ASSERT(!gate_accepts(
        empty_ctx, "qwen35", PlacementBackend::Cuda));

    BackendArgs huge_ctx = base;
    huge_ctx.device.max_ctx = INT_MAX;
    TEST_ASSERT(!gate_accepts(
        huge_ctx, "qwen35", PlacementBackend::Cuda));

    BackendArgs max_ctx = base;
    max_ctx.device.max_ctx = INT_MAX - PAGED_BLOCK_SIZE + 1;
    TEST_ASSERT(gate_accepts(
        max_ctx, "qwen35", PlacementBackend::Cuda));

    // None of these are rules about paged attention itself: without the flag
    // every one of them is a supported qwen35 launch.
    for (BackendArgs * args : {&draft, &ddtree, &windowed, &empty_ctx,
                               &huge_ctx}) {
        args->paged_attention = false;
        TEST_ASSERT(gate_accepts(*args, "qwen35", PlacementBackend::Cuda));
    }
}

static void test_feature_gate_parallel_and_kv_pool_rules() {
    // A valid paged qwen35 monolithic launch is the baseline every rule
    // below perturbs.
    BackendArgs paged;
    paged.model_path = "/nonexistent/model.gguf";
    paged.paged_attention = true;

    // --max-concurrency is validated even without any other flag: zero decode
    // slots is meaningless on every backend.
    BackendArgs plain;
    plain.model_path = "/nonexistent/model.gguf";
    plain.max_concurrency = 0;
    TEST_ASSERT(!gate_accepts(plain, "qwen35", PlacementBackend::Cuda));
    plain.max_concurrency = 1;
    TEST_ASSERT(gate_accepts(plain, "qwen35", PlacementBackend::Cuda));

    // More than one slot exists only in the paged qwen35 backend.
    BackendArgs dense;
    dense.model_path = "/nonexistent/model.gguf";
    dense.max_concurrency = 2;
    TEST_ASSERT(!gate_accepts(dense, "qwen35", PlacementBackend::Cuda));

    BackendArgs parallel = paged;
    parallel.max_concurrency = 2;
    TEST_ASSERT(gate_accepts(
        parallel, "qwen35", PlacementBackend::Cuda));

    // Slot counts need not be powers of two. Decode graph buckets pad via
    // active_slot_ids rather than changing the physical slot allocation.
    parallel.max_concurrency = 3;
    TEST_ASSERT(gate_accepts(
        parallel, "qwen35", PlacementBackend::Cuda));

    // 64 slots is the top of the supported range.
    parallel.max_concurrency = 64;
    TEST_ASSERT(gate_accepts(
        parallel, "qwen35", PlacementBackend::Cuda));
    parallel.max_concurrency = 65;
    TEST_ASSERT(!gate_accepts(
        parallel, "qwen35", PlacementBackend::Cuda));

    // --kv-pool-tokens sizes the shared pool, so it needs slots to share.
    BackendArgs pool = paged;
    pool.kv_pool_tokens = 4096;
    TEST_ASSERT(!gate_accepts(pool, "qwen35", PlacementBackend::Cuda));
    pool.max_concurrency = 2;
    TEST_ASSERT(gate_accepts(pool, "qwen35", PlacementBackend::Cuda));

    // The pool must hold at least one block, and stay addressable with int
    // after rounding up to whole blocks.
    pool.kv_pool_tokens = PAGED_BLOCK_SIZE - 1;
    TEST_ASSERT(!gate_accepts(pool, "qwen35", PlacementBackend::Cuda));
    pool.kv_pool_tokens = PAGED_BLOCK_SIZE;
    TEST_ASSERT(gate_accepts(pool, "qwen35", PlacementBackend::Cuda));
    const long long max_pool_tokens =
        ((long long)INT_MAX - PAGED_BLOCK_SIZE) /
        PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE;
    pool.kv_pool_tokens = max_pool_tokens + 1;
    TEST_ASSERT(!gate_accepts(pool, "qwen35", PlacementBackend::Cuda));
    pool.kv_pool_tokens = max_pool_tokens;
    TEST_ASSERT(gate_accepts(pool, "qwen35", PlacementBackend::Cuda));

    // The automatic pool is memory-derived, so a logical slot/context product
    // larger than the physical tensor address space is legal.
    BackendArgs overflow = paged;
    overflow.max_concurrency = 2;
    overflow.device.max_ctx = 1 << 30;
    TEST_ASSERT(gate_accepts(
        overflow, "qwen35", PlacementBackend::Cuda));
    // An explicit addressable pool remains accepted as well.
    overflow.kv_pool_tokens = 1 << 20;
    TEST_ASSERT(gate_accepts(
        overflow, "qwen35", PlacementBackend::Cuda));
}

// ── Inert-flag warnings ─────────────────────────────────────────────────
// Warnings must never gate admission, so each case also asserts the same
// configuration passes check_feature_compatibility().

static std::vector<std::string> warn_result(
    const BackendArgs & args,
    const std::string & arch,
    const BackendFeatureConfig & features = {}) {
    TEST_ASSERT(check_feature_compatibility(
        args, features, arch, compiled_placement_backend(),
        compiled_placement_backend()).empty());
    return collect_feature_warnings(args, features, arch);
}

static bool warns_about(const std::vector<std::string> & warnings,
                        const std::string & flag) {
    for (const std::string & w : warnings) {
        if (w.rfind(flag + " ignored:", 0) == 0) return true;
    }
    return false;
}

static void test_feature_warnings_silent_when_supported() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";
    args.ddtree_mode = true;
    args.fa_window = 512;
    args.draft_swa_window = 2048;
    // qwen35 forwards every one of these.
    TEST_ASSERT(warn_result(args, "qwen35").empty());
}

static void test_feature_warnings_report_inert_draft() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";

    // qwen3 and deepseek4 never forward a draft model.
    TEST_ASSERT(warns_about(warn_result(args, "qwen3"), "--draft"));
    TEST_ASSERT(warns_about(warn_result(args, "deepseek4"), "--draft"));
    // laguna and gemma4 forward it only when monolithic.
    TEST_ASSERT(!warns_about(warn_result(args, "laguna"), "--draft"));
    TEST_ASSERT(!warns_about(warn_result(args, "gemma4"), "--draft"));

    BackendArgs split = args;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", split.device));
    const std::vector<std::string> w = collect_feature_warnings(split, {}, "laguna");
    TEST_ASSERT(warns_about(w, "--draft"));
    TEST_ASSERT(w[0].find("single-device placement") != std::string::npos);
}

static void test_feature_warnings_report_inert_decode_tunables() {
    BackendArgs ddtree;
    ddtree.model_path = "/nonexistent/model.gguf";
    ddtree.ddtree_mode = true;
    TEST_ASSERT(warns_about(warn_result(ddtree, "gemma4"), "--ddtree"));
    TEST_ASSERT(!warns_about(warn_result(ddtree, "laguna"), "--ddtree"));

    BackendArgs vw;
    vw.model_path = "/nonexistent/model.gguf";
    vw.verify_width = 8;
    TEST_ASSERT(!warns_about(warn_result(vw, "laguna"), "--verify-width"));
    TEST_ASSERT(warns_about(warn_result(vw, "qwen35"), "--verify-width"));

    BackendArgs fa;
    fa.model_path = "/nonexistent/model.gguf";
    fa.fa_window = 4096;
    // gemma4 honors --fa-window on both paths; laguna has no such option.
    TEST_ASSERT(!warns_about(warn_result(fa, "gemma4"), "--fa-window"));
    TEST_ASSERT(warns_about(warn_result(fa, "laguna"), "--fa-window"));

    BackendArgs swa;
    swa.model_path = "/nonexistent/model.gguf";
    swa.draft_swa_window = 2048;
    TEST_ASSERT(!warns_about(warn_result(swa, "qwen35moe"), "--draft-swa"));
    TEST_ASSERT(warns_about(warn_result(swa, "gemma4"), "--draft-swa"));
}

static void test_feature_warnings_report_inert_moe_options() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";

    BackendFeatureConfig moe_opts;
    moe_opts.routing_stats_requested = true;
    moe_opts.adaptive_experts_requested = true;

    TEST_ASSERT(warn_result(args, "laguna", moe_opts).empty());
    TEST_ASSERT(warn_result(args, "qwen35moe", moe_opts).empty());
    TEST_ASSERT(warn_result(args, "qwen35", moe_opts).size() == 2);
    TEST_ASSERT(warn_result(args, "deepseek4", moe_opts).size() == 2);
}

static void test_model_capability_tables() {
    // Table integrity: one row per architecture, no blanks, no duplicates.
    for (const ArchCapabilities & row : kArchCapabilities) {
        TEST_ASSERT(row.arch != nullptr && row.arch[0] != '\0');
        TEST_ASSERT(find_arch_capabilities(row.arch) == &row);
    }

    // arch_is_supported() must match create_backend()'s dispatch chain.
    for (const char * arch : {"qwen35", "qwen35moe", "laguna",
                              "qwen3", "gemma4", "deepseek4"}) {
        TEST_ASSERT(arch_is_supported(arch));
    }
    TEST_ASSERT(!arch_is_supported(""));
    TEST_ASSERT(!arch_is_supported("qwen36"));  // model_card has a branch; the factory does not
    TEST_ASSERT(!arch_is_supported("llama"));

    TEST_ASSERT(arch_has_expert_offload("laguna"));
    TEST_ASSERT(arch_has_expert_offload("qwen35moe"));
    TEST_ASSERT(!arch_has_expert_offload("qwen35"));
    // deepseek4 is mixture-of-experts but has no hot/cold offload path.
    TEST_ASSERT(!arch_has_expert_offload("deepseek4"));

    // Every capability predicate must be false for an architecture the
    // factory cannot build, so no rule can admit an unbuildable model.
    TEST_ASSERT(!arch_supports_layer_split("qwen36"));
    TEST_ASSERT(!arch_supports_remote_draft("qwen36"));
    TEST_ASSERT(!arch_supports_pflash_compression("qwen36"));
    TEST_ASSERT(!arch_supports_decode_draft("qwen36", false));
    TEST_ASSERT(!arch_supports_ddtree("qwen36", false));
    TEST_ASSERT(!arch_supports_verify_width("qwen36", false));
    TEST_ASSERT(!arch_supports_fa_window("qwen36", false));
    TEST_ASSERT(!arch_supports_draft_swa("qwen36", false));
    TEST_ASSERT(!arch_supports_paged_attention("qwen36", false));

    // Paged decode lives in the monolithic qwen35 backend alone.
    TEST_ASSERT(arch_supports_paged_attention("qwen35", false));
    TEST_ASSERT(!arch_supports_paged_attention("qwen35", true));
    TEST_ASSERT(!arch_supports_paged_attention("qwen35moe", false));
}

int main() {
    std::fprintf(stderr, "\n\u2500\u2500 Backend feature/architecture gate \u2500\u2500\n");
    RUN_TEST(test_feature_gate_accepts_plain_launch);
    RUN_TEST(test_feature_gate_rejects_undetected_arch);
    RUN_TEST(test_feature_gate_requires_compiled_target_backend);
    RUN_TEST(test_feature_gate_ipc_options_require_ipc_binary);
    RUN_TEST(test_feature_gate_mixed_draft_placement_requires_ipc);
    RUN_TEST(test_feature_gate_pflash_requires_drafter_and_supported_arch);
    RUN_TEST(test_feature_gate_validates_target_split_topology);
    RUN_TEST(test_feature_gate_tensor_parallel_requirements);
    RUN_TEST(test_feature_gate_ds4_prefill_requires_deepseek4);
    RUN_TEST(test_feature_gate_approximate_ds4_prefill_requires_local_hip);
    RUN_TEST(test_feature_gate_ds4_decode_options_require_monolithic_hip);
    RUN_TEST(test_feature_gate_remote_draft_requires_supported_arch);
    RUN_TEST(test_feature_gate_layer_split_requires_supported_arch);
    RUN_TEST(test_feature_gate_paged_attention_requires_supported_monolithic);
    RUN_TEST(test_feature_gate_deepseek4_paged_reference_constraints);
    RUN_TEST(test_feature_gate_paged_attention_requires_plain_ar_decode);
    RUN_TEST(test_feature_gate_parallel_and_kv_pool_rules);
    RUN_TEST(test_feature_warnings_silent_when_supported);
    RUN_TEST(test_feature_warnings_report_inert_draft);
    RUN_TEST(test_feature_warnings_report_inert_decode_tunables);
    RUN_TEST(test_feature_warnings_report_inert_moe_options);
    RUN_TEST(test_model_capability_tables);

    std::fprintf(stderr,
        "\n\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n"
        " Results: %d assertions, %d failures\n"
        "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n",
        test_count, test_failures);
    if (test_failures == 0) std::fprintf(stderr, "ALL PASSED\n");
    return test_failures == 0 ? 0 : 1;
}
