// Shared paged-attention sizing and compatibility rules.

#pragma once

#include <climits>
#include <string>

namespace dflash::common {

constexpr int PAGED_BLOCK_SIZE = 16;

constexpr int paged_block_count(int max_ctx) {
    return (max_ctx + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;
}

constexpr int paged_token_capacity(int max_ctx) {
    return paged_block_count(max_ctx) * PAGED_BLOCK_SIZE;
}

// TODO(#558): once the cross-feature gate lands in main, move these rules
// into check_feature_compatibility() and delete both this struct and the
// function below; the geometry helpers above stay. Six of the eight rules
// read straight from BackendArgs plus the gate's `arch` parameter. Three
// things need doing by hand:
//   - `pflash` and `kvflash` need new BackendArgs fields. They are the
//     reason both non-server callers construct this struct with them
//     hardcoded to false — neither layer can see ServerConfig or the env.
//   - `accelerator_build` needs has_compiled_accelerator_backend() moved
//     out of backend_factory.cpp into placement/placement_backend.h, next
//     to compiled_placement_backend().
//   - the prefix/prefill/disk-cache rule in server_main stays where it is:
//     it is pure ServerConfig, and naming one feature is not enough to earn
//     a place in the gate.
// The 256-wide K/V head check in Qwen35Backend::init() does not move either
// — it needs loaded tensor dims, which GgufModelInfo does not carry.
struct PagedAttentionOptions {
    bool enabled = false;
    const char * architecture = nullptr;  // null when not known yet
    bool accelerator_build = true;
    bool layer_split = false;
    bool remote_target_shard = false;
    bool draft = false;
    bool remote_draft = false;
    bool ddtree = false;
    int fa_window = 0;
    bool pflash = false;
    bool kvflash = false;
    int max_ctx = 0;
};

inline std::string validate_paged_attention_options(
    const PagedAttentionOptions & options) {
    if (!options.enabled) return {};
    if (options.architecture &&
        std::string(options.architecture) != "qwen35") {
        return "supports only Qwen3.5/Qwen3.6 dense targets";
    }
    if (!options.accelerator_build) {
        return "requires a CUDA or HIP build";
    }
    if (options.layer_split || options.remote_target_shard) {
        return "requires one local target device";
    }
    if (options.draft || options.remote_draft || options.ddtree) {
        return "requires autoregressive decode without a draft or DDTree";
    }
    if (options.fa_window != 0) {
        return "requires full attention (--fa-window 0)";
    }
    if (options.pflash) {
        return "cannot be combined with PFlash prefill compression";
    }
    if (options.kvflash) {
        return "cannot be combined with KVFlash";
    }
    if (options.max_ctx <= 0 ||
        options.max_ctx > INT_MAX - PAGED_BLOCK_SIZE + 1) {
        return "requires max_ctx in [1, INT_MAX - PAGED_BLOCK_SIZE + 1]";
    }
    return {};
}

}  // namespace dflash::common
