// Backend factory implementation.

#include "backend_factory.h"
#include "backend_plan_internal.h"
#include "gguf_inspect.h"
#include "model_capabilities.h"
#include "platform_env.h"

#include "qwen35_backend.h"
#include "qwen35moe_backend.h"
#include "bailingmoe3_backend.h"
#include "laguna_backend.h"
#include "laguna_layer_split_adapter.h"
#include "qwen3_backend.h"
#include "gemma4_backend.h"
#include "gemma4_layer_split_adapter.h"
#include "deepseek4_backend.h"
#include "deepseek4_layer_split_adapter.h"
#include "layer_split_backend.h"
#include "qwen35_layer_split_adapter.h"

#include <cstdio>
#include <algorithm>
#include <type_traits>

namespace dflash::common {

namespace {

// ─── Capability table ↔ dispatch cross-check ────────────────────────────
// Every option an architecture can receive arrives through a named field on
// its backend config struct. Detecting whether that field exists lets the
// compiler verify model_capabilities.h against the structs this file feeds:
// a row claiming support an architecture has nowhere to store, or a config
// carrying a field the table calls unsupported, fails the build here.
//
// This is the strongest check the language allows. It cannot see whether the
// dispatch below actually *assigns* a field it has — that is what the unit
// tests and the table's dispatch-matching row order are for.

#define DFLASH_ARCH_FIELD_TRAIT(trait_name, field_name)                  \
    template <class T, class = void>                                     \
    struct trait_name : std::false_type {};                              \
    template <class T>                                                   \
    struct trait_name<T, std::void_t<decltype(T::field_name)>>           \
        : std::true_type {}

DFLASH_ARCH_FIELD_TRAIT(has_draft_path,        draft_path);
DFLASH_ARCH_FIELD_TRAIT(has_draft_block_size,  draft_block_size);
DFLASH_ARCH_FIELD_TRAIT(has_fa_window,         fa_window);
DFLASH_ARCH_FIELD_TRAIT(has_verify_width,      verify_width);
DFLASH_ARCH_FIELD_TRAIT(has_draft_swa,         draft_swa_window);
DFLASH_ARCH_FIELD_TRAIT(has_ddtree_mode,       ddtree_mode);
DFLASH_ARCH_FIELD_TRAIT(has_max_verify_tokens, max_verify_tokens);
DFLASH_ARCH_FIELD_TRAIT(has_paged_attention,   paged_attention);

#undef DFLASH_ARCH_FIELD_TRAIT

// DDTree reaches qwen35's layer-split path as a max_verify_tokens budget
// rather than a ddtree_mode flag, so either field counts as a carrier.
template <class T>
struct has_ddtree : std::bool_constant<has_ddtree_mode<T>::value ||
                                       has_max_verify_tokens<T>::value> {};

// Architectures with no layer-split adapter. Pairing one of these with the
// split half of a check requires the row to be Monolithic or Never, which
// table_split_coherent() already guarantees.
struct NoLayerSplitConfig {};

constexpr bool monolithic_carries(FeatureSupport support) {
    return support != FeatureSupport::Never;
}
constexpr bool layer_split_carries(FeatureSupport support) {
    return support == FeatureSupport::Both;
}

#define DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, trait, field)        \
    static_assert(                                                            \
        trait<Mono>::value ==                                                 \
            monolithic_carries(arch_capabilities(arch_name).field),           \
        arch_name ": monolithic config and capability table disagree on "     \
        #field);                                                              \
    static_assert(                                                            \
        trait<Split>::value ==                                                \
            layer_split_carries(arch_capabilities(arch_name).field),          \
        arch_name ": layer-split config and capability table disagree on "    \
        #field)

#define DFLASH_CHECK_ARCH(arch_name, Mono, Split)                             \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_draft_path,   decode_draft); \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_ddtree,       ddtree);       \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_verify_width, verify_width); \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_fa_window,    fa_window);    \
    DFLASH_CHECK_ARCH_OPTION(arch_name, Mono, Split, has_draft_swa,    draft_swa)

DFLASH_CHECK_ARCH("qwen35",    Qwen35Config,          Qwen35LayerSplitAdapterConfig);
DFLASH_CHECK_ARCH("qwen35moe", Qwen35Config,          NoLayerSplitConfig);
DFLASH_CHECK_ARCH("bailingmoe3", BailingMoe3Config,   NoLayerSplitConfig);
DFLASH_CHECK_ARCH("laguna",    LagunaBackendArgs,     LagunaLayerSplitAdapterConfig);
DFLASH_CHECK_ARCH("qwen3",     Qwen3BackendConfig,    NoLayerSplitConfig);
DFLASH_CHECK_ARCH("gemma4",    Gemma4BackendConfig,   Gemma4LayerSplitAdapterConfig);
DFLASH_CHECK_ARCH("deepseek4", DeepSeek4BackendConfig, DeepSeek4LayerSplitAdapterConfig);

// These sit outside the bundle because the field-presence trait cannot
// separate qwen35 from qwen35moe: they share Qwen35Config, while the factory
// forwards both fields only for dense qwen35. Pairing the MoE Never rows with
// that shared struct would fail a check that is really about dispatch.
DFLASH_CHECK_ARCH_OPTION("qwen35", Qwen35Config, Qwen35LayerSplitAdapterConfig,
                         has_paged_attention, paged_attn);
DFLASH_CHECK_ARCH_OPTION("qwen35", Qwen35Config, Qwen35LayerSplitAdapterConfig,
                         has_draft_block_size, draft_block_size);

#undef DFLASH_CHECK_ARCH
#undef DFLASH_CHECK_ARCH_OPTION

std::unique_ptr<ModelBackend> construct_backend(
    const BackendArgs & args,
    const std::string & arch) {
    if (arch == "qwen35") {
        if (args.device.is_layer_split()) {
            Qwen35LayerSplitAdapterConfig cfg;
            cfg.target_path        = args.model_path.c_str();
            cfg.draft_path         = args.draft_path
                ? args.draft_path->c_str()
                : nullptr;
            cfg.device             = args.device;
            cfg.draft_gpu          = args.draft_device.gpu;
            cfg.remote_draft       = args.remote_draft;
            cfg.remote_target_shard = args.remote_target_shard;
            cfg.fa_window          = args.fa_window;
            cfg.kq_stride_pad      = args.kq_stride_pad;
            cfg.draft_swa_window   = args.draft_swa_window;
            cfg.draft_ctx_max      = args.draft_ctx_max;
            cfg.chunk              = args.chunk;
            cfg.max_verify_tokens  = args.ddtree_mode
                ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, args.ddtree_budget + 1)
                : DFLASH27B_DRAFT_BLOCK_SIZE;
            cfg.run_dflash         = args.draft_path.has_value();

            auto adapter = std::make_unique<Qwen35LayerSplitAdapter>(cfg);
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(qwen35) init failed\n");
                return nullptr;
            }
            return backend;
        }

        Qwen35Config cfg;
        cfg.target_path        = args.model_path.c_str();
        cfg.draft_path         = args.draft_path
            ? args.draft_path->c_str()
            : nullptr;
        cfg.device             = args.device;
        cfg.draft_gpu          = args.draft_device.gpu;
        cfg.remote_draft       = args.remote_draft;
        cfg.stream_fd          = args.stream_fd;
        cfg.fa_window          = args.fa_window;
        cfg.paged_attention    = args.paged_attention;
        cfg.max_concurrency    = args.max_concurrency;
        cfg.kv_pool_tokens     = args.kv_pool_tokens;
        cfg.kq_stride_pad      = args.kq_stride_pad;
        cfg.draft_block_size   = args.draft_block_size;
        cfg.draft_swa_window   = args.draft_swa_window;
        cfg.draft_ctx_max      = args.draft_ctx_max;
        cfg.fast_rollback      = args.fast_rollback;
        cfg.seq_verify         = args.seq_verify;
        cfg.ddtree_mode        = args.ddtree_mode;
        cfg.ddtree_budget      = args.ddtree_budget;
        cfg.ddtree_temp        = args.ddtree_temp;
        cfg.ddtree_chain_seed  = args.ddtree_chain_seed;
        cfg.ddtree_tau         = args.ddtree_tau;
        cfg.use_feature_mirror = args.use_feature_mirror;

        auto backend = std::make_unique<Qwen35Backend>(cfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "qwen35moe") {
        Qwen35Config cfg;
        cfg.target_path        = args.model_path.c_str();
        cfg.draft_path         = args.draft_path
            ? args.draft_path->c_str()
            : nullptr;
        cfg.device             = args.device;
        cfg.draft_gpu          = args.draft_device.gpu;
        cfg.stream_fd          = args.stream_fd;
        cfg.fa_window          = args.fa_window;
        cfg.kq_stride_pad      = args.kq_stride_pad;
        cfg.draft_swa_window   = args.draft_swa_window;
        cfg.draft_ctx_max      = args.draft_ctx_max;
        cfg.fast_rollback      = args.fast_rollback;
        cfg.seq_verify         = args.seq_verify;
        cfg.ddtree_mode        = args.ddtree_mode;
        cfg.ddtree_budget      = args.ddtree_budget;
        cfg.ddtree_temp        = args.ddtree_temp;
        cfg.ddtree_chain_seed  = args.ddtree_chain_seed;
        cfg.ddtree_tau         = args.ddtree_tau;
        cfg.use_feature_mirror = args.use_feature_mirror;

        auto backend = std::make_unique<Qwen35MoeBackend>(cfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35MoeBackend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "bailingmoe3") {
        BailingMoe3Config cfg;
        cfg.model_path = args.model_path.c_str();
        cfg.device = args.device;
        cfg.stream_fd = args.stream_fd;

        auto backend = std::make_unique<BailingMoe3Backend>(cfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] BailingMoe3Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "laguna") {
        if (args.device.is_layer_split()) {
            LagunaLayerSplitAdapterConfig cfg;
            cfg.target_path = args.model_path.c_str();
            cfg.device      = args.device;
            cfg.remote_target_shard = args.remote_target_shard;
            cfg.chunk       = args.chunk;

            auto adapter = std::make_unique<LagunaLayerSplitAdapter>(cfg);
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(laguna) init failed\n");
                return nullptr;
            }
            return backend;
        }

        LagunaBackendArgs lcfg;
        lcfg.target_path = args.model_path.c_str();
        lcfg.draft_path  = args.draft_path
            ? args.draft_path->c_str()
            : "";
        lcfg.draft_gpu   = args.draft_device.gpu;
        lcfg.draft_ctx_max = args.draft_ctx_max;
        lcfg.ddtree_mode = args.ddtree_mode;
        lcfg.ddtree_budget = args.ddtree_budget;
        lcfg.ddtree_temp = args.ddtree_temp;
        lcfg.verify_width = args.verify_width;
        lcfg.device      = args.device;
        lcfg.max_ctx     = args.device.max_ctx;
        lcfg.chunk       = args.chunk;
        // kv_type defaults to Q8_0 in LagunaBackendArgs

        auto backend = std::make_unique<LagunaBackend>(lcfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] LagunaBackend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "qwen3") {
        Qwen3BackendConfig qcfg;
        qcfg.model_path = args.model_path.c_str();
        qcfg.device     = args.device;
        qcfg.stream_fd  = args.stream_fd;
        qcfg.chunk      = args.chunk;

        auto backend = std::make_unique<Qwen3Backend>(qcfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen3Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "gemma4") {
        if (args.device.is_layer_split()) {
            Gemma4LayerSplitAdapterConfig cfg;
            cfg.target_path = args.model_path.c_str();
            cfg.device      = args.device;
            cfg.remote_target_shard = args.remote_target_shard;
            cfg.chunk       = args.chunk;
            cfg.fa_window   = args.fa_window;

            auto adapter = std::make_unique<Gemma4LayerSplitAdapter>(cfg);
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(gemma4) init failed\n");
                return nullptr;
            }
            return backend;
        }

        Gemma4BackendConfig gcfg;
        gcfg.model_path    = args.model_path.c_str();
        gcfg.draft_path    = args.draft_path
            ? args.draft_path->c_str()
            : nullptr;
        gcfg.draft_gpu     = args.draft_device.gpu;
        gcfg.draft_ctx_max = args.draft_ctx_max;
        gcfg.device        = args.device;
        gcfg.stream_fd     = args.stream_fd;
        gcfg.chunk         = args.chunk;
        // Gemma4Backend reads this into its cache (gemma4_backend.cpp) exactly
        // as the layer-split adapter does; leaving it unset silently dropped
        // --fa-window on single-device gemma4.
        gcfg.fa_window     = args.fa_window;

        auto backend = std::make_unique<Gemma4Backend>(gcfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Gemma4Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "deepseek4") {
        // Approximate prefill and the fused decode options are gated against
        // non-monolithic-HIP placement in check_feature_compatibility().

        // A single local device uses the monolithic backend. Reserve the
        // layer-split adapter for explicit multi-device placement or remote
        // target shards.
        if (!args.device.is_layer_split() &&
            !args.remote_target_shard.enabled()) {
            DeepSeek4BackendConfig cfg;
            cfg.model_path = args.model_path.c_str();
            cfg.device     = args.device;
            cfg.stream_fd  = args.stream_fd;
            cfg.max_ctx    = args.device.max_ctx;
            cfg.chunk      = args.chunk;
            cfg.expert_top_k = args.ds4_expert_top_k;
            cfg.fused_decode = args.ds4_fused_decode;
            cfg.fused_verify_f16_kv = args.ds4_fused_verify_f16_kv;
            cfg.prefill_mode = args.ds4_prefill_mode;

            auto backend = std::make_unique<DeepSeek4Backend>(cfg);
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] DeepSeek4Backend init failed\n");
                return nullptr;
            }
            return backend;
        }

        // Explicit local splits and CUDA/HIP remote splits use the adapter.
        DeepSeek4LayerSplitAdapterConfig cfg;
        cfg.target_path        = args.model_path.c_str();
        cfg.device             = args.device;
        cfg.remote_target_shard = args.remote_target_shard;
        cfg.chunk              = args.chunk;

        auto adapter = std::make_unique<DeepSeek4LayerSplitAdapter>(cfg);
        auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] LayerSplitBackend(deepseek4) init failed\n");
            return nullptr;
        }
        return backend;

    } else {
        std::fprintf(stderr, "[backend_factory] unsupported architecture: %s\n",
                     arch.c_str());
        return nullptr;
    }
}

}  // namespace

BackendPreparation prepare_backend(
    BackendArgs args,
    BackendAdmissionContext admission) {
    if (args.model_path.empty()) {
        return BackendPreparationFailure{
            BackendPreparationError::InvalidRequest,
            "model_path is empty",
            {}};
    }

    GgufModelInfo model = inspect_gguf_model_info(args.model_path.c_str());
    return detail::BackendPlanBuilder::resolve(
        std::move(args),
        std::move(admission),
        std::move(model),
        compiled_placement_backend());
}

BackendRuntime::BackendRuntime(BackendPlan plan)
    : plan_(std::move(plan)) {}

BackendRuntime::~BackendRuntime() = default;

std::unique_ptr<BackendRuntime> create_backend(BackendPlan plan) {
    auto runtime = std::unique_ptr<BackendRuntime>(
        new BackendRuntime(std::move(plan)));

    switch (runtime->plan_.specla_environment_) {
        case BackendPlan::SpeclaEnvironmentAction::Preserve:
            break;
        case BackendPlan::SpeclaEnvironmentAction::Enable:
            set_environment_variable("DFLASH_SPECLA", "1", true);
            if (runtime->plan_.args_.specla_top_k_explicit) {
                const std::string top_k =
                    std::to_string(runtime->plan_.args_.specla_top_k);
                set_environment_variable(
                    "DFLASH_SPECLA_TOPK", top_k.c_str(), true);
            }
            break;
        case BackendPlan::SpeclaEnvironmentAction::Disable:
            unset_environment_variable("DFLASH_SPECLA");
            break;
    }

    std::fprintf(
        stderr,
        "[backend_factory] detected arch=%s\n",
        runtime->plan_.arch().c_str());
    runtime->backend_ = construct_backend(
        runtime->plan_.args_, runtime->plan_.arch());
    if (!runtime->backend_) {
        return nullptr;
    }
    return runtime;
}

}  // namespace dflash::common
