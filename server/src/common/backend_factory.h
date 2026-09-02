// Backend planning and arch-detecting ModelBackend construction.
//
// BackendArgs is mutable input. prepare_backend() consumes it, resolves model
// and placement facts, normalizes backend policy, and returns an immutable
// BackendPlan. create_backend() accepts only that plan.

#pragma once

#include "backend_args.h"
#include "gguf_inspect.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace dflash::common {

namespace detail {
class BackendPlanBuilder;
}

class BackendPlan;
class BackendRuntime;
class ModelBackend;

// The sole construction entry point. It consumes a valid plan and returns a
// runtime that keeps the plan's owned path storage alive.
std::unique_ptr<BackendRuntime> create_backend(BackendPlan plan);

class BackendPlan final {
public:
    BackendPlan(BackendPlan &&) noexcept = default;
    BackendPlan & operator=(BackendPlan &&) = delete;
    BackendPlan(const BackendPlan &) = delete;
    BackendPlan & operator=(const BackendPlan &) = delete;

    const BackendArgs & args() const { return args_; }
    const GgufModelInfo & model() const { return model_; }
    const std::string & arch() const { return model_.arch; }
    const std::vector<std::string> & warnings() const { return warnings_; }

private:
    enum class SpeclaEnvironmentAction {
        Preserve,
        Enable,
        Disable,
    };

    BackendPlan() = default;

    BackendArgs args_;
    GgufModelInfo model_;
    std::vector<std::string> warnings_;
    SpeclaEnvironmentAction specla_environment_ =
        SpeclaEnvironmentAction::Preserve;

    friend class detail::BackendPlanBuilder;
    friend std::unique_ptr<BackendRuntime> create_backend(BackendPlan plan);
};

enum class BackendPreparationError {
    InvalidRequest,
    ModelInspection,
    FeatureCompatibility,
};

struct BackendPreparationFailure {
    BackendPreparationError error;
    std::string message;
    std::vector<std::string> warnings;
};

using BackendPreparation =
    std::variant<BackendPlan, BackendPreparationFailure>;

// Consumes the mutable request. Successful preparation performs one GGUF
// inspection and returns the only value accepted by backend construction.
BackendPreparation prepare_backend(
    BackendArgs args,
    BackendAdmissionContext admission = {});

// Owns the immutable plan together with the backend. Several existing backend
// config structs retain c_str() pointers, so the plan must share their
// lifetime and be destroyed after the backend.
class BackendRuntime final {
public:
    ~BackendRuntime();

    BackendRuntime(const BackendRuntime &) = delete;
    BackendRuntime & operator=(const BackendRuntime &) = delete;
    BackendRuntime(BackendRuntime &&) = delete;
    BackendRuntime & operator=(BackendRuntime &&) = delete;

    ModelBackend & backend() { return *backend_; }
    const ModelBackend & backend() const { return *backend_; }
    const BackendPlan & plan() const { return plan_; }

private:
    explicit BackendRuntime(BackendPlan plan);

    // Declaration order is intentional: backend_ is destroyed before plan_,
    // keeping its borrowed path pointers valid through backend teardown.
    BackendPlan plan_;
    std::unique_ptr<ModelBackend> backend_;

    friend std::unique_ptr<BackendRuntime> create_backend(BackendPlan plan);
};

}  // namespace dflash::common
