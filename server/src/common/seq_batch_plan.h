// Model-neutral prefill planning for continuous batching.
//
// The HTTP scheduler owns policy (arrival order, fairness, token budgets),
// while a SeqEngine owns model state and lowers selected slices into its graph
// representation. Keeping this helper GPU- and model-free makes the policy
// testable without Qwen, paged attention, or a device runtime.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

struct PrefillCandidate {
    int slot = -1;
    uint64_t order = 0;
};

struct PrefillSlice {
    int slot = -1;
    int max_tokens = 0;
};

// Hard limits advertised by the engine. They describe what one engine step
// can execute at a particular decode width, not which request should run.
struct StepPlanLimits {
    int max_prefill_sequences = 1;
    int max_prefill_tokens_per_sequence = 512;
    int max_prefill_tokens_total = 512;
};

// Soft serving policy owned by the scheduler. Values are deliberately not
// tied to a model family or GPU architecture and may be exposed as runtime
// configuration by the server.
struct StepPlanPolicy {
    int prefill_quantum = 512;
    int prefill_token_budget = 4096;
    int mixed_prefill_token_budget = 2048;
};

// Select the oldest eligible sequences, then distribute the step's token
// budget in round-robin quanta. Strict FIFO determines cohort membership;
// round-robin allocation prevents the first member from taking the entire
// budget. The engine may consume fewer tokens and reports actual progress.
inline std::vector<PrefillSlice> plan_prefill_slices(
        const std::vector<PrefillCandidate> & candidates,
        bool has_decode,
        const StepPlanLimits & limits,
        const StepPlanPolicy & policy,
        size_t round_robin_start = 0) {
    std::vector<PrefillSlice> slices;
    if (candidates.empty() || limits.max_prefill_sequences <= 0 ||
        limits.max_prefill_tokens_per_sequence <= 0 ||
        limits.max_prefill_tokens_total <= 0 ||
        policy.prefill_quantum <= 0) {
        return slices;
    }

    int budget = has_decode ? policy.mixed_prefill_token_budget
                            : policy.prefill_token_budget;
    budget = std::min(budget, limits.max_prefill_tokens_total);
    if (budget <= 0) return slices;

    std::vector<PrefillCandidate> ordered = candidates;
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const PrefillCandidate & a, const PrefillCandidate & b) {
            if (a.order != b.order) return a.order < b.order;
            return a.slot < b.slot;
        });

    const int selected = std::min(
        (int)ordered.size(), limits.max_prefill_sequences);
    slices.reserve((size_t)selected);
    for (int i = 0; i < selected; ++i) {
        slices.push_back({ordered[(size_t)i].slot, 0});
    }

    while (budget > 0) {
        bool granted = false;
        for (size_t offset = 0; offset < slices.size(); ++offset) {
            const size_t idx =
                (round_robin_start + offset) % slices.size();
            PrefillSlice & slice = slices[idx];
            const int room =
                limits.max_prefill_tokens_per_sequence - slice.max_tokens;
            if (room <= 0) continue;
            const int grant = std::min({policy.prefill_quantum, room, budget});
            if (grant <= 0) continue;
            slice.max_tokens += grant;
            budget -= grant;
            granted = true;
            if (budget == 0) break;
        }
        if (!granted) break;
    }

    slices.erase(
        std::remove_if(slices.begin(), slices.end(),
            [](const PrefillSlice & slice) { return slice.max_tokens <= 0; }),
        slices.end());
    return slices;
}

}  // namespace dflash::common
