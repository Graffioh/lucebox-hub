// Model-neutral prefill planning for continuous batching.
//
// The HTTP scheduler owns arrival order and fairness, while a SeqEngine owns
// model state, advertises the useful work envelope for a step, and lowers the
// selected slices into its graph representation. Keeping this helper GPU- and
// model-free makes the policy testable without Qwen, paged attention, or a
// device runtime.

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

// Limits advertised by the engine. They describe the useful work envelope for
// one engine step at a particular decode width, not which request should run.
struct StepPlanLimits {
    int max_prefill_sequences = 1;
    int max_prefill_tokens_per_sequence = 512;
    int max_prefill_tokens_total = 512;
    int prefill_allocation_quantum = 512;
};

// Select the oldest eligible sequences, then distribute the step's token
// capacity in engine-owned quanta. Strict FIFO determines cohort membership;
// a rotating cursor prevents the oldest member from always winning a partial
// final round. The engine may consume fewer tokens and reports actual progress.
inline std::vector<PrefillSlice> plan_prefill_slices(
        const std::vector<PrefillCandidate> & candidates,
        const StepPlanLimits & limits,
        size_t round_robin_start = 0) {
    std::vector<PrefillSlice> slices;
    if (candidates.empty() || limits.max_prefill_sequences <= 0 ||
        limits.max_prefill_tokens_per_sequence <= 0 ||
        limits.max_prefill_tokens_total <= 0 ||
        limits.prefill_allocation_quantum <= 0) {
        return slices;
    }

    std::vector<PrefillCandidate> ordered = candidates;
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const PrefillCandidate & a, const PrefillCandidate & b) {
            if (a.order != b.order) return a.order < b.order;
            return a.slot < b.slot;
        });

    const int selected = std::min(
        (int)ordered.size(), limits.max_prefill_sequences);
    int budget = std::min(
        limits.max_prefill_tokens_total,
        selected * limits.max_prefill_tokens_per_sequence);
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
            const int grant = std::min({
                limits.prefill_allocation_quantum, room, budget});
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
