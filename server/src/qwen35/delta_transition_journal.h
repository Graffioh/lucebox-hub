#pragma once

#include <cstddef>
#include <vector>

namespace dflash::qwen35 {

// Host-side contract for the compact journal emitted by a future GDN verify
// kernel. The persistent state uses the kernel's transposed layout:
// state[col * rows + row]. A transition contains exactly the values needed to
// repeat the state update, but none of the model projections:
//
//     state' = gate * state + key (outer-product) delta
//
// In scalar mode gate has one value. In row-wise mode it has one value per
// state row. key and delta are the normalized/resolved values produced by
// verification; in particular, delta is captured after the state-dependent
// k^T S reduction. Therefore this journal is valid only for a chain replayed
// from the same base recurrent state.
enum class DeltaTransitionGateMode {
    Scalar,
    RowWise,
};

struct DeltaTransition {
    DeltaTransitionGateMode gate_mode = DeltaTransitionGateMode::Scalar;
    std::vector<float> gate;
    std::vector<float> key;
    std::vector<float> delta;
};

struct DeltaTransitionJournal {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<DeltaTransition> transitions;
};

// Resolve the state-dependent delta exactly once, as the verification kernel
// would. gate contains already-exponentiated multipliers (not raw/log gates).
// The output is unchanged when validation fails.
bool capture_delta_transition(
    const std::vector<float> & state,
    size_t rows,
    size_t cols,
    const std::vector<float> & key,
    const std::vector<float> & value,
    const std::vector<float> & gate,
    float beta,
    DeltaTransitionGateMode gate_mode,
    DeltaTransition & output);

// Apply one captured transition without evaluating projections or recomputing
// delta. The state is unchanged when validation fails.
bool apply_delta_transition(
    const DeltaTransition & transition,
    size_t rows,
    size_t cols,
    std::vector<float> & state);

// Commit transitions [0, accepted) to a persistent state. accepted == 0 is a
// no-op. Oversized or malformed prefixes fail before modifying state.
bool commit_delta_transition_prefix(
    const DeltaTransitionJournal & journal,
    size_t accepted,
    std::vector<float> & state);

// Compact recurrent journal footprint per head/token, excluding allocator
// alignment. Runtime storage may further share keys across grouped V heads.
size_t delta_transition_float_count(
    size_t rows,
    size_t cols,
    DeltaTransitionGateMode gate_mode);

} // namespace dflash::qwen35
