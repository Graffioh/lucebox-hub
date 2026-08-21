#include "qwen35/delta_transition_journal.h"

#include <cmath>
#include <limits>
#include <utility>

namespace dflash::qwen35 {
namespace {

bool matrix_size(size_t rows, size_t cols, size_t & elements) {
    if (rows == 0 || cols == 0 ||
        rows > std::numeric_limits<size_t>::max() / cols) {
        return false;
    }
    elements = rows * cols;
    return true;
}

bool transition_shape_valid(
    const DeltaTransition & transition,
    size_t rows,
    size_t cols) {
    const size_t expected_gate =
        transition.gate_mode == DeltaTransitionGateMode::Scalar ? 1 : rows;
    return transition.gate.size() == expected_gate &&
           transition.key.size() == rows &&
           transition.delta.size() == cols;
}

float gate_at(const DeltaTransition & transition, size_t row) {
    return transition.gate_mode == DeltaTransitionGateMode::Scalar
        ? transition.gate[0]
        : transition.gate[row];
}

} // namespace

bool capture_delta_transition(
    const std::vector<float> & state,
    size_t rows,
    size_t cols,
    const std::vector<float> & key,
    const std::vector<float> & value,
    const std::vector<float> & gate,
    float beta,
    DeltaTransitionGateMode gate_mode,
    DeltaTransition & output) {
    size_t state_elements = 0;
    const size_t expected_gate =
        gate_mode == DeltaTransitionGateMode::Scalar ? 1 : rows;
    if (!matrix_size(rows, cols, state_elements) ||
        state.size() != state_elements || key.size() != rows ||
        value.size() != cols || gate.size() != expected_gate) {
        return false;
    }

    DeltaTransition next;
    next.gate_mode = gate_mode;
    next.gate = gate;
    next.key = key;
    next.delta.resize(cols);

    for (size_t col = 0; col < cols; ++col) {
        float projection = 0.0f;
        for (size_t row = 0; row < rows; ++row) {
            const float row_gate = gate_mode == DeltaTransitionGateMode::Scalar
                ? 1.0f
                : gate[row];
            projection += row_gate * state[col * rows + row] * key[row];
        }
        const float scalar_gate = gate_mode == DeltaTransitionGateMode::Scalar
            ? gate[0]
            : 1.0f;
        next.delta[col] =
            (value[col] - scalar_gate * projection) * beta;
    }

    output = std::move(next);
    return true;
}

bool apply_delta_transition(
    const DeltaTransition & transition,
    size_t rows,
    size_t cols,
    std::vector<float> & state) {
    size_t state_elements = 0;
    if (!matrix_size(rows, cols, state_elements) ||
        state.size() != state_elements ||
        !transition_shape_valid(transition, rows, cols)) {
        return false;
    }

    for (size_t col = 0; col < cols; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t index = col * rows + row;
            state[index] = std::fma(
                transition.key[row], transition.delta[col],
                gate_at(transition, row) * state[index]);
        }
    }
    return true;
}

bool commit_delta_transition_prefix(
    const DeltaTransitionJournal & journal,
    size_t accepted,
    std::vector<float> & state) {
    size_t state_elements = 0;
    if (!matrix_size(journal.rows, journal.cols, state_elements) ||
        state.size() != state_elements ||
        accepted > journal.transitions.size()) {
        return false;
    }
    for (size_t i = 0; i < accepted; ++i) {
        if (!transition_shape_valid(
                journal.transitions[i], journal.rows, journal.cols)) {
            return false;
        }
    }
    for (size_t i = 0; i < accepted; ++i) {
        // Already validated, so this cannot partially fail.
        apply_delta_transition(
            journal.transitions[i], journal.rows, journal.cols, state);
    }
    return true;
}

size_t delta_transition_float_count(
    size_t rows,
    size_t cols,
    DeltaTransitionGateMode gate_mode) {
    const size_t gate_values =
        gate_mode == DeltaTransitionGateMode::Scalar ? 1 : rows;
    if (rows > std::numeric_limits<size_t>::max() - cols ||
        rows + cols > std::numeric_limits<size_t>::max() - gate_values) {
        return 0;
    }
    return rows + cols + gate_values;
}

} // namespace dflash::qwen35
