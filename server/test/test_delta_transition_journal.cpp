#include "qwen35/delta_transition_journal.h"
#include "host_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using dflash::qwen35::DeltaTransition;
using dflash::qwen35::DeltaTransitionGateMode;
using dflash::qwen35::DeltaTransitionJournal;
using dflash::qwen35::apply_delta_transition;
using dflash::qwen35::capture_delta_transition;
using dflash::qwen35::commit_delta_transition_prefix;
using dflash::qwen35::delta_transition_float_count;

static int g_checks = 0;

namespace {

struct RawStep {
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> gate;
    float beta = 0.0f;
};

bool near(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float scale = std::max(
            1.0f, std::max(std::fabs(lhs[i]), std::fabs(rhs[i])));
        if (std::fabs(lhs[i] - rhs[i]) > 4e-6f * scale) return false;
    }
    return true;
}

void replay_reference(
    std::vector<float> & state,
    size_t rows,
    size_t cols,
    const RawStep & step,
    DeltaTransitionGateMode gate_mode) {
    for (size_t col = 0; col < cols; ++col) {
        float projection = 0.0f;
        for (size_t row = 0; row < rows; ++row) {
            const float projection_gate =
                gate_mode == DeltaTransitionGateMode::RowWise
                    ? step.gate[row]
                    : 1.0f;
            projection += projection_gate *
                state[col * rows + row] * step.key[row];
        }
        const float scalar_gate =
            gate_mode == DeltaTransitionGateMode::Scalar
                ? step.gate[0]
                : 1.0f;
        const float delta =
            (step.value[col] - scalar_gate * projection) * step.beta;
        for (size_t row = 0; row < rows; ++row) {
            const size_t index = col * rows + row;
            const float update_gate =
                gate_mode == DeltaTransitionGateMode::Scalar
                    ? step.gate[0]
                    : step.gate[row];
            state[index] = std::fma(
                step.key[row], delta, update_gate * state[index]);
        }
    }
}

std::vector<float> initial_state(size_t rows, size_t cols, int layer) {
    std::vector<float> state(rows * cols);
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = 0.013f * static_cast<float>(i + 1) -
            0.07f * static_cast<float>(layer + 1);
    }
    return state;
}

std::vector<RawStep> make_steps(
    size_t rows,
    size_t cols,
    size_t count,
    int layer,
    DeltaTransitionGateMode gate_mode) {
    std::vector<RawStep> steps(count);
    for (size_t t = 0; t < count; ++t) {
        RawStep & step = steps[t];
        step.key.resize(rows);
        step.value.resize(cols);
        step.gate.resize(
            gate_mode == DeltaTransitionGateMode::Scalar ? 1 : rows);
        for (size_t row = 0; row < rows; ++row) {
            step.key[row] = 0.021f * static_cast<float>(row + 1) -
                0.009f * static_cast<float>(t + layer);
        }
        for (size_t col = 0; col < cols; ++col) {
            step.value[col] = 0.031f * static_cast<float>(col + 1) +
                0.017f * static_cast<float>(t + 2 * layer);
        }
        for (size_t row = 0; row < step.gate.size(); ++row) {
            step.gate[row] = 0.78f +
                0.011f * static_cast<float>((row + t + layer) % 7);
        }
        step.beta = 0.42f + 0.03f * static_cast<float>(t % 4);
    }
    return steps;
}

void prove_all_prefixes(DeltaTransitionGateMode gate_mode) {
    constexpr size_t rows = 8;
    constexpr size_t cols = 7;
    constexpr size_t tokens = 6;
    constexpr int layers = 2;

    for (int layer = 0; layer < layers; ++layer) {
        const std::vector<float> base = initial_state(rows, cols, layer);
        const std::vector<RawStep> steps =
            make_steps(rows, cols, tokens, layer, gate_mode);

        DeltaTransitionJournal journal;
        journal.rows = rows;
        journal.cols = cols;
        std::vector<float> verify_state = base;
        for (const RawStep & step : steps) {
            DeltaTransition transition;
            CHECK(capture_delta_transition(
                verify_state, rows, cols, step.key, step.value, step.gate,
                step.beta, gate_mode, transition));
            CHECK(apply_delta_transition(
                transition, rows, cols, verify_state));
            journal.transitions.push_back(transition);
        }

        for (size_t accepted = 1; accepted <= tokens; ++accepted) {
            std::vector<float> replayed = base;
            for (size_t t = 0; t < accepted; ++t) {
                replay_reference(
                    replayed, rows, cols, steps[t], gate_mode);
            }

            std::vector<float> committed = base;
            CHECK(commit_delta_transition_prefix(
                journal, accepted, committed));
            CHECK(near(committed, replayed));
        }

        std::vector<float> full_replay = base;
        for (const RawStep & step : steps) {
            replay_reference(full_replay, rows, cols, step, gate_mode);
        }
        CHECK(near(verify_state, full_replay));
    }
}

} // namespace

int main() {
    prove_all_prefixes(DeltaTransitionGateMode::Scalar);
    prove_all_prefixes(DeltaTransitionGateMode::RowWise);

    CHECK(delta_transition_float_count(
        128, 128, DeltaTransitionGateMode::Scalar) == 257);
    CHECK(delta_transition_float_count(
        128, 128, DeltaTransitionGateMode::RowWise) == 384);

    DeltaTransitionJournal malformed;
    malformed.rows = 2;
    malformed.cols = 2;
    malformed.transitions.push_back(DeltaTransition{});
    std::vector<float> state = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> original = state;
    CHECK(!commit_delta_transition_prefix(malformed, 1, state));
    CHECK(state == original);
    CHECK(!commit_delta_transition_prefix(malformed, 2, state));
    CHECK(state == original);
    CHECK(commit_delta_transition_prefix(malformed, 0, state));
    CHECK(state == original);

    std::printf("delta transition journal tests passed: %d checks\n", g_checks);
    return 0;
}
