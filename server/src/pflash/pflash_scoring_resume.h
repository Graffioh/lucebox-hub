#pragma once

// Exact drafter-session resume for the strict scorer (qwen35_drafter.cpp).
//
// A fresh scoring call over S tokens runs blocks 0..14 in 1024-token ubatches from token 0, cut once at S - 64, and
// projects keys / probe / request logits in 8192-token chunks from token 0. A resumed call reproduces those values
// bit for bit only when it uses the same partition for every token it recomputes and reads stored values that a
// fresh call would also have computed in that partition:
//   * it resumes at a multiple of kScoringResumeGrid (8192, a multiple of the 1024 ubatch), so its ubatches and key
//     chunks fall on the fresh call's grid, and at or before S - 64, so the cut stays where the fresh call has it;
//   * the session keeps one recurrent-state snapshot, at pflash_scoring_checkpoint(S) of the call that stored it;
//   * stored block-14 rows are exact only below pflash_scoring_exact_rows_end(S) of the call that computed them
//     (they sit in full grid ubatches there; later rows met the S - 64 cut).
// Nothing here approximates: a value that is not exactly available sends the call back to token 0.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::pflash {

constexpr int kScoringResumeGrid = 8192;
constexpr int kScoringUbatch = 1024;
constexpr int kScoringCutBack = 64;

// The snapshot position a call over ``S`` tokens keeps: the last multiple of the grid at or before S - 64 (0 when
// there is none: nothing to resume from).
int pflash_scoring_checkpoint(int S);

// Rows this call computes at positions below this are exact for any later prompt that shares them.
int pflash_scoring_exact_rows_end(int S);

// Where a call over ``S`` tokens resumes from a session whose snapshot is ``checkpoint`` and that shares
// ``shared`` leading tokens with it: the checkpoint when it is on the grid, > 0, <= shared, <= S - 64 and
// <= ``cap``; else 0 (a fresh call).
int pflash_scoring_resume(int checkpoint, int shared, int S, int cap);

// Exact F32 block-14 rows of query token sets (the query step's candidate and END sets), kept per session for later
// turns whose activations are gone. Only sets whose last token is below the storing call's exact-rows end are put.
class PFlashExactRowSets {
public:
    explicit PFlashExactRowSets(size_t max_sets = 8) : max_sets_(max_sets) {}

    // Store ``rows`` ([hidden, tokens.size()]) for the ascending ``tokens``; replaces an equal set, evicts the
    // least recently stored past max_sets. False (nothing stored) when tokens are empty, not ascending, reach
    // ``exact_end``, or rows have the wrong size.
    bool put(const std::vector<int> & tokens, const float * rows, int hidden, int exact_end);
    // The rows of exactly this token set, when stored.
    bool lookup(const std::vector<int> & tokens, int hidden, std::vector<float> & rows) const;
    // Drop every set with a token at or after ``position`` (the prompt diverged there). Returns how many.
    int invalidate_from(int position);
    void clear() { sets_.clear(); }
    size_t size() const { return sets_.size(); }

private:
    struct Set {
        std::vector<int> tokens;
        std::vector<float> rows;
    };
    size_t max_sets_;
    std::vector<Set> sets_;   // oldest first
};

} // namespace luce::pflash
