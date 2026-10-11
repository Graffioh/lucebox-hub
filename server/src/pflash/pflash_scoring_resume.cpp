#include "pflash/pflash_scoring_resume.h"

#include <algorithm>

namespace luce::pflash {

static_assert(kScoringResumeGrid % kScoringUbatch == 0, "the resume grid must hold whole ubatches");

int pflash_scoring_checkpoint(int S) {
    const int cut = S - kScoringCutBack;
    if (cut < kScoringResumeGrid) return 0;
    return cut / kScoringResumeGrid * kScoringResumeGrid;
}

int pflash_scoring_exact_rows_end(int S) {
    const int cut = S - kScoringCutBack;
    if (cut < kScoringUbatch) return 0;
    return cut / kScoringUbatch * kScoringUbatch;
}

int pflash_scoring_resume(int checkpoint, int shared, int S, int cap) {
    if (checkpoint <= 0 || checkpoint % kScoringResumeGrid != 0) return 0;
    if (checkpoint > shared || checkpoint > S - kScoringCutBack || checkpoint > cap) return 0;
    return checkpoint;
}

bool PFlashExactRowSets::put(const std::vector<int> & tokens, const float * rows, int hidden, int exact_end) {
    if (tokens.empty() || hidden <= 0 || !rows || tokens.front() < 0 || tokens.back() >= exact_end) return false;
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] <= tokens[i - 1]) return false;
    }
    sets_.erase(std::remove_if(sets_.begin(), sets_.end(),
                               [&tokens](const Set & s) { return s.tokens == tokens; }),
                sets_.end());
    Set set;
    set.tokens = tokens;
    set.rows.assign(rows, rows + (size_t) hidden * tokens.size());
    sets_.push_back(std::move(set));
    while (sets_.size() > max_sets_) sets_.erase(sets_.begin());
    return true;
}

bool PFlashExactRowSets::lookup(const std::vector<int> & tokens, int hidden, std::vector<float> & rows) const {
    for (const auto & s : sets_) {
        if (s.tokens == tokens && s.rows.size() == (size_t) hidden * tokens.size()) {
            rows = s.rows;
            return true;
        }
    }
    return false;
}

int PFlashExactRowSets::invalidate_from(int position) {
    const size_t before = sets_.size();
    sets_.erase(std::remove_if(sets_.begin(), sets_.end(),
                               [position](const Set & s) { return s.tokens.back() >= position; }),
                sets_.end());
    return (int) (before - sets_.size());
}

} // namespace luce::pflash
