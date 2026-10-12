// Exact drafter-session resume (pflash_scoring_resume.h): the grid checkpoint, the exact-rows bound, the resume
// rule and the F32 query-set store. Model-free.

#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_scoring_resume.h"

#include <algorithm>
#include <vector>

using namespace luce::pflash;

namespace {

struct PFlashScoringResumeFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// Ubatch boundaries of a call over S tokens resumed at r (the drafter's ubatch_at): 1024 steps from r, cut at S-64.
std::vector<int> ubatch_starts(int r, int S) {
    std::vector<int> out;
    const int cut = std::max(r, S - kScoringCutBack);
    for (int start = r; start < S;) {
        out.push_back(start);
        start += std::min(kScoringUbatch, (start < cut ? cut : S) - start);
    }
    return out;
}

} // namespace

TEST_CASE(PFlashScoringResumeFixture, checkpoint_is_the_last_grid_point_before_the_cut) {
    CHECK(pflash_scoring_checkpoint(0) == 0);
    CHECK(pflash_scoring_checkpoint(8192) == 0);                 // cut 8128 < 8192
    CHECK(pflash_scoring_checkpoint(8192 + 64) == 8192);
    CHECK(pflash_scoring_checkpoint(8192 + 63) == 0);
    CHECK(pflash_scoring_checkpoint(63565) == 57344);
    CHECK(pflash_scoring_checkpoint(65600) == 65536);
    CHECK(pflash_scoring_checkpoint(65599) == 57344);
}

TEST_CASE(PFlashScoringResumeFixture, exact_rows_end_is_the_last_full_ubatch_before_the_cut) {
    CHECK(pflash_scoring_exact_rows_end(1000) == 0);
    CHECK(pflash_scoring_exact_rows_end(1024 + 64) == 1024);
    CHECK(pflash_scoring_exact_rows_end(63565) == 63488);
    for (int S : {9000, 63565, 70000, 131072}) {
        CHECK(pflash_scoring_exact_rows_end(S) >= pflash_scoring_checkpoint(S));
        CHECK(pflash_scoring_exact_rows_end(S) <= S - kScoringCutBack);
    }
}

TEST_CASE(PFlashScoringResumeFixture, resume_only_at_a_grid_checkpoint_inside_the_shared_prefix) {
    // A follow-up: the previous prompt's checkpoint, all of it shared.
    CHECK(pflash_scoring_resume(57344, 63560, 63790, 1 << 30) == 57344);
    // Diverged before the checkpoint (an edit): fresh.
    CHECK(pflash_scoring_resume(57344, 57343, 63790, 1 << 30) == 0);
    // The checkpoint itself shared is enough.
    CHECK(pflash_scoring_resume(57344, 57344, 63790, 1 << 30) == 57344);
    // A shorter prompt whose S - 64 cut falls before the checkpoint: fresh.
    CHECK(pflash_scoring_resume(57344, 57380, 57380, 1 << 30) == 0);
    CHECK(pflash_scoring_resume(57344, 57408, 57408, 1 << 30) == 57344);
    // The cap (a redo) and an off-grid or empty checkpoint.
    CHECK(pflash_scoring_resume(57344, 63560, 63790, 0) == 0);
    CHECK(pflash_scoring_resume(57344, 63560, 63790, 57343) == 0);
    CHECK(pflash_scoring_resume(57345, 63560, 63790, 1 << 30) == 0);
    CHECK(pflash_scoring_resume(0, 63560, 63790, 1 << 30) == 0);
    // A retry (the same prompt): resumes at its own checkpoint.
    CHECK(pflash_scoring_resume(57344, 63565, 63565, 1 << 30) == 57344);
}

TEST_CASE(PFlashScoringResumeFixture, a_resumed_call_lands_on_the_fresh_partition) {
    // For every resume the rule allows, the ubatches and 8192 key chunks from the resume point are exactly the
    // fresh call's from that point on.
    for (int S : {16500, 63565, 63790, 65600, 74200, 98304 + 64}) {
        const std::vector<int> fresh = ubatch_starts(0, S);
        for (int prev : {16400, 57400, 63565, 65599, 65600, 74100}) {
            const int r = pflash_scoring_resume(pflash_scoring_checkpoint(prev), std::min(prev, S), S, 1 << 30);
            if (r == 0) continue;
            CHECK(r % kScoringResumeGrid == 0);
            std::vector<int> tail;
            for (const int b : fresh) {
                if (b >= r) tail.push_back(b);
            }
            CHECK(ubatch_starts(r, S) == tail);
            // The fresh call has a ubatch boundary at r (so the state there is the one the snapshot holds).
            bool boundary = false;
            for (const int b : fresh) boundary = boundary || b == r;
            CHECK(boundary);
        }
    }
}

TEST_CASE(PFlashScoringResumeFixture, query_sets_store_exact_rows_only) {
    PFlashExactRowSets sets(2);
    const int hidden = 2;
    const float rows3[] = {1, 2, 3, 4, 5, 6};
    std::vector<float> out;
    // Below the exact end: stored and found by the exact token list.
    CHECK(sets.put({10, 11, 12}, rows3, hidden, 13));
    CHECK(sets.lookup({10, 11, 12}, hidden, out));
    CHECK(out == std::vector<float>(rows3, rows3 + 6));
    CHECK(!sets.lookup({10, 11}, hidden, out));
    // Reaching the exact end, unsorted or empty: refused.
    CHECK(!sets.put({20, 21, 22}, rows3, hidden, 22));
    CHECK(!sets.put({12, 11, 10}, rows3, hidden, 100));
    CHECK(!sets.put({}, rows3, hidden, 100));
    CHECK(!sets.lookup({20, 21, 22}, hidden, out));
    // Bounded: the oldest goes first; an equal set is replaced, not duplicated.
    const float rows1a[] = {7, 8};
    const float rows1b[] = {9, 9};
    CHECK(sets.put({30}, rows1a, hidden, 100));
    CHECK(sets.put({30}, rows1b, hidden, 100));
    CHECK(sets.size() == 2);
    CHECK(sets.lookup({30}, hidden, out));
    CHECK(out == std::vector<float>({9, 9}));
    CHECK(sets.put({40}, rows1a, hidden, 100));
    CHECK(sets.size() == 2);
    CHECK(!sets.lookup({10, 11, 12}, hidden, out));
    // A divergence drops every set that reaches it.
    CHECK(sets.invalidate_from(40) == 1);
    CHECK(sets.lookup({30}, hidden, out));
    CHECK(sets.invalidate_from(30) == 1);
    CHECK(sets.size() == 0);
}

// Session capacity: headroom S + S/2 + 4096, clamped to the context limit, never below S padded to 256.
TEST_CASE(PFlashScoringResumeFixture, session_capacity_is_headroom_clamped_to_the_context_limit) {
    using luce::pflash::pflash_session_capacity;
    CHECK(pflash_session_capacity(100000, 0) == 100000 + 50000 + 4096);
    CHECK(pflash_session_capacity(100000, 262144) == 100000 + 50000 + 4096);
    CHECK(pflash_session_capacity(200000, 262144) == 262144);
    CHECK(pflash_session_capacity(217177, 225280) == 225280);
    CHECK(pflash_session_capacity(250000, 225280) == 250112);   // the prompt itself always fits (padded to 256)
    CHECK(pflash_session_capacity(200000, 225000) == 225024);   // limit rounded up to 256
    for (int S : {1, 255, 256, 8191, 65536, 131071, 225279, 262144}) {
        for (int limit : {0, 1, 4096, 131072, 225280, 262144}) {
            const int c = pflash_session_capacity(S, limit);
            CHECK(c >= (S + 255) / 256 * 256);
            CHECK(c <= std::max((S + 255) / 256 * 256, limit > 0 ? (limit + 255) / 256 * 256 : S + S / 2 + 4096));
        }
    }
}

TEST_CASE(PFlashScoringResumeFixture, session_ctx_limit_is_the_smaller_known_context) {
    using luce::pflash::pflash_session_ctx_limit;
    CHECK(pflash_session_ctx_limit(262144, 225280) == 225280);
    CHECK(pflash_session_ctx_limit(262144, 0) == 262144);
    CHECK(pflash_session_ctx_limit(0, 139264) == 139264);
    CHECK(pflash_session_ctx_limit(0, 0) == 0);
    CHECK(pflash_session_ctx_limit(-1, 8192) == 8192);
}
