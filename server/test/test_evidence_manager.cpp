// Unit tests for the multi-turn PFlash evidence manager — pure planning
// logic: chunk recovery, append/rebuild planning, ledger commits. No GPU,
// no model files, no HTTP.

#include "CppUnitTestFramework.hpp"
#include "server/evidence_manager.h"

#include <string>
#include <vector>

using namespace dflash::common;

namespace {
struct EvidenceManagerFixture {};

static std::vector<int32_t> range_ids(int begin, int end) {
    std::vector<int32_t> ids;
    ids.reserve((size_t) (end - begin));
    for (int value = begin; value < end; ++value) ids.push_back(value);
    return ids;
}

static PrefixHash test_revision(int seed) {
    const std::vector<int32_t> ids{seed, seed + 1, seed + 7};
    return hash_prefix(ids.data(), (int) ids.size());
}

static std::string fake_decode(int begin, int end) {
    return "[tok " + std::to_string(begin) + ".." + std::to_string(end) + "]";
}

static EvidenceConfig test_config() {
    EvidenceConfig config;
    config.chunk_size = 4;
    config.rebuild_window = 2;
    config.context_budget = 0;
    config.anchor_every_turn = false;
    return config;
}
}  // namespace

// ── recover_selected_chunks ────────────────────────────────────────────

TEST_CASE(EvidenceManagerFixture, recover_all_kept) {
    const auto input = range_ids(0, 16);          // 4 chunks of 4
    const auto kept = recover_selected_chunks(input, input, 4, 16);
    CHECK(kept == std::vector<int>({0, 1, 2, 3}));
}

TEST_CASE(EvidenceManagerFixture, recover_dropped_middle_chunks) {
    const auto input = range_ids(0, 16);
    // Selector keeps chunks 0 and 3.
    std::vector<int32_t> compressed = range_ids(0, 4);
    const auto tail = range_ids(12, 16);
    compressed.insert(compressed.end(), tail.begin(), tail.end());
    const auto kept = recover_selected_chunks(input, compressed, 4, 16);
    CHECK(kept == std::vector<int>({0, 3}));
}

TEST_CASE(EvidenceManagerFixture, recover_clips_at_source_end) {
    // Source occupies 10 tokens; the scorer input appends a query tail.
    auto input = range_ids(0, 16);
    const int source_end = 10;
    // Chunks 0,1 fully inside the source; chunk 2 straddles; chunk 3 is
    // query-only. Selector keeps chunks 0 and 2.
    std::vector<int32_t> compressed = range_ids(0, 4);
    const auto straddle = range_ids(8, 12);
    compressed.insert(compressed.end(), straddle.begin(), straddle.end());
    const auto kept = recover_selected_chunks(
        input, compressed, 4, source_end);
    CHECK(kept == std::vector<int>({0, 2}));
}

TEST_CASE(EvidenceManagerFixture, recover_ignores_query_chunks) {
    auto input = range_ids(0, 16);
    const int source_end = 8;   // chunks 2,3 are the appended query
    std::vector<int32_t> compressed = range_ids(0, 4);
    const auto query = range_ids(8, 16);
    compressed.insert(compressed.end(), query.begin(), query.end());
    const auto kept = recover_selected_chunks(
        input, compressed, 4, source_end);
    CHECK(kept == std::vector<int>({0}));
}

TEST_CASE(EvidenceManagerFixture, recover_non_grid_output_under_recovers) {
    const auto input = range_ids(0, 16);
    // Output that is not chunk-aligned cannot be attributed; fail safe.
    const std::vector<int32_t> compressed{5, 6, 7, 8, 9, 10};
    const auto kept = recover_selected_chunks(input, compressed, 4, 16);
    CHECK(kept.empty());
}

TEST_CASE(EvidenceManagerFixture, recover_empty_inputs) {
    const auto input = range_ids(0, 8);
    CHECK(recover_selected_chunks(input, {}, 4, 8).empty());
    CHECK(recover_selected_chunks({}, input, 4, 8).empty());
    CHECK(recover_selected_chunks(input, input, 0, 8).empty());
}

// ── spans_for_chunks ───────────────────────────────────────────────────

TEST_CASE(EvidenceManagerFixture, spans_merge_adjacent_chunks) {
    const auto spans = spans_for_chunks({3, 1, 2}, 4, 16);
    REQUIRE(spans.size() == 1);
    CHECK(spans[0].segment == 1);
    CHECK(spans[0].begin == 4);
    CHECK(spans[0].end == 16);
}

TEST_CASE(EvidenceManagerFixture, spans_clip_and_split) {
    const auto spans = spans_for_chunks({0, 2, 3}, 4, 10);
    REQUIRE(spans.size() == 2);
    CHECK(spans[0].begin == 0);
    CHECK(spans[0].end == 4);
    CHECK(spans[1].segment == 2);
    CHECK(spans[1].begin == 8);
    CHECK(spans[1].end == 10);   // straddling chunk clipped to source
}

// ── plan + commit: append flow ─────────────────────────────────────────

TEST_CASE(EvidenceManagerFixture, first_turn_appends_fresh_selection) {
    EvidenceLedger ledger;
    const auto plan = plan_evidence_turn(
        ledger, test_revision(1), {0, 2, 5},
        /*source_tokens=*/24, /*dialogue_tokens=*/64,
        /*question_turns=*/1, test_config());
    CHECK(plan.mode == EvidenceMode::Append);
    CHECK(plan.placement == EvidencePlacement::CurrentQuestion);
    CHECK(plan.ledger_reset);
    CHECK(plan.emit_chunks == std::vector<int>({0, 2, 5}));

    commit_evidence_turn(ledger, plan, {"q1"}, "BLOCK", test_revision(1), 24,
                         test_config().rebuild_window);
    CHECK(ledger.turn_index == 1);
    CHECK(ledger.resident.count(0) == 1);
    CHECK(ledger.resident.count(5) == 1);
    REQUIRE(ledger.committed_users.size() == 1);
    CHECK(ledger.committed_users[0] == "BLOCK\n\nq1");
    CHECK(!ledger.anchored);
}

TEST_CASE(EvidenceManagerFixture, append_turn_emits_only_missing) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();

    auto plan = plan_evidence_turn(
        ledger, revision, {0, 2}, 24, 64, 1, config);
    commit_evidence_turn(ledger, plan, {"q1"}, "B1", revision, 24,
                         config.rebuild_window);

    // Turn 2 selects chunks 2 and 5; only 5 is missing (chunk 5 covers
    // source tokens [20,24) — the clipped tail).
    plan = plan_evidence_turn(
        ledger, revision, {2, 5}, 24, 96, 2, config);
    CHECK(plan.mode == EvidenceMode::Append);
    CHECK(!plan.ledger_reset);
    CHECK(plan.emit_chunks == std::vector<int>({5}));
    CHECK(plan.resident_tokens == 8 + 4);  // chunks 0,2 resident + 5 new

    commit_evidence_turn(ledger, plan, {"q1", "q2"}, "B2", revision, 24,
                         config.rebuild_window);
    REQUIRE(ledger.committed_users.size() == 2);
    CHECK(ledger.committed_users[0] == "B1\n\nq1");
    CHECK(ledger.committed_users[1] == "B2\n\nq2");
    CHECK(ledger.resident.count(5) == 1);
}

TEST_CASE(EvidenceManagerFixture, append_turn_with_full_overlap_emits_none) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();

    auto plan = plan_evidence_turn(
        ledger, revision, {0, 2}, 24, 64, 1, config);
    commit_evidence_turn(ledger, plan, {"q1"}, "B1", revision, 24,
                         config.rebuild_window);

    plan = plan_evidence_turn(ledger, revision, {0, 2}, 24, 96, 2, config);
    CHECK(plan.mode == EvidenceMode::Append);
    CHECK(plan.emit_chunks.empty());

    commit_evidence_turn(ledger, plan, {"q1", "q2"}, "", revision, 24,
                         config.rebuild_window);
    CHECK(ledger.committed_users[1] == "q2");
}

// ── plan + commit: rebuild triggers ────────────────────────────────────

TEST_CASE(EvidenceManagerFixture, capacity_overflow_triggers_rebuild) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();
    config.context_budget = 40;
    config.rebuild_window = 1;

    auto plan = plan_evidence_turn(
        ledger, revision, {0, 1, 2}, 24, 20, 1, config);
    commit_evidence_turn(ledger, plan, {"q1"}, "B1", revision, 24,
                         config.rebuild_window);

    // Turn 2: resident 12 + missing 8 + dialogue 20 = 40 == budget: append.
    plan = plan_evidence_turn(ledger, revision, {1, 3, 4}, 24, 20, 2, config);
    CHECK(plan.mode == EvidenceMode::Append);

    commit_evidence_turn(ledger, plan, {"q1", "q2"}, "B2", revision, 24,
                         config.rebuild_window);

    // Turn 3: resident 20 + missing 4 + dialogue 20 = 44 > 40: rebuild.
    plan = plan_evidence_turn(ledger, revision, {4, 5}, 24, 20, 3, config);
    CHECK(plan.mode == EvidenceMode::Rebuild);
    CHECK(plan.placement == EvidencePlacement::Anchor);
    // retained = fresh {4,5} + last-1 selection {1,3,4} = {1,3,4,5},
    // 16 tokens + 20 dialogue = 36 <= 40: no shed.
    CHECK(plan.emit_chunks == std::vector<int>({1, 3, 4, 5}));
    CHECK(!plan.capacity_shed);

    commit_evidence_turn(
        ledger, plan, {"q1", "q2", "q3"}, "ANCHOR", revision, 24,
        config.rebuild_window);
    CHECK(ledger.anchored);
    CHECK(ledger.anchor_block == "ANCHOR");
    REQUIRE(ledger.committed_users.size() == 3);
    CHECK(ledger.committed_users[0] == "q1");   // bare under anchor layout
    CHECK(ledger.resident.count(5) == 1);
}

TEST_CASE(EvidenceManagerFixture, rebuild_sheds_history_when_still_over) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();   // rebuild_window = 2
    config.context_budget = 44;

    auto plan = plan_evidence_turn(
        ledger, revision, {0, 1, 2}, 28, 20, 1, config);
    commit_evidence_turn(ledger, plan, {"q1"}, "B1", revision, 28,
                         config.rebuild_window);

    // Turn 2: resident 12 + missing 8 + dialogue 20 = 40 <= 44: append.
    plan = plan_evidence_turn(ledger, revision, {3, 4}, 28, 20, 2, config);
    CHECK(plan.mode == EvidenceMode::Append);
    commit_evidence_turn(ledger, plan, {"q1", "q2"}, "B2", revision, 28,
                         config.rebuild_window);

    // Turn 3: resident 20 + missing 8 + 20 = 48 > 44 → rebuild. Retained =
    // fresh {5,6} + last-2 selections {3,4},{0,1,2} = {0..6}, 28+20=48 > 44
    // → shed history, carry only the fresh selection.
    plan = plan_evidence_turn(ledger, revision, {5, 6}, 28, 20, 3, config);
    CHECK(plan.mode == EvidenceMode::Rebuild);
    CHECK(plan.capacity_shed);
    CHECK(plan.emit_chunks == std::vector<int>({5, 6}));
}

TEST_CASE(EvidenceManagerFixture, source_revision_change_forces_reset) {
    EvidenceLedger ledger;
    auto config = test_config();

    auto plan = plan_evidence_turn(
        ledger, test_revision(1), {0, 1}, 24, 20, 1, config);
    commit_evidence_turn(ledger, plan, {"q1"}, "B1", test_revision(1), 24,
                         config.rebuild_window);

    // Same shape but a different source revision: reset + append fresh.
    plan = plan_evidence_turn(
        ledger, test_revision(2), {2, 3}, 24, 40, 2, config);
    CHECK(plan.mode == EvidenceMode::Append);
    CHECK(plan.ledger_reset);
    CHECK(plan.emit_chunks == std::vector<int>({2, 3}));

    commit_evidence_turn(ledger, plan, {"q1", "q2"}, "B2",
                         test_revision(2), 24, config.rebuild_window);
    CHECK(ledger.turn_index == 2);
    // Old resident set was cleared before the new emit chunks were added.
    CHECK(ledger.resident.count(0) == 0);
    CHECK(ledger.resident.count(2) == 1);
    CHECK(ledger.committed_users[0] == "q1");
    CHECK(ledger.committed_users[1] == "B2\n\nq2");
}

TEST_CASE(EvidenceManagerFixture, question_desync_forces_reset) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();

    auto plan = plan_evidence_turn(
        ledger, revision, {0}, 24, 20, 1, config);
    commit_evidence_turn(ledger, plan, {"q1"}, "B1", revision, 24,
                         config.rebuild_window);

    // Client sends turn 4 while the ledger served only turn 1 → desync.
    plan = plan_evidence_turn(
        ledger, revision, {1, 2}, 24, 60, 4, config);
    CHECK(plan.ledger_reset);
    CHECK(plan.mode == EvidenceMode::Append);
    CHECK(plan.emit_chunks == std::vector<int>({1, 2}));

    commit_evidence_turn(ledger, plan, {"q1", "q2", "q3", "q4"}, "B4",
                         revision, 24, config.rebuild_window);
    REQUIRE(ledger.committed_users.size() == 4);
    CHECK(ledger.committed_users[0] == "q1");
    CHECK(ledger.committed_users[2] == "q3");
    CHECK(ledger.committed_users[3] == "B4\n\nq4");
    CHECK(ledger.turn_index == 4);
}

TEST_CASE(EvidenceManagerFixture, anchor_every_turn_rebuilds) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();
    config.anchor_every_turn = true;

    auto plan = plan_evidence_turn(
        ledger, revision, {0, 2}, 24, 20, 1, config);
    CHECK(plan.mode == EvidenceMode::Rebuild);
    CHECK(plan.placement == EvidencePlacement::Anchor);
    commit_evidence_turn(ledger, plan, {"q1"}, "A1", revision, 24,
                         config.rebuild_window);

    // Second turn rebuilds again even under budget, carrying fresh + recent.
    plan = plan_evidence_turn(ledger, revision, {3}, 24, 20, 2, config);
    CHECK(plan.mode == EvidenceMode::Rebuild);
    for (int chunk : {0, 2, 3}) {
        CHECK(std::find(plan.emit_chunks.begin(), plan.emit_chunks.end(),
                        chunk) != plan.emit_chunks.end());
    }
}

TEST_CASE(EvidenceManagerFixture, recent_window_bounds_carried_history) {
    EvidenceLedger ledger;
    const auto revision = test_revision(1);
    auto config = test_config();  // rebuild_window = 2
    config.context_budget = 0;

    // Four append turns, each selecting a disjoint chunk.
    for (int turn = 1; turn <= 4; ++turn) {
        std::vector<std::string> questions;
        for (int i = 1; i <= turn; ++i) {
            questions.push_back("q" + std::to_string(i));
        }
        auto plan = plan_evidence_turn(
            ledger, revision, {turn + 10}, 64, 20, turn, config);
        commit_evidence_turn(ledger, plan, questions,
                             "B" + std::to_string(turn), revision, 64,
                             config.rebuild_window);
    }
    // recent holds at most rebuild_window + 1 selections.
    CHECK((int) ledger.recent.size() <= 3);
}

// ── render helpers ─────────────────────────────────────────────────────

TEST_CASE(EvidenceManagerFixture, render_block_has_span_headers) {
    const std::vector<EvidenceSpan> spans{
        {2, 8, 12}, {7, 28, 32},
    };
    const std::string block = render_evidence_block(spans, fake_decode);
    CHECK(block.find("segment 2") != std::string::npos);
    CHECK(block.find("8-12") != std::string::npos);
    CHECK(block.find("[tok 8..12]") != std::string::npos);
    CHECK(block.find("segment 7") != std::string::npos);
    CHECK(block.find("[tok 28..32]") != std::string::npos);
}

TEST_CASE(EvidenceManagerFixture, compose_message_orders_block_then_query) {
    CHECK(compose_evidence_message("BLOCK", "q") == "BLOCK\n\nq");
    CHECK(compose_evidence_message("", "q") == "q");
    CHECK(compose_evidence_message("BLOCK", "") == "BLOCK");
}
