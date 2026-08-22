#include "server/observability.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace dflash::common::observability;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (false)

int main() {
    ObservabilityState disabled({});
    CHECK(disabled.job_queued() == 0);
    CHECK(disabled.queue_depth() == 0);
    CHECK(disabled.begin_step(4) == nullptr);
    CHECK(disabled.snapshot_json().find("\"enabled\":false") !=
          std::string::npos);

    const auto output = std::filesystem::temp_directory_path() /
        ("lucebox-observability-" + std::to_string(steady_time_ns()) +
         ".jsonl");
    ObservabilityConfig config;
    config.enabled = true;
    config.output_path = output.string();
    config.max_rounds = 1;
    config.max_requests = 1;
    config.max_token_bursts = 1;
    ObservabilityState state(config);

    const uint64_t queued_ns = state.job_queued();
    CHECK(queued_ns > 0);
    CHECK(state.queue_depth() == 1);
    state.job_dequeued();
    CHECK(state.queue_depth() == 0);

    state.set_live_slots(3);
    CHECK(state.snapshot().live_slots == 3);

    state.record_request_admitted(7, "response", 10, queued_ns,
                                  queued_ns + 10);
    StepProfile * step = state.begin_step(4);
    CHECK(step != nullptr);
    step->path = StepPath::Speculative;
    step->executed_decode_lanes = 4;
    step->spec_eligible_lanes = 4;
    step->spec_attempted_lanes = 4;
    step->spec_proposed_draft_tokens = 12;
    step->spec_accepted_draft_tokens = 8;
    step->kv_blocks_total = 100;
    step->kv_blocks_free_after = 80;
    LaneProfile lane;
    lane.request_id = 7;
    lane.slot = 0;
    lane.scheduler_consumed_tokens = 3;
    step->add_lane(lane);
    step->add_phase({Phase::TargetCompute, 1, 20});
    state.record_prefill_completed(7, queued_ns + 20);
    state.record_token_burst(7, step->round_id, queued_ns + 30, 3);
    state.record_request_finished(7, true, 3, queued_ns + 40);
    state.commit_step(step);

    StepProfile * dropped = state.begin_step(4);
    dropped->kv_blocks_total = 100;
    dropped->kv_blocks_free_after = 80;
    state.commit_step(dropped);
    const LiveMetricsSnapshot snapshot = state.snapshot();
    CHECK(snapshot.rounds == 2);
    CHECK(snapshot.kv_blocks_free == 80);
    CHECK(snapshot.durable_decode_tokens == 3);
    CHECK(snapshot.requests_completed == 1);
    CHECK(snapshot.dropped_steps == 1);
    state.flush();
    std::ifstream input(output);
    const std::string jsonl{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    CHECK(jsonl.find("lucebox.concurrency.v1\"}\n{\"type\":\"step\"") !=
          std::string::npos);
    CHECK(jsonl.find("\"type\":\"request\"") != std::string::npos);
    CHECK(jsonl.find("\"type\":\"token_burst\"") != std::string::npos);
    CHECK(jsonl.find("\"dropped_steps\":1") != std::string::npos);
    std::filesystem::remove(output);

    std::printf("test_observability: passed\n");
    return 0;
}
