// Same-process autoregressive outcome cache for asynchronous speculative
// drafting. A small vocabulary-compatible Qwen3.5 model runs on the draft GPU
// while the primary GPU performs ordinary target verification.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35ShadowStats {
    int launches = 0;
    int hits = 0;
    int key_misses = 0;
    int not_ready = 0;
    int launch_skips = 0;
    int draft_rows_saved = 0;
    double compute_ms = 0.0;
};

// A shadow result is reusable only for the exact authoritative transition it
// modeled. Keep this predicate independent of GPU work so the fail-closed key
// contract can be unit tested.
bool qwen35_shadow_outcome_matches(
    int source_committed,
    int predicted_depth,
    int predicted_endpoint,
    int32_t predicted_pending,
    int actual_endpoint,
    int32_t actual_pending,
    bool used_fast_rollback,
    int actual_commit_count);

bool qwen35_shadow_accepted_tokens_match(
    const std::vector<int32_t> & proposal,
    int predicted_depth,
    const std::vector<int32_t> & actual_tokens);

// The worker owns an independent low-priority GPU stream and three device-
// local AR caches: authoritative prefix, hypothetical endpoint, and proposal
// generation. No model activations or KV state cross the host or an IPC
// boundary; only token IDs are passed to the worker.
class Qwen35ShadowDrafter {
public:
    Qwen35ShadowDrafter();
    ~Qwen35ShadowDrafter();

    Qwen35ShadowDrafter(const Qwen35ShadowDrafter &) = delete;
    Qwen35ShadowDrafter & operator=(const Qwen35ShadowDrafter &) = delete;

    bool init(const char * model_path,
              int gpu,
              int max_ctx,
              int target_vocab,
              int32_t target_eos,
              int32_t target_eot);

    // Prefill is asynchronous and normally overlaps target prefill. A request
    // that reaches decode first simply skips shadows until the base is ready.
    void begin_request(const std::vector<int32_t> & prompt);

    // Consume an exact outcome-cache hit without waiting. The proposal starts
    // with pending_token and is safe to pass to ordinary target verification.
    bool try_take(int endpoint_pos,
                  int32_t pending_token,
                  std::vector<int32_t> & proposal);

    // Launch one likely accepted-length outcome. Returns immediately; false
    // means the worker/base was not ready and ordinary drafting should proceed.
    bool launch(const std::vector<int32_t> & current_proposal,
                int source_committed,
                int min_depth);

    // Publish the target-authoritative transition. A matching branch becomes
    // reusable; a miss asynchronously advances the base AR state with
    // actual_tokens. This never waits for shadow compute.
    void resolve(int endpoint_pos,
                 int32_t pending_token,
                 bool fast_rolled_back,
                 int commit_count,
                 const std::vector<int32_t> & actual_tokens);

    // Do not delay response completion for unused work. A later request
    // invalidates stale work by epoch and asynchronously resets its caches.
    void finish_request();

    Qwen35ShadowStats stats() const;
    const std::string & model_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dflash::common
