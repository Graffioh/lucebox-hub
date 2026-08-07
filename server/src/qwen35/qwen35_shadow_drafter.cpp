#include "qwen35_shadow_drafter.h"

#include "attn_masks.h"
#include "graph_builders.h"
#include "internal.h"
#include "step_graph.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace dflash::common {

namespace {

bool copy_cache(const TargetCache & src,
                TargetCache & dst,
                ggml_backend_t backend) {
    if (!backend || src.attn_k.size() != dst.attn_k.size() ||
        src.attn_v.size() != dst.attn_v.size() ||
        src.ssm_state.size() != dst.ssm_state.size() ||
        src.conv_state.size() != dst.conv_state.size()) {
        return false;
    }

    auto copy_vec = [&](const std::vector<ggml_tensor *> & from,
                        const std::vector<ggml_tensor *> & to) {
        for (size_t i = 0; i < from.size(); ++i) {
            if ((!from[i]) != (!to[i])) return false;
            if (!from[i]) continue;
            if (from[i]->type != to[i]->type ||
                !ggml_are_same_shape(from[i], to[i]) ||
                !ggml_are_same_stride(from[i], to[i])) {
                return false;
            }
            ggml_backend_tensor_copy_async(
                backend, backend, from[i], to[i]);
        }
        return true;
    };

    if (!copy_vec(src.attn_k, dst.attn_k) ||
        !copy_vec(src.attn_v, dst.attn_v) ||
        !copy_vec(src.ssm_state, dst.ssm_state) ||
        !copy_vec(src.conv_state, dst.conv_state)) {
        return false;
    }
    ggml_backend_synchronize(backend);
    dst.cur_pos = src.cur_pos;
    dst.last_tok = src.last_tok;
    return true;
}

int select_depth(const std::vector<uint64_t> & histogram,
                 int min_depth,
                 int max_depth) {
    if (min_depth > max_depth) return -1;
    int best = -1;
    uint64_t best_count = 0;
    for (int depth = min_depth; depth <= max_depth; ++depth) {
        const uint64_t count = depth < (int)histogram.size()
            ? histogram[(size_t)depth] : 0;
        if (count > best_count) {
            best = depth;
            best_count = count;
        }
    }
    if (best >= 0) return best;
    return std::clamp((min_depth + max_depth) / 2, min_depth, max_depth);
}

}  // namespace

bool qwen35_shadow_accepted_tokens_match(
        const std::vector<int32_t> & proposal,
        int predicted_depth,
        const std::vector<int32_t> & actual_tokens) {
    return predicted_depth > 0 &&
        (int)proposal.size() >= predicted_depth &&
        (int)actual_tokens.size() == predicted_depth &&
        std::equal(actual_tokens.begin(), actual_tokens.end(),
                   proposal.begin());
}

bool qwen35_shadow_outcome_matches(
        int source_committed,
        int predicted_depth,
        int predicted_endpoint,
        int32_t predicted_pending,
        int actual_endpoint,
        int32_t actual_pending,
        bool used_fast_rollback,
        int actual_commit_count) {
    return used_fast_rollback && source_committed >= 0 &&
        predicted_depth > 0 && actual_commit_count == predicted_depth &&
        predicted_endpoint == source_committed + predicted_depth &&
        actual_endpoint == predicted_endpoint &&
        actual_pending == predicted_pending;
}

struct Qwen35ShadowDrafter::Impl {
    enum class JobKind { None, Prefill, Branch, Advance };

    struct Job {
        JobKind kind = JobKind::None;
        std::vector<int32_t> tokens;
        int source_committed = -1;
        int depth = -1;
        uint64_t id = 0;
        uint64_t epoch = 0;
    };

    struct Transition {
        int source_pos = -1;
        bool fast_rolled_back = false;
        int endpoint_pos = -1;
        int32_t pending_token = -1;
        int commit_count = 0;
        std::vector<int32_t> actual_tokens;
        uint64_t epoch = 0;
    };

    struct BranchResult {
        bool set = false;
        uint64_t job_id = 0;
        int endpoint_pos = -1;
        int32_t pending_token = -1;
        std::vector<int32_t> proposal;
    };

    ~Impl() { shutdown(); }

    bool init(const char * path,
              int gpu,
              int requested_max_ctx,
              int target_vocab,
              int32_t target_eos,
              int32_t target_eot) {
        model_path = path ? path : "";
        max_ctx = requested_max_ctx;
        backend = ggml_backend_cuda_init(gpu);
        if (!backend) {
            std::fprintf(stderr,
                "async-shadow-ar: failed to initialize gpu=%d\n", gpu);
            return false;
        }
        const bool low_priority =
            ggml_backend_cuda_set_low_priority_stream(backend);

        if (!load_target_gguf(model_path, backend, weights)) {
            std::fprintf(stderr, "async-shadow-ar: model load failed: %s\n",
                         dflash27b_last_error());
            return false;
        }
        if (weights.is_moe || weights.n_vocab != target_vocab ||
            (target_eos >= 0 && weights.eos_id >= 0 &&
             weights.eos_id != target_eos) ||
            (target_eot >= 0 && weights.eos_chat_id >= 0 &&
             weights.eos_chat_id != target_eot)) {
            std::fprintf(stderr,
                "async-shadow-ar: incompatible model (dense=%d vocab=%d/%d "
                "eos=%d/%d eot=%d/%d)\n",
                weights.is_moe ? 0 : 1, weights.n_vocab, target_vocab,
                weights.eos_id, target_eos,
                weights.eos_chat_id, target_eot);
            return false;
        }

        auto make_cache = [&](TargetCache & cache) {
            return create_target_cache_partial(
                weights, max_ctx, /*max_verify_tokens=*/1, backend, cache,
                /*prefill_only=*/true, /*layer_begin=*/0,
                /*layer_end=*/weights.n_layer,
                /*allocate_target_feat=*/false);
        };
        if (!make_cache(base_cache) || !make_cache(endpoint_cache) ||
            !make_cache(work_cache)) {
            std::fprintf(stderr,
                "async-shadow-ar: cache allocation failed: %s\n",
                dflash27b_last_error());
            return false;
        }

        worker = std::thread([this] { worker_loop(); });
        initialized = true;
        std::fprintf(stderr,
            "async-shadow-ar: model=%s gpu=%d layers=%d hidden=%d "
            "vocab=%d ctx=%d stream_priority=%s transport=same-process-device\n",
            model_path.c_str(), gpu, weights.n_layer, weights.n_embd,
            weights.n_vocab, max_ctx,
            low_priority ? "low" : "device-default");
        return true;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
            cv.notify_all();
        }
        if (worker.joinable()) worker.join();
        step_graph_destroy(step_graph);
        free_target_cache(work_cache);
        free_target_cache(endpoint_cache);
        free_target_cache(base_cache);
        free_target_weights(weights);
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        initialized = false;
    }

    bool eval_tokens(TargetCache & cache,
                     const std::vector<int32_t> & tokens,
                     int32_t * next_token) {
        if (tokens.empty()) {
            if (next_token) *next_token = cache.last_tok;
            return true;
        }
        if (cache.cur_pos < 0 ||
            cache.cur_pos + (int)tokens.size() > max_ctx) {
            return false;
        }

        constexpr int kPrefillChunk = 256;
        int offset = 0;
        int32_t last = -1;
        while (offset < (int)tokens.size()) {
            const int n = std::min(kPrefillChunk,
                                   (int)tokens.size() - offset);
            const int pos = cache.cur_pos;
            if (!build_target_step(
                    step_graph, weights, cache, backend,
                    pos, n, /*with_mask=*/n > 1,
                    /*capture=*/false,
                    /*capture_delta_intermediate=*/false,
                    /*fa_window=*/0,
                    /*last_token_logits_only=*/true,
                    KQ_MASK_PAD)) {
                return false;
            }

            embed.resize((size_t)weights.n_embd * (size_t)n);
            if (!weights.embedder.embed(tokens.data() + offset, n,
                                        embed.data())) {
                return false;
            }
            ggml_backend_tensor_set(step_graph.inp_embed, embed.data(), 0,
                                    sizeof(float) * embed.size());

            positions.resize((size_t)4 * (size_t)n);
            for (int i = 0; i < n; ++i) {
                positions[(size_t)4 * i + 0] = pos + i;
                positions[(size_t)4 * i + 1] = pos + i;
                positions[(size_t)4 * i + 2] = pos + i;
                positions[(size_t)4 * i + 3] = 0;
            }
            ggml_backend_tensor_set(step_graph.positions, positions.data(), 0,
                                    sizeof(int32_t) * positions.size());

            if (step_graph.attn_mask) {
                std::vector<uint16_t> mask;
                const int kv_len = pos + n;
                build_causal_mask(mask, kv_len, n, pos, KQ_MASK_PAD,
                                  /*win_start=*/0,
                                  (int)step_graph.attn_mask->ne[0]);
                ggml_backend_tensor_set(step_graph.attn_mask, mask.data(), 0,
                                        sizeof(uint16_t) * mask.size());
            }
            if (step_graph.kv_write_rows) {
                rows.assign((size_t)weights.n_head_kv * (size_t)n, 0);
                for (int h = 0; h < weights.n_head_kv; ++h) {
                    for (int i = 0; i < n; ++i) {
                        rows[(size_t)h * n + i] = pos + i;
                    }
                }
                ggml_backend_tensor_set(step_graph.kv_write_rows, rows.data(),
                                        0, sizeof(int64_t) * rows.size());
            }

            if (ggml_backend_graph_compute(backend, step_graph.gf) !=
                GGML_STATUS_SUCCESS) {
                return false;
            }
            ggml_backend_tensor_get(step_graph.argmax_tokens, &last, 0,
                                    sizeof(last));
            cache.cur_pos += n;
            cache.last_tok = last;
            offset += n;
        }
        if (next_token) *next_token = last;
        return true;
    }

    bool build_branch(const Job & current,
                      int & endpoint_pos,
                      int32_t & pending_token,
                      std::vector<int32_t> & future_proposal) {
        if (current.source_committed != base_cache.cur_pos ||
            current.depth <= 0 ||
            current.depth > (int)current.tokens.size()) {
            return false;
        }
        if (!copy_cache(base_cache, endpoint_cache, backend)) return false;

        const std::vector<int32_t> accepted(
            current.tokens.begin(), current.tokens.begin() + current.depth);
        if (!eval_tokens(endpoint_cache, accepted, &pending_token)) return false;
        endpoint_pos = current.source_committed + current.depth;

        if (!copy_cache(endpoint_cache, work_cache, backend)) return false;
        future_proposal.assign(current.tokens.size(), -1);
        future_proposal[0] = pending_token;
        int32_t input = pending_token;
        for (size_t i = 1; i < future_proposal.size(); ++i) {
            int32_t next = -1;
            if (!eval_tokens(work_cache, {input}, &next)) return false;
            future_proposal[i] = next;
            input = next;
        }
        return true;
    }

    void worker_loop() {
        for (;;) {
            Job current;
            Transition transition;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] {
                    return stop || job_pending ||
                        (base_ready && !transitions.empty());
                });
                if (stop) return;
                if (job_pending) {
                    current = job;
                    job_pending = false;
                } else {
                    transition = std::move(transitions.front());
                    transitions.pop_front();
                    cached_ready = false;
                    base_ready = false;
                    current.kind = JobKind::Advance;
                    current.tokens = transition.actual_tokens;
                    current.source_committed = transition.source_pos;
                    current.epoch = transition.epoch;
                }
            }

            if (current.kind == JobKind::Prefill) {
                reset_target_cache(base_cache);
                base_cache.cur_pos = 0;
                const bool ok = eval_tokens(base_cache, current.tokens, nullptr);
                std::lock_guard<std::mutex> lock(mu);
                if (current.epoch == request_epoch) {
                    base_ready = ok;
                    if (job_active && job.id == current.id) {
                        job_active = false;
                        job = {};
                    }
                }
                cv.notify_all();
                continue;
            }

            if (current.kind == JobKind::Advance) {
                const bool source_matches =
                    current.source_committed == base_cache.cur_pos;
                const bool ok = source_matches &&
                    eval_tokens(base_cache, current.tokens, nullptr);
                std::lock_guard<std::mutex> lock(mu);
                if (current.epoch == request_epoch) {
                    base_ready = ok;
                }
                cv.notify_all();
                continue;
            }

            const auto started = std::chrono::steady_clock::now();
            int predicted_endpoint = -1;
            int32_t predicted_pending = -1;
            std::vector<int32_t> future;
            const bool built = build_branch(
                current, predicted_endpoint, predicted_pending, future);
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();

            {
                std::unique_lock<std::mutex> lock(mu);
                if (current.epoch != request_epoch) continue;
                request_stats.compute_ms += elapsed_ms;
                if (!built) {
                    base_ready = true;
                    if (job_active && job.id == current.id) {
                        job_active = false;
                        job = {};
                    }
                    cv.notify_all();
                    continue;
                }
                branch_result.set = true;
                branch_result.job_id = current.id;
                branch_result.endpoint_pos = predicted_endpoint;
                branch_result.pending_token = predicted_pending;
                branch_result.proposal = future;
                cv.notify_all();
                cv.wait(lock, [&] {
                    return stop || current.epoch != request_epoch ||
                        direct_done_job_id == current.id ||
                        !transitions.empty();
                });
                if (stop) return;
                if (current.epoch != request_epoch ||
                    direct_done_job_id == current.id) {
                    continue;
                }
                transition = std::move(transitions.front());
                transitions.pop_front();
                branch_result = {};
            }

            const bool match = qwen35_shadow_outcome_matches(
                current.source_committed, current.depth,
                predicted_endpoint, predicted_pending,
                transition.endpoint_pos, transition.pending_token,
                transition.fast_rolled_back, transition.commit_count) &&
                qwen35_shadow_accepted_tokens_match(
                    current.tokens, current.depth, transition.actual_tokens);
            bool synced = true;
            if (match) {
                std::swap(base_cache, endpoint_cache);
            } else {
                synced = transition.source_pos == base_cache.cur_pos &&
                    eval_tokens(base_cache, transition.actual_tokens, nullptr);
            }

            {
                std::lock_guard<std::mutex> lock(mu);
                if (current.epoch != request_epoch) continue;
                if (match && transitions.empty()) {
                    cached_endpoint = predicted_endpoint;
                    cached_pending = predicted_pending;
                    cached_proposal = std::move(future);
                    cached_ready = true;
                } else {
                    request_stats.key_misses++;
                }
                base_ready = synced;
                if (job_active && job.id == current.id) {
                    job_active = false;
                    job = {};
                }
                cv.notify_all();
            }
        }
    }

    void begin_request(const std::vector<int32_t> & prompt) {
        if (!initialized) return;
        std::lock_guard<std::mutex> lock(mu);
        request_epoch++;
        request_stats = {};
        outcome_histogram.clear();
        transitions.clear();
        branch_result = {};
        cached_ready = false;
        base_ready = false;
        if ((int)prompt.size() > max_ctx) {
            job = {};
            job_active = false;
            job_pending = false;
            request_stats.launch_skips++;
            cv.notify_all();
            return;
        }
        job.kind = JobKind::Prefill;
        job.tokens = prompt;
        job.id = next_job_id++;
        job.epoch = request_epoch;
        job_active = true;
        job_pending = true;
        cv.notify_all();
    }

    bool try_take(int endpoint_pos,
                  int32_t pending_token,
                  std::vector<int32_t> & proposal) {
        std::lock_guard<std::mutex> lock(mu);
        if (cached_ready && cached_endpoint == endpoint_pos &&
            cached_pending == pending_token) {
            proposal = cached_proposal;
            cached_ready = false;
            request_stats.hits++;
            request_stats.draft_rows_saved += (int)proposal.size();
            return true;
        }
        if (cached_ready) cached_ready = false;
        if (job_active && job.kind == JobKind::Branch &&
            !not_ready_counted) {
            const auto it = std::find_if(
                transitions.begin(), transitions.end(),
                [&](const Transition & candidate) {
                    return candidate.endpoint_pos == endpoint_pos &&
                        candidate.pending_token == pending_token;
                });
            if (it != transitions.end()) {
                request_stats.not_ready++;
                not_ready_counted = true;
            }
        }
        return false;
    }

    bool launch(const std::vector<int32_t> & proposal,
                int source_committed,
                int min_depth) {
        std::lock_guard<std::mutex> lock(mu);
        // A proposal that missed its consumption deadline is stale, but its
        // endpoint cache was already adopted and remains the correct base.
        cached_ready = false;
        if (!initialized || job_active || !base_ready ||
            !transitions.empty() || proposal.size() < 2 ||
            source_committed != base_cache.cur_pos) {
            request_stats.launch_skips++;
            return false;
        }
        const int max_depth = (int)proposal.size() - 1;
        const int depth = select_depth(outcome_histogram,
                                       std::max(1, min_depth), max_depth);
        if (depth < 0) {
            request_stats.launch_skips++;
            return false;
        }

        job.kind = JobKind::Branch;
        job.tokens = proposal;
        job.source_committed = source_committed;
        job.depth = depth;
        job.id = next_job_id++;
        job.epoch = request_epoch;
        branch_result = {};
        base_ready = false;
        job_active = true;
        job_pending = true;
        not_ready_counted = false;
        request_stats.launches++;
        cv.notify_all();
        return true;
    }

    void resolve(int endpoint_pos,
                 int32_t pending_token,
                 bool fast_rolled_back,
                 int commit_count,
                 const std::vector<int32_t> & actual_tokens) {
        std::lock_guard<std::mutex> lock(mu);
        if (fast_rolled_back && commit_count >= 0) {
            if ((int)outcome_histogram.size() <= commit_count) {
                outcome_histogram.resize((size_t)commit_count + 1, 0);
            }
            outcome_histogram[(size_t)commit_count]++;
        }

        const int source_pos = endpoint_pos - (int)actual_tokens.size();
        if (job_active && job.kind == JobKind::Branch &&
            branch_result.set && branch_result.job_id == job.id &&
            transitions.empty() &&
            qwen35_shadow_outcome_matches(
                job.source_committed, job.depth,
                branch_result.endpoint_pos, branch_result.pending_token,
                endpoint_pos, pending_token,
                fast_rolled_back, commit_count) &&
            qwen35_shadow_accepted_tokens_match(
                job.tokens, job.depth, actual_tokens)) {
            // GPU work completed before target resolution. Publish the exact
            // result here so worker wake-up latency cannot turn it into a miss.
            // The worker is asleep and no device operation occurs here.
            std::swap(base_cache, endpoint_cache);
            cached_endpoint = branch_result.endpoint_pos;
            cached_pending = branch_result.pending_token;
            cached_proposal = std::move(branch_result.proposal);
            cached_ready = true;
            base_ready = true;
            direct_done_job_id = job.id;
            job_active = false;
            job = {};
            branch_result = {};
            cv.notify_all();
            return;
        }

        transitions.push_back({
            source_pos,
            fast_rolled_back,
            endpoint_pos,
            pending_token,
            commit_count,
            actual_tokens,
            request_epoch,
        });
        cv.notify_all();
    }

    void finish_request() {
        std::lock_guard<std::mutex> lock(mu);
        request_epoch++;
        job = {};
        job_active = false;
        job_pending = false;
        base_ready = false;
        cached_ready = false;
        transitions.clear();
        branch_result = {};
        cv.notify_all();
    }

    Qwen35ShadowStats stats() const {
        std::lock_guard<std::mutex> lock(mu);
        return request_stats;
    }

    mutable std::mutex mu;
    std::condition_variable cv;
    std::thread worker;
    bool stop = false;
    bool initialized = false;
    bool job_pending = false;
    bool job_active = false;
    bool base_ready = false;
    bool not_ready_counted = false;
    Job job;
    std::deque<Transition> transitions;
    BranchResult branch_result;
    uint64_t request_epoch = 0;
    uint64_t next_job_id = 1;
    uint64_t direct_done_job_id = 0;

    bool cached_ready = false;
    int cached_endpoint = -1;
    int32_t cached_pending = -1;
    std::vector<int32_t> cached_proposal;
    std::vector<uint64_t> outcome_histogram;
    Qwen35ShadowStats request_stats;

    std::string model_path;
    int max_ctx = 0;
    ggml_backend_t backend = nullptr;
    TargetWeights weights;
    TargetCache base_cache;
    TargetCache endpoint_cache;
    TargetCache work_cache;
    StepGraph step_graph;
    std::vector<float> embed;
    std::vector<int32_t> positions;
    std::vector<int64_t> rows;
};

Qwen35ShadowDrafter::Qwen35ShadowDrafter()
    : impl_(std::make_unique<Impl>()) {}

Qwen35ShadowDrafter::~Qwen35ShadowDrafter() = default;

bool Qwen35ShadowDrafter::init(const char * model_path,
                               int gpu,
                               int max_ctx,
                               int target_vocab,
                               int32_t target_eos,
                               int32_t target_eot) {
    return impl_->init(model_path, gpu, max_ctx, target_vocab,
                       target_eos, target_eot);
}

void Qwen35ShadowDrafter::begin_request(
    const std::vector<int32_t> & prompt) {
    impl_->begin_request(prompt);
}

bool Qwen35ShadowDrafter::try_take(
    int endpoint_pos,
    int32_t pending_token,
    std::vector<int32_t> & proposal) {
    return impl_->try_take(endpoint_pos, pending_token, proposal);
}

bool Qwen35ShadowDrafter::launch(
    const std::vector<int32_t> & current_proposal,
    int source_committed,
    int min_depth) {
    return impl_->launch(current_proposal, source_committed, min_depth);
}

void Qwen35ShadowDrafter::resolve(
    int endpoint_pos,
    int32_t pending_token,
    bool fast_rolled_back,
    int commit_count,
    const std::vector<int32_t> & actual_tokens) {
    impl_->resolve(endpoint_pos, pending_token, fast_rolled_back,
                   commit_count, actual_tokens);
}

void Qwen35ShadowDrafter::finish_request() {
    impl_->finish_request();
}

Qwen35ShadowStats Qwen35ShadowDrafter::stats() const {
    return impl_->stats();
}

const std::string & Qwen35ShadowDrafter::model_path() const {
    return impl_->model_path;
}

}  // namespace dflash::common
