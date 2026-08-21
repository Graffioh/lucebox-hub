#include "common/dspark_head.h"
#include "host_check.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    constexpr int hidden = 3;
    constexpr int vocab = 5;
    constexpr int rank = 2;
    constexpr int q_len = 4;
    constexpr int lanes = 2;
    constexpr int confidence_dim = hidden + rank;

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) return 1;

    ggml_init_params weights_params{};
    weights_params.mem_size = ggml_tensor_overhead() * 8;
    weights_params.no_alloc = true;
    ggml_context * weights_ctx = ggml_init(weights_params);
    CHECK(weights_ctx != nullptr);
    if (!weights_ctx) {
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * lm_head = ggml_new_tensor_2d(
        weights_ctx, GGML_TYPE_F32, hidden, vocab);
    ggml_tensor * markov_w1 = ggml_new_tensor_2d(
        weights_ctx, GGML_TYPE_F32, rank, vocab);
    ggml_tensor * markov_w2 = ggml_new_tensor_2d(
        weights_ctx, GGML_TYPE_F32, rank, vocab);
    ggml_tensor * confidence_w = ggml_new_tensor_2d(
        weights_ctx, GGML_TYPE_F32, confidence_dim, 1);
    ggml_tensor * confidence_b = ggml_new_tensor_1d(
        weights_ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t weights_buf =
        ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    CHECK(weights_buf != nullptr);
    if (!weights_buf) {
        ggml_free(weights_ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> lm((size_t)hidden * vocab);
    std::vector<float> w1((size_t)rank * vocab);
    std::vector<float> w2((size_t)rank * vocab);
    std::vector<float> cw((size_t)confidence_dim);
    for (int token = 0; token < vocab; ++token) {
        for (int h = 0; h < hidden; ++h) {
            lm[(size_t)token * hidden + h] =
                0.031f * (float)(token + 1) * (float)(h + 1);
        }
        for (int r = 0; r < rank; ++r) {
            w1[(size_t)token * rank + r] =
                0.017f * (float)(token + 1 + r);
            w2[(size_t)token * rank + r] =
                0.013f * (float)(token + 1) * (float)(r + 1);
        }
    }
    for (int i = 0; i < confidence_dim; ++i) {
        cw[(size_t)i] = 0.021f * (float)(i + 1);
    }
    const float cb = -0.11f;
    ggml_backend_tensor_set(lm_head, lm.data(), 0, sizeof(float) * lm.size());
    ggml_backend_tensor_set(
        markov_w1, w1.data(), 0, sizeof(float) * w1.size());
    ggml_backend_tensor_set(
        markov_w2, w2.data(), 0, sizeof(float) * w2.size());
    ggml_backend_tensor_set(
        confidence_w, cw.data(), 0, sizeof(float) * cw.size());
    ggml_backend_tensor_set(confidence_b, &cb, 0, sizeof(cb));

    DraftWeights dw;
    dw.n_embd = hidden;
    dw.block_size = q_len;
    dw.dspark.enabled = true;
    dw.dspark.markov_rank = rank;
    dw.dspark.vocab_size = vocab;
    dw.dspark.confidence_dim = confidence_dim;
    dw.dspark.markov_w1 = markov_w1;
    dw.dspark.markov_w2 = markov_w2;
    dw.dspark.confidence_w = confidence_w;
    dw.dspark.confidence_b = confidence_b;

    std::vector<std::vector<float>> hidden_host(
        lanes, std::vector<float>((size_t)hidden * q_len));
    std::vector<std::vector<float>> prenorm_host(
        lanes, std::vector<float>((size_t)hidden * q_len));
    const int32_t seeds[lanes] = {1, 3};
    for (int lane = 0; lane < lanes; ++lane) {
        for (int position = 0; position < q_len; ++position) {
            for (int h = 0; h < hidden; ++h) {
                const size_t index =
                    (size_t)position * hidden + h;
                hidden_host[(size_t)lane][index] =
                    0.07f * (float)(1 + lane + 2 * position + h);
                prenorm_host[(size_t)lane][index] =
                    hidden_host[(size_t)lane][index] +
                    0.019f * (float)(h + 1);
            }
        }
    }

    std::vector<std::vector<int32_t>> serial_tokens(lanes);
    std::vector<std::vector<float>> serial_confidence(lanes);
    for (int lane = 0; lane < lanes; ++lane) {
        CHECK(dspark_markov_correct_greedy_chain_fused(
            dw, backend, lm_head, hidden_host[(size_t)lane].data(),
            q_len, seeds[lane], serial_tokens[(size_t)lane],
            &serial_confidence[(size_t)lane],
            prenorm_host[(size_t)lane].data()));
    }

    std::vector<uint8_t> arena(4u * 1024u * 1024u);
    ggml_init_params graph_params{};
    graph_params.mem_size = arena.size();
    graph_params.mem_buffer = arena.data();
    graph_params.no_alloc = true;
    ggml_context * graph_ctx = ggml_init(graph_params);
    CHECK(graph_ctx != nullptr);
    ggml_cgraph * graph =
        ggml_new_graph_custom(graph_ctx, 2048, false);

    std::vector<ggml_tensor *> hidden_inputs(lanes);
    std::vector<ggml_tensor *> prenorm_inputs(lanes);
    for (int lane = 0; lane < lanes; ++lane) {
        hidden_inputs[(size_t)lane] = ggml_new_tensor_2d(
            graph_ctx, GGML_TYPE_F32, hidden, q_len);
        prenorm_inputs[(size_t)lane] = ggml_new_tensor_2d(
            graph_ctx, GGML_TYPE_F32, hidden, q_len);
        ggml_set_input(hidden_inputs[(size_t)lane]);
        ggml_set_input(prenorm_inputs[(size_t)lane]);
    }
    ggml_tensor * seed_input =
        ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, lanes);
    ggml_set_input(seed_input);

    DSparkBatchedChainOutputs outputs;
    CHECK(build_dspark_markov_batched_chain(
        graph_ctx, graph, dw, lm_head,
        hidden_inputs, prenorm_inputs, seed_input,
        q_len, true, outputs));
    CHECK(outputs.n_lanes == lanes);
    CHECK(outputs.q_len == q_len);
    CHECK(outputs.tokens.size() == (size_t)(q_len - 1));
    CHECK(outputs.confidence.size() == (size_t)(q_len - 1));

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    CHECK(allocator != nullptr);
    const bool graph_allocated =
        allocator && ggml_gallocr_alloc_graph(allocator, graph);
    CHECK(graph_allocated);
    if (graph_allocated) {
        for (int lane = 0; lane < lanes; ++lane) {
            ggml_backend_tensor_set(
                hidden_inputs[(size_t)lane],
                hidden_host[(size_t)lane].data(), 0,
                sizeof(float) * hidden_host[(size_t)lane].size());
            ggml_backend_tensor_set(
                prenorm_inputs[(size_t)lane],
                prenorm_host[(size_t)lane].data(), 0,
                sizeof(float) * prenorm_host[(size_t)lane].size());
        }
        ggml_backend_tensor_set(
            seed_input, seeds, 0, sizeof(seeds));
        CHECK(ggml_backend_graph_compute(backend, graph) ==
              GGML_STATUS_SUCCESS);

        std::vector<int32_t> depth_tokens(
            (size_t)(q_len - 1) * lanes);
        std::vector<float> depth_confidence(
            (size_t)(q_len - 1) * lanes);
        for (int depth = 0; depth < q_len - 1; ++depth) {
            ggml_backend_tensor_get_async(
                backend, outputs.tokens[(size_t)depth],
                depth_tokens.data() + (size_t)depth * lanes,
                0, sizeof(int32_t) * lanes);
            ggml_backend_tensor_get_async(
                backend, outputs.confidence[(size_t)depth],
                depth_confidence.data() + (size_t)depth * lanes,
                0, sizeof(float) * lanes);
        }
        ggml_backend_synchronize(backend);

        for (int lane = 0; lane < lanes; ++lane) {
            CHECK(serial_tokens[(size_t)lane].size() == (size_t)q_len);
            CHECK(serial_confidence[(size_t)lane].size() ==
                  (size_t)(q_len - 1));
            CHECK(serial_tokens[(size_t)lane][0] == seeds[lane]);
            for (int depth = 0; depth < q_len - 1; ++depth) {
                CHECK(serial_tokens[(size_t)lane][(size_t)depth + 1] ==
                      depth_tokens[(size_t)depth * lanes + lane]);
                CHECK(std::fabs(
                    serial_confidence[(size_t)lane][(size_t)depth] -
                    depth_confidence[
                        (size_t)depth * lanes + lane]) < 1e-6f);
            }
        }
    }
    if (allocator) ggml_gallocr_free(allocator);

    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weights_buf);
    ggml_free(weights_ctx);
    ggml_backend_free(backend);
    std::printf(
        "DSpark batched-head parity tests passed: %d checks\n",
        g_checks);
    return 0;
}
