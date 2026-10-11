// Stage 2 of PFlash query identification (look-back combiner), STAGE2-CONTRACT.md version 3
// (round10/cacheaware; trainer mirror luce_box request_probe/stage2_contract.py).
//
// Pass 1 of a scoring call, when a stage-2 GGUF (PFLASH_REQUEST_STAGE2_GGUF, arch "requeststage2") is loaded next
// to a cand_v5 request probe and the prompt has a user message L:
//   R         = min(256, |post-latest stage-1 run tokens|) query slots reserved for the post-latest cues
//   seed      = L's stage-1 run tokens (<= 1024 - R by score)
//   m_all     = mean of x̂ over L's content tokens, m_s1 = mean over the seed (= m_all when the seed is empty)
//   u_all/u_s1, message head -> raw logits l_lb (look-back) and l_vague (budget widening)
//   a_h(t)    = mean over the seed of softmax_{k in E}(<Q_h(q), K_{h/4}(k)> / 16), E = content tokens of the
//               non-system messages before L (u314 head Q/K, NoPE)
//   x(t)      = 143 features (A_h = log1p(N a_h), messages back, role, position, length, s1, c(t), c*u_all,
//               u_all, u_s1), s2(t) = fc2(GELU_erf(fc1(LayerNorm(x))))
//   query     = seed > promoted (s2 runs, <= min(512, 1024 - R - |seed|)) > post-latest s1 runs (>= R), <= 1024,
//               all pinned
//               (pflash_stage2_assemble in pflash_request_probe.h); l_vague >= tau_fb widens the budget x1.5.
// x̂ = RMSNorm(row) * blk.15.attn_norm (as stored), the head's K-graph input; rows are F32, never BF16.
//
// Two implementations of the numbers: a plain C++ CPU path in double (pflash_stage2_cpu_reference; the parity
// reference, small prompts) and the device path the drafter serves with (ggml graphs: c(t) and the m_all sums in
// the key graph, the look-back attention in seed tiles with an exact softmax over E, the token MLP). Both are
// checked against the trainer's float64 fixtures (test_pflash_request_stage2.cpp).

#pragma once

#include "pflash_request_probe.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace luce::pflash {

// ---------------------------------------------------------------- contract strings and shapes (section 3)
inline constexpr const char * kStage2Arch = "requeststage2";
inline constexpr const char * kStage2Schema = "qwen3_5_0_8b_request_stage2_v3";
inline constexpr const char * kStage2FeatureVersion =
    "s2_features_v4_lognd8_back_role3_relpos_loglen_s1c16_emb32x4_seedcap1024minusR";
inline constexpr const char * kStage2AssemblyVersion = "s2_query_v4_reserve256_seed_lookback512_postlatest_pinned";
inline constexpr const char * kStage2FallbackVersion = "s2_fallback_v2_budget_x1.5_rawfits";
inline constexpr const char * kStage2BaseModel = "Qwen/Qwen3.5-0.8B";
inline constexpr const char * kStage2FeatureTap = "post_block14_residual_before_block15";
inline constexpr const char * kStage2TauSourceCalibration = "calibration-pool";
inline constexpr const char * kStage2TauSourceDev = "dev-provisional";
// Environment: the stage-2 GGUF, and the parity dump of x(t), s2(t), a_h(t) per pass 1 (a file path; "1" is
// rejected so a dump never lands in the working directory by accident).
inline constexpr const char * kStage2GgufEnv = "PFLASH_REQUEST_STAGE2_GGUF";
inline constexpr const char * kStage2FeaturesEnv = "PFLASH_TRACE_STAGE2_FEATURES";

inline constexpr int kStage2Hidden = 1024;      // block-14 row width
inline constexpr int kStage2Heads = 8;          // u314 query heads
inline constexpr int kStage2KvHeads = 2;
inline constexpr int kStage2HeadDim = 256;
inline constexpr int kStage2Embed = 32;
inline constexpr int kStage2MlpHidden = 64;
inline constexpr int kStage2MsgHidden = 32;
inline constexpr int kStage2Fixed = 15;
inline constexpr int kStage2InputDim = kStage2Fixed + 4 * kStage2Embed;   // 143
inline constexpr float kStage2S1Clamp = 16.0f;
inline constexpr float kStage2BudgetFactor = 1.5f;
// Feature columns (section 4.4).
inline constexpr int kStage2ColA = 0;           // 8 heads
inline constexpr int kStage2ColBack = 8;        // log1p(d)
inline constexpr int kStage2ColRole = 9;        // user, assistant, tool
inline constexpr int kStage2ColRelpos = 12;
inline constexpr int kStage2ColLogLen = 13;
inline constexpr int kStage2ColS1 = 14;
inline constexpr int kStage2ColC = 15;
inline constexpr int kStage2ColCxAll = 47;
inline constexpr int kStage2ColUAll = 79;
inline constexpr int kStage2ColUS1 = 111;

// ---------------------------------------------------------------- the GGUF
// Host copies, numpy (= torch) row-major layout: a Linear(in, out) weight is [out][in].
struct PFlashStage2Weights {
    std::vector<float> cand_w, cand_b;           // [32][1024], [32]
    std::vector<float> lat_all_w, lat_all_b;     // [32][1024], [32]
    std::vector<float> lat_s1_w, lat_s1_b;       // [32][1024], [32]
    std::vector<float> norm_w, norm_b;           // [143]
    std::vector<float> fc1_w, fc1_b;             // [64][143], [64]
    std::vector<float> fc2_w, fc2_b;             // [64], [1]
    std::vector<float> msg_fc1_w, msg_fc1_b;     // [32][2048], [32]
    std::vector<float> msg_fc2_w, msg_fc2_b;     // [2][32] (row 0 look-back, row 1 vague), [2]
    float norm_eps = 0.0f;
    float rms_eps = 0.0f;
    float tau2 = 0.0f;
    float tau_lb = 0.0f;
    float tau_fb = 0.0f;
    std::string tau_source;
};

// What the loaded GGUF must pair with (section 3: every pin compared exactly).
struct PFlashStage2Pins {
    std::string drafter_sha256;              // sha256 of the loaded drafter GGUF
    std::string head_sha256;                 // sha256 of the loaded scoring-head GGUF
    std::string stage1_checkpoint_sha256;    // requestprobe.checkpoint_sha256 of the loaded stage-1 GGUF
    std::string stage1_candidate_rule;       // requestprobe.candidate_rule of the loaded stage-1 GGUF
    float drafter_rms_eps = 0.0f;            // qwen35.attention.layer_norm_rms_epsilon
};

// Loads ``path`` and checks every required key, version string, constant, pin and tensor (names, F32, shapes, no
// extra tensors). Fails closed: false + ``error``, ``out`` untouched.
bool pflash_stage2_load(const std::string & path, const PFlashStage2Pins & pins,
                        PFlashStage2Weights & out, std::string & error);

// ---------------------------------------------------------------- layout of one prompt (section 1)
struct PFlashStage2Layout {
    int latest = -1;                 // L, -1 = none
    PFlashTokenSpan latest_span{0, 0};
    std::vector<int> earlier;        // E, ascending positions
    std::vector<int> back;           // per E token: d = r(L) - r(j) >= 1
    std::vector<uint8_t> role;       // per E token: 0 user, 1 assistant, 2 tool
    std::vector<int> offset;         // per E token: t - b_j
    std::vector<int> length;         // per E token: n_j
};
// E and its structural columns. Fails ("message spans out of order or overlapping") unless the non-system
// messages' non-empty spans are ascending and disjoint, every span inside [0, n_tokens).
bool pflash_stage2_layout(const std::vector<PFlashRequestMessage> & messages, int n_tokens,
                          PFlashStage2Layout & out, std::string & error);

// ---------------------------------------------------------------- shared math (CPU, double accumulation)
// x̂ = x / sqrt(mean(x^2) + eps) * w_in, ``hidden`` wide.
void pflash_stage2_xhat(const float * x, const float * w_in, int hidden, float eps, double * out);

// u_all, u_s1 and the message head from m_all / m_s1 (each kStage2Hidden wide).
struct PFlashStage2Latest {
    std::vector<double> m_all, m_s1;     // [1024]
    std::vector<float> u_all, u_s1;      // [32]
    float lookback_logit = 0.0f;
    float vague_logit = 0.0f;
};
void pflash_stage2_message_head(const PFlashStage2Weights & w, PFlashStage2Latest & latest);

// x(t) for every E token, [N][143] F32. ``a``: [N][8] look-back mass (zeros for an empty seed); ``s1``: raw
// stage-1 logits over all tokens; ``cand``: [N][32] c(t) of the E tokens.
void pflash_stage2_features(const PFlashStage2Layout & layout, const std::vector<float> & a,
                            const std::vector<float> & s1, const std::vector<float> & cand,
                            const PFlashStage2Latest & latest, std::vector<float> & x);

// s2 = fc2(GELU_erf(fc1(LayerNorm(x)))) on the CPU, double accumulation; x [n][143] -> [n].
void pflash_stage2_mlp_cpu(const PFlashStage2Weights & w, const std::vector<float> & x, int n,
                           std::vector<float> & s2);

// The fixed look-back weights for the CPU path: head q/k ([2048][1024], [512][1024]) and the drafter's block-15
// norms (stored 1 + w, used as stored).
struct PFlashStage2HeadQK {
    std::vector<float> wq, wk, w_in, w_qn, w_kn;
    float rms_eps = 0.0f;
};

// The whole pass 1 on the CPU from F32 rows ([n_tokens][1024]) and s1 (over all tokens): seed, layout, m_all,
// m_s1, message head, a_h, x and s2 over E (computed whatever l_lb is). The seed is capped at 1024 - R (contract v3).
// False + error on a bad layout.
struct PFlashStage2Pass1 {
    PFlashStage2Layout layout;
    std::vector<int> seed;
    bool seed_capped = false;
    int reserve = 0;             // R
    PFlashStage2Latest latest;
    std::vector<float> a;        // [N][8]
    std::vector<float> x;        // [N][143]
    std::vector<float> s2;       // [N]
};
bool pflash_stage2_cpu_reference(const float * rows, int n_tokens,
                                 const std::vector<PFlashRequestMessage> & messages,
                                 const std::vector<float> & s1, float tau1,
                                 const PFlashStage2HeadQK & head, const PFlashStage2Weights & w,
                                 PFlashStage2Pass1 & out, std::string & error);

// s2 over every token (NaN outside E), the assembly's input.
std::vector<float> pflash_stage2_full_s2(const PFlashStage2Layout & layout, const std::vector<float> & s2,
                                         int n_tokens);

// ---------------------------------------------------------------- device path (ggml)
// Device copies of the tensors the graphs read: cand, norm, fc1, fc2 (lat_* and msg_* stay on the host).
struct PFlashStage2Device {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * cand_w = nullptr;   // ne [1024, 32]
    ggml_tensor * cand_b = nullptr;   // [32]
    ggml_tensor * norm_w = nullptr;   // [143]
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * fc1_w = nullptr;    // [143, 64]
    ggml_tensor * fc1_b = nullptr;    // [64]
    ggml_tensor * fc2_w = nullptr;    // [64, 1]
    ggml_tensor * fc2_b = nullptr;    // [1]
};
bool pflash_stage2_device_upload(ggml_backend_t backend, const PFlashStage2Weights & w,
                                 PFlashStage2Device & dev, std::string & error);
void pflash_stage2_device_free(PFlashStage2Device & dev);

// Key-graph pieces on x̂ = ``x_norm`` [hidden, n]: c(t) [32, n], and the column sum of [begin, end) [hidden].
ggml_tensor * pflash_stage2_build_cand(ggml_context * ctx, const PFlashStage2Device & dev, ggml_tensor * x_norm);
ggml_tensor * pflash_stage2_build_column_sum(ggml_context * ctx, ggml_tensor * x_norm, int begin, int end);

// Look-back attention on the device. ``keys``: [D, Hk, >= max(earlier)+1] F32 K vectors (the scoring session's,
// K_g(k) = RMSNorm(Wk x̂) * w_kn); ``wq`` [hidden, H*D] (head Q), ``w_in`` [hidden], ``w_qn`` [D]. Seed rows are the
// raw F32 rows ([nseed][hidden]). The E keys are gathered once; the seed runs in tiles whose [N, tile, 8] logits
// stay under ``tile_bytes``; each tile takes an exact softmax over all of E and adds its per-key mass into one
// [N, 8] accumulator. ``a`` = [N][8] mean mass (zeros for an empty seed or E).
struct PFlashStage2AttentionInputs {
    ggml_backend_t backend = nullptr;
    ggml_tensor * keys = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * w_in = nullptr;
    ggml_tensor * w_qn = nullptr;
    float rms_eps = 0.0f;
    size_t tile_bytes = (size_t) 64 << 20;
};
bool pflash_stage2_attention_gpu(const PFlashStage2AttentionInputs & in, const std::vector<int> & earlier,
                                 const float * seed_rows, int nseed, std::vector<float> & a, int & tiles,
                                 std::string & error);

// The token MLP on the device: x [n][143] -> s2 [n].
bool pflash_stage2_mlp_gpu(ggml_backend_t backend, const PFlashStage2Device & dev, float norm_eps,
                           const std::vector<float> & x, int n, std::vector<float> & s2, std::string & error);

// PFLASH_TRACE_STAGE2_FEATURES: appends one record per pass 1: char[4] "S2F2", i32 n_tokens, i32 N, i32 dim=143,
// i32 seed count, i32 message count M, f32 lookback_logit, f32 vague_logit, i32 messages[M][3] (role 0 system,
// 1 user, 2 assistant, 3 tool; span begin; span end), i32 seed[seed count], i32 earlier[N], f32 a[N][8],
// f32 x[N][143], f32 s2[N] (a, x zeros and s2 NaN when a_h / s2 were not computed). False when the write fails.
bool pflash_stage2_dump_features(const char * path, int n_tokens, const std::vector<PFlashRequestMessage> & messages,
                                 const PFlashStage2Layout & layout, const std::vector<int> & seed,
                                 const PFlashStage2Latest & latest, const std::vector<float> & a,
                                 const std::vector<float> & x, const std::vector<float> & s2);

} // namespace luce::pflash
