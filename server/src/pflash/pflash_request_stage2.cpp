#include "pflash_request_stage2.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace luce::pflash {

namespace {

constexpr int kH = kStage2Heads;
constexpr int kHk = kStage2KvHeads;
constexpr int kD = kStage2HeadDim;
constexpr int kE = kStage2Embed;

double gelu_erf(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

bool meta_string(gguf_context * g, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return false;
    out = gguf_get_val_str(g, id);
    return true;
}

bool meta_f32(gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return false;
    out = gguf_get_val_f32(g, id);
    return true;
}

bool meta_u32(gguf_context * g, const char * key, uint32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return false;
    out = gguf_get_val_u32(g, id);
    return true;
}

bool same_bits(float a, float b) {
    uint32_t x = 0, y = 0;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

} // namespace

// ---------------------------------------------------------------- loader

bool pflash_stage2_load(const std::string & path, const PFlashStage2Pins & pins,
                        PFlashStage2Weights & out, std::string & error) {
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        error = "stage-2 GGUF could not be opened: " + path;
        return false;
    }
    const auto fail = [&](const std::string & message) {
        gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        error = "stage-2 GGUF " + path + ": " + message;
        return false;
    };
    std::string value;
    const auto need_string = [&](const char * key, const std::string & expected) {
        if (!meta_string(g, key, value)) return std::string(key) + " is missing or not a string";
        if (value != expected) return std::string(key) + " is \"" + value + "\", the runtime needs \"" + expected + "\"";
        return std::string();
    };
    if (!meta_string(g, "general.architecture", value) || value != kStage2Arch) {
        return fail("general.architecture is not requeststage2");
    }
    const struct { const char * key; std::string expected; } strings[] = {
        {"requeststage2.schema", kStage2Schema},
        {"requeststage2.feature_version", kStage2FeatureVersion},
        {"requeststage2.assembly_version", kStage2AssemblyVersion},
        {"requeststage2.fallback_version", kStage2FallbackVersion},
        {"requeststage2.base_model", kStage2BaseModel},
        {"requeststage2.feature_tap", kStage2FeatureTap},
        {"requeststage2.stage1_candidate_rule", kRequestCandidateRuleV5},
    };
    for (const auto & s : strings) {
        const std::string problem = need_string(s.key, s.expected);
        if (!problem.empty()) return fail(problem);
    }
    if (pins.stage1_candidate_rule != kRequestCandidateRuleV5) {
        return fail(std::string("the loaded stage-1 GGUF's candidate rule is \"") + pins.stage1_candidate_rule +
                    "\"; stage 2 pairs only with " + kRequestCandidateRuleV5);
    }
    // Pins: the drafter, the scoring head and the stage-1 checkpoint this stage 2 was trained and exported with.
    const struct { const char * key; const std::string & expected; const char * what; } shas[] = {
        {"requeststage2.runtime_gguf_sha256", pins.drafter_sha256, "the loaded drafter GGUF"},
        {"requeststage2.head_gguf_sha256", pins.head_sha256, "the loaded scoring-head GGUF"},
        {"requeststage2.stage1_checkpoint_sha256", pins.stage1_checkpoint_sha256,
         "the loaded stage-1 GGUF's requestprobe.checkpoint_sha256"},
    };
    for (const auto & s : shas) {
        if (s.expected.empty()) return fail(std::string(s.key) + ": no value to pin against (" + s.what + ")");
        if (!meta_string(g, s.key, value)) return fail(std::string(s.key) + " is missing or not a string");
        if (value != s.expected) return fail(std::string(s.key) + " does not match " + s.what);
    }
    PFlashStage2Weights w;
    float s1_clamp = 0.0f;
    float budget_factor = 0.0f;
    if (!meta_f32(g, "requeststage2.norm_eps", w.norm_eps) || !(w.norm_eps > 0.0f) || !std::isfinite(w.norm_eps)) {
        return fail("requeststage2.norm_eps is missing or not a positive f32");
    }
    if (!meta_f32(g, "requeststage2.rms_eps", w.rms_eps) || !same_bits(w.rms_eps, pins.drafter_rms_eps)) {
        return fail("requeststage2.rms_eps is missing or differs from the drafter's layer_norm_rms_epsilon");
    }
    if (!meta_f32(g, "requeststage2.s1_clamp", s1_clamp) || s1_clamp != kStage2S1Clamp) {
        return fail("requeststage2.s1_clamp is missing or not 16");
    }
    if (!meta_f32(g, "requeststage2.budget_factor", budget_factor) || budget_factor != kStage2BudgetFactor) {
        return fail("requeststage2.budget_factor is missing or not 1.5 (the runtime constant)");
    }
    const struct { const char * key; float * dst; } taus[] = {
        {"requeststage2.tau2", &w.tau2}, {"requeststage2.tau_lb", &w.tau_lb}, {"requeststage2.tau_fb", &w.tau_fb},
    };
    for (const auto & t : taus) {
        if (!meta_f32(g, t.key, *t.dst) || std::isnan(*t.dst)) {
            return fail(std::string(t.key) + " is missing, not an f32, or NaN");
        }
    }
    if (!meta_string(g, "requeststage2.tau_source", w.tau_source) ||
        (w.tau_source != kStage2TauSourceCalibration && w.tau_source != kStage2TauSourceDev)) {
        return fail("requeststage2.tau_source is missing or not calibration-pool / dev-provisional");
    }
    const struct { const char * key; uint32_t expected; } u32s[] = {
        {"requeststage2.query_cap", (uint32_t) kRequestQueryCap},
        {"requeststage2.promoted_cap", (uint32_t) kStage2PromotedCap},
        {"requeststage2.post_reserve", (uint32_t) kRequestPostReserve},
        {"requeststage2.min_run", (uint32_t) kRequestMinRun},
        {"requeststage2.input_dim", (uint32_t) kStage2InputDim},
        {"requeststage2.hidden", (uint32_t) kStage2MlpHidden},
        {"requeststage2.embed", (uint32_t) kStage2Embed},
        {"requeststage2.heads", (uint32_t) kStage2Heads},
    };
    for (const auto & u : u32s) {
        uint32_t got = 0;
        if (!meta_u32(g, u.key, got) || got != u.expected) {
            return fail(std::string(u.key) + " is missing, not a u32, or not " + std::to_string(u.expected));
        }
    }
    // Tensors: ggml ne is the numpy shape reversed; all F32; nothing else in the file.
    struct Contract {
        const char * name;
        int64_t ne0;
        int64_t ne1;   // 0 = 1-D
        std::vector<float> * dst;
    };
    const Contract tensors[] = {
        {"requeststage2.cand.weight", kStage2Hidden, kE, &w.cand_w},
        {"requeststage2.cand.bias", kE, 0, &w.cand_b},
        {"requeststage2.lat_all.weight", kStage2Hidden, kE, &w.lat_all_w},
        {"requeststage2.lat_all.bias", kE, 0, &w.lat_all_b},
        {"requeststage2.lat_s1.weight", kStage2Hidden, kE, &w.lat_s1_w},
        {"requeststage2.lat_s1.bias", kE, 0, &w.lat_s1_b},
        {"requeststage2.norm.weight", kStage2InputDim, 0, &w.norm_w},
        {"requeststage2.norm.bias", kStage2InputDim, 0, &w.norm_b},
        {"requeststage2.fc1.weight", kStage2InputDim, kStage2MlpHidden, &w.fc1_w},
        {"requeststage2.fc1.bias", kStage2MlpHidden, 0, &w.fc1_b},
        {"requeststage2.fc2.weight", kStage2MlpHidden, 0, &w.fc2_w},
        {"requeststage2.fc2.bias", 1, 0, &w.fc2_b},
        {"requeststage2.msg_fc1.weight", 2 * kStage2Hidden, kStage2MsgHidden, &w.msg_fc1_w},
        {"requeststage2.msg_fc1.bias", kStage2MsgHidden, 0, &w.msg_fc1_b},
        {"requeststage2.msg_fc2.weight", kStage2MsgHidden, 2, &w.msg_fc2_w},
        {"requeststage2.msg_fc2.bias", 2, 0, &w.msg_fc2_b},
    };
    if (gguf_get_n_tensors(g) != (int64_t) (sizeof(tensors) / sizeof(tensors[0]))) {
        return fail("tensor count is " + std::to_string(gguf_get_n_tensors(g)) + ", the contract has 16");
    }
    for (const auto & t : tensors) {
        ggml_tensor * src = data_ctx ? ggml_get_tensor(data_ctx, t.name) : nullptr;
        const int dims = t.ne1 == 0 ? 1 : 2;
        if (!src || src->type != GGML_TYPE_F32 || ggml_n_dims(src) != dims || src->ne[0] != t.ne0 ||
            (dims == 2 && src->ne[1] != t.ne1) || !ggml_is_contiguous(src)) {
            return fail(std::string("tensor contract mismatch: ") + t.name);
        }
        const size_t n = (size_t) ggml_nelements(src);
        t.dst->assign((const float *) src->data, (const float *) src->data + n);
        for (const float v : *t.dst) {
            if (!std::isfinite(v)) return fail(std::string("non-finite value in ") + t.name);
        }
    }
    gguf_free(g);
    ggml_free(data_ctx);
    out = std::move(w);
    return true;
}

// ---------------------------------------------------------------- layout

bool pflash_stage2_layout(const std::vector<PFlashRequestMessage> & messages, int n_tokens,
                          PFlashStage2Layout & out, std::string & error) {
    out = PFlashStage2Layout{};
    int last_end = 0;
    for (const auto & m : messages) {
        if (m.role == PFlashRequestRole::System || m.span.end <= m.span.begin) continue;
        if (m.span.begin < last_end || m.span.begin < 0 || m.span.end > n_tokens) {
            error = "message spans out of order, overlapping or out of range";
            return false;
        }
        last_end = m.span.end;
    }
    out.latest = pflash_request_latest_user(messages);
    if (out.latest < 0) return true;
    out.latest_span = messages[(size_t) out.latest].span;
    std::vector<int> rank(messages.size(), -1);
    int r = 0;
    for (size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role != PFlashRequestRole::System) rank[i] = r++;
    }
    for (int i = 0; i < out.latest; ++i) {
        const auto & m = messages[(size_t) i];
        if (m.role == PFlashRequestRole::System || m.span.end <= m.span.begin) continue;
        const uint8_t role = m.role == PFlashRequestRole::User ? 0 : m.role == PFlashRequestRole::Assistant ? 1 : 2;
        const int n = m.span.end - m.span.begin;
        for (int t = m.span.begin; t < m.span.end; ++t) {
            out.earlier.push_back(t);
            out.back.push_back(rank[(size_t) out.latest] - rank[(size_t) i]);
            out.role.push_back(role);
            out.offset.push_back(t - m.span.begin);
            out.length.push_back(n);
        }
    }
    return true;
}

// ---------------------------------------------------------------- shared math

void pflash_stage2_xhat(const float * x, const float * w_in, int hidden, float eps, double * out) {
    double ss = 0.0;
    for (int k = 0; k < hidden; ++k) ss += (double) x[k] * (double) x[k];
    const double inv = 1.0 / std::sqrt(ss / hidden + (double) eps);
    for (int k = 0; k < hidden; ++k) out[k] = (double) x[k] * inv * (double) w_in[k];
}

namespace {

// y[o] = W[o][:] . x + b[o], W row-major [out][in].
void linear(const std::vector<float> & W, const std::vector<float> & b, const double * x, int in, int outs,
            double * y) {
    for (int o = 0; o < outs; ++o) {
        const float * row = W.data() + (size_t) o * in;
        double acc = (double) b[(size_t) o];
        for (int k = 0; k < in; ++k) acc += (double) row[k] * x[k];
        y[o] = acc;
    }
}

} // namespace

void pflash_stage2_message_head(const PFlashStage2Weights & w, PFlashStage2Latest & latest) {
    std::vector<double> u((size_t) kE);
    latest.u_all.assign((size_t) kE, 0.0f);
    latest.u_s1.assign((size_t) kE, 0.0f);
    linear(w.lat_all_w, w.lat_all_b, latest.m_all.data(), kStage2Hidden, kE, u.data());
    for (int k = 0; k < kE; ++k) latest.u_all[(size_t) k] = (float) u[(size_t) k];
    linear(w.lat_s1_w, w.lat_s1_b, latest.m_s1.data(), kStage2Hidden, kE, u.data());
    for (int k = 0; k < kE; ++k) latest.u_s1[(size_t) k] = (float) u[(size_t) k];
    std::vector<double> z((size_t) 2 * kStage2Hidden);
    std::copy(latest.m_all.begin(), latest.m_all.end(), z.begin());
    std::copy(latest.m_s1.begin(), latest.m_s1.end(), z.begin() + kStage2Hidden);
    std::vector<double> h((size_t) kStage2MsgHidden);
    linear(w.msg_fc1_w, w.msg_fc1_b, z.data(), 2 * kStage2Hidden, kStage2MsgHidden, h.data());
    for (auto & v : h) v = gelu_erf(v);
    double logits[2];
    linear(w.msg_fc2_w, w.msg_fc2_b, h.data(), kStage2MsgHidden, 2, logits);
    latest.lookback_logit = (float) logits[0];
    latest.vague_logit = (float) logits[1];
}

void pflash_stage2_features(const PFlashStage2Layout & layout, const std::vector<float> & a,
                            const std::vector<float> & s1, const std::vector<float> & cand,
                            const PFlashStage2Latest & latest, std::vector<float> & x) {
    const size_t N = layout.earlier.size();
    x.assign(N * (size_t) kStage2InputDim, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        float * row = x.data() + i * (size_t) kStage2InputDim;
        for (int h = 0; h < kH; ++h) {
            row[kStage2ColA + h] = (float) std::log1p((double) N * (double) a[i * kH + (size_t) h]);
        }
        row[kStage2ColBack] = (float) std::log1p((double) layout.back[i]);
        row[kStage2ColRole + layout.role[i]] = 1.0f;
        row[kStage2ColRelpos] = (float) ((double) layout.offset[i] / (double) layout.length[i]);
        row[kStage2ColLogLen] = (float) std::log1p((double) layout.length[i]);
        float v = s1[(size_t) layout.earlier[i]];
        if (v > kStage2S1Clamp) v = kStage2S1Clamp;
        else if (v < -kStage2S1Clamp) v = -kStage2S1Clamp;   // NaN stays NaN
        row[kStage2ColS1] = v;
        const float * c = cand.data() + i * (size_t) kE;
        for (int k = 0; k < kE; ++k) {
            row[kStage2ColC + k] = c[k];
            row[kStage2ColCxAll + k] = (float) ((double) c[k] * (double) latest.u_all[(size_t) k]);
            row[kStage2ColUAll + k] = latest.u_all[(size_t) k];
            row[kStage2ColUS1 + k] = latest.u_s1[(size_t) k];
        }
    }
}

void pflash_stage2_mlp_cpu(const PFlashStage2Weights & w, const std::vector<float> & x, int n,
                           std::vector<float> & s2) {
    s2.assign((size_t) n, 0.0f);
    std::vector<double> v((size_t) kStage2InputDim);
    std::vector<double> h((size_t) kStage2MlpHidden);
    const double eps = (double) w.norm_eps;
    for (int i = 0; i < n; ++i) {
        const float * row = x.data() + (size_t) i * kStage2InputDim;
        double mu = 0.0;
        for (int k = 0; k < kStage2InputDim; ++k) mu += (double) row[k];
        mu /= kStage2InputDim;
        double var = 0.0;
        for (int k = 0; k < kStage2InputDim; ++k) var += ((double) row[k] - mu) * ((double) row[k] - mu);
        var /= kStage2InputDim;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int k = 0; k < kStage2InputDim; ++k) {
            v[(size_t) k] = ((double) row[k] - mu) * inv * (double) w.norm_w[(size_t) k] + (double) w.norm_b[(size_t) k];
        }
        linear(w.fc1_w, w.fc1_b, v.data(), kStage2InputDim, kStage2MlpHidden, h.data());
        double acc = (double) w.fc2_b[0];
        for (int k = 0; k < kStage2MlpHidden; ++k) acc += (double) w.fc2_w[(size_t) k] * gelu_erf(h[(size_t) k]);
        s2[(size_t) i] = (float) acc;
    }
}

std::vector<float> pflash_stage2_full_s2(const PFlashStage2Layout & layout, const std::vector<float> & s2,
                                         int n_tokens) {
    std::vector<float> out((size_t) std::max(0, n_tokens), std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < layout.earlier.size() && i < s2.size(); ++i) out[(size_t) layout.earlier[i]] = s2[i];
    return out;
}

bool pflash_stage2_cpu_reference(const float * rows, int n_tokens,
                                 const std::vector<PFlashRequestMessage> & messages,
                                 const std::vector<float> & s1, float tau1,
                                 const PFlashStage2HeadQK & head, const PFlashStage2Weights & w,
                                 PFlashStage2Pass1 & out, std::string & error) {
    out = PFlashStage2Pass1{};
    if ((int) s1.size() != n_tokens) {
        error = "s1 does not cover the prompt";
        return false;
    }
    if (!pflash_stage2_layout(messages, n_tokens, out.layout, error)) return false;
    const int hidden = kStage2Hidden;
    const float eps = head.rms_eps;
    const auto xhat = [&](int t, double * dst) {
        pflash_stage2_xhat(rows + (size_t) t * hidden, head.w_in.data(), hidden, eps, dst);
    };
    const int L = out.layout.latest;
    std::vector<int> post;
    out.seed = pflash_request_capped_seed(s1, tau1, messages, kRequestQueryCap, kRequestPostReserve, out.reserve,
                                          post, &out.seed_capped);
    auto & lat = out.latest;
    lat.m_all.assign((size_t) hidden, 0.0);
    std::vector<double> xh((size_t) hidden);
    if (L >= 0 && out.layout.latest_span.end > out.layout.latest_span.begin) {
        const auto span = out.layout.latest_span;
        for (int t = span.begin; t < span.end; ++t) {
            xhat(t, xh.data());
            for (int k = 0; k < hidden; ++k) lat.m_all[(size_t) k] += xh[(size_t) k];
        }
        for (auto & v : lat.m_all) v /= (double) (span.end - span.begin);
    }
    if (out.seed.empty()) {
        lat.m_s1 = lat.m_all;
    } else {
        lat.m_s1.assign((size_t) hidden, 0.0);
        for (const int t : out.seed) {
            xhat(t, xh.data());
            for (int k = 0; k < hidden; ++k) lat.m_s1[(size_t) k] += xh[(size_t) k];
        }
        for (auto & v : lat.m_s1) v /= (double) out.seed.size();
    }
    pflash_stage2_message_head(w, lat);

    const auto & E = out.layout.earlier;
    const size_t N = E.size();
    // c(t) and K of the E tokens, Q of the seed (double; F32 storage of c as the device keeps it).
    std::vector<float> cand(N * (size_t) kE);
    std::vector<double> K(N * (size_t) kHk * kD);
    std::vector<double> proj((size_t) kH * kD);
    const auto head_norm = [&](double * v, const std::vector<float> & wn) {
        double ss = 0.0;
        for (int d = 0; d < kD; ++d) ss += v[d] * v[d];
        const double inv = 1.0 / std::sqrt(ss / kD + (double) eps);
        for (int d = 0; d < kD; ++d) v[d] = v[d] * inv * (double) wn[(size_t) d];
    };
    const std::vector<float> zero_b((size_t) kH * kD, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        xhat(E[i], xh.data());
        double c[kE];
        linear(w.cand_w, w.cand_b, xh.data(), hidden, kE, c);
        for (int k = 0; k < kE; ++k) cand[i * kE + (size_t) k] = (float) c[k];
        linear(head.wk, zero_b, xh.data(), hidden, kHk * kD, proj.data());
        for (int g = 0; g < kHk; ++g) head_norm(proj.data() + (size_t) g * kD, head.w_kn);
        std::copy_n(proj.data(), (size_t) kHk * kD, K.data() + i * (size_t) kHk * kD);
    }
    std::vector<double> a(N * (size_t) kH, 0.0);
    if (!out.seed.empty() && N > 0) {
        std::vector<double> logits(N);
        for (const int q : out.seed) {
            xhat(q, xh.data());
            linear(head.wq, zero_b, xh.data(), hidden, kH * kD, proj.data());
            for (int h = 0; h < kH; ++h) head_norm(proj.data() + (size_t) h * kD, head.w_qn);
            for (int h = 0; h < kH; ++h) {
                const double * qv = proj.data() + (size_t) h * kD;
                const int g = h / (kH / kHk);
                double mx = -std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < N; ++i) {
                    const double * kv = K.data() + (i * kHk + (size_t) g) * kD;
                    double dot = 0.0;
                    for (int d = 0; d < kD; ++d) dot += qv[d] * kv[d];
                    logits[i] = dot / 16.0;
                    mx = std::max(mx, logits[i]);
                }
                double sum = 0.0;
                for (size_t i = 0; i < N; ++i) { logits[i] = std::exp(logits[i] - mx); sum += logits[i]; }
                for (size_t i = 0; i < N; ++i) a[i * kH + (size_t) h] += logits[i] / sum;
            }
        }
        for (auto & v : a) v /= (double) out.seed.size();
    }
    out.a.resize(a.size());
    for (size_t k = 0; k < a.size(); ++k) out.a[k] = (float) a[k];
    // A_h from the double mass (the F32 copy in ``out.a`` is what the device path produces).
    pflash_stage2_features(out.layout, out.a, s1, cand, lat, out.x);
    for (size_t i = 0; i < N; ++i) {
        for (int h = 0; h < kH; ++h) {
            out.x[i * kStage2InputDim + (size_t) (kStage2ColA + h)] =
                (float) std::log1p((double) N * a[i * kH + (size_t) h]);
        }
    }
    pflash_stage2_mlp_cpu(w, out.x, (int) N, out.s2);
    return true;
}

// ---------------------------------------------------------------- device path

bool pflash_stage2_device_upload(ggml_backend_t backend, const PFlashStage2Weights & w,
                                 PFlashStage2Device & dev, std::string & error) {
    pflash_stage2_device_free(dev);
    ggml_init_params ip{};
    ip.mem_size = 16 * ggml_tensor_overhead();
    ip.no_alloc = true;
    dev.ctx = ggml_init(ip);
    if (!dev.ctx) {
        error = "stage-2 device context allocation failed";
        return false;
    }
    struct T { ggml_tensor ** dst; int64_t ne0; int64_t ne1; const std::vector<float> * src; const char * name; };
    const T tensors[] = {
        {&dev.cand_w, kStage2Hidden, kE, &w.cand_w, "requeststage2.cand.weight"},
        {&dev.cand_b, kE, 1, &w.cand_b, "requeststage2.cand.bias"},
        {&dev.norm_w, kStage2InputDim, 1, &w.norm_w, "requeststage2.norm.weight"},
        {&dev.norm_b, kStage2InputDim, 1, &w.norm_b, "requeststage2.norm.bias"},
        {&dev.fc1_w, kStage2InputDim, kStage2MlpHidden, &w.fc1_w, "requeststage2.fc1.weight"},
        {&dev.fc1_b, kStage2MlpHidden, 1, &w.fc1_b, "requeststage2.fc1.bias"},
        {&dev.fc2_w, kStage2MlpHidden, 1, &w.fc2_w, "requeststage2.fc2.weight"},
        {&dev.fc2_b, 1, 1, &w.fc2_b, "requeststage2.fc2.bias"},
    };
    for (const auto & t : tensors) {
        if ((int64_t) t.src->size() != t.ne0 * t.ne1) {
            pflash_stage2_device_free(dev);
            error = std::string("stage-2 weights have the wrong size: ") + t.name;
            return false;
        }
        *t.dst = t.ne1 == 1 && t.dst != &dev.fc2_w
            ? ggml_new_tensor_1d(dev.ctx, GGML_TYPE_F32, t.ne0)
            : ggml_new_tensor_2d(dev.ctx, GGML_TYPE_F32, t.ne0, t.ne1);
        ggml_set_name(*t.dst, t.name);
    }
    dev.buf = ggml_backend_alloc_ctx_tensors(dev.ctx, backend);
    if (!dev.buf) {
        pflash_stage2_device_free(dev);
        error = "stage-2 device buffer allocation failed";
        return false;
    }
    for (const auto & t : tensors) {
        ggml_backend_tensor_set(*t.dst, t.src->data(), 0, t.src->size() * sizeof(float));
    }
    return true;
}

void pflash_stage2_device_free(PFlashStage2Device & dev) {
    if (dev.buf) ggml_backend_buffer_free(dev.buf);
    if (dev.ctx) ggml_free(dev.ctx);
    dev = PFlashStage2Device{};
}

ggml_tensor * pflash_stage2_build_cand(ggml_context * ctx, const PFlashStage2Device & dev, ggml_tensor * x_norm) {
    return ggml_add(ctx, ggml_mul_mat(ctx, dev.cand_w, x_norm), dev.cand_b);
}

ggml_tensor * pflash_stage2_build_column_sum(ggml_context * ctx, ggml_tensor * x_norm, int begin, int end) {
    ggml_tensor * cols = ggml_view_2d(ctx, x_norm, x_norm->ne[0], end - begin, x_norm->nb[1],
                                      (size_t) begin * x_norm->nb[1]);
    ggml_tensor * sums = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, cols)));   // [1, hidden]
    return ggml_reshape_1d(ctx, sums, x_norm->ne[0]);
}

namespace {

// A context of ``n`` tensor slots for a graph of ``nodes`` nodes.
ggml_context * graph_ctx(size_t tensors, size_t nodes) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(nodes, false) + 64 * 1024;
    ip.no_alloc = true;
    return ggml_init(ip);
}

struct Persistent {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~Persistent() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

bool run_graph(ggml_backend_t backend, ggml_gallocr_t alloc, ggml_cgraph * gf) {
    return ggml_gallocr_alloc_graph(alloc, gf) && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

constexpr int kKeyChunk = 8192;   // no tensor puts a sequence length into a HIP grid y/z dimension

} // namespace

bool pflash_stage2_attention_gpu(const PFlashStage2AttentionInputs & in, const std::vector<int> & earlier,
                                 const float * seed_rows, int nseed, std::vector<float> & a, int & tiles,
                                 std::string & error) {
    const int N = (int) earlier.size();
    const int hidden = kStage2Hidden;
    a.assign((size_t) N * kH, 0.0f);
    tiles = 0;
    if (N == 0 || nseed <= 0) return true;
    if (!in.backend || !in.keys || in.keys->type != GGML_TYPE_F32 || in.keys->ne[0] != kD ||
        in.keys->ne[1] != kHk || !ggml_is_contiguous(in.keys) || !in.wq || in.wq->ne[0] != hidden ||
        in.wq->ne[1] != kH * kD || !in.w_in || in.w_in->ne[0] != hidden || !in.w_qn || in.w_qn->ne[0] != kD) {
        error = "stage-2 look-back: tensor shapes do not match the u314 head";
        return false;
    }
    for (int i = 0; i < N; ++i) {
        if (earlier[(size_t) i] < 0 || earlier[(size_t) i] >= in.keys->ne[2] ||
            (i > 0 && earlier[(size_t) i] <= earlier[(size_t) i - 1])) {
            error = "stage-2 look-back: earlier positions out of range or not ascending";
            return false;
        }
    }
    // Tile of seed queries: its [N, tile, 8] F32 logits stay under tile_bytes.
    // Rounded down to a multiple of 16 queries when it holds 16 or more (64 GEMM columns per KV group).
    const size_t per_query = (size_t) N * kH * sizeof(float);
    size_t fit = std::max<size_t>(1, in.tile_bytes / per_query);
    if (fit >= 16) fit -= fit % 16;
    const int tile = (int) std::min<size_t>((size_t) nseed, fit);

    Persistent p;
    p.ctx = graph_ctx(8, 0);
    if (!p.ctx) { error = "stage-2 look-back: context allocation failed"; return false; }
    ggml_tensor * idx = ggml_new_tensor_1d(p.ctx, GGML_TYPE_I32, N);
    ggml_tensor * kp = ggml_new_tensor_3d(p.ctx, GGML_TYPE_F32, kD, N, kHk);       // E keys, per group
    ggml_tensor * xs = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, hidden, nseed);    // seed rows
    ggml_tensor * qp = ggml_new_tensor_3d(p.ctx, GGML_TYPE_F32, kD, nseed, kH);    // seed Q, per head
    ggml_tensor * acc = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, N, kH);           // per-key mass sums
    ggml_tensor * logits = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, (int64_t) N * tile * kH);
    p.buf = ggml_backend_alloc_ctx_tensors(p.ctx, in.backend);
    if (!p.buf) { error = "stage-2 look-back: device buffer allocation failed"; return false; }
    ggml_backend_tensor_set(idx, earlier.data(), 0, (size_t) N * sizeof(int32_t));
    ggml_backend_tensor_set(xs, seed_rows, 0, (size_t) hidden * nseed * sizeof(float));
    ggml_backend_tensor_memset(acc, 0, 0, ggml_nbytes(acc));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(in.backend));
    struct AllocFree { ggml_gallocr_t a; ~AllocFree() { ggml_gallocr_free(a); } } alloc_free{alloc};
    {
        // Gather the E keys into [D, N, Hk] (contiguous per group) and project the seed's Q.
        const int chunks = (N + kKeyChunk - 1) / kKeyChunk;
        ggml_context * ctx = graph_ctx((size_t) 16 + 8 * chunks, 1024);
        if (!ctx) { error = "stage-2 look-back: graph context allocation failed"; return false; }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        ggml_tensor * keys2d = ggml_view_2d(ctx, in.keys, (int64_t) kD * kHk, in.keys->ne[2], in.keys->nb[2], 0);
        for (int b = 0; b < N; b += kKeyChunk) {
            const int n = std::min(kKeyChunk, N - b);
            ggml_tensor * ids = ggml_view_1d(ctx, idx, n, (size_t) b * sizeof(int32_t));
            ggml_tensor * rows = ggml_get_rows(ctx, keys2d, ids);                         // [D*Hk, n]
            ggml_tensor * perm = ggml_permute(ctx, ggml_reshape_3d(ctx, rows, kD, kHk, n), 0, 2, 1, 3);
            ggml_tensor * dst = ggml_view_3d(ctx, kp, kD, n, kHk, kp->nb[1], kp->nb[2], (size_t) b * kp->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, perm, dst));
        }
        ggml_tensor * q_in = ggml_mul(ctx, ggml_rms_norm(ctx, xs, in.rms_eps), in.w_in);
        ggml_tensor * Q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, in.wq, q_in), kD, kH, nseed);
        Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, in.rms_eps), in.w_qn);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3), qp));   // [D, nseed, H]
        const bool ok = run_graph(in.backend, alloc, gf);
        ggml_free(ctx);
        if (!ok) { error = "stage-2 look-back: key gather / seed Q graph failed"; return false; }
    }
    const int chunks = (N + kKeyChunk - 1) / kKeyChunk;
    for (int q0 = 0; q0 < nseed; q0 += tile) {
        const int n = std::min(tile, nseed - q0);
        ggml_context * ctx = graph_ctx((size_t) 32 + 12 * chunks * kHk, 2048);
        if (!ctx) { error = "stage-2 look-back: tile context allocation failed"; return false; }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);
        // This tile's logits as a contiguous [N, n, H] block of the buffer.
        ggml_tensor * lg = ggml_view_3d(ctx, logits, N, n, kH, (size_t) N * sizeof(float),
                                        (size_t) N * n * sizeof(float), 0);
        ggml_tensor * qt = ggml_cont(ctx, ggml_view_3d(ctx, qp, kD, n, kH, qp->nb[1], qp->nb[2],
                                                       (size_t) q0 * qp->nb[1]));      // [D, n, H]
        const int per_group = kH / kHk;
        for (int g = 0; g < kHk; ++g) {
            // Heads g*4 .. g*4+3 read KV group g (torch repeat_interleave = the runtime's ggml_repeat layout).
            ggml_tensor * qg = ggml_view_2d(ctx, qt, kD, (int64_t) n * per_group, qt->nb[1],
                                            (size_t) g * per_group * qt->nb[2]);
            for (int b = 0; b < N; b += kKeyChunk) {
                const int nk = std::min(kKeyChunk, N - b);
                ggml_tensor * kg = ggml_view_2d(ctx, kp, kD, nk, kp->nb[1],
                                                (size_t) g * kp->nb[2] + (size_t) b * kp->nb[1]);
                ggml_tensor * part = ggml_mul_mat(ctx, kg, qg);                            // [nk, n*4]
                ggml_tensor * dst = ggml_view_3d(ctx, lg, nk, n, per_group, lg->nb[1], lg->nb[2],
                                                 (size_t) b * sizeof(float) + (size_t) g * per_group * lg->nb[2]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_3d(ctx, part, nk, n, per_group), dst));
            }
        }
        // Exact softmax over all of E per (query, head), then the per-key sum over this tile's queries.
        ggml_tensor * probs = ggml_soft_max_ext_inplace(ctx, lg, nullptr, 1.0f / std::sqrt((float) kD), 0.0f);
        ggml_tensor * per_key = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, probs, 1, 0, 2, 3)));  // [1,N,H]
        ggml_tensor * sum = ggml_add(ctx, acc, ggml_reshape_2d(ctx, per_key, N, kH));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, sum, acc));
        const bool ok = run_graph(in.backend, alloc, gf);
        ggml_free(ctx);
        if (!ok) { error = "stage-2 look-back: attention tile graph failed"; return false; }
        ++tiles;
    }
    std::vector<float> host((size_t) N * kH);
    ggml_backend_tensor_get(acc, host.data(), 0, host.size() * sizeof(float));
    for (int h = 0; h < kH; ++h) {
        for (int i = 0; i < N; ++i) {
            const float v = host[(size_t) h * N + (size_t) i];
            if (!std::isfinite(v)) {
                error = "stage-2 look-back: non-finite attention mass";
                return false;
            }
            a[(size_t) i * kH + (size_t) h] = (float) ((double) v / (double) nseed);
        }
    }
    return true;
}

bool pflash_stage2_mlp_gpu(ggml_backend_t backend, const PFlashStage2Device & dev, float norm_eps,
                           const std::vector<float> & x, int n, std::vector<float> & s2, std::string & error) {
    s2.assign((size_t) std::max(0, n), 0.0f);
    if (n <= 0) return true;
    if ((int64_t) x.size() != (int64_t) n * kStage2InputDim || !dev.buf) {
        error = "stage-2 MLP: bad input";
        return false;
    }
    Persistent p;
    p.ctx = graph_ctx(4, 0);
    if (!p.ctx) { error = "stage-2 MLP: context allocation failed"; return false; }
    ggml_tensor * xin = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, kStage2InputDim, n);
    ggml_tensor * out = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, n);
    p.buf = ggml_backend_alloc_ctx_tensors(p.ctx, backend);
    if (!p.buf) { error = "stage-2 MLP: device buffer allocation failed"; return false; }
    ggml_backend_tensor_set(xin, x.data(), 0, x.size() * sizeof(float));
    const int chunks = (n + kKeyChunk - 1) / kKeyChunk;
    ggml_context * ctx = graph_ctx((size_t) 16 + 16 * chunks, 4096);
    if (!ctx) { error = "stage-2 MLP: graph context allocation failed"; return false; }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (int b = 0; b < n; b += kKeyChunk) {
        const int m = std::min(kKeyChunk, n - b);
        ggml_tensor * xv = ggml_view_2d(ctx, xin, kStage2InputDim, m, xin->nb[1], (size_t) b * xin->nb[1]);
        ggml_tensor * h = ggml_norm(ctx, xv, norm_eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, dev.norm_w), dev.norm_b);
        h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, dev.fc1_w, h), dev.fc1_b));   // [64, m]
        ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, dev.fc2_w, h), dev.fc2_b);          // [1, m]
        ggml_tensor * dst = ggml_view_1d(ctx, out, m, (size_t) b * sizeof(float));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, o, m), dst));
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool ok = run_graph(backend, alloc, gf);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (!ok) { error = "stage-2 MLP: graph failed"; return false; }
    ggml_backend_tensor_get(out, s2.data(), 0, (size_t) n * sizeof(float));
    return true;
}

bool pflash_stage2_dump_features(const char * path, int n_tokens, const std::vector<PFlashRequestMessage> & messages,
                                 const PFlashStage2Layout & layout, const std::vector<int> & seed,
                                 const PFlashStage2Latest & latest, const std::vector<float> & a,
                                 const std::vector<float> & x, const std::vector<float> & s2) {
    const int32_t N = (int32_t) layout.earlier.size();
    std::vector<float> a_out = a;
    std::vector<float> x_out = x;
    a_out.resize((size_t) N * kH, 0.0f);
    x_out.resize((size_t) N * kStage2InputDim, 0.0f);
    std::vector<float> s2_out((size_t) N, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < s2.size() && i < s2_out.size(); ++i) s2_out[i] = s2[i];
    std::vector<int32_t> m;
    for (const auto & message : messages) {
        m.push_back((int32_t) message.role);
        m.push_back(message.span.begin);
        m.push_back(message.span.end);
    }
    const int32_t header[5] = {n_tokens, N, kStage2InputDim, (int32_t) seed.size(), (int32_t) messages.size()};
    const float logits[2] = {latest.lookback_logit, latest.vague_logit};
    FILE * file = std::fopen(path, "ab");
    if (!file) return false;
    bool ok = std::fwrite("S2F2", 1, 4, file) == 4 &&
        std::fwrite(header, sizeof(header), 1, file) == 1 &&
        std::fwrite(logits, sizeof(logits), 1, file) == 1 &&
        std::fwrite(m.data(), sizeof(int32_t), m.size(), file) == m.size() &&
        std::fwrite(seed.data(), sizeof(int32_t), seed.size(), file) == seed.size() &&
        std::fwrite(layout.earlier.data(), sizeof(int32_t), (size_t) N, file) == (size_t) N &&
        std::fwrite(a_out.data(), sizeof(float), a_out.size(), file) == a_out.size() &&
        std::fwrite(x_out.data(), sizeof(float), x_out.size(), file) == x_out.size() &&
        std::fwrite(s2_out.data(), sizeof(float), s2_out.size(), file) == s2_out.size();
    ok = std::fclose(file) == 0 && ok;
    return ok;
}

} // namespace luce::pflash
