// Shared CPU sampler chain. See sampler.h for the protocol overview.

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
#include "geometric_sampler_cuda.h"
#endif

namespace dflash::common {

namespace {

// Binary search (quickselect via std::nth_element) for the smallest index
// `cut` such that the descending-by-value prefix cand[0,cut) has cumulative
// mass >= target, where "mass" of an element is given by `mass_of`. At each
// level, partitioning [lo,hi) at its midpoint puts exactly the top (mid-lo)
// elements of that range into [lo,mid) (in some order) — whichever half still
// contains the boundary is recursed into and the other is discarded. Mutates
// `cand` in place; only the final small base-case range ends up sorted, since
// order doesn't matter for the caller's draw, only which elements make the
// cut. Each level's cost is proportional to its (shrinking) range, so total
// work is O(cand.size()), not O(cand.size() log cand.size()) like a full sort,
// regardless of where the cutoff lands.
template <typename MassFn>
size_t nucleus_cutoff(std::vector<std::pair<float, int>> & cand, double target, MassFn mass_of) {
    constexpr size_t kBaseCase = 64;
    size_t lo = 0, hi = cand.size();
    while (hi - lo > kBaseCase) {
        const size_t mid = lo + (hi - lo) / 2;
        std::nth_element(cand.begin() + lo, cand.begin() + mid, cand.begin() + hi,
                         [](auto & a, auto & b){ return a.first > b.first; });
        double mass = 0.0;
        for (size_t i = lo; i < mid; i++) mass += mass_of(cand[i]);
        if (mass >= target) {
            hi = mid;        // cutoff lies within [lo, mid)
        } else {
            target -= mass;  // [lo, mid) fully included; keep searching [mid, hi)
            lo = mid;
        }
    }
    // Base case: small enough range left, sort it and walk the exact cumulative
    // cutoff (cand[0, lo) is already fully confirmed included from above).
    std::sort(cand.begin() + lo, cand.begin() + hi,
             [](auto & a, auto & b){ return a.first > b.first; });
    size_t cut = hi;
    double cum = 0.0;
    for (size_t i = lo; i < hi; i++) {
        cum += mass_of(cand[i]);
        if (cum >= target) { cut = i + 1; break; }
    }
    return cut;
}

// Draws a token from `cand`, whose .first fields are proportional
// probabilities (need not already sum to 1 or be sorted). `r_uniform` is a
// pre-drawn uniform in [0,1) supplied by the caller (drawn once per
// sample_logits call) so every path — GPU, GPU-assisted top_p, or CPU —
// consumes the same single RNG value.
int draw_from_weights(const std::vector<std::pair<float, int>> & cand, double r_uniform) {
    double Z = 0.0;
    for (auto & c : cand) Z += c.first;
    const double r = r_uniform * Z;
    double acc = 0.0;
    for (auto & c : cand) {
        acc += c.first;
        if (r <= acc) return c.second;
    }
    return cand.back().second;
}

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
// Given probabilities the GPU already computed (penalties + softmax(temp)
// applied, summing to ~1) for a pure top_p (no top_k) config, find the
// nucleus and draw. Skips all exp()/Z bookkeeping the raw-logit path needs,
// since the input is already normalized.
//
// Only worth calling for top_p without top_k: top_k's CPU cost is already
// cheap (partial_sort scales with k, not vocab — measured ~270-300us at
// vocab=151936), so a GPU round trip (kernel + D2H copy, ~500-800us) makes it
// slower, not faster. top_p's CPU cost without top_k is dominated by
// nucleus_cutoff's O(vocab) std::nth_element passes regardless of who
// computed the input probabilities, so skipping the CPU-side exp() pass here
// is a net win (measured ~1.4x faster end-to-end at vocab=151936).
int sample_from_gpu_probs(std::vector<float> & probs, double top_p, double r_uniform) {
    std::vector<std::pair<float, int>> cand(probs.size());
    for (size_t i = 0; i < probs.size(); i++) cand[i] = {probs[i], (int)i};

    double Z = 0.0;
    for (auto & c : cand) Z += c.first;
    const double target = top_p * Z;
    const size_t cut = nucleus_cutoff(cand, target, [](auto & c){ return (double)c.first; });
    cand.resize(cut);

    return draw_from_weights(cand, r_uniform);
}
#endif

}  // namespace

bool SamplerDistribution::valid() const {
    if (probabilities.empty() || support.empty()) return false;
    double total = 0.0;
    for (float probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0f) return false;
        total += probability;
    }
    double support_total = 0.0;
    for (int32_t token : support) {
        if (token < 0 || static_cast<size_t>(token) >= probabilities.size()) {
            return false;
        }
        support_total += probabilities[static_cast<size_t>(token)];
    }
    return std::fabs(total - 1.0) <= 1e-4 &&
           std::fabs(support_total - total) <= 1e-4;
}

float SamplerDistribution::probability_of(int32_t token) const {
    if (token < 0 || static_cast<size_t>(token) >= probabilities.size()) {
        return 0.0f;
    }
    return probabilities[static_cast<size_t>(token)];
}

bool SamplerDistribution::deterministic() const {
    int positive = 0;
    for (int32_t token : support) {
        if (probability_of(token) > 0.0f && ++positive > 1) return false;
    }
    return positive == 1;
}

int sample_distribution(
        const SamplerDistribution & distribution,
        double r_uniform) {
    if (distribution.probabilities.empty() || distribution.support.empty() ||
        !std::isfinite(r_uniform) || r_uniform < 0.0 || r_uniform >= 1.0) {
        return -1;
    }
    double total = 0.0;
    for (int32_t token : distribution.support) {
        if (token < 0 ||
            static_cast<size_t>(token) >= distribution.probabilities.size()) {
            return -1;
        }
        total += distribution.probability_of(token);
    }
    if (!(total > 0.0)) return -1;

    const double target = r_uniform * total;
    double cumulative = 0.0;
    int last_positive = -1;
    for (int32_t token : distribution.support) {
        const float probability = distribution.probability_of(token);
        if (probability <= 0.0f) continue;
        last_positive = token;
        cumulative += probability;
        if (target <= cumulative) return token;
    }
    return last_positive;
}

bool redirect_distribution_mass(
        SamplerDistribution & distribution,
        const int32_t * source_ids,
        int source_count,
        int32_t replacement) {
    if (!distribution.valid() || source_count < 0 ||
        (source_count > 0 && !source_ids) || replacement < 0 ||
        static_cast<size_t>(replacement) >= distribution.probabilities.size()) {
        return false;
    }

    double redirected = 0.0;
    for (int i = 0; i < source_count; ++i) {
        const int32_t source = source_ids[i];
        if (source < 0 ||
            static_cast<size_t>(source) >= distribution.probabilities.size() ||
            source == replacement) {
            return false;
        }
        redirected += distribution.probabilities[static_cast<size_t>(source)];
        distribution.probabilities[static_cast<size_t>(source)] = 0.0f;
    }
    distribution.probabilities[static_cast<size_t>(replacement)] +=
        static_cast<float>(redirected);
    if (std::find(distribution.support.begin(), distribution.support.end(),
                  replacement) == distribution.support.end()) {
        distribution.support.push_back(replacement);
    }
    return true;
}

static bool build_sampler_candidates(
        const float * logits_in,
        int vocab,
        const SamplerCfg & cfg,
        const std::vector<int32_t> & history,
        std::vector<std::pair<float, int>> & candidates) {
    candidates.clear();
    if (!logits_in || vocab <= 0) return false;
    candidates.resize(static_cast<size_t>(vocab));
    for (int token = 0; token < vocab; ++token) {
        candidates[static_cast<size_t>(token)] = {logits_in[token], token};
    }

    if (cfg.rep_pen > 1.0f && !history.empty()) {
        const int window = std::min(
            static_cast<int>(history.size()), cfg.rep_window);
        const int from = static_cast<int>(history.size()) - window;
        std::unordered_set<int> seen;
        for (int i = from; i < static_cast<int>(history.size()); ++i) {
            seen.insert(history[static_cast<size_t>(i)]);
        }
        for (auto & candidate : candidates) {
            if (seen.count(candidate.second) != 0) {
                candidate.first = candidate.first > 0.0f
                    ? candidate.first / cfg.rep_pen
                    : candidate.first * cfg.rep_pen;
            }
        }
    }

    if ((cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f) &&
        !history.empty()) {
        const int window = std::min(
            static_cast<int>(history.size()), cfg.rep_window);
        const int from = static_cast<int>(history.size()) - window;
        std::unordered_map<int, int> counts;
        for (int i = from; i < static_cast<int>(history.size()); ++i) {
            ++counts[history[static_cast<size_t>(i)]];
        }
        for (auto & candidate : candidates) {
            const auto it = counts.find(candidate.second);
            if (it != counts.end()) {
                candidate.first -= cfg.freq_pen * it->second;
                candidate.first -= cfg.pres_pen;
            }
        }
    }

    if (cfg.temp <= 0.0f) {
        const int token = std::max_element(
            candidates.begin(), candidates.end(),
            [](const auto & a, const auto & b) {
                return a.first < b.first;
            })->second;
        candidates.assign(1, {1.0f, token});
        return true;
    }

    const bool need_top_k = cfg.top_k > 0 && cfg.top_k < vocab;
    const bool need_top_p = cfg.top_p > 0.0f && cfg.top_p < 1.0f;
    if (need_top_k) {
        std::partial_sort(
            candidates.begin(), candidates.begin() + cfg.top_k,
            candidates.end(),
            [](const auto & a, const auto & b) {
                return a.first > b.first;
            });
        candidates.resize(static_cast<size_t>(cfg.top_k));
    }

    const float inverse_temperature =
        1.0f / std::max(1e-3f, cfg.temp);
    const float max_logit = need_top_k
        ? candidates.front().first
        : std::max_element(
              candidates.begin(), candidates.end(),
              [](const auto & a, const auto & b) {
                  return a.first < b.first;
              })->first;
    const float scaled_max = max_logit * inverse_temperature;

    if (need_top_p && !need_top_k) {
        double denominator = 0.0;
        for (const auto & candidate : candidates) {
            denominator += std::exp(
                static_cast<double>(candidate.first) *
                    inverse_temperature -
                scaled_max);
        }
        const double target = static_cast<double>(cfg.top_p) * denominator;
        const size_t cutoff = nucleus_cutoff(
            candidates, target, [&](const auto & candidate) {
                return std::exp(
                    static_cast<double>(candidate.first) *
                        inverse_temperature -
                    scaled_max);
            });
        candidates.resize(cutoff);
    }

    double denominator = 0.0;
    for (auto & candidate : candidates) {
        candidate.first = std::exp(
            candidate.first * inverse_temperature - scaled_max);
        denominator += candidate.first;
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
    for (auto & candidate : candidates) {
        candidate.first =
            static_cast<float>(candidate.first / denominator);
    }

    if (need_top_p && need_top_k) {
        double cumulative = 0.0;
        size_t cutoff = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumulative += candidates[i].first;
            if (cumulative >= cfg.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
    }
    return true;
}

bool build_sampler_distribution(
        const float * logits_in,
        int vocab,
        const SamplerCfg & cfg,
        const std::vector<int32_t> & history,
        SamplerDistribution & out) {
    out.probabilities.clear();
    out.support.clear();

    static thread_local std::vector<std::pair<float, int>> candidates;
    if (!build_sampler_candidates(
            logits_in, vocab, cfg, history, candidates)) {
        return false;
    }

    double total = 0.0;
    for (const auto & candidate : candidates) total += candidate.first;
    if (!(total > 0.0) || !std::isfinite(total)) return false;

    out.probabilities.assign(static_cast<size_t>(vocab), 0.0f);
    out.support.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        out.probabilities[static_cast<size_t>(candidate.second)] =
            static_cast<float>(candidate.first / total);
        out.support.push_back(candidate.second);
    }
    return true;
}

int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng) {
    double r_uniform = 0.0;
    if (cfg.temp > 0.0f) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        r_uniform = uniform(rng);
    }

#ifdef DFLASH27B_HAVE_GPU_SAMPLER
    if (gpu_sampler_enabled() && gpu_sampler_supports(cfg)) {
        const int token = geometric_sample_logits_cuda(
            logits_in, vocab, cfg, history, r_uniform,
            /*logits_on_device=*/false);
        if (token >= 0) return token;
    }

    const bool need_top_k = cfg.top_k > 0 && cfg.top_k < vocab;
    const bool need_top_p = cfg.top_p > 0.0f && cfg.top_p < 1.0f;
    if (cfg.temp > 0.0f && need_top_p && !need_top_k &&
        gpu_sampler_enabled()) {
        std::vector<float> gpu_probabilities(static_cast<size_t>(vocab));
        if (geometric_compute_probs_cuda(
                logits_in, vocab, cfg, history, gpu_probabilities.data(),
                /*logits_on_device=*/false)) {
            return sample_from_gpu_probs(
                gpu_probabilities, cfg.top_p, r_uniform);
        }
    }
#endif

    static thread_local std::vector<std::pair<float, int>> candidates;
    if (!build_sampler_candidates(
            logits_in, vocab, cfg, history, candidates)) {
        return -1;
    }
    return draw_from_weights(candidates, r_uniform);
}

bool parse_sampler_token(std::string & line, SamplerCfg & out) {
    auto pos = line.find(" samp=");
    if (pos == std::string::npos) return false;
    auto end = line.find(' ', pos + 1);
    std::string tok = (end == std::string::npos)
                          ? line.substr(pos + 6)
                          : line.substr(pos + 6, end - (pos + 6));
    line.erase(pos, (end == std::string::npos ? std::string::npos : end - pos));
    float t = 0.0f, tp = 1.0f, rp = 1.0f, fp = 0.0f, pp = 0.0f;
    int   tk = 0;
    unsigned long long sd = 0;
    int n = std::sscanf(tok.c_str(), "%f,%f,%d,%f,%llu,%f,%f",
                        &t, &tp, &tk, &rp, &sd, &fp, &pp);
    if (n < 1) return false;
    out.temp     = t;
    out.top_p    = tp;
    out.top_k    = tk;
    out.rep_pen  = rp;
    out.seed     = sd;
    out.freq_pen = fp;
    out.pres_pen = pp;
    return true;
}

}  // namespace dflash::common
