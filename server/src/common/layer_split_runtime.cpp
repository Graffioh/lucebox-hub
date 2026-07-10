#include "layer_split_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace dflash::common {

namespace {

bool logits_debug_enabled() {
    const char * value = std::getenv("DFLASH_DS4_LOGITS_DEBUG");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

void log_logits_debug(const std::vector<float> & logits,
                      int step,
                      int selected_token) {
    if (!logits_debug_enabled()) return;

    size_t finite_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    float finite_min = std::numeric_limits<float>::infinity();
    float finite_max = -std::numeric_limits<float>::infinity();
    std::array<std::pair<float, int>, 5> top{};
    for (auto & entry : top) {
        entry = {-std::numeric_limits<float>::infinity(), -1};
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        const float value = logits[i];
        if (std::isnan(value)) {
            ++nan_count;
            continue;
        }
        if (!std::isfinite(value)) {
            ++inf_count;
            continue;
        }

        ++finite_count;
        finite_min = std::min(finite_min, value);
        finite_max = std::max(finite_max, value);
        for (size_t rank = 0; rank < top.size(); ++rank) {
            if (value <= top[rank].first) continue;
            for (size_t shift = top.size() - 1; shift > rank; --shift) {
                top[shift] = top[shift - 1];
            }
            top[rank] = {value, (int)i};
            break;
        }
    }

    const float selected_value =
        selected_token >= 0 && (size_t)selected_token < logits.size()
            ? logits[(size_t)selected_token]
            : std::numeric_limits<float>::quiet_NaN();
    std::fprintf(stderr,
        "[deepseek4-logits] step=%d size=%zu selected=%d selected_value=%g "
        "finite=%zu nan=%zu inf=%zu finite_min=%g finite_max=%g "
        "top=[%d:%g,%d:%g,%d:%g,%d:%g,%d:%g]\n",
        step, logits.size(), selected_token, selected_value,
        finite_count, nan_count, inf_count, finite_min, finite_max,
        top[0].second, top[0].first,
        top[1].second, top[1].first,
        top[2].second, top[2].first,
        top[3].second, top[3].first,
        top[4].second, top[4].first);
}

}  // namespace

bool run_layer_split_ar_decode(
        int last_tok,
        int committed,
        int n_gen,
        int vocab,
        const std::vector<float> & prefill_last_logits,
        const SamplerCfg & sampler,
        std::mt19937_64 & rng,
        const LayerSplitForwardStep & forward_one,
        const std::function<bool(int)> & is_eos,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io,
        bool forward_provides_argmax) {
    if (n_gen <= 0) return true;

    const bool require_logits =
        sampler.needs_logit_processing() || !forward_provides_argmax;

    if (require_logits) {
        if ((int)prefill_last_logits.size() != vocab) return false;
        last_tok = sample_logits(prefill_last_logits.data(), vocab, sampler,
                                 out_tokens, rng);
        log_logits_debug(prefill_last_logits, 0, last_tok);
    }

    out_tokens.push_back(last_tok);
    io.emit(last_tok);
    if (io.cancelled) {
        io.emit(-1);
        return true;
    }
    if (is_eos(last_tok)) {
        io.emit(-1);
        return true;
    }
    ++committed;

    std::vector<float> logits_buf;
    for (int i = 1; i < n_gen; ++i) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        logits_buf.clear();
        if (!forward_one(one, committed, next_tok,
                         require_logits ? &logits_buf : nullptr)) {
            return false;
        }
        if (require_logits) {
            if ((int)logits_buf.size() != vocab) return false;
            next_tok = sample_logits(logits_buf.data(), vocab, sampler,
                                     out_tokens, rng);
            log_logits_debug(logits_buf, i, next_tok);
        }

        last_tok = next_tok;
        out_tokens.push_back(last_tok);
        io.emit(last_tok);
        ++committed;
        if (io.cancelled) break;
        if (is_eos(last_tok)) break;
    }

    io.emit(-1);
    return true;
}

}  // namespace dflash::common
