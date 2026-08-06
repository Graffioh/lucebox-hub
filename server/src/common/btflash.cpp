#include "btflash.h"

#include "attn_masks.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dflash::common {

std::string validate_btflash_config(const BTFlashConfig & config) {
    if (!config.enabled()) return {};
    if (config.k != 2 && config.k != 4) {
        return "btflash.k must be 2 or 4 in the BT1 prototype";
    }
    if (config.horizon != 8 && config.horizon != 16) {
        return "btflash.horizon must be 8 or 16 in the BT1 prototype";
    }
    if (config.row_budget < config.k || config.row_budget > 64) {
        return "btflash.row_budget must be between k and 64";
    }
    if (config.survivors != 1) {
        return "btflash.survivors must be 1 in the BT1 prototype";
    }
    if (config.fork != "fixed") {
        return "btflash.fork must be fixed in the BT1 prototype";
    }
    if (config.select != "logprob") {
        return "btflash.select must be logprob in the BT1 prototype";
    }
    if (config.fork_tokens < 1 || config.fork_tokens > 256) {
        return "btflash.fork_tokens must be between 1 and 256";
    }
    return {};
}

uint64_t btflash_branch_seed(uint64_t fork_seed, int branch_id) {
    uint64_t z = fork_seed + 0x9E3779B97F4A7C15ULL *
        (uint64_t)(branch_id + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double btflash_token_logprob(const float * logits, int vocab, int token) {
    if (!logits || vocab <= 0 || token < 0 || token >= vocab) {
        return -std::numeric_limits<double>::infinity();
    }
    float max_logit = logits[0];
    for (int i = 1; i < vocab; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        sum += std::exp((double)logits[i] - (double)max_logit);
    }
    return (double)logits[token] - (double)max_logit - std::log(sum);
}

int btflash_select_normalized_logprob(const std::vector<double> & logprob_sums,
                                      const std::vector<int> & token_counts) {
    if (logprob_sums.empty() || logprob_sums.size() != token_counts.size()) {
        return -1;
    }
    int best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < logprob_sums.size(); ++i) {
        if (token_counts[i] <= 0) continue;
        const double score = logprob_sums[i] / (double)token_counts[i];
        if (best < 0 || score > best_score) {
            best = (int)i;
            best_score = score;
        }
    }
    return best;
}

void build_btflash_mask(std::vector<uint16_t> & out,
                        int shared_length,
                        int completed_steps,
                        int width,
                        int kv_pad,
                        int q_pad) {
    out.assign((size_t)kv_pad * (size_t)q_pad, F16_NEG_INF);
    if (shared_length < 0 || completed_steps < 0 || width <= 0 ||
        kv_pad <= 0 || q_pad < width) {
        return;
    }
    const int current_start = shared_length + completed_steps * width;
    for (int branch = 0; branch < width; ++branch) {
        uint16_t * row = out.data() + (size_t)branch * (size_t)kv_pad;
        for (int key = 0; key < shared_length && key < kv_pad; ++key) {
            row[key] = F16_ZERO;
        }
        for (int step = 0; step < completed_steps; ++step) {
            const int key = shared_length + step * width + branch;
            if (key < kv_pad) row[key] = F16_ZERO;
        }
        const int self = current_start + branch;
        if (self < kv_pad) row[self] = F16_ZERO;
    }
}

}  // namespace dflash::common
