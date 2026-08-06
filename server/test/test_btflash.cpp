#include "CppUnitTestFramework.hpp"
#include "btflash.h"
#include "attn_masks.h"

#include <cmath>
#include <vector>

using namespace dflash::common;

namespace {
struct BTFlashFixture {};
}

TEST_CASE(BTFlashFixture, bt1_config_accepts_only_prototype_shape) {
    BTFlashConfig disabled;
    CHECK(validate_btflash_config(disabled).empty());

    BTFlashConfig config;
    config.row_budget = 16;
    config.k = 4;
    config.horizon = 16;
    CHECK(validate_btflash_config(config).empty());

    config.k = 8;
    CHECK(!validate_btflash_config(config).empty());
    config.k = 4;
    config.select = "verifier";
    CHECK(!validate_btflash_config(config).empty());
}

TEST_CASE(BTFlashFixture, branch_seeds_are_stable_and_distinct) {
    const uint64_t a = btflash_branch_seed(1234, 0);
    const uint64_t b = btflash_branch_seed(1234, 1);
    CHECK(a == btflash_branch_seed(1234, 0));
    CHECK(a != b);
    CHECK(a != btflash_branch_seed(5678, 0));
}

TEST_CASE(BTFlashFixture, normalized_logprob_selector_is_length_normalized) {
    const std::vector<double> sums = {-4.0, -3.0, -1.0};
    const std::vector<int> counts = {2, 1, 0};
    CHECK(btflash_select_normalized_logprob(sums, counts) == 0);
    CHECK(btflash_select_normalized_logprob({}, {}) == -1);

    const float logits[] = {0.0f, 1.0f, 2.0f};
    const double p2 = btflash_token_logprob(logits, 3, 2);
    const double p0 = btflash_token_logprob(logits, 3, 0);
    CHECK(std::isfinite(p2));
    CHECK(p2 > p0);
}

TEST_CASE(BTFlashFixture, mask_exposes_shared_prefix_and_same_branch_only) {
    constexpr int shared = 3;
    constexpr int width = 2;
    constexpr int completed = 2;
    constexpr int kv_pad = 16;
    constexpr int q_pad = 32;
    std::vector<uint16_t> mask;
    build_btflash_mask(mask, shared, completed, width, kv_pad, q_pad);
    CHECK(mask.size() == (size_t)kv_pad * q_pad);

    for (int branch = 0; branch < width; ++branch) {
        const uint16_t * row = mask.data() + (size_t)branch * kv_pad;
        CHECK(row[0] == F16_ZERO);
        CHECK(row[1] == F16_ZERO);
        CHECK(row[2] == F16_ZERO);
        CHECK(row[shared + branch] == F16_ZERO);
        CHECK(row[shared + width + branch] == F16_ZERO);
        CHECK(row[shared + completed * width + branch] == F16_ZERO);
        CHECK(row[shared + (1 - branch)] == F16_NEG_INF);
        CHECK(row[shared + width + (1 - branch)] == F16_NEG_INF);
        CHECK(row[shared + completed * width + (1 - branch)] == F16_NEG_INF);
    }
    CHECK(mask[(size_t)width * kv_pad] == F16_NEG_INF);
}
