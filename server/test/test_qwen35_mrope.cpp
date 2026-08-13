#include "CppUnitTestFramework.hpp"

#include "qwen35/qwen35_mrope.h"

#include <cstdint>
#include <vector>

namespace {

using namespace dflash::common;

struct Qwen35MropeFixture {};

TEST_CASE(Qwen35MropeFixture, text_positions_are_axis_major) {
    const std::vector<int32_t> got = qwen35_text_mrope_positions(7, 3);
    const std::vector<int32_t> want = {
        7, 8, 9,
        7, 8, 9,
        7, 8, 9,
        0, 0, 0,
    };
    CHECK(got == want);
}

TEST_CASE(Qwen35MropeFixture, one_token_layout_and_empty_input) {
    CHECK(qwen35_text_mrope_positions(11, 1) ==
          std::vector<int32_t>({11, 11, 11, 0}));
    CHECK(qwen35_text_mrope_positions(11, 0).empty());
}

}  // namespace
