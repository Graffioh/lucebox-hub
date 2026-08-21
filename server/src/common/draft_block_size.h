#pragma once

namespace dflash::common {

// A checkpoint's declared block size is its supported proposal horizon.
// Runtime tuning may shorten that horizon, but widening it would ask the
// drafter to predict positions it was not trained or published to serve.
constexpr bool draft_block_size_override_supported(
        int requested, int checkpoint_block_size) {
    return requested == 0 ||
           (requested >= 2 && checkpoint_block_size >= 2 &&
            requested <= checkpoint_block_size);
}

}  // namespace dflash::common
