// JPEG and PNG decoding to 8-bit RGB. Model independent: every vision model
// starts from these pixels and does its own resizing and patching.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::vision {

struct EncodedImageView {
    const std::uint8_t * data = nullptr;
    std::size_t size = 0;
};

// Checked before any decoded buffer is allocated.
struct DecodeLimits {
    std::size_t max_encoded_bytes = 16ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_pixels = 64ULL * 1024ULL * 1024ULL;
    std::uint32_t max_dimension = 65'535;
};

// Borrowed row-major RGB, three bytes per pixel.
struct DecodedRgbView {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    const std::uint8_t * data = nullptr;
    std::size_t size = 0;
};

struct DecodedRgb {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;

    DecodedRgbView view() const {
        return {width, height, pixels.data(), pixels.size()};
    }
};

enum class DecodeError {
    None = 0,
    EmptyInput,
    EncodedTooLarge,
    UnsupportedFormat,
    MalformedImage,
    DecodedTooLarge,
    AllocationFailed,
};

struct DecodeStatus {
    DecodeError code = DecodeError::None;
    std::string message;

    explicit operator bool() const { return code == DecodeError::None; }
};

struct DecodeResult {
    DecodeStatus status;
    DecodedRgb image;

    explicit operator bool() const { return static_cast<bool>(status); }
};

DecodeResult decode_image(
    const EncodedImageView & encoded,
    const DecodeLimits & limits = {});

const char * decode_error_name(DecodeError error);

}  // namespace luce::vision
