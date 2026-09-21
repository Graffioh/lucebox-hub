// Bicubic resize of 8-bit RGB that reproduces Pillow's Image.resize(BICUBIC)
// byte for byte, including its antialiasing when shrinking. Vision models are
// trained on images resized this way, so every model's preprocessing uses it.
// Reference: Pillow 12.3.0 src/libImaging/Resample.c.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace luce::vision {

// `input` is row-major RGB, three bytes per pixel. Returns false and sets
// `error` on invalid sizes or allocation failure; `output` is then unspecified.
bool resize_rgb_bicubic(const std::vector<std::uint8_t> & input, int input_width, int input_height,
                        int output_width, int output_height,
                        std::vector<std::uint8_t> & output, std::string & error);

}  // namespace luce::vision
