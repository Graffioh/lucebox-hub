#include "image_resize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace luce::vision {
namespace {

struct Status {
    bool good = true;
    std::string message;
    explicit operator bool() const { return good; }
};
// Pillow's fixed-point precision for 8-bit resampling.
constexpr int kPrecisionBits = 22;

Status ok() { return {}; }
Status fail(std::string message) { return {false, std::move(message)}; }

bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

struct Coefficients {
    int kernel_size = 0;
    std::vector<int> bounds;
    std::vector<std::int32_t> weights;
};

// Pillow 12.3.0 Resample.c reference used for byte parity:
// https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

Status precompute_coefficients(int input_size, int output_size, Coefficients & out) {
    if (input_size <= 0 || output_size <= 0) {
        return fail("resample dimensions must be positive");
    }
    const float input_begin = 0.0F;
    const float input_end = static_cast<float>(input_size);
    double filter_scale =
        (static_cast<double>(input_end) - static_cast<double>(input_begin)) / output_size;
    const double scale = filter_scale;
    if (filter_scale < 1.0) {
        filter_scale = 1.0;
    }
    const double support = 2.0 * filter_scale;
    const int kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::uint64_t coefficient_count = 0;
    if (!checked_mul(static_cast<std::uint64_t>(output_size),
                     static_cast<std::uint64_t>(kernel_size), coefficient_count) ||
        coefficient_count > std::numeric_limits<std::size_t>::max()) {
        return fail("resample coefficient count overflow");
    }

    std::vector<double> floating(static_cast<std::size_t>(coefficient_count), 0.0);
    out.bounds.resize(static_cast<std::size_t>(output_size) * 2);
    out.weights.resize(static_cast<std::size_t>(coefficient_count));
    const double inverse_filter_scale = 1.0 / filter_scale;
    for (int output = 0; output < output_size; ++output) {
        const double center = input_begin + (output + 0.5) * scale;
        int first = static_cast<int>(center - support + 0.5);
        if (first < 0) {
            first = 0;
        }
        int count = static_cast<int>(center + support + 0.5);
        if (count > input_size) {
            count = input_size;
        }
        count -= first;
        double sum = 0.0;
        const std::size_t base = static_cast<std::size_t>(output) * kernel_size;
        for (int index = 0; index < count; ++index) {
            const double weight = bicubic(
                (index + first - center + 0.5) * inverse_filter_scale);
            floating[base + index] = weight;
            sum += weight;
        }
        if (sum != 0.0) {
            for (int index = 0; index < count; ++index) {
                floating[base + index] /= sum;
            }
        }
        out.bounds[static_cast<std::size_t>(output) * 2] = first;
        out.bounds[static_cast<std::size_t>(output) * 2 + 1] = count;
    }

    constexpr double scale_to_fixed = static_cast<double>(1U << kPrecisionBits);
    for (std::size_t index = 0; index < floating.size(); ++index) {
        const double value = floating[index] * scale_to_fixed;
        out.weights[index] = static_cast<std::int32_t>(
            value < 0.0 ? value - 0.5 : value + 0.5);
    }
    out.kernel_size = kernel_size;
    return ok();
}

std::uint8_t clip_fixed(std::int32_t value) {
    const std::int32_t rounded = value >> kPrecisionBits;
    return static_cast<std::uint8_t>(std::clamp<std::int32_t>(rounded, 0, 255));
}

Status resize_horizontal(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_width,
    std::vector<std::uint8_t> & output) {
    Coefficients coeffs;
    if (const auto status = precompute_coefficients(input_width, output_width, coeffs); !status) {
        return status;
    }
    output.resize(static_cast<std::size_t>(output_width) * input_height * 3);
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const int first = coeffs.bounds[static_cast<std::size_t>(x) * 2];
            const int count = coeffs.bounds[static_cast<std::size_t>(x) * 2 + 1];
            const std::int32_t * weights =
                coeffs.weights.data() + static_cast<std::size_t>(x) * coeffs.kernel_size;
            for (int channel = 0; channel < 3; ++channel) {
                std::int32_t sum = 1 << (kPrecisionBits - 1);
                for (int index = 0; index < count; ++index) {
                    const std::size_t source =
                        (static_cast<std::size_t>(y) * input_width + first + index) * 3 + channel;
                    sum += static_cast<std::int32_t>(input[source]) * weights[index];
                }
                const std::size_t destination =
                    (static_cast<std::size_t>(y) * output_width + x) * 3 + channel;
                output[destination] = clip_fixed(sum);
            }
        }
    }
    return ok();
}

Status resize_vertical(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_height,
    std::vector<std::uint8_t> & output) {
    Coefficients coeffs;
    if (const auto status = precompute_coefficients(input_height, output_height, coeffs); !status) {
        return status;
    }
    output.resize(static_cast<std::size_t>(input_width) * output_height * 3);
    for (int y = 0; y < output_height; ++y) {
        const int first = coeffs.bounds[static_cast<std::size_t>(y) * 2];
        const int count = coeffs.bounds[static_cast<std::size_t>(y) * 2 + 1];
        const std::int32_t * weights =
            coeffs.weights.data() + static_cast<std::size_t>(y) * coeffs.kernel_size;
        for (int x = 0; x < input_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                std::int32_t sum = 1 << (kPrecisionBits - 1);
                for (int index = 0; index < count; ++index) {
                    const std::size_t source =
                        (static_cast<std::size_t>(first + index) * input_width + x) * 3 + channel;
                    sum += static_cast<std::int32_t>(input[source]) * weights[index];
                }
                const std::size_t destination =
                    (static_cast<std::size_t>(y) * input_width + x) * 3 + channel;
                output[destination] = clip_fixed(sum);
            }
        }
    }
    return ok();
}

Status resize_impl(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    std::vector<std::uint8_t> & output) {
    if (input_width == output_width && input_height == output_height) {
        output = input;
        return ok();
    }

    std::vector<std::uint8_t> intermediate;
    const bool vertical_first =
        static_cast<std::uint64_t>(input_height) >
            static_cast<std::uint64_t>(input_width) * 100 &&
        output_height < input_height;
    if (vertical_first) {
        if (const auto status = resize_vertical(
                input, input_width, input_height, output_height, intermediate);
            !status) {
            return status;
        }
        if (output_width == input_width) {
            output = std::move(intermediate);
            return ok();
        }
        return resize_horizontal(
            intermediate, input_width, output_height, output_width, output);
    }

    const std::vector<std::uint8_t> * vertical_input = &input;
    int vertical_width = input_width;
    if (output_width != input_width) {
        if (const auto status = resize_horizontal(
                input, input_width, input_height, output_width, intermediate);
            !status) {
            return status;
        }
        vertical_input = &intermediate;
        vertical_width = output_width;
    }
    if (output_height != input_height) {
        return resize_vertical(
            *vertical_input, vertical_width, input_height, output_height, output);
    }
    output = *vertical_input;
    return ok();
}

}  // namespace

bool resize_rgb_bicubic(const std::vector<std::uint8_t> & input, int input_width, int input_height,
                        int output_width, int output_height,
                        std::vector<std::uint8_t> & output, std::string & error) {
    error.clear();
    if (input_width <= 0 || input_height <= 0 || output_width <= 0 || output_height <= 0 ||
        input.size() != static_cast<std::size_t>(input_width) * input_height * 3) {
        error = "resize needs positive sizes and width*height*3 input bytes";
        return false;
    }
    try {
        const Status status = resize_impl(input, input_width, input_height, output_width, output_height, output);
        if (!status) error = status.message;
        return bool(status);
    } catch (const std::bad_alloc &) {
        error = "resize allocation failed";
        return false;
    }
}

}  // namespace luce::vision
