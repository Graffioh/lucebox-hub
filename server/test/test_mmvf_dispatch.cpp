#include "ggml.h"
#include "mmvf.cuh"

#include <array>
#include <cstdio>
#include <cstring>

namespace {

constexpr int AMD_CC = 0x1000000;

bool expect(bool condition, const char * message) {
    if (condition) return true;
    std::fprintf(stderr, "test_mmvf_dispatch: %s\n", message);
    return false;
}

std::array<size_t, GGML_MAX_DIMS> contiguous_f16_strides(
        const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    std::array<size_t, GGML_MAX_DIMS> nb{};
    nb[0] = sizeof(uint16_t);
    for (size_t i = 1; i < nb.size(); ++i) {
        nb[i] = nb[i - 1] * static_cast<size_t>(ne[i - 1]);
    }
    return nb;
}

} // namespace

int main(int argc, char ** argv) {
    const bool disabled = argc == 2 && std::strcmp(argv[1], "disabled") == 0;
    if (argc > 2 || (argc == 2 && !disabled)) {
        std::fprintf(stderr, "usage: test_mmvf_dispatch [disabled]\n");
        return 2;
    }

    const std::array<int64_t, GGML_MAX_DIMS> hc = {16384, 24, 1, 1};
    const auto hc_nb = contiguous_f16_strides(hc);
    bool ok = true;
    ok &= expect(
        ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1151, hc.data(), hc_nb.data(), 4) ==
            !disabled,
        disabled ? "kill switch did not restore gfx1151 q4 fallback"
                 : "exact gfx1151 q4 HC projection did not select MMVF");

    for (int q = 1; q <= 3; ++q) {
        ok &= expect(
            ggml_cuda_should_use_mmvf(
                GGML_TYPE_F16, AMD_CC + 0x1151, hc.data(), hc_nb.data(), q),
            "gfx1151 narrow generic MMVF admission changed");
    }
    ok &= expect(
        !ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1150, hc.data(), hc_nb.data(), 4),
        "gfx1150 inherited the gfx1151-only HC admission");
    ok &= expect(
        !ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1151, hc.data(), hc_nb.data(), 5),
        "gfx1151 HC q5 was admitted");

    auto near_shape = hc;
    near_shape[1] = 23;
    const auto near_shape_nb = contiguous_f16_strides(near_shape);
    ok &= expect(
        !ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1151, near_shape.data(),
            near_shape_nb.data(), 4),
        "near-miss HC shape was admitted");

    auto higher_rank = hc;
    higher_rank[2] = 2;
    const auto higher_rank_nb = contiguous_f16_strides(higher_rank);
    ok &= expect(
        !ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1151, higher_rank.data(),
            higher_rank_nb.data(), 4),
        "higher-rank HC tensor was admitted");

    auto bad_nb = hc_nb;
    bad_nb[1] += sizeof(uint16_t);
    ok &= expect(
        !ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1151, hc.data(), bad_nb.data(), 4),
        "misaligned HC stride was admitted");

    ok &= expect(
        ggml_cuda_should_use_mmvf(
            GGML_TYPE_F16, AMD_CC + 0x1201, hc.data(), hc_nb.data(), 4),
        "generic RDNA4 q4 admission changed");
    return ok ? 0 : 1;
}
