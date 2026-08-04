#include "mmq-streamk-schedule.h"

#include <climits>
#include <cstdint>
#include <cstdio>

namespace {
constexpr int kIterK = 256;
constexpr int kNsm86 = 82;
int failures = 0;

void expect(const char * name, int actual, int wanted) {
    const bool pass = actual == wanted;
    std::printf("[%s] %-34s actual=%d expected=%d\n", pass ? "PASS" : "FAIL", name, actual, wanted);
    failures += pass ? 0 : 1;
}
} // namespace

int main() {
    // Validated SM86 path: avoid empty CTAs for shallow K.
    expect("sm86 shallow pure tiling", mmq_stream_k_nblocks(20, 82, 256, kIterK, true, true), 20);
    expect("sm86 partial fill",        mmq_stream_k_nblocks(20, 82, 512, kIterK, true, true), 40);
    expect("sm86 deep K unchanged",    mmq_stream_k_nblocks(20, 82, 5120, kIterK, true, true), 82);
    expect("sm86 single shallow",      mmq_stream_k_nblocks(1, 82, 256, kIterK, true, true), 1);
    expect("sm86 single deep capped",  mmq_stream_k_nblocks(1, 82, 5120, kIterK, true, true), 20);

    // Existing >=90% NVIDIA tiling behavior remains unchanged.
    expect("nvidia 90pct tiling",       mmq_stream_k_nblocks(74, 82, 5120, kIterK, true, false), 74);
    expect("nvidia full SM tiling",     mmq_stream_k_nblocks(82, 82, 5120, kIterK, true, false), 82);

    // Fail closed: non-SM86 and non-NVIDIA paths retain the old nsm schedule.
    expect("other nvidia unchanged",    mmq_stream_k_nblocks(20, 82, 256, kIterK, true, false), 82);
    expect("non-nvidia unchanged",      mmq_stream_k_nblocks(20, 82, 256, kIterK, false, false), 82);

    // Invalid inputs remain bounded, and 64-bit arithmetic avoids overflow.
    expect("invalid ntiles",            mmq_stream_k_nblocks(0, 82, 256, kIterK, true, true), 1);
    expect("invalid nsm",               mmq_stream_k_nblocks(20, 0, 256, kIterK, true, true), 1);
    expect("invalid K",                 mmq_stream_k_nblocks(20, 82, 0, kIterK, true, true), 1);
    expect("invalid iter",              mmq_stream_k_nblocks(20, 82, 256, 0, true, true), 1);
    expect("large shape no overflow",   mmq_stream_k_nblocks(INT_MAX / 2, 82, INT64_MAX / 4, kIterK, true, true), INT_MAX / 2);

    // Fixup predicate itself remains unchanged.
    expect("tiling skips fixup", mmq_stream_k_fixup_needed(20, 20), 0);
    expect("partial needs fixup", mmq_stream_k_fixup_needed(20, 40), 1);
    expect("deep needs fixup",    mmq_stream_k_fixup_needed(20, 82), 1);

    if (failures) {
        std::fprintf(stderr, "FAILED: %d schedule checks\n", failures);
        return 1;
    }
    std::puts("ALL PASS: SM86 Stream-K schedule checks");
    return 0;
}
