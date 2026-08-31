#include "ds4_mix_mmv_dispatch.h"

#include <cstdio>

namespace {

bool expect(bool condition, const char * message) {
    if (condition) return true;
    std::fprintf(stderr, "test_ds4_mix_mmv_dispatch: %s\n", message);
    return false;
}

} // namespace

int main() {
    bool ok = true;

    ok &= expect(!ds4_mix_mmv_q6_env_enabled(nullptr),
                 "an unset opt-in was enabled");
    ok &= expect(!ds4_mix_mmv_q6_env_enabled(""),
                 "an empty opt-in was enabled");
    ok &= expect(!ds4_mix_mmv_q6_env_enabled("0"),
                 "value 0 enabled q6");
    ok &= expect(!ds4_mix_mmv_q6_env_enabled("true"),
                 "a non-numeric value enabled q6");
    ok &= expect(!ds4_mix_mmv_q6_env_enabled("01"),
                 "a zero-prefixed value enabled q6");
    ok &= expect(!ds4_mix_mmv_q6_env_enabled("1 "),
                 "a suffixed value enabled q6");
    ok &= expect(ds4_mix_mmv_q6_env_enabled("1"),
                 "the exact opt-in value did not enable q6");

    for (int64_t q = 1; q <= 5; ++q) {
        ok &= expect(ds4_mix_mmv_supported_tokens(q, false),
                     "the qualified q1-q5 range changed");
        ok &= expect(ds4_mix_mmv_supported_tokens(q, true),
                     "the q6 opt-in rejected a narrower width");
    }
    ok &= expect(!ds4_mix_mmv_supported_tokens(0, false),
                 "zero tokens were accepted");
    ok &= expect(!ds4_mix_mmv_supported_tokens(6, false),
                 "q6 was enabled by default");
    ok &= expect(ds4_mix_mmv_supported_tokens(6, true),
                 "the q6 opt-in did not enable q6");
    ok &= expect(!ds4_mix_mmv_supported_tokens(7, true),
                 "the q6 opt-in widened q7");

    return ok ? 0 : 1;
}
