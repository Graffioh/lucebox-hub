// SpecLA runtime mode (docs/SPECLA.md, arXiv:2607.16673).
//
// DFLASH_SPECLA=1 switches the single-target qwen35 spec-decode verify path
// to factor-capture verification: capture-mode verify runs the
// topology-masked parallel builder, never mutates the durable SSM/conv
// state, and records compact per-token factors instead of dense per-token
// state checkpoints. Accepted state is reconstructed by DeltaConstruct at
// commit. Non-capture forwards (prefill, AR decode, replay) keep their
// state writebacks, so the legacy restore+replay fallback stays correct.
#pragma once

#include <cstdlib>
#include <cstring>

namespace dflash::common {

inline bool specla_enabled() {
    static const bool on = []() {
        const char * v = std::getenv("DFLASH_SPECLA");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

}  // namespace dflash::common
