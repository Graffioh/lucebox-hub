// Shared result of claiming a concurrent sequence slot.

#pragma once

#include <string>

namespace dflash::common {

struct SeqAdmissionResult {
    bool ok = false;
    // Full right now (no free slot / currently-free pool blocks, or another
    // admission still prefilling). Retrying after progress can succeed.
    bool busy = false;
    int slot = -1;
    std::string error;
};

}  // namespace dflash::common
