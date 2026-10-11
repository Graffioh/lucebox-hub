#pragma once

// Host copies of drafter scoring sessions (qwen35_drafter.cpp), so a session survives the drafter being released
// (--draft-residency auto: ReleaseAfterUse frees the drafter after every compression).
//
// When the drafter is released, every live session's device tensors (blocks 0..14 cache incl. the recurrent-state
// snapshot, block-15 keys) are copied byte for byte to host RAM next to the session's host bookkeeping. When the
// drafter is loaded again, a session is restored into freshly allocated tensors of the same layout before the next
// scoring call, and resume runs unchanged. A copy is restored only under the identity it was stored with (drafter
// file, build, session parameters) and only if every tensor matches in type, shape, strides and size; anything else
// drops it and the call scores from token 0. Nothing is approximated.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::pflash {

struct PFlashHostTensor {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
    size_t nb[GGML_MAX_DIMS] = {0, 0, 0, 0};
    std::vector<uint8_t> bytes;   // ggml_nbytes(tensor) raw bytes
};

// Every tensor of ``ctx`` that owns device memory (allocated, not a view), in creation order.
std::vector<ggml_tensor *> pflash_owned_tensors(ggml_context * ctx);

// Copies each tensor's bytes to host. Fails (``out`` cleared, reason in ``error``) on a null, unallocated or view tensor.
bool pflash_tensors_to_host(const std::vector<ggml_tensor *> & tensors, std::vector<PFlashHostTensor> & out,
                            std::string & error);

// Writes the host copies back into ``tensors``. Every pair is checked first (count, type, shape, strides, byte size,
// allocated); on any mismatch nothing is written and the reason is in ``error``. Names are kept for the log only: an
// unnamed tensor is renamed by the first graph that reads it (ggml's "leaf_<n>"), so a fresh allocation of the same
// layout does not carry the stored name.
bool pflash_tensors_from_host(const std::vector<PFlashHostTensor> & host, const std::vector<ggml_tensor *> & tensors,
                              std::string & error);

size_t pflash_host_tensor_bytes(const std::vector<PFlashHostTensor> & host);

// "<path>|dev:ino|size|mtime_ns" for an existing file, "<path>|missing" otherwise, "-" for an empty path: a file
// identity cheap enough to take at every load (the drafter's sha256 is not always computed).
std::string pflash_file_identity(const std::string & path);

// Why a stored identity does not match the current one: "" when equal, else the first differing "key=value" field
// (identities are "key=value" lines).
std::string pflash_identity_mismatch(const std::string & stored, const std::string & current);

} // namespace luce::pflash
