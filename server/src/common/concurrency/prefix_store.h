// Model-neutral checkpoint protocol for continuous-batching prefix reuse.
//
// The scheduler owns token-prefix lookup and LRU policy. A SeqEngine receives
// only opaque checkpoint identities and logical token positions. The concrete
// engine owns checkpoint payloads and cache-layout-specific copies. Copied
// pages today and a shared-page/radix engine later use the same protocol.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dflash::common {

struct PrefixStoreRef {
    uint64_t id = 0;
    int tokens = 0;

    bool valid() const { return id != 0 && tokens > 0; }
};

inline bool operator==(PrefixStoreRef a, PrefixStoreRef b) {
    return a.id == b.id && a.tokens == b.tokens;
}

inline bool operator!=(PrefixStoreRef a, PrefixStoreRef b) {
    return !(a == b);
}

struct PrefixCaptureTicket {
    uint64_t id = 0;
    PrefixStoreRef checkpoint;

    bool valid() const { return id != 0 && checkpoint.valid(); }
};

inline bool operator==(const PrefixCaptureTicket & a,
                       const PrefixCaptureTicket & b) {
    return a.id == b.id && a.checkpoint == b.checkpoint;
}

inline bool operator!=(const PrefixCaptureTicket & a,
                       const PrefixCaptureTicket & b) {
    return !(a == b);
}

struct PrefixStorePlan {
    PrefixStoreRef restore;
    PrefixCaptureTicket capture;

    bool empty() const { return !restore.valid() && !capture.valid(); }
};

struct PrefixStoreAdmission {
    PrefixStoreRef restored;
    PrefixStoreRef invalidated;
    PrefixCaptureTicket capture;
    // Wall time spent validating/copying a requested restore. Non-zero for
    // both successful restores and invalidations so operators can see stalls.
    uint64_t restore_elapsed_us = 0;
};

struct PrefixStoreEvent {
    enum class Status {
        none,
        saved,
        failed,
    };

    Status status = Status::none;
    PrefixCaptureTicket ticket;
    std::string error;
    // Actual committed payload size. Present only for `saved`.
    size_t bytes = 0;
    // Wall time spent in the capture attempt, including a failed copy.
    uint64_t elapsed_us = 0;

    bool attempted() const { return status != Status::none; }
};

}  // namespace dflash::common
