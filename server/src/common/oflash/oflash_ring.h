// oflash_ring.h — engine side of the OFlash capture ring (OFLASH.md §3-4).
//
// Single producer (the decode thread) appending records to a POSIX shm
// segment; single consumer (the trainer sidecar) advancing `tail` from
// another process. The producer never blocks and never waits on the
// consumer: when a record does not fit, it is dropped and counted.
//
// The segment is created with shm_open(name) so a trainer can attach by
// name (the supervisor hands the name over on spawn / via the control
// socket). The engine owns the segment lifetime and shm_unlinks it on
// destruction — attached consumers keep their mapping until they close.

#pragma once

#include "oflash_format.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace dflash::common::oflash {

class OFlashRing {
public:
    OFlashRing() = default;
    ~OFlashRing() { close(); }
    OFlashRing(const OFlashRing &) = delete;
    OFlashRing & operator=(const OFlashRing &) = delete;

    // Create /dev/shm/<name> sized to hold `capacity_bytes` of record data
    // plus the header, and fill the immutable stream facts. Returns false
    // on any syscall failure (the runtime then disables capture; serving
    // is never affected).
    bool create(const std::string & name,
                uint64_t capacity_bytes,
                uint64_t drafter_hash,
                uint64_t target_hash,
                uint64_t drafter_semantics_hash,
                uint32_t n_capture_layers,
                uint32_t hidden,
                uint32_t block_size,
                uint32_t topk,
                uint32_t vocab);

    // Append one record. `header.size_bytes` is computed here from
    // `payload_bytes` (rounded up to 8); the caller fills every other
    // header field. Drops (and counts) when the ring lacks space.
    // Returns true when the record was published.
    bool push(OFlashRecordHeader header,
              const void * payload, size_t payload_bytes);

    // Two-part variant so callers can assemble feat + label sections
    // without an intermediate copy of the (large) feature block.
    bool push2(OFlashRecordHeader header,
               const void * payload_a, size_t a_bytes,
               const void * payload_b, size_t b_bytes);

    void close();  // munmap + shm_unlink (owner only)

    bool     active() const   { return hdr_ != nullptr; }
    const std::string & name() const { return name_; }
    uint64_t dropped() const;
    uint64_t written() const  { return records_written_; }
    // Bytes currently unconsumed (approximate; consumer races are fine).
    uint64_t backlog() const;

private:
    bool reserve(size_t bytes, uint64_t & write_at);
    void publish(uint64_t new_head);
    uint8_t * slot(uint64_t logical) const;

    OFlashRingHeader * hdr_ = nullptr;
    uint8_t * data_ = nullptr;
    uint64_t  capacity_ = 0;
    size_t    map_bytes_ = 0;
    std::string name_;
    uint64_t  records_written_ = 0;
};

}  // namespace dflash::common::oflash
