// oflash_ring.cpp — SPSC shared-memory capture ring (engine producer).

#include "oflash_ring.h"

#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common::oflash {

namespace {

// The header lives in shared memory; cursor accesses must be atomic with
// the right ordering. std::atomic_ref is C++20, so wrap the primitives.
inline uint64_t load_acquire(const uint64_t * p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
inline void store_release(uint64_t * p, uint64_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
inline void add_relaxed(uint64_t * p, uint64_t v) {
    __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

constexpr size_t align8(size_t n) { return (n + 7) & ~(size_t)7; }

}  // namespace

bool OFlashRing::create(const std::string & name,
                        uint64_t capacity_bytes,
                        uint64_t drafter_hash,
                        uint32_t n_capture_layers,
                        uint32_t hidden,
                        uint32_t block_size,
                        uint32_t topk,
                        uint32_t vocab) {
#if defined(_WIN32)
    (void)name; (void)capacity_bytes; (void)drafter_hash;
    (void)n_capture_layers; (void)hidden; (void)block_size;
    (void)topk; (void)vocab;
    return false;
#else
    close();
    capacity_bytes = align8(capacity_bytes);
    if (capacity_bytes < (uint64_t)1 << 20) {
        std::fprintf(stderr, "[oflash] ring capacity too small (%llu)\n",
                     (unsigned long long)capacity_bytes);
        return false;
    }
    // A stale segment from a crashed run may linger; recreate cleanly.
    ::shm_unlink(name.c_str());
    int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        std::perror("[oflash] shm_open");
        return false;
    }
    const size_t total = sizeof(OFlashRingHeader) + (size_t)capacity_bytes;
    if (::ftruncate(fd, (off_t)total) != 0) {
        std::perror("[oflash] ftruncate");
        ::close(fd);
        ::shm_unlink(name.c_str());
        return false;
    }
    void * mapped = ::mmap(nullptr, total, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        std::perror("[oflash] mmap");
        ::shm_unlink(name.c_str());
        return false;
    }

    hdr_       = static_cast<OFlashRingHeader *>(mapped);
    data_      = reinterpret_cast<uint8_t *>(mapped) + sizeof(OFlashRingHeader);
    capacity_  = capacity_bytes;
    map_bytes_ = total;
    name_      = name;

    std::memset(hdr_, 0, sizeof(*hdr_));
    hdr_->capacity         = capacity_bytes;
    hdr_->data_offset      = sizeof(OFlashRingHeader);
    hdr_->drafter_hash     = drafter_hash;
    hdr_->n_capture_layers = n_capture_layers;
    hdr_->hidden           = hidden;
    hdr_->block_size       = block_size;
    hdr_->topk             = topk;
    hdr_->vocab            = vocab;
    hdr_->version          = OFLASH_RING_VERSION;
    // Magic last: a consumer that maps mid-init sees magic==0 and waits.
    __atomic_store_n(&hdr_->magic, OFLASH_RING_MAGIC, __ATOMIC_RELEASE);
    return true;
#endif
}

void OFlashRing::close() {
#if !defined(_WIN32)
    if (hdr_) {
        ::munmap(hdr_, map_bytes_);
        ::shm_unlink(name_.c_str());
    }
#endif
    hdr_ = nullptr;
    data_ = nullptr;
    capacity_ = 0;
    map_bytes_ = 0;
    name_.clear();
}

uint8_t * OFlashRing::slot(uint64_t logical) const {
    return data_ + (logical % capacity_);
}

// Reserve `bytes` of contiguous space. On success `write_at` is the
// logical offset to write at; a PAD record may have been emitted to skip
// a too-small buffer tail. Fails (drop) when the consumer is too far
// behind.
bool OFlashRing::reserve(size_t bytes, uint64_t & write_at) {
    const uint64_t head = hdr_->head;  // producer-owned, plain load
    const uint64_t tail = load_acquire(&hdr_->tail);
    const uint64_t used = head - tail;
    uint64_t avail = capacity_ - used;

    const uint64_t to_end = capacity_ - (head % capacity_);
    uint64_t pad = 0;
    if (bytes > to_end) {
        pad = to_end;  // wrap: dead space needs a PAD record
    }
    if (pad + bytes > avail) return false;

    if (pad) {
        // The buffer tail is a multiple of 8 but can be SMALLER than a full
        // record header; a PAD needs only its first 8 bytes (type + size)
        // to be skippable, so clamp the write to the space that exists.
        OFlashRecordHeader ph{};
        ph.type = OFLASH_REC_PAD;
        ph.size_bytes = (uint32_t)pad;
        std::memcpy(slot(head), &ph,
                    pad < sizeof(ph) ? (size_t)pad : sizeof(ph));
        // Not published separately — the consumer sees PAD and the next
        // record together once the real record's head store lands.
    }
    write_at = head + pad;
    return true;
}

void OFlashRing::publish(uint64_t new_head) {
    store_release(&hdr_->head, new_head);
}

bool OFlashRing::push(OFlashRecordHeader header,
                      const void * payload, size_t payload_bytes) {
    return push2(header, payload, payload_bytes, nullptr, 0);
}

bool OFlashRing::push2(OFlashRecordHeader header,
                       const void * payload_a, size_t a_bytes,
                       const void * payload_b, size_t b_bytes) {
    if (!hdr_) return false;
    const size_t need = align8(sizeof(header) + a_bytes + b_bytes);
    // A record larger than half the ring is a config error, not load.
    if (need > capacity_ / 2) {
        add_relaxed(&hdr_->dropped_records, 1);
        add_relaxed(&hdr_->dropped_bytes, need);
        return false;
    }
    uint64_t at = 0;
    if (!reserve(need, at)) {
        add_relaxed(&hdr_->dropped_records, 1);
        add_relaxed(&hdr_->dropped_bytes, need);
        return false;
    }
    header.size_bytes = (uint32_t)need;
    uint8_t * dst = slot(at);
    std::memcpy(dst, &header, sizeof(header));
    if (a_bytes) std::memcpy(dst + sizeof(header), payload_a, a_bytes);
    if (b_bytes) std::memcpy(dst + sizeof(header) + a_bytes, payload_b, b_bytes);
    const size_t tail_pad = need - sizeof(header) - a_bytes - b_bytes;
    if (tail_pad) std::memset(dst + need - tail_pad, 0, tail_pad);
    publish(at + need);
    records_written_++;
    return true;
}

uint64_t OFlashRing::dropped() const {
    return hdr_ ? __atomic_load_n(&hdr_->dropped_records, __ATOMIC_RELAXED) : 0;
}

uint64_t OFlashRing::backlog() const {
    if (!hdr_) return 0;
    return hdr_->head - load_acquire(&hdr_->tail);
}

}  // namespace dflash::common::oflash
