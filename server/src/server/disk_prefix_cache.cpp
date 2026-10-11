// Disk-backed prefix cache implementation.

#include "disk_prefix_cache.h"
#include "common/sha1.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <system_error>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace luce::common {

namespace fs = std::filesystem;

// ─── Utility ────────────────────────────────────────────────────────────

static std::string hex(const uint8_t * data, int len) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (int i = 0; i < len; ++i) {
        out.push_back(hex_chars[data[i] >> 4]);
        out.push_back(hex_chars[data[i] & 0x0f]);
    }
    return out;
}

static bool mkdir_p(const std::string & path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) return true;
    fs::create_directories(path, ec);
    return fs::is_directory(path, ec);
}

static uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
}

const char * disk_prefix_cache_mode_name(DiskPrefixCacheMode mode) {
    switch (mode) {
        case DiskPrefixCacheMode::Off:   return "off";
        case DiskPrefixCacheMode::Full:  return "full";
        case DiskPrefixCacheMode::Auto:  return "auto";
        case DiskPrefixCacheMode::Fixed: return "fixed";
    }
    return "full";
}

std::string disk_prefix_cache_policy_name(const DiskPrefixCachePolicy & policy) {
    std::string base;
    if (policy.mode == DiskPrefixCacheMode::Fixed) {
        base = "fixed:" + std::to_string(policy.fixed_tokens);
    } else if (policy.mode == DiskPrefixCacheMode::Auto) {
        base = "auto:" + std::to_string(policy.auto_window);
    } else {
        base = disk_prefix_cache_mode_name(policy.mode);
    }
    if (policy.compress) base += "+compress";
    return base;
}

bool parse_disk_prefix_cache_policy(const std::string & value,
                                    DiskPrefixCachePolicy & out) {
    std::string v;
    v.reserve(value.size());
    for (char c : value) v.push_back((char)std::tolower((unsigned char)c));

    if (v == "off" || v == "none" || v == "disabled") {
        out = {};
        out.mode = DiskPrefixCacheMode::Off;
        return true;
    }
    if (v == "full" || v == "full-prefix") {
        out = {};
        out.mode = DiskPrefixCacheMode::Full;
        return true;
    }
    if (v == "auto") {
        out = {};
        out.mode = DiskPrefixCacheMode::Auto;
        return true;
    }

    const std::string auto_prefix = "auto:";
    if (v.rfind(auto_prefix, 0) == 0) {
        char * end = nullptr;
        long n = std::strtol(v.c_str() + auto_prefix.size(), &end, 10);
        if (!end || *end != '\0' || n <= 0 || n > 1000000) return false;
        out = {};
        out.mode = DiskPrefixCacheMode::Auto;
        out.auto_window = (int)n;
        return true;
    }

    char * end = nullptr;
    long n = std::strtol(v.c_str(), &end, 10);
    if (end && *end == '\0' && n > 0 && n <= 1000000) {
        out = {};
        out.mode = DiskPrefixCacheMode::Fixed;
        out.fixed_tokens = (int)n;
        return true;
    }
    return false;
}

bool apply_request_scope_override(DiskPrefixCachePolicy & server_policy,
                                  const std::string & scope_str) {
    DiskPrefixCachePolicy parsed;
    if (!parse_disk_prefix_cache_policy(scope_str, parsed)) {
        return false;
    }
    // Preserve server-level flags (e.g. compress) across the scope override.
    parsed.compress = server_policy.compress;
    server_policy = parsed;
    return true;
}

static bool valid_boundary(int n, int full_len) {
    return n > 0 && n <= full_len;
}

std::vector<int> disk_prefix_cache_full_lookup_lengths(
        int prompt_len,
        const std::vector<int> & boundaries,
        int min_tokens) {
    std::vector<int> out;
    // Nothing shorter than the persistence minimum is ever written, so such
    // lengths are not worth a probe.
    if (prompt_len >= min_tokens) out.push_back(prompt_len);
    for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
        if (*it >= min_tokens && *it < prompt_len) out.push_back(*it);
    }
    return out;
}

int disk_prefix_cache_fixed_boundary(const DiskPrefixCachePolicy & policy,
                                     int full_len,
                                     int min_tokens) {
    if (policy.mode != DiskPrefixCacheMode::Fixed) return 0;
    if (policy.fixed_tokens < min_tokens) return 0;
    return valid_boundary(policy.fixed_tokens, full_len) ? policy.fixed_tokens : 0;
}

static int lcp_len(const std::vector<int32_t> & a,
                   const std::vector<int32_t> & b) {
    const int n = std::min((int)a.size(), (int)b.size());
    int i = 0;
    while (i < n && a[(size_t)i] == b[(size_t)i]) i++;
    return i;
}

static int floor_to_safe_boundary(int n, const std::vector<int> & safe_boundaries) {
    if (n <= 0) return 0;
    if (safe_boundaries.empty()) return n;

    int best = 0;
    for (int b : safe_boundaries) {
        if (b > 0 && b <= n) best = std::max(best, b);
    }
    return best;
}

int disk_prefix_cache_auto_boundary(
    const std::vector<int32_t> & prompt_ids,
    const std::vector<std::vector<int32_t>> & recent_prompts,
    int window,
    const std::vector<int> & safe_boundaries,
    int min_tokens) {
    if (prompt_ids.empty() || recent_prompts.empty() || window <= 0) return 0;

    const int n_recent = std::min(window, (int)recent_prompts.size());
    int common = 0;
    for (int i = 0; i < n_recent; ++i) {
        common = std::max(common, lcp_len(prompt_ids, recent_prompts[(size_t)i]));
    }
    common = floor_to_safe_boundary(common, safe_boundaries);
    return common >= min_tokens ? common : 0;
}

// Little-endian I/O helpers.
static void write_u32(FILE * f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void write_u64(FILE * f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static void write_i64(FILE * f, int64_t v)  { std::fwrite(&v, 8, 1, f); }
static void write_u16(FILE * f, uint16_t v) { std::fwrite(&v, 2, 1, f); }

static bool read_u32(FILE * f, uint32_t & out) { return std::fread(&out, 4, 1, f) == 1; }
static bool read_u64(FILE * f, uint64_t & out) { return std::fread(&out, 8, 1, f) == 1; }
static bool read_i64(FILE * f, int64_t & out)  { return std::fread(&out, 8, 1, f) == 1; }
static bool read_u16(FILE * f, uint16_t & out) { return std::fread(&out, 2, 1, f) == 1; }

// 64-bit safe absolute seek.
static bool seek_to(FILE * f, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, (__int64)offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)offset, SEEK_SET) == 0;
#endif
}

// Skip the tensor table and key tokens of a file positioned after its
// header, then read the auxiliary blob.
static bool read_aux_section(FILE * f, uint32_t n_tensors, uint32_t token_count,
                             std::string & aux) {
    for (uint32_t i = 0; i < n_tensors; ++i) {
        uint16_t name_len = 0;
        if (!read_u16(f, name_len) ||
            std::fseek(f, (long)name_len + 4 + 32 + 8, SEEK_CUR) != 0) {
            return false;
        }
    }
    // Key tokens: up to 4 * 2^32 bytes, so step in bounded seeks.
    uint64_t skip = 4ull * token_count;
    while (skip > 0) {
        const long step = (long)std::min<uint64_t>(skip, 1u << 30);
        if (std::fseek(f, step, SEEK_CUR) != 0) return false;
        skip -= (uint64_t)step;
    }
    uint32_t len = 0;
    if (!read_u32(f, len) || len > (64u << 20)) return false;
    aux.resize(len);
    return len == 0 || std::fread(aux.data(), 1, len, f) == len;
}

// Flush a written file to stable storage.
static bool sync_file(FILE * f) {
#if defined(_WIN32)
    return _commit(_fileno(f)) == 0;
#else
    return ::fdatasync(fileno(f)) == 0;
#endif
}

// Make a rename inside `dir` durable (POSIX; a no-op elsewhere).
static void sync_directory(const std::string & dir) {
#if !defined(_WIN32)
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        (void)::fsync(fd);
        ::close(fd);
    }
#else
    (void)dir;
#endif
}

// ─── Construction ───────────────────────────────────────────────────────

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend)
    : config_(cfg), backend_(backend) {}

uint64_t DiskPrefixCache::now() const {
    return clock_ ? clock_() : now_unix();
}

static bool has_suffix(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Validate a snapshot file without reading tensor data: header, version,
// tensor table, payload sum and exact file size. A file cut short by a crash
// or a full disk, or written by an older format, fails here.
static bool validate_snapshot_file(const std::string & path,
                                   DiskCacheHeader & hdr,
                                   uint64_t & file_size,
                                   const char *& why) {
    std::error_code ec;
    file_size = (uint64_t) fs::file_size(path, ec);
    if (ec) { why = "unreadable size"; return false; }
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { why = "unreadable"; return false; }
    bool ok = false;
    do {
        DiskCacheHeader h{};
        if (std::fread(h.magic, 1, 4, f) != 4 ||
            std::memcmp(h.magic, "DKVC", 4) != 0) { why = "bad magic"; break; }
        if (!read_u32(f, h.version) || h.version != DISK_CACHE_VERSION) {
            why = "old or unknown version"; break;
        }
        if (std::fseek(f, 0, SEEK_SET) != 0) { why = "seek"; break; }
        // Re-read through the regular header parser.
        uint8_t raw[DISK_CACHE_HEADER_SIZE];
        if (std::fread(raw, 1, sizeof(raw), f) != sizeof(raw)) {
            why = "short header"; break;
        }
        std::memcpy(&h.cur_pos,       raw + 24, 4);
        std::memcpy(&h.n_tensors,     raw + 28, 4);
        std::memcpy(&h.token_count,   raw + 32, 4);
        std::memcpy(h.layout_id,      raw + 8, 16);
        std::memcpy(h.token_hash,     raw + 36, 16);
        std::memcpy(&h.payload_bytes, raw + 52, 8);
        std::memcpy(&h.created_at,    raw + 60, 8);
        std::memcpy(&h.last_used,     raw + DISK_CACHE_LAST_USED_OFFSET, 8);
        std::memcpy(&h.last_tok,      raw + 76, 4);
        std::memcpy(h.identity_salt,  raw + 80, 16);
        if (h.cur_pos == 0 || h.cur_pos > h.token_count) {
            why = "snapshot past key"; break;
        }
        if (h.n_tensors == 0 || h.n_tensors > 4096) { why = "tensor count"; break; }
        uint64_t table_bytes = 0;
        uint64_t payload = 0;
        bool table_ok = true;
        for (uint32_t i = 0; i < h.n_tensors && table_ok; ++i) {
            uint16_t name_len = 0;
            uint32_t type = 0;
            int64_t ne = 0;
            uint64_t nbytes = 0;
            table_ok = read_u16(f, name_len) && name_len < GGML_MAX_NAME &&
                std::fseek(f, name_len, SEEK_CUR) == 0 && read_u32(f, type) &&
                type < GGML_TYPE_COUNT;
            for (int d = 0; d < 4 && table_ok; ++d) table_ok = read_i64(f, ne) && ne >= 0;
            table_ok = table_ok && read_u64(f, nbytes);
            payload += nbytes;
            table_bytes += 2 + name_len + 4 + 32 + 8;
        }
        if (!table_ok) { why = "bad tensor table"; break; }
        if (payload != h.payload_bytes) { why = "payload mismatch"; break; }
        uint32_t aux_len = 0;
        const uint64_t aux_at = DISK_CACHE_HEADER_SIZE + table_bytes +
            4ull * h.token_count;
        if (aux_at + 4 > file_size ||
            !seek_to(f, aux_at) || !read_u32(f, aux_len)) {
            why = "truncated metadata"; break;
        }
        const uint64_t expected = aux_at + 4 + aux_len + payload;
        if (file_size != expected) { why = "truncated or oversized"; break; }
        hdr = h;
        ok = true;
    } while (false);
    std::fclose(f);
    return ok;
}

// ─── Initialization ─────────────────────────────────────────────────────

bool DiskPrefixCache::init() {
    if (disabled()) return true;

    if (!mkdir_p(config_.cache_dir)) {
        std::fprintf(stderr, "[disk-cache] failed to create dir: %s\n",
                     config_.cache_dir.c_str());
        return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    scan_all();
    // Budget and TTL hold from the first moment, before any request.
    expire_locked();
    make_room_locked(0);

    // Restore into the layout this identity used most recently; learn_layout
    // confirms it against the live model before anything is written.
    const DiskEntry * best = nullptr;
    for (const auto & e : entries_) {
        if (e.own_salt && (!best || e.last_used > best->last_used)) best = &e;
    }
    if (best) {
        layout_id_ = best->layout_id;
        layout_known_ = true;
        layout_from_disk_ = true;
        layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    }

    std::fprintf(stderr,
                 "[disk-cache] initialized dir=%s budget=%.1f GB ttl=%s "
                 "files=%zu used=%.1f MB layout=%s\n",
                 config_.cache_dir.c_str(),
                 (double)config_.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                 config_.ttl_seconds
                     ? (std::to_string(config_.ttl_seconds) + "s").c_str()
                     : "never",
                 entries_.size(), (double)total_bytes_ / (1024.0 * 1024.0),
                 layout_known_ ? hex(layout_id_.data(), 16).c_str() : "pending");
    return true;
}

// ─── Layout fingerprint ─────────────────────────────────────────────────

void DiskPrefixCache::compute_layout_id(ggml_context * ctx) {
    // Collect tensor metadata sorted by name for deterministic fingerprint.
    struct TInfo { std::string name; uint32_t type; int64_t ne[4]; };
    std::vector<TInfo> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        TInfo ti;
        ti.name = t->name;
        ti.type = (uint32_t)t->type;
        ti.ne[0] = t->ne[0];
        ti.ne[1] = 1;  // normalize sequence-length dimension
        ti.ne[2] = t->ne[2];
        ti.ne[3] = t->ne[3];
        tensors.push_back(std::move(ti));
    }
    std::sort(tensors.begin(), tensors.end(), [](const TInfo & a, const TInfo & b) {
        return a.name < b.name;
    });

    // Build a single buffer and hash it.
    // Prepend identity_salt_ so that config/model differences (model file,
    // KV types, max_ctx, chat_template) rotate the layout_id independently of
    // tensor structure. All-zero salt (the default) is back-compatible.
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), identity_salt_.begin(), identity_salt_.end());
    for (const auto & ti : tensors) {
        buf.insert(buf.end(), ti.name.begin(), ti.name.end());
        buf.insert(buf.end(), (uint8_t *)&ti.type, (uint8_t *)&ti.type + 4);
        buf.insert(buf.end(), (uint8_t *)ti.ne, (uint8_t *)ti.ne + 32);
    }

    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::memcpy(layout_id_.data(), digest, 16);
}

void DiskPrefixCache::learn_layout(int slot) {
    if (disabled()) return;
    if (layout_known_ && !layout_from_disk_) return;  // already verified from live model

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) {
        if (backend_.snapshot_used(slot)) {
            backend_snapshot_unsupported_ = true;
            std::fprintf(stderr,
                         "[disk-cache] disabled: backend snapshots are "
                         "memory-only\n");
        }
        return;
    }

    std::array<uint8_t, 16> prev_id = layout_id_;
    bool had_disk_layout = layout_from_disk_;

    compute_layout_id(ref.ctx);

    if (had_disk_layout && std::memcmp(prev_id.data(), layout_id_.data(), 16) != 0) {
        // The files stay indexed (budget/TTL) but are never restored.
        std::fprintf(stderr, "[disk-cache] layout mismatch: disk=%s model=%s — switching\n",
                     hex(prev_id.data(), 16).c_str(),
                     hex(layout_id_.data(), 16).c_str());
    }

    layout_known_ = true;
    layout_from_disk_ = false;
    layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    mkdir_p(layout_dir_);

    std::fprintf(stderr, "[disk-cache] layout learned: %s\n",
                 hex(layout_id_.data(), 16).c_str());
}

// ─── Directory scanning ─────────────────────────────────────────────────

void DiskPrefixCache::scan_all() {
    entries_.clear();
    total_bytes_ = 0;
    size_t removed = 0;

    std::error_code ec;
    for (const auto & de : fs::directory_iterator(config_.cache_dir, ec)) {
        const std::string base = de.path().filename().string();
        std::error_code dec;
        if (base.empty() || base[0] == '.' || !de.is_directory(dec)) continue;
        bool kept_any = false;
        std::error_code sec;
        for (const auto & se : fs::directory_iterator(de.path(), sec)) {
            const std::string name = se.path().filename().string();
            const std::string path = se.path().string();
            std::error_code rec;
            if (has_suffix(name, ".dkv.tmp")) {
                // A write the previous process never renamed into place.
                fs::remove(path, rec);
                ++removed;
                continue;
            }
            if (!has_suffix(name, ".dkv")) continue;

            DiskCacheHeader hdr{};
            uint64_t size = 0;
            const char * why = "";
            bool ok = validate_snapshot_file(path, hdr, size, why);
            if (ok && (hex(hdr.layout_id, 16) != base ||
                       hex(hdr.token_hash, 16) + ".dkv" != name)) {
                ok = false;
                why = "name does not match header";
            }
            if (!ok) {
                std::fprintf(stderr, "[disk-cache] removing %s: %s\n",
                             path.c_str(), why);
                fs::remove(path, rec);
                ++removed;
                continue;
            }

            DiskEntry entry;
            entry.path = path;
            std::memcpy(entry.token_hash.data(), hdr.token_hash, 16);
            std::memcpy(entry.layout_id.data(), hdr.layout_id, 16);
            entry.token_count = hdr.token_count;
            entry.cur_pos     = hdr.cur_pos;
            entry.last_used   = hdr.last_used;
            entry.file_size   = size;
            entry.own_salt    = std::memcmp(hdr.identity_salt,
                                            identity_salt_.data(), 16) == 0;
            total_bytes_ += entry.file_size;
            entries_.push_back(std::move(entry));
            kept_any = true;
        }
        std::error_code rec;
        if (!kept_any) fs::remove(de.path(), rec);  // only removes if empty
    }

    std::fprintf(stderr, "[disk-cache] scanned %zu files, %.1f MB (%zu removed)\n",
                 entries_.size(), (double)total_bytes_ / (1024.0 * 1024.0),
                 removed);
}

// ─── Index maintenance ──────────────────────────────────────────────────

bool DiskPrefixCache::expired(const DiskEntry & e, uint64_t t) const {
    return config_.ttl_seconds > 0 && t > e.last_used &&
           t - e.last_used > config_.ttl_seconds;
}

void DiskPrefixCache::remove_entry(size_t idx, const char * why) {
    const DiskEntry & e = entries_[idx];
    std::fprintf(stderr, "[disk-cache] %s: removing %s (%u tokens, %.1f MB)\n",
                 why, e.path.c_str(), e.token_count,
                 (double)e.file_size / (1024.0 * 1024.0));
    std::error_code ec;
    fs::remove(e.path, ec);
    total_bytes_ -= std::min<size_t>(total_bytes_, e.file_size);
    entries_.erase(entries_.begin() + (long)idx);
}

void DiskPrefixCache::expire_locked() {
    const uint64_t t = now();
    for (size_t i = entries_.size(); i-- > 0;) {
        if (expired(entries_[i], t)) remove_entry(i, "ttl expired");
    }
}

void DiskPrefixCache::make_room_locked(uint64_t incoming) {
    while (!entries_.empty() &&
           (uint64_t)total_bytes_ + incoming > (uint64_t)config_.budget_bytes) {
        size_t lru = 0;
        for (size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].last_used < entries_[lru].last_used) lru = i;
        }
        remove_entry(lru, "budget (lru)");
    }
}

// ─── Lookup ─────────────────────────────────────────────────────────────

bool DiskPrefixCache::lookup(const std::vector<int32_t> & prompt_ids, int slot) {
    if (disabled() || !layout_known_) return false;

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx < 0) return false;

    if (expired(entries_[(size_t)idx], now())) {
        remove_entry((size_t)idx, "ttl expired");
        return false;
    }
    if (!read_file(entries_[(size_t)idx].path, slot, prompt_ids)) {
        // File is corrupt or incompatible — remove it.
        remove_entry((size_t)idx, "lookup failed");
        return false;
    }

    if (layout_from_disk_) {
        const std::array<uint8_t, 16> disk_id = layout_id_;
        auto ref = backend_.snapshot_ref(slot);
        if (!ref.ctx) {
            backend_.snapshot_free(slot);
            return false;
        }
        compute_layout_id(ref.ctx);
        if (std::memcmp(disk_id.data(), layout_id_.data(), 16) != 0) {
            std::fprintf(stderr,
                         "[disk-cache] adopted layout mismatch: disk=%s model=%s\n",
                         hex(disk_id.data(), 16).c_str(),
                         hex(layout_id_.data(), 16).c_str());
            backend_.snapshot_free(slot);
            layout_from_disk_ = false;
            layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
            mkdir_p(layout_dir_);
            return false;
        }
        layout_from_disk_ = false;
    }

    // Persist last_used in the file so LRU order and the TTL survive a
    // restart. Not fsynced: losing it in a crash only ages the entry.
    DiskEntry & entry = entries_[(size_t)idx];
    entry.last_used = now();
    if (FILE * f = std::fopen(entry.path.c_str(), "r+b")) {
        if (std::fseek(f, DISK_CACHE_LAST_USED_OFFSET, SEEK_SET) == 0) {
            write_u64(f, entry.last_used);
        }
        std::fclose(f);
    }
    return true;
}

void DiskPrefixCache::for_each_aux(
        const std::function<void(const std::string &)> & visit) {
    if (disabled()) return;
    std::vector<std::pair<uint64_t, std::string>> files;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto & e : entries_) {
            if (e.own_salt) files.push_back({e.last_used, e.path});
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto & file : files) {
        FILE * f = std::fopen(file.second.c_str(), "rb");
        if (!f) continue;
        DiskCacheHeader hdr{};
        std::string aux;
        const bool ok = read_header(f, hdr) &&
            read_aux_section(f, hdr.n_tensors, hdr.token_count, aux);
        std::fclose(f);
        if (ok && !aux.empty()) visit(aux);
    }
}

DiskPrefixCache::PrefixProbe DiskPrefixCache::longest_prefix(
        const std::vector<int32_t> & prompt_ids) {
    PrefixProbe out;
    if (disabled() || !layout_known_) return out;
    std::lock_guard<std::mutex> lock(mu_);
    const uint64_t t = now();
    std::vector<uint32_t> lengths;
    for (const auto & e : entries_) {
        if (e.layout_id != layout_id_ || expired(e, t)) continue;
        if (e.token_count > prompt_ids.size()) continue;
        if ((int)e.token_count < config_.min_tokens) continue;
        lengths.push_back(e.token_count);
    }
    std::sort(lengths.begin(), lengths.end(), std::greater<uint32_t>());
    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
    for (uint32_t len : lengths) {
        const int idx = find_entry(hash_prefix(prompt_ids.data(), (int)len));
        if (idx >= 0) {
            out.len = (int)len;
            out.cur_pos = (int)entries_[(size_t)idx].cur_pos;
            return out;
        }
    }
    return out;
}

int DiskPrefixCache::prune_ancestors(const std::vector<int32_t> & tokens,
                                     int keep_max) {
    if (disabled() || !layout_known_) return 0;
    std::lock_guard<std::mutex> lock(mu_);
    int removed = 0;
    for (size_t i = entries_.size(); i-- > 0;) {
        const DiskEntry & e = entries_[i];
        if (e.layout_id != layout_id_) continue;
        if ((int)e.token_count <= keep_max || e.token_count >= tokens.size()) continue;
        if (hash_prefix(tokens.data(), (int)e.token_count) != e.token_hash) continue;
        remove_entry(i, "superseded by deeper session state");
        ++removed;
    }
    return removed;
}

// ─── Save ───────────────────────────────────────────────────────────────

bool DiskPrefixCache::save(int slot, const std::vector<int32_t> & prompt_ids,
                           const std::string & aux) {
    if (disabled()) return false;

    // Learn layout on first save.
    if (!layout_known_) {
        learn_layout(slot);
        if (!layout_known_) return false;
    }

    // Check minimum token threshold.
    if ((int)prompt_ids.size() < config_.min_tokens) return false;

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return false;

    // A file is keyed by the hash of `prompt_ids` only, and a hit restores
    // the whole snapshot. A snapshot that extends past its key would be
    // restored for any prompt sharing just the key prefix, with unverified
    // tokens in between, so such files are never written.
    if (ref.cur_pos > (int)prompt_ids.size()) {
        std::fprintf(stderr,
                     "[disk-cache] skip save: snapshot pos=%d exceeds key of "
                     "%zu tokens\n", ref.cur_pos, prompt_ids.size());
        return false;
    }

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);
    expire_locked();

    // Already on disk: count it as a use.
    const int existing = find_entry(hash);
    if (existing >= 0) {
        entries_[(size_t)existing].last_used = now();
        return true;
    }

    // Make room first, so the budget holds while the file is being written.
    uint64_t payload = 0;
    size_t table_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t;
         t = ggml_get_next_tensor(ref.ctx, t)) {
        payload += ggml_nbytes(t);
        table_bytes += 2 + std::strlen(t->name) + 4 + 32 + 8;
    }
    const uint64_t est_size = DISK_CACHE_HEADER_SIZE + table_bytes +
        4ull * prompt_ids.size() + 4 + aux.size() + payload;
    if (est_size > (uint64_t)config_.budget_bytes) {
        std::fprintf(stderr, "[disk-cache] skip save: %.1f MB exceeds budget\n",
                     (double)est_size / (1024.0 * 1024.0));
        return false;
    }
    make_room_locked(est_size);

    std::string path = make_path(hash);
    std::string tmp_path = path + ".tmp";
    mkdir_p(layout_dir_);

    const auto t0 = std::chrono::steady_clock::now();
    if (!write_file(tmp_path, ref, prompt_ids, aux)) {
        std::fprintf(stderr, "[disk-cache] write failed (%s), skipping %s\n",
                     std::strerror(errno), path.c_str());
        std::remove(tmp_path.c_str());
        return false;
    }

    // Atomic rename.
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }
    sync_directory(layout_dir_);

    // Update index.
    DiskEntry entry;
    entry.path        = path;
    std::memcpy(entry.token_hash.data(), hash.data(), 16);
    entry.layout_id   = layout_id_;
    entry.token_count = (uint32_t)prompt_ids.size();
    entry.cur_pos     = (uint32_t)ref.cur_pos;
    entry.last_used   = now();
    entry.own_salt    = true;
    entry.file_size   = est_size;

    total_bytes_ += entry.file_size;
    entries_.push_back(std::move(entry));

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[disk-cache] saved %s (%u tokens, %d pos, %.1f MB, %.2fs)\n",
                 hex(hash.data(), 16).c_str(),
                 (uint32_t)prompt_ids.size(), ref.cur_pos,
                 (double)est_size / (1024.0 * 1024.0), secs);
    return true;
}

// ─── Continued checkpoints ──────────────────────────────────────────────

bool DiskPrefixCache::maybe_store_continued(int slot,
                                            const std::vector<int32_t> & all_tokens,
                                            int cur_pos) {
    if (disabled()) return false;
    if (cur_pos < config_.min_tokens) return false;

    const int interval = config_.continued_interval;
    if (interval <= 0) return false;

    // The interval only paces the checkpoints: fire once each time cur_pos
    // crosses a new multiple of it. The file itself is keyed by the full
    // token prefix the snapshot covers (cur_pos tokens), so a hit can only
    // come from a prompt that actually contains everything in the snapshot.
    int target = (cur_pos / interval) * interval;
    if (target <= continued_last_store_pos_) return false;
    if (target < config_.min_tokens) return false;
    if (cur_pos > (int)all_tokens.size()) return false;

    std::vector<int32_t> prefix(all_tokens.begin(), all_tokens.begin() + cur_pos);
    bool ok = save(slot, prefix);
    if (ok) {
        continued_last_store_pos_ = target;
        std::fprintf(stderr, "[disk-cache] continued checkpoint at %d tokens\n", cur_pos);
    }
    return ok;
}

// ─── Cold prefix boundary ───────────────────────────────────────────────

int DiskPrefixCache::cold_prefix_boundary(const std::vector<int32_t> & prompt_ids,
                                          const std::vector<int> & boundaries) {
    if (disabled() || !layout_known_) return 0;

    const int prompt_len = (int)prompt_ids.size();
    if (prompt_len <= config_.cold_max_tokens) return 0;
    if (boundaries.empty()) return 0;

    // Find the last turn boundary <= cold_max_tokens.
    int best = 0;
    for (int b : boundaries) {
        if (b <= config_.cold_max_tokens && b >= config_.min_tokens) {
            best = b;
        }
    }
    if (best == 0) return 0;

    // Check if we already have a disk entry covering this prefix.
    PrefixHash hash = hash_prefix(prompt_ids.data(), best);
    std::lock_guard<std::mutex> lock(mu_);
    if (find_entry(hash) >= 0) return 0;  // already cached

    return best;
}

// ─── Helpers ────────────────────────────────────────────────────────────

std::string DiskPrefixCache::make_path(const PrefixHash & hash) const {
    return layout_dir_ + "/" + hex(hash.data(), 16) + ".dkv";
}

int DiskPrefixCache::find_entry(const PrefixHash & hash) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[i].token_hash == hash &&
            entries_[i].layout_id == layout_id_) return i;
    }
    return -1;
}

// ─── File I/O: Write ────────────────────────────────────────────────────

bool DiskPrefixCache::write_file(const std::string & path,
                                 const ModelBackend::SnapshotRef & ref,
                                 const std::vector<int32_t> & prompt_ids,
                                 const std::string & aux) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    // Count tensors and compute payload size.
    uint32_t n_tensors = 0;
    uint64_t payload_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        n_tensors++;
        payload_bytes += ggml_nbytes(t);
    }

    // Write header.
    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memcpy(hdr.layout_id, layout_id_.data(), 16);
    hdr.cur_pos       = (uint32_t)ref.cur_pos;
    hdr.n_tensors     = n_tensors;
    hdr.token_count   = (uint32_t)prompt_ids.size();
    PrefixHash ph = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    std::memcpy(hdr.token_hash, ph.data(), 16);
    hdr.payload_bytes = payload_bytes;
    hdr.created_at    = now();
    hdr.last_used     = hdr.created_at;
    hdr.last_tok      = ref.last_tok;
    std::memcpy(hdr.identity_salt, identity_salt_.data(), 16);

    bool ok = write_header(f, hdr);

    // Write tensor table.
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); ok && t; t = ggml_get_next_tensor(ref.ctx, t)) {
        uint16_t name_len = (uint16_t)std::strlen(t->name);
        write_u16(f, name_len);
        std::fwrite(t->name, 1, name_len, f);
        write_u32(f, (uint32_t)t->type);
        for (int d = 0; d < 4; ++d) write_i64(f, t->ne[d]);
        write_u64(f, ggml_nbytes(t));
        ok = !std::ferror(f);
    }

    // The exact key: a hit compares it with the prompt before restoring.
    ok = ok && std::fwrite(prompt_ids.data(), sizeof(int32_t), prompt_ids.size(), f) ==
        prompt_ids.size();
    write_u32(f, (uint32_t)aux.size());
    ok = ok && (aux.empty() || std::fwrite(aux.data(), 1, aux.size(), f) == aux.size());

    // Write tensor data: the snapshot's bytes as stored, no conversion.
    std::vector<uint8_t> buf(4 * 1024 * 1024);  // 4 MB transfer buffer
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); ok && t; t = ggml_get_next_tensor(ref.ctx, t)) {
        size_t nbytes = ggml_nbytes(t);
        size_t offset = 0;
        while (ok && offset < nbytes) {
            size_t chunk = std::min(buf.size(), nbytes - offset);
            ggml_backend_tensor_get(t, buf.data(), offset, chunk);
            ok = std::fwrite(buf.data(), 1, chunk, f) == chunk;
            offset += chunk;
        }
    }

    // Durable before the rename publishes it.
    ok = ok && std::fflush(f) == 0 && sync_file(f);
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

// ─── File I/O: Read ─────────────────────────────────────────────────────

bool DiskPrefixCache::read_file(const std::string & path, int slot,
                                const std::vector<int32_t> & prompt_ids) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    DiskCacheHeader hdr{};
    if (!read_header(f, hdr)) { std::fclose(f); return false; }

    // Validate.
    if (std::memcmp(hdr.magic, "DKVC", 4) != 0) { std::fclose(f); return false; }
    if (hdr.version != DISK_CACHE_VERSION) { std::fclose(f); return false; }
    if (std::memcmp(hdr.layout_id, layout_id_.data(), 16) != 0) { std::fclose(f); return false; }
    if (std::memcmp(hdr.identity_salt, identity_salt_.data(), 16) != 0) {
        std::fprintf(stderr, "[disk-cache] rejecting %s: written by another "
                     "model/KV type/config\n", path.c_str());
        std::fclose(f);
        return false;
    }
    // Never restore a snapshot that extends past the tokens its key covers
    // (see save()). Older files written that way are dropped by the caller.
    if (hdr.cur_pos > hdr.token_count) {
        std::fprintf(stderr,
                     "[disk-cache] rejecting %s: snapshot pos=%u exceeds key of "
                     "%u tokens\n", path.c_str(), hdr.cur_pos, hdr.token_count);
        std::fclose(f);
        return false;
    }
    if (hdr.token_count != prompt_ids.size()) { std::fclose(f); return false; }

    // Read tensor table.
    std::vector<DiskTensorEntry> table;
    table.reserve(hdr.n_tensors);
    for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
        DiskTensorEntry ent;
        uint16_t name_len;
        if (!read_u16(f, name_len)) { std::fclose(f); return false; }
        if (name_len >= GGML_MAX_NAME) { std::fclose(f); return false; }
        char name_buf[GGML_MAX_NAME] = {};
        if (std::fread(name_buf, 1, name_len, f) != name_len) { std::fclose(f); return false; }
        ent.name = name_buf;
        if (!read_u32(f, ent.type) || ent.type >= GGML_TYPE_COUNT) { std::fclose(f); return false; }
        for (int d = 0; d < 4; ++d) {
            if (!read_i64(f, ent.ne[d])) { std::fclose(f); return false; }
        }
        uint64_t nbytes_raw;
        if (!read_u64(f, nbytes_raw)) { std::fclose(f); return false; }
        ent.nbytes = (size_t)nbytes_raw;
        table.push_back(std::move(ent));
    }

    // The stored key must be exactly this prompt (not just its hash): a
    // state keyed on another token stream, e.g. a compressed view, can
    // never be restored for it.
    std::vector<int32_t> stored(prompt_ids.size());
    if (std::fread(stored.data(), sizeof(int32_t), stored.size(), f) != stored.size() ||
        stored != prompt_ids) {
        std::fprintf(stderr, "[disk-cache] rejecting %s: stored tokens differ "
                     "from the prompt\n", path.c_str());
        std::fclose(f);
        return false;
    }
    uint32_t aux_len = 0;
    if (!read_u32(f, aux_len) || std::fseek(f, (long)aux_len, SEEK_CUR) != 0) {
        std::fclose(f);
        return false;
    }

    // Allocate ggml context + buffer.
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(hdr.n_tensors + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::fclose(f); return false; }

    // Create tensors.
    for (const auto & ent : table) {
        ggml_tensor * t = ggml_new_tensor(ctx, (ggml_type)ent.type, 4, ent.ne);
        if (!t || ggml_nbytes(t) != ent.nbytes) {
            ggml_free(ctx);
            std::fclose(f);
            return false;
        }
        ggml_set_name(t, ent.name.c_str());
    }

    // Host buffer from the CPU buffer type: no backend object to keep alive;
    // the adopting backend frees the buffer in snapshot_free().
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_free(ctx);
        std::fclose(f);
        return false;
    }

    // Read tensor data.
    std::vector<uint8_t> read_buf(4 * 1024 * 1024);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        size_t nbytes = ggml_nbytes(t);
        size_t offset = 0;
        while (offset < nbytes) {
            size_t chunk = std::min(read_buf.size(), nbytes - offset);
            if (std::fread(read_buf.data(), 1, chunk, f) != chunk) {
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                std::fclose(f);
                return false;
            }
            ggml_backend_tensor_set(t, read_buf.data(), offset, chunk);
            offset += chunk;
        }
    }
    std::fclose(f);

    // Hand off to backend.
    if (!backend_.snapshot_adopt(slot, ctx, buf, (int)hdr.cur_pos, hdr.last_tok)) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    return true;
}

// ─── Header I/O ─────────────────────────────────────────────────────────

bool DiskPrefixCache::write_header(FILE * f, const DiskCacheHeader & hdr) {
    std::fwrite(hdr.magic, 1, 4, f);
    write_u32(f, hdr.version);
    std::fwrite(hdr.layout_id, 1, 16, f);
    write_u32(f, hdr.cur_pos);
    write_u32(f, hdr.n_tensors);
    write_u32(f, hdr.token_count);
    std::fwrite(hdr.token_hash, 1, 16, f);
    write_u64(f, hdr.payload_bytes);
    write_u64(f, hdr.created_at);
    write_u64(f, hdr.last_used);
    write_u32(f, (uint32_t)hdr.last_tok);  // stored as u32, cast back on read
    std::fwrite(hdr.identity_salt, 1, 16, f);
    // Total: 4+4+16+4+4+4+16+8+8+8+4+16 = 96 bytes. No extra padding needed.
    return !std::ferror(f);
}

bool DiskPrefixCache::read_header(FILE * f, DiskCacheHeader & hdr) {
    if (std::fread(hdr.magic, 1, 4, f) != 4) return false;
    if (!read_u32(f, hdr.version)) return false;
    if (std::fread(hdr.layout_id, 1, 16, f) != 16) return false;
    if (!read_u32(f, hdr.cur_pos)) return false;
    if (!read_u32(f, hdr.n_tensors)) return false;
    if (!read_u32(f, hdr.token_count)) return false;
    if (std::fread(hdr.token_hash, 1, 16, f) != 16) return false;
    if (!read_u64(f, hdr.payload_bytes)) return false;
    if (!read_u64(f, hdr.created_at)) return false;
    if (!read_u64(f, hdr.last_used)) return false;
    uint32_t last_tok_raw;
    if (!read_u32(f, last_tok_raw)) return false;
    hdr.last_tok = (int32_t)last_tok_raw;
    if (std::fread(hdr.identity_salt, 1, 16, f) != 16) return false;
    return true;
}

}  // namespace luce::common
