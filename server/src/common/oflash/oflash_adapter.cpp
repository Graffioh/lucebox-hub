// oflash_adapter.cpp — adapter file parsing, LoRA slot management, store.

#include "oflash_adapter.h"
#include "oflash_format.h"

#include "internal.h"               // DraftWeights
#include "common/gguf_mmap.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common::oflash {

using json = nlohmann::json;

namespace {

// ── Minimal safetensors reader (with __metadata__) ──────────────────
// Format: u64 LE header length, JSON header, raw tensor data. The repo's
// only other safetensors parser (draft_safetensors_loader.cpp) is file-local
// and drops __metadata__, which OFlash needs for the drafter-hash refusal.

struct StTensor {
    std::string dtype;              // "F32" | "F16" | "BF16"
    std::vector<int64_t> shape;     // row-major
    uint64_t begin = 0, end = 0;    // offsets into the data section
};

struct StFile {
    GgufMmap map;
    uint64_t data_start = 0;
    std::unordered_map<std::string, StTensor> tensors;
    std::unordered_map<std::string, std::string> metadata;
};

bool st_open(const std::string & path, StFile & f, std::string & error) {
    if (!f.map.open(path.c_str(), error)) return false;
    const uint8_t * base = (const uint8_t *)f.map.data();
    const size_t size = f.map.size();
    if (size < 8) { error = "safetensors too small: " + path; return false; }
    uint64_t hlen = 0;
    std::memcpy(&hlen, base, 8);
    if (hlen > size - 8) { error = "safetensors header overruns file"; return false; }
    f.data_start = 8 + hlen;

    json h = json::parse(base + 8, base + 8 + hlen, nullptr,
                         /*allow_exceptions=*/false);
    if (h.is_discarded() || !h.is_object()) {
        error = "safetensors header is not valid JSON";
        return false;
    }
    try {
        for (auto it = h.begin(); it != h.end(); ++it) {
            if (it.key() == "__metadata__") {
                if (!it.value().is_object()) continue;
                for (auto mit = it.value().begin(); mit != it.value().end(); ++mit) {
                    if (mit.value().is_string()) {
                        f.metadata[mit.key()] = mit.value().get<std::string>();
                    }
                }
                continue;
            }
            const json & v = it.value();
            if (!v.is_object() || !v.contains("dtype") ||
                !v.contains("shape") || !v.contains("data_offsets")) {
                error = "safetensors entry malformed: " + it.key();
                return false;
            }
            StTensor t;
            t.dtype = v["dtype"].get<std::string>();
            for (const auto & d : v["shape"]) {
                t.shape.push_back(d.get<int64_t>());
            }
            t.begin = v["data_offsets"][0].get<uint64_t>();
            t.end   = v["data_offsets"][1].get<uint64_t>();
            if (t.begin > t.end || t.end > size - f.data_start) {
                error = "safetensors tensor out of bounds: " + it.key();
                return false;
            }
            f.tensors.emplace(it.key(), std::move(t));
        }
    } catch (const json::exception & e) {
        error = std::string("safetensors header has invalid field types: ") +
                e.what();
        return false;
    }
    return true;
}

size_t st_dtype_size(const std::string & d) {
    if (d == "F32") return 4;
    if (d == "F16" || d == "BF16") return 2;
    return 0;
}

// Convert one tensor's payload to F16 in place-preserving element order.
bool st_to_f16(const StFile & f, const std::string & name, const StTensor & t,
               std::vector<uint16_t> & out, std::string & error) {
    const size_t esz = st_dtype_size(t.dtype);
    if (esz == 0) { error = "unsupported adapter dtype " + t.dtype; return false; }
    const uint64_t bytes = t.end - t.begin;
    const size_t n = bytes / esz;
    if (n * esz != bytes) { error = "tensor byte size not divisible by dtype"; return false; }
    const uint8_t * src = (const uint8_t *)f.map.data() + f.data_start + t.begin;
    out.resize(n);
    if (t.dtype == "F16") {
        std::memcpy(out.data(), src, bytes);
    } else if (t.dtype == "F32") {
        std::vector<float> tmp(n);
        std::memcpy(tmp.data(), src, bytes);
        if (std::any_of(tmp.begin(), tmp.end(),
                        [](float v) { return !std::isfinite(v); })) {
            error = "adapter tensor contains NaN/Inf: " + name;
            return false;
        }
        ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *)out.data(), (int64_t)n);
    } else {  // BF16 → F32 → F16
        std::vector<float> tmp(n);
        for (size_t i = 0; i < n; i++) {
            uint16_t bits = 0;
            std::memcpy(&bits, src + i * sizeof(bits), sizeof(bits));
            if ((bits & 0x7f80u) == 0x7f80u) {
                error = "adapter tensor contains NaN/Inf: " + name;
                return false;
            }
            uint32_t u = (uint32_t)bits << 16;
            std::memcpy(&tmp[i], &u, 4);
        }
        ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *)out.data(), (int64_t)n);
    }
    // This also rejects finite F32/BF16 values that overflow during the F16
    // conversion. Uploading an infinity is unsafe regardless of its source.
    if (std::any_of(out.begin(), out.end(), [](uint16_t bits) {
            return (bits & 0x7c00u) == 0x7c00u;
        })) {
        error = "adapter tensor contains NaN/Inf after F16 conversion: " + name;
        return false;
    }
    return true;
}

int64_t st_numel(const StTensor & t) {
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    return n;
}

}  // namespace

std::vector<OFlashLoraTensorSpec> oflash_lora_expected_tensors(
        const DraftWeights & dw, int rank) {
    std::vector<OFlashLoraTensorSpec> v;
    const int64_t hidden = dw.n_embd;
    const int64_t q_dim  = (int64_t)dw.head_dim * dw.n_head;
    const int64_t kv_dim = (int64_t)dw.head_dim * dw.n_head_kv;
    const int64_t fc_in  = (int64_t)dw.n_target_layers * hidden;
    auto pair = [&](const std::string & base, int64_t in, int64_t out) {
        v.push_back({base + ".lora_a", in, rank});
        v.push_back({base + ".lora_b", rank, out});
    };
    pair("dflash.fc", fc_in, hidden);
    for (int i = 0; i < dw.n_layer; i++) {
        const std::string blk = "blk." + std::to_string(i);
        // The GGUF scalar can drift during re-export. LoRA shapes must match
        // the loaded matrices that ggml_mul_mat actually consumes.
        const DraftLayer * layer = (size_t)i < dw.layers.size()
            ? &dw.layers[(size_t)i] : nullptr;
        const int64_t up_out = layer && layer->w_up
            ? layer->w_up->ne[1] : dw.n_ff;
        const int64_t down_in = layer && layer->w_down
            ? layer->w_down->ne[0] : dw.n_ff;
        pair(blk + ".attn_q",      hidden, q_dim);
        pair(blk + ".attn_k",      hidden, kv_dim);
        pair(blk + ".attn_v",      hidden, kv_dim);
        pair(blk + ".attn_output", q_dim,  hidden);
        pair(blk + ".ffn_up",      hidden, up_out);
        pair(blk + ".ffn_down",    down_in, hidden);
    }
    return v;
}

bool oflash_adapter_load(const std::string & path,
                         const DraftWeights & dw,
                         int rank,
                         float alpha,
                         const std::string & drafter_sha256,
                         OFlashAdapterHost & out,
                         std::string & error) {
    StFile f;
    if (!st_open(path, f, error)) return false;

    auto meta = [&](const char * key) -> std::string {
        auto it = f.metadata.find(key);
        return it == f.metadata.end() ? std::string() : it->second;
    };
    if (meta("oflash.format") != "1") {
        error = "adapter format version mismatch (oflash.format="
              + meta("oflash.format") + ")";
        return false;
    }
    const std::string file_sha = meta("oflash.drafter_sha256");
    if (file_sha.empty() ||
        drafter_sha256.compare(0, file_sha.size(), file_sha) != 0 ||
        file_sha.size() < 16) {
        error = "adapter drafter hash mismatch (adapter=" + file_sha +
                " loaded=" + drafter_sha256.substr(0, 16) + "...)";
        return false;
    }
    const std::string rank_s = meta("oflash.rank");
    if (rank_s != std::to_string(rank)) {
        error = "adapter rank " + rank_s + " != engine rank " +
                std::to_string(rank);
        return false;
    }
    const std::string alpha_s = meta("oflash.alpha");
    char * alpha_end = nullptr;
    errno = 0;
    const float file_alpha = std::strtof(alpha_s.c_str(), &alpha_end);
    if (alpha_s.empty() || errno != 0 || !alpha_end || *alpha_end != '\0' ||
        !std::isfinite(file_alpha) ||
        std::fabs(file_alpha - alpha) >
            1e-6f * std::max(1.0f, std::fabs(alpha))) {
        error = "adapter alpha " + alpha_s + " != engine alpha " +
                std::to_string(alpha);
        return false;
    }
    uint64_t generation = 0;
    {
        const std::string g = meta("oflash.generation");
        char * end = nullptr;
        errno = 0;
        generation = std::strtoull(g.c_str(), &end, 10);
        if (g.empty() || errno != 0 || !end || *end != '\0') {
            error = "adapter has invalid oflash.generation";
            return false;
        }
    }

    // Every expected tensor must exist with the exact shape; extras refuse
    // (a mismatched trainer is a bug we want loud, not partially applied).
    const auto specs = oflash_lora_expected_tensors(dw, rank);
    if (f.tensors.size() != specs.size()) {
        error = "adapter has " + std::to_string(f.tensors.size()) +
                " tensors, expected " + std::to_string(specs.size());
        return false;
    }
    out.tensors.clear();
    for (const auto & s : specs) {
        auto it = f.tensors.find(s.name);
        if (it == f.tensors.end()) {
            error = "adapter missing tensor " + s.name;
            return false;
        }
        const StTensor & t = it->second;
        // safetensors row-major [rows, cols] maps to ggml ne{cols, rows}:
        // spec dims are ggml ne{in_dim=ne0, out_dim=ne1} → shape [ne1, ne0].
        if (t.shape.size() != 2 || t.shape[0] != s.out_dim ||
            t.shape[1] != s.in_dim) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "adapter tensor %s shape [%lld,%lld], expected [%lld,%lld]",
                s.name.c_str(),
                t.shape.size() > 0 ? (long long)t.shape[0] : -1LL,
                t.shape.size() > 1 ? (long long)t.shape[1] : -1LL,
                (long long)s.out_dim, (long long)s.in_dim);
            error = buf;
            return false;
        }
        if (st_numel(t) != s.in_dim * s.out_dim) {
            error = "adapter tensor numel mismatch: " + s.name;
            return false;
        }
        const size_t elem_size = st_dtype_size(t.dtype);
        const uint64_t expected_bytes =
            (uint64_t)s.in_dim * (uint64_t)s.out_dim * elem_size;
        if (elem_size == 0 || t.end - t.begin != expected_bytes) {
            error = "adapter tensor byte size mismatch: " + s.name;
            return false;
        }
        std::vector<uint16_t> data;
        if (!st_to_f16(f, s.name, t, data, error)) return false;
        out.tensors.emplace(s.name, std::move(data));
    }
    out.generation = generation;
    out.path = path;
    return true;
}

OFlashLoraWeights * oflash_lora_create(const DraftWeights & dw,
                                       ggml_backend_t backend,
                                       int rank,
                                       float alpha,
                                       std::string & error) {
    const auto specs = oflash_lora_expected_tensors(dw, rank);
    // 2 tensors per pair, 1 fc pair + 6 pairs per layer.
    const size_t n_tensors = specs.size();
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * (n_tensors + 8);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    auto * lw = new OFlashLoraWeights();
    lw->rank  = rank;
    lw->alpha = alpha;
    lw->scale = rank > 0 ? alpha / (float)rank : 0.0f;
    lw->ctx = ggml_init(ip);
    if (!lw->ctx) {
        error = "oflash lora ggml_init failed";
        delete lw;
        return nullptr;
    }

    lw->layers.resize((size_t)dw.n_layer);
    std::unordered_map<std::string, ggml_tensor *> by_name;
    for (const auto & s : specs) {
        ggml_tensor * t = ggml_new_tensor_2d(lw->ctx, GGML_TYPE_F16,
                                             s.in_dim, s.out_dim);
        ggml_set_name(t, s.name.c_str());
        by_name[s.name] = t;
    }
    auto pick = [&](const std::string & base) {
        OFlashLoraPair p;
        p.a = by_name.at(base + ".lora_a");
        p.b = by_name.at(base + ".lora_b");
        return p;
    };
    lw->fc = pick("dflash.fc");
    for (int i = 0; i < dw.n_layer; i++) {
        const std::string blk = "blk." + std::to_string(i);
        OFlashLoraLayer & L = lw->layers[(size_t)i];
        L.wq     = pick(blk + ".attn_q");
        L.wk     = pick(blk + ".attn_k");
        L.wv     = pick(blk + ".attn_v");
        L.wo     = pick(blk + ".attn_output");
        L.w_up   = pick(blk + ".ffn_up");
        L.w_down = pick(blk + ".ffn_down");
    }

    lw->buf = ggml_backend_alloc_ctx_tensors(lw->ctx, backend);
    if (!lw->buf) {
        error = "oflash lora buffer alloc failed";
        oflash_lora_free(lw);
        return nullptr;
    }
    // Zero-fill so serving before the first adapter equals the base drafter.
    OFlashAdapterHost zeros;  // empty tensors map → zero upload
    if (!oflash_lora_upload(*lw, zeros, error)) {
        oflash_lora_free(lw);
        return nullptr;
    }
    return lw;
}

void oflash_lora_free(OFlashLoraWeights * lw) {
    if (!lw) return;
    if (lw->buf) ggml_backend_buffer_free(lw->buf);
    if (lw->ctx) ggml_free(lw->ctx);
    delete lw;
}

bool oflash_lora_upload(OFlashLoraWeights & lw,
                        const OFlashAdapterHost & host,
                        std::string & error) {
    std::vector<uint16_t> zeros;
    for (ggml_tensor * t = ggml_get_first_tensor(lw.ctx); t != nullptr;
         t = ggml_get_next_tensor(lw.ctx, t)) {
        const size_t n = (size_t)ggml_nelements(t);
        auto it = host.tensors.find(ggml_get_name(t));
        if (it != host.tensors.end()) {
            if (it->second.size() != n) {
                error = std::string("adapter upload size mismatch: ") +
                        ggml_get_name(t);
                return false;
            }
            ggml_backend_tensor_set(t, it->second.data(), 0, n * 2);
        } else if (host.tensors.empty()) {
            if (zeros.size() < n) zeros.assign(n, 0);
            ggml_backend_tensor_set(t, zeros.data(), 0, n * 2);
        } else {
            error = std::string("adapter missing tensor at upload: ") +
                    ggml_get_name(t);
            return false;
        }
    }
    lw.generation = host.generation;
    return true;
}

// ── Profile store ───────────────────────────────────────────────────

std::string oflash_profile_dir(const std::string & base_dir,
                               const std::string & drafter_sha256,
                               const std::string & profile) {
    const std::string hash16 = drafter_sha256.substr(0, 16);
    return base_dir + "/" + hash16 + "/" + profile;
}

bool oflash_store_read_promoted(const std::string & profile_dir,
                                std::string & adapter_path,
                                uint64_t & generation) {
    std::ifstream in(profile_dir + "/promoted.json");
    if (!in.good()) return false;
    json j = json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object() ||
        !j.contains("adapter") || !j.contains("generation")) {
        return false;
    }
    try {
        adapter_path = j["adapter"].get<std::string>();
        generation   = j["generation"].get<uint64_t>();
    } catch (const json::exception &) {
        return false;
    }
    // Relative paths are relative to the profile dir (portable store).
    if (!adapter_path.empty() && adapter_path[0] != '/') {
        adapter_path = profile_dir + "/" + adapter_path;
    }
    return true;
}

bool oflash_store_write_promoted(const std::string & profile_dir,
                                 const std::string & adapter_path,
                                 uint64_t generation) {
#if defined(_WIN32)
    (void)profile_dir; (void)adapter_path; (void)generation;
    return false;
#else
    // Store the basename when the adapter lives inside the profile dir.
    std::string rel = adapter_path;
    if (rel.rfind(profile_dir + "/", 0) == 0) {
        rel = rel.substr(profile_dir.size() + 1);
    }
    json j = {{"adapter", rel}, {"generation", generation}};
    const std::string tmp = profile_dir + "/promoted.json.tmp";
    {
        std::ofstream outf(tmp, std::ios::trunc);
        if (!outf.good()) return false;
        outf << j.dump(2) << "\n";
        if (!outf.good()) return false;
    }
    if (::rename(tmp.c_str(), (profile_dir + "/promoted.json").c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
#endif
}

}  // namespace dflash::common::oflash
