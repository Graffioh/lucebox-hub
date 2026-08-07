#include "deepseek4_paged_cache.h"

#include "deepseek4_page_layout.h"

#ifndef DFLASH_DS4_PLAN_ONLY
#include "deepseek4_internal.h"
#endif

#include <cstdio>
#include <limits>
#include <memory>

namespace dflash::common {
namespace {
bool add_mul(uint64_t & dst, uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    const uint64_t v = a * b;
    if (dst > std::numeric_limits<uint64_t>::max() - v) return false;
    dst += v;
    return true;
}
}

bool prepare_deepseek4_gathered_lane_rows(
        const int32_t * slots, const int64_t * positions, uint32_t lanes,
        const int32_t * block_tables, uint32_t block_table_stride,
        uint32_t physical_blocks, uint32_t ratio,
        std::vector<DeepSeek4GatheredLaneRows> & out) {
    if (!slots || !positions || !block_tables || !block_table_stride ||
        !physical_blocks || (ratio != 0 && ratio != 4 && ratio != 128)) {
        return false;
    }
    std::vector<DeepSeek4GatheredLaneRows> prepared(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        auto & rows = prepared[lane];
        rows.slot = slots[lane];
        rows.position = positions[lane];
        if (rows.slot < 0) continue; // Padding must remain entirely passive.
        if (rows.position < 0) return false;
        const uint64_t pos = static_cast<uint64_t>(rows.position);
        // The current row is appended in-graph, so retain at most the 127
        // preceding rows that can coexist with it in the 128-row SWA window.
        const uint64_t first_raw = pos >= DS4_PAGE_TOKENS
            ? pos - DS4_PAGE_TOKENS + 1 : 0;
        rows.raw_history.reserve(static_cast<size_t>(pos - first_raw));
        for (uint64_t p = first_raw; p < pos; ++p) {
            rows.raw_history.push_back(
                int64_t(rows.slot) * DS4_PAGE_TOKENS + ds4_raw_ring_row(p));
        }
        rows.raw_scatter = int64_t(rows.slot) * DS4_PAGE_TOKENS +
                           ds4_raw_ring_row(pos);
        if (!ratio) continue;

        // Every completed group before the current token contributes one
        // chronological row.  Looking up each logical page (rather than
        // assuming contiguous physical pages) is the reference behaviour.
        const uint64_t completed = pos / ratio;
        rows.compressed_history.reserve(static_cast<size_t>(completed));
        for (uint64_t group = 0; group < completed; ++group) {
            const uint64_t end_token = group * ratio + ratio - 1;
            const uint64_t logical_block = end_token / DS4_PAGE_TOKENS;
            if (logical_block >= block_table_stride) return false;
            const int32_t physical =
                block_tables[size_t(lane) * block_table_stride + logical_block];
            if (physical < 0 || uint32_t(physical) >= physical_blocks) return false;
            uint64_t row = 0; bool emitted = false;
            if (!ds4_compressed_page_row(end_token, uint32_t(physical), ratio,
                                         row, emitted) || !emitted ||
                row > uint64_t(INT64_MAX)) return false;
            rows.compressed_history.push_back(static_cast<int64_t>(row));
        }
        const uint64_t logical_block = pos / DS4_PAGE_TOKENS;
        if (logical_block >= block_table_stride) return false;
        const int32_t physical =
            block_tables[size_t(lane) * block_table_stride + logical_block];
        if (physical < 0 || uint32_t(physical) >= physical_blocks) return false;
        uint64_t scatter = 0;
        if (!ds4_compressed_page_row(pos, uint32_t(physical), ratio, scatter,
                                     rows.compressed_emitted) ||
            scatter > uint64_t(INT64_MAX)) return false;
        if (rows.compressed_emitted) rows.compressed_scatter = int64_t(scatter);
    }
    out = std::move(prepared);
    return true;
}

bool plan_deepseek4_paged_cache(uint32_t head_dim, uint32_t indexer_head_dim,
                                uint32_t slots, uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<uint32_t> & ratios,
                                DeepSeek4PagedCachePlan & out) {
    DeepSeek4PagedCachePlan p;
    if (!head_dim || !indexer_head_dim || !slots || !max_ctx ||
        !physical_blocks || ratios.empty() ||
        physical_blocks > UINT32_MAX / DS4_PAGE_TOKENS) return false;
    p.slots = slots; p.max_ctx = max_ctx; p.physical_blocks = physical_blocks;
    p.max_blocks_per_sequence = 1 + (max_ctx - 1) / DS4_PAGE_TOKENS;
    p.ratios = ratios;
    p.physical_rows.resize(ratios.size());
    // block table, lengths, and active IDs, all I32.
    if (!add_mul(p.metadata_bytes, p.max_blocks_per_sequence, uint64_t(slots) * 4) ||
        !add_mul(p.metadata_bytes, slots, 8)) return false;
    for (size_t i = 0; i < ratios.size(); ++i) {
        const uint32_t r = ratios[i];
        if (r != 0 && r != 4 && r != 128) return false;
        if (!add_mul(p.raw_bytes, uint64_t(head_dim) * DS4_PAGE_TOKENS * 2, slots)) return false;
        if (!r) continue;
        uint64_t rows = 0;
        if (!ds4_compressed_page_capacity(physical_blocks, r, rows)) return false;
        p.physical_rows[i] = rows;
        if (!add_mul(p.compressed_bytes, uint64_t(head_dim) * 2, rows)) return false;
        const uint64_t width = uint64_t(head_dim) * (r == 4 ? 2 : 1);
        const uint64_t state_rows = r == 4 ? 8 : 128;
        if (!add_mul(p.state_bytes, width * state_rows * 8, slots)) return false; // KV + score F32
        if (r == 4) {
            if (!add_mul(p.compressed_bytes, uint64_t(indexer_head_dim) * 2, rows)) return false;
            if (!add_mul(p.state_bytes, uint64_t(indexer_head_dim) * 2 * 8 * 8, slots)) return false;
        }
    }
    p.total_persistent_bytes = p.metadata_bytes;
    if (p.total_persistent_bytes > UINT64_MAX - p.raw_bytes) return false;
    p.total_persistent_bytes += p.raw_bytes;
    if (p.total_persistent_bytes > UINT64_MAX - p.compressed_bytes) return false;
    p.total_persistent_bytes += p.compressed_bytes;
    if (p.total_persistent_bytes > UINT64_MAX - p.state_bytes) return false;
    p.total_persistent_bytes += p.state_bytes;
    out = std::move(p);
    return true;
}

#ifndef DFLASH_DS4_PLAN_ONLY
bool create_deepseek4_paged_cache(ggml_backend_t backend,
                                  const DeepSeek4Weights & w, uint32_t slots,
                                  uint32_t max_ctx, uint32_t physical_blocks,
                                  DeepSeek4PagedCache & out) {
    free_deepseek4_paged_cache(out);
    DeepSeek4PagedCachePlan plan;
    if (!backend || w.n_layer <= 0 || w.compress_ratios.size() != size_t(w.n_layer) ||
        !plan_deepseek4_paged_cache(w.head_dim, w.n_indexer_head_dim, slots,
                                    max_ctx, physical_blocks, w.compress_ratios, plan)) return false;
    try { out.pool = std::make_unique<PagedKvPool>(physical_blocks, slots, DS4_PAGE_TOKENS); }
    catch (...) { free_deepseek4_paged_cache(out); return false; }
    out.plan = plan;
    out.layers.resize(w.n_layer);
    ggml_init_params ip{ggml_tensor_overhead() * size_t(w.n_layer * 9 + 8) + 4096, nullptr, true};
    out.ctx = ggml_init(ip);
    if (!out.ctx) { free_deepseek4_paged_cache(out); return false; }
    out.block_table = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, plan.max_blocks_per_sequence, slots);
    out.sequence_lengths = ggml_new_tensor_1d(out.ctx, GGML_TYPE_I32, slots);
    out.active_slot_ids = ggml_new_tensor_1d(out.ctx, GGML_TYPE_I32, slots);
    for (int il = 0; il < w.n_layer; ++il) {
        auto & l = out.layers[il]; const uint32_t r = plan.ratios[il];
        l.ratio = r; l.physical_rows = plan.physical_rows[il];
        l.raw_kv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16, w.head_dim, DS4_PAGE_TOKENS, slots);
        if (!r) continue;
        l.comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.head_dim, l.physical_rows);
        const int64_t width = int64_t(w.head_dim) * (r == 4 ? 2 : 1), sr = r == 4 ? 8 : 128;
        l.attn_compressor.state_kv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, width, sr, slots);
        l.attn_compressor.state_score = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, width, sr, slots);
        if (r == 4) {
            l.index_comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.n_indexer_head_dim, l.physical_rows);
            const int64_t iw = int64_t(w.n_indexer_head_dim) * 2;
            l.indexer_compressor.state_kv = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, iw, 8, slots);
            l.indexer_compressor.state_score = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, iw, 8, slots);
        }
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) { free_deepseek4_paged_cache(out); return false; }
    ggml_backend_buffer_clear(out.buf, 0);
    // One shared, contiguous legacy cache is intentionally retained for prefill.
    if (!create_deepseek4_cache(backend, w, int(max_ctx), out.prefill_staging)) {
        free_deepseek4_paged_cache(out); return false;
    }
    return true;
}

void free_deepseek4_paged_cache(DeepSeek4PagedCache & c) {
    deepseek4_release_paged_gathered_runtime(c);
    free_deepseek4_cache(c.prefill_staging);
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.pool.reset(); c.layers.clear(); c.block_table = nullptr;
    c.sequence_lengths = nullptr; c.active_slot_ids = nullptr; c.plan = {};
}
#endif
} // namespace dflash::common
