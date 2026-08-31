#include "deepseek4_paged_segments.h"

#include "common/paged_attention_config.h"
#include "deepseek4_page_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <unordered_set>
#include <utility>

namespace dflash::common {
namespace {

static_assert(DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS <=
              uint32_t(DEEPSEEK4_MAX_PAGED_SEQUENCES));
static_assert(DEEPSEEK4_PAGED_SEGMENT_SHAPE_KEY_V1 != 0x5041474544LL);

bool checked_table_size(uint32_t slots, uint32_t stride, size_t & size) {
    if (!slots || !stride ||
        size_t(slots) > std::numeric_limits<size_t>::max() / stride) {
        return false;
    }
    size = size_t(slots) * stride;
    return true;
}

bool flattened_raw_row(int32_t slot, uint64_t position, int64_t & row) {
    if (slot < 0) return false;
    const uint64_t base = uint64_t(uint32_t(slot)) * DS4_PAGE_TOKENS;
    const uint64_t value = base + ds4_raw_ring_row(position);
    if (value > uint64_t(INT64_MAX)) return false;
    row = static_cast<int64_t>(value);
    return true;
}

bool segment_table_row(const DeepSeek4PagedStepLayout & layout,
                       const DeepSeek4PagedSegment & segment,
                       uint64_t logical_block,
                       int32_t & physical_block) {
    if (logical_block >= layout.block_table_stride ||
        segment.compact_block_table_offset >
            layout.compact_block_tables.size() ||
        logical_block > std::numeric_limits<size_t>::max() -
            segment.compact_block_table_offset) {
        return false;
    }
    const size_t index = segment.compact_block_table_offset +
                         static_cast<size_t>(logical_block);
    if (index >= layout.compact_block_tables.size()) return false;
    physical_block = layout.compact_block_tables[index];
    return physical_block >= 0 &&
           uint32_t(physical_block) < layout.physical_blocks;
}

bool same_token_rows(const DeepSeek4SegmentTokenRows & lhs,
                     const DeepSeek4SegmentTokenRows & rhs) {
    return lhs.raw_write_row == rhs.raw_write_row &&
           lhs.initial_raw_suffix == rhs.initial_raw_suffix &&
           lhs.prior_segment_rows == rhs.prior_segment_rows &&
           lhs.attention_ape_phase == rhs.attention_ape_phase &&
           lhs.attention_state_row == rhs.attention_state_row &&
           lhs.indexer_ape_phase == rhs.indexer_ape_phase &&
           lhs.indexer_state_row == rhs.indexer_state_row &&
           lhs.sees_emitted_compressed_row ==
               rhs.sees_emitted_compressed_row;
}

bool same_layer_rows(const std::vector<DeepSeek4LayerSegmentRows> & lhs,
                     const std::vector<DeepSeek4LayerSegmentRows> & rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const DeepSeek4LayerSegmentRows & a = lhs[i];
        const DeepSeek4LayerSegmentRows & b = rhs[i];
        if (a.slot != b.slot ||
            a.compression_ratio != b.compression_ratio ||
            a.attention_state_slot != b.attention_state_slot ||
            a.indexer_state_slot != b.indexer_state_slot ||
            a.immutable_raw_history != b.immutable_raw_history ||
            a.immutable_compressed_history !=
                b.immutable_compressed_history ||
            a.tokens.size() != b.tokens.size() ||
            a.emission_token != b.emission_token ||
            a.compressed_write_row != b.compressed_write_row ||
            a.compressed_position != b.compressed_position) {
            return false;
        }
        for (size_t token = 0; token < a.tokens.size(); ++token) {
            if (!same_token_rows(a.tokens[token], b.tokens[token])) {
                return false;
            }
        }
    }
    return true;
}

int ratio_index(uint32_t ratio) {
    if (ratio == 0) return 0;
    if (ratio == 4) return 1;
    if (ratio == 128) return 2;
    return -1;
}

} // namespace

bool prepare_deepseek4_paged_step_layout(
        const std::vector<DeepSeek4PagedSegmentSpec> & specs,
        const std::vector<int32_t> & slot_block_tables,
        uint32_t slot_count,
        uint32_t max_context,
        uint32_t block_table_stride,
        uint32_t physical_blocks,
        DeepSeek4PagedStepLayout & out) {
    size_t expected_table_size = 0;
    if (specs.empty() ||
        specs.size() > size_t(DEEPSEEK4_MAX_PAGED_SEQUENCES) ||
        !max_context || !physical_blocks ||
        physical_blocks > UINT32_MAX / DS4_PAGE_TOKENS ||
        !checked_table_size(slot_count, block_table_stride,
                            expected_table_size) ||
        slot_block_tables.size() != expected_table_size ||
        block_table_stride > UINT32_MAX / specs.size() ||
        uint64_t(block_table_stride) * DS4_PAGE_TOKENS < max_context) {
        return false;
    }

    uint32_t total_rows = 0;
    bool saw_prefill = false;
    std::unordered_set<int32_t> slot_owners;
    std::unordered_set<int32_t> physical_pages;
    std::unordered_set<int64_t> pool_rows;
    DeepSeek4PagedStepLayout prepared;
    prepared.slot_count = slot_count;
    prepared.max_context = max_context;
    prepared.block_table_stride = block_table_stride;
    prepared.physical_blocks = physical_blocks;
    prepared.segments.reserve(specs.size());

    for (size_t segment_index = 0; segment_index < specs.size();
         ++segment_index) {
        const DeepSeek4PagedSegmentSpec & spec = specs[segment_index];
        const bool decode = spec.kind == DeepSeek4PagedSegmentKind::decode;
        const bool prefill = spec.kind == DeepSeek4PagedSegmentKind::prefill;
        if (!decode && !prefill) return false;
        if (decode && saw_prefill) return false;
        saw_prefill = saw_prefill || prefill;
        if (spec.slot < 0 || uint32_t(spec.slot) >= slot_count ||
            !slot_owners.insert(spec.slot).second ||
            spec.start_position < 0 || spec.start_position > INT32_MAX ||
            (decode && spec.row_count != 1) ||
            (prefill && (spec.row_count < 1 ||
                         spec.row_count > DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS)) ||
            (decode && spec.prompt_tail) ||
            spec.token_ids.size() != spec.row_count ||
            spec.pool_write_rows.size() != spec.row_count ||
            (spec.new_blocks.empty() != (spec.first_new_block < 0))) {
            return false;
        }
        if (total_rows + spec.row_count >
            uint32_t(DEEPSEEK4_MAX_PAGED_SEQUENCES)) {
            return false;
        }
        const int64_t end_position = spec.start_position +
                                     int64_t(spec.row_count) - 1;
        if (end_position < spec.start_position || end_position > INT32_MAX ||
            uint64_t(end_position) >= max_context ||
            (spec.needs_full_logits && prefill && !spec.prompt_tail)) {
            return false;
        }

        DeepSeek4PagedSegment segment;
        segment.kind = spec.kind;
        segment.slot = spec.slot;
        segment.row_offset = total_rows;
        segment.row_count = spec.row_count;
        segment.start_position = spec.start_position;
        segment.prompt_tail = spec.prompt_tail;
        segment.output_row = decode
            ? int32_t(total_rows)
            : (spec.prompt_tail
                ? int32_t(total_rows + spec.row_count - 1) : -1);
        segment.phase4 = uint8_t(uint64_t(spec.start_position) % 4);
        segment.phase128 = uint8_t(
            uint64_t(spec.start_position) % DS4_PAGE_TOKENS);
        if (prepared.compact_block_tables.size() > UINT32_MAX) return false;
        segment.compact_block_table_offset = static_cast<uint32_t>(
            prepared.compact_block_tables.size());
        segment.pool_write_rows = spec.pool_write_rows;
        segment.raw_ring_write_rows.reserve(spec.row_count);

        const size_t source_offset = size_t(uint32_t(spec.slot)) *
                                     block_table_stride;
        prepared.compact_block_tables.insert(
            prepared.compact_block_tables.end(),
            slot_block_tables.begin() + source_offset,
            slot_block_tables.begin() + source_offset + block_table_stride);

        const uint64_t last_logical_block =
            uint64_t(end_position) / DS4_PAGE_TOKENS;
        if (last_logical_block >= block_table_stride) return false;
        for (uint64_t logical = 0; logical <= last_logical_block; ++logical) {
            const int32_t physical = slot_block_tables[
                source_offset + static_cast<size_t>(logical)];
            if (physical < 0 || uint32_t(physical) >= physical_blocks) {
                return false;
            }
            // A physical page has one logical owner across the whole step.
            // Repeating it in one table is also malformed aliasing.
            if (!physical_pages.insert(physical).second) return false;
        }

        int32_t derived_first_new_block = -1;
        std::vector<int32_t> derived_new_blocks;
        for (uint32_t local_row = 0; local_row < spec.row_count;
             ++local_row) {
            if (spec.token_ids[local_row] < 0 ||
                spec.pool_write_rows[local_row] < 0 ||
                !pool_rows.insert(spec.pool_write_rows[local_row]).second) {
                return false;
            }
            const uint64_t position = uint64_t(spec.start_position) + local_row;
            const uint64_t logical_block = position / DS4_PAGE_TOKENS;
            const uint64_t block_offset = position % DS4_PAGE_TOKENS;
            const int32_t physical = slot_block_tables[
                source_offset + static_cast<size_t>(logical_block)];
            const uint64_t expected_pool_row =
                uint64_t(uint32_t(physical)) * DS4_PAGE_TOKENS + block_offset;
            if (expected_pool_row > uint64_t(INT64_MAX) ||
                uint64_t(spec.pool_write_rows[local_row]) != expected_pool_row) {
                return false;
            }
            int64_t raw_row = -1;
            if (!flattened_raw_row(spec.slot, position, raw_row)) return false;
            segment.raw_ring_write_rows.push_back(raw_row);
            if (block_offset == 0) {
                if (logical_block > uint64_t(INT32_MAX)) return false;
                if (derived_first_new_block < 0) {
                    derived_first_new_block = int32_t(logical_block);
                } else if (logical_block != uint64_t(
                               derived_first_new_block +
                               int32_t(derived_new_blocks.size()))) {
                    return false;
                }
                derived_new_blocks.push_back(physical);
            }
            prepared.token_ids.push_back(spec.token_ids[local_row]);
            prepared.positions.push_back(static_cast<int64_t>(position));
            prepared.row_to_segment.push_back(
                static_cast<int32_t>(segment_index));
        }
        if (spec.first_new_block != derived_first_new_block ||
            spec.new_blocks != derived_new_blocks) {
            return false;
        }
        segment.first_new_block = spec.first_new_block;
        segment.new_blocks = spec.new_blocks;

        total_rows += spec.row_count;
        prepared.segments.push_back(std::move(segment));
    }

    prepared.row_to_public_output.assign(total_rows, -1);
    prepared.requested_full_logits.assign(total_rows, 0);
    for (size_t i = 0; i < prepared.segments.size(); ++i) {
        const int32_t output_row = prepared.segments[i].output_row;
        if (output_row < 0) continue;
        if (uint32_t(output_row) >= total_rows) return false;
        prepared.row_to_public_output[size_t(output_row)] =
            static_cast<int32_t>(prepared.public_output_rows.size());
        prepared.public_output_rows.push_back(output_row);
        if (specs[i].needs_full_logits) {
            prepared.requested_full_logits[size_t(output_row)] = 1;
        }
    }

    out = std::move(prepared);
    return true;
}

bool prepare_deepseek4_paged_layer_rows(
        const DeepSeek4PagedStepLayout & layout,
        uint32_t compression_ratio,
        std::vector<DeepSeek4LayerSegmentRows> & out) {
    if ((compression_ratio != 0 && compression_ratio != 4 &&
         compression_ratio != 128) ||
        layout.segments.empty() || !layout.block_table_stride ||
        !layout.slot_count || !layout.max_context || !layout.physical_blocks ||
        uint64_t(layout.block_table_stride) * DS4_PAGE_TOKENS <
            layout.max_context ||
        layout.token_ids.size() != layout.positions.size() ||
        layout.token_ids.size() != layout.row_to_segment.size() ||
        layout.token_ids.size() != layout.row_to_public_output.size() ||
        layout.token_ids.size() != layout.requested_full_logits.size() ||
        layout.token_ids.size() >
            size_t(DEEPSEEK4_MAX_PAGED_SEQUENCES)) {
        return false;
    }
    size_t expected_compact_size = 0;
    if (layout.segments.size() > UINT32_MAX ||
        !checked_table_size(static_cast<uint32_t>(layout.segments.size()),
                            layout.block_table_stride,
                            expected_compact_size) ||
        layout.compact_block_tables.size() != expected_compact_size) {
        return false;
    }

    std::vector<DeepSeek4LayerSegmentRows> prepared;
    prepared.reserve(layout.segments.size());
    uint32_t expected_offset = 0;
    size_t expected_public_output = 0;
    bool saw_prefill = false;
    std::unordered_set<int32_t> segment_slots;
    std::unordered_set<int32_t> physical_pages;
    for (size_t segment_index = 0; segment_index < layout.segments.size();
         ++segment_index) {
        const DeepSeek4PagedSegment & segment = layout.segments[segment_index];
        const bool decode =
            segment.kind == DeepSeek4PagedSegmentKind::decode;
        const bool prefill =
            segment.kind == DeepSeek4PagedSegmentKind::prefill;
        if ((!decode && !prefill) ||
            (decode && saw_prefill) ||
            segment.slot < 0 || uint32_t(segment.slot) >= layout.slot_count ||
            segment.row_offset != expected_offset ||
            !segment.row_count ||
            segment.row_count > DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS ||
            segment.start_position < 0 || segment.start_position > INT32_MAX ||
            uint64_t(segment.start_position) + segment.row_count >
                layout.max_context ||
            segment.phase4 != uint64_t(segment.start_position) % 4 ||
            segment.phase128 !=
                uint64_t(segment.start_position) % DS4_PAGE_TOKENS ||
            size_t(segment.row_offset) + segment.row_count >
                layout.positions.size() ||
            segment.compact_block_table_offset !=
                segment_index * size_t(layout.block_table_stride) ||
            size_t(segment.compact_block_table_offset) +
                    layout.block_table_stride >
                layout.compact_block_tables.size() ||
            segment.raw_ring_write_rows.size() != segment.row_count ||
            segment.pool_write_rows.size() != segment.row_count ||
            (segment.new_blocks.empty() !=
             (segment.first_new_block < 0))) {
            return false;
        }
        saw_prefill = saw_prefill || prefill;
        if (!segment_slots.insert(segment.slot).second) return false;
        const int32_t expected_output_row = decode
            ? int32_t(segment.row_offset)
            : (segment.prompt_tail
                ? int32_t(segment.row_offset + segment.row_count - 1) : -1);
        if ((decode && segment.prompt_tail) ||
            segment.output_row != expected_output_row) {
            return false;
        }
        if ((decode && segment.row_count != 1) ||
            (prefill && segment.row_count >
                            DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS)) {
            return false;
        }
        int32_t derived_first_new_block = -1;
        size_t derived_new_block_count = 0;
        for (uint32_t i = 0; i < segment.row_count; ++i) {
            const size_t row = size_t(segment.row_offset) + i;
            const uint64_t position =
                uint64_t(segment.start_position) + i;
            int32_t physical = -1;
            if (!segment_table_row(
                    layout, segment, position / DS4_PAGE_TOKENS,
                    physical)) {
                return false;
            }
            const uint64_t expected_pool_row =
                uint64_t(uint32_t(physical)) * DS4_PAGE_TOKENS +
                position % DS4_PAGE_TOKENS;
            int64_t expected_raw_row = -1;
            if (layout.row_to_segment[row] != int32_t(segment_index) ||
                layout.positions[row] != segment.start_position + i ||
                layout.token_ids[row] < 0 ||
                layout.requested_full_logits[row] > 1 ||
                expected_pool_row > uint64_t(INT64_MAX) ||
                segment.pool_write_rows[i] != int64_t(expected_pool_row) ||
                !flattened_raw_row(segment.slot, position,
                                   expected_raw_row) ||
                segment.raw_ring_write_rows[i] != expected_raw_row) {
                return false;
            }
            if (position % DS4_PAGE_TOKENS == 0) {
                const uint64_t logical = position / DS4_PAGE_TOKENS;
                if (logical > uint64_t(INT32_MAX)) return false;
                if (derived_first_new_block < 0) {
                    derived_first_new_block = int32_t(logical);
                }
                if (logical != uint64_t(derived_first_new_block) +
                                   derived_new_block_count ||
                    derived_new_block_count >= segment.new_blocks.size() ||
                    segment.new_blocks[derived_new_block_count] != physical) {
                    return false;
                }
                ++derived_new_block_count;
            }
            const bool is_output = int32_t(row) == expected_output_row;
            if (is_output) {
                if (expected_public_output >=
                        layout.public_output_rows.size() ||
                    layout.public_output_rows[expected_public_output] !=
                        int32_t(row) ||
                    layout.row_to_public_output[row] !=
                        int32_t(expected_public_output)) {
                    return false;
                }
                ++expected_public_output;
            } else if (layout.row_to_public_output[row] != -1 ||
                       layout.requested_full_logits[row]) {
                return false;
            }
        }
        if (segment.first_new_block != derived_first_new_block ||
            segment.new_blocks.size() != derived_new_block_count) {
            return false;
        }
        const uint64_t segment_end =
            uint64_t(segment.start_position) + segment.row_count - 1;
        for (uint64_t logical = 0;
             logical <= segment_end / DS4_PAGE_TOKENS; ++logical) {
            int32_t physical = -1;
            if (!segment_table_row(layout, segment, logical, physical) ||
                !physical_pages.insert(physical).second) {
                return false;
            }
        }
        const uint64_t start = uint64_t(segment.start_position);
        if (compression_ratio && segment.row_count >
                compression_ratio - start % compression_ratio) {
            return false;
        }

        DeepSeek4LayerSegmentRows rows;
        rows.slot = segment.slot;
        rows.compression_ratio = compression_ratio;
        rows.attention_state_slot = compression_ratio ? segment.slot : -1;
        rows.indexer_state_slot = compression_ratio == 4 ? segment.slot : -1;

        const uint64_t first_raw = start >= DS4_PAGE_TOKENS
            ? start - DS4_PAGE_TOKENS + 1 : 0;
        rows.immutable_raw_history.reserve(
            static_cast<size_t>(start - first_raw));
        for (uint64_t position = first_raw; position < start; ++position) {
            int64_t raw_row = -1;
            if (!flattened_raw_row(segment.slot, position, raw_row)) {
                return false;
            }
            rows.immutable_raw_history.push_back(raw_row);
        }

        if (compression_ratio) {
            const uint64_t completed = start / compression_ratio;
            rows.immutable_compressed_history.reserve(
                static_cast<size_t>(completed));
            for (uint64_t group = 0; group < completed; ++group) {
                const uint64_t end_token =
                    group * compression_ratio + compression_ratio - 1;
                int32_t physical = -1;
                if (!segment_table_row(layout, segment,
                                       end_token / DS4_PAGE_TOKENS,
                                       physical)) {
                    return false;
                }
                uint64_t compressed_row = 0;
                bool emitted = false;
                if (!ds4_compressed_page_row(
                        end_token, uint32_t(physical), compression_ratio,
                        compressed_row, emitted) || !emitted ||
                    compressed_row > uint64_t(INT64_MAX)) {
                    return false;
                }
                rows.immutable_compressed_history.push_back(
                    static_cast<int64_t>(compressed_row));
            }
        }

        rows.tokens.reserve(segment.row_count);
        for (uint32_t i = 0; i < segment.row_count; ++i) {
            const uint64_t position = start + i;
            DeepSeek4SegmentTokenRows token_rows;
            if (!flattened_raw_row(segment.slot, position,
                                   token_rows.raw_write_row) ||
                token_rows.raw_write_row != segment.raw_ring_write_rows[i]) {
                return false;
            }
            const size_t max_initial =
                DS4_PAGE_TOKENS - 1 - size_t(i);
            token_rows.initial_raw_suffix = static_cast<uint16_t>(
                std::min(rows.immutable_raw_history.size(), max_initial));
            token_rows.prior_segment_rows = static_cast<uint8_t>(i);
            if (compression_ratio) {
                const uint32_t phase = static_cast<uint32_t>(
                    position % compression_ratio);
                token_rows.attention_ape_phase = int32_t(phase);
                token_rows.attention_state_row = compression_ratio == 4
                    ? int64_t(compression_ratio + phase)
                    : int64_t(phase);
                if (compression_ratio == 4) {
                    token_rows.indexer_ape_phase = int32_t(phase);
                    token_rows.indexer_state_row =
                        int64_t(compression_ratio + phase);
                }
            }
            rows.tokens.push_back(token_rows);
        }

        if (compression_ratio) {
            const uint64_t tail = start + segment.row_count - 1;
            int32_t physical = -1;
            if (!segment_table_row(layout, segment,
                                   tail / DS4_PAGE_TOKENS, physical)) {
                return false;
            }
            uint64_t compressed_row = 0;
            bool emitted = false;
            if (!ds4_compressed_page_row(
                    tail, uint32_t(physical), compression_ratio,
                    compressed_row, emitted) ||
                compressed_row > uint64_t(INT64_MAX)) {
                return false;
            }
            if (emitted) {
                rows.emission_token = int32_t(segment.row_count - 1);
                rows.compressed_write_row = int64_t(compressed_row);
                const uint64_t compressed_position =
                    tail + 1 - compression_ratio;
                if (compressed_position > uint64_t(INT32_MAX)) return false;
                rows.compressed_position = int32_t(compressed_position);
                rows.tokens.back().sees_emitted_compressed_row = true;
            }
        }

        expected_offset += segment.row_count;
        prepared.push_back(std::move(rows));
    }
    if (expected_offset != layout.positions.size() ||
        expected_public_output != layout.public_output_rows.size()) {
        return false;
    }
    out = std::move(prepared);
    return true;
}

bool build_deepseek4_paged_segment_shape_key(
        const DeepSeek4PagedStepLayout & layout,
        const std::vector<std::vector<DeepSeek4LayerSegmentRows>> & layer_rows,
        bool token_id_mode,
        bool hybrid_mode,
        std::vector<int64_t> & out) {
    out.clear();
    const size_t q = layout.token_ids.size();
    const size_t segment_count = layout.segments.size();
    if (!q || q > size_t(DEEPSEEK4_MAX_PAGED_SEQUENCES) ||
        !segment_count || segment_count > q || layer_rows.empty() ||
        layer_rows.size() > size_t(INT64_MAX)) {
        return false;
    }

    bool saw_prefill = false;
    for (const DeepSeek4PagedSegment & segment : layout.segments) {
        saw_prefill = saw_prefill ||
            segment.kind == DeepSeek4PagedSegmentKind::prefill;
    }
    if (!saw_prefill) return false;

    std::array<std::vector<DeepSeek4LayerSegmentRows>, 3> canonical;
    std::array<bool, 3> prepared = {false, false, false};
    if (!prepare_deepseek4_paged_layer_rows(layout, 0, canonical[0])) {
        return false;
    }
    prepared[0] = true;
    for (const std::vector<DeepSeek4LayerSegmentRows> & layer : layer_rows) {
        if (layer.size() != segment_count) return false;
        const uint32_t ratio = layer.front().compression_ratio;
        const int index = ratio_index(ratio);
        if (index < 0) return false;
        if (!prepared[size_t(index)]) {
            if (!prepare_deepseek4_paged_layer_rows(
                    layout, ratio, canonical[size_t(index)])) {
                return false;
            }
            prepared[size_t(index)] = true;
        }
        if (!same_layer_rows(layer, canonical[size_t(index)])) {
            return false;
        }
    }

    const size_t per_layer = segment_count * 7 + q * 2;
    const size_t fixed = 5 + segment_count * 6;
    if (per_layer &&
        layer_rows.size() >
            (std::numeric_limits<size_t>::max() - fixed) / per_layer) {
        return false;
    }
    std::vector<int64_t> key;
    key.reserve(fixed + layer_rows.size() * per_layer);
    key.push_back(DEEPSEEK4_PAGED_SEGMENT_SHAPE_KEY_V1);
    key.push_back(static_cast<int64_t>(q));
    key.push_back(static_cast<int64_t>(segment_count));
    key.push_back(token_id_mode ? 1 : 0);
    key.push_back(hybrid_mode ? 1 : 0);
    for (const DeepSeek4PagedSegment & segment : layout.segments) {
        key.push_back(segment.slot);
        key.push_back(segment.row_offset);
        key.push_back(segment.row_count);
        key.push_back(segment.kind == DeepSeek4PagedSegmentKind::decode
                          ? 0 : 1);
        key.push_back(segment.phase4);
        key.push_back(segment.phase128);
    }

    for (const std::vector<DeepSeek4LayerSegmentRows> & layer : layer_rows) {
        for (size_t segment_index = 0; segment_index < segment_count;
             ++segment_index) {
            const DeepSeek4PagedSegment & segment =
                layout.segments[segment_index];
            const DeepSeek4LayerSegmentRows & rows = layer[segment_index];
            const uint32_t ratio = rows.compression_ratio;
            const bool emits = rows.emission_token >= 0;
            const int64_t raw_live = static_cast<int64_t>(
                rows.immutable_raw_history.size());
            const int64_t comp_live = static_cast<int64_t>(
                rows.immutable_compressed_history.size());
            const int64_t state_publish_rows =
                segment.kind == DeepSeek4PagedSegmentKind::prefill && ratio
                    ? (ratio == 4 ? 8 : 128) : 0;
            key.push_back(ratio);
            key.push_back(raw_live);
            key.push_back(std::max<int64_t>(1, raw_live));
            key.push_back(comp_live);
            key.push_back(std::max<int64_t>(1, comp_live));
            key.push_back(emits ? 1 : 0);
            key.push_back(state_publish_rows);
            for (const DeepSeek4SegmentTokenRows & token : rows.tokens) {
                key.push_back(token.initial_raw_suffix);
                key.push_back(token.prior_segment_rows);
            }
        }
    }
    out = std::move(key);
    return true;
}

} // namespace dflash::common
