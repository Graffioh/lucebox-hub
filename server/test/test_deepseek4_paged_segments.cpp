#include "deepseek4/deepseek4_paged_segments.h"
#include "deepseek4/deepseek4_paged_cache.h"
#include "host_check.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

namespace {

std::vector<int32_t> block_tables(uint32_t slots, uint32_t stride) {
    return std::vector<int32_t>(size_t(slots) * stride, -1);
}

void set_table(std::vector<int32_t> & tables, uint32_t stride, int32_t slot,
               std::initializer_list<int32_t> physical) {
    size_t index = size_t(uint32_t(slot)) * stride;
    for (int32_t page : physical) tables[index++] = page;
}

DeepSeek4PagedSegmentSpec segment(
        DeepSeek4PagedSegmentKind kind,
        int32_t slot,
        int64_t start,
        uint32_t count,
        const std::vector<int32_t> & tables,
        uint32_t stride,
        bool prompt_tail = false,
        bool full_logits = false) {
    DeepSeek4PagedSegmentSpec spec;
    spec.kind = kind;
    spec.slot = slot;
    spec.start_position = start;
    spec.row_count = count;
    spec.prompt_tail = prompt_tail;
    spec.needs_full_logits = full_logits;
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t position = uint64_t(start) + i;
        const int32_t physical = tables[
            size_t(uint32_t(slot)) * stride + position / 128];
        spec.token_ids.push_back(1000 + int32_t(position));
        spec.pool_write_rows.push_back(
            int64_t(uint64_t(uint32_t(physical)) * 128 + position % 128));
        if (position % 128 == 0) {
            if (spec.first_new_block < 0) {
                spec.first_new_block = int32_t(position / 128);
            }
            spec.new_blocks.push_back(physical);
        }
    }
    return spec;
}

void check_singleton_parity(DeepSeek4PagedSegmentKind kind,
                            int64_t position,
                            uint32_t ratio) {
    std::vector<int32_t> tables = block_tables(3, 3);
    set_table(tables, 3, 2, {5, 1, 7});
    DeepSeek4PagedStepLayout layout;
    CHECK(prepare_deepseek4_paged_step_layout(
        {segment(kind, 2, position, 1, tables, 3)},
        tables, 3, 384, 3, 8, layout));

    std::vector<DeepSeek4LayerSegmentRows> segment_rows;
    CHECK(prepare_deepseek4_paged_layer_rows(
        layout, ratio, segment_rows));
    const int32_t slot = 2;
    std::vector<DeepSeek4GatheredLaneRows> gathered;
    CHECK(prepare_deepseek4_gathered_lane_rows(
        &slot, &position, 1, layout.compact_block_tables.data(),
        layout.block_table_stride, layout.physical_blocks, ratio, gathered));
    CHECK(gathered.size() == 1 && segment_rows.size() == 1);
    CHECK(segment_rows[0].immutable_raw_history == gathered[0].raw_history);
    CHECK(segment_rows[0].immutable_compressed_history ==
          gathered[0].compressed_history);
    CHECK(segment_rows[0].tokens[0].raw_write_row == gathered[0].raw_scatter);
    CHECK((segment_rows[0].emission_token == 0) ==
          gathered[0].compressed_emitted);
    CHECK(segment_rows[0].compressed_write_row ==
          gathered[0].compressed_scatter);
}

} // namespace

int main() {
    // A singleton segment is address-for-address identical to the established
    // gathered q=1 lane for decode and prefill, including fragmented pages.
    for (DeepSeek4PagedSegmentKind kind : {
             DeepSeek4PagedSegmentKind::decode,
             DeepSeek4PagedSegmentKind::prefill}) {
        for (int64_t position : {0LL, 3LL, 127LL, 259LL}) {
            for (uint32_t ratio : {0u, 4u, 128u}) {
                check_singleton_parity(kind, position, ratio);
            }
        }
    }

    // C1 prompt widths one through four retain chronological flattened rows.
    for (uint32_t width = 1; width <= 4; ++width) {
        std::vector<int32_t> tables = block_tables(1, 1);
        set_table(tables, 1, 0, {0});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, width,
                     tables, 1, true, width == 4)},
            tables, 1, 128, 1, 1, layout));
        CHECK(layout.segments.size() == 1);
        CHECK(layout.segments[0].row_offset == 0);
        CHECK(layout.segments[0].row_count == width);
        CHECK(layout.segments[0].output_row == int32_t(width - 1));
        CHECK(layout.segments[0].first_new_block == 0);
        CHECK(layout.segments[0].new_blocks == std::vector<int32_t>({0}));
        CHECK(layout.positions.size() == width);
        CHECK(layout.public_output_rows ==
              std::vector<int32_t>({int32_t(width - 1)}));
        CHECK(layout.requested_full_logits.back() == (width == 4));

        std::vector<DeepSeek4LayerSegmentRows> rows;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 4, rows));
        CHECK(rows.size() == 1 && rows[0].tokens.size() == width);
        for (uint32_t i = 0; i < width; ++i) {
            CHECK(rows[0].tokens[i].raw_write_row == int64_t(i));
            CHECK(rows[0].tokens[i].initial_raw_suffix == 0);
            CHECK(rows[0].tokens[i].prior_segment_rows == i);
            CHECK(rows[0].tokens[i].attention_ape_phase == int32_t(i));
            CHECK(rows[0].tokens[i].attention_state_row == int64_t(4 + i));
            CHECK(rows[0].tokens[i].indexer_ape_phase == int32_t(i));
            CHECK(rows[0].tokens[i].indexer_state_row == int64_t(4 + i));
        }
        CHECK(rows[0].emission_token == (width == 4 ? 3 : -1));
        CHECK(rows[0].compressed_write_row == (width == 4 ? 0 : -1));
        CHECK(rows[0].tokens.back().sees_emitted_compressed_row ==
              (width == 4));
    }

    // Two ragged prompt owners pack as 4+2 rows and compact one table each.
    {
        std::vector<int32_t> tables = block_tables(2, 1);
        set_table(tables, 1, 0, {3});
        set_table(tables, 1, 1, {7});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 4,
                        tables, 1),
                segment(DeepSeek4PagedSegmentKind::prefill, 1, 0, 2,
                        tables, 1),
            },
            tables, 2, 128, 1, 8, layout));
        CHECK(layout.token_ids.size() == 6);
        CHECK(layout.segments[0].row_offset == 0);
        CHECK(layout.segments[1].row_offset == 4);
        CHECK(layout.segments[0].compact_block_table_offset == 0);
        CHECK(layout.segments[1].compact_block_table_offset == 1);
        CHECK(layout.compact_block_tables == std::vector<int32_t>({3, 7}));
        CHECK(layout.row_to_segment ==
              std::vector<int32_t>({0, 0, 0, 0, 1, 1}));

        std::vector<DeepSeek4LayerSegmentRows> ratio4;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 4, ratio4));
        CHECK(ratio4.size() == 2);
        CHECK(ratio4[0].slot == 0 && ratio4[1].slot == 1);
        CHECK(ratio4[0].attention_state_slot == 0);
        CHECK(ratio4[1].attention_state_slot == 1);
        CHECK(ratio4[0].indexer_state_slot == 0);
        CHECK(ratio4[1].indexer_state_slot == 1);
        CHECK(ratio4[0].immutable_raw_history.empty());
        CHECK(ratio4[1].immutable_raw_history.empty());
        CHECK(ratio4[0].tokens.back().raw_write_row == 3);
        CHECK(ratio4[1].tokens.front().raw_write_row == 128);
        CHECK(ratio4[1].tokens.back().raw_write_row == 129);
        CHECK(ratio4[0].emission_token == 3);
        CHECK(ratio4[0].compressed_write_row == 96);
        CHECK(ratio4[1].emission_token == -1);

        std::vector<DeepSeek4LayerSegmentRows> ratio128;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 128, ratio128));
        CHECK(ratio128.size() == 2);
        CHECK(ratio128[0].attention_state_slot == 0);
        CHECK(ratio128[1].attention_state_slot == 1);
        CHECK(ratio128[0].indexer_state_slot == -1);
        CHECK(ratio128[1].indexer_state_slot == -1);
        CHECK(ratio128[0].emission_token == -1);
        CHECK(ratio128[1].emission_token == -1);
    }

    // Mixed service rows keep every decode first and map only the prompt tail
    // to a model output. Full-logit selection remains dynamic row metadata.
    {
        std::vector<int32_t> tables = block_tables(3, 1);
        set_table(tables, 1, 0, {0});
        set_table(tables, 1, 1, {1});
        set_table(tables, 1, 2, {2});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::decode, 0, 12, 1,
                        tables, 1, false, true),
                segment(DeepSeek4PagedSegmentKind::decode, 1, 9, 1,
                        tables, 1),
                segment(DeepSeek4PagedSegmentKind::prefill, 2, 20, 4,
                        tables, 1, true),
            },
            tables, 3, 128, 1, 3, layout));
        CHECK(layout.token_ids.size() == 6);
        CHECK(layout.public_output_rows ==
              std::vector<int32_t>({0, 1, 5}));
        CHECK(layout.row_to_public_output ==
              std::vector<int32_t>({0, 1, -1, -1, -1, 2}));
        CHECK(layout.requested_full_logits ==
              std::vector<uint8_t>({1, 0, 0, 0, 0, 0}));
        std::vector<DeepSeek4LayerSegmentRows> rows;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 0, rows));
        CHECK(rows.size() == 3);
        CHECK(rows[0].attention_state_slot == -1);
        CHECK(rows[2].tokens[3].prior_segment_rows == 3);

        std::vector<DeepSeek4LayerSegmentRows> ratio4;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 4, ratio4));
        CHECK(ratio4[0].tokens[0].attention_state_row == 4);
        CHECK(ratio4[1].tokens[0].attention_state_row == 5);
        CHECK(ratio4[2].emission_token == 3);
        CHECK(ratio4[2].compressed_write_row == 69);
        std::vector<DeepSeek4LayerSegmentRows> ratio128;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 128, ratio128));
        CHECK(ratio128[0].tokens[0].attention_state_row == 12);
        CHECK(ratio128[1].tokens[0].attention_state_row == 9);
        CHECK(ratio128[2].emission_token == -1);
    }

    // The existing service-width ceiling still accepts six independent
    // singleton owners without inventing padding or row aliases.
    {
        std::vector<int32_t> tables = block_tables(6, 1);
        std::vector<DeepSeek4PagedSegmentSpec> specs;
        for (int32_t slot = 0; slot < 6; ++slot) {
            set_table(tables, 1, slot, {slot});
            specs.push_back(segment(
                DeepSeek4PagedSegmentKind::decode, slot, 0, 1,
                tables, 1));
        }
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            specs, tables, 6, 128, 1, 6, layout));
        CHECK(layout.segments.size() == 6);
        CHECK(layout.public_output_rows ==
              std::vector<int32_t>({0, 1, 2, 3, 4, 5}));
        std::vector<DeepSeek4LayerSegmentRows> rows;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 4, rows));
        CHECK(rows.size() == 6);
    }

    // Snapshot geometry survives a raw-ring wrap. Each later query drops one
    // old snapshot row and reads one more predecessor through the F16 ring.
    {
        std::vector<int32_t> tables = block_tables(4, 2);
        set_table(tables, 2, 3, {4, 2});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 3, 127, 4,
                     tables, 2)},
            tables, 4, 256, 2, 5, layout));
        CHECK(layout.segments[0].raw_ring_write_rows ==
              std::vector<int64_t>({511, 384, 385, 386}));
        CHECK(layout.segments[0].first_new_block == 1);
        CHECK(layout.segments[0].new_blocks == std::vector<int32_t>({2}));
        std::vector<DeepSeek4LayerSegmentRows> rows;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 0, rows));
        CHECK(rows[0].immutable_raw_history.size() == 127);
        CHECK(rows[0].immutable_raw_history.front() == 384);
        CHECK(rows[0].immutable_raw_history.back() == 510);
        for (uint32_t i = 0; i < 4; ++i) {
            CHECK(rows[0].tokens[i].initial_raw_suffix == 127 - i);
            CHECK(rows[0].tokens[i].prior_segment_rows == i);
        }

        DeepSeek4PagedStepLayout before_page;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 3, 127, 1,
                     tables, 2)},
            tables, 4, 256, 2, 5, before_page));
        std::vector<DeepSeek4LayerSegmentRows> boundary4;
        CHECK(prepare_deepseek4_paged_layer_rows(
            before_page, 4, boundary4));
        CHECK(boundary4[0].compressed_write_row == 159);
        std::vector<DeepSeek4LayerSegmentRows> boundary128;
        CHECK(prepare_deepseek4_paged_layer_rows(
            before_page, 128, boundary128));
        CHECK(boundary128[0].compressed_write_row == 4);

        DeepSeek4PagedStepLayout after_page;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 3, 128, 4,
                     tables, 2)},
            tables, 4, 256, 2, 5, after_page));
        CHECK(after_page.segments[0].pool_write_rows ==
              std::vector<int64_t>({256, 257, 258, 259}));
        CHECK(after_page.segments[0].first_new_block == 1);
        CHECK(after_page.segments[0].new_blocks ==
              std::vector<int32_t>({2}));
        CHECK(prepare_deepseek4_paged_layer_rows(
            after_page, 4, boundary4));
        CHECK(boundary4[0].emission_token == 3);
        CHECK(boundary4[0].compressed_write_row == 64);
    }

    // A single safe tail can terminate both the ratio-4 and ratio-128 state
    // machines. The terminal compressed row is visible only to that query.
    {
        std::vector<int32_t> tables = block_tables(1, 1);
        set_table(tables, 1, 0, {5});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 0, 124, 4,
                     tables, 1)},
            tables, 1, 128, 1, 6, layout));
        std::vector<DeepSeek4LayerSegmentRows> ratio4;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 4, ratio4));
        CHECK(ratio4[0].immutable_compressed_history.size() == 31);
        CHECK(ratio4[0].immutable_compressed_history.front() == 160);
        CHECK(ratio4[0].immutable_compressed_history.back() == 190);
        CHECK(ratio4[0].emission_token == 3);
        CHECK(ratio4[0].compressed_write_row == 191);
        CHECK(ratio4[0].compressed_position == 124);
        CHECK(!ratio4[0].tokens[2].sees_emitted_compressed_row);
        CHECK(ratio4[0].tokens[3].sees_emitted_compressed_row);

        std::vector<DeepSeek4LayerSegmentRows> ratio128;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 128, ratio128));
        CHECK(ratio128[0].immutable_compressed_history.empty());
        CHECK(ratio128[0].emission_token == 3);
        CHECK(ratio128[0].compressed_write_row == 5);
        CHECK(ratio128[0].compressed_position == 0);
        CHECK(ratio128[0].tokens[0].attention_ape_phase == 124);
        CHECK(ratio128[0].tokens[3].attention_state_row == 127);
        CHECK(ratio128[0].tokens[3].indexer_ape_phase == -1);
        CHECK(ratio128[0].indexer_state_slot == -1);
    }

    // Compressed history follows fragmented logical pages rather than
    // assuming that physical rows are contiguous.
    {
        std::vector<int32_t> tables = block_tables(3, 3);
        set_table(tables, 3, 2, {5, 1, 7});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 2, 258, 2,
                     tables, 3)},
            tables, 3, 384, 3, 8, layout));
        std::vector<DeepSeek4LayerSegmentRows> rows;
        CHECK(prepare_deepseek4_paged_layer_rows(layout, 4, rows));
        CHECK(rows[0].immutable_raw_history.size() == 127);
        CHECK(rows[0].immutable_raw_history.front() == 259);
        CHECK(rows[0].immutable_raw_history.back() == 257);
        CHECK(rows[0].immutable_compressed_history.size() == 64);
        CHECK(rows[0].immutable_compressed_history.front() == 160);
        CHECK(rows[0].immutable_compressed_history[31] == 191);
        CHECK(rows[0].immutable_compressed_history[32] == 32);
        CHECK(rows[0].immutable_compressed_history.back() == 63);
        CHECK(rows[0].emission_token == 1);
        CHECK(rows[0].compressed_write_row == 224);
    }

    // Invalid ownership, shape, ordering, addressing, and unsafe compressor
    // spans fail before any graph or device state exists.
    {
        std::vector<int32_t> tables = block_tables(2, 1);
        set_table(tables, 1, 0, {0});
        set_table(tables, 1, 1, {1});
        DeepSeek4PagedStepLayout layout;
        CHECK(!prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 1,
                        tables, 1),
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 1, 1,
                        tables, 1),
            },
            tables, 2, 128, 1, 2, layout));

        std::vector<int32_t> aliased = {0, 0};
        CHECK(!prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 1,
                        aliased, 1),
                segment(DeepSeek4PagedSegmentKind::prefill, 1, 0, 1,
                        aliased, 1),
            },
            aliased, 2, 128, 1, 1, layout));

        CHECK(!prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 4,
                        tables, 1),
                segment(DeepSeek4PagedSegmentKind::prefill, 1, 0, 3,
                        tables, 1),
            },
            tables, 2, 128, 1, 2, layout));
        CHECK(!prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 1,
                        tables, 1),
                segment(DeepSeek4PagedSegmentKind::decode, 1, 0, 1,
                        tables, 1),
            },
            tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec overflow;
        overflow.slot = 0;
        overflow.start_position = std::numeric_limits<int64_t>::max();
        overflow.row_count = 1;
        overflow.token_ids = {1};
        overflow.pool_write_rows = {0};
        CHECK(!prepare_deepseek4_paged_step_layout(
            {overflow}, tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec too_wide = segment(
            DeepSeek4PagedSegmentKind::prefill, 0, 0, 4, tables, 1);
        too_wide.row_count = 5;
        too_wide.token_ids.push_back(9);
        too_wide.pool_write_rows.push_back(4);
        CHECK(!prepare_deepseek4_paged_step_layout(
            {too_wide}, tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec bad_decode = segment(
            DeepSeek4PagedSegmentKind::decode, 0, 0, 1, tables, 1);
        bad_decode.row_count = 2;
        bad_decode.token_ids.push_back(9);
        bad_decode.pool_write_rows.push_back(1);
        CHECK(!prepare_deepseek4_paged_step_layout(
            {bad_decode}, tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec bad_pool = segment(
            DeepSeek4PagedSegmentKind::prefill, 0, 0, 1, tables, 1);
        bad_pool.pool_write_rows[0] = 17;
        CHECK(!prepare_deepseek4_paged_step_layout(
            {bad_pool}, tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec bad_logits = segment(
            DeepSeek4PagedSegmentKind::prefill, 0, 0, 1, tables, 1,
            false, true);
        CHECK(!prepare_deepseek4_paged_step_layout(
            {bad_logits}, tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec missing_delta = segment(
            DeepSeek4PagedSegmentKind::prefill, 0, 0, 1, tables, 1);
        missing_delta.first_new_block = -1;
        missing_delta.new_blocks.clear();
        CHECK(!prepare_deepseek4_paged_step_layout(
            {missing_delta}, tables, 2, 128, 1, 2, layout));

        DeepSeek4PagedSegmentSpec wrong_delta = segment(
            DeepSeek4PagedSegmentKind::prefill, 0, 0, 1, tables, 1);
        wrong_delta.new_blocks[0] = 1;
        CHECK(!prepare_deepseek4_paged_step_layout(
            {wrong_delta}, tables, 2, 128, 1, 2, layout));

        CHECK(!prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 1,
                     tables, 1)},
            tables, 2, 128, 1, std::numeric_limits<uint32_t>::max(),
            layout));
    }

    // A valid layout can still be rejected per layer if it would cross a
    // compressor boundary instead of ending at the first one.
    {
        std::vector<int32_t> tables = block_tables(1, 1);
        set_table(tables, 1, 0, {0});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 0, 2, 3,
                     tables, 1)},
            tables, 1, 128, 1, 1, layout));
        std::vector<DeepSeek4LayerSegmentRows> rows;
        CHECK(!prepare_deepseek4_paged_layer_rows(layout, 4, rows));
        CHECK(!prepare_deepseek4_paged_layer_rows(layout, 16, rows));

        std::vector<int32_t> two_pages = block_tables(1, 2);
        set_table(two_pages, 2, 0, {0, 1});
        CHECK(prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 0, 126, 3,
                     two_pages, 2)},
            two_pages, 1, 256, 2, 2, layout));
        CHECK(!prepare_deepseek4_paged_layer_rows(layout, 128, rows));

        CHECK(!prepare_deepseek4_paged_step_layout(
            {segment(DeepSeek4PagedSegmentKind::prefill, 0, 129, 2,
                     two_pages, 2)},
            two_pages, 1, 130, 2, 2, layout));
    }

    // A non-tail prefill advances host state but has no model output; the
    // later tail owns exactly its final global row.
    {
        std::vector<int32_t> tables = block_tables(2, 1);
        set_table(tables, 1, 0, {0});
        set_table(tables, 1, 1, {1});
        DeepSeek4PagedStepLayout layout;
        CHECK(prepare_deepseek4_paged_step_layout(
            {
                segment(DeepSeek4PagedSegmentKind::prefill, 0, 0, 2,
                        tables, 1),
                segment(DeepSeek4PagedSegmentKind::prefill, 1, 8, 3,
                        tables, 1, true, true),
            },
            tables, 2, 128, 1, 2, layout));
        CHECK(layout.segments[0].output_row == -1);
        CHECK(layout.segments[1].output_row == 4);
        CHECK(layout.public_output_rows == std::vector<int32_t>({4}));
        CHECK(layout.row_to_public_output ==
              std::vector<int32_t>({-1, -1, -1, -1, 0}));
        CHECK(layout.requested_full_logits ==
              std::vector<uint8_t>({0, 0, 0, 0, 1}));

        std::vector<DeepSeek4LayerSegmentRows> rows;
        DeepSeek4PagedStepLayout stale = layout;
        stale.segments[1].phase4++;
        CHECK(!prepare_deepseek4_paged_layer_rows(stale, 4, rows));

        stale = layout;
        stale.segments[1].slot = stale.segments[0].slot;
        CHECK(!prepare_deepseek4_paged_layer_rows(stale, 4, rows));

        stale = layout;
        stale.segments[0].first_new_block = -1;
        stale.segments[0].new_blocks.clear();
        CHECK(!prepare_deepseek4_paged_layer_rows(stale, 4, rows));

        stale = layout;
        stale.compact_block_tables[1] = 0;
        stale.segments[1].pool_write_rows = {8, 9, 10};
        stale.segments[1].new_blocks.clear();
        stale.segments[1].first_new_block = -1;
        CHECK(!prepare_deepseek4_paged_layer_rows(stale, 4, rows));

        stale = layout;
        stale.public_output_rows[0] = 3;
        CHECK(!prepare_deepseek4_paged_layer_rows(stale, 4, rows));
    }

    std::printf("OK test_deepseek4_paged_segments (%d checks)\n", g_checks);
    return 0;
}
