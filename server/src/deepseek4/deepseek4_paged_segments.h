// Pure host-side layout for DeepSeek V4 paged service segments.
#pragma once

#include <cstdint>
#include <vector>

namespace dflash::common {

inline constexpr uint32_t DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS = 4;
inline constexpr int64_t DEEPSEEK4_PAGED_SEGMENT_SHAPE_KEY_V1 =
    0x4453345345470001LL;

enum class DeepSeek4PagedSegmentKind : uint8_t {
    decode,
    prefill,
};

// One caller-owned slot in a service step. Tokens and pool rows are in
// chronological order starting at start_position. pool_write_rows are the
// physical token rows returned by PagedKvPool::append().
struct DeepSeek4PagedSegmentSpec {
    DeepSeek4PagedSegmentKind kind = DeepSeek4PagedSegmentKind::prefill;
    int32_t slot = -1;
    int64_t start_position = -1;
    uint32_t row_count = 0;
    bool prompt_tail = false;
    bool needs_full_logits = false;
    std::vector<int32_t> token_ids;
    std::vector<int64_t> pool_write_rows;
    // Authoritative block-table delta returned by the staged pool append.
    int32_t first_new_block = -1;
    std::vector<int32_t> new_blocks;
};

// DS4-private lowering of a segment into the flattened model row axis.
struct DeepSeek4PagedSegment {
    DeepSeek4PagedSegmentKind kind = DeepSeek4PagedSegmentKind::prefill;
    int32_t slot = -1;
    uint32_t row_offset = 0;
    uint32_t row_count = 0;
    int64_t start_position = -1;
    bool prompt_tail = false;
    int32_t output_row = -1;
    uint8_t phase4 = 0;
    uint8_t phase128 = 0;
    uint32_t compact_block_table_offset = 0;
    std::vector<int64_t> pool_write_rows;
    std::vector<int64_t> raw_ring_write_rows;
    // The block-table delta implied by chronological append rows. With the
    // four-row cap this is normally empty or one entry, but keeping the
    // vector shape matches PagedKvPool and makes provenance explicit.
    int32_t first_new_block = -1;
    std::vector<int32_t> new_blocks;
};

// Per-token state and history geometry for one segment in one MLA layer.
// initial_raw_suffix is a count taken from the tail of immutable_raw_history.
// It is followed by prior_segment_rows F16 readbacks from this segment; only
// the query's own projected KV remains F32.
struct DeepSeek4SegmentTokenRows {
    int64_t raw_write_row = -1;
    uint16_t initial_raw_suffix = 0;
    uint8_t prior_segment_rows = 0;
    int32_t attention_ape_phase = -1;
    int64_t attention_state_row = -1;
    int32_t indexer_ape_phase = -1;
    int64_t indexer_state_row = -1;
    bool sees_emitted_compressed_row = false;
};

// All history is chronological. immutable_raw_history is captured before any
// raw-ring scatter for this segment. A safe segment has zero or one compressed
// emission, and an emission is always produced by its terminal token.
struct DeepSeek4LayerSegmentRows {
    int32_t slot = -1;
    uint32_t compression_ratio = 0;
    int32_t attention_state_slot = -1;
    int32_t indexer_state_slot = -1;
    std::vector<int64_t> immutable_raw_history;
    std::vector<int64_t> immutable_compressed_history;
    std::vector<DeepSeek4SegmentTokenRows> tokens;
    int32_t emission_token = -1;
    int64_t compressed_write_row = -1;
    int32_t compressed_position = -1;
};

struct DeepSeek4PagedStepLayout {
    uint32_t slot_count = 0;
    uint32_t max_context = 0;
    uint32_t block_table_stride = 0;
    uint32_t physical_blocks = 0;
    std::vector<DeepSeek4PagedSegment> segments;
    std::vector<int32_t> token_ids;
    std::vector<int64_t> positions;
    // Populated by the model's embedding lookup after pure geometry planning.
    std::vector<float> embeddings;
    std::vector<int32_t> row_to_segment;
    // Dynamic output metadata. public_output_rows contains only decode rows
    // and completed prompt tails; row_to_public_output maps a model row to its
    // index there, or -1. requested_full_logits is row-major and is never
    // graph-cache key material.
    std::vector<int32_t> public_output_rows;
    std::vector<int32_t> row_to_public_output;
    std::vector<uint8_t> requested_full_logits;
    // One full block-table column per segment owner, not per model row.
    std::vector<int32_t> compact_block_tables;
};

// Validate and lower a decode-first list of unique slot owners. Decode
// segments contain exactly one row, prefill segments contain one to four, and
// the entire step contains at most six rows. The source table is slot-major.
// max_context is the cache plan's exclusive logical-position bound. Every
// visible physical page must have exactly one segment owner.
bool prepare_deepseek4_paged_step_layout(
    const std::vector<DeepSeek4PagedSegmentSpec> & specs,
    const std::vector<int32_t> & slot_block_tables,
    uint32_t slot_count,
    uint32_t max_context,
    uint32_t block_table_stride,
    uint32_t physical_blocks,
    DeepSeek4PagedStepLayout & out);

// Derive the immutable histories, raw writes, compressor/indexer phases and
// optional terminal compressed destination for one layer. Ratios 0, 4 and 128
// are supported. A segment that passes the next compression boundary fails.
bool prepare_deepseek4_paged_layer_rows(
    const DeepSeek4PagedStepLayout & layout,
    uint32_t compression_ratio,
    std::vector<DeepSeek4LayerSegmentRows> & out);

// Build only the graph topology key. Token values, physical cache addresses,
// output selection, and other uploaded values intentionally do not fragment
// replay. layer_rows is layer-major and must describe every layout segment.
bool build_deepseek4_paged_segment_shape_key(
    const DeepSeek4PagedStepLayout & layout,
    const std::vector<std::vector<DeepSeek4LayerSegmentRows>> & layer_rows,
    bool token_id_mode,
    bool hybrid_mode,
    std::vector<int64_t> & out);

} // namespace dflash::common
