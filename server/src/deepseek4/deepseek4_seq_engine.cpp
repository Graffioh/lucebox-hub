#include "deepseek4_seq_engine.h"

#include "deepseek4_backend.h"
#include "deepseek4_dspark.h"
#include "deepseek4_page_layout.h"
#include "deepseek4_paged_segments.h"
#include "common/adaptive_spec_width.h"
#include "common/sampler.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace dflash::common {
namespace {

enum class Ds4HostStageStatus : uint8_t {
    staged,
    row_failed,
    fatal,
};

struct Ds4HostStage {
    Ds4HostStageStatus status = Ds4HostStageStatus::fatal;
    int64_t start_position = -1;
    std::vector<int64_t> pool_write_rows;
    int32_t first_new_block = -1;
    std::vector<int32_t> new_blocks;
    std::string error;
};

class Ds4HostStepTransaction {
public:
    Ds4HostStepTransaction(
            SeqSlotManager & slots,
            std::vector<int32_t> & host_tables,
            uint32_t table_stride,
            uint32_t physical_blocks)
        : slots_(slots), host_tables_(host_tables),
          table_stride_(table_stride), physical_blocks_(physical_blocks) {
        appends_.reserve(DEEPSEEK4_MAX_PAGED_SEQUENCES);
        patches_.reserve(DEEPSEEK4_MAX_PAGED_SEQUENCES);
    }

    ~Ds4HostStepTransaction() {
        if (phase_ != Phase::armed) return;
        std::string error;
        if (!rollback(error)) {
            std::fprintf(stderr,
                "[deepseek4-parallel] host staging rollback failed: %s\n",
                error.c_str());
        }
    }

    Ds4HostStepTransaction(const Ds4HostStepTransaction &) = delete;
    Ds4HostStepTransaction & operator=(
        const Ds4HostStepTransaction &) = delete;
    Ds4HostStepTransaction(Ds4HostStepTransaction &&) = delete;
    Ds4HostStepTransaction & operator=(Ds4HostStepTransaction &&) = delete;

    Ds4HostStage stage_decode(const SeqEngine::StepInput & input) {
        return stage_decode_tokens(
            input.slot, std::vector<int32_t>{input.token},
            DeepSeek4PagedSegmentKind::decode);
    }

    Ds4HostStage stage_decode_tokens(
            int slot_id,
            const std::vector<int32_t> & tokens,
            DeepSeek4PagedSegmentKind kind =
                DeepSeek4PagedSegmentKind::speculative_decode) {
        Ds4HostStage stage;
        const bool ordinary = kind == DeepSeek4PagedSegmentKind::decode;
        const bool speculative =
            kind == DeepSeek4PagedSegmentKind::speculative_decode;
        if ((!ordinary && !speculative) ||
            (ordinary && tokens.size() != 1) ||
            (speculative &&
             (tokens.empty() ||
              tokens.size() > DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS)) ||
            std::any_of(tokens.begin(), tokens.end(),
                [](int32_t token) { return token < 0; }) ||
            !slots_.is_active(slot_id) || slots_.is_prefilling(slot_id)) {
            stage.status = Ds4HostStageStatus::row_failed;
            stage.error = "invalid DeepSeek4 decode staging slot";
            return stage;
        }

        const int previous_cur_pos = slots_.slot(slot_id).cur_pos;
        const SeqSlotManager::StepAppend append =
            slots_.append_tokens(
                slot_id, tokens.data(), static_cast<int>(tokens.size()));
        if (!append.ok) {
            stage.status = append.rollback_failed
                ? Ds4HostStageStatus::fatal
                : Ds4HostStageStatus::row_failed;
            stage.error = append.rollback_failed
                ? "DeepSeek4 decode allocator rollback failed"
                : append.busy
                ? "paged KV pool exhausted during DeepSeek4 decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "DeepSeek4 decode K/V append failed";
            return stage;
        }

        AppendRecord record;
        record.kind = kind;
        record.slot = slot_id;
        record.count = static_cast<int32_t>(tokens.size());
        record.previous_cur_pos = previous_cur_pos;
        record.first_patch = patches_.size();
        appends_.push_back(record);

        stage.start_position = append.position;
        stage.pool_write_rows = append.physical_rows;
        stage.first_new_block = append.first_new_block;
        stage.new_blocks = append.new_blocks;

        const SeqSlot & slot = slots_.slot(slot_id);
        const bool scalar_delta_matches = append.new_blocks.empty()
            ? append.new_block < 0 && append.new_block_index < 0
            : append.new_block == append.new_blocks.front() &&
              append.new_block_index == append.first_new_block;
        if (append.count != static_cast<int>(tokens.size()) ||
            append.position != previous_cur_pos ||
            append.physical_rows.size() != tokens.size() ||
            append.physical_row < 0 ||
            append.physical_row != append.physical_rows.front() ||
            std::any_of(append.physical_rows.begin(),
                append.physical_rows.end(),
                [](int64_t row) { return row < 0; }) ||
            slot.cur_pos != previous_cur_pos ||
            slot.staged_tokens != tokens ||
            !scalar_delta_matches) {
            std::string rollback_error;
            const bool rewound = rollback_last(rollback_error);
            stage.status = rewound
                ? Ds4HostStageStatus::row_failed
                : Ds4HostStageStatus::fatal;
            stage.error = rewound
                ? "malformed successful DeepSeek4 decode append"
                : "DeepSeek4 decode append rollback failed: " +
                      rollback_error;
            return stage;
        }

        std::string patch_error;
        if (!patch_last_append(append.first_new_block, append.new_blocks,
                               patch_error)) {
            std::string rollback_error;
            const bool rewound = rollback_last(rollback_error);
            stage.status = rewound
                ? Ds4HostStageStatus::row_failed
                : Ds4HostStageStatus::fatal;
            stage.error = rewound
                ? patch_error
                : "DeepSeek4 decode host-table rollback failed: " +
                      rollback_error;
            return stage;
        }

        stage.status = Ds4HostStageStatus::staged;
        return stage;
    }

    Ds4HostStage stage_prefill(int slot_id, int count) {
        Ds4HostStage stage;
        if (count < 1 || !slots_.is_prefilling(slot_id)) {
            stage.error = "invalid DeepSeek4 prefill staging request";
            return stage;
        }

        const int previous_cur_pos = slots_.slot(slot_id).cur_pos;
        SeqSlotManager::PrefillChunk chunk =
            slots_.append_prefill(slot_id, count);
        if (!chunk.ok) {
            stage.error = "DeepSeek4 prefill K/V append failed";
            return stage;
        }

        AppendRecord record;
        record.kind = DeepSeek4PagedSegmentKind::prefill;
        record.slot = slot_id;
        record.count = count;
        record.previous_cur_pos = previous_cur_pos;
        record.first_patch = patches_.size();
        appends_.push_back(record);

        stage.start_position = previous_cur_pos;
        stage.pool_write_rows = chunk.rows;
        stage.first_new_block = chunk.first_new_block;
        stage.new_blocks = chunk.new_blocks;

        const bool rows_valid = chunk.rows.size() == size_t(count) &&
            std::all_of(chunk.rows.begin(), chunk.rows.end(),
                [](int64_t row) { return row >= 0; });
        if (!rows_valid ||
            slots_.slot(slot_id).cur_pos != previous_cur_pos + count) {
            std::string rollback_error;
            const bool rewound = rollback_last(rollback_error);
            stage.error = rewound
                ? "malformed successful DeepSeek4 prefill append"
                : "DeepSeek4 prefill append rollback failed: " +
                      rollback_error;
            return stage;
        }

        std::string patch_error;
        if (!patch_last_append(chunk.first_new_block, chunk.new_blocks,
                               patch_error)) {
            std::string rollback_error;
            const bool rewound = rollback_last(rollback_error);
            stage.error = rewound
                ? patch_error
                : "DeepSeek4 prefill host-table rollback failed: " +
                      rollback_error;
            return stage;
        }

        stage.status = Ds4HostStageStatus::staged;
        return stage;
    }

    bool rollback(std::string & error) noexcept {
        if (phase_ != Phase::armed) {
            error = "DeepSeek4 host transaction crossed its mutation barrier";
            return false;
        }
        std::string first_error;
        while (!appends_.empty()) {
            std::string current_error;
            if (!rollback_last(current_error) && first_error.empty()) {
                first_error = current_error;
            }
        }
        phase_ = Phase::complete;
        error = std::move(first_error);
        return error.empty();
    }

    void cross_device_mutation_barrier() noexcept {
        assert(phase_ == Phase::armed);
        phase_ = Phase::submitted;
    }

    void rearm_after_device_restore() noexcept {
        assert(phase_ == Phase::submitted);
        phase_ = Phase::armed;
    }

    void finish_success() noexcept {
        assert(phase_ == Phase::submitted);
        appends_.clear();
        patches_.clear();
        phase_ = Phase::complete;
    }

private:
    enum class Phase : uint8_t { armed, submitted, complete };

    struct BlockPatch {
        size_t flat_index = 0;
        int32_t previous_physical = -1;
    };

    struct AppendRecord {
        DeepSeek4PagedSegmentKind kind =
            DeepSeek4PagedSegmentKind::prefill;
        int32_t slot = -1;
        int32_t count = 0;
        int32_t previous_cur_pos = -1;
        size_t first_patch = 0;
    };

    bool patch_last_append(
            int32_t first_new_block,
            const std::vector<int32_t> & new_blocks,
            std::string & error) {
        if (phase_ != Phase::armed || appends_.empty()) {
            error = "DeepSeek4 host-table patch has no staged append";
            return false;
        }
        AppendRecord & record = appends_.back();
        if (new_blocks.empty() != (first_new_block < 0)) {
            error = "inconsistent DeepSeek4 block-table delta";
            return false;
        }
        if (new_blocks.empty()) return true;
        if (first_new_block < 0 ||
            uint64_t(uint32_t(first_new_block)) + new_blocks.size() >
                table_stride_ ||
            record.slot < 0) {
            error = "DeepSeek4 block-table delta is out of range";
            return false;
        }

        const size_t base = size_t(uint32_t(record.slot)) * table_stride_;
        for (size_t i = 0; i < new_blocks.size(); ++i) {
            const int32_t physical = new_blocks[i];
            const size_t flat_index =
                base + size_t(uint32_t(first_new_block)) + i;
            if (physical < 0 || uint32_t(physical) >= physical_blocks_ ||
                flat_index >= host_tables_.size() ||
                host_tables_[flat_index] != -1) {
                error = "invalid DeepSeek4 block-table patch";
                return false;
            }
            patches_.push_back({flat_index, host_tables_[flat_index]});
            host_tables_[flat_index] = physical;
        }
        return true;
    }

    bool rollback_last(std::string & error) noexcept {
        if (appends_.empty()) {
            error = "DeepSeek4 host transaction has no append to roll back";
            return false;
        }
        const AppendRecord record = appends_.back();
        appends_.pop_back();

        bool ok = true;
        auto remember = [&](const std::string & message) {
            if (ok) error = message;
            ok = false;
        };

        if (record.first_patch > patches_.size()) {
            remember("invalid DeepSeek4 host patch rollback range");
        }
        const size_t begin = std::min(record.first_patch, patches_.size());
        const size_t end = patches_.size();
        for (size_t i = end; i > begin; --i) {
            const BlockPatch & patch = patches_[i - 1];
            if (patch.flat_index >= host_tables_.size()) {
                remember("invalid DeepSeek4 host patch index during rollback");
                continue;
            }
            host_tables_[patch.flat_index] = patch.previous_physical;
        }
        patches_.resize(begin);

        const bool allocator_rewound =
            record.kind == DeepSeek4PagedSegmentKind::prefill
                ? slots_.rollback_prefill(record.slot, record.count)
                : slots_.rollback_step(record.slot);
        if (!allocator_rewound) {
            remember("DeepSeek4 staged allocator rollback failed");
        }
        if (!slots_.is_active(record.slot) ||
            slots_.slot(record.slot).cur_pos != record.previous_cur_pos) {
            remember("DeepSeek4 slot position did not rewind exactly");
        }
        if (record.kind != DeepSeek4PagedSegmentKind::prefill &&
            slots_.is_active(record.slot) &&
            !slots_.slot(record.slot).staged_tokens.empty()) {
            remember("DeepSeek4 decode tokens remained staged after rollback");
        }
        return ok;
    }

    SeqSlotManager & slots_;
    std::vector<int32_t> & host_tables_;
    uint32_t table_stride_ = 0;
    uint32_t physical_blocks_ = 0;
    std::vector<BlockPatch> patches_;
    std::vector<AppendRecord> appends_;
    Phase phase_ = Phase::armed;
};

struct Ds4PagedSnapshotSpan {
    ggml_tensor * tensor = nullptr;
    size_t offset = 0;
    std::vector<uint8_t> bytes;
};

struct Ds4PagedSpecSnapshot {
    std::vector<Ds4PagedSnapshotSpan> spans;
};

bool snapshot_paged_span(
        ggml_tensor * tensor, size_t offset, size_t bytes,
        Ds4PagedSpecSnapshot & snapshot, std::string & error) {
    if (!tensor || !bytes || offset > ggml_nbytes(tensor) ||
        bytes > ggml_nbytes(tensor) - offset) {
        error = "DeepSeek4 speculative snapshot span is out of range";
        return false;
    }
    for (const Ds4PagedSnapshotSpan & existing : snapshot.spans) {
        if (existing.tensor == tensor && existing.offset == offset &&
            existing.bytes.size() == bytes) {
            return true;
        }
    }
    Ds4PagedSnapshotSpan span;
    span.tensor = tensor;
    span.offset = offset;
    span.bytes.resize(bytes);
    ggml_backend_tensor_get(
        tensor, span.bytes.data(), offset, span.bytes.size());
    snapshot.spans.push_back(std::move(span));
    return true;
}

bool snapshot_paged_row(
        ggml_tensor * tensor, int64_t row, size_t base_offset,
        Ds4PagedSpecSnapshot & snapshot, std::string & error) {
    if (!tensor || row < 0 || row >= tensor->ne[1]) {
        error = "DeepSeek4 speculative snapshot row is out of range";
        return false;
    }
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    if (size_t(row) >
        (std::numeric_limits<size_t>::max() - base_offset) / tensor->nb[1]) {
        error = "DeepSeek4 speculative snapshot offset overflow";
        return false;
    }
    return snapshot_paged_span(
        tensor, base_offset + size_t(row) * tensor->nb[1], row_bytes,
        snapshot, error);
}

bool save_paged_spec_snapshot(
        const DeepSeek4PagedCache & cache,
        const DeepSeek4PagedStepLayout & layout,
        Ds4PagedSpecSnapshot & snapshot,
        std::string & error) {
    snapshot.spans.clear();
    if (cache.layers.size() != cache.plan.ratios.size()) {
        error = "DeepSeek4 speculative snapshot cache geometry is invalid";
        return false;
    }
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const DeepSeek4PagedLayerCache & layer = cache.layers[il];
        std::vector<DeepSeek4LayerSegmentRows> layer_rows;
        if (!prepare_deepseek4_paged_layer_rows(
                layout, cache.plan.ratios[il], layer_rows) ||
            layer_rows.size() != layout.segments.size()) {
            error = "DeepSeek4 speculative snapshot layer layout failed";
            return false;
        }
        for (size_t segment_index = 0;
             segment_index < layout.segments.size(); ++segment_index) {
            const DeepSeek4PagedSegment & segment =
                layout.segments[segment_index];
            const DeepSeek4LayerSegmentRows & rows =
                layer_rows[segment_index];
            if (segment.kind !=
                    DeepSeek4PagedSegmentKind::speculative_decode ||
                segment.slot < 0 ||
                rows.tokens.size() != segment.row_count || !layer.raw_kv) {
                error = "DeepSeek4 speculative snapshot received a non-candidate segment";
                return false;
            }

            for (const DeepSeek4SegmentTokenRows & token : rows.tokens) {
                if (token.raw_write_row < 0 ||
                    token.raw_write_row / int64_t(DS4_PAGE_TOKENS) !=
                        segment.slot) {
                    error = "DeepSeek4 speculative raw snapshot owner is invalid";
                    return false;
                }
                if (!snapshot_paged_row(
                        layer.raw_kv,
                        token.raw_write_row % int64_t(DS4_PAGE_TOKENS),
                        size_t(segment.slot) * layer.raw_kv->nb[2],
                        snapshot, error)) {
                    return false;
                }
            }
            if (!rows.compression_ratio) continue;

            std::vector<int64_t> state_rows;
            if (rows.compression_ratio == 4 && rows.emission_token >= 0) {
                state_rows = {0, 1, 2, 3, 4, 5, 6, 7};
            } else {
                state_rows.reserve(rows.tokens.size());
                for (const DeepSeek4SegmentTokenRows & token : rows.tokens) {
                    state_rows.push_back(token.attention_state_row);
                }
            }
            auto snapshot_state = [&](ggml_tensor * tensor) {
                if (!tensor || segment.slot >= tensor->ne[2]) {
                    error = "DeepSeek4 speculative state snapshot is out of range";
                    return false;
                }
                for (int64_t row : state_rows) {
                    if (!snapshot_paged_row(
                            tensor, row,
                            size_t(segment.slot) * tensor->nb[2],
                            snapshot, error)) {
                        return false;
                    }
                }
                return true;
            };
            if (!snapshot_state(layer.attn_compressor.state_kv) ||
                !snapshot_state(layer.attn_compressor.state_score) ||
                (rows.compression_ratio == 4 &&
                 (!snapshot_state(layer.indexer_compressor.state_kv) ||
                  !snapshot_state(layer.indexer_compressor.state_score)))) {
                return false;
            }
            if (rows.emission_token >= 0 &&
                (!snapshot_paged_row(
                     layer.comp_kv, rows.compressed_write_row, 0,
                     snapshot, error) ||
                 (rows.compression_ratio == 4 &&
                  !snapshot_paged_row(
                      layer.index_comp_kv, rows.compressed_write_row, 0,
                      snapshot, error)))) {
                return false;
            }
        }
    }
    return true;
}

bool restore_paged_spec_snapshot(
        const Ds4PagedSpecSnapshot & snapshot,
        std::string & error) {
    for (auto it = snapshot.spans.rbegin();
         it != snapshot.spans.rend(); ++it) {
        if (!it->tensor || it->bytes.empty() ||
            it->offset > ggml_nbytes(it->tensor) ||
            it->bytes.size() > ggml_nbytes(it->tensor) - it->offset) {
            error = "DeepSeek4 speculative restore span is out of range";
            return false;
        }
        ggml_backend_tensor_set(
            it->tensor, it->bytes.data(), it->offset, it->bytes.size());
    }
    return true;
}

} // namespace

DeepSeek4SeqEngine::DeepSeek4SeqEngine(
        DeepSeek4Backend & backend, PagedKvPool & pool, int max_ctx,
        uint32_t table_stride)
    : b_(backend), slots_(pool, max_ctx), stride_(table_stride),
      host_tables_((size_t)pool.max_sequences() * table_stride, -1),
      dspark_slots_(size_t(pool.max_sequences())) {
    bool adaptive = adaptive_spec_width_globally_enabled();
    if (const char * raw = std::getenv("DFLASH_DS4_ADAPTIVE_WIDTH")) {
        adaptive = raw[0] && std::strcmp(raw, "0") != 0;
    }
    const int max_width = paged_dspark_max_width();
    for (DsparkSlotState & state : dspark_slots_) {
        state.width = std::make_unique<AdaptiveSpecWidth>(
            max_width, 2, adaptive);
    }
}

bool DeepSeek4SeqEngine::paged_dspark_enabled() const {
    const char * value = std::getenv("DFLASH_DS4_PAGED_SPEC");
    return value && value[0] && std::strcmp(value, "0") != 0 &&
        b_.spec_enabled_ && b_.spec_drafter_ &&
        b_.spec_drafter_->dspark_enabled;
}

int DeepSeek4SeqEngine::paged_dspark_max_width() const {
    int cap = 4;
    const char * q5 = std::getenv("DFLASH_DS4_Q5_VERIFY");
    if (q5 && q5[0] == '1' && q5[1] == '\0') cap = 5;
    if (const char * value = std::getenv("DFLASH_DS4_SPEC_Q")) {
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && end && *end == '\0' && parsed >= 2 &&
            parsed <= cap) {
            cap = static_cast<int>(parsed);
        }
    }
    return cap;
}

int DeepSeek4SeqEngine::dspark_feature_width() const {
    if (!b_.spec_drafter_ || b_.w_.n_embd <= 0 ||
        b_.spec_drafter_->n_target_layers <= 0 ||
        b_.spec_drafter_->n_target_layers >
            std::numeric_limits<int>::max() / b_.w_.n_embd) {
        return 0;
    }
    return b_.spec_drafter_->n_target_layers * b_.w_.n_embd;
}

void DeepSeek4SeqEngine::append_dspark_feature(
        int slot, const float * feature) {
    const int width = dspark_feature_width();
    if (!feature || width <= 0 || b_.w_.n_swa <= 0 || slot < 0 ||
        size_t(slot) >= dspark_slots_.size()) {
        return;
    }
    std::vector<float> & ring = dspark_slots_[size_t(slot)].features;
    if (ring.size() % size_t(width) != 0) ring.clear();
    const size_t rows = ring.size() / size_t(width);
    if (rows >= size_t(b_.w_.n_swa)) {
        std::memmove(
            ring.data(), ring.data() + width,
            (ring.size() - size_t(width)) * sizeof(float));
        ring.resize(ring.size() - size_t(width));
    }
    ring.insert(ring.end(), feature, feature + width);
}

bool DeepSeek4SeqEngine::token_is_eos(int32_t token) const {
    return deepseek4_is_eos_tok(token, b_.w_);
}

StepPlanLimits DeepSeek4SeqEngine::step_plan_limits(
        int decode_rows) const {
    const int rows = std::max(
        0, DEEPSEEK4_MAX_PAGED_SEQUENCES - decode_rows);
    return {
        rows,
        std::min(int(DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS), rows),
        rows,
        1,
    };
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    using AdmitStatus = AdmitResult::Status;
    AdmitResult result = slots_.admit(
        request_id, prompt, sampler);
    if (result.status != AdmitStatus::admitted) return result;
    if (result.slot < 0 || result.slot >= slots_.slot_count()) {
        result.status = AdmitStatus::failed;
        result.error = "invalid DeepSeek4 serving slot";
        return result;
    }
    std::fill_n(host_tables_.data() + (size_t)result.slot * stride_,
                stride_, -1);
    DsparkSlotState & dspark = dspark_slots_[size_t(result.slot)];
    dspark.features.clear();
    if (dspark.width) dspark.width->reset();
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)result.slot);
    return result;
}

bool DeepSeek4SeqEngine::set_block(int slot, int logical, int32_t physical) {
    if (slot < 0 || slot >= slots_.slot_count() || logical < 0 ||
        (uint32_t) logical >= stride_) return false;
    host_tables_[(size_t)slot * stride_ + (size_t)logical] = physical;
    return true;
}

SeqEngine::StepResult DeepSeek4SeqEngine::step(const StepPlan & plan) {
    StepResult result;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    if (inputs.size() > size_t(DEEPSEEK4_MAX_PAGED_SEQUENCES) ||
        int(inputs.size()) != slots_.decoding_count()) {
        return fail_step("decode plan does not cover every live DeepSeek4 slot");
    }
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & input : inputs) {
        if (input.slot < 0 || input.slot >= n_slots || input.token < 0 ||
            decode_seen[(size_t)input.slot] ||
            !slots_.is_active(input.slot) ||
            slots_.is_prefilling(input.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 decode row");
        }
        decode_seen[(size_t)input.slot] = 1;
    }

    const StepPlanLimits limits = step_plan_limits((int)inputs.size());
    if (limits.max_prefill_sequences < 0 ||
        plan.prefills.size() > size_t(limits.max_prefill_sequences)) {
        return fail_step("DeepSeek4 step exceeds the six-lane graph");
    }
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    int64_t planned_prefill_tokens = 0;
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 prefill segment");
        }
        prefill_seen[(size_t)slice.slot] = 1;
        planned_prefill_tokens += slice.max_tokens;
        if (planned_prefill_tokens > limits.max_prefill_tokens_total) {
            return fail_step("DeepSeek4 prefill plan exceeds its row budget");
        }
    }
    if (inputs.empty() && plan.prefills.empty()) return result;

    const DeepSeek4PagedCache & cache = b_.paged_cache_;
    size_t expected_host_table_size = 0;
    const bool host_table_size_valid = cache.plan.slots && stride_ &&
        size_t(cache.plan.slots) <=
            std::numeric_limits<size_t>::max() / size_t(stride_);
    if (host_table_size_valid) {
        expected_host_table_size = size_t(cache.plan.slots) * stride_;
    }
    if (!host_table_size_valid ||
        host_tables_.size() != expected_host_table_size ||
        cache.plan.slots != uint32_t(n_slots) ||
        cache.plan.max_ctx != uint32_t(slots_.max_context()) ||
        cache.plan.max_blocks_per_sequence != stride_ ||
        !cache.plan.physical_blocks || !cache.pool ||
        cache.pool->max_sequences() != cache.plan.slots ||
        cache.pool->physical_block_count() != cache.plan.physical_blocks ||
        b_.w_.n_layer <= 0 ||
        b_.w_.compress_ratios.size() != size_t(b_.w_.n_layer) ||
        cache.plan.ratios.size() != size_t(b_.w_.n_layer) ||
        cache.layers.size() != size_t(b_.w_.n_layer)) {
        return fail_step("invalid DeepSeek4 paged serving geometry");
    }
    for (size_t layer = 0; layer < size_t(b_.w_.n_layer); ++layer) {
        if (b_.w_.compress_ratios[layer] != cache.plan.ratios[layer] ||
            cache.layers[layer].ratio != cache.plan.ratios[layer]) {
            return fail_step("DeepSeek4 paged compression ratios disagree");
        }
    }

    if (plan.prefills.empty()) {
        return paged_dspark_enabled()
            ? step_decode_with_dspark(inputs)
            : step_decode_only(inputs);
    }
    return step_with_paged_segments(plan);
}

SeqEngine::StepResult DeepSeek4SeqEngine::step_decode_only(
        const std::vector<StepInput> & inputs,
        const std::vector<int> * capture_layer_ids,
        std::vector<float> * capture) {
    StepResult result;
    if (capture) capture->clear();
    auto fail_step = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    std::vector<int32_t> lane_tokens;
    std::vector<int64_t> lane_positions;
    std::vector<int32_t> lane_slots;
    std::vector<int> decode_lanes;
    lane_tokens.reserve(inputs.size());
    lane_positions.reserve(inputs.size());
    lane_slots.reserve(inputs.size());
    decode_lanes.reserve(inputs.size());
    result.decode.reserve(inputs.size());

    for (const StepInput & input : inputs) {
        DecodeOutput out;
        out.slot = input.slot;
        const SeqSlotManager::StepAppend append =
            slots_.append_token(input.slot, input.token);
        if (!append.ok) {
            if (append.rollback_failed) {
                return fail_step(
                    "DeepSeek4 decode allocator rollback failed");
            }
            out.failed = true;
            out.error = append.busy
                ? "paged KV pool exhausted during DeepSeek4 decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "DeepSeek4 decode K/V append failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        if (append.new_block >= 0 &&
            !set_block(input.slot, append.new_block_index,
                       append.new_block)) {
            out.failed = true;
            out.error = "DeepSeek4 decode block-table update failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        decode_lanes.push_back((int)lane_tokens.size());
        lane_tokens.push_back(input.token);
        lane_positions.push_back(append.position);
        lane_slots.push_back(input.slot);
        result.decode.push_back(std::move(out));
    }

    if (lane_tokens.empty()) return result;
    if (lane_tokens.size() > DEEPSEEK4_MAX_PAGED_SEQUENCES) {
        return fail_step("DeepSeek4 gathered step exceeds six lanes");
    }

    std::vector<float> embeddings(
        (size_t)b_.w_.n_embd * lane_tokens.size());
    if (!b_.w_.embedder.embed(lane_tokens.data(), (int)lane_tokens.size(),
                              embeddings.data())) {
        return fail_step("DeepSeek4 token embedding failed");
    }

    std::vector<int32_t> compact_tables(
        lane_tokens.size() * stride_, -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane) {
        std::copy_n(
            host_tables_.data() + (size_t)lane_slots[lane] * stride_,
            stride_, compact_tables.data() + lane * stride_);
    }

    std::vector<uint8_t> logit_lanes(lane_tokens.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane >= 0 &&
            slots_.slot(inputs[i].slot).sampler.needs_logit_processing()) {
            logit_lanes[(size_t) lane] = 1;
        }
    }
    std::vector<float> logits;
    std::vector<int32_t> argmax;
    if (!deepseek4_paged_gathered_step(
            b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
            embeddings.data(), lane_tokens.data(), lane_positions.data(),
            lane_slots.data(), (uint32_t)lane_tokens.size(),
            compact_tables.data(), stride_, logit_lanes.data(), logits, argmax,
            b_.moe_hybrid_.get(), b_.routing_stats_.get(),
            capture_layer_ids, capture)) {
        return fail_step("DeepSeek4 gathered paged graph failed");
    }

    auto sample_lane = [&](int slot_id, int lane) {
        SeqSlot & slot = slots_.slot(slot_id);
        if (!slot.sampler.needs_logit_processing()) {
            return argmax[(size_t)lane];
        }
        return sample_logits(
            logits.data() + (size_t)lane * b_.w_.n_vocab,
            b_.w_.n_vocab, slot.sampler, slot.sample_history, slot.rng);
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane < 0) continue;
        DecodeOutput & out = result.decode[i];
        slots_.commit_step(out.slot);
        out.token = sample_lane(out.slot, lane);
    }
    return result;
}

SeqEngine::StepResult DeepSeek4SeqEngine::step_decode_with_dspark(
        const std::vector<StepInput> & inputs) {
    struct DraftLane {
        std::vector<int32_t> proposal;
        int candidate_cap = 0;
        int allocated = 0;
        int accepted = 0;
    };
    struct CandidateLane {
        size_t input_index = 0;
        int slot = -1;
        std::vector<int32_t> tokens;
    };
    struct CandidateExecution {
        DeepSeek4PagedStepLayout layout;
        Ds4PagedSpecSnapshot snapshot;
        std::vector<int32_t> argmax;
        std::vector<float> capture;
        bool submitted = false;
        bool fatal = false;
    };

    const DSparkDrafter * drafter = b_.spec_drafter_.get();
    const int feature_width = dspark_feature_width();
    if (!drafter || feature_width <= 0 ||
        drafter->capture_layer_ids.empty()) {
        return step_decode_only(inputs);
    }

    std::vector<DraftLane> drafts(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const StepInput & input = inputs[i];
        const SeqSlot & slot = slots_.slot(input.slot);
        DsparkSlotState & state = dspark_slots_[size_t(input.slot)];
        if (!input.allow_speculation ||
            slot.sampler.needs_logit_processing() ||
            !state.width || state.features.empty() ||
            state.features.size() % size_t(feature_width) != 0 ||
            slot.cur_pos >= slots_.max_context()) {
            continue;
        }
        const int context_tokens = static_cast<int>(
            state.features.size() / size_t(feature_width));
        const int proposal_width = state.width->next_width(
            paged_dspark_max_width());
        if (!deepseek4_dspark_propose_chain(
                b_.backend_, b_.w_, *drafter, state.features.data(),
                context_tokens, slot.cur_pos, input.token, proposal_width,
                drafts[i].proposal)) {
            drafts[i].proposal.clear();
        }
    }

    std::vector<float> root_capture;
    StepResult root = step_decode_only(
        inputs, &drafter->capture_layer_ids, &root_capture);
    if (!root.error.empty()) return root;

    size_t successful_roots = 0;
    for (const DecodeOutput & output : root.decode) {
        successful_roots += output.failed ? 0 : 1;
    }
    if (root_capture.size() != successful_roots * size_t(feature_width)) {
        root.decode.clear();
        root.prefills.clear();
        root.error = "DeepSeek4 paged DSpark root capture is malformed";
        return root;
    }
    size_t capture_row = 0;
    for (const DecodeOutput & output : root.decode) {
        if (output.failed) continue;
        append_dspark_feature(
            output.slot,
            root_capture.data() + capture_row * size_t(feature_width));
        ++capture_row;
    }

    int candidate_budget = std::max(
        0, int(DEEPSEEK4_MAX_PAGED_SEQUENCES) -
               static_cast<int>(successful_roots));
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i >= root.decode.size() || root.decode[i].failed ||
            drafts[i].proposal.size() < 2 ||
            drafts[i].proposal[1] != root.decode[i].token) {
            continue;
        }
        const SeqSlot & slot = slots_.slot(inputs[i].slot);
        int desired = std::min({
            int(drafts[i].proposal.size()) - 1,
            int(DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS),
            slots_.max_context() - slot.cur_pos,
        });
        for (int candidate = 0; candidate < desired; ++candidate) {
            if (token_is_eos(drafts[i].proposal[size_t(candidate + 1)])) {
                desired = candidate + 1;
                break;
            }
        }
        drafts[i].candidate_cap = desired > 0
            ? deepseek4_safe_compressor_batch_tokens(
                  b_.w_, slot.cur_pos, desired)
            : 0;
    }
    while (candidate_budget > 0) {
        bool granted = false;
        for (DraftLane & draft : drafts) {
            if (draft.allocated >= draft.candidate_cap) continue;
            ++draft.allocated;
            --candidate_budget;
            granted = true;
            if (!candidate_budget) break;
        }
        if (!granted) break;
    }

    std::vector<CandidateLane> offered;
    for (size_t i = 0; i < drafts.size(); ++i) {
        if (drafts[i].allocated < 1) continue;
        CandidateLane lane;
        lane.input_index = i;
        lane.slot = inputs[i].slot;
        lane.tokens.assign(
            drafts[i].proposal.begin() + 1,
            drafts[i].proposal.begin() + 1 + drafts[i].allocated);
        offered.push_back(std::move(lane));
    }
    if (offered.empty()) return root;

    auto execute_candidates = [&](const std::vector<CandidateLane> & lanes,
                                  Ds4HostStepTransaction & transaction,
                                  CandidateExecution & execution,
                                  std::string & error) {
        execution = {};
        std::vector<DeepSeek4PagedSegmentSpec> specs;
        specs.reserve(lanes.size());
        for (const CandidateLane & lane : lanes) {
            Ds4HostStage stage = transaction.stage_decode_tokens(
                lane.slot, lane.tokens);
            if (stage.status != Ds4HostStageStatus::staged) {
                execution.fatal =
                    stage.status == Ds4HostStageStatus::fatal;
                error = stage.error.empty()
                    ? "DeepSeek4 speculative host staging failed"
                    : stage.error;
                return false;
            }
            DeepSeek4PagedSegmentSpec spec;
            spec.kind = DeepSeek4PagedSegmentKind::speculative_decode;
            spec.slot = lane.slot;
            spec.start_position = stage.start_position;
            spec.row_count = static_cast<uint32_t>(lane.tokens.size());
            spec.token_ids = lane.tokens;
            spec.pool_write_rows = std::move(stage.pool_write_rows);
            spec.first_new_block = stage.first_new_block;
            spec.new_blocks = std::move(stage.new_blocks);
            specs.push_back(std::move(spec));
        }
        if (!prepare_deepseek4_paged_step_layout(
                specs, host_tables_, b_.paged_cache_.plan.slots,
                b_.paged_cache_.plan.max_ctx,
                b_.paged_cache_.plan.max_blocks_per_sequence,
                b_.paged_cache_.plan.physical_blocks, execution.layout)) {
            error = "DeepSeek4 speculative segment layout failed";
            return false;
        }
        const size_t q = execution.layout.token_ids.size();
        if (q < 1 || q > DEEPSEEK4_MAX_PAGED_SEQUENCES ||
            q > std::numeric_limits<size_t>::max() / size_t(b_.w_.n_embd)) {
            error = "DeepSeek4 speculative segment width is invalid";
            return false;
        }
        execution.layout.embeddings.resize(q * size_t(b_.w_.n_embd));
        if (!b_.w_.embedder.embed(
                execution.layout.token_ids.data(), static_cast<int>(q),
                execution.layout.embeddings.data())) {
            error = "DeepSeek4 speculative embedding failed";
            return false;
        }

        DeepSeek4PagedSegmentPreparedStep * raw_prepared = nullptr;
        if (!deepseek4_prepare_paged_segment_step(
                b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
                execution.layout, b_.moe_hybrid_.get(),
                raw_prepared, &drafter->capture_layer_ids) ||
            !raw_prepared) {
            if (raw_prepared) {
                deepseek4_discard_paged_segment_step(raw_prepared);
            }
            error = "DeepSeek4 speculative graph prepare failed";
            return false;
        }
        using PreparedPtr = std::unique_ptr<
            DeepSeek4PagedSegmentPreparedStep,
            void (*)(DeepSeek4PagedSegmentPreparedStep *)>;
        PreparedPtr prepared(
            raw_prepared, deepseek4_discard_paged_segment_step);
        if (!save_paged_spec_snapshot(
                b_.paged_cache_, execution.layout,
                execution.snapshot, error)) {
            return false;
        }

        transaction.cross_device_mutation_barrier();
        execution.submitted = true;
        std::vector<float> logits;
        if (!deepseek4_compute_paged_segment_step(
                *prepared, logits, execution.argmax,
                nullptr, &execution.capture)) {
            error = "DeepSeek4 speculative graph failed";
            return false;
        }
        if (!logits.empty() || execution.argmax.size() != q ||
            execution.capture.size() != q * size_t(feature_width)) {
            error = "DeepSeek4 speculative outputs are malformed";
            return false;
        }
        return true;
    };

    auto rollback_candidates = [&](Ds4HostStepTransaction & transaction,
                                   const CandidateExecution & execution,
                                   std::string & error) {
        if (execution.submitted) {
            if (!restore_paged_spec_snapshot(execution.snapshot, error)) {
                return false;
            }
            transaction.rearm_after_device_restore();
        }
        return transaction.rollback(error);
    };

    auto fallback_to_root = [&](const std::string & reason) {
        std::fprintf(stderr,
            "[deepseek4-parallel] paged DSpark fell back to gathered decode: %s\n",
            reason.c_str());
        return std::move(root);
    };
    auto fatal_after_root = [&](const std::string & error) {
        root.decode.clear();
        root.prefills.clear();
        root.error = error;
        return std::move(root);
    };

    Ds4HostStepTransaction transaction(
        slots_, host_tables_, stride_, b_.paged_cache_.plan.physical_blocks);
    CandidateExecution first;
    std::string error;
    if (!execute_candidates(offered, transaction, first, error)) {
        const std::string primary = error;
        std::string rollback_error;
        if (!rollback_candidates(transaction, first, rollback_error)) {
            return fatal_after_root(
                "DeepSeek4 paged DSpark recovery failed after " + primary +
                ": " + rollback_error);
        }
        if (first.fatal) return fatal_after_root(primary);
        return fallback_to_root(primary);
    }

    bool all_accepted = true;
    for (size_t lane_index = 0; lane_index < offered.size(); ++lane_index) {
        const CandidateLane & lane = offered[lane_index];
        const DeepSeek4PagedSegment & segment =
            first.layout.segments[lane_index];
        DraftLane & draft = drafts[lane.input_index];
        int accepted = 1;
        for (int candidate = 1;
             candidate < static_cast<int>(lane.tokens.size()); ++candidate) {
            const size_t posterior_row =
                size_t(segment.row_offset) + size_t(candidate - 1);
            if (draft.proposal[size_t(candidate + 1)] !=
                    first.argmax[posterior_row]) {
                break;
            }
            ++accepted;
        }
        draft.accepted = accepted;
        all_accepted = all_accepted && accepted == int(lane.tokens.size());
    }

    const CandidateExecution * committed_execution = &first;
    std::vector<CandidateLane> committed_lanes = offered;
    std::unique_ptr<Ds4HostStepTransaction> replay_transaction;
    CandidateExecution replay;
    if (!all_accepted) {
        std::string rollback_error;
        if (!rollback_candidates(transaction, first, rollback_error)) {
            return fatal_after_root(
                "DeepSeek4 paged DSpark rejection rollback failed: " +
                rollback_error);
        }
        for (CandidateLane & lane : committed_lanes) {
            lane.tokens.resize(
                size_t(drafts[lane.input_index].accepted));
        }
        replay_transaction = std::make_unique<Ds4HostStepTransaction>(
            slots_, host_tables_, stride_,
            b_.paged_cache_.plan.physical_blocks);
        error.clear();
        if (!execute_candidates(
                committed_lanes, *replay_transaction, replay, error)) {
            const std::string primary = error;
            std::string replay_rollback_error;
            if (!rollback_candidates(
                    *replay_transaction, replay,
                    replay_rollback_error)) {
                return fatal_after_root(
                    "DeepSeek4 paged DSpark replay recovery failed after " +
                    primary + ": " + replay_rollback_error);
            }
            if (replay.fatal) return fatal_after_root(primary);
            return fallback_to_root(primary);
        }
        committed_execution = &replay;
    }

    for (size_t lane_index = 0;
         lane_index < committed_lanes.size(); ++lane_index) {
        const CandidateLane & lane = committed_lanes[lane_index];
        const DeepSeek4PagedSegment & segment =
            committed_execution->layout.segments[lane_index];
        DecodeOutput & output = root.decode[lane.input_index];
        slots_.commit_step(lane.slot);
        output.committed_tokens = lane.tokens;
        output.token = committed_execution->argmax[
            size_t(segment.row_offset + segment.row_count - 1)];
        for (uint32_t local = 0; local < segment.row_count; ++local) {
            const size_t row = size_t(segment.row_offset) + local;
            append_dspark_feature(
                lane.slot,
                committed_execution->capture.data() +
                    row * size_t(feature_width));
        }
        DsparkSlotState & state = dspark_slots_[size_t(lane.slot)];
        if (state.width) {
            state.width->observe(
                drafts[lane.input_index].accepted + 1,
                drafts[lane.input_index].allocated + 1);
        }
    }
    if (replay_transaction) {
        replay_transaction->finish_success();
    } else {
        transaction.finish_success();
    }
    return root;
}

SeqEngine::StepResult DeepSeek4SeqEngine::step_with_paged_segments(
        const StepPlan & plan) {
    StepResult result;
    result.decode.resize(plan.decode.size());
    result.prefills.resize(plan.prefills.size());
    for (size_t i = 0; i < plan.decode.size(); ++i) {
        result.decode[i].slot = plan.decode[i].slot;
    }
    for (size_t i = 0; i < plan.prefills.size(); ++i) {
        result.prefills[i].slot = plan.prefills[i].slot;
    }

    Ds4HostStepTransaction transaction(
        slots_, host_tables_, stride_, b_.paged_cache_.plan.physical_blocks);
    auto fail_before_submit = [&](const std::string & primary_error) {
        result.decode.clear();
        result.prefills.clear();
        std::string rollback_error;
        if (!transaction.rollback(rollback_error)) {
            result.error = "DeepSeek4 host rollback failed after " +
                primary_error + ": " + rollback_error;
        } else {
            result.error = primary_error;
        }
        return std::move(result);
    };
    auto fail_after_submit = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    std::vector<int32_t> decode_segment(plan.decode.size(), -1);
    std::vector<int32_t> prefill_segment(plan.prefills.size(), -1);
    std::vector<DeepSeek4PagedSegmentSpec> specs;
    specs.reserve(plan.decode.size() + plan.prefills.size());

    size_t successful_decode_rows = 0;
    for (size_t i = 0; i < plan.decode.size(); ++i) {
        const StepInput & input = plan.decode[i];
        Ds4HostStage stage = transaction.stage_decode(input);
        if (stage.status == Ds4HostStageStatus::row_failed) {
            result.decode[i].failed = true;
            result.decode[i].error = std::move(stage.error);
            continue;
        }
        if (stage.status != Ds4HostStageStatus::staged) {
            return fail_before_submit(stage.error.empty()
                ? "DeepSeek4 decode host staging failed"
                : stage.error);
        }

        DeepSeek4PagedSegmentSpec spec;
        spec.kind = DeepSeek4PagedSegmentKind::decode;
        spec.slot = input.slot;
        spec.start_position = stage.start_position;
        spec.row_count = 1;
        spec.prompt_tail = false;
        spec.needs_full_logits =
            slots_.slot(input.slot).sampler.needs_logit_processing();
        spec.token_ids = {input.token};
        spec.pool_write_rows = std::move(stage.pool_write_rows);
        spec.first_new_block = stage.first_new_block;
        spec.new_blocks = std::move(stage.new_blocks);
        decode_segment[i] = static_cast<int32_t>(specs.size());
        specs.push_back(std::move(spec));
        ++successful_decode_rows;
    }

    int rows_left = DEEPSEEK4_MAX_PAGED_SEQUENCES -
        static_cast<int>(successful_decode_rows);
    for (size_t i = 0; i < plan.prefills.size(); ++i) {
        const PrefillSlice & slice = plan.prefills[i];
        const SeqSlot & seq = slots_.slot(slice.slot);
        const int start = seq.cur_pos;
        if (start < 0 || start > seq.prompt_len ||
            seq.prompt_len < 1 ||
            seq.sample_history.size() < size_t(seq.prompt_len)) {
            return fail_before_submit(
                "invalid DeepSeek4 prompt state during segment staging");
        }
        const int prompt_remaining = seq.prompt_len - start;
        const int bounded = std::min({
            slice.max_tokens,
            prompt_remaining,
            rows_left,
            int(DEEPSEEK4_MAX_PAGED_SEGMENT_ROWS),
        });
        const int count = bounded > 0
            ? deepseek4_safe_compressor_batch_tokens(
                  b_.w_, start, bounded)
            : 0;
        if (count < 1 || count > bounded ||
            size_t(start) > seq.sample_history.size() ||
            size_t(count) > seq.sample_history.size() - size_t(start)) {
            return fail_before_submit(
                "DeepSeek4 prefill segment has no safe token range");
        }

        std::vector<int32_t> prompt_tokens(
            seq.sample_history.begin() + start,
            seq.sample_history.begin() + start + count);
        if (std::any_of(prompt_tokens.begin(), prompt_tokens.end(),
                        [](int32_t token) { return token < 0; })) {
            return fail_before_submit(
                "DeepSeek4 prefill segment contains an invalid token");
        }
        const bool prompt_tail = start + count == seq.prompt_len;
        const bool needs_full_logits = prompt_tail &&
            seq.sampler.needs_logit_processing();

        Ds4HostStage stage =
            transaction.stage_prefill(slice.slot, count);
        if (stage.status != Ds4HostStageStatus::staged ||
            stage.start_position != start) {
            return fail_before_submit(stage.error.empty()
                ? "DeepSeek4 prefill host staging failed"
                : stage.error);
        }

        DeepSeek4PagedSegmentSpec spec;
        spec.kind = DeepSeek4PagedSegmentKind::prefill;
        spec.slot = slice.slot;
        spec.start_position = start;
        spec.row_count = static_cast<uint32_t>(count);
        spec.prompt_tail = prompt_tail;
        spec.needs_full_logits = needs_full_logits;
        spec.token_ids = std::move(prompt_tokens);
        spec.pool_write_rows = std::move(stage.pool_write_rows);
        spec.first_new_block = stage.first_new_block;
        spec.new_blocks = std::move(stage.new_blocks);
        prefill_segment[i] = static_cast<int32_t>(specs.size());
        specs.push_back(std::move(spec));
        rows_left -= count;
    }

    DeepSeek4PagedStepLayout layout;
    if (!prepare_deepseek4_paged_step_layout(
            specs, host_tables_, b_.paged_cache_.plan.slots,
            b_.paged_cache_.plan.max_ctx,
            b_.paged_cache_.plan.max_blocks_per_sequence,
            b_.paged_cache_.plan.physical_blocks, layout)) {
        return fail_before_submit(
            "DeepSeek4 paged segment layout failed");
    }

    const size_t q = layout.token_ids.size();
    if (b_.w_.n_embd <= 0 ||
        q > std::numeric_limits<size_t>::max() /
                size_t(b_.w_.n_embd)) {
        return fail_before_submit(
            "DeepSeek4 segment embedding size overflow");
    }
    const size_t embedding_count = q * size_t(b_.w_.n_embd);
    layout.embeddings.resize(embedding_count);
    if (!b_.w_.embedder.embed(
            layout.token_ids.data(), static_cast<int>(q),
            layout.embeddings.data())) {
        return fail_before_submit(
            "DeepSeek4 segment embedding failed");
    }

    if (layout.segments.size() != specs.size() ||
        layout.positions.size() != q ||
        layout.row_to_segment.size() != q ||
        layout.row_to_public_output.size() != q ||
        layout.requested_full_logits.size() != q) {
        return fail_before_submit(
            "DeepSeek4 segment output geometry is inconsistent");
    }
    auto mapped_output_is_valid = [&](const DeepSeek4PagedSegment & segment) {
        if (segment.output_row < 0 ||
            size_t(segment.output_row) >= q) {
            return false;
        }
        const int32_t public_index =
            layout.row_to_public_output[size_t(segment.output_row)];
        return public_index >= 0 &&
            size_t(public_index) < layout.public_output_rows.size() &&
            layout.public_output_rows[size_t(public_index)] ==
                segment.output_row;
    };

    size_t expected_public_outputs = 0;
    for (size_t i = 0; i < decode_segment.size(); ++i) {
        const int32_t segment_index = decode_segment[i];
        if (segment_index < 0) {
            if (!result.decode[i].failed) {
                return fail_before_submit(
                    "DeepSeek4 decode row lost its staged mapping");
            }
            continue;
        }
        if (size_t(segment_index) >= layout.segments.size()) {
            return fail_before_submit(
                "DeepSeek4 decode segment mapping is out of range");
        }
        const DeepSeek4PagedSegment & segment =
            layout.segments[size_t(segment_index)];
        const bool needs_full_logits =
            slots_.slot(plan.decode[i].slot)
                .sampler.needs_logit_processing();
        if (segment.kind != DeepSeek4PagedSegmentKind::decode ||
            segment.slot != plan.decode[i].slot ||
            segment.row_count != 1 || segment.prompt_tail ||
            segment.output_row != int32_t(segment.row_offset) ||
            !mapped_output_is_valid(segment) ||
            layout.requested_full_logits[
                size_t(segment.output_row)] !=
                    uint8_t(needs_full_logits)) {
            return fail_before_submit(
                "DeepSeek4 decode segment output mapping is invalid");
        }
        ++expected_public_outputs;
    }
    for (size_t i = 0; i < prefill_segment.size(); ++i) {
        const int32_t segment_index = prefill_segment[i];
        if (segment_index < 0 ||
            size_t(segment_index) >= layout.segments.size()) {
            return fail_before_submit(
                "DeepSeek4 prefill segment mapping is out of range");
        }
        const DeepSeek4PagedSegment & segment =
            layout.segments[size_t(segment_index)];
        if (size_t(segment.row_offset) > q ||
            size_t(segment.row_count) > q - size_t(segment.row_offset)) {
            return fail_before_submit(
                "DeepSeek4 prefill segment span is out of range");
        }
        const bool tail = segment.prompt_tail;
        const bool needs_full_logits = tail &&
            slots_.slot(plan.prefills[i].slot)
                .sampler.needs_logit_processing();
        if (segment.kind != DeepSeek4PagedSegmentKind::prefill ||
            segment.slot != plan.prefills[i].slot ||
            segment.row_count != specs[size_t(segment_index)].row_count ||
            (tail &&
             segment.output_row != int32_t(
                 segment.row_offset + segment.row_count - 1)) ||
            (!tail && segment.output_row != -1) ||
            (tail && !mapped_output_is_valid(segment))) {
            return fail_before_submit(
                "DeepSeek4 prefill segment output mapping is invalid");
        }
        for (uint32_t local_row = 0; local_row < segment.row_count;
             ++local_row) {
            const size_t row = size_t(segment.row_offset) + local_row;
            const bool output_row =
                int32_t(row) == segment.output_row;
            if ((!output_row &&
                 (layout.row_to_public_output[row] != -1 ||
                  layout.requested_full_logits[row])) ||
                (output_row &&
                 layout.requested_full_logits[row] !=
                     uint8_t(needs_full_logits))) {
                return fail_before_submit(
                    "DeepSeek4 prompt row exposes an invalid output");
            }
        }
        expected_public_outputs += tail ? 1 : 0;
    }
    if (layout.public_output_rows.size() != expected_public_outputs) {
        return fail_before_submit(
            "DeepSeek4 public segment output count is inconsistent");
    }

    DeepSeek4PagedSegmentPreparedStep * raw_prepared = nullptr;
    const bool capture_dspark = paged_dspark_enabled();
    const std::vector<int> * capture_ids = capture_dspark
        ? &b_.spec_drafter_->capture_layer_ids : nullptr;
    if (!deepseek4_prepare_paged_segment_step(
            b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
            layout, b_.moe_hybrid_.get(), raw_prepared, capture_ids) ||
        !raw_prepared) {
        if (raw_prepared) {
            deepseek4_discard_paged_segment_step(raw_prepared);
        }
        return fail_before_submit(
            "DeepSeek4 paged segment prepare failed");
    }
    using PreparedStepPtr = std::unique_ptr<
        DeepSeek4PagedSegmentPreparedStep,
        void (*)(DeepSeek4PagedSegmentPreparedStep *)>;
    PreparedStepPtr prepared(
        raw_prepared, deepseek4_discard_paged_segment_step);

    transaction.cross_device_mutation_barrier();

    std::vector<float> logits;
    std::vector<int32_t> argmax;
    std::vector<float> capture;
    if (!deepseek4_compute_paged_segment_step(
            *prepared, logits, argmax, b_.routing_stats_.get(),
            capture_dspark ? &capture : nullptr)) {
        return fail_after_submit(
            "DeepSeek4 paged segment graph failed");
    }

    const bool any_full_logits = std::any_of(
        layout.requested_full_logits.begin(),
        layout.requested_full_logits.end(),
        [](uint8_t requested) { return requested != 0; });
    const size_t public_output_count = layout.public_output_rows.size();
    if (argmax.size() != q && argmax.size() != public_output_count) {
        return fail_after_submit(
            "DeepSeek4 paged segment output width is malformed");
    }
    const bool outputs_are_packed =
        argmax.size() == public_output_count;
    size_t expected_logits = 0;
    if (any_full_logits) {
        if (b_.w_.n_vocab <= 0 ||
            argmax.size() > std::numeric_limits<size_t>::max() /
                    size_t(b_.w_.n_vocab)) {
            return fail_after_submit(
                "DeepSeek4 segment logits size overflow");
        }
        expected_logits = argmax.size() * size_t(b_.w_.n_vocab);
    }
    if ((any_full_logits && logits.size() != expected_logits) ||
        (!any_full_logits && !logits.empty())) {
        return fail_after_submit(
            "DeepSeek4 paged segment outputs are malformed");
    }
    const int feature_width = capture_dspark ? dspark_feature_width() : 0;
    if (capture_dspark &&
        (feature_width <= 0 ||
         capture.size() != q * size_t(feature_width))) {
        return fail_after_submit(
            "DeepSeek4 paged segment DSpark capture is malformed");
    }

    auto sample_row = [&](int slot_id, int32_t model_row) {
        SeqSlot & seq = slots_.slot(slot_id);
        const int32_t public_row =
            layout.row_to_public_output[size_t(model_row)];
        const size_t output_row = outputs_are_packed
            ? size_t(public_row) : size_t(model_row);
        if (!seq.sampler.needs_logit_processing()) {
            return argmax[output_row];
        }
        return sample_logits(
            logits.data() + output_row * size_t(b_.w_.n_vocab),
            b_.w_.n_vocab, seq.sampler, seq.sample_history, seq.rng);
    };

    for (size_t i = 0; i < plan.decode.size(); ++i) {
        const int32_t segment_index = decode_segment[i];
        if (segment_index < 0) continue;
        DecodeOutput & out = result.decode[i];
        const int32_t model_row =
            layout.segments[size_t(segment_index)].output_row;
        slots_.commit_step(out.slot);
        out.token = sample_row(out.slot, model_row);
    }
    for (size_t i = 0; i < plan.prefills.size(); ++i) {
        PrefillOutput & out = result.prefills[i];
        const DeepSeek4PagedSegment & segment =
            layout.segments[size_t(prefill_segment[i])];
        if (!segment.prompt_tail) {
            out.status = PrefillOutput::Status::advanced;
            continue;
        }
        out.token = sample_row(out.slot, segment.output_row);
        out.status = PrefillOutput::Status::completed;
        slots_.commit_prefill(out.slot);
    }
    if (capture_dspark) {
        for (const DeepSeek4PagedSegment & segment : layout.segments) {
            for (uint32_t local = 0; local < segment.row_count; ++local) {
                const size_t row = size_t(segment.row_offset) + local;
                append_dspark_feature(
                    segment.slot,
                    capture.data() + row * size_t(feature_width));
            }
        }
    }
    transaction.finish_success();
    return result;
}

void DeepSeek4SeqEngine::retire(int slot) {
    if (!slots_.is_active(slot)) return;
    slots_.retire(slot);
    if (slot >= 0 && size_t(slot) < dspark_slots_.size()) {
        DsparkSlotState & state = dspark_slots_[size_t(slot)];
        state.features.clear();
        if (state.width) state.width->reset();
    }
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)slot);
    std::fill_n(host_tables_.data() + (size_t)slot * stride_, stride_, -1);
}

} // namespace dflash::common
