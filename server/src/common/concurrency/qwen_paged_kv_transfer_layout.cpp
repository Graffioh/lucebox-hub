#include "qwen_paged_kv_transfer.h"

#include <limits>

namespace dflash::common {
namespace {

bool fail(QwenPagedKvBlockLayout & out, std::string * error,
          const char * message) {
    out = {};
    if (error) *error = message;
    return false;
}

bool checked_mul(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t & out) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) return false;
    out = lhs + rhs;
    return true;
}

}  // namespace

bool plan_qwen_paged_kv_block_layout(
        const std::vector<QwenPagedKvTensorLayout> & tensors,
        uint32_t block_size,
        QwenPagedKvBlockLayout & out,
        std::string * error) {
    out = {};
    if (tensors.empty() || block_size == 0) {
        return fail(out, error, "empty paged K/V layout or zero block size");
    }

    const uint64_t physical_rows = tensors.front().physical_rows;
    const uint64_t heads = tensors.front().heads;
    if (physical_rows == 0 || heads == 0 ||
        physical_rows % block_size != 0 ||
        physical_rows / block_size > std::numeric_limits<uint32_t>::max()) {
        return fail(out, error, "invalid paged K/V physical dimensions");
    }
    out.physical_block_count =
        static_cast<uint32_t>(physical_rows / block_size);

    try {
        out.tensor_offsets.reserve(tensors.size());
        out.tensor_head_bytes.reserve(tensors.size());
    } catch (...) {
        return fail(out, error, "paged K/V layout allocation failed");
    }

    size_t host_offset = 0;
    for (const QwenPagedKvTensorLayout & tensor : tensors) {
        if (tensor.row_bytes == 0 || tensor.row_stride < tensor.row_bytes ||
            tensor.head_stride == 0 || tensor.storage_bytes == 0 ||
            tensor.physical_rows != physical_rows || tensor.heads != heads) {
            return fail(out, error, "inconsistent paged K/V tensor layout");
        }
        if (physical_rows > std::numeric_limits<size_t>::max() ||
            heads > std::numeric_limits<size_t>::max()) {
            return fail(out, error, "paged K/V dimensions exceed host size_t");
        }

        size_t last_row_offset = 0;
        size_t last_head_offset = 0;
        size_t used_end = 0;
        if (!checked_mul(static_cast<size_t>(physical_rows - 1),
                         tensor.row_stride, last_row_offset) ||
            !checked_mul(static_cast<size_t>(heads - 1),
                         tensor.head_stride, last_head_offset) ||
            !checked_add(last_head_offset, last_row_offset, used_end) ||
            !checked_add(used_end, tensor.row_bytes, used_end) ||
            used_end > tensor.storage_bytes) {
            return fail(out, error, "paged K/V tensor strides exceed storage");
        }

        // Heads must not overlap. Larger head strides (alignment/padding) are
        // fine because device copies retain the original pitch.
        size_t plane_span = 0;
        if (!checked_mul(static_cast<size_t>(physical_rows - 1),
                         tensor.row_stride, plane_span) ||
            !checked_add(plane_span, tensor.row_bytes, plane_span) ||
            tensor.head_stride < plane_span) {
            return fail(out, error, "overlapping paged K/V head planes");
        }

        size_t head_bytes = 0;
        size_t tensor_bytes = 0;
        size_t next_offset = 0;
        if (!checked_mul(tensor.row_bytes, block_size, head_bytes) ||
            !checked_mul(head_bytes, static_cast<size_t>(heads),
                         tensor_bytes) ||
            !checked_add(host_offset, tensor_bytes, next_offset)) {
            return fail(out, error, "paged K/V host block size overflow");
        }
        try {
            out.tensor_offsets.push_back(host_offset);
            out.tensor_head_bytes.push_back(head_bytes);
        } catch (...) {
            return fail(out, error, "paged K/V layout allocation failed");
        }
        host_offset = next_offset;
    }

    if (host_offset == 0) {
        return fail(out, error, "zero-byte paged K/V host block");
    }
    out.block_bytes = host_offset;
    if (error) error->clear();
    return true;
}

}  // namespace dflash::common
