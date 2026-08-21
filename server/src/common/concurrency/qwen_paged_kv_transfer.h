// Production HIP/CUDA block transfers for Qwen paged K/V residency.
//
// A host image packs the payload bytes for one physical block from every
// full-attention K and V tensor. Device row/head strides are retained in the
// copy plan, so quantized K/V types and padded tensor layouts are copied
// without reinterpretation.

#pragma once

#include "paged_kv_residency.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct TargetCache;

// GPU-independent input/output for block-layout validation. storage_bytes is
// the accessible span starting at the tensor data pointer (ggml_nbytes for a
// normal cache tensor).
struct QwenPagedKvTensorLayout {
    size_t row_bytes = 0;
    size_t row_stride = 0;
    size_t head_stride = 0;
    size_t storage_bytes = 0;
    uint64_t physical_rows = 0;
    uint64_t heads = 0;
};

struct QwenPagedKvBlockLayout {
    size_t block_bytes = 0;
    uint32_t physical_block_count = 0;
    // One entry per input tensor. Tensor payloads are packed consecutively;
    // each head occupies tensor_head_bytes[i] bytes in the host image.
    std::vector<size_t> tensor_offsets;
    std::vector<size_t> tensor_head_bytes;
};

// Validates all dimensions/strides and computes the packed host block image.
// Every tensor must cover the same physical row count and head count. Row
// padding is supported; padding bytes are not copied into the host image.
bool plan_qwen_paged_kv_block_layout(
    const std::vector<QwenPagedKvTensorLayout> & tensors,
    uint32_t block_size,
    QwenPagedKvBlockLayout & out,
    std::string * error = nullptr);

// Owns the dedicated nonblocking transfer stream. callbacks() retains shared
// ownership of the stream/layout state, so this wrapper may be destroyed as
// soon as callbacks are handed to PagedKvResidencyManager. TargetCache and
// its K/V buffers must outlive those callbacks. Before asking the residency
// manager to evict, the engine must have completed the compute work that last
// wrote the source block; the manager's synchronize callback supplies the
// opposite copy-stream -> later-attention/append barrier.
class QwenPagedKvResidencyTransfer {
public:
    static std::unique_ptr<QwenPagedKvResidencyTransfer> create(
        const TargetCache & cache,
        ggml_backend_t backend,
        int device,
        uint32_t block_size,
        std::string * error = nullptr);

    ~QwenPagedKvResidencyTransfer();

    QwenPagedKvResidencyTransfer(
        const QwenPagedKvResidencyTransfer &) = delete;
    QwenPagedKvResidencyTransfer & operator=(
        const QwenPagedKvResidencyTransfer &) = delete;

    size_t block_bytes() const noexcept;
    PagedKvResidencyTransferOps callbacks() const;

private:
    struct State;
    explicit QwenPagedKvResidencyTransfer(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;
};

}  // namespace dflash::common
