#include "qwen_paged_kv_transfer.h"

#include "common/gpu_runtime_compat.h"
#include "internal.h"

#include "ggml.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace dflash::common {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool runtime_pointer_is_device(const void * pointer, int device) {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    return attributes.type == hipMemoryTypeDevice &&
           attributes.device == device;
#else
    return attributes.type == cudaMemoryTypeDevice &&
           attributes.device == device;
#endif
}

bool tensor_is_device_backed(const ggml_tensor * tensor, int device,
                             std::string * error) {
    if (!tensor || !tensor->buffer || !tensor->data) {
        set_error(error, "paged K/V tensor is null or unallocated");
        return false;
    }
    const ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(tensor->buffer);
    if (!buft || ggml_backend_buft_is_meta(buft)) {
        set_error(error,
                  "meta/tensor-parallel paged K/V buffers are unsupported");
        return false;
    }
    if (ggml_backend_buft_is_host(buft)) {
        set_error(error, "host-backed paged K/V buffers are unsupported");
        return false;
    }
    const ggml_backend_dev_t tensor_device =
        ggml_backend_buft_get_device(buft);
    if (!tensor_device) {
        set_error(error, "paged K/V buffer has no backend device");
        return false;
    }
    const enum ggml_backend_dev_type tensor_device_type =
        ggml_backend_dev_type(tensor_device);
    if (tensor_device_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        tensor_device_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        set_error(error, "paged K/V buffer is not GPU device memory");
        return false;
    }
    if (!runtime_pointer_is_device(tensor->data, device)) {
        set_error(error,
                  "paged K/V tensor is not on the requested HIP/CUDA device");
        return false;
    }
    return true;
}

}  // namespace

struct QwenPagedKvResidencyTransfer::State {
    struct TensorCopy {
        uint8_t * device_data = nullptr;
        QwenPagedKvTensorLayout layout;
        size_t host_offset = 0;
        size_t host_head_bytes = 0;
    };

    int device = -1;
    uint32_t block_size = 0;
    size_t block_bytes = 0;
    uint32_t physical_block_count = 0;
    cudaStream_t stream = nullptr;
    bool stream_may_reference_host = false;
    std::vector<TensorCopy> tensors;
    std::vector<void *> pinned_allocations;

    ~State() {
        if (device >= 0 && cudaSetDevice(device) == cudaSuccess) {
            if (stream && cudaStreamSynchronize(stream) != cudaSuccess) {
                // Match PagedKvResidencyManager's fail-safe teardown rule:
                // leak stream/backing rather than free host memory that a
                // failed runtime may still reference.
                pinned_allocations.clear();
                stream = nullptr;
                return;
            }
            for (void * pointer : pinned_allocations) {
                if (pointer) (void)cudaFreeHost(pointer);
            }
            pinned_allocations.clear();
            if (stream) (void)cudaStreamDestroy(stream);
        }
        stream = nullptr;
    }

    bool select_device() const {
        return device >= 0 && cudaSetDevice(device) == cudaSuccess;
    }

    void * allocate_pinned(size_t bytes) {
        if (bytes != block_bytes || !select_device()) return nullptr;
        void * pointer = nullptr;
        if (cudaMallocHost(&pointer, bytes) != cudaSuccess || !pointer) {
            (void)cudaGetLastError();
            return nullptr;
        }
        try {
            pinned_allocations.push_back(pointer);
        } catch (...) {
            (void)cudaFreeHost(pointer);
            return nullptr;
        }
        return pointer;
    }

    void free_pinned(void * pointer) {
        if (!pointer) return;
        const auto found = std::find(
            pinned_allocations.begin(), pinned_allocations.end(), pointer);
        if (found == pinned_allocations.end()) return;
        if (stream_may_reference_host) {
            if (!stream || !select_device() ||
                cudaStreamSynchronize(stream) != cudaSuccess) {
                return;
            }
            stream_may_reference_host = false;
        }
        if (select_device() && cudaFreeHost(pointer) == cudaSuccess) {
            pinned_allocations.erase(found);
        }
    }

    bool synchronize() {
        if (!stream || !select_device()) return false;
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaGetLastError();
            stream_may_reference_host = true;
            return false;
        }
        stream_may_reference_host = false;
        return true;
    }

    bool queue_copy(uint32_t physical_block, void * host, size_t bytes,
                    bool copy_out) {
        if (!host || bytes != block_bytes || !stream ||
            physical_block >= physical_block_count || !select_device()) {
            return false;
        }

        const size_t first_row =
            static_cast<size_t>(physical_block) * block_size;
        uint8_t * const host_bytes = static_cast<uint8_t *>(host);
        for (const TensorCopy & tensor : tensors) {
            uint8_t * const device_block = tensor.device_data +
                first_row * tensor.layout.row_stride;
            uint8_t * const host_tensor =
                host_bytes + tensor.host_offset;

            cudaError_t status = cudaSuccess;
            if (tensor.layout.row_stride == tensor.layout.row_bytes) {
                // Common Qwen cache layout: block rows are contiguous inside
                // each head plane, so one 2D copy covers every head.
                status = copy_out
                    ? cudaMemcpy2DAsync(
                          host_tensor, tensor.host_head_bytes,
                          device_block, tensor.layout.head_stride,
                          tensor.host_head_bytes, tensor.layout.heads,
                          cudaMemcpyDeviceToHost, stream)
                    : cudaMemcpy2DAsync(
                          device_block, tensor.layout.head_stride,
                          host_tensor, tensor.host_head_bytes,
                          tensor.host_head_bytes, tensor.layout.heads,
                          cudaMemcpyHostToDevice, stream);
            } else {
                // Preserve uncommon row padding with one pitched async copy
                // per head. Host images remain tightly packed payload bytes.
                for (size_t head = 0;
                     head < static_cast<size_t>(tensor.layout.heads);
                     ++head) {
                    uint8_t * const device_head =
                        device_block + head * tensor.layout.head_stride;
                    uint8_t * const host_head =
                        host_tensor + head * tensor.host_head_bytes;
                    status = copy_out
                        ? cudaMemcpy2DAsync(
                              host_head, tensor.layout.row_bytes,
                              device_head, tensor.layout.row_stride,
                              tensor.layout.row_bytes, block_size,
                              cudaMemcpyDeviceToHost, stream)
                        : cudaMemcpy2DAsync(
                              device_head, tensor.layout.row_stride,
                              host_head, tensor.layout.row_bytes,
                              tensor.layout.row_bytes, block_size,
                              cudaMemcpyHostToDevice, stream);
                    if (status != cudaSuccess) break;
                }
            }
            if (status != cudaSuccess) {
                // A callback that returns false is not marked pending by the
                // residency manager. Drain any prefix already queued here so
                // its pinned buffer can still be released safely.
                (void)cudaGetLastError();
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    stream_may_reference_host = true;
                }
                return false;
            }
        }
        return true;
    }
};

QwenPagedKvResidencyTransfer::QwenPagedKvResidencyTransfer(
        std::shared_ptr<State> state)
    : state_(std::move(state)) {}

QwenPagedKvResidencyTransfer::~QwenPagedKvResidencyTransfer() = default;

std::unique_ptr<QwenPagedKvResidencyTransfer>
QwenPagedKvResidencyTransfer::create(
        const TargetCache & cache,
        ggml_backend_t backend,
        int device,
        uint32_t block_size,
        std::string * error) {
    if (error) error->clear();
    if (!backend || device < 0 || block_size == 0) {
        set_error(error, "invalid paged K/V transfer backend/device/block size");
        return nullptr;
    }
    const ggml_backend_dev_t backend_device = ggml_backend_get_device(backend);
    if (!backend_device) {
        set_error(error, "paged K/V transfer backend has no device");
        return nullptr;
    }
    const enum ggml_backend_dev_type backend_type =
        ggml_backend_dev_type(backend_device);
    if (backend_type == GGML_BACKEND_DEVICE_TYPE_META ||
        ggml_backend_buft_is_meta(
            ggml_backend_get_default_buffer_type(backend))) {
        set_error(error,
                  "meta/tensor-parallel paged K/V transfer is unsupported");
        return nullptr;
    }
    if (backend_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        backend_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        set_error(error, "paged K/V transfer requires a GPU backend");
        return nullptr;
    }
    if (cache.backend && cache.backend != backend) {
        set_error(error, "paged K/V cache belongs to a different backend");
        return nullptr;
    }
    if (cache.attn_k.empty() ||
        cache.attn_k.size() != cache.attn_v.size()) {
        set_error(error, "paged K/V cache has no complete attention layers");
        return nullptr;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device >= device_count || cudaSetDevice(device) != cudaSuccess) {
        (void)cudaGetLastError();
        set_error(error, "invalid HIP/CUDA device for paged K/V transfer");
        return nullptr;
    }

    std::vector<const ggml_tensor *> cache_tensors;
    std::vector<QwenPagedKvTensorLayout> layouts;
    try {
        cache_tensors.reserve(cache.attn_k.size() * 2);
        layouts.reserve(cache.attn_k.size() * 2);
    } catch (...) {
        set_error(error, "paged K/V transfer layout allocation failed");
        return nullptr;
    }

    for (size_t layer = 0; layer < cache.attn_k.size(); ++layer) {
        const ggml_tensor * pair[] = {
            cache.attn_k[layer], cache.attn_v[layer],
        };
        if (!pair[0] || !pair[1]) {
            set_error(error,
                      "partial/tensor-parallel paged K/V cache is unsupported");
            return nullptr;
        }
        if (pair[0]->ne[1] != pair[1]->ne[1] ||
            pair[0]->ne[2] != pair[1]->ne[2]) {
            set_error(error, "K/V cache pair dimensions do not match");
            return nullptr;
        }
        for (const ggml_tensor * tensor : pair) {
            if (!tensor_is_device_backed(tensor, device, error)) return nullptr;
            if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
                tensor->ne[2] <= 0 || tensor->ne[3] != 1 ||
                tensor->ne[0] % ggml_blck_size(tensor->type) != 0) {
                set_error(error, "unsupported paged K/V tensor dimensions");
                return nullptr;
            }
            const size_t row_bytes =
                ggml_row_size(tensor->type, tensor->ne[0]);
            cache_tensors.push_back(tensor);
            layouts.push_back({
                row_bytes,
                tensor->nb[1],
                tensor->nb[2],
                ggml_nbytes(tensor),
                static_cast<uint64_t>(tensor->ne[1]),
                static_cast<uint64_t>(tensor->ne[2]),
            });
        }
    }

    QwenPagedKvBlockLayout plan;
    if (!plan_qwen_paged_kv_block_layout(
            layouts, block_size, plan, error)) {
        return nullptr;
    }

    std::shared_ptr<State> state;
    try {
        state = std::make_shared<State>();
        state->device = device;
        state->block_size = block_size;
        state->block_bytes = plan.block_bytes;
        state->physical_block_count = plan.physical_block_count;
        state->tensors.reserve(cache_tensors.size());
        for (size_t i = 0; i < cache_tensors.size(); ++i) {
            state->tensors.push_back({
                static_cast<uint8_t *>(cache_tensors[i]->data),
                layouts[i],
                plan.tensor_offsets[i],
                plan.tensor_head_bytes[i],
            });
        }
    } catch (...) {
        set_error(error, "paged K/V transfer state allocation failed");
        return nullptr;
    }

    if (cudaStreamCreateWithFlags(
            &state->stream, cudaStreamNonBlocking) != cudaSuccess) {
        (void)cudaGetLastError();
        set_error(error, "failed to create paged K/V transfer stream");
        return nullptr;
    }

    try {
        return std::unique_ptr<QwenPagedKvResidencyTransfer>(
            new QwenPagedKvResidencyTransfer(std::move(state)));
    } catch (...) {
        set_error(error, "paged K/V transfer wrapper allocation failed");
        return nullptr;
    }
}

size_t QwenPagedKvResidencyTransfer::block_bytes() const noexcept {
    return state_ ? state_->block_bytes : 0;
}

PagedKvResidencyTransferOps
QwenPagedKvResidencyTransfer::callbacks() const {
    PagedKvResidencyTransferOps result;
    const std::shared_ptr<State> state = state_;
    if (!state) return result;
    result.allocate_pinned = [state](size_t bytes) {
        return state->allocate_pinned(bytes);
    };
    result.free_pinned = [state](void * pointer) {
        state->free_pinned(pointer);
    };
    result.copy_out_async =
        [state](PagedKvSequenceHandle, uint32_t, uint32_t physical_block,
                void * host, size_t bytes) {
            return state->queue_copy(
                physical_block, host, bytes, /*copy_out=*/true);
        };
    result.copy_in_async =
        [state](PagedKvSequenceHandle, uint32_t, uint32_t physical_block,
                const void * host, size_t bytes) {
            return state->queue_copy(
                physical_block, const_cast<void *>(host), bytes,
                /*copy_out=*/false);
        };
    result.synchronize = [state] { return state->synchronize(); };
    return result;
}

}  // namespace dflash::common
