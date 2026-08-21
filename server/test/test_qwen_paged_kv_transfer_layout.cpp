#define GENERATE_UNIT_TEST_MAIN
#include "CppUnitTestFramework.hpp"
#include "common/concurrency/qwen_paged_kv_transfer.h"

#include <limits>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {
struct QwenPagedKvTransferLayoutFixture {};

QwenPagedKvTensorLayout packed(size_t row_bytes, uint64_t rows,
                               uint64_t heads) {
    return {
        row_bytes,
        row_bytes,
        row_bytes * static_cast<size_t>(rows),
        row_bytes * static_cast<size_t>(rows) *
            static_cast<size_t>(heads),
        rows,
        heads,
    };
}
}  // namespace

TEST_CASE(QwenPagedKvTransferLayoutFixture,
          packs_mixed_kv_types_and_all_heads) {
    const std::vector<QwenPagedKvTensorLayout> tensors = {
        packed(8, 64, 2),
        packed(4, 64, 2),
        packed(6, 64, 2),
        packed(2, 64, 2),
    };
    QwenPagedKvBlockLayout plan;
    std::string error;
    CHECK(plan_qwen_paged_kv_block_layout(
        tensors, /*block_size=*/16, plan, &error));
    CHECK(error.empty());
    CHECK(plan.physical_block_count == 4);
    CHECK(plan.tensor_offsets ==
          std::vector<size_t>({0, 256, 384, 576}));
    CHECK(plan.tensor_head_bytes ==
          std::vector<size_t>({128, 64, 96, 32}));
    CHECK(plan.block_bytes == 640);
}

TEST_CASE(QwenPagedKvTransferLayoutFixture,
          accepts_row_and_head_padding_without_storing_padding) {
    QwenPagedKvTensorLayout tensor;
    tensor.row_bytes = 6;
    tensor.row_stride = 8;
    tensor.head_stride = 520;
    tensor.physical_rows = 64;
    tensor.heads = 2;
    tensor.storage_bytes = 520 + 63 * 8 + 6;

    QwenPagedKvBlockLayout plan;
    CHECK(plan_qwen_paged_kv_block_layout(
        {tensor}, /*block_size=*/16, plan));
    CHECK(plan.tensor_head_bytes == std::vector<size_t>({96}));
    CHECK(plan.block_bytes == 192);
}

TEST_CASE(QwenPagedKvTransferLayoutFixture,
          rejects_mismatched_rows_short_storage_and_overflow) {
    QwenPagedKvBlockLayout plan;
    std::string error;
    CHECK(!plan_qwen_paged_kv_block_layout(
        {packed(8, 64, 2), packed(8, 32, 2)}, 16, plan, &error));
    CHECK(!error.empty());

    QwenPagedKvTensorLayout short_tensor = packed(8, 64, 2);
    short_tensor.storage_bytes--;
    CHECK(!plan_qwen_paged_kv_block_layout(
        {short_tensor}, 16, plan, &error));

    QwenPagedKvTensorLayout overflow = packed(8, 64, 2);
    overflow.row_bytes = std::numeric_limits<size_t>::max();
    overflow.row_stride = overflow.row_bytes;
    overflow.head_stride = overflow.row_bytes;
    overflow.storage_bytes = overflow.row_bytes;
    CHECK(!plan_qwen_paged_kv_block_layout(
        {overflow}, 16, plan, &error));
}
