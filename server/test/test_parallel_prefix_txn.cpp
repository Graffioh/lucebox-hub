#include "server/parallel_prefix_txn.h"
#include "host_check.h"

#include <type_traits>
#include <utility>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

namespace {

struct FakePolicy {
    std::vector<int> cancelled;
    std::vector<int> aborted;
    std::vector<int> confirmed;
    size_t confirmed_bytes = 0;

    void cancel_inline_snap(int slot) { cancelled.push_back(slot); }
    void abort_inline_snap(int slot) { aborted.push_back(slot); }
    void confirm_inline_snap(
            int slot, int, const std::vector<int32_t> &,
            bool, size_t resident_bytes) {
        confirmed.push_back(slot);
        confirmed_bytes = resident_bytes;
    }
};

struct FakeEngine {
    std::vector<PrefixStoreRef> discarded;

    void discard_prefix_store(PrefixStoreRef checkpoint) {
        discarded.push_back(checkpoint);
    }
};

using Txn = BasicPrefixCaptureTxn<FakePolicy, FakeEngine>;

PrefixCaptureTicket ticket(uint64_t id = 11) {
    PrefixCaptureTicket value;
    value.id = id;
    value.checkpoint = {7, 16};
    return value;
}

PrefixStoreEvent event(
        PrefixStoreEvent::Status status, PrefixCaptureTicket value) {
    PrefixStoreEvent out;
    out.status = status;
    out.ticket = value;
    if (status == PrefixStoreEvent::Status::saved) {
        out.bytes = 4096;
    }
    if (status == PrefixStoreEvent::Status::failed) {
        out.error = "copy failed";
    }
    return out;
}

}  // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<Txn>);
    static_assert(!std::is_copy_assignable_v<Txn>);
    static_assert(std::is_move_constructible_v<Txn>);
    static_assert(std::is_move_assignable_v<Txn>);

    {
        FakePolicy policy;
        FakeEngine engine;
        {
            Txn first(policy, engine, /*policy_slot=*/3, ticket());
            Txn owner(std::move(first));
            CHECK(!first.active());
            CHECK(owner.active());
        }
        CHECK(policy.cancelled == std::vector<int>({3}));
        CHECK(policy.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    {
        FakePolicy policy;
        FakeEngine engine;
        Txn txn(policy, engine, /*policy_slot=*/3, ticket());
        CHECK(txn.resolve(
                  event(PrefixStoreEvent::Status::saved, ticket()),
                  std::vector<int32_t>(16, 1)) ==
              Txn::Resolution::saved);
        CHECK(!txn.active());
        CHECK(policy.confirmed_bytes == 4096);
        CHECK(policy.confirmed == std::vector<int>({3}));
        CHECK(policy.cancelled.empty());
        CHECK(policy.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    {
        FakePolicy policy;
        FakeEngine engine;
        Txn txn(policy, engine, /*policy_slot=*/3, ticket());
        CHECK(txn.resolve(
                  event(PrefixStoreEvent::Status::failed, ticket()),
                  std::vector<int32_t>(16, 1)) ==
              Txn::Resolution::failed);
        CHECK(policy.cancelled == std::vector<int>({3}));
        CHECK(policy.aborted.empty());
        CHECK(engine.discarded.empty());
    }

    {
        FakePolicy policy;
        FakeEngine engine;
        Txn txn(policy, engine, /*policy_slot=*/3, ticket());
        PrefixCaptureTicket unrelated = ticket(/*id=*/99);
        unrelated.checkpoint = {63, 16};
        CHECK(txn.resolve(
                  event(PrefixStoreEvent::Status::saved, unrelated),
                  std::vector<int32_t>(16, 1)) ==
              Txn::Resolution::mismatched);
        CHECK(policy.cancelled.empty());
        CHECK(policy.aborted == std::vector<int>({3}));
        CHECK(engine.discarded == std::vector<PrefixStoreRef>({{7, 16}}));
    }

    {
        FakePolicy policy;
        FakeEngine engine;
        Txn txn(policy, engine, /*policy_slot=*/3, ticket());
        txn.abort();
        CHECK(policy.aborted == std::vector<int>({3}));
        CHECK(engine.discarded == std::vector<PrefixStoreRef>({{7, 16}}));
    }

    std::printf("OK test_parallel_prefix_txn (%d checks)\n", g_checks);
    return 0;
}
