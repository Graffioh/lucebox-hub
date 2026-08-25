// Move-only ownership for one continuous-batching prefix capture.
//
// Policy metadata and checkpoint payload have different owners. This object
// keeps their resolution ordered and makes every early exit cancel exactly
// one untouched reservation. A malformed saved outcome discards only this
// transaction's payload and metadata; it never trusts an event-supplied id.

#pragma once

#include "common/concurrency/prefix_store.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace dflash::common {

template <typename Policy, typename Engine>
class BasicPrefixCaptureTxn {
public:
    enum class Resolution {
        inactive,
        saved,
        failed,
        mismatched,
    };

    BasicPrefixCaptureTxn() = default;

    BasicPrefixCaptureTxn(
            Policy & policy,
            Engine & engine,
            int policy_slot,
            PrefixCaptureTicket ticket)
        : policy_(&policy),
          engine_(&engine),
          policy_slot_(policy_slot),
          ticket_(ticket) {}

    ~BasicPrefixCaptureTxn() { cancel(); }

    BasicPrefixCaptureTxn(const BasicPrefixCaptureTxn &) = delete;
    BasicPrefixCaptureTxn & operator=(
        const BasicPrefixCaptureTxn &) = delete;

    BasicPrefixCaptureTxn(BasicPrefixCaptureTxn && other) noexcept {
        take(std::move(other));
    }

    BasicPrefixCaptureTxn & operator=(
            BasicPrefixCaptureTxn && other) noexcept {
        if (this == &other) return *this;
        cancel();
        take(std::move(other));
        return *this;
    }

    bool active() const {
        return policy_ && engine_ && policy_slot_ >= 0 && ticket_.valid();
    }

    const PrefixCaptureTicket & ticket() const { return ticket_; }

    Resolution resolve(
            const PrefixStoreEvent & event,
            const std::vector<int32_t> & prompt) {
        if (!active()) return Resolution::inactive;
        if (!event.attempted() || event.ticket != ticket_) {
            if (event.status == PrefixStoreEvent::Status::saved) {
                discard_saved();
            } else {
                cancel();
            }
            return Resolution::mismatched;
        }
        if (event.status == PrefixStoreEvent::Status::saved) {
            policy_->confirm_inline_snap(
                policy_slot_, ticket_.checkpoint.tokens, prompt,
                /*protect=*/false, event.bytes);
            clear();
            return Resolution::saved;
        }
        if (event.status == PrefixStoreEvent::Status::failed) {
            cancel();
            return Resolution::failed;
        }
        cancel();
        return Resolution::mismatched;
    }

    void cancel() {
        if (!active()) return;
        policy_->cancel_inline_snap(policy_slot_);
        clear();
    }

    void abort() {
        // A successful capture is resolved and cleared synchronously. An
        // externally aborted active transaction therefore has only armed the
        // capture and must preserve any incumbent in the reserved slot.
        cancel();
    }

private:
    void discard_saved() {
        engine_->discard_prefix_store(ticket_.checkpoint);
        policy_->abort_inline_snap(policy_slot_);
        clear();
    }

    void clear() {
        policy_ = nullptr;
        engine_ = nullptr;
        policy_slot_ = -1;
        ticket_ = {};
    }

    void take(BasicPrefixCaptureTxn && other) {
        policy_ = other.policy_;
        engine_ = other.engine_;
        policy_slot_ = other.policy_slot_;
        ticket_ = other.ticket_;
        other.clear();
    }

    Policy * policy_ = nullptr;
    Engine * engine_ = nullptr;
    int policy_slot_ = -1;
    PrefixCaptureTicket ticket_;
};

}  // namespace dflash::common
