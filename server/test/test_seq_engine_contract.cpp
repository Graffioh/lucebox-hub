// Host-side test for the SeqEngine contract checker (seq_engine_contract.h).
//
// Two jobs, and the second is the one that matters:
//
//   1. FakeSeqEngine, a conforming in-memory engine, passes the checker. It
//      doubles as the reference a new backend can read to see what admit /
//      step / retire owe the scheduler, with no GPU.
//   2. Every deliberately broken variant of it is REJECTED, with the expected
//      violation named. Without this half, a checker that returned an empty
//      list unconditionally would still "pass" — the mutants are what make
//      its assertions load-bearing.
//
// No model, ggml, or GPU. Run it against a real engine (Qwen3.5's or a new
// backend's) by calling check_seq_engine_contract() from a GPU-side test.

#include "seq_engine_contract.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,     \
                         #cond);                                             \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Each flag breaks exactly one clause of the contract; the default (all
// false) is the conforming engine.
struct Faults {
    bool hard_error_when_full       = false; // busy condition reports false
    bool reuse_live_slot            = false; // hands slot 0 to every request
    bool drop_one_output            = false; // answers n-1 decode inputs
    bool pause_decode_during_prefill = false; // omits live decode outputs
    bool accept_partial_batch       = false; // steps a batch missing a slot
    bool retire_leaks               = false; // retire() never frees the slot
    bool keeps_stale_outputs        = false; // appends to `outputs` as-is
};

// ── A conforming SeqEngine with no model behind it ──────────────────────
//
// Slots hold nothing but a fed-token log; the "sampler" is arithmetic.
class FakeSeqEngine final : public SeqEngine {
public:
    explicit FakeSeqEngine(int slots, Faults faults = Faults())
        : faults_(faults),
          active_((size_t)slots, false),
          prefilling_((size_t)slots, false),
          fed_((size_t)slots) {}

    int slot_count() const override { return (int)active_.size(); }

    bool token_is_eos(int32_t token) const override { return token == kEos; }

    bool prefill_pending() const override { return pending_slot_ >= 0; }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg &,
                      int n_gen) override {
        (void)request_id;
        AdmitResult r;
        if (prompt.empty() || n_gen < 1) {
            r.error = "invalid request";   // hard error: retrying cannot help
            return r;
        }
        if (pending_slot_ >= 0) {
            r.busy = !faults_.hard_error_when_full;
            r.error = "one prefill is already pending";
            return r;
        }
        int slot = -1;
        if (faults_.reuse_live_slot) {
            slot = 0;
        } else {
            for (size_t i = 0; i < active_.size(); i++) {
                if (!active_[i]) { slot = (int)i; break; }
            }
        }
        if (slot < 0) {
            r.busy  = !faults_.hard_error_when_full;
            r.error = "all slots are live";
            return r;
        }
        active_[(size_t)slot] = true;
        prefilling_[(size_t)slot] = true;
        fed_[(size_t)slot].clear();
        pending_slot_ = slot;
        prefill_steps_left_ = 2;
        r.ok = true;
        r.slot = slot;
        r.n_gen_cap = n_gen;
        return r;
    }

    bool step(const std::vector<StepInput> & inputs,
              std::vector<StepOutput> & outputs) override {
        if (!faults_.keeps_stale_outputs) outputs.clear();
        if (!faults_.accept_partial_batch &&
            (int)inputs.size() != decoding_count()) {
            return false;   // never advance state for a slot we were not given
        }
        if (inputs.empty() && pending_slot_ < 0) return true;
        size_t emit = inputs.size();
        if (faults_.drop_one_output && emit > 0) emit--;
        if (faults_.pause_decode_during_prefill && pending_slot_ >= 0) {
            emit = 0;
        }
        for (size_t i = 0; i < emit; i++) {
            const StepInput & in = inputs[i];
            StepOutput out;
            out.slot = in.slot;
            if (in.slot < 0 || in.slot >= slot_count() ||
                !active_[(size_t)in.slot] ||
                prefilling_[(size_t)in.slot] || in.token < 0) {
                out.failed = true;
                outputs.push_back(out);
                continue;
            }
            fed_[(size_t)in.slot].push_back(in.token);
            out.token  = kFirstToken + in.slot +
                         (int32_t)fed_[(size_t)in.slot].size();
            outputs.push_back(out);
        }

        if (pending_slot_ >= 0 && --prefill_steps_left_ == 0) {
            const int slot = pending_slot_;
            pending_slot_ = -1;
            prefilling_[(size_t)slot] = false;
            StepOutput out;
            out.slot = slot;
            out.token = kFirstToken + slot;
            out.prefill_done = true;
            out.prefill_s = 0.001;
            outputs.push_back(out);
        }
        return true;
    }

    void retire(int slot) override {
        if (slot < 0 || slot >= slot_count()) return;
        if (faults_.retire_leaks) return;
        if (pending_slot_ == slot) {
            pending_slot_ = -1;
            prefill_steps_left_ = 0;
        }
        active_[(size_t)slot] = false;
        prefilling_[(size_t)slot] = false;
        fed_[(size_t)slot].clear();
    }

private:
    static constexpr int32_t kEos        = 2;
    static constexpr int32_t kFirstToken = 100;   // generated ids never hit EOS

    int decoding_count() const {
        int n = 0;
        for (size_t i = 0; i < active_.size(); i++) {
            n += active_[i] && !prefilling_[i] ? 1 : 0;
        }
        return n;
    }

    Faults                            faults_;
    std::vector<bool>                 active_;
    std::vector<bool>                 prefilling_;
    std::vector<std::vector<int32_t>> fed_;
    int                               pending_slot_ = -1;
    int                               prefill_steps_left_ = 0;
};

static void print_violations(const char * label,
                             const std::vector<std::string> & v) {
    for (const std::string & s : v) {
        std::fprintf(stderr, "  [%s] %s\n", label, s.c_str());
    }
}

static bool mentions(const std::vector<std::string> & v,
                     const std::string & needle) {
    for (const std::string & s : v) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

int main() {
    // ── A conforming engine passes, at several slot counts ──────────────
    // slot_count()==1 matters on its own: the checker must not depend on
    // there being a second slot to omit from a batch.
    for (const int slots : {1, 2, 4, 8}) {
        FakeSeqEngine engine(slots);
        const std::vector<std::string> v = check_seq_engine_contract(engine);
        if (!v.empty()) print_violations("conforming", v);
        CHECK(v.empty());
    }

    // ── Every broken clause is caught, and named ────────────────────────
    struct Case {
        const char * label;
        bool Faults::*broken;   // the one clause this variant violates
        const char * expect;    // substring of the violation it must produce
    };
    const Case cases[] = {
        {"hard-error-when-full", &Faults::hard_error_when_full,
         "must report busy=true"},
        {"reuse-live-slot", &Faults::reuse_live_slot,
         "reused a slot that is live"},
        {"drop-one-output", &Faults::drop_one_output,
         "one output per input"},
        {"pause-decode-during-prefill", &Faults::pause_decode_during_prefill,
         "prefill step left a decoding slot without an output"},
        {"accept-partial-batch", &Faults::accept_partial_batch,
         "omits a live slot"},
        {"retire-leaks", &Faults::retire_leaks,
         "after a retire() freed a slot"},
        {"keeps-stale-outputs", &Faults::keeps_stale_outputs,
         "must clear `outputs`"},
    };
    for (const Case & c : cases) {
        Faults faults;
        faults.*c.broken = true;
        FakeSeqEngine engine(4, faults);
        const std::vector<std::string> v = check_seq_engine_contract(engine);
        if (!mentions(v, c.expect)) {
            std::fprintf(stderr,
                         "FAIL %s: expected a violation mentioning \"%s\"\n",
                         c.label, c.expect);
            print_violations(c.label, v);
        }
        g_checks++;
        CHECK(mentions(v, c.expect));
    }

    std::printf("test_seq_engine_contract: %d checks passed\n", g_checks);
    return 0;
}
