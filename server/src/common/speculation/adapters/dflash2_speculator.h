#pragma once

#include "common/dflash2_benefit.h"
#include "common/speculation/speculator.h"
#include "internal.h"

#include "ggml-backend.h"

#include <string>

namespace dflash::common {

class DFlash2Speculator final : public Speculator {
public:
    DFlash2Speculator(
        const DraftWeights & weights,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        DFlash2BenefitModelSignature signature,
        DFlash2BenefitConfig config = {});

    const std::string & score_kind() const override { return score_kind_; }
    int max_block_size() const override;
    uint32_t input_requirements() const override {
        return SpeculatorInputHidden;
    }
    bool ready() const override { return error_.empty(); }
    const std::string & error() const override { return error_; }

    bool propose(const SpeculatorBatchInput & input,
                 std::vector<SpecProposal> & output) override;

private:
    const DraftWeights & weights_;
    ggml_backend_t backend_ = nullptr;
    ggml_tensor * lm_head_ = nullptr;
    DFlash2BenefitProvider benefit_;
    std::string score_kind_;
    std::string error_;
};

}  // namespace dflash::common
