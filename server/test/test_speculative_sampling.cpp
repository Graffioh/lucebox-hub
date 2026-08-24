#include "common/sampler.h"
#include "common/speculative_sampling.h"
#include "host_check.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

namespace {

bool near(float actual, float expected, float tolerance = 1e-6f) {
    return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    {
        const float logits[] = {3.0f, 2.0f, 1.0f};
        SamplerCfg cfg;
        cfg.temp = 0.0f;
        cfg.freq_pen = 2.5f;
        const std::vector<int32_t> history = {0};
        SamplerDistribution dist;
        CHECK(build_sampler_distribution(logits, 3, cfg, history, dist));
        CHECK(dist.valid());
        CHECK(dist.support.size() == 1);
        CHECK(dist.support[0] == 1);
        CHECK(near(dist.probability_of(1), 1.0f));
    }

    {
        const float logits[] = {1.2f, -0.5f, 2.1f, 0.3f, 1.7f};
        SamplerCfg cfg;
        cfg.temp = 0.7f;
        cfg.top_k = 4;
        cfg.top_p = 0.82f;
        cfg.rep_pen = 1.15f;
        cfg.freq_pen = 0.2f;
        cfg.pres_pen = 0.1f;
        const std::vector<int32_t> history = {2, 4, 2};
        SamplerDistribution dist;
        CHECK(build_sampler_distribution(logits, 5, cfg, history, dist));
        CHECK(dist.valid());
        CHECK(!dist.support.empty());

        std::mt19937_64 direct_rng(9001);
        std::mt19937_64 dist_rng(9001);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int i = 0; i < 64; ++i) {
            const int direct = sample_logits(logits, 5, cfg, history, direct_rng);
            const int materialized =
                sample_distribution(dist, uniform(dist_rng));
            CHECK(direct == materialized);
        }
    }

    {
        SamplerDistribution dist;
        dist.probabilities = {0.1f, 0.2f, 0.3f, 0.4f};
        dist.support = {0, 1, 2, 3};
        const int32_t eos_ids[] = {1, 3};
        CHECK(redirect_distribution_mass(dist, eos_ids, 2, 2));
        CHECK(near(dist.probability_of(0), 0.1f));
        CHECK(near(dist.probability_of(1), 0.0f));
        CHECK(near(dist.probability_of(2), 0.9f));
        CHECK(near(dist.probability_of(3), 0.0f));
        CHECK(dist.valid());
    }

    {
        SamplerDistribution target;
        target.probabilities = {0.6f, 0.3f, 0.1f};
        target.support = {0, 1, 2};
        SparseProposalDistribution proposal;
        proposal.token_ids = {0, 1, 2};
        proposal.probabilities = {0.2f, 0.5f, 0.3f};
        CHECK(proposal.valid(3));
        CHECK(near(speculative_acceptance_probability(target, proposal, 0), 1.0f));
        CHECK(near(speculative_acceptance_probability(target, proposal, 1), 0.6f));
        CHECK(near(speculative_acceptance_probability(target, proposal, 2), 1.0f / 3.0f));
        int token = -1;
        CHECK(sample_speculative_residual(target, proposal, 0.0, token));
        CHECK(token == 0);
        CHECK(sample_speculative_residual(target, proposal, 0.999, token));
        CHECK(token == 0);
    }

    {
        SamplerDistribution target;
        target.probabilities = {0.5f, 0.3f, 0.2f};
        target.support = {0, 1, 2};
        SparseProposalDistribution proposal;
        proposal.token_ids = {0, 2};
        proposal.probabilities = {0.25f, 0.75f};
        CHECK(proposal.valid(3));
        CHECK(near(speculative_acceptance_probability(target, proposal, 2),
                   0.2f / 0.75f));
        int token = -1;
        CHECK(sample_speculative_residual(target, proposal, 0.0, token));
        CHECK(token == 0);
        CHECK(sample_speculative_residual(target, proposal, 0.999, token));
        CHECK(token == 1);
    }

    std::printf("speculative sampling tests passed: %d checks\n", g_checks);
    return 0;
}
