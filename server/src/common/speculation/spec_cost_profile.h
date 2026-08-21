// Generic startup profiling protocol for monotone speculation cost tables.
#pragma once

#include "common/speculation/speculation_gate.h"

#include <functional>
#include <string>
#include <vector>

namespace dflash::common {

struct SpecProfileGrid {
    std::vector<int> tree_rows;
    std::vector<int> step_rows;
    std::vector<int> draft_lanes;
};

SpecProfileGrid build_spec_profile_grid(
    int max_concurrency, int tree_width, int max_accept,
    const std::function<int(int)> & bucket);

struct SpecCostProfileResult {
    SpecCostTables tables;
    std::string error;

    bool ok() const { return error.empty() && tables.valid(); }
};

// Returns an empty path when disk caching is disabled or no cache root exists.
std::string spec_cost_profile_cache_path(const std::string & identity);

bool load_spec_cost_profile(
    const std::string & path, const std::string & identity,
    SpecCostTables & tables, std::string & error);

bool save_spec_cost_profile(
    const std::string & path, const std::string & identity,
    const SpecCostTables & tables, std::string & error);

class SpecCostProfiler {
public:
    using Runner = std::function<double(int rows)>;

    SpecCostProfileResult profile(
        const SpecProfileGrid & grid,
        Runner tree_runner,
        Runner step_runner,
        Runner draft_runner,
        std::string speculator_id,
        int repetitions = 5) const;
};

}  // namespace dflash::common
