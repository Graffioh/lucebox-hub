#include "common/speculation/spec_cost_profile.h"
#include "common/sha1.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace dflash::common {
namespace {

void sort_unique_positive(std::vector<int> & values) {
    values.erase(
        std::remove_if(
            values.begin(), values.end(),
            [](int value) { return value <= 0; }),
        values.end());
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

struct SeriesResult {
    SpecCostSeries table;
    std::string error;
};

SeriesResult profile_monotonic_costs(
        std::vector<int> indices,
        const SpecCostProfiler::Runner & runner,
        int repetitions) {
    SeriesResult result;
    if (!runner) {
        result.error = "profiling runner is missing";
        return result;
    }
    if (repetitions <= 0) {
        result.error = "profiling repetitions must be positive";
        return result;
    }
    sort_unique_positive(indices);
    if (indices.empty()) {
        result.error = "profiling grid is empty";
        return result;
    }

    result.table.indices = indices;
    result.table.costs.reserve(indices.size());
    for (int index : indices) {
        (void)runner(index);
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(repetitions));
        for (int rep = 0; rep < repetitions; ++rep) {
            const double sample = runner(index);
            if (!std::isfinite(sample) || sample <= 0.0) {
                result.error = "profiling runner returned an invalid cost";
                result.table = {};
                return result;
            }
            samples.push_back(sample);
        }
        std::sort(samples.begin(), samples.end());
        double median = samples[static_cast<size_t>(repetitions) / 2];
        if (repetitions % 2 == 0) {
            median = 0.5 * (
                samples[static_cast<size_t>(repetitions) / 2 - 1] +
                median);
        }
        if (!result.table.costs.empty()) {
            median = std::max(median, result.table.costs.back());
        }
        result.table.costs.push_back(median);
    }
    return result;
}


constexpr int kProfileCacheVersion = 1;
constexpr size_t kMaxSeriesEntries = 4096;

bool read_series(
        std::istream & input, const char * expected,
        SpecCostSeries & series) {
    std::string name;
    size_t count = 0;
    if (!(input >> name >> count) || name != expected ||
        count == 0 || count > kMaxSeriesEntries) {
        return false;
    }
    series.indices.resize(count);
    series.costs.resize(count);
    for (size_t i = 0; i < count; ++i) {
        if (!(input >> series.indices[i] >> series.costs[i])) return false;
    }
    return true;
}

void write_series(
        std::ostream & output, const char * name,
        const SpecCostSeries & series) {
    output << name << ' ' << series.indices.size() << '\n';
    output << std::setprecision(17);
    for (size_t i = 0; i < series.indices.size(); ++i) {
        output << series.indices[i] << ' ' << series.costs[i] << '\n';
    }
}

std::string hex_sha1(const std::string & value) {
    uint8_t digest[20];
    sha1_hash(value.data(), value.size(), digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : digest) out << std::setw(2) << (unsigned)byte;
    return out.str();
}

}  // namespace

std::string spec_cost_profile_cache_path(const std::string & identity) {
    if (identity.empty()) return {};
    if (const char * configured = std::getenv("DFLASH_SPEC_PROFILE_PATH")) {
        if (std::string(configured) == "0") return {};
        if (*configured) return configured;
    }
    std::filesystem::path root;
    if (const char * xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        root = xdg;
    } else if (const char * home = std::getenv("HOME"); home && *home) {
        root = std::filesystem::path(home) / ".cache";
    } else {
        return {};
    }
    return (root / "lucebox" /
            ("spec-cost-v1-" + hex_sha1(identity) + ".profile")).string();
}

bool load_spec_cost_profile(
        const std::string & path, const std::string & identity,
        SpecCostTables & tables, std::string & error) {
    tables = {};
    error.clear();
    if (path.empty()) {
        error = "profile cache disabled";
        return false;
    }
    std::ifstream input(path);
    if (!input) {
        error = "profile cache miss";
        return false;
    }
    std::string magic;
    int version = 0;
    std::string stored_identity;
    SpecCostTables loaded;
    if (!(input >> magic >> version) || magic != "dflash-spec-cost-profile" ||
        version != kProfileCacheVersion ||
        !(input >> std::quoted(stored_identity)) ||
        !(input >> std::quoted(loaded.speculator_id)) ||
        !read_series(input, "tree", loaded.tree_cost) ||
        !read_series(input, "step", loaded.step_cost) ||
        !read_series(input, "draft", loaded.draft_cost)) {
        error = "invalid profile cache";
        return false;
    }
    input >> std::ws;
    if (!input.eof() || stored_identity != identity || !loaded.valid()) {
        error = stored_identity != identity
            ? "profile cache identity mismatch" : "invalid profile cache";
        return false;
    }
    tables = std::move(loaded);
    return true;
}

bool save_spec_cost_profile(
        const std::string & path, const std::string & identity,
        const SpecCostTables & tables, std::string & error) {
    error.clear();
    if (path.empty()) return true;
    if (identity.empty() || !tables.valid()) {
        error = "refusing to save invalid profile cache";
        return false;
    }
    const std::filesystem::path destination(path);
    std::error_code ec;
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path(), ec);
        if (ec) {
            error = "could not create profile cache directory";
            return false;
        }
    }
    const std::filesystem::path temporary =
        destination.string() + ".tmp." + std::to_string(getpid());
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            error = "could not open temporary profile cache";
            return false;
        }
        output << "dflash-spec-cost-profile " << kProfileCacheVersion << '\n'
               << std::quoted(identity) << '\n'
               << std::quoted(tables.speculator_id) << '\n';
        write_series(output, "tree", tables.tree_cost);
        write_series(output, "step", tables.step_cost);
        write_series(output, "draft", tables.draft_cost);
        output.flush();
        if (!output) {
            error = "could not write profile cache";
            output.close();
            std::filesystem::remove(temporary, ec);
            return false;
        }
    }
    std::filesystem::rename(temporary, destination, ec);
    if (ec) {
        error = "could not publish profile cache";
        std::filesystem::remove(temporary, ec);
        return false;
    }
    return true;
}

SpecProfileGrid build_spec_profile_grid(
        int max_concurrency, int tree_width, int max_accept,
        const std::function<int(int)> & bucket) {
    SpecProfileGrid grid;
    if (max_concurrency <= 0 || tree_width <= 0 || max_accept <= 0) {
        return grid;
    }
    auto bucketed = [&](int lanes) {
        if (lanes <= 0) return 0;
        return std::max(lanes, bucket ? bucket(lanes) : lanes);
    };
    for (int lanes = 1; lanes <= max_concurrency; ++lanes) {
        grid.tree_rows.push_back(bucketed(lanes) * tree_width);
        grid.draft_lanes.push_back(lanes);
    }
    for (int concurrency = 1; concurrency <= max_concurrency; ++concurrency) {
        grid.step_rows.push_back(bucketed(concurrency));
        for (int spec_lanes = 1; spec_lanes <= concurrency; ++spec_lanes) {
            const int ar_rows = bucketed(concurrency - spec_lanes);
            for (int accepted = spec_lanes;
                 accepted <= spec_lanes * max_accept; ++accepted) {
                grid.step_rows.push_back(accepted + ar_rows);
            }
        }
    }
    sort_unique_positive(grid.tree_rows);
    sort_unique_positive(grid.step_rows);
    sort_unique_positive(grid.draft_lanes);
    return grid;
}

SpecCostProfileResult SpecCostProfiler::profile(
        const SpecProfileGrid & grid,
        Runner tree_runner,
        Runner step_runner,
        Runner draft_runner,
        std::string speculator_id,
        int repetitions) const {
    SpecCostProfileResult result;
    if (speculator_id.empty()) {
        result.error = "speculator id is empty";
        return result;
    }

    SeriesResult tree = profile_monotonic_costs(
        grid.tree_rows, tree_runner, repetitions);
    SeriesResult step = profile_monotonic_costs(
        grid.step_rows, step_runner, repetitions);
    SeriesResult draft = profile_monotonic_costs(
        grid.draft_lanes, draft_runner, repetitions);
    if (!tree.error.empty() || !step.error.empty() || !draft.error.empty()) {
        result.error = tree.error + step.error + draft.error;
        return result;
    }

    result.tables.tree_cost = std::move(tree.table);
    result.tables.step_cost = std::move(step.table);
    result.tables.draft_cost = std::move(draft.table);
    result.tables.speculator_id = std::move(speculator_id);
    return result;
}

}  // namespace dflash::common
