#include "common/speculation/spec_cost_profile.h"
#include "host_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <unistd.h>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    const SpecProfileGrid grid = build_spec_profile_grid(
        3, 16, 4, [](int lanes) { return lanes; });
    CHECK((grid.tree_rows == std::vector<int>{16, 32, 48}));
    CHECK((grid.draft_lanes == std::vector<int>{1, 2, 3}));
    CHECK(std::binary_search(grid.step_rows.begin(), grid.step_rows.end(), 1));
    CHECK(std::binary_search(grid.step_rows.begin(), grid.step_rows.end(), 12));
    CHECK(std::is_sorted(grid.step_rows.begin(), grid.step_rows.end()));
    CHECK(std::adjacent_find(grid.step_rows.begin(), grid.step_rows.end()) ==
          grid.step_rows.end());

    const SpecProfileGrid bucketed = build_spec_profile_grid(
        5, 7, 7, [](int lanes) {
            if (lanes <= 1) return 1;
            if (lanes <= 2) return 2;
            if (lanes <= 4) return 4;
            return 6;
        });
    CHECK((bucketed.tree_rows == std::vector<int>{7, 14, 28, 42}));
    CHECK(bucketed.step_rows.back() == 35);
    CHECK(build_spec_profile_grid(0, 16, 4, {}).tree_rows.empty());

    SpecProfileGrid small;
    small.tree_rows = {2, 1, 2};
    small.step_rows = {4};
    small.draft_lanes = {3};
    std::unordered_map<int, int> tree_calls;
    const double noise[] = {-2.0, 1.0, 0.0, 2.0, -1.0};
    auto tree_runner = [&](int index) {
        const int call = tree_calls[index]++;
        if (call == 0) return 10000.0;
        const double intended = index == 1 ? 10.0 : 5.0;
        return intended + noise[(call - 1) % 5];
    };
    int step_calls = 0;
    int draft_calls = 0;
    const SpecCostProfileResult profiled = SpecCostProfiler{}.profile(
        small,
        tree_runner,
        [&](int index) {
            ++step_calls;
            return index == 4 ? 20.0 : 0.0;
        },
        [&](int index) {
            ++draft_calls;
            return index == 3 ? 30.0 : 0.0;
        },
        "adapter-score-v1");
    CHECK(profiled.ok());
    CHECK(profiled.tables.speculator_id == "adapter-score-v1");
    CHECK((profiled.tables.tree_cost.indices == std::vector<int>{1, 2}));
    CHECK(profiled.tables.tree_cost.costs[0] == 10.0);
    CHECK(profiled.tables.tree_cost.costs[1] == 10.0);
    CHECK(tree_calls[1] == 6 && tree_calls[2] == 6);
    CHECK(step_calls == 6);
    CHECK(draft_calls == 6);

    SpecCostProfileResult bad = SpecCostProfiler{}.profile(
        {}, [](int) { return 1.0; }, [](int) { return 1.0; },
        [](int) { return 1.0; }, "adapter-score-v1");
    CHECK(!bad.ok() && !bad.error.empty());
    bad = SpecCostProfiler{}.profile(
        small, [](int) { return 1.0; }, [](int) { return 1.0; },
        [](int) { return 1.0; }, "", 5);
    CHECK(!bad.ok());
    int invalid_calls = 0;
    bad = SpecCostProfiler{}.profile(
        {{1}, {1}, {1}},
        [&](int) {
            ++invalid_calls;
            return invalid_calls == 2
                ? std::numeric_limits<double>::quiet_NaN() : 1.0;
        },
        [](int) { return 1.0; },
        [](int) { return 1.0; },
        "adapter-score-v1");
    CHECK(!bad.ok());
    CHECK(bad.tables.tree_cost.indices.empty());

    const std::filesystem::path cache_dir =
        std::filesystem::temp_directory_path() /
        ("dflash-spec-profile-test-" + std::to_string(getpid()));
    const std::filesystem::path cache_path = cache_dir / "profile";
    std::string cache_error;
    CHECK(save_spec_cost_profile(
        cache_path.string(), "identity-a", profiled.tables, cache_error));
    SpecCostTables loaded;
    CHECK(load_spec_cost_profile(
        cache_path.string(), "identity-a", loaded, cache_error));
    CHECK(loaded.speculator_id == profiled.tables.speculator_id);
    CHECK(loaded.tree_cost.indices == profiled.tables.tree_cost.indices);
    CHECK(loaded.tree_cost.costs == profiled.tables.tree_cost.costs);
    CHECK(!load_spec_cost_profile(
        cache_path.string(), "identity-b", loaded, cache_error));
    CHECK(loaded.tree_cost.indices.empty());

    {
        std::ofstream corrupt(cache_path, std::ios::trunc);
        corrupt << "not a profile\n";
    }
    CHECK(!load_spec_cost_profile(
        cache_path.string(), "identity-a", loaded, cache_error));
    std::error_code remove_error;
    std::filesystem::remove_all(cache_dir, remove_error);

    std::printf("spec cost profile tests passed: %d checks\n", g_checks);
    return 0;
}
