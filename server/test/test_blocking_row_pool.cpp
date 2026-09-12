#include "common/blocking_row_pool.h"
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <stdexcept>

using dflash::common::BlockingRowPool;
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void mixed_widths(unsigned workers) {
    BlockingRowPool pool(workers);
    // Changing the active set exposes late inactive workers: they must not
    // execute a new callback under a stale generation or acknowledge it twice.
    for (int step = 0; step < 25000; ++step) {
        const int rows = step % 3 == 0 ? 4 : step % 3 == 1 ? 24 : 1;
        std::array<std::atomic<int>, 24> hits{};
        pool.run_custom(rows, [&](int row) {
            hits[(size_t) row].fetch_add(1, std::memory_order_relaxed);
        });
        for (int row = 0; row < 24; ++row) {
            check(hits[(size_t) row].load() == (row < rows ? 1 : 0),
                  "job returned before exactly-once completion");
        }
    }
}

static void concurrent_clients() {
    BlockingRowPool pool(8);
    std::atomic<int> errors{0};
    std::vector<std::thread> clients;
    for (int client = 0; client < 4; ++client) {
        clients.emplace_back([&, client] {
            for (int step = 0; step < 1000; ++step) {
                std::array<int, 24> output{};
                pool.run_chunks(24, [&](int begin, int end) {
                    for (int row = begin; row < end; ++row) output[row] = row + client + step;
                });
                for (int row = 0; row < 24; ++row) {
                    if (output[row] != row + client + step) ++errors;
                }
            }
        });
    }
    for (auto & client : clients) client.join();
    check(errors == 0, "concurrent clients mixed jobs");
}

static void boundaries_and_idle() {
    bool rejected = false;
    try { BlockingRowPool invalid(0); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "zero workers accepted");
    BlockingRowPool pool(8);
    int calls = 0;
    pool.run_custom(0, [&](int) { ++calls; });
    pool.run_custom(-1, [&](int) { ++calls; });
    check(calls == 0, "empty job called callback");
    std::atomic<int64_t> total{0};
    pool.run_chunks(INT_MAX, [&](int begin, int end) {
        total.fetch_add((int64_t) end - begin);
    });
    check(total == INT_MAX, "large row partition overflowed");
    for (int step = 0; step < 4; ++step) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        pool.run_custom(1, [&](int row) { check(row == 0, "wrong row"); ++calls; });
    }
    check(calls == 4, "idle workers missed wakeup");
}

int main() {
    try {
        for (unsigned workers : {1u, 2u, 4u, 8u}) mixed_widths(workers);
        concurrent_clients();
        boundaries_and_idle();
        std::puts("blocking row pool: 100000 mixed-width jobs, 4000 concurrent jobs, boundaries/idle passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
