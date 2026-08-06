// oflash_supervisor.h — trainer sidecar lifecycle (OFLASH.md §3, §6.2).
//
// Spawns the (Python) trainer, watches it, and relays control messages.
// Deliberately NOT BackendIpcProcess: that class blocks forever in waitpid
// and treats spawn failure as fatal at its call sites, while OFlash's
// contract is "this feature must never take down inference" — spawn failure
// degrades to capture-only, a wedged child is killed after a grace period,
// and a dead child is respawned with backoff (3 attempts max).
//
// Transport (mirrors the repo's daemon conventions, not UDS — none exists):
//   parent → child  newline text commands on the child's stdin
//                   ("promote <gen>", "rollback <gen>", "disable", "quit")
//   child → parent  one int32 ready-status after attaching the ring, then
//                   newline JSON events on the inherited --stream-fd:
//                   {"event":"swap_ready","path":"...","generation":N}
//
// All child I/O happens on the supervisor thread; the decode thread only
// polls take_pending_swap() (mutex'd mailbox) and enqueues outbound lines.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dflash::common::oflash {

struct OFlashSupervisorConfig {
    std::string trainer_bin;              // executable path (execv, no PATH)
    std::string drafter_path;             // positional arg for the trainer
    std::vector<std::string> args;        // extra --flag=value args
    int ready_timeout_ms = 60000;
    int max_respawns     = 3;
};

struct OFlashPendingSwap {
    std::string path;
    uint64_t generation = 0;
};

class OFlashSupervisor {
public:
    OFlashSupervisor() = default;
    ~OFlashSupervisor() { stop(); }
    OFlashSupervisor(const OFlashSupervisor &) = delete;
    OFlashSupervisor & operator=(const OFlashSupervisor &) = delete;

    // Launches the supervisor thread; returns immediately. The thread owns
    // spawn + handshake + read loop + respawn. False only on thread-spawn
    // failure or empty trainer_bin.
    bool start(const OFlashSupervisorConfig & cfg);

    // Ask the child to exit, reap with grace, join the thread. Idempotent.
    void stop();

    bool trainer_alive() const;
    uint64_t respawns() const;

    // Decode-thread API: newest swap request, if any (newer overwrites
    // older — only the latest adapter matters).
    bool take_pending_swap(OFlashPendingSwap & out);

    // Queue a control line for the child ("promote 3", "rollback 3", ...).
    void send_line(const std::string & line);

private:
    void run();                                   // supervisor thread body
    bool spawn_once(std::string & error);         // fork/exec + handshake
    void reap_child(bool graceful);
    void drain_outbox_locked();

    OFlashSupervisorConfig cfg_;
    std::thread thread_;
    mutable std::mutex mu_;
    std::vector<std::string> outbox_;
    OFlashPendingSwap pending_;
    bool has_pending_ = false;
    bool stopping_ = false;
    bool alive_ = false;
    uint64_t respawns_ = 0;

    // POSIX child state (unused on Windows; feature is stubbed there).
    long child_pid_ = -1;
    int  child_stdin_ = -1;
    int  child_stream_ = -1;
};

}  // namespace dflash::common::oflash
