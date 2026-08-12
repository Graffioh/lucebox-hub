// oflash_supervisor.cpp — trainer sidecar spawn/supervise/relay.

#include "oflash_supervisor.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include <unistd.h>
#endif

namespace dflash::common::oflash {

using json = nlohmann::json;

#if defined(_WIN32)

bool OFlashSupervisor::start(const OFlashSupervisorConfig &) { return false; }
void OFlashSupervisor::stop() {}
bool OFlashSupervisor::trainer_alive() const { return false; }
bool OFlashSupervisor::trainer_disabled() const { return false; }
uint64_t OFlashSupervisor::respawns() const { return 0; }
bool OFlashSupervisor::take_pending_swap(OFlashPendingSwap &) { return false; }
bool OFlashSupervisor::clear_pending_swaps() { return false; }
void OFlashSupervisor::begin_rollback(uint64_t) {}
bool OFlashSupervisor::rollback_pending() const { return false; }
void OFlashSupervisor::send_line(const std::string &) {}
void OFlashSupervisor::run() {}
bool OFlashSupervisor::spawn_once(std::string &) { return false; }
void OFlashSupervisor::reap_child(bool) {}
void OFlashSupervisor::drain_outbox_locked() {}

#else

namespace {

void close_inherited_fds_except(int keep_fd) {
#if defined(__linux__) && defined(SYS_close_range)
    bool range_ok = true;
    if (keep_fd > 3) {
        range_ok = ::syscall(SYS_close_range, 3u,
                             (unsigned)keep_fd - 1, 0u) == 0;
    }
    range_ok = ::syscall(SYS_close_range, (unsigned)keep_fd + 1,
                         UINT_MAX, 0u) == 0 && range_ok;
    if (range_ok) return;
#endif
    const long open_max = ::sysconf(_SC_OPEN_MAX);
    const int limit = open_max > 0 ? (int)open_max : 65536;
    for (int fd = 3; fd < limit; ++fd) {
        if (fd != keep_fd) ::close(fd);
    }
}

}  // namespace

bool OFlashSupervisor::start(const OFlashSupervisorConfig & cfg) {
    if (cfg.trainer_bin.empty()) return false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (thread_.joinable()) return false;
        cfg_ = cfg;
        stopping_ = false;
        alive_ = false;
        trainer_disabled_ = false;
        outbox_.clear();
        has_pending_ = false;
        rollback_pending_ = false;
        rollback_generation_ = 0;
        respawns_ = 0;
    }
    try {
        thread_ = std::thread([this] { run(); });
    } catch (...) {
        std::fprintf(stderr, "[oflash] supervisor thread spawn failed\n");
        return false;
    }
    return true;
}

void OFlashSupervisor::stop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_ && !thread_.joinable()) return;
        stopping_ = true;
        outbox_.push_back("quit");
    }
    if (thread_.joinable()) thread_.join();
}

bool OFlashSupervisor::trainer_alive() const {
    std::lock_guard<std::mutex> lock(mu_);
    return alive_;
}

bool OFlashSupervisor::trainer_disabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return trainer_disabled_;
}

uint64_t OFlashSupervisor::respawns() const {
    std::lock_guard<std::mutex> lock(mu_);
    return respawns_;
}

bool OFlashSupervisor::take_pending_swap(OFlashPendingSwap & out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!has_pending_) return false;
    out = pending_;
    has_pending_ = false;
    return true;
}

bool OFlashSupervisor::clear_pending_swaps() {
    std::lock_guard<std::mutex> lock(mu_);
    const bool cleared = has_pending_;
    pending_ = {};
    has_pending_ = false;
    return cleared;
}

void OFlashSupervisor::begin_rollback(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mu_);
    pending_ = {};
    has_pending_ = false;
    rollback_pending_ = true;
    rollback_generation_ = generation;
    outbox_.push_back("rollback " + std::to_string(generation));
}

bool OFlashSupervisor::rollback_pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    return rollback_pending_;
}

void OFlashSupervisor::send_line(const std::string & line) {
    std::lock_guard<std::mutex> lock(mu_);
    outbox_.push_back(line);
}

bool OFlashSupervisor::spawn_once(std::string & error) {
    int in_pipe[2]  = {-1, -1};   // parent writes → child stdin
    int strm_pipe[2] = {-1, -1};  // child writes → parent reads
    if (::pipe(in_pipe) != 0 || ::pipe(strm_pipe) != 0) {
        error = "pipe() failed";
        if (in_pipe[0] >= 0) { ::close(in_pipe[0]); ::close(in_pipe[1]); }
        return false;
    }

    std::vector<std::string> argv_s;
    argv_s.push_back(cfg_.trainer_bin);
    argv_s.push_back(cfg_.drafter_path);
    for (const auto & a : cfg_.args) argv_s.push_back(a);
    argv_s.push_back("--stream-fd=" + std::to_string(strm_pipe[1]));

    const pid_t pid = ::fork();
    if (pid < 0) {
        error = "fork() failed";
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        ::close(strm_pipe[0]); ::close(strm_pipe[1]);
        return false;
    }
    if (pid == 0) {
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        ::close(strm_pipe[0]);
        // The server is multi-threaded and owns listening/client sockets.
        // The trainer needs only stdin, stdout/stderr, and its event stream;
        // close every other inherited descriptor before exec.
        close_inherited_fds_except(strm_pipe[1]);
        std::vector<char *> argv;
        argv.reserve(argv_s.size() + 1);
        for (auto & s : argv_s) argv.push_back(const_cast<char *>(s.c_str()));
        argv.push_back(nullptr);
        ::execv(argv[0], argv.data());
        std::fprintf(stderr, "[oflash-trainer] execv %s failed: %s\n",
                     argv[0], std::strerror(errno));
        ::_exit(127);
    }

    ::close(in_pipe[0]);
    ::close(strm_pipe[1]);
    child_pid_    = pid;
    child_stdin_  = in_pipe[1];
    child_stream_ = strm_pipe[0];

    // Ready handshake: one int32 (0 = ok). The trainer sends it right after
    // attaching the ring, BEFORE loading torch/the mirror, so this is fast.
    // Poll in slices so stop() never waits the whole handshake timeout.
    int pr = 0;
    for (int waited = 0; waited < cfg_.ready_timeout_ms; waited += 200) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) { reap_child(false); error = "stopping"; return false; }
        }
        struct pollfd pfd { child_stream_, POLLIN, 0 };
        pr = ::poll(&pfd, 1, 200);
        if (pr != 0) break;
    }
    if (pr <= 0) {
        error = pr == 0 ? "trainer ready handshake timed out"
                        : "poll on trainer stream failed";
        reap_child(false);
        return false;
    }
    int32_t status = -1;
    ssize_t n = ::read(child_stream_, &status, sizeof(status));
    if (n != (ssize_t)sizeof(status) || status != 0) {
        error = "trainer init failed (status=" + std::to_string(status) + ")";
        reap_child(false);
        return false;
    }
    return true;
}

void OFlashSupervisor::reap_child(bool graceful) {
    if (child_pid_ < 0) return;
    if (child_stdin_ >= 0) { ::close(child_stdin_); child_stdin_ = -1; }
    const pid_t pid = (pid_t)child_pid_;
    const int grace_ms = graceful ? 2000 : 200;
    int waited = 0;
    int wstatus = 0;
    while (waited < grace_ms) {
        const pid_t wr = ::waitpid(pid, &wstatus, WNOHANG);
        if (wr == pid || (wr == -1 && errno == ECHILD)) {
            goto reaped;
        }
        ::usleep(50 * 1000);
        waited += 50;
    }
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &wstatus, 0);
reaped:
    if (child_stream_ >= 0) { ::close(child_stream_); child_stream_ = -1; }
    child_pid_ = -1;
}

// mu_ must be held. Writes queued lines to the child's stdin; EPIPE just
// means the read loop will notice EOF shortly.
void OFlashSupervisor::drain_outbox_locked() {
    if (child_stdin_ < 0) { outbox_.clear(); return; }
    for (const auto & line : outbox_) {
        const std::string msg = line + "\n";
        ssize_t off = 0;
        while (off < (ssize_t)msg.size()) {
            const ssize_t n = ::write(child_stdin_, msg.data() + off,
                                      msg.size() - (size_t)off);
            if (n <= 0) {
                if (errno == EINTR) continue;
                return;  // child gone; keep remaining lines dropped
            }
            off += n;
        }
    }
    outbox_.clear();
}

void OFlashSupervisor::run() {
    // Writes to a dead child's pipe must surface as EPIPE, not SIGPIPE.
    // http_server.cpp does the same process-wide when serving starts; the
    // supervisor may run before that.
    ::signal(SIGPIPE, SIG_IGN);

    uint64_t attempts = 0;
    int backoff_ms = 1000;
    std::string line_buf;

    while (true) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) break;
        }
        std::string error;
        if (!spawn_once(error)) {
            std::fprintf(stderr, "[oflash] trainer spawn failed: %s\n",
                         error.c_str());
            attempts++;
            if (attempts > (uint64_t)cfg_.max_respawns) {
                std::fprintf(stderr,
                    "[oflash] trainer respawn limit reached; capture-only\n");
                break;
            }
            for (int slept = 0; slept < backoff_ms; slept += 100) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    if (stopping_) return;
                }
                ::usleep(100 * 1000);
            }
            backoff_ms *= 2;
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            alive_ = true;
            trainer_disabled_ = false;
            if (attempts > 0) respawns_++;
            // A successful pipe write to the old child did not prove that it
            // processed the rollback. Preserve and resend until acknowledged.
            if (rollback_pending_) {
                outbox_.push_back(
                    "rollback " + std::to_string(rollback_generation_));
            }
            drain_outbox_locked();
        }
        std::fprintf(stderr, "[oflash] trainer ready (pid=%ld)\n", child_pid_);

        // Read loop: JSON lines from the child; 200ms poll so the outbox
        // and the stop flag stay responsive.
        line_buf.clear();
        bool child_eof = false;
        while (!child_eof) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stopping_) break;
                drain_outbox_locked();
            }
            struct pollfd pfd { child_stream_, POLLIN, 0 };
            const int pr = ::poll(&pfd, 1, 200);
            if (pr < 0) {
                if (errno == EINTR) continue;
                child_eof = true;
                break;
            }
            if (pr == 0) continue;
            char buf[4096];
            const ssize_t n = ::read(child_stream_, buf, sizeof(buf));
            if (n <= 0) { child_eof = true; break; }
            line_buf.append(buf, (size_t)n);
            size_t nl;
            while ((nl = line_buf.find('\n')) != std::string::npos) {
                const std::string line = line_buf.substr(0, nl);
                line_buf.erase(0, nl + 1);
                json j = json::parse(line, nullptr, false);
                if (j.is_discarded() || !j.is_object()) continue;
                const std::string ev = j.value("event", "");
                if (ev == "swap_ready") {
                    std::lock_guard<std::mutex> lock(mu_);
                    if (rollback_pending_) {
                        std::fprintf(stderr,
                            "[oflash] discarding adapter announcement during "
                            "rollback barrier\n");
                    } else {
                        pending_.path = j.value("path", "");
                        pending_.generation =
                            j.value("generation", (uint64_t)0);
                        has_pending_ = !pending_.path.empty();
                    }
                } else if (ev == "rollback_ack") {
                    const uint64_t generation =
                        j.value("generation", (uint64_t)0);
                    std::lock_guard<std::mutex> lock(mu_);
                    if (rollback_pending_ &&
                        generation == rollback_generation_) {
                        // Stream ordering proves every pre-rollback export was
                        // parsed (and discarded) before this acknowledgement.
                        pending_ = {};
                        has_pending_ = false;
                        rollback_pending_ = false;
                    }
                } else if (ev == "training_disabled") {
                    std::lock_guard<std::mutex> lock(mu_);
                    trainer_disabled_ = true;
                    std::fprintf(stderr,
                        "[oflash-trainer] training disabled: %s\n",
                        j.value("reason", "unspecified").c_str());
                } else if (ev == "log") {
                    std::fprintf(stderr, "[oflash-trainer] %s\n",
                                 j.value("message", "").c_str());
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            alive_ = false;
            if (stopping_) break;
        }
        std::fprintf(stderr, "[oflash] trainer exited; will respawn\n");
        reap_child(false);
        attempts++;
        if (attempts > (uint64_t)cfg_.max_respawns) {
            std::fprintf(stderr,
                "[oflash] trainer respawn limit reached; capture-only\n");
            break;
        }
        for (int slept = 0; slept < backoff_ms; slept += 100) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stopping_) return;
            }
            ::usleep(100 * 1000);
        }
        backoff_ms *= 2;
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        drain_outbox_locked();  // best-effort "quit"
        alive_ = false;
    }
    reap_child(true);
}

#endif  // _WIN32

}  // namespace dflash::common::oflash
