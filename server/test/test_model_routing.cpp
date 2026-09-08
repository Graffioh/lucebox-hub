// Real HTTP + existing batch schedulers, with deterministic host-only engines.
// Protects cross-model admission, tokenization, retirement and worker ownership.
#include "CppUnitTestFramework.hpp"
#include "server/http_server.h"
#include "common/concurrency/seq_engine.h"
#include "gguf.h"

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace dflash::common;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
struct ModelRoutingFixture {};
#define ROUTING_CHECK(expression) do { \
    if (!(expression)) throw std::runtime_error( \
        std::string("routing check at line ") + std::to_string(__LINE__) + ": " #expression); \
} while (false)

class Socket {
public:
    explicit Socket(int fd) : fd_(fd) { ROUTING_CHECK(fd >= 0); }
    ~Socket() { close(); }
    Socket(Socket && other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    Socket(const Socket &) = delete;
    int get() const { return fd_; }
    void close() { if (fd_ >= 0) ::close(fd_); fd_ = -1; }
    void send(const std::string & data) {
        size_t offset = 0;
        while (offset < data.size()) {
            const auto n = ::send(fd_, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
            ROUTING_CHECK(n > 0);
            offset += (size_t)n;
        }
    }
    std::string read() {
        std::string response;
        const auto deadline = Clock::now() + 5s;
        for (;;) {
            ROUTING_CHECK(Clock::now() < deadline);
            pollfd pfd{fd_, POLLIN, 0};
            const int ready = poll(&pfd, 1, 100);
            ROUTING_CHECK(ready >= 0);
            if (!ready) continue;
            char bytes[4096];
            const auto n = recv(fd_, bytes, sizeof(bytes), 0);
            ROUTING_CHECK(n >= 0);
            if (n == 0) return response;
            response.append(bytes, (size_t)n);
        }
    }
private:
    int fd_;
};

json response_body(const std::string & response, int status = 200) {
    ROUTING_CHECK(response.rfind("HTTP/1.1 " + std::to_string(status) + " ", 0) == 0);
    const auto boundary = response.find("\r\n\r\n");
    ROUTING_CHECK(boundary != std::string::npos);
    return json::parse(response.substr(boundary + 4));
}

class HeldEngine final : public SeqEngine {
public:
    int slot_count() const override { return 2; }
    int max_context() const override { return 64; }
    bool token_is_eos(int32_t token) const override { return token == 2; }
    StepPlanLimits step_plan_limits(int) const override { return {2, 64, 128}; }
    AdmitResult admit(uint64_t, const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_next) {
            fail_next = false;
            return {AdmitResult::Status::failed, -1, "injected admission failure"};
        }
        for (int i = 0; i < 2; ++i) {
            if (active[i]) continue;
            active[i] = true;
            prompts.push_back(prompt);
            temperatures.push_back(sampler.temp);
            ++admissions;
            cv.notify_all();
            return {AdmitResult::Status::admitted, i, {}};
        }
        return {AdmitResult::Status::busy, -1, {}};
    }
    StepResult step(const StepPlan & plan) override {
        std::unique_lock<std::mutex> lock(mu);
        if (block_step) {
            inside_block = true;
            cv.notify_all();
            cv.wait(lock, [&] { return !block_step; });
        }
        StepResult result;
        for (const auto & slice : plan.prefills) {
            result.prefills.push_back({slice.slot,
                release ? PrefillOutput::Status::completed : PrefillOutput::Status::advanced,
                release ? 0 : -1, {}});
        }
        for (const auto & input : plan.decode) result.decode.push_back({input.slot, 2, false, {}});
        // An unfinished fake prefill yields control to the real scheduler,
        // allowing admission and cancellation without timing-based completion.
        lock.unlock();
        std::this_thread::yield();
        return result;
    }
    void retire(int slot) override {
        std::lock_guard<std::mutex> lock(mu);
        active.at((size_t)slot) = false;
        ++retirements;
        cv.notify_all();
    }
    void wait_admissions(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return admissions >= count; }));
    }
    void wait_retirements(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return retirements >= count; }));
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mu);
        release = true;
        block_step = false;
        cv.notify_all();
    }
    std::mutex mu;
    std::condition_variable cv;
    std::array<bool, 2> active{};
    bool release = false, fail_next = false, block_step = false, inside_block = false;
    int admissions = 0, retirements = 0;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<float> temperatures;
};

struct RoutedBackend : ModelBackend {
    HeldEngine engine;
    SeqEngine * seq_engine() override { return &engine; }
    void print_ready_banner() const override {}
    bool park(ParkTarget) override { return true; }
    bool unpark(ParkTarget) override { return true; }
    bool is_target_parked() const override { return false; }
    GenerateResult generate_impl(const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int snapshot_cur_pos(int) const override { return 0; }
    GenerateResult restore_and_generate_impl(int, const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool handle_compress(const std::string &, const DaemonIO &) override { return false; }
    void free_drafter() override {}
    void shutdown() override {}
};

// Exercises the existing ModelBackend::generate path without a SeqEngine.
// The gate models work boundaries where real backends poll DaemonIO cancellation.
struct HeldSingleBackend final : RoutedBackend {
    SeqEngine * seq_engine() override { return nullptr; }
    GenerateResult generate_impl(const GenerateRequest & req, const DaemonIO & io) override {
        std::unique_lock<std::mutex> lock(mu);
        ++calls;
        prompts.push_back(req.prompt);
        temperatures.push_back(req.sampler.temp);
        cv.notify_all();
        const auto deadline = Clock::now() + 5s;
        while (!released && !io.is_cancelled() && Clock::now() < deadline) {
            cv.wait_for(lock, 10ms);
        }
        GenerateResult result;
        if (io.is_cancelled()) {
            ++cancellations;
            cv.notify_all();
        } else if (!released) {
            return result; // Bounded failure if the server never cancels or releases us.
        } else {
            result.tokens = {0, 2};
            for (int32_t token : result.tokens) io.emit(token);
        }
        result.succeed();
        return result;
    }
    void wait_calls(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return calls >= count; }));
    }
    void wait_cancellations(int count) {
        std::unique_lock<std::mutex> lock(mu);
        ROUTING_CHECK(cv.wait_for(lock, 5s, [&] { return cancellations >= count; }));
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mu);
        released = true;
        cv.notify_all();
    }
    std::mutex mu;
    std::condition_variable cv;
    bool released = false;
    int calls = 0, cancellations = 0;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<float> temperatures;
};

void load_tokenizer(Tokenizer & tokenizer, bool second) {
    gguf_context * ctx = gguf_init_empty();
    const char * first[] = {"q", "x", "<eos>", "y", "s"};
    const char * other[] = {"s", "y", "<eos>", "x", "q"};
    const uint32_t types[] = {1, 1, 3, 1, 1};
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", second ? other : first, 5);
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_UINT32, types, 5);
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 2);
    const auto path = std::filesystem::temp_directory_path() /
        ("luce-routing-" + std::to_string(getpid()) + (second ? "-b.gguf" : "-a.gguf"));
    gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    const bool loaded = tokenizer.load_from_gguf(path.c_str());
    std::filesystem::remove(path);
    ROUTING_CHECK(loaded);
}

class RunningModels {
public:
    RoutedBackend first, second;
    HeldSingleBackend single;
    Tokenizer first_tok, second_tok;
    std::unique_ptr<HttpServer> listener, peer;
    std::thread runner;
    int port = 0;
    std::atomic<int> result{-1};

    explicit RunningModels(bool single_peer = false) {
        load_tokenizer(first_tok, false);
        load_tokenizer(second_tok, true);
        // Obtain a loopback test port from the OS, then hand it to HttpServer.
        Socket reservation(socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ROUTING_CHECK(bind(reservation.get(), (sockaddr *)&address, sizeof(address)) == 0);
        socklen_t length = sizeof(address);
        ROUTING_CHECK(getsockname(reservation.get(), (sockaddr *)&address, &length) == 0);
        port = ntohs(address.sin_port);
        ServerConfig config;
        config.host = "127.0.0.1";
        config.port = port;
        config.model_name = "qwen";
        config.max_ctx = 64;
        config.default_max_tokens = 4;
        config.prefix_cache_cap = 0;
        config.ppp_enabled = false;
        config.admission_coalesce_ms = 0;
        config.chat_template_src = "{{ messages[0]['content'] }}";
        config.sampler_defaults.has_temperature = true;
        config.sampler_defaults.temperature = 0.2f;
        listener = std::make_unique<HttpServer>(first, first_tok, config);
        config.model_name = "ds4";
        config.chat_template_src = "y{{ messages[0]['content'] }}";
        config.sampler_defaults.temperature = 0.7f;
        ModelBackend & peer_backend = single_peer ? static_cast<ModelBackend &>(single) : second;
        peer = std::make_unique<HttpServer>(peer_backend, second_tok, config);
        reservation.close();
        runner = std::thread([this] { result = listener->run({listener.get(), peer.get()}); });
        try {
            // Wait on actual listener readiness, not a fixed startup sleep.
            const auto deadline = Clock::now() + 5s;
            for (;;) {
                ROUTING_CHECK(result.load() == -1);
                Socket probe(socket(AF_INET, SOCK_STREAM, 0));
                address.sin_port = htons(port);
                if (connect(probe.get(), (sockaddr *)&address, sizeof(address)) == 0) break;
                ROUTING_CHECK(Clock::now() < deadline);
                std::this_thread::yield();
            }
        } catch (...) { stop(); throw; }
    }
    ~RunningModels() { stop(); }
    void stop() {
        listener->request_stop();
        first.engine.finish();
        second.engine.finish();
        single.finish();
        if (runner.joinable()) runner.join();
    }
    Socket connect_client() {
        Socket client(socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        ROUTING_CHECK(connect(client.get(), (sockaddr *)&address, sizeof(address)) == 0);
        return client;
    }
    Socket post(json body, const std::string & path = "/v1/chat/completions") {
        auto client = connect_client();
        const auto text = body.dump();
        client.send("POST " + path + " HTTP/1.1\r\nHost: localhost\r\nContent-Length: " +
                    std::to_string(text.size()) + "\r\n\r\n" + text);
        return client;
    }
    json get(const std::string & path) {
        auto client = connect_client();
        client.send("GET " + path + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
        return response_body(client.read());
    }
    void wait_load(int first_count, int second_count) {
        const auto deadline = Clock::now() + 5s;
        for (;;) {
            const auto status = get("/status/json")["models"];
            if (status[0]["in_flight"] == first_count && status[1]["in_flight"] == second_count) return;
            ROUTING_CHECK(Clock::now() < deadline);
            std::this_thread::yield();
        }
    }
};

json chat(const char * model = "auto", bool stream = false) {
    return {{"model", model}, {"stream", stream}, {"max_tokens", 4},
            {"messages", {{{"role", "user"}, {"content", "x"}}}}};
}
} // namespace

TEST_CASE(ModelRoutingFixture, test_two_plus_two_preserves_batching_and_independent_progress) {
    RunningModels models;
    std::vector<Socket> clients;
    for (int i = 0; i < 4; ++i) {
        clients.push_back(models.post(chat()));
        (i % 2 ? models.second : models.first).engine.wait_admissions(i / 2 + 1);
    }
    models.wait_load(2, 2);
    response_body(models.post(chat()).read(), 503);
    response_body(models.post(chat("qwen")).read(), 503);
    response_body(models.post(chat("ds4")).read(), 503);
    models.first.engine.finish();
    for (int i : {0, 2}) {
        const auto body = response_body(clients[i].read());
        ROUTING_CHECK(body["model"] == "qwen");
        ROUTING_CHECK(body["choices"][0]["message"]["content"] == "q");
    }
    models.wait_load(0, 2); // DS4 cannot stall Qwen's worker or response path.
    models.second.engine.finish();
    for (int i : {1, 3}) {
        const auto body = response_body(clients[i].read());
        ROUTING_CHECK(body["model"] == "ds4");
        ROUTING_CHECK(body["choices"][0]["message"]["content"] == "s");
    }
    models.wait_load(0, 0);
    ROUTING_CHECK(models.first.engine.prompts[0] == std::vector<int32_t>({1}));
    ROUTING_CHECK(models.second.engine.prompts[0] == std::vector<int32_t>({1, 3}));
    ROUTING_CHECK(models.first.engine.temperatures[0] == 0.2f);
    ROUTING_CHECK(models.second.engine.temperatures[0] == 0.7f);
    const auto list = models.get("/v1/models")["data"];
    ROUTING_CHECK(list.size() == 3);
    ROUTING_CHECK(list[0]["id"] == "qwen" && list[1]["id"] == "ds4" && list[2]["id"] == "auto");
    const auto codex = models.get("/v1/models?client_version=test")["models"];
    ROUTING_CHECK(codex.size() == 3 && codex[1]["slug"] == "ds4" && codex[2]["slug"] == "auto");
    const auto props = models.get("/props");
    ROUTING_CHECK(props["server"]["props_schema"] == 2);
    ROUTING_CHECK(props["models"][1]["props"]["model_alias"] == "ds4");
}

TEST_CASE(ModelRoutingFixture, test_rejected_requests_and_engine_failure_release_capacity) {
    RunningModels models;
    response_body(models.post(chat("missing")).read(), 404);
    auto invalid = chat(); invalid.erase("messages");
    response_body(models.post(invalid).read(), 400);
    invalid = chat(); invalid["model"] = 42;
    response_body(models.post(invalid).read(), 400);
    invalid = chat(); invalid["messages"][0]["content"] = std::string(100, 'x');
    response_body(models.post(invalid).read(), 400);
    response_body(models.post(chat(), "/v1/messages/count_tokens").read(), 400);
    ROUTING_CHECK(response_body(models.post(chat("ds4"), "/v1/messages/count_tokens").read())["input_tokens"] == 2);
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.fail_next = true;
    }
    response_body(models.post(chat("qwen")).read(), 500);
    models.wait_load(0, 0);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(models.post(chat("qwen")).read())["model"] == "qwen");
}

TEST_CASE(ModelRoutingFixture, test_disconnect_holds_capacity_until_engine_retirement) {
    RunningModels models;
    auto a = models.post(chat("qwen"));
    auto b = models.post(chat("qwen"));
    models.first.engine.wait_admissions(2);
    {
        std::unique_lock<std::mutex> lock(models.first.engine.mu);
        models.first.engine.block_step = true;
        ROUTING_CHECK(models.first.engine.cv.wait_for(lock, 5s, [&] { return models.first.engine.inside_block; }));
    }
    // A TCP reset is an unambiguous disconnect. FIN alone also represents a
    // legitimate HTTP client that half-closes its write side and keeps reading.
    linger reset{1, 0};
    ROUTING_CHECK(setsockopt(a.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0);
    a.close(); // A running step still owns this request's state.
    response_body(models.post(chat("qwen")).read(), 503);
    {
        std::lock_guard<std::mutex> lock(models.first.engine.mu);
        models.first.engine.block_step = false;
        models.first.engine.cv.notify_all();
    }
    models.first.engine.wait_retirements(1);
    models.wait_load(1, 0);
    auto replacement = models.post(chat("qwen"));
    models.first.engine.wait_admissions(3);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(b.read())["model"] == "qwen");
    ROUTING_CHECK(response_body(replacement.read())["model"] == "qwen");
    models.wait_load(0, 0);
}

TEST_CASE(ModelRoutingFixture, test_response_protocols_use_selected_model_and_terminal_event) {
    RunningModels models;
    models.second.engine.finish();
    const auto stream = models.post(chat("ds4", true)).read();
    ROUTING_CHECK(stream.rfind("HTTP/1.1 200 ", 0) == 0);
    ROUTING_CHECK(stream.find("\"model\":\"ds4\"") != std::string::npos);
    ROUTING_CHECK(stream.find("\"content\":\"s\"") != std::string::npos);
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
    const auto message = response_body(models.post(chat("ds4"), "/v1/messages").read());
    ROUTING_CHECK(message["model"] == "ds4");
    ROUTING_CHECK(message["content"][0]["text"] == "s");
    json request = {{"model", "ds4"}, {"input", {{{"role", "user"}, {"content", "x"}}}},
                    {"max_output_tokens", 4}};
    const auto response = response_body(models.post(request, "/v1/responses").read());
    ROUTING_CHECK(response["model"] == "ds4");
    ROUTING_CHECK(response["output"][0]["content"][0]["text"] == "s");
    models.wait_load(0, 0);
}

TEST_CASE(ModelRoutingFixture, test_shutdown_drains_both_models_and_incomplete_upload) {
    RunningModels models;
    auto a = models.post(chat("qwen"));
    auto b = models.post(chat("ds4"));
    auto incomplete = models.connect_client();
    incomplete.send("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n{");
    models.first.engine.wait_admissions(1);
    models.second.engine.wait_admissions(1);
    models.listener->request_stop();
    models.runner.join();
    ROUTING_CHECK(models.result == 0);
    ROUTING_CHECK(models.first.engine.retirements == 1);
    ROUTING_CHECK(models.second.engine.retirements == 1);
}

TEST_CASE(ModelRoutingFixture, test_hybrid_admission_uses_single_capacity_and_independent_workers) {
    RunningModels models(true);
    const auto props = models.get("/props")["models"];
    ROUTING_CHECK(props[0]["capacity"] == 2 && props[0]["execution_mode"] == "batched");
    ROUTING_CHECK(props[1]["capacity"] == 1 && props[1]["execution_mode"] == "single-request");
    auto first = models.post(chat());
    models.first.engine.wait_admissions(1);
    auto second = models.post(chat());
    models.single.wait_calls(1);
    auto third = models.post(chat());
    models.first.engine.wait_admissions(2);
    models.wait_load(2, 1);
    response_body(models.post(chat()).read(), 503);
    response_body(models.post(chat("ds4")).read(), 503);
    models.first.engine.finish();
    ROUTING_CHECK(response_body(first.read())["model"] == "qwen");
    ROUTING_CHECK(response_body(third.read())["model"] == "qwen");
    models.wait_load(0, 1);
    models.single.finish();
    const auto body = response_body(second.read());
    ROUTING_CHECK(body["model"] == "ds4");
    ROUTING_CHECK(body["choices"][0]["message"]["content"] == "s");
    models.wait_load(0, 0);
    ROUTING_CHECK(models.single.prompts[0] == std::vector<int32_t>({1, 3}));
    ROUTING_CHECK(models.single.temperatures[0] == 0.7f);
}

TEST_CASE(ModelRoutingFixture, test_hybrid_disconnect_cancels_single_worker_and_reuses_capacity) {
    RunningModels models(true);
    auto client = models.post(chat("ds4", true));
    models.single.wait_calls(1);
    linger reset{1, 0};
    ROUTING_CHECK(setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)) == 0);
    client.close();
    models.single.wait_cancellations(1);
    models.wait_load(0, 0);
    models.single.finish();
    const auto stream = models.post(chat("ds4", true)).read();
    ROUTING_CHECK(stream.find("\"model\":\"ds4\"") != std::string::npos);
    ROUTING_CHECK(stream.find("\"content\":\"s\"") != std::string::npos);
    ROUTING_CHECK(stream.find("data: [DONE]") != std::string::npos);
}

TEST_CASE(ModelRoutingFixture, test_hybrid_shutdown_cancels_single_worker_before_destroying_contexts) {
    RunningModels models(true);
    auto first = models.post(chat("qwen"));
    auto second = models.post(chat("ds4"));
    models.first.engine.wait_admissions(1);
    models.single.wait_calls(1);
    models.listener->request_stop();
    models.single.wait_cancellations(1);
    models.runner.join();
    ROUTING_CHECK(models.result == 0);
    ROUTING_CHECK(models.first.engine.retirements == 1);
}
#undef ROUTING_CHECK
#endif
