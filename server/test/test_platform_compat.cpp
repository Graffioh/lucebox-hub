#include "common/platform_env.h"
#include "server/socket_handle.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace dflash::common;

namespace {

int fail(const char * message) {
    std::fprintf(stderr, "platform compatibility test failed: %s\n", message);
    return 1;
}

}  // namespace

int main() {
    constexpr const char * kEnvName = "DFLASH_PLATFORM_COMPAT_TEST";

    if (unset_environment_variable(kEnvName) != 0) {
        return fail("could not clear test environment variable");
    }
    if (set_environment_variable(kEnvName, "original", true) != 0) {
        return fail("could not set environment variable");
    }
    if (set_environment_variable(kEnvName, "replacement", false) != 0) {
        return fail("non-overwriting environment update failed");
    }
    const char * value = std::getenv(kEnvName);
    if (value == nullptr || std::strcmp(value, "original") != 0) {
        return fail("overwrite=false replaced the existing value");
    }
    if (set_environment_variable(kEnvName, "replacement", true) != 0) {
        return fail("overwriting environment update failed");
    }
    value = std::getenv(kEnvName);
    if (value == nullptr || std::strcmp(value, "replacement") != 0) {
        return fail("overwrite=true did not replace the existing value");
    }
    if (unset_environment_variable(kEnvName) != 0 ||
        std::getenv(kEnvName) != nullptr) {
        return fail("could not remove environment variable");
    }

    if (socket_is_valid(kInvalidSocket)) {
        return fail("invalid socket sentinel reported as valid");
    }

#if defined(_WIN32)
    static_assert(sizeof(SocketHandle) == sizeof(void *),
                  "Win64 socket handles must remain pointer-width");
    WSADATA wsa_data{};
    const int wsa_error = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (wsa_error != 0) {
        return fail("WSAStartup failed");
    }
#endif

    const SocketHandle socket_handle = ::socket(AF_INET, SOCK_STREAM, 0);
    if (!socket_is_valid(socket_handle)) {
#if defined(_WIN32)
        WSACleanup();
#endif
        return fail("could not create a TCP socket");
    }

#if defined(_WIN32)
    closesocket(socket_handle);
    WSACleanup();
#else
    ::close(socket_handle);
#endif

    return 0;
}
