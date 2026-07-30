// Native socket handle type shared by the HTTP server interface.

#pragma once

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
// Winsock must precede any transitive inclusion of windows.h.
#include <winsock2.h>
#endif

namespace dflash::common {

#if defined(_WIN32)
using SocketHandle = SOCKET;
inline constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
inline constexpr SocketHandle kInvalidSocket = -1;
#endif

inline constexpr bool socket_is_valid(SocketHandle socket) noexcept {
    return socket != kInvalidSocket;
}

}  // namespace dflash::common
