// Cross-platform environment variable helpers.

#pragma once

#include <cstdlib>

namespace dflash::common {

// Match POSIX setenv() semantics on every platform. In particular,
// overwrite=false must preserve an existing value; _putenv_s() does not
// provide that behavior by itself.
inline int set_environment_variable(
        const char * name, const char * value, bool overwrite) {
#if defined(_WIN32)
    if (!overwrite && std::getenv(name) != nullptr) {
        return 0;
    }
    return ::_putenv_s(name, value);
#else
    return ::setenv(name, value, overwrite ? 1 : 0);
#endif
}

inline int unset_environment_variable(const char * name) {
#if defined(_WIN32)
    return ::_putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}

}  // namespace dflash::common
