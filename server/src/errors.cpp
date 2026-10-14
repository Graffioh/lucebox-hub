// Last-error string used by loaders and graph builders, one per thread: a
// caller reads the error its own failing call set, and the pointer stays
// valid until that thread sets another. A concurrent server scores PFlash
// prompts on one thread while another serves, and neither may read, or
// free, the other's. Consumed by tests and the test_dflash driver via
// luce_last_error().

#include "luce.h"
#include "internal.h"

#include <string>

namespace luce::common {

namespace {
thread_local std::string g_last_error;
}

void set_last_error(std::string msg) {
    g_last_error = std::move(msg);
}

} // namespace luce::common

extern "C" const char * luce_last_error(void) {
    return luce::common::g_last_error.c_str();
}
