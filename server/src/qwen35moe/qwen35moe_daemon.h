// Qwen35MoE daemon entry point.

#pragma once

#include "qwen35_daemon.h"

namespace luce::common {

using Qwen35MoeDaemonArgs = Qwen35DaemonArgs;

int run_qwen35moe_daemon(const Qwen35MoeDaemonArgs & args);

}  // namespace luce::common
