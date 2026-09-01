#include "kernel_qualification.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace kq = lucebox::kernel_qualification;

namespace {

bool require(bool condition, const char * message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    bool ok = true;

    const kq::Metric within = kq::compare_f32(
        "output", {1.0f, 2.0f}, {1.0f, 2.00001f}, 2.0e-5);
    ok = require(within.passed && within.finite,
                 "comparison inside tolerance must pass") && ok;

    const kq::Metric outside = kq::compare_f32(
        "state", {1.0f, 2.1f}, {1.0f, 2.0f}, 1.0e-3);
    ok = require(!outside.passed && outside.reason == "tolerance exceeded",
                 "comparison outside tolerance must fail") && ok;

    const kq::Metric nonfinite = kq::compare_f32(
        "nonfinite", {std::numeric_limits<float>::quiet_NaN()}, {0.0f}, 1.0);
    ok = require(!nonfinite.passed && !nonfinite.finite,
                 "non-finite values must fail") && ok;

    const kq::CaseResult passed = kq::evaluate(
        "gdn", {within}, {"gdn.grouped_cols", "gdn.grouped_cols", true});
    const kq::CaseResult wrong_route = kq::evaluate(
        "gdn", {within}, {"gdn.grouped_cols", "gdn.scalar", false});
    const kq::CaseResult skipped = kq::unsupported(
        "gdn", "GPU backend unavailable");
    ok = require(passed.status == kq::Status::pass,
                 "matching numerics and route must pass") && ok;
    ok = require(wrong_route.status == kq::Status::fail,
                 "a route mismatch must fail") && ok;
    ok = require(skipped.status == kq::Status::unsupported,
                 "unsupported must not be reported as pass") && ok;
    ok = require(kq::exit_code({skipped}) == 77,
                 "unsupported-only reports must use the CTest skip code") && ok;

    std::ostringstream report;
    kq::write_json(report, "self-test", {passed});
    const std::string json = report.str();
    ok = require(
        json.find("\"schema\":\"lucebox.kernel_qualification.v1\"") !=
            std::string::npos &&
        json.find("\"status\":\"pass\"") != std::string::npos,
        "report must include the stable schema and aggregate status") && ok;

    if (ok) std::cout << json;
    return ok ? 0 : 1;
}
