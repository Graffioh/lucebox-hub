#include "pflash/pflash_query_candidates.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace luce::pflash {

namespace {

constexpr const char * kCandidatesEnv = "PFLASH_QUERY_CANDIDATES";
constexpr const char * kPinEnv = "PFLASH_QUERY_PIN";
constexpr const char * kUnitScoreEnv = "PFLASH_QUERY_UNIT_SCORE";
constexpr const char * kAggregateEnv = "PFLASH_QUERY_AGG";
constexpr const char * kTemperatureEnv = "PFLASH_QUERY_TEMP";
constexpr const char * kNullMarginEnv = "PFLASH_QUERY_NULL_MARGIN";
constexpr const char * kExtentEnv = "PFLASH_QUERY_EXTENT";
constexpr const char * kPinMarginEnv = "PFLASH_QUERY_PIN_MARGIN";
constexpr const char * kTwEnv = "PFLASH_QUERY_TW";
constexpr const char * kEndEnv = "PFLASH_QUERY_END";

bool parse_int(const char * raw, int lo, int hi, int & out) {
    if (!raw || !*raw) return false;
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' || v < lo || v > hi) return false;
    out = (int) v;
    return true;
}

bool parse_float(const char * raw, float & out) {
    if (!raw || !*raw) return false;
    errno = 0;
    char * end = nullptr;
    const float v = std::strtof(raw, &end);
    if (errno == ERANGE || end == raw || *end != '\0' || !std::isfinite(v)) return false;
    out = v;
    return true;
}

} // namespace

const char * pflash_unit_score_name(PFlashUnitScore mode) noexcept {
    return mode == PFlashUnitScore::Max ? "max" : "mean";
}

const char * pflash_query_aggregate_name(PFlashQueryAggregate mode) noexcept {
    return mode == PFlashQueryAggregate::Softmax ? "softmax" : "wmax";
}

bool pflash_unit_score_from_name(const std::string & name, PFlashUnitScore & out) {
    if (name == "max") { out = PFlashUnitScore::Max; return true; }
    if (name == "mean") { out = PFlashUnitScore::Mean; return true; }
    return false;
}

bool pflash_query_aggregate_from_name(const std::string & name, PFlashQueryAggregate & out) {
    if (name == "softmax") { out = PFlashQueryAggregate::Softmax; return true; }
    if (name == "wmax") { out = PFlashQueryAggregate::WeightedMax; return true; }
    return false;
}

bool pflash_query_config_from_env(PFlashQueryConfig & out, std::string & error) {
    error.clear();
    PFlashQueryConfig config;
    const char * k_raw = std::getenv(kCandidatesEnv);
    const char * pin_raw = std::getenv(kPinEnv);
    if (k_raw && !parse_int(k_raw, 0, kQueryCandidatesMax, config.candidates)) {
        error = std::string(kCandidatesEnv) + " must be an integer in [0, " + std::to_string(kQueryCandidatesMax) + "]";
        return false;
    }
    if (pin_raw && !parse_int(pin_raw, 0, kQueryPinMax, config.pin_tokens)) {
        error = std::string(kPinEnv) + " must be an integer in [0, " + std::to_string(kQueryPinMax) + "]";
        return false;
    }
    const char * unit_raw = std::getenv(kUnitScoreEnv);
    const char * agg_raw = std::getenv(kAggregateEnv);
    const char * temp_raw = std::getenv(kTemperatureEnv);
    const char * margin_raw = std::getenv(kNullMarginEnv);
    if (!config.any()) {
        // The knobs alone change nothing: refuse them, so a typo in the switch cannot pass silently.
        for (const char * name : {kUnitScoreEnv, kAggregateEnv, kTemperatureEnv, kNullMarginEnv, kExtentEnv,
                                  kPinMarginEnv, kTwEnv, kEndEnv}) {
            if (std::getenv(name)) {
                error = std::string(name) + " is set without " + kCandidatesEnv + " or " + kPinEnv;
                return false;
            }
        }
        out = config;
        return true;
    }
    if (unit_raw && !pflash_unit_score_from_name(unit_raw, config.unit_score)) {
        error = std::string(kUnitScoreEnv) + " must be max or mean";
        return false;
    }
    if (agg_raw && !pflash_query_aggregate_from_name(agg_raw, config.aggregate)) {
        error = std::string(kAggregateEnv) + " must be softmax or wmax";
        return false;
    }
    if (temp_raw && (!parse_float(temp_raw, config.temperature) || !(config.temperature > 0.0f))) {
        error = std::string(kTemperatureEnv) + " must be a finite number > 0";
        return false;
    }
    if (margin_raw && (!parse_float(margin_raw, config.null_margin) || config.null_margin < 0.0f)) {
        error = std::string(kNullMarginEnv) + " must be a finite number >= 0";
        return false;
    }
    if (const char * raw = std::getenv(kExtentEnv)) {
        if (std::strcmp(raw, "gap") == 0) {
            config.extent_gap = true;
        } else if ((config.extent_gap = false), !parse_float(raw, config.extent_margin) || config.extent_margin < 0.0f) {
            error = std::string(kExtentEnv) + " must be a finite number >= 0 or gap";
            return false;
        }
    }
    if (const char * raw = std::getenv(kPinMarginEnv)) {
        if (std::strcmp(raw, "inf") == 0) {
            config.pin_gap = false;
            config.pin_margin = std::numeric_limits<float>::infinity();
        } else if (std::strcmp(raw, "gap") == 0) {
            config.pin_gap = true;
        } else if ((config.pin_gap = false), !parse_float(raw, config.pin_margin) || config.pin_margin < 0.0f) {
            error = std::string(kPinMarginEnv) + " must be a finite number >= 0, inf or gap";
            return false;
        }
    }
    if (const char * raw = std::getenv(kTwEnv)) {
        if (std::strcmp(raw, "score") == 0) {
            config.tw_score = true;
        } else if (!parse_float(raw, config.tw_share) || !(config.tw_share > 0.0f) || !(config.tw_share < 1.0f)) {
            error = std::string(kTwEnv) + " must be a share in (0, 1) or score";
            return false;
        }
    }
    if (const char * raw = std::getenv(kEndEnv)) {
        if (!parse_float(raw, config.end_share) || !(config.end_share > 0.0f) || !(config.end_share < 1.0f)) {
            error = std::string(kEndEnv) + " must be a share in (0, 1)";
            return false;
        }
    }
    if (!(config.tw_share + config.end_share < 1.0f)) {
        error = std::string(kTwEnv) + " + " + kEndEnv + " shares must be < 1";
        return false;
    }
    if ((std::getenv(kTwEnv) || std::getenv(kEndEnv)) && config.candidates <= 0) {
        error = std::string(kTwEnv) + " / " + kEndEnv + " need " + kCandidatesEnv;
        return false;
    }
    out = config;
    return true;
}

PFlashQueryUnits pflash_query_units(const std::vector<PFlashTokenSpan> & segments,
                                    const std::vector<PFlashRequestMessage> & messages,
                                    const std::vector<float> & s1) {
    PFlashQueryUnits out;
    out.latest = pflash_request_latest_user(messages);
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto & s = segments[i];
        if (s.end <= s.begin || (i > 0 && s.begin != segments[i - 1].end) || (i == 0 && s.begin != 0)) {
            out.error = "segments_invalid";
            return out;
        }
    }
    const auto build = [&](bool system) {
        std::vector<PFlashQueryUnit> units;
        for (size_t m = 0; m < messages.size(); ++m) {
            const auto & msg = messages[m];
            if ((msg.role == PFlashRequestRole::System) != system) continue;
            const PFlashTokenSpan span = msg.span;
            if (span.end <= span.begin) continue;
            if (span.begin < 0 || span.end > (int) s1.size()) {
                out.error = "logits_missing";
                return units;
            }
            // First segment ending after span.begin.
            auto it = std::upper_bound(segments.begin(), segments.end(), span.begin,
                [](int pos, const PFlashTokenSpan & seg) { return pos < seg.end; });
            for (; it != segments.end() && it->begin < span.end; ++it) {
                const int b = std::max(it->begin, span.begin);
                const int e = std::min(it->end, span.end);
                if (e <= b) continue;
                double sum = 0.0;
                int n = 0;
                float mx = -std::numeric_limits<float>::infinity();
                for (int t = b; t < e; ++t) {
                    const float v = s1[(size_t) t];
                    if (!std::isfinite(v)) continue;
                    sum += v;
                    ++n;
                    mx = std::max(mx, v);
                }
                if (n == 0) continue;
                PFlashQueryUnit u;
                u.span = {b, e};
                u.message = (int) m;
                u.latest = (int) m == out.latest;
                u.score_max = mx;
                u.score_mean = (float) (sum / n);
                units.push_back(u);
            }
        }
        std::sort(units.begin(), units.end(), [](const PFlashQueryUnit & a, const PFlashQueryUnit & b) {
            return a.span.begin < b.span.begin;
        });
        return units;
    };
    out.units = build(false);
    if (!out.error.empty()) {
        out.units.clear();
        return out;
    }
    if (out.units.empty()) {
        out.units = build(true);
        out.system_used = !out.units.empty();
        if (!out.error.empty()) out.units.clear();
    }
    return out;
}

std::vector<int> pflash_query_rank(const PFlashQueryUnits & units, const std::vector<int> & which,
                                   PFlashUnitScore mode) {
    std::vector<int> order = which;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto & ua = units.units[(size_t) a];
        const auto & ub = units.units[(size_t) b];
        const float sa = ua.score(mode);
        const float sb = ub.score(mode);
        if (sa != sb) return sa > sb;
        return ua.span.begin > ub.span.begin;
    });
    return order;
}

std::vector<int> pflash_query_unit_tokens(PFlashTokenSpan span, const std::vector<float> & s1, int cap) {
    std::vector<int> tokens;
    for (int t = span.begin; t < span.end; ++t) tokens.push_back(t);
    if (cap <= 0 || (int) tokens.size() <= cap) return tokens;
    std::vector<int> ranked = pflash_request_by_score(tokens, s1);
    ranked.resize((size_t) cap);
    std::sort(ranked.begin(), ranked.end());
    return ranked;
}

std::vector<double> pflash_query_softmax(const std::vector<float> & scores, float temperature) {
    std::vector<double> w(scores.size(), 0.0);
    if (scores.empty()) return w;
    double mx = -std::numeric_limits<double>::infinity();
    for (const float s : scores) mx = std::max(mx, (double) s);
    double total = 0.0;
    for (size_t k = 0; k < scores.size(); ++k) {
        w[k] = std::exp(((double) scores[k] - mx) / (double) temperature);
        total += w[k];
    }
    for (auto & v : w) v /= total;
    return w;
}

std::vector<int> pflash_query_extent_units(const PFlashQueryUnits & units, int message, PFlashUnitScore mode,
                                           float margin, bool gap) {
    std::vector<int> mine;
    for (size_t i = 0; i < units.units.size(); ++i) {
        if (units.units[i].message == message) mine.push_back((int) i);
    }
    if (mine.empty()) return mine;
    const auto ranked = pflash_query_rank(units, mine, mode);
    if (gap) {
        if ((int) ranked.size() < kQueryNullMinUnits) return ranked;
        size_t cut = 1;
        float best = -1.0f;
        for (size_t i = 0; i + 1 < ranked.size(); ++i) {
            const float drop = units.units[(size_t) ranked[i]].score(mode) - units.units[(size_t) ranked[i + 1]].score(mode);
            if (drop > best) {          // the first largest drop (ties keep the shorter extent)
                best = drop;
                cut = i + 1;
            }
        }
        return std::vector<int>(ranked.begin(), ranked.begin() + (long) cut);
    }
    const float top = units.units[(size_t) ranked.front()].score(mode);
    std::vector<int> out;
    for (const int u : ranked) {
        if (!(units.units[(size_t) u].score(mode) >= top - margin)) break;   // inf margin: every unit
        out.push_back(u);
    }
    return out;
}

PFlashQueryExtent pflash_query_extent(const PFlashQueryUnits & units, int message, const std::vector<float> & s1,
                                      PFlashUnitScore mode, float margin, int cap, bool gap) {
    PFlashQueryExtent out;
    const auto eligible = pflash_query_extent_units(units, message, mode, margin, gap);
    if (eligible.empty()) return out;
    out.top = units.units[(size_t) eligible.front()].score(mode);
    int total = 0;
    for (const int u : eligible) {
        const auto & unit = units.units[(size_t) u];
        const int len = unit.span.end - unit.span.begin;
        if (cap > 0 && total + len > cap) {
            if (out.units.empty()) {
                // The top unit alone exceeds the cap: its cap best tokens (a query, not a pin).
                out.units.push_back(u);
                out.tokens = pflash_query_unit_tokens(unit.span, s1, cap);
                out.cut = true;
                return out;
            }
            ++out.skipped;
            continue;
        }
        total += len;
        out.units.push_back(u);
        for (int t = unit.span.begin; t < unit.span.end; ++t) out.tokens.push_back(t);
    }
    std::sort(out.tokens.begin(), out.tokens.end());
    return out;
}

PFlashQueryCandidateSet pflash_query_candidate_set(const PFlashQueryUnits & units,
                                                   const std::vector<float> & s1,
                                                   const PFlashQueryConfig & config) {
    PFlashQueryCandidateSet out;
    out.latest = units.latest;
    out.latest_z = std::numeric_limits<float>::quiet_NaN();
    const PFlashUnitScore mode = config.unit_score;
    const int k = std::max(1, config.candidates);
    std::vector<int> latest_units;
    std::vector<int> messages;          // messages with a unit, ascending
    for (size_t i = 0; i < units.units.size(); ++i) {
        const auto & u = units.units[i];
        if (u.latest) latest_units.push_back((int) i);
        if (std::find(messages.begin(), messages.end(), u.message) == messages.end()) messages.push_back(u.message);
    }
    std::sort(messages.begin(), messages.end());
    out.latest_units = (int) latest_units.size();
    bool latest_null = false;
    if (!latest_units.empty()) {
        const auto ranked = pflash_query_rank(units, latest_units, mode);
        out.latest_top = units.units[(size_t) ranked.front()].score(mode);
        if ((int) latest_units.size() >= kQueryNullMinUnits) {
            std::vector<float> sorted;
            for (const int u : latest_units) sorted.push_back(units.units[(size_t) u].score(mode));
            std::sort(sorted.begin(), sorted.end());
            const size_t n = sorted.size();
            const float median = n % 2 ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
            std::vector<float> dev;
            for (const float v : sorted) dev.push_back(std::fabs(v - median));
            std::sort(dev.begin(), dev.end());
            const float mad = n % 2 ? dev[n / 2] : 0.5f * (dev[n / 2 - 1] + dev[n / 2]);
            const float spread = std::max(1.4826f * mad, kQuerySpreadFloor);
            out.latest_median = median;
            out.latest_spread = spread;
            out.latest_z = (out.latest_top - median) / spread;
            latest_null = config.null_margin > 0.0f && out.latest_z < config.null_margin;
        }
        out.latest_null = latest_null;
    }
    // Messages by their top unit's score (descending; ties to the later message), the latest first unless NULL.
    struct Top { int message; float score; int begin; };
    std::vector<Top> tops;
    for (const int m : messages) {
        if (m == units.latest) continue;
        float best = -std::numeric_limits<float>::infinity();
        int begin = -1;
        for (const auto & u : units.units) {
            if (u.message != m) continue;
            const float v = u.score(mode);
            if (v > best || (v == best && u.span.begin > begin)) { best = v; begin = u.span.begin; }
        }
        tops.push_back({m, best, begin});
    }
    std::stable_sort(tops.begin(), tops.end(), [](const Top & a, const Top & b) {
        if (a.score != b.score) return a.score > b.score;
        return a.message > b.message;
    });
    std::vector<int> order;
    if (!latest_units.empty() && !latest_null) order.push_back(units.latest);
    for (const auto & t : tops) order.push_back(t.message);
    if (order.empty() && !latest_units.empty()) order.push_back(units.latest);   // NULL and nothing else: never empty
    if ((int) order.size() > k) order.resize((size_t) k);
    std::vector<float> scores;
    for (const int m : order) {
        const auto extent = pflash_query_extent(units, m, s1, mode, config.extent_margin, kQueryExtentQueryCap,
                                                config.extent_gap);
        if (extent.tokens.empty()) continue;
        PFlashQueryCandidate c;
        c.message = m;
        c.units = extent.units;
        c.tokens = extent.tokens;
        c.score = extent.top;
        scores.push_back(c.score);
        out.candidates.push_back(std::move(c));
    }
    const auto w = pflash_query_softmax(scores, config.temperature);
    for (size_t i = 0; i < out.candidates.size(); ++i) out.candidates[i].weight = w[i];
    return out;
}

PFlashQueryBlend pflash_query_blend(const PFlashQueryCandidateSet & set, const PFlashQueryConfig & config,
                                    float tw_score_value) {
    PFlashQueryBlend out;
    std::vector<float> scores;
    for (const auto & c : set.candidates) scores.push_back(c.score);
    const bool tw_in_softmax = config.tw_score && std::isfinite(tw_score_value);
    if (tw_in_softmax) scores.push_back(tw_score_value);
    auto w = pflash_query_softmax(scores, config.temperature);
    double rest = 1.0 - (double) config.end_share;
    if (!config.tw_score) rest -= (double) config.tw_share;
    if (tw_in_softmax) {
        out.tw = w.back() * rest;
        w.pop_back();
    } else if (!config.tw_score) {
        out.tw = (double) config.tw_share;
    }
    // TW "score" without a finite score (no content token in its windows): TW stays out of the blend.
    for (const double v : w) out.probe.push_back(v * rest);
    out.end = (double) config.end_share;
    if (out.probe.empty()) {
        // No probe candidate: the remaining weight goes to TW / END (normalised); the caller fails closed before.
        const double total = out.tw + out.end;
        if (total > 0.0) { out.tw /= total; out.end /= total; }
    }
    return out;
}

bool pflash_query_aggregate(const std::vector<std::vector<float>> & masses,
                            const std::vector<double> & weights,
                            PFlashQueryAggregate mode,
                            std::vector<float> & out,
                            std::string & error) {
    error.clear();
    if (masses.empty() || masses.size() != weights.size()) {
        error = "query aggregate: candidate masses and weights differ in count";
        return false;
    }
    const size_t n = masses.front().size();
    for (const auto & m : masses) {
        if (m.size() != n) {
            error = "query aggregate: candidate masses differ in length";
            return false;
        }
    }
    out.assign(n, 0.0f);
    if (mode == PFlashQueryAggregate::Softmax) {
        std::vector<double> acc(n, 0.0);
        for (size_t k = 0; k < masses.size(); ++k) {
            for (size_t t = 0; t < n; ++t) acc[t] += weights[k] * (double) masses[k][t];
        }
        for (size_t t = 0; t < n; ++t) out[t] = (float) acc[t];
        return true;
    }
    std::vector<double> acc(n, 0.0);
    double total = 0.0;
    for (size_t t = 0; t < n; ++t) {
        double best = 0.0;
        for (size_t k = 0; k < masses.size(); ++k) best = std::max(best, weights[k] * (double) masses[k][t]);
        acc[t] = best;
        total += best;
    }
    for (size_t t = 0; t < n; ++t) out[t] = total > 0.0 ? (float) (acc[t] / total) : 0.0f;
    return true;
}

PFlashQueryPin pflash_query_pin(const PFlashQueryUnits & units, int cap, PFlashUnitScore mode, float margin,
                                bool gap) {
    PFlashQueryPin pin;
    if (cap <= 0 || units.latest < 0) return pin;
    std::vector<PFlashTokenSpan> spans;
    for (const int u : pflash_query_extent_units(units, units.latest, mode, margin, gap)) {
        const auto & unit = units.units[(size_t) u];
        const int len = unit.span.end - unit.span.begin;
        if (pin.tokens + len > cap) {
            ++pin.skipped;
            continue;
        }
        pin.tokens += len;
        ++pin.units;
        spans.push_back(unit.span);
    }
    std::sort(spans.begin(), spans.end(), [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
        return a.begin < b.begin;
    });
    for (const auto & sp : spans) {
        if (!pin.spans.empty() && sp.begin <= pin.spans.back().end) {
            pin.spans.back().end = std::max(pin.spans.back().end, sp.end);
        } else {
            pin.spans.push_back(sp);
        }
    }
    return pin;
}

std::string pflash_ids_fingerprint(const std::vector<int32_t> & ids) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const int32_t token : ids) {
        const uint32_t value = static_cast<uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(value >> shift);
            hash *= UINT64_C(1099511628211);
        }
    }
    char encoded[17];
    std::snprintf(encoded, sizeof(encoded), "%016llx", static_cast<unsigned long long>(hash));
    return encoded;
}

const char * pflash_grid_mode_name(PFlashGridMode mode) noexcept {
    switch (mode) {
        case PFlashGridMode::S1: return "s1";
        case PFlashGridMode::P1: return "p1";
        case PFlashGridMode::Candidates: return "cand";
        case PFlashGridMode::Oracle: return "oracle";
    }
    return "?";
}

bool pflash_query_grid_parse(const std::string & text, std::vector<PFlashGridConfig> & out, std::string & error) {
    error.clear();
    out.clear();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const std::exception & e) {
        error = std::string("query grid: invalid JSON: ") + e.what();
        return false;
    }
    if (!j.is_object() || !j.contains("configs") || !j["configs"].is_array() || j["configs"].empty()) {
        error = "query grid: need {\"configs\": [...]} with at least one config";
        return false;
    }
    for (const auto & c : j["configs"]) {
        if (!c.is_object()) { error = "query grid: a config is not an object"; return false; }
        PFlashGridConfig g;
        // Grid files are explicit: every blend knob starts OFF (not at the runtime defaults).
        g.query.extent_gap = false;
        g.query.pin_gap = false;
        g.query.tw_share = 0.0f;
        g.query.end_share = 0.0f;
        try {
            g.name = c.at("name").get<std::string>();
            const std::string mode = c.value("mode", std::string("cand"));
            if (mode == "s1") g.mode = PFlashGridMode::S1;
            else if (mode == "p1") g.mode = PFlashGridMode::P1;
            else if (mode == "cand") g.mode = PFlashGridMode::Candidates;
            else if (mode == "oracle") g.mode = PFlashGridMode::Oracle;
            else { error = "query grid: unknown mode " + mode; return false; }
            for (const auto & [key, value] : c.items()) {
                if (key == "name" || key == "mode") continue;
                if (key == "k") g.query.candidates = value.get<int>();
                else if (key == "unit_score") {
                    if (!pflash_unit_score_from_name(value.get<std::string>(), g.query.unit_score)) {
                        error = "query grid: unit_score must be max or mean"; return false;
                    }
                } else if (key == "agg") {
                    if (!pflash_query_aggregate_from_name(value.get<std::string>(), g.query.aggregate)) {
                        error = "query grid: agg must be softmax or wmax"; return false;
                    }
                } else if (key == "temp") g.query.temperature = value.get<float>();
                else if (key == "null_margin") g.query.null_margin = value.get<float>();
                else if (key == "extent") {
                    if (value.is_string() && value.get<std::string>() == "gap") g.query.extent_gap = true;
                    else g.query.extent_margin = value.get<float>();
                }
                else if (key == "pin") g.query.pin_tokens = value.get<int>();
                else if (key == "pin_margin") {
                    if (value.is_string() && value.get<std::string>() == "inf") {
                        g.query.pin_margin = std::numeric_limits<float>::infinity();
                    } else if (value.is_string() && value.get<std::string>() == "gap") {
                        g.query.pin_gap = true;
                    } else {
                        g.query.pin_margin = value.get<float>();
                    }
                } else if (key == "tw") {
                    if (value.is_string() && value.get<std::string>() == "score") g.query.tw_score = true;
                    else g.query.tw_share = value.get<float>();
                } else if (key == "end") g.query.end_share = value.get<float>();
                else if (key == "oracle_pin") g.oracle_pin = value.get<bool>();
                else { error = "query grid: unknown key " + key + " in config " + g.name; return false; }
            }
        } catch (const std::exception & e) {
            error = std::string("query grid: bad config: ") + e.what();
            return false;
        }
        if (g.name.empty()) { error = "query grid: empty config name"; return false; }
        for (const auto & other : out) {
            if (other.name == g.name) { error = "query grid: duplicate config name " + g.name; return false; }
        }
        const auto & q = g.query;
        if ((g.mode == PFlashGridMode::Candidates) != (q.candidates > 0) || q.candidates < 0 ||
            q.candidates > kQueryCandidatesMax) {
            error = "query grid: " + g.name + ": k in [1, 8] exactly for mode cand";
            return false;
        }
        if (!(q.temperature > 0.0f) || !std::isfinite(q.temperature) || !(q.null_margin >= 0.0f) ||
            !(q.extent_margin >= 0.0f) || q.pin_tokens < 0 || q.pin_tokens > kQueryPinMax || !(q.pin_margin >= 0.0f) ||
            q.tw_share < 0.0f || q.end_share < 0.0f || !(q.tw_share + q.end_share < 1.0f) ||
            ((q.tw() || q.end_share > 0.0f) && g.mode != PFlashGridMode::Candidates)) {
            error = "query grid: " + g.name + ": a knob is out of range";
            return false;
        }
        out.push_back(std::move(g));
    }
    return true;
}

bool pflash_query_oracle_parse(const std::string & text, int n_tokens, PFlashQueryOracle & out, std::string & error) {
    error.clear();
    out = {};
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const std::exception & e) {
        error = std::string("query oracle: invalid JSON: ") + e.what();
        return false;
    }
    const auto spans_of = [&](const char * key, std::vector<PFlashTokenSpan> & spans) -> bool {
        if (!j.contains(key) || !j[key].is_array()) { error = std::string("query oracle: no ") + key; return false; }
        for (const auto & s : j[key]) {
            if (!s.is_array() || s.size() != 2) { error = "query oracle: a span is not [b, e]"; return false; }
            const int b = s[0].get<int>();
            const int e = s[1].get<int>();
            if (b < 0 || e <= b || e > n_tokens) { error = "query oracle: span outside the input"; return false; }
            spans.push_back({b, e});
        }
        std::sort(spans.begin(), spans.end(), [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
            return a.begin < b.begin;
        });
        return true;
    };
    std::vector<PFlashTokenSpan> query;
    try {
        if (!spans_of("query", query) || !spans_of("pin", out.pin)) return false;
    } catch (const std::exception & e) {
        error = std::string("query oracle: ") + e.what();
        return false;
    }
    for (const auto & s : query) {
        for (int t = s.begin; t < s.end; ++t) out.query.push_back(t);
    }
    std::sort(out.query.begin(), out.query.end());
    out.query.erase(std::unique(out.query.begin(), out.query.end()), out.query.end());
    std::vector<PFlashTokenSpan> merged;
    for (const auto & s : out.pin) {
        if (!merged.empty() && s.begin <= merged.back().end) merged.back().end = std::max(merged.back().end, s.end);
        else merged.push_back(s);
    }
    out.pin.swap(merged);
    if (out.query.empty()) { error = "query oracle: empty query"; return false; }
    return true;
}

} // namespace luce::pflash
