#include "pflash_selection.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace luce::pflash {

namespace {

constexpr const char * kModeEnv = "PFLASH_SELECT_MODE";
constexpr const char * kChunkEnv = "PFLASH_SELECT_CHUNK_SIZE";
constexpr const char * kQueryEnv = "PFLASH_SELECT_QUERY_TOKENS";
constexpr const char * kQueryParserEnv = "PFLASH_SELECT_QUERY_PARSER";
constexpr const char * kTopPEnv = "PFLASH_SELECT_TOP_P";
constexpr const char * kTopKEnv = "PFLASH_SELECT_TOPK";
constexpr const char * kSegmentsEnv = "PFLASH_SELECT_SEGMENTS";
constexpr const char * kSelectEnv = "PFLASH_SELECT_SCORE";
constexpr const char * kScorerEnv = "PFLASH_SELECT_SCORER";
constexpr const char * kSplitEnv = "PFLASH_SELECT_SPLIT";
constexpr const char * kContainerHeadersEnv = "PFLASH_SELECT_CONTAINER_HEADERS";
constexpr const char * kContainerHeaderMaxEnv = "PFLASH_SELECT_CONTAINER_HEADER_MAX";
constexpr const char * kCutMarkerEnv = "PFLASH_ASSEMBLY_CUT_MARKER";
constexpr const char * kStructHeadersEnv = "PFLASH_SELECT_STRUCT_HEADERS";

// "1" on, unset or "0" off, anything else invalid.
enum class Switch { Off, On, Invalid };
Switch parse_switch(const char * raw) {
    if (!raw || std::strcmp(raw, "0") == 0) return Switch::Off;
    if (std::strcmp(raw, "1") == 0) return Switch::On;
    return Switch::Invalid;
}

PFlashSelectionResult invalid_result(std::string error) {
    PFlashSelectionResult result;
    result.error = std::move(error);
    return result;
}

bool parse_int(const char * raw, int & out) {
    if (!raw || !*raw) return false;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parse_double(const char * raw, double & out) {
    if (!raw || !*raw) return false;
    errno = 0;
    char * end = nullptr;
    const double value = std::strtod(raw, &end);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

int scheduled_chunk_size(int input_tokens) {
    if (input_tokens < 500) return 128;
    if (input_tokens < 3000) return 512;
    return 1024;
}

} // namespace

bool pflash_container_headers_requested() noexcept {
    return parse_switch(std::getenv(kContainerHeadersEnv)) == Switch::On;
}

bool pflash_cut_markers_requested() noexcept {
    return parse_switch(std::getenv(kCutMarkerEnv)) == Switch::On;
}

bool pflash_struct_headers_requested() noexcept {
    return parse_switch(std::getenv(kStructHeadersEnv)) == Switch::On;
}

bool has_pflash_selection_environment() noexcept {
    return std::getenv(kModeEnv) != nullptr ||
           std::getenv(kChunkEnv) != nullptr ||
           std::getenv(kQueryEnv) != nullptr ||
           std::getenv(kQueryParserEnv) != nullptr ||
           std::getenv(kTopPEnv) != nullptr ||
           std::getenv(kTopKEnv) != nullptr ||
           std::getenv(kSegmentsEnv) != nullptr ||
           std::getenv(kSelectEnv) != nullptr ||
           std::getenv(kScorerEnv) != nullptr ||
           std::getenv(kSplitEnv) != nullptr;
}

bool pflash_chunk_is_structurally_required(
        int begin,
        int end,
        int query_begin,
        int query_end,
        int input_tokens,
        const std::vector<luce::common::PFlashTokenSpan> &
            required_instruction_spans,
        bool query_suffix_structural) noexcept {
    if (begin < 0 || end <= begin || query_begin < 0 ||
        query_end < query_begin || input_tokens < query_end ||
        end > input_tokens) {
        return false;
    }
    const bool query_chunk = begin < query_end && end > query_begin;
    const bool structural_suffix_chunk = query_suffix_structural &&
        begin < input_tokens && end > query_end;
    if (query_chunk || structural_suffix_chunk) return true;
    for (const auto & span : required_instruction_spans) {
        if (begin < span.end && end > span.begin) return true;
    }
    return false;
}

bool validate_pflash_instruction_spans(
        const std::vector<luce::common::PFlashTokenSpan> & spans,
        int input_tokens,
        std::string & error) noexcept {
    error.clear();
    if (input_tokens < 0) {
        error = "PFlash input token count must not be negative";
        return false;
    }
    if (spans.size() > luce::common::kPFlashMaxInstructionSpans) {
        error = "PFlash has too many instruction spans";
        return false;
    }
    int previous_end = 0;
    for (const auto & span : spans) {
        if (span.begin < 0 || span.end <= span.begin ||
            span.end > input_tokens) {
            error = "PFlash instruction span is outside the input";
            return false;
        }
        if (span.begin < previous_end) {
            error = "PFlash instruction spans must be ordered and non-overlapping";
            return false;
        }
        previous_end = span.end;
    }
    return true;
}

PFlashSelectionResult select_pflash_candidates(
        const std::vector<PFlashSelectionCandidate> & candidates,
        const PFlashSelectionPolicy & policy,
        PFlashSelectionMode mode) {
    if (mode == PFlashSelectionMode::Legacy) {
        return invalid_result("legacy mode does not use strict PFlash selection");
    }
    if (policy.token_budget <= 0) {
        return invalid_result("PFlash token budget must be positive");
    }
    if (!std::isfinite(policy.top_p) || policy.top_p <= 0.0 || policy.top_p > 1.0) {
        return invalid_result("PFlash top_p must be finite and in (0, 1]");
    }
    if (mode == PFlashSelectionMode::TopK && policy.top_k <= 0) {
        return invalid_result("PFlash top_k must be positive");
    }

    std::vector<const PFlashSelectionCandidate *> source_ranges;
    source_ranges.reserve(candidates.size());
    std::vector<size_t> ordinals;
    ordinals.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        if (candidate.begin < 0 || candidate.end <= candidate.begin) {
            return invalid_result("PFlash candidate range is invalid");
        }
        if (!std::isfinite(candidate.score)) {
            return invalid_result("PFlash candidate score must be finite");
        }
        source_ranges.push_back(&candidate);
        ordinals.push_back(candidate.ordinal);
    }

    std::sort(ordinals.begin(), ordinals.end());
    if (std::adjacent_find(ordinals.begin(), ordinals.end()) != ordinals.end()) {
        return invalid_result("PFlash candidate ordinals must be unique");
    }
    std::sort(source_ranges.begin(), source_ranges.end(),
        [](const auto * left, const auto * right) {
            if (left->begin != right->begin) return left->begin < right->begin;
            return left->end < right->end;
        });
    for (size_t index = 1; index < source_ranges.size(); ++index) {
        if (source_ranges[index - 1]->end > source_ranges[index]->begin) {
            return invalid_result("PFlash candidate ranges must not overlap");
        }
    }

    PFlashSelectionResult result;
    result.ok = true;
    result.stop = PFlashSelectionStop::CandidatesExhausted;
    std::vector<const PFlashSelectionCandidate *> selected_candidates;
    selected_candidates.reserve(candidates.size());

    std::vector<const PFlashSelectionCandidate *> optional;
    optional.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        if (candidate.mandatory) {
            const int length = candidate.end - candidate.begin;
            if (length > policy.token_budget - result.retained_tokens) {
                result = {};
                result.stop = PFlashSelectionStop::MandatoryQueryExceedsBudget;
                result.error = "mandatory PFlash retention tokens exceed the token budget";
                return result;
            }
            selected_candidates.push_back(&candidate);
            result.retained_tokens += length;
        } else {
            optional.push_back(&candidate);
        }
    }

    std::sort(optional.begin(), optional.end(),
        [](const auto * left, const auto * right) {
            const double left_score = std::max(0.0, left->score);
            const double right_score = std::max(0.0, right->score);
            if (left_score != right_score) return left_score > right_score;
            return left->ordinal < right->ordinal;
        });

    double max_score = 0.0;
    for (const auto * candidate : optional) {
        max_score = std::max(max_score, std::max(0.0, candidate->score));
    }
    double scaled_total = 0.0;
    if (max_score > 0.0) {
        for (const auto * candidate : optional) {
            scaled_total += std::max(0.0, candidate->score) / max_score;
        }
    }

    int kept_optional = 0;
    for (const auto * candidate : optional) {
        if (mode == PFlashSelectionMode::CumulativeTopP &&
            result.retained_mass >= policy.top_p) {
            result.stop = PFlashSelectionStop::TopPReached;
            break;
        }
        // Rank rule: K optional candidates in score order, the budget below
        // still a ceiling. K binding here means the budget never was.
        if (mode == PFlashSelectionMode::TopK && kept_optional >= policy.top_k) {
            result.stop = PFlashSelectionStop::TopKReached;
            break;
        }

        const int length = candidate->end - candidate->begin;
        if (length > policy.token_budget - result.retained_tokens) {
            result.stop = PFlashSelectionStop::BudgetReached;
            if (policy.skip_oversized) continue;
            break;
        }

        selected_candidates.push_back(candidate);
        result.retained_tokens += length;
        ++kept_optional;
        if (!optional.empty()) {
            result.retained_mass += max_score > 0.0
                ? (std::max(0.0, candidate->score) / max_score) / scaled_total
                : 1.0 / static_cast<double>(optional.size());
        }
    }

    std::sort(selected_candidates.begin(), selected_candidates.end(),
        [](const auto * left, const auto * right) {
            if (left->begin != right->begin) return left->begin < right->begin;
            return left->end < right->end;
        });
    result.ordinals.reserve(selected_candidates.size());
    for (const auto * candidate : selected_candidates) {
        result.ordinals.push_back(candidate->ordinal);
    }
    return result;
}

const char * pflash_selection_mode_name(PFlashSelectionMode mode) noexcept {
    switch (mode) {
        case PFlashSelectionMode::Legacy: return "legacy";
        case PFlashSelectionMode::BudgetOnly: return "budget_only";
        case PFlashSelectionMode::CumulativeTopP: return "top_p";
        case PFlashSelectionMode::TopK: return "top_k";
    }
    return "unknown";
}

const char * pflash_selection_stop_name(PFlashSelectionStop stop) noexcept {
    switch (stop) {
        case PFlashSelectionStop::TopPReached: return "top_p_reached";
        case PFlashSelectionStop::TopKReached: return "top_k_reached";
        case PFlashSelectionStop::BudgetReached: return "budget_reached";
        case PFlashSelectionStop::CandidatesExhausted: return "candidates_exhausted";
        case PFlashSelectionStop::InvalidInput: return "invalid_input";
        case PFlashSelectionStop::MandatoryQueryExceedsBudget:
            return "mandatory_query_exceeds_budget";
    }
    return "unknown";
}

const char * pflash_query_parser_name(PFlashQueryParser parser) noexcept {
    switch (parser) {
        case PFlashQueryParser::SemanticUser: return "latest_user";
        case PFlashQueryParser::ArbitraryTail: return "arbitrary_tail";
    }
    return "unknown";
}

bool resolve_pflash_selection(
        int input_tokens,
        int legacy_chunk_size,
        PFlashSelectionConfig & out,
        std::string & error) {
    error.clear();
    if (input_tokens < 0) {
        error = "PFlash input token count must not be negative";
        return false;
    }
    if (legacy_chunk_size <= 0) {
        error = "PFlash legacy chunk size must be positive";
        return false;
    }

    const char * mode_raw = std::getenv(kModeEnv);
    const char * chunk_raw = std::getenv(kChunkEnv);
    const char * query_raw = std::getenv(kQueryEnv);
    const char * query_parser_raw = std::getenv(kQueryParserEnv);
    const char * top_p_raw = std::getenv(kTopPEnv);
    const char * top_k_raw = std::getenv(kTopKEnv);
    const char * segments_raw = std::getenv(kSegmentsEnv);
    const char * select_raw = std::getenv(kSelectEnv);
    const char * scorer_raw = std::getenv(kScorerEnv);
    const char * split_raw = std::getenv(kSplitEnv);

    PFlashSelectionConfig config;
    config.configured = mode_raw || chunk_raw || query_raw ||
        query_parser_raw || top_p_raw || top_k_raw || segments_raw ||
        select_raw || scorer_raw || split_raw;
    if (scorer_raw) {
        if (std::strcmp(scorer_raw, "head") == 0) {
            config.scorer = PFlashScorer::Head;
        } else if (std::strcmp(scorer_raw, "legacy") == 0) {
            config.scorer = PFlashScorer::Legacy;
        } else if (std::strcmp(scorer_raw, "split") == 0) {
            config.scorer = PFlashScorer::Split;
        } else {
            error = std::string(kScorerEnv) + " must be head, legacy or split";
            return false;
        }
    }
    if (split_raw) {
        char * end = nullptr;
        errno = 0;
        const double value = std::strtod(split_raw, &end);
        if (errno != 0 || end == split_raw || *end != '\0' || !(value > 0.0 && value < 1.0)) {
            error = std::string(kSplitEnv) + " must be a fraction in (0, 1)";
            return false;
        }
        config.split_fraction = value;
    }
    if (segments_raw) {
        if (std::strcmp(segments_raw, "fixed") == 0) {
            config.segmentation = PFlashSegmentation::Fixed;
        } else if (std::strcmp(segments_raw, "probe") == 0) {
            config.segmentation = PFlashSegmentation::Probe;
        } else if (std::strcmp(segments_raw, "auto") != 0) {
            error = std::string(kSegmentsEnv) + " must be auto, fixed or probe";
            return false;
        }
    }
    if (select_raw) {
        if (std::strcmp(select_raw, "sum") == 0) {
            config.candidate_score = PFlashCandidateScore::Sum;
        } else if (std::strcmp(select_raw, "density") == 0) {
            config.candidate_score = PFlashCandidateScore::Density;
        } else if (std::strcmp(select_raw, "auto") != 0) {
            error = std::string(kSelectEnv) + " must be auto, sum or density";
            return false;
        }
    }
    config.chunk_size = legacy_chunk_size;

    if (mode_raw) {
        if (std::strcmp(mode_raw, "budget_only") == 0) {
            config.mode = PFlashSelectionMode::BudgetOnly;
        } else if (std::strcmp(mode_raw, "top_p") == 0) {
            config.mode = PFlashSelectionMode::CumulativeTopP;
        } else if (std::strcmp(mode_raw, "top_k") == 0) {
            config.mode = PFlashSelectionMode::TopK;
        } else {
            error = std::string(kModeEnv) +
                " must be budget_only, top_p or top_k";
            return false;
        }
    }
    config.selection_active = config.mode != PFlashSelectionMode::Legacy;

    if (chunk_raw) {
        if (!parse_int(chunk_raw, config.chunk_size) || config.chunk_size <= 0) {
            error = std::string(kChunkEnv) + " must be a positive integer";
            return false;
        }
    } else if (config.selection_active) {
        config.chunk_size = scheduled_chunk_size(input_tokens);
    }

    if (query_raw &&
        (!parse_int(query_raw, config.query_tokens) ||
         config.query_tokens < 1 || config.query_tokens > 512)) {
        error = std::string(kQueryEnv) + " must be an integer in [1, 512]";
        return false;
    }

    if (query_parser_raw) {
        if (std::strcmp(query_parser_raw, "latest_user") == 0) {
            config.query_parser = PFlashQueryParser::SemanticUser;
        } else if (std::strcmp(query_parser_raw, "arbitrary_tail") == 0) {
            config.query_parser = PFlashQueryParser::ArbitraryTail;
        } else {
            error = std::string(kQueryParserEnv) +
                " must be latest_user or arbitrary_tail";
            return false;
        }
    }

    if (top_p_raw &&
        (!parse_double(top_p_raw, config.top_p) ||
         config.top_p <= 0.0 || config.top_p > 1.0)) {
        error = std::string(kTopPEnv) + " must be finite and in (0, 1]";
        return false;
    }

    if (top_k_raw && (!parse_int(top_k_raw, config.top_k) || config.top_k <= 0)) {
        error = std::string(kTopKEnv) + " must be a positive integer";
        return false;
    }
    if (config.mode == PFlashSelectionMode::TopK && config.top_k <= 0) {
        error = std::string(kTopKEnv) + " is required when " +
            std::string(kModeEnv) + " is top_k";
        return false;
    }

    // Assembly switches: parsed whatever the mode, applied only by strict
    // selection. They do not mark the selection as configured.
    const Switch headers = parse_switch(std::getenv(kContainerHeadersEnv));
    const Switch markers = parse_switch(std::getenv(kCutMarkerEnv));
    if (headers == Switch::Invalid) {
        error = std::string(kContainerHeadersEnv) + " must be 0 or 1";
        return false;
    }
    if (markers == Switch::Invalid) {
        error = std::string(kCutMarkerEnv) + " must be 0 or 1";
        return false;
    }
    const Switch structure = parse_switch(std::getenv(kStructHeadersEnv));
    if (structure == Switch::Invalid) {
        error = std::string(kStructHeadersEnv) + " must be 0 or 1";
        return false;
    }
    config.container_headers = headers == Switch::On;
    config.cut_markers = markers == Switch::On;
    config.struct_headers = structure == Switch::On;
    if (const char * raw = std::getenv(kContainerHeaderMaxEnv)) {
        if (!parse_int(raw, config.container_header_max) ||
            config.container_header_max < 1 ||
            config.container_header_max > 4096) {
            error = std::string(kContainerHeaderMaxEnv) +
                " must be an integer in [1, 4096]";
            return false;
        }
    }

    out = config;
    return true;
}

std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
        const std::vector<float> & boundary_scores,
        int input_tokens,
        float threshold,
        int min_segment,
        int max_segment,
        const std::vector<int> & forced_cuts,
        const std::vector<float> & split_scores) {
    using luce::common::PFlashTokenSpan;
    std::vector<PFlashTokenSpan> spans;
    if (input_tokens <= 0 || (int) boundary_scores.size() < input_tokens ||
        min_segment < 1 || max_segment < min_segment) {
        return spans;
    }
    // Sub-unit scores feed only the oversize interior argmax;
    // the boundary threshold and merge floor always read the unit scores.
    const std::vector<float> & interior =
        (int) split_scores.size() >= input_tokens ? split_scores : boundary_scores;
    std::vector<uint8_t> forced((size_t) input_tokens + 1, 0);
    for (int cut : forced_cuts) {
        if (cut > 0 && cut < input_tokens) forced[(size_t) cut] = 1;
    }
    std::vector<int> cuts;
    cuts.push_back(0);
    for (int token = 1; token < input_tokens; ++token) {
        const bool wanted = forced[(size_t) token] ||
            (std::isfinite(boundary_scores[(size_t) token]) &&
             boundary_scores[(size_t) token] > threshold);
        if (!wanted) continue;
        if (!forced[(size_t) token] && token - cuts.back() < min_segment) continue;
        cuts.push_back(token);
    }
    cuts.push_back(input_tokens);
    for (size_t index = 1; index < cuts.size(); ++index) {
        int begin = cuts[index - 1];
        const int end = cuts[index];
        while (end - begin > max_segment) {
            // Split at the best-scoring interior token in the second half of
            // the next max_segment piece (the distance guard keeps the split
            // off the near edge), honoring the min_segment margins on both
            // sides; else on a fixed grid.
            const int lo = std::max(begin + min_segment, begin + max_segment / 2);
            const int hi = std::min(end - min_segment, begin + max_segment);
            int best = -1;
            float best_score = 0.0f;
            for (int token = lo; token <= hi; ++token) {
                const float score = interior[(size_t) token];
                if (std::isfinite(score) && score > best_score) {
                    best_score = score;
                    best = token;
                }
            }
            if (best < 0) best = begin + max_segment;
            spans.push_back({begin, best});
            begin = best;
        }
        spans.push_back({begin, end});
    }
    return spans;
}

PFlashSelectionResult select_pflash_split(
        const std::vector<PFlashSelectionCandidate> & head,
        const std::vector<PFlashSelectionCandidate> & other,
        const PFlashSelectionPolicy & policy,
        double head_fraction,
        PFlashSelectionMode mode) {
    PFlashSelectionResult result;
    if (mode == PFlashSelectionMode::TopK) {
        // A per-pass K would keep up to 2K segments, which is not the rule the
        // mode names; fail closed rather than quietly double it.
        result.stop = PFlashSelectionStop::InvalidInput;
        result.error = "split selection does not support top_k";
        return result;
    }
    if (head.size() != other.size() || !(head_fraction > 0.0 && head_fraction < 1.0)) {
        result.stop = PFlashSelectionStop::InvalidInput;
        result.error = "split selection needs matching candidate lists and a fraction in (0, 1)";
        return result;
    }
    for (size_t i = 0; i < head.size(); ++i) {
        if (head[i].ordinal != other[i].ordinal || head[i].begin != other[i].begin ||
            head[i].end != other[i].end || head[i].mandatory != other[i].mandatory) {
            result.stop = PFlashSelectionStop::InvalidInput;
            result.error = "split selection candidate lists describe different spans";
            return result;
        }
    }
    PFlashSelectionPolicy first = policy;
    first.token_budget = static_cast<int>(policy.token_budget * head_fraction);
    // Mandatory spans must fit even when the head's share is small.
    int mandatory = 0;
    for (const auto & c : head) if (c.mandatory) mandatory += c.end - c.begin;
    first.token_budget = (std::max)(first.token_budget, (std::min)(mandatory, policy.token_budget));
    const PFlashSelectionResult pass1 = select_pflash_candidates(head, first, mode);
    if (!pass1.ok) return pass1;
    std::vector<uint8_t> taken(head.size(), 0);
    for (size_t ordinal : pass1.ordinals) {
        for (size_t i = 0; i < head.size(); ++i) if (head[i].ordinal == ordinal) taken[i] = 1;
    }
    std::vector<PFlashSelectionCandidate> rest;
    for (size_t i = 0; i < other.size(); ++i) {
        if (taken[i]) continue;
        PFlashSelectionCandidate c = other[i];
        c.mandatory = false;  // mandatory spans were charged in pass 1
        rest.push_back(c);
    }
    PFlashSelectionPolicy second = policy;
    second.token_budget = policy.token_budget - pass1.retained_tokens;
    const PFlashSelectionResult pass2 = second.token_budget > 0
        ? select_pflash_candidates(rest, second, mode)
        : PFlashSelectionResult{};
    result.ok = true;
    result.ordinals = pass1.ordinals;
    result.ordinals.insert(result.ordinals.end(), pass2.ordinals.begin(), pass2.ordinals.end());
    std::sort(result.ordinals.begin(), result.ordinals.end());
    result.retained_tokens = pass1.retained_tokens + pass2.retained_tokens;
    result.retained_mass = pass1.retained_mass;  // the head's normalised mass share
    result.stop = second.token_budget > 0 ? pass2.stop : pass1.stop;
    return result;
}

const char * pflash_scorer_name(PFlashScorer scorer) noexcept {
    switch (scorer) {
        case PFlashScorer::Head: return "head";
        case PFlashScorer::Legacy: return "legacy";
        case PFlashScorer::Split: return "split";
    }
    return "unknown";
}

const char * pflash_segmentation_name(PFlashSegmentation segmentation) noexcept {
    switch (segmentation) {
        case PFlashSegmentation::Auto: return "auto";
        case PFlashSegmentation::Fixed: return "fixed";
        case PFlashSegmentation::Probe: return "probe";
    }
    return "unknown";
}

const char * pflash_candidate_score_name(PFlashCandidateScore score) noexcept {
    switch (score) {
        case PFlashCandidateScore::Auto: return "auto";
        case PFlashCandidateScore::Sum: return "sum";
        case PFlashCandidateScore::Density: return "density";
    }
    return "unknown";
}

namespace {

using luce::common::PFlashTokenSpan;

bool inside_any(int token, const std::vector<PFlashTokenSpan> & spans) {
    for (const auto & span : spans) {
        if (token >= span.begin && token < span.end) return true;
    }
    return false;
}

// Sorted, merged copy of ``spans`` (touching spans merge).
std::vector<PFlashTokenSpan> merge_spans(std::vector<PFlashTokenSpan> spans) {
    std::sort(spans.begin(), spans.end(),
        [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
            return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
        });
    std::vector<PFlashTokenSpan> merged;
    for (const auto & span : spans) {
        if (span.end <= span.begin) continue;
        if (!merged.empty() && span.begin <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, span.end);
        } else {
            merged.push_back(span);
        }
    }
    return merged;
}

int span_tokens(const std::vector<PFlashTokenSpan> & spans) {
    int total = 0;
    for (const auto & span : spans) total += span.end - span.begin;
    return total;
}

} // namespace

std::vector<int> pflash_container_starts(
        const std::vector<float> & container_probs,
        int input_tokens,
        float threshold,
        int min_spacing,
        const std::vector<PFlashTokenSpan> & excluded) {
    std::vector<int> starts;
    if (input_tokens <= 0 || (int) container_probs.size() < input_tokens) {
        return starts;
    }
    min_spacing = std::max(1, min_spacing);
    for (int token = 0; token < input_tokens; ++token) {
        const float p = container_probs[(size_t) token];
        if (!std::isfinite(p) || !(p > threshold)) continue;
        if (inside_any(token, excluded)) continue;
        if (!starts.empty() && token - starts.back() < min_spacing) continue;
        starts.push_back(token);
    }
    return starts;
}

std::vector<int> pflash_union_starts(const std::vector<int> & a,
                                     const std::vector<int> & b) {
    std::vector<int> out = a;
    out.insert(out.end(), b.begin(), b.end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

PFlashContainers pflash_container_headers(
        const std::vector<int32_t> & ids,
        const std::vector<int> & starts,
        const std::vector<PFlashTokenSpan> & excluded,
        const std::vector<uint8_t> & newline_vocab,
        int header_max) {
    PFlashContainers out;
    const int input_tokens = (int) ids.size();
    header_max = std::max(1, header_max);
    for (size_t index = 0; index < starts.size(); ++index) {
        const int start = starts[index];
        if (start < 0 || start >= input_tokens) continue;
        if (!out.starts.empty() && start <= out.starts.back()) continue;
        int end = index + 1 < starts.size()
            ? std::min(starts[index + 1], input_tokens) : input_tokens;
        for (const auto & span : excluded) {
            if (span.begin > start && span.begin < end) end = span.begin;
        }
        if (end <= start) continue;
        const int cap = std::min(end, start + header_max);
        int header_end = cap;
        for (int token = start + 1; token < cap; ++token) {
            const int32_t id = ids[(size_t) token];
            if (id >= 0 && (size_t) id < newline_vocab.size() &&
                newline_vocab[(size_t) id]) {
                header_end = token + 1;
                break;
            }
        }
        out.starts.push_back(start);
        out.extents.push_back({start, end});
        out.headers.push_back({start, header_end});
    }
    return out;
}

PFlashAssemblyResult pflash_assemble_selection(
        const std::vector<PFlashSelectionCandidate> & candidates,
        const std::vector<size_t> & selected_ordinals,
        const PFlashAssemblyPolicy & policy) {
    PFlashAssemblyResult result;
    if (policy.token_budget <= 0 || policy.marker_tokens < 0) {
        result.error = "PFlash assembly budget must be positive";
        return result;
    }
    const PFlashContainers * containers = policy.containers;
    if (containers && (containers->extents.size() != containers->headers.size())) {
        result.error = "PFlash container extents and headers differ in length";
        return result;
    }
    std::unordered_map<size_t, size_t> index_of;
    for (size_t i = 0; i < candidates.size(); ++i) index_of[candidates[i].ordinal] = i;
    // Selected candidates in source order.
    std::vector<const PFlashSelectionCandidate *> selected;
    for (size_t ordinal : selected_ordinals) {
        const auto found = index_of.find(ordinal);
        if (found == index_of.end()) {
            result.error = "PFlash assembly got an unknown ordinal";
            return result;
        }
        selected.push_back(&candidates[found->second]);
    }
    std::sort(selected.begin(), selected.end(),
        [](const auto * a, const auto * b) { return a->begin < b->begin; });
    // Drop order: the reverse of the selection order (lowest clamped score
    // first, the higher ordinal first among equals).
    std::vector<const PFlashSelectionCandidate *> droppable;
    for (const auto * candidate : selected) {
        if (!candidate->mandatory) droppable.push_back(candidate);
    }
    std::sort(droppable.begin(), droppable.end(),
        [](const auto * a, const auto * b) {
            const double sa = std::max(0.0, a->score);
            const double sb = std::max(0.0, b->score);
            if (sa != sb) return sa < sb;
            return a->ordinal > b->ordinal;
        });
    std::vector<uint8_t> dropped_flag(candidates.size(), 0);
    size_t next_drop = 0;

    for (;;) {
        ++result.passes;
        std::vector<PFlashTokenSpan> segment_spans;
        for (const auto * candidate : selected) {
            if (dropped_flag[index_of[candidate->ordinal]]) continue;
            segment_spans.push_back({candidate->begin, candidate->end});
        }
        const std::vector<PFlashTokenSpan> segments_merged = merge_spans(segment_spans);
        std::vector<PFlashTokenSpan> all = segment_spans;
        std::vector<PFlashTokenSpan> headers;
        if (containers && !containers->extents.empty()) {
            // Containers holding a selected non-mandatory candidate.
            std::vector<uint8_t> has_content(containers->extents.size(), 0);
            const auto & extents = containers->extents;
            for (const auto * candidate : selected) {
                if (candidate->mandatory || dropped_flag[index_of[candidate->ordinal]]) continue;
                // First container ending after the candidate begins.
                auto it = std::upper_bound(extents.begin(), extents.end(), candidate->begin,
                    [](int value, const PFlashTokenSpan & span) { return value < span.end; });
                for (; it != extents.end() && it->begin < candidate->end; ++it) {
                    has_content[(size_t) (it - extents.begin())] = 1;
                }
            }
            for (size_t c = 0; c < extents.size(); ++c) {
                if (has_content[c]) headers.push_back(containers->headers[c]);
            }
            all.insert(all.end(), headers.begin(), headers.end());
        }
        const std::vector<PFlashTokenSpan> kept = merge_spans(all);
        const int kept_tokens = span_tokens(kept);
        const int markers = policy.marker_tokens > 0 && !kept.empty()
            ? (int) kept.size() - 1 : 0;
        const int total = kept_tokens + markers * policy.marker_tokens;
        if (total <= policy.token_budget || next_drop >= droppable.size()) {
            result.ok = true;
            result.kept = kept;
            result.header_tokens = kept_tokens - span_tokens(segments_merged);
            result.cut_markers = markers;
            result.retained_tokens = total;
            for (const auto & header : headers) {
                // Added when some header token was not already a segment's.
                bool covered = false;
                for (const auto & span : segments_merged) {
                    if (span.begin <= header.begin && span.end >= header.end) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) result.headers_added.push_back(header);
            }
            for (const auto * candidate : selected) {
                if (!dropped_flag[index_of[candidate->ordinal]]) {
                    result.ordinals.push_back(candidate->ordinal);
                }
            }
            std::sort(result.ordinals.begin(), result.ordinals.end());
            return result;
        }
        // Drop the lowest-ranked candidate and recompute: dropping it may
        // also free its container's header or a marker, so one at a time
        // drops no more than the overflow needs.
        const auto * candidate = droppable[next_drop++];
        dropped_flag[index_of[candidate->ordinal]] = 1;
        result.dropped.push_back(candidate->ordinal);
    }
}

std::vector<int32_t> pflash_assemble_ids(
        const std::vector<int32_t> & ids,
        const std::vector<PFlashTokenSpan> & kept,
        const std::vector<int32_t> & marker_ids) {
    std::vector<int32_t> out;
    int previous_end = -1;
    for (const auto & span : kept) {
        const int begin = std::max(0, span.begin);
        const int end = std::min((int) ids.size(), span.end);
        if (end <= begin) continue;
        if (previous_end >= 0 && begin > previous_end) {
            out.insert(out.end(), marker_ids.begin(), marker_ids.end());
        }
        out.insert(out.end(), ids.begin() + begin, ids.begin() + end);
        previous_end = end;
    }
    return out;
}

} // namespace luce::pflash
