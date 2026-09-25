// PFlash scoring pipeline glue. See pflash_compress.h.

#include "pflash_compress.h"

#include "pflash_selection.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

int env_int(const char * name, int fallback) {
    if (const char * v = std::getenv(name)) {
        int x = std::atoi(v);
        if (x >= 0) return x;
    }
    return fallback;
}

float env_float(const char * name, float def) {
    if (const char * v = std::getenv(name)) {
        try { return std::stof(v); } catch (...) {}
    }
    return def;
}

void force_chunk_neighborhood(std::vector<uint8_t> & forced, int n_chunks,
                                     int chunk, int radius) {
    int lo = std::max(0, chunk - radius);
    int hi = std::min(n_chunks - 1, chunk + radius);
    for (int c = lo; c <= hi; ++c) forced[(size_t)c] = 1;
}

void write_compression_trace(
        int input_tokens,
        float keep_ratio,
        int chunk_size,
        int n_lookahead,
        int pool_kernel,
        int n_keep,
        const std::vector<std::pair<float, int>> & chunk_means,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        const std::vector<int32_t> & compressed_ids,
        const PFlashTraceFields * trace_fields) {
    const char * path = std::getenv("PFLASH_TRACE_PATH");
    if (!path || !*path) return;

    FILE * file = std::fopen(path, "a");
    if (!file) {
        std::fprintf(stderr, "[pflash-trace] cannot append %s\n", path);
        return;
    }

    std::vector<float> scores(selected.size(), 0.0f);
    for (const auto & chunk : chunk_means) {
        scores[(size_t)chunk.second] = chunk.first;
    }
    const bool has_exact_scores = trace_fields &&
        trace_fields->exact_chunk_scores &&
        trace_fields->exact_chunk_scores->size() == scores.size();
    if (trace_fields &&
        trace_fields->selector_mode !=
            luce::pflash::PFlashSelectionMode::Legacy &&
        !has_exact_scores) {
        std::fclose(file);
        std::fprintf(stderr, "[pflash-trace] exact strict scores unavailable\n");
        return;
    }

    std::fprintf(file,
        "{\"schema_version\":%d,\"input_tokens\":%d,\"keep_ratio\":%.9g",
        trace_fields ? 3 : 1, input_tokens, keep_ratio);
    if (trace_fields) {
        std::fputs(",\"input_ids\":[", file);
        for (size_t index = 0; index < trace_fields->input_ids->size(); ++index) {
            if (index) std::fputc(',', file);
            std::fprintf(file, "%d", (*trace_fields->input_ids)[index]);
        }
        std::fprintf(file,
            "],\"query_begin\":%d,\"query_end\":%d,"
            "\"selector_mode\":\"%s\",\"query_parser\":\"%s\","
            "\"token_budget\":%d,\"top_k\":%d,"
            "\"retained_tokens\":%d",
            trace_fields->query_begin, trace_fields->query_end,
            luce::pflash::pflash_selection_mode_name(
                trace_fields->selector_mode),
            luce::pflash::pflash_query_parser_name(trace_fields->query_parser),
            trace_fields->token_budget, trace_fields->top_k,
            trace_fields->retained_tokens);
        std::fputs(",\"required_instruction_spans\":[", file);
        if (trace_fields->required_instruction_spans) {
            for (size_t index = 0;
                 index < trace_fields->required_instruction_spans->size();
                 ++index) {
                if (index) std::fputc(',', file);
                const auto & span =
                    (*trace_fields->required_instruction_spans)[index];
                std::fprintf(file, "[%d,%d]", span.begin, span.end);
            }
        }
        std::fputc(']', file);
        if (trace_fields->selector_mode ==
            luce::pflash::PFlashSelectionMode::Legacy) {
            std::fputs(",\"stop_reason\":null,\"retained_mass\":null", file);
        } else {
            std::fprintf(file,
                ",\"stop_reason\":\"%s\",\"retained_mass\":%.17g",
                luce::pflash::pflash_selection_stop_name(trace_fields->stop),
                trace_fields->retained_mass);
        }
    }
    if (trace_fields) {
        std::fprintf(file, ",\"segmentation\":\"%s\",\"candidate_score\":\"%s\",\"scorer\":\"%s\",\"split_fraction\":%.4f",
                     trace_fields->segmentation, trace_fields->candidate_score,
                     trace_fields->scorer, trace_fields->split_fraction);
        if (trace_fields->other_chunk_scores) {
            std::fputs(",\"other_chunk_scores\":[", file);
            for (size_t index = 0; index < trace_fields->other_chunk_scores->size(); ++index) {
                const double score = (*trace_fields->other_chunk_scores)[index];
                if (index) std::fputc(',', file);
                if (std::isfinite(score)) std::fprintf(file, "%.9g", score); else std::fputs("null", file);
            }
            std::fputc(']', file);
        }
        if (trace_fields->segments) {
            std::fputs(",\"segments\":[", file);
            for (size_t index = 0; index < trace_fields->segments->size(); ++index) {
                const auto & span = (*trace_fields->segments)[index];
                std::fprintf(file, "%s[%d,%d]", index ? "," : "", span.begin, span.end);
            }
            std::fputc(']', file);
        }
        if (trace_fields->record_starts) {
            std::fputs(",\"record_starts\":[", file);
            for (size_t index = 0; index < trace_fields->record_starts->size(); ++index) {
                std::fprintf(file, "%s%d", index ? "," : "",
                             (*trace_fields->record_starts)[index]);
            }
            std::fputc(']', file);
        }
        if (trace_fields->headers_added) {
            std::fputs(",\"headers_added\":[", file);
            for (size_t index = 0; index < trace_fields->headers_added->size(); ++index) {
                const auto & span = (*trace_fields->headers_added)[index];
                std::fprintf(file, "%s[%d,%d]", index ? "," : "", span.begin, span.end);
            }
            std::fputc(']', file);
        }
        if (trace_fields->header_tokens >= 0) {
            std::fprintf(file, ",\"header_tokens\":%d", trace_fields->header_tokens);
        }
        if (trace_fields->struct_starts) {
            std::fputs(",\"struct_starts\":[", file);
            for (size_t index = 0; index < trace_fields->struct_starts->size(); ++index) {
                std::fprintf(file, "%s%d", index ? "," : "",
                             (*trace_fields->struct_starts)[index]);
            }
            std::fputc(']', file);
        }
        if (trace_fields->struct_headers_added) {
            std::fputs(",\"struct_headers_added\":[", file);
            for (size_t index = 0; index < trace_fields->struct_headers_added->size(); ++index) {
                const auto & span = (*trace_fields->struct_headers_added)[index];
                std::fprintf(file, "%s[%d,%d]", index ? "," : "", span.begin, span.end);
            }
            std::fputc(']', file);
        }
        if (trace_fields->cut_markers >= 0) {
            std::fprintf(file, ",\"cut_markers\":%d", trace_fields->cut_markers);
        }
        if (trace_fields->drafter_profile && trace_fields->drafter_profile->valid) {
            const auto & prof = *trace_fields->drafter_profile;
            std::fprintf(file,
                ",\"drafter_profile\":{\"S\":%d,\"resume\":%d,\"new_tokens\":%d,"
                "\"ubatches\":%d,\"mask_ms\":%.6g,\"graph_ms\":%.6g,"
                "\"attn_ms\":%.6g,\"deltanet_ms\":%.6g,\"score_ms\":%.6g,"
                "\"total_ms\":%.6g}",
                prof.input_tokens, prof.resume, prof.new_tokens, prof.ubatches,
                prof.mask_ms, prof.graph_ms, prof.attn_ms, prof.deltanet_ms,
                prof.score_ms, prof.total_ms);
        }
        if (trace_fields->assembly_dropped) {
            std::fputs(",\"assembly_dropped\":[", file);
            for (size_t index = 0; index < trace_fields->assembly_dropped->size(); ++index) {
                std::fprintf(file, "%s%zu", index ? "," : "",
                             (*trace_fields->assembly_dropped)[index]);
            }
            std::fputc(']', file);
        }
    }
    std::fprintf(file,
        ",\"chunk_size\":%d,\"n_lookahead\":%d,\"pool_kernel\":%d,"
        "\"n_keep\":%d,\"chunk_scores\":[",
        chunk_size, n_lookahead, pool_kernel, n_keep);
    for (size_t index = 0; index < scores.size(); ++index) {
        if (index) std::fputc(',', file);
        const double score = has_exact_scores
            ? (*trace_fields->exact_chunk_scores)[index]
            : (double) scores[index];
        if (std::isfinite(score)) {
            std::fprintf(file, has_exact_scores ? "%.17g" : "%.9g", score);
        } else {
            std::fputs("null", file);
        }
    }
    std::fputs("],\"selected_chunks\":[", file);
    bool first = true;
    for (size_t index = 0; index < selected.size(); ++index) {
        if (!selected[index]) continue;
        if (!first) std::fputc(',', file);
        std::fprintf(file, "%zu", index);
        first = false;
    }
    std::fputs("],\"forced_chunks\":[", file);
    first = true;
    for (size_t index = 0; index < forced.size(); ++index) {
        if (!forced[index]) continue;
        if (!first) std::fputc(',', file);
        std::fprintf(file, "%zu", index);
        first = false;
    }
    std::fputs("],\"compressed_ids\":[", file);
    for (size_t index = 0; index < compressed_ids.size(); ++index) {
        if (index) std::fputc(',', file);
        std::fprintf(file, "%d", compressed_ids[index]);
    }
    std::fputs("]}\n", file);
    std::fclose(file);
}

namespace {
thread_local std::vector<PFlashTokenSpan> g_last_kept_spans;
thread_local PFlashScoringStats g_last_scoring_stats;
thread_local std::vector<PFlashCandidateLift> g_last_candidate_lifts;
thread_local PFlashDrafterProfile g_last_drafter_profile;
} // namespace

const PFlashDrafterProfile & pflash_last_drafter_profile() {
    return g_last_drafter_profile;
}

void pflash_set_drafter_profile(const PFlashDrafterProfile & profile) {
    g_last_drafter_profile = profile;
}

const std::vector<PFlashCandidateLift> & pflash_last_candidate_lifts() {
    return g_last_candidate_lifts;
}

const PFlashScoringStats & pflash_last_scoring_stats() {
    return g_last_scoring_stats;
}

void pflash_set_scoring_stats(const PFlashScoringStats & stats) {
    g_last_scoring_stats = stats;
}

const std::vector<PFlashTokenSpan> & pflash_last_kept_spans() {
    return g_last_kept_spans;
}

void pflash_clear_kept_spans() {
    g_last_kept_spans.clear();
    g_last_scoring_stats = {};
    g_last_candidate_lifts.clear();
    g_last_drafter_profile = {};
}

std::vector<int32_t> select_pflash_chunks(
        const std::vector<int32_t> & ids,
        const std::vector<float> & token_scores,
        float keep_ratio,
        int n_lookahead,
        int score_query_end,
        int pool_kernel,
        const luce::pflash::PFlashSelectionConfig & config,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        bool direct_mass,
        bool write_trace,
        const std::vector<PFlashTokenSpan> * segments,
        bool density,
        const std::vector<float> * other_token_scores,
        double split_fraction,
        const luce::pflash::PFlashContainers * containers) {
    const int input_tokens = (int) ids.size();
    const int query_end = score_query_end < 0 ? input_tokens : score_query_end;
    const int query_tokens = std::min(n_lookahead, query_end);
    const int query_begin = query_end - query_tokens;
    const int selector_budget = (int) std::floor(
        (double) input_tokens * (double) keep_ratio);
    // Fixed grid unless the caller provides variable-length segments.
    const int n_chunks = segments
        ? (int) segments->size()
        : (input_tokens + config.chunk_size - 1) / config.chunk_size;

    std::vector<luce::pflash::PFlashSelectionCandidate> candidates;
    std::vector<std::pair<float, int>> chunk_means;
    std::vector<double> exact_chunk_scores;
    candidates.reserve((size_t) n_chunks);
    chunk_means.reserve((size_t) n_chunks);
    exact_chunk_scores.reserve((size_t) n_chunks);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int begin = segments ? (*segments)[(size_t) chunk].begin : chunk * config.chunk_size;
        const int end = segments ? (*segments)[(size_t) chunk].end
                                 : std::min(input_tokens, begin + config.chunk_size);
        double score = 0.0;
        for (int token = begin; token < end; ++token) {
            score += token_scores[(size_t) token];
        }
        if (!direct_mass || density) {
            score /= (double) std::max(1, end - begin);
        }
        const bool mandatory =
            luce::pflash::pflash_chunk_is_structurally_required(
                begin, end, query_begin, query_end, input_tokens,
                required_instruction_spans,
                /*query_suffix_structural=*/ !config.query_suffix_candidates);
        candidates.push_back({(size_t) chunk, begin, end, score, mandatory});
        chunk_means.push_back({(float) score, chunk});
        exact_chunk_scores.push_back(score);
    }
    // Two-scorer selection: the other scorer's mean per-token score over the
    // same spans (its native ranking rule).
    std::vector<luce::pflash::PFlashSelectionCandidate> other_candidates;
    std::vector<double> other_scores;
    const bool split = other_token_scores != nullptr && split_fraction > 0.0;
    if (split) {
        for (const auto & candidate : candidates) {
            double score = 0.0;
            for (int token = candidate.begin; token < candidate.end; ++token) {
                score += (*other_token_scores)[(size_t) token];
            }
            score /= (double) std::max(1, candidate.end - candidate.begin);
            other_candidates.push_back({candidate.ordinal, candidate.begin, candidate.end, score, candidate.mandatory});
            other_scores.push_back(score);
        }
    }

    const luce::pflash::PFlashSelectionPolicy policy{selector_budget, config.top_p,
                                                      /*skip_oversized=*/ segments != nullptr,
                                                      config.top_k};
    const auto selected = split
        ? luce::pflash::select_pflash_split(candidates, other_candidates, policy, split_fraction, config.mode)
        : luce::pflash::select_pflash_candidates(candidates, policy, config.mode);
    if (!selected.ok) {
        set_last_error("PFlash selection failed: " + selected.error);
        std::fprintf(stderr,
            "[pflash-select] ERROR mode=%s budget=%d stop=%s: %s\n",
            luce::pflash::pflash_selection_mode_name(config.mode),
            selector_budget,
            luce::pflash::pflash_selection_stop_name(selected.stop),
            selected.error.c_str());
        std::fflush(stderr);
        return {};
    }

    // Assembly experiments (container headers, request-structure headers,
    // cut markers): off by default, and then nothing below changes.
    const bool with_headers = (config.container_headers || config.struct_headers) &&
        containers != nullptr;
    const bool with_markers = config.cut_markers && !config.cut_marker_ids.empty();
    luce::pflash::PFlashAssemblyResult assembly;
    if (with_headers || with_markers) {
        luce::pflash::PFlashAssemblyPolicy assembly_policy;
        assembly_policy.token_budget = selector_budget;
        assembly_policy.containers = with_headers ? containers : nullptr;
        assembly_policy.marker_tokens =
            with_markers ? (int) config.cut_marker_ids.size() : 0;
        assembly = luce::pflash::pflash_assemble_selection(
            candidates, selected.ordinals, assembly_policy);
        if (!assembly.ok) {
            set_last_error("PFlash assembly failed: " + assembly.error);
            std::fprintf(stderr, "[pflash-select] ERROR assembly: %s\n",
                         assembly.error.c_str());
            std::fflush(stderr);
            return {};
        }
    }
    const bool assembled = with_headers || with_markers;
    const std::vector<size_t> & final_ordinals =
        assembled ? assembly.ordinals : selected.ordinals;
    // The added headers that open a request-structure unit.
    static const std::vector<int> no_starts;
    const std::vector<int> & struct_starts =
        with_headers ? containers->struct_starts : no_starts;
    std::vector<PFlashTokenSpan> struct_headers_added;
    for (const auto & header : assembly.headers_added) {
        if (std::binary_search(struct_starts.begin(), struct_starts.end(), header.begin)) {
            struct_headers_added.push_back(header);
        }
    }

    std::vector<uint8_t> selected_mask((size_t) n_chunks, 0);
    std::vector<uint8_t> mandatory_mask((size_t) n_chunks, 0);
    for (const auto & candidate : candidates) {
        if (candidate.mandatory) mandatory_mask[candidate.ordinal] = 1;
    }
    for (size_t ordinal : final_ordinals) {
        if (ordinal >= selected_mask.size()) {
            set_last_error("PFlash selector returned an invalid ordinal");
            return {};
        }
        selected_mask[ordinal] = 1;
    }

    std::vector<int32_t> output;
    output.reserve((size_t) selected.retained_tokens);
    g_last_kept_spans.clear();
    g_last_candidate_lifts.clear();
    if (direct_mass) {
        // Head mass sums to one over the keys, so uniform attention gives
        // each token 1/input of it.
        for (const auto & candidate : candidates) {
            double mass = 0.0;
            for (int token = candidate.begin; token < candidate.end; ++token) {
                mass += token_scores[(size_t) token];
            }
            const int length = std::max(1, candidate.end - candidate.begin);
            g_last_candidate_lifts.push_back(
                {{candidate.begin, candidate.end},
                 mass / (double) length * (double) input_tokens});
        }
    }
    if (assembled) {
        // Kept ranges are arbitrary token ranges now (header spans are
        // sub-segment); markers go between ranges that are not adjacent.
        static const std::vector<int32_t> no_markers;
        output = luce::pflash::pflash_assemble_ids(
            ids, assembly.kept, with_markers ? config.cut_marker_ids : no_markers);
        g_last_kept_spans = assembly.kept;
    } else {
        for (const auto & candidate : candidates) {
            if (!selected_mask[candidate.ordinal]) continue;
            output.insert(output.end(),
                          ids.begin() + candidate.begin,
                          ids.begin() + candidate.end);
            if (!g_last_kept_spans.empty() &&
                g_last_kept_spans.back().end == candidate.begin) {
                g_last_kept_spans.back().end = candidate.end;
            } else {
                g_last_kept_spans.push_back({candidate.begin, candidate.end});
            }
        }
    }

    std::fprintf(stderr,
        "[pflash-select] selected mode=%s scorer=%s segments=%s score=%s chunk=%d query=%d "
        "budget=%d selected_tokens=%zu chunks=%zu/%d stop=%s mass=%.9g\n",
        luce::pflash::pflash_selection_mode_name(config.mode),
        split ? "split" : "single",
        segments ? "probe" : "fixed", density ? "density" : "sum",
        segments ? 0 : config.chunk_size, query_tokens, selector_budget, output.size(),
        final_ordinals.size(), n_chunks,
        luce::pflash::pflash_selection_stop_name(selected.stop),
        selected.retained_mass);
    if (assembled) {
        std::fprintf(stderr,
            "[pflash-assembly] containers=%zu headers_added=%zu header_tokens=%d "
            "cut_markers=%d marker_tokens=%d dropped=%zu passes=%d "
            "retained=%d budget=%d\n",
            with_headers ? containers->starts.size() : (size_t) 0,
            assembly.headers_added.size(), assembly.header_tokens,
            assembly.cut_markers,
            with_markers ? (int) config.cut_marker_ids.size() : 0,
            assembly.dropped.size(), assembly.passes,
            assembly.retained_tokens, selector_budget);
    }
    if (config.struct_headers) {
        std::fprintf(stderr,
            "[pflash-assembly] struct_starts=%zu struct_headers_added=%zu\n",
            struct_starts.size(), struct_headers_added.size());
    }
    std::fflush(stderr);

    if (write_trace) {
        const int trace_chunk = segments ? 0 : config.chunk_size;
        const int n_keep_approx = segments
            ? (int) final_ordinals.size()
            : std::max(1, (selector_budget + config.chunk_size - 1) / config.chunk_size);
        PFlashTraceFields strict_fields{
            &ids, query_begin, query_end, config.mode, config.query_parser,
            selector_budget,
            selected.stop,
            assembled ? assembly.retained_tokens : selected.retained_tokens,
            selected.retained_mass,
            &exact_chunk_scores, &required_instruction_spans};
        strict_fields.segments = segments;
        strict_fields.segmentation = segments ? "probe" : "fixed";
        strict_fields.candidate_score = density ? "density" : "sum";
        strict_fields.scorer = split ? "split" : luce::pflash::pflash_scorer_name(config.scorer);
        strict_fields.split_fraction = split ? split_fraction : 0.0;
        strict_fields.other_chunk_scores = split ? &other_scores : nullptr;
        strict_fields.top_k =
            config.mode == luce::pflash::PFlashSelectionMode::TopK ? config.top_k : 0;
        if (with_headers) {
            // The probe's record starts: every start unless the request's
            // structure gave some too.
            if (config.container_headers) {
                strict_fields.record_starts = containers->struct_starts.empty()
                    ? &containers->starts : &containers->record_starts;
            }
            strict_fields.headers_added = &assembly.headers_added;
            strict_fields.header_tokens = assembly.header_tokens;
        }
        if (config.struct_headers) {
            strict_fields.struct_starts = &struct_starts;
            strict_fields.struct_headers_added = &struct_headers_added;
        }
        if (with_markers) strict_fields.cut_markers = assembly.cut_markers;
        if (assembled) strict_fields.assembly_dropped = &assembly.dropped;
        if (g_last_drafter_profile.valid) {
            strict_fields.drafter_profile = &g_last_drafter_profile;
        }
        write_compression_trace(
            input_tokens, keep_ratio, trace_chunk, query_tokens,
            pool_kernel, n_keep_approx, chunk_means, selected_mask,
            mandatory_mask, output, &strict_fields);
    }
    return output;
}

} // namespace luce::common
