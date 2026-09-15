#include "server/evidence_manager.h"

#include <algorithm>
#include <cstring>

namespace dflash::common {

std::vector<int> recover_selected_chunks(
        const std::vector<int32_t> & input_ids,
        const std::vector<int32_t> & compressed_ids,
        int chunk_size,
        int source_end) {
    std::vector<int> kept;
    if (chunk_size <= 0 || input_ids.empty() || compressed_ids.empty() ||
        source_end <= 0) {
        return kept;
    }

    const int input_tokens = (int) input_ids.size();
    const int compressed_tokens = (int) compressed_ids.size();
    const int n_chunks = (input_tokens + chunk_size - 1) / chunk_size;

    int cursor = 0;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int begin = chunk * chunk_size;
        if (begin >= source_end) break;
        const int end = std::min(input_tokens, begin + chunk_size);
        const int len = end - begin;
        // A later (shorter, trailing) chunk can still fit what remains of
        // the output, so only a full match advances the cursor.
        if (cursor + len > compressed_tokens) continue;
        if (std::memcmp(compressed_ids.data() + cursor,
                        input_ids.data() + begin,
                        (size_t) len * sizeof(int32_t)) == 0) {
            kept.push_back(chunk);
            cursor += len;
        }
    }
    return kept;
}

std::vector<EvidenceSpan> spans_for_chunks(
        const std::vector<int> & chunks, int chunk_size, int source_end) {
    std::vector<EvidenceSpan> spans;
    if (chunk_size <= 0 || source_end <= 0) return spans;

    std::vector<int> sorted = chunks;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    for (int chunk : sorted) {
        if (chunk < 0) continue;
        const int begin = chunk * chunk_size;
        if (begin >= source_end) break;
        const int end = std::min(source_end, begin + chunk_size);
        if (!spans.empty() && spans.back().end == begin) {
            spans.back().end = end;
        } else {
            spans.push_back(EvidenceSpan{chunk, begin, end});
        }
    }
    return spans;
}

static int spans_token_total(const std::vector<EvidenceSpan> & spans) {
    int total = 0;
    for (const auto & span : spans) total += span.tokens();
    return total;
}

EvidencePlan plan_evidence_turn(
        const EvidenceLedger & ledger,
        const PrefixHash & source_revision,
        const std::vector<int> & fresh_chunks,
        int source_tokens,
        int dialogue_tokens,
        int question_turns,
        const EvidenceConfig & config) {
    EvidencePlan plan;
    plan.question_turns = question_turns;

    plan.fresh = fresh_chunks;
    std::sort(plan.fresh.begin(), plan.fresh.end());
    plan.fresh.erase(std::unique(plan.fresh.begin(), plan.fresh.end()),
                     plan.fresh.end());

    // The ledger matches this request only when it served the immediately
    // preceding turn of the same source revision. A stale ledger means the
    // committed prefix is gone (restart) or the request stream diverged
    // (truncated/branched history, edited source); plan from the request.
    const bool stale =
        ledger.turn_index == 0 ||
        ledger.source_revision != source_revision ||
        question_turns != ledger.turn_index + 1;
    plan.ledger_reset = stale;

    const int budget = config.context_budget;

    // After a reset nothing from the old ledger is in the served prompt.
    const int resident_tokens = stale ? 0 : spans_token_total(spans_for_chunks(
        std::vector<int>(ledger.resident.begin(), ledger.resident.end()),
        config.chunk_size, source_tokens));

    // Anchor-every-turn is the placement experiment's source-first arm: a
    // consolidated rebuild each turn. Under append placement a stale ledger
    // still appends — the fresh selection lands in the current question.
    bool rebuild = config.anchor_every_turn;
    std::vector<int> emit_chunks;
    if (!rebuild) {
        if (stale) {
            emit_chunks = plan.fresh;
        } else {
            for (int chunk : plan.fresh) {
                if (ledger.resident.count(chunk) == 0) {
                    emit_chunks.push_back(chunk);
                }
            }
        }
        const int emit_tokens = spans_token_total(
            spans_for_chunks(emit_chunks, config.chunk_size, source_tokens));
        plan.projected_total =
            resident_tokens + emit_tokens + dialogue_tokens;
        rebuild = !stale && budget > 0 && plan.projected_total > budget;
    }

    if (rebuild) {
        plan.mode = EvidenceMode::Rebuild;
        plan.placement = EvidencePlacement::Anchor;

        // retained = fresh selection plus the last-K turn selections.
        std::unordered_set<int> retained(plan.fresh.begin(), plan.fresh.end());
        int kept_windows = 0;
        for (auto it = ledger.recent.rbegin();
             it != ledger.recent.rend() && kept_windows < config.rebuild_window;
             ++it, ++kept_windows) {
            retained.insert(it->begin(), it->end());
        }
        emit_chunks.assign(retained.begin(), retained.end());
        std::sort(emit_chunks.begin(), emit_chunks.end());

        plan.emit = spans_for_chunks(
            emit_chunks, config.chunk_size, source_tokens);
        plan.emit_tokens = spans_token_total(plan.emit);
        plan.projected_total = plan.emit_tokens + dialogue_tokens;

        // High-water fallback: if even the consolidated rebuild overflows,
        // shed the history selections and carry only the fresh selection —
        // the current question must fit.
        if (budget > 0 && plan.projected_total > budget &&
            (int) retained.size() > (int) plan.fresh.size()) {
            emit_chunks = plan.fresh;
            plan.emit = spans_for_chunks(
                emit_chunks, config.chunk_size, source_tokens);
            plan.emit_tokens = spans_token_total(plan.emit);
            plan.projected_total = plan.emit_tokens + dialogue_tokens;
            plan.capacity_shed = true;
        }
        plan.emit_chunks = std::move(emit_chunks);
        plan.resident_tokens = plan.emit_tokens;
    } else {
        plan.mode = EvidenceMode::Append;
        plan.placement = EvidencePlacement::CurrentQuestion;
        plan.emit_chunks = std::move(emit_chunks);
        plan.emit = spans_for_chunks(
            plan.emit_chunks, config.chunk_size, source_tokens);
        plan.emit_tokens = spans_token_total(plan.emit);
        plan.resident_tokens = resident_tokens + plan.emit_tokens;
        if (plan.projected_total == 0) {
            plan.projected_total = plan.resident_tokens + dialogue_tokens;
        }
    }
    return plan;
}

void commit_evidence_turn(
        EvidenceLedger & ledger,
        const EvidencePlan & plan,
        const std::vector<std::string> & question_texts,
        const std::string & block_text,
        const PrefixHash & source_revision,
        int source_tokens,
        int rebuild_window) {
    ledger.source_revision = source_revision;
    ledger.source_tokens = source_tokens;

    if (plan.mode == EvidenceMode::Rebuild) {
        ledger.resident.clear();
        for (int chunk : plan.emit_chunks) ledger.resident.insert(chunk);
        ledger.anchor_block = block_text;
        ledger.anchored = true;
        // Under the anchor layout every question re-renders bare.
        ledger.committed_users = question_texts;
    } else {
        if (plan.ledger_reset) {
            // Prior questions could not have carried committed evidence;
            // they re-render as the client sent them.
            ledger.committed_users.assign(
                question_texts.begin(),
                question_texts.empty() ? question_texts.begin()
                                       : question_texts.end() - 1);
            ledger.anchor_block.clear();
            ledger.anchored = false;
            ledger.resident.clear();
        }
        for (int chunk : plan.emit_chunks) ledger.resident.insert(chunk);
        const std::string & current =
            question_texts.empty() ? std::string() : question_texts.back();
        ledger.committed_users.push_back(
            compose_evidence_message(block_text, current));
    }

    ledger.recent.push_back(plan.fresh);
    while ((int) ledger.recent.size() > rebuild_window + 1) {
        ledger.recent.pop_front();
    }

    ledger.turn_index = plan.question_turns;
}

std::string render_evidence_block(
        const std::vector<EvidenceSpan> & spans,
        const std::function<std::string(int, int)> & decode_source) {
    std::string block =
        "The following source segments were selected as evidence for this "
        "session:\n\n";
    for (const auto & span : spans) {
        block += "=== segment " + std::to_string(span.segment) +
                 " (source tokens " + std::to_string(span.begin) + "-" +
                 std::to_string(span.end) + ") ===\n";
        block += decode_source(span.begin, span.end);
        block += "\n\n";
    }
    return block;
}

std::string compose_evidence_message(
        const std::string & block_text, const std::string & question_text) {
    if (block_text.empty()) return question_text;
    if (question_text.empty()) return block_text;
    return block_text + "\n\n" + question_text;
}

}  // namespace dflash::common
