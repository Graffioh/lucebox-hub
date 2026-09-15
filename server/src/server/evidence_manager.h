#pragma once

// Multi-turn PFlash evidence manager — pure planning layer.
//
// A managed session (request field `pflash_evidence`, requires `session_id`)
// keeps one immutable long source under a drafter-side evidence ledger. Each
// turn rescores the source against the current question, diffs the fresh
// chunk selection against what the served prompt already carries, and either
// appends the missing evidence inside the current user message (cheap: the
// inline prefix cache then diff-prefills only the new turn) or rebuilds the
// prompt with all retained evidence consolidated into the first user message.
//
// The module is deliberately free of HTTP, tokenizer, and backend types so
// the ledger, chunk recovery, and placement decisions are unit-testable
// without model files or a GPU.

#include "server/prefix_cache.h"  // PrefixHash, hash_prefix

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace dflash::common {

struct EvidenceConfig {
    // Selector chunk grid. Must match the grid the scorer used so recovered
    // selections map back onto the same spans.
    int  chunk_size = 256;
    // Rebuilds carry the fresh selection plus the selections of the last K
    // managed turns, so evidence that was recently needed stays resident.
    int  rebuild_window = 3;
    // Drafter-token budget for evidence + dialogue. <= 0 disables the
    // capacity check (rebuilds then only trigger on revision/desync or the
    // anchor placement mode).
    int  context_budget = 0;
    // Placement experiment arm: when true every turn consolidates retained
    // evidence into the first user message (source-first single block).
    // When false, new evidence appends inside the current user message and
    // rebuilds happen only on capacity/revision/desync.
    bool anchor_every_turn = false;
};

// One retained evidence span inside the session source.
struct EvidenceSpan {
    int segment = -1;  // first chunk index of the (possibly merged) span
    int begin = 0;     // source-token range [begin, end)
    int end = 0;
    int tokens() const { return end - begin; }
};

enum class EvidenceMode {
    // Emit only the missing chunks; they are rendered inside the current
    // user message before the question text.
    Append,
    // Re-render the prompt: all retained evidence consolidated in the first
    // user message, bare question/answer history re-rendered after it.
    Rebuild,
};

enum class EvidencePlacement {
    CurrentQuestion,  // emit block inside the current user message
    Anchor,           // emit block as the first user message (source slot)
};

struct EvidencePlan {
    EvidenceMode      mode = EvidenceMode::Append;
    EvidencePlacement placement = EvidencePlacement::CurrentQuestion;
    // The ledger's committed state does not match this request (fresh
    // session, source revision change, or user-turn desync); commit must
    // rebuild it from the request rather than from stored state.
    bool ledger_reset = false;
    // This turn's fresh chunk selection (sorted unique chunk ids).
    std::vector<int> fresh;
    // Chunk ids whose source ranges must be emitted this turn.
    std::vector<int> emit_chunks;
    // Merged source ranges for `emit_chunks` (display unit for the block).
    std::vector<EvidenceSpan> emit;
    // Non-source user messages in the request (prior questions + current).
    int question_turns = 0;
    int resident_tokens = 0;   // resident evidence tokens after this plan
    int emit_tokens = 0;       // source tokens emitted this turn
    int projected_total = 0;   // resident + dialogue + reserve estimate
    // Rebuild wanted more than the budget; history selections were shed.
    bool capacity_shed = false;
};

struct EvidenceLedger {
    PrefixHash source_revision{};
    int        source_tokens = 0;
    // Managed turns served (== committed_users.size()).
    int        turn_index = 0;
    // Chunk ids present in the currently-served prompt.
    std::unordered_set<int> resident;
    // Fresh selections of the last-K managed turns (oldest first).
    std::deque<std::vector<int>> recent;
    // Committed content of each non-source user message, in order. Serving
    // these verbatim keeps the rendered prefix byte-stable across turns so
    // the inline prefix cache diff-prefills only the new turn.
    std::vector<std::string> committed_users;
    // Rendered anchor content for the first user message when the committed
    // layout carries consolidated evidence there (rebuild layout).
    std::string anchor_block;
    bool        anchored = false;
};

// Recover which fixed-grid chunks of `input_ids` survive in
// `compressed_ids`. The LongAttnComp selector emits kept chunks whole and in
// ascending order, so a monotone chunk-anchored scan is exact for grid
// output. Chunks are eligible when they intersect [0, source_end); a
// straddling chunk (covering the source tail and the appended query) counts
// when its whole input span was emitted — the emitted source range is
// clipped to source_end downstream. Output that is not chunk-aligned
// (variable-segment selectors) under-recovers rather than misattributing.
std::vector<int> recover_selected_chunks(
    const std::vector<int32_t> & input_ids,
    const std::vector<int32_t> & compressed_ids,
    int chunk_size,
    int source_end);

// Merge chunk ids into maximal contiguous source ranges clipped to
// [0, source_end). Chunks are expected sorted; unsorted input is tolerated.
std::vector<EvidenceSpan> spans_for_chunks(
    const std::vector<int> & chunks, int chunk_size, int source_end);

// Decide append vs rebuild for this turn. Pure: does not mutate `ledger`.
// `dialogue_tokens` is the caller's drafter-token estimate of all
// non-evidence prompt content (system, questions, answers) plus the
// generation reserve; `question_turns` is the count of non-source user
// messages in the request.
EvidencePlan plan_evidence_turn(
    const EvidenceLedger & ledger,
    const PrefixHash & source_revision,
    const std::vector<int> & fresh_chunks,
    int source_tokens,
    int dialogue_tokens,
    int question_turns,
    const EvidenceConfig & config);

// Apply `plan` after the managed prompt rendered and tokenized
// successfully. `question_texts` are the client-sent contents of the
// non-source user messages in order (size == plan.question_turns);
// `block_text` is the rendered evidence block for plan.emit.
// `rebuild_window` bounds how many past selections the ledger keeps.
void commit_evidence_turn(
    EvidenceLedger & ledger,
    const EvidencePlan & plan,
    const std::vector<std::string> & question_texts,
    const std::string & block_text,
    const PrefixHash & source_revision,
    int source_tokens,
    int rebuild_window);

// Render the evidence block shared by append and anchor placement. Spans
// keep source order; each carries a segment-id + source-token header so
// placement variants stay like-for-like.
std::string render_evidence_block(
    const std::vector<EvidenceSpan> & spans,
    const std::function<std::string(int, int)> & decode_source);

// Compose a committed user content for append placement.
std::string compose_evidence_message(
    const std::string & block_text, const std::string & question_text);

}  // namespace dflash::common
