// PFlash fold selection. See pflash_fold.h.

#include "pflash_fold.h"

#include "server/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace luce::pflash {

using luce::common::PFlashTokenSpan;

namespace {

constexpr size_t npos = std::string::npos;

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}
bool is_ident_start(char c) {
    return std::isalpha((unsigned char) c) || c == '_';
}
bool is_ident(char c) {
    return std::isalnum((unsigned char) c) || c == '_';
}

struct Line {
    size_t begin;   // first byte
    size_t end;     // the newline, or the body end
    size_t next;    // the next line's first byte (newline included here)
    size_t first;   // first non-space byte, == end when blank
};

std::vector<Line> lines_of(const std::string & t, PFlashTextSpan body) {
    std::vector<Line> out;
    size_t pos = body.begin;
    while (pos < body.end) {
        size_t nl = t.find('\n', pos);
        const size_t end = (nl == npos || nl >= body.end) ? body.end : nl;
        const size_t next = end < body.end ? end + 1 : body.end;
        size_t first = pos;
        while (first < end && is_space(t[first])) ++first;
        out.push_back({pos, end, next, first});
        pos = next;
    }
    return out;
}

int column_of(const std::string & t, size_t begin, size_t first) {
    int col = 0;
    for (size_t i = begin; i < first; ++i) col = t[i] == '\t' ? (col / 8 + 1) * 8 : col + 1;
    return col;
}

std::vector<PFlashTextSpan> sorted_unique(std::vector<PFlashTextSpan> spans) {
    std::sort(spans.begin(), spans.end(), [](const auto & a, const auto & b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
    return spans;
}

// Merged coverage of possibly nested spans, ascending.
std::vector<PFlashTextSpan> coverage_of(const std::vector<PFlashTextSpan> & spans) {
    std::vector<PFlashTextSpan> out;
    for (const auto & s : sorted_unique(spans)) {
        if (!out.empty() && s.begin <= out.back().end) {
            out.back().end = std::max(out.back().end, s.end);
        } else {
            out.push_back(s);
        }
    }
    return out;
}

bool covered(const std::vector<PFlashTextSpan> & coverage, size_t pos) {
    auto it = std::upper_bound(coverage.begin(), coverage.end(), pos,
        [](size_t p, const PFlashTextSpan & s) { return p < s.begin; });
    return it != coverage.begin() && pos < std::prev(it)->end;
}

// ── Python-like blocks ─────────────────────────────────────────────────

// ``def name(`` / ``async def name(`` / ``class Name(`` or ``class Name:``.
bool python_header(const std::string & t, size_t i, size_t end) {
    const auto word = [&](const char * w) {
        const size_t n = std::strlen(w);
        if (i + n <= end && t.compare(i, n, w) == 0 &&
            (i + n == end || !is_ident(t[i + n]))) {
            i += n;
            return true;
        }
        return false;
    };
    const auto spaces = [&]() {
        const size_t from = i;
        while (i < end && is_space(t[i])) ++i;
        return i > from;
    };
    if (word("async") && !spaces()) return false;
    bool is_class = false;
    if (word("class")) {
        is_class = true;
    } else if (!word("def")) {
        return false;
    }
    if (!spaces() || i >= end || !is_ident_start(t[i])) return false;
    while (i < end && is_ident(t[i])) ++i;
    spaces();
    if (i >= end) return false;
    return is_class ? (t[i] == '(' || t[i] == ':') : (t[i] == '(' || t[i] == '[');
}

// Brackets and strings of one line, Python lexing: comments end the line,
// triple-quoted strings may span lines.
void scan_python_line(const std::string & t, size_t i, size_t end,
                      char & triple, int & depth) {
    while (i < end) {
        const char c = t[i];
        if (triple) {
            if (c == '\\') { i += 2; continue; }
            if (c == triple && i + 2 < end && t[i + 1] == triple && t[i + 2] == triple) {
                triple = 0;
                i += 3;
                continue;
            }
            ++i;
            continue;
        }
        if (c == '#') return;
        if (c == '"' || c == '\'') {
            if (i + 2 < end && t[i + 1] == c && t[i + 2] == c) {
                triple = c;
                i += 3;
                continue;
            }
            ++i;
            while (i < end && t[i] != c) i += t[i] == '\\' ? 2 : 1;
            ++i;
            continue;
        }
        if (c == '(' || c == '[' || c == '{') {
            ++depth;
        } else if ((c == ')' || c == ']' || c == '}') && depth > 0) {
            --depth;
        }
        ++i;
    }
}

// ── C-like blocks ──────────────────────────────────────────────────────

bool in_list(const std::string & w, std::initializer_list<const char *> list) {
    for (const char * x : list) if (w == x) return true;
    return false;
}

// Does the statement text before a '{' (comments dropped, strings as "")
// open a function or a type body?
bool brace_header(const std::string & s) {
    std::vector<std::string> words;   // identifiers at bracket depth 0
    int depth = 0;
    bool assign = false;
    size_t first_paren = npos;
    for (size_t i = 0; i < s.size();) {
        const char c = s[i];
        if (c == '(' || c == '[') {
            if (depth == 0 && c == '(' && first_paren == npos) first_paren = i;
            ++depth;
            ++i;
            continue;
        }
        if (c == ')' || c == ']') {
            if (depth > 0) --depth;
            ++i;
            continue;
        }
        if (depth == 0 && c == '=') {
            const char prev = i > 0 ? s[i - 1] : ' ';
            const char next = i + 1 < s.size() ? s[i + 1] : ' ';
            if (next != '=' && next != '>' && prev != '=' && prev != '!' &&
                prev != '<' && prev != '>') {
                assign = true;
            }
        }
        if (is_ident_start(c)) {
            const size_t from = i;
            while (i < s.size() && is_ident(s[i])) ++i;
            if (depth == 0) words.push_back(s.substr(from, i - from));
            continue;
        }
        ++i;
    }
    if (words.empty()) return false;
    const std::string & first = words.front();
    if (in_list(first, {"if", "else", "for", "while", "switch", "do", "try",
                        "catch", "finally", "match", "loop", "unsafe", "return",
                        "case", "default", "synchronized", "using", "lock",
                        "fixed", "checked", "unchecked", "foreach", "select",
                        "go", "defer", "with", "when", "until", "elif", "throw",
                        "yield", "await", "def", "lambda"})) {
        return false;
    }
    if (in_list(first, {"namespace", "extern", "module", "mod", "package"})) return false;
    size_t tail = s.find_last_not_of(' ');
    const bool arrow = tail != npos && tail >= 1 && s[tail - 1] == '=' && s[tail] == '>';
    bool fn_word = false;
    bool type_word = false;
    for (const auto & w : words) {
        fn_word = fn_word || in_list(w, {"function", "func", "fn", "fun"});
        type_word = type_word || in_list(w, {"class", "struct", "interface", "enum",
                                             "union", "trait", "impl", "record",
                                             "protocol", "extension"});
    }
    // Prose and markup (LaTeX, templates) are not declarations.
    if (words.size() > 16 || s.find_first_of("$\\") != npos) return false;
    if (assign) return arrow || fn_word;
    if (type_word || fn_word) return true;
    // ``name(...)`` plus qualifiers: a declaration-shaped prefix of at most
    // ten words, only type-like text after the parameter list.
    if (first_paren == npos) return false;
    size_t before = first_paren;
    while (before > 0 && s[before - 1] == ' ') --before;
    if (before == 0 || !(is_ident(s[before - 1]) || s[before - 1] == '>')) return false;
    int prefix_words = 0;
    for (size_t i = 0; i < first_paren; ++i) {
        const char c = s[i];
        if (is_ident_start(c) && (i == 0 || !is_ident(s[i - 1]))) ++prefix_words;
        if (!is_ident(c) && !std::strchr(" :<>,*&~[]@.", c)) return false;
    }
    if (prefix_words > 10) return false;
    size_t close = first_paren;
    for (int d = 0; close < s.size(); ++close) {
        if (s[close] == '(') ++d;
        if (s[close] == ')' && --d == 0) break;
    }
    for (size_t i = close + 1; i < s.size(); ++i) {
        if (!is_ident(s[i]) && !std::strchr(" :<>,*&[]()-.?|'", s[i])) return false;
    }
    return true;
}

// ── text blocks ────────────────────────────────────────────────────────

struct RecordHeader {
    size_t line;
    std::string key;
    long number;
};

// "Document 3:", "Passage 12.", "Chapter 4", "## Session 7 -", "[4]".
bool record_header(const std::string & t, const Line & line, RecordHeader & out) {
    size_t i = line.first;
    const size_t end = line.end;
    while (i < end && t[i] == '#') ++i;
    while (i < end && is_space(t[i])) ++i;
    if (i + 1 < end && t[i] == '*' && t[i + 1] == '*') i += 2;
    std::string key;
    if (i < end && t[i] == '[') {
        key = "[";
        ++i;
    } else {
        if (i >= end || !std::isupper((unsigned char) t[i])) return false;
        for (int w = 0; w < 2; ++w) {
            const size_t from = i;
            while (i < end && std::isalpha((unsigned char) t[i]) && i - from < 24) ++i;
            if (i - from < 2 || (i < end && std::isalpha((unsigned char) t[i]))) {
                if (w == 0) return false;
                i = from;
                break;
            }
            key.append(t, from, i - from);
            key += ' ';
            if (i < end && t[i] == ' ' && i + 1 < end &&
                std::isalpha((unsigned char) t[i + 1])) {
                ++i;
                continue;
            }
            break;
        }
        const size_t gap = i;
        while (i < end && (t[i] == ' ' || t[i] == '#')) ++i;
        if (i == gap) return false;          // "Document 3", not "ks2"
    }
    const size_t digits = i;
    long number = 0;
    while (i < end && std::isdigit((unsigned char) t[i]) && i - digits < 6) {
        number = number * 10 + (t[i] - '0');
        ++i;
    }
    if (i == digits || (i < end && std::isdigit((unsigned char) t[i]))) return false;
    while (i < end && is_space(t[i])) ++i;
    char delim = '\n';
    if (i < end) {
        delim = t[i];
        if (!std::strchr(":.)]-", delim)) return false;
        if (key == "[") {
            // "[4] Title", not a citation that wrapped to a line start.
            if (delim != ']') return false;
            size_t k = i + 1;
            while (k < end && is_space(t[k])) ++k;
            if (k < end && (k == i + 1 || !std::isupper((unsigned char) t[k]))) return false;
        }
    }
    out = {line.begin, key + delim, number};
    return true;
}

int heading_level(const std::string & t, const Line & line) {
    if (line.first != line.begin) return 0;
    size_t i = line.first;
    while (i < line.end && t[i] == '#') ++i;
    const int level = (int) (i - line.first);
    if (level < 1 || level > 6 || i + 1 >= line.end || t[i] != ' ' ||
        is_space(t[i + 1])) {
        return 0;
    }
    return level;
}

// Coverage lookups over sorted spans ignore the rest of the body: a line is
// in code when its first byte is.
bool line_in(const std::vector<PFlashTextSpan> & coverage, const Line & line) {
    return covered(coverage, line.begin);
}

bool starts_with(const std::string & t, size_t i, size_t end, const char * prefix) {
    const size_t n = std::strlen(prefix);
    return i + n <= end && t.compare(i, n, prefix) == 0;
}

// A line that reads as pasted material rather than the user's own prose.
bool paste_line(const std::string & t, const Line & line) {
    const size_t i = line.first;
    const size_t end = line.end;
    if (i == end) return false;
    if (line.first - line.begin >= 4 || (line.first > line.begin && t[line.begin] == '\t')) {
        return true;                                     // indented code or data
    }
    const char c = t[i];
    if (starts_with(t, i, end, "```") || starts_with(t, i, end, "~~~")) return true;
    if (c == '#') {
        size_t j = i;
        while (j < end && t[j] == '#') ++j;
        if (j - i <= 6 && j < end && t[j] == ' ') return true;       // heading
        if (starts_with(t, i, end, "#include") || starts_with(t, i, end, "#!") ||
            starts_with(t, i, end, "#define") || starts_with(t, i, end, "#[")) {
            return true;
        }
    }
    if (std::strchr("{[|", c)) return true;              // JSON, arrays, tables, "[user]"
    if (c == '<' && i + 1 < end &&
        (std::isalpha((unsigned char) t[i + 1]) || std::strchr("/!?", t[i + 1]))) {
        return true;                                     // markup
    }
    size_t k = i;
    while (k < end && std::strchr("-=*_~", t[k])) ++k;
    if (k - i >= 3 && k == end) return true;             // separator rule
    for (const char * prefix : {"File:", "Path:", "Source:", "Repository:", "Package:",
                                "import ", "package ", "func ", "fn ",
                                "//", "/*", "$ ", ">>> ", "@"}) {
        if (starts_with(t, i, end, prefix)) return true;
    }
    if (python_header(t, i, end)) return true;
    size_t last = end;
    while (last > i && is_space(t[last - 1])) --last;
    if (last > i && std::strchr("{};", t[last - 1])) return true;   // code-shaped
    RecordHeader header;
    return record_header(t, line, header);
}

} // namespace

PFlashQueryBlocks pflash_query_blocks(const std::string & t, PFlashTextSpan message) {
    PFlashQueryBlocks out;
    const auto lines = lines_of(t, message);
    if (lines.empty()) return out;
    const size_t total = message.end - message.begin;
    const auto trimmed = [&](size_t from, size_t to) -> PFlashTextSpan {
        // Lines [from, to) without leading and trailing blank lines.
        while (from < to && lines[from].first == lines[from].end) ++from;
        while (to > from && lines[to - 1].first == lines[to - 1].end) --to;
        if (from >= to) return {};
        size_t end = lines[to - 1].end;
        while (end > lines[to - 1].first && is_space(t[end - 1])) --end;
        return {lines[from].first, end};
    };
    size_t first_paste = lines.size();
    size_t last_paste = lines.size();
    for (size_t j = 0; j < lines.size(); ++j) {
        if (!paste_line(t, lines[j])) continue;
        if (first_paste == lines.size()) first_paste = j;
        last_paste = j;
    }
    out.structured = first_paste < lines.size();
    const auto small = [&](PFlashTextSpan s) { return (s.end - s.begin) * 2 <= total; };
    if (out.structured) {
        const PFlashTextSpan head = trimmed(0, first_paste);
        if (head.end > head.begin && small(head)) {
            out.head = head;
            out.head_rule = "paste";
        }
        size_t para = lines.size();
        while (para > 0 && lines[para - 1].first == lines[para - 1].end) --para;
        size_t start = para;
        while (start > 0 && lines[start - 1].first != lines[start - 1].end) --start;
        if (start > last_paste && start < para) {
            const PFlashTextSpan tail = trimmed(start, para);
            if (tail.end > tail.begin && small(tail)) out.tail = tail;
        }
        return out;
    }
    // No paste-like line: a short opening question before a long text.
    size_t para = 0;
    while (para < lines.size() && lines[para].first == lines[para].end) ++para;
    size_t stop = para;
    while (stop < lines.size() && lines[stop].first != lines[stop].end) ++stop;
    const PFlashTextSpan head = trimmed(para, stop);
    if (head.end > head.begin && stop < lines.size() &&
        (head.end - head.begin) * 9 <= total) {
        size_t last = head.end;
        while (last > head.begin && is_space(t[last - 1])) --last;
        if (last > head.begin && t[last - 1] == '?') {
            out.head = head;
            out.head_rule = "question";
        }
    }
    return out;
}

std::vector<PFlashTextSpan> pflash_message_bodies(const std::string & text) {
    static const char * kStart = "<|im_start|>";
    static const char * kEnd = "<|im_end|>";
    std::vector<PFlashTextSpan> bodies;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t s = text.find(kStart, pos);
        const size_t e = text.find(kEnd, pos);
        const size_t marker = std::min(s, e);
        const size_t stop = marker == npos ? text.size() : marker;
        if (stop > pos) bodies.push_back({pos, stop});
        if (marker == npos) break;
        if (marker == s) {
            // Skip the role line.
            const size_t nl = text.find('\n', s);
            pos = nl == npos ? text.size() : nl + 1;
        } else {
            pos = e + std::strlen(kEnd);
        }
    }
    return bodies;
}

std::vector<PFlashTextSpan> pflash_python_blocks(
        const std::string & t, PFlashTextSpan body) {
    struct Open { int indent; size_t start; };
    std::vector<PFlashTextSpan> out;
    std::vector<Open> open;
    size_t last_end = npos;   // end (newline included) of the last code line
    size_t decorator = npos;
    int decorator_indent = -1;
    char triple = 0;
    int depth = 0;
    bool after_blank = false;
    const auto close = [&](int indent) {
        while (!open.empty() && open.back().indent >= indent) {
            if (last_end != npos && last_end > open.back().start) {
                out.push_back({open.back().start, last_end});
            }
            open.pop_back();
        }
    };
    for (const Line & line : lines_of(t, body)) {
        if (line.first == line.end) {
            if (!triple) after_blank = true;
            continue;
        }
        // A bracket still open after a blank line, at a column-0 line, is
        // prose or broken code rather than a continuation.
        if (depth > 0 && !triple && after_blank && line.first == line.begin &&
            !std::strchr(")]}", t[line.first])) {
            depth = 0;
        }
        // So is one still open at a def/class line (a broken signature).
        if (depth > 0 && !triple && python_header(t, line.first, line.end)) depth = 0;
        after_blank = false;
        const bool continued = triple || depth > 0;
        if (!continued) {
            if (t[line.first] == '#') continue;   // comments neither open nor close
            const int indent = column_of(t, line.begin, line.first);
            close(indent);
            if (t[line.first] == '@') {
                if (decorator == npos) {
                    decorator = line.begin;
                    decorator_indent = indent;
                }
            } else {
                if (python_header(t, line.first, line.end)) {
                    const size_t start =
                        decorator != npos && decorator_indent == indent ? decorator : line.begin;
                    open.push_back({indent, start});
                }
                decorator = npos;
            }
        }
        scan_python_line(t, continued ? line.begin : line.first, line.end, triple, depth);
        last_end = line.next;
    }
    close(-1);
    return sorted_unique(std::move(out));
}

std::vector<PFlashTextSpan> pflash_brace_blocks(
        const std::string & t, PFlashTextSpan body) {
    struct Frame { bool block; size_t start; int paren; };
    std::vector<Frame> frames;
    std::vector<PFlashTextSpan> out;
    std::string stmt;           // significant text of the current statement
    size_t stmt_at = npos;
    bool stmt_long = false;
    int paren = 0;
    bool line_start = true;
    bool line_blank = true;
    bool new_line = false;      // a line ended at bracket depth 0
    const size_t e = body.end;
    const auto reset = [&]() {
        stmt.clear();
        stmt_at = npos;
        stmt_long = false;
    };
    const auto append = [&](char c) {
        if (stmt.size() < 1024) {
            stmt += c;
        } else {
            stmt_long = true;
        }
    };
    const auto line_begin = [&](size_t p) {
        while (p > body.begin && t[p - 1] != '\n') --p;
        return p;
    };
    const auto line_next = [&](size_t p) {
        const size_t nl = t.find('\n', p);
        return nl == npos || nl >= e ? e : nl + 1;
    };
    const auto skip_to_newline = [&](size_t p) {
        const size_t nl = t.find('\n', p);
        return nl == npos || nl >= e ? e : nl;
    };
    size_t i = body.begin;
    while (i < e) {
        const char c = t[i];
        if (c == '\n') {
            if (line_blank) {
                reset();
                paren = 0;
            }
            new_line = paren == 0;
            line_start = true;
            line_blank = true;
            ++i;
            continue;
        }
        if (is_space(c)) {
            if (!stmt.empty() && stmt.back() != ' ') append(' ');
            ++i;
            continue;
        }
        const bool first_on_line = line_start;
        line_start = false;
        line_blank = false;
        if (c == '#' && first_on_line) {          // preprocessor, attribute, shell comment
            i = skip_to_newline(i);
            continue;
        }
        if (c == '/' && i + 1 < e && t[i + 1] == '/') {
            i = skip_to_newline(i);
            continue;
        }
        if (c == '/' && i + 1 < e && t[i + 1] == '*') {
            const size_t z = t.find("*/", i + 2);
            i = z == npos || z + 2 > e ? e : z + 2;
            continue;
        }
        // A statement ends with its line unless a brace opens the next one
        // (Allman style) or a bracket is open: Python and prose have no
        // ';' to end them.
        if (new_line) {
            new_line = false;
            if (c != '{') reset();
        }
        if (stmt_at == npos) stmt_at = i;
        if (c == '"' || c == '`') {
            size_t j = i + 1;
            while (j < e && t[j] != c && !(c == '"' && t[j] == '\n')) j += t[j] == '\\' ? 2 : 1;
            i = j < e && t[j] == c ? j + 1 : std::min(j, e);
            append('"');
            append('"');
            continue;
        }
        if (c == '\'') {
            // A char literal ('x', '\n', '\x41', 'é'); otherwise a lifetime
            // or an apostrophe, kept as an ordinary character.
            size_t j = npos;
            if (i + 1 < e && t[i + 1] == '\\') {
                size_t k = i + 3;
                while (k < e && k < i + 12 && t[k] != '\'' && t[k] != '\n') ++k;
                if (k < e && t[k] == '\'') j = k;
            } else if (i + 2 < e && t[i + 2] == '\'' && t[i + 1] != '\n') {
                j = i + 2;
            } else if (i + 1 < e && (unsigned char) t[i + 1] >= 0x80) {
                size_t k = i + 2;
                while (k < e && k < i + 6 && (unsigned char) t[k] >= 0x80) ++k;
                if (k < e && t[k] == '\'') j = k;
            }
            if (j != npos) {
                i = j + 1;
                append('\'');
                append('\'');
                continue;
            }
        }
        if (c == '(' || c == '[') {
            ++paren;
        } else if ((c == ')' || c == ']') && paren > 0) {
            --paren;
        } else if (c == ';' && paren == 0) {
            reset();
            ++i;
            continue;
        } else if (c == '{') {
            const bool block = paren == 0 && !stmt_long && brace_header(stmt);
            frames.push_back({block, block ? line_begin(stmt_at) : 0, paren});
            paren = 0;
            reset();
            ++i;
            continue;
        } else if (c == '}') {
            if (!frames.empty()) {
                const Frame frame = frames.back();
                frames.pop_back();
                if (frame.block) out.push_back({frame.start, line_next(i)});
                paren = frame.paren;
            }
            reset();
            ++i;
            continue;
        }
        append(c);
        ++i;
    }
    return sorted_unique(std::move(out));
}

std::vector<PFlashTextSpan> pflash_text_blocks(
        const std::string & t, PFlashTextSpan body,
        const std::vector<PFlashTextSpan> & code, const char ** source,
        bool paragraphs) {
    const auto coverage = coverage_of(code);
    const auto lines = lines_of(t, body);
    std::vector<PFlashTextSpan> out;
    if (source) *source = "none";
    // A code body keeps its code blocks only: numbered lines in code are
    // comments and data, not records.
    int nonblank = 0;
    int in_code = 0;
    for (const Line & line : lines) {
        if (line.first == line.end) continue;
        ++nonblank;
        in_code += line_in(coverage, line) ? 1 : 0;
    }
    if (nonblank > 0 && (double) in_code >= kPFlashFoldCodeLineShare * (double) nonblank) {
        return out;
    }

    // Numbered records: the widest recurring header family.
    std::map<std::string, std::vector<RecordHeader>> families;
    for (const Line & line : lines) {
        if (line.first == line.end || line_in(coverage, line)) continue;
        RecordHeader header;
        if (record_header(t, line, header)) families[header.key].push_back(header);
    }
    const std::vector<RecordHeader> * best = nullptr;
    size_t best_width = 0;
    for (const auto & [key, headers] : families) {
        std::set<long> numbers;
        for (const auto & h : headers) numbers.insert(h.number);
        if (numbers.size() < 3) continue;
        const size_t width = headers.back().line - headers.front().line;
        if (!best || width > best_width ||
            (width == best_width && headers.size() > best->size())) {
            best = &headers;
            best_width = width;
        }
    }
    if (best) {
        for (size_t j = 0; j < best->size(); ++j) {
            const size_t end = j + 1 < best->size() ? (*best)[j + 1].line : body.end;
            out.push_back({(*best)[j].line, end});
        }
        if (source) *source = "records";
        return out;
    }
    // Markdown fences (``` or ~~~ at a line start) hold code examples in a
    // text; code anywhere else makes this a code body, which keeps only its
    // code blocks and records.
    std::vector<PFlashTextSpan> fences;
    size_t fence_open = npos;
    for (const Line & line : lines) {
        if (line.end - line.first >= 3 &&
            (t.compare(line.first, 3, "```") == 0 || t.compare(line.first, 3, "~~~") == 0)) {
            if (fence_open == npos) {
                fence_open = line.begin;
            } else {
                fences.push_back({fence_open, line.next});
                fence_open = npos;
            }
        }
    }
    if (fence_open != npos) fences.push_back({fence_open, body.end});
    for (const auto & block : code) {
        if (!covered(fences, block.begin)) return out;
    }

    // Markdown sections: a heading through the next heading of its level or
    // above.
    std::vector<std::pair<size_t, int>> headings;
    for (const Line & line : lines) {
        if (covered(fences, line.begin)) continue;
        const int level = heading_level(t, line);
        if (level > 0) headings.push_back({line.begin, level});
    }
    if (headings.size() >= 2) {
        for (size_t j = 0; j < headings.size(); ++j) {
            size_t end = body.end;
            for (size_t k = j + 1; k < headings.size(); ++k) {
                if (headings[k].second <= headings[j].second) {
                    end = headings[k].first;
                    break;
                }
            }
            out.push_back({headings[j].first, end});
        }
        if (source) *source = "headings";
        return out;
    }

    // Paragraphs: blank-line blocks, a long block split into its lines; a
    // fenced example is one block.
    if (!paragraphs) return out;
    const auto flush = [&](size_t j, size_t k) {
        if (j >= k) return;
        const PFlashTextSpan block{lines[j].begin, lines[k - 1].next};
        if (block.end - block.begin <= kPFlashFoldParagraphBytes) {
            out.push_back(block);
        } else {
            for (size_t m = j; m < k; ++m) out.push_back({lines[m].begin, lines[m].next});
        }
    };
    size_t run = 0;
    for (size_t j = 0; j <= lines.size(); ++j) {
        const bool fenced = j < lines.size() && covered(fences, lines[j].begin);
        if (j == lines.size() || lines[j].first == lines[j].end || fenced) {
            flush(run, j);
            run = j + 1;
        }
    }
    out.insert(out.end(), fences.begin(), fences.end());
    out = sorted_unique(std::move(out));
    if (source && !out.empty()) *source = "paragraphs";
    return out;
}

PFlashFoldStructure pflash_fold_structure(const std::string & text, bool paragraphs) {
    PFlashFoldStructure s;
    int rank = 0;   // records > headings > paragraphs > none, for the label
    for (const auto & body : pflash_message_bodies(text)) {
        auto code = pflash_python_blocks(text, body);
        const auto braces = pflash_brace_blocks(text, body);
        code.insert(code.end(), braces.begin(), braces.end());
        code = sorted_unique(std::move(code));
        const char * source = "none";
        const auto blocks = pflash_text_blocks(text, body, code, &source, paragraphs);
        const auto coverage = coverage_of(code);
        for (const Line & line : lines_of(text, body)) {
            if (line.first == line.end) continue;
            ++s.nonblank_lines;
            if (line_in(coverage, line)) ++s.code_lines;
        }
        s.code_blocks += code.size();
        s.blocks.insert(s.blocks.end(), code.begin(), code.end());
        s.blocks.insert(s.blocks.end(), blocks.begin(), blocks.end());
        const int r = std::strcmp(source, "records") == 0 ? 3
            : std::strcmp(source, "headings") == 0 ? 2
            : std::strcmp(source, "paragraphs") == 0 ? 1 : 0;
        if (r > rank) {
            rank = r;
            s.text_source = source;
        }
    }
    s.blocks = sorted_unique(std::move(s.blocks));
    s.code_like = s.nonblank_lines > 0 &&
        (double) s.code_lines >= kPFlashFoldCodeLineShare * (double) s.nonblank_lines;
    return s;
}

PFlashTokenSpan pflash_text_to_token_span(
        const std::vector<size_t> & token_begin, PFlashTextSpan span) {
    if (token_begin.empty() || span.end <= span.begin) return {0, 0};
    int first = (int) (std::upper_bound(token_begin.begin(), token_begin.end(), span.begin) -
                       token_begin.begin()) - 1;
    first = std::max(first, 0);
    const int end = (int) (std::lower_bound(token_begin.begin(), token_begin.end(), span.end) -
                           token_begin.begin());
    return {first, std::max(first, end)};
}

std::vector<PFlashTokenSpan> pflash_fold_token_blocks(
        const std::vector<PFlashTextSpan> & blocks,
        const std::vector<size_t> & token_begin,
        const std::vector<PFlashSelectionCandidate> & candidates) {
    std::vector<PFlashTokenSpan> mandatory;
    for (const auto & c : candidates) {
        if (c.mandatory) mandatory.push_back({c.begin, c.end});
    }
    std::sort(mandatory.begin(), mandatory.end(), [](const auto & a, const auto & b) {
        return a.begin < b.begin;
    });
    std::vector<PFlashTokenSpan> out;
    for (const auto & block : blocks) {
        PFlashTokenSpan span = pflash_text_to_token_span(token_begin, block);
        for (const auto & m : mandatory) {
            if (m.end <= span.begin) continue;
            if (m.begin <= span.begin) {
                span.begin = m.end;       // starts inside a kept span
                continue;
            }
            if (m.begin < span.end) span.end = m.begin;
            break;
        }
        if (span.end > span.begin) out.push_back(span);
    }
    return out;
}

PFlashFoldResult select_pflash_fold(
        const std::vector<PFlashSelectionCandidate> & candidates,
        const std::vector<PFlashTokenSpan> & blocks,
        const PFlashFoldPolicy & policy) {
    PFlashFoldResult result;
    if (policy.token_budget <= 0 || policy.k <= 0 || policy.cap <= 0 ||
        policy.head < 0 || policy.head > policy.cap) {
        result.error = "PFlash fold needs a positive budget, k and cap, and head <= cap";
        return result;
    }
    int limit = 0;
    std::vector<const PFlashSelectionCandidate *> ranges;
    for (const auto & c : candidates) {
        if (c.begin < 0 || c.end <= c.begin || !std::isfinite(c.score)) {
            result.error = "PFlash candidate range or score is invalid";
            return result;
        }
        ranges.push_back(&c);
        limit = std::max(limit, c.end);
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto * a, const auto * b) {
        return a->begin < b->begin;
    });
    std::set<size_t> ordinals;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if ((i > 0 && ranges[i - 1]->end > ranges[i]->begin) ||
            !ordinals.insert(ranges[i]->ordinal).second) {
            result.error = "PFlash candidates must be unique and non-overlapping";
            return result;
        }
    }
    for (const auto & b : blocks) limit = std::max(limit, b.end);

    std::vector<uint8_t> kept((size_t) limit, 0);
    int used = 0;
    for (const auto & c : candidates) {
        if (!c.mandatory) continue;
        used += c.end - c.begin;
        std::fill(kept.begin() + c.begin, kept.begin() + c.end, 1);
    }
    if (used > policy.token_budget) {
        result.stop = PFlashSelectionStop::MandatoryQueryExceedsBudget;
        result.error = "mandatory PFlash retention tokens exceed the token budget";
        return result;
    }

    // Anchors: by density, or by mass (density x length) on code.
    std::vector<const PFlashSelectionCandidate *> order;
    for (const auto & c : candidates) if (!c.mandatory) order.push_back(&c);
    const auto key = [&](const PFlashSelectionCandidate * c) {
        const double d = std::max(0.0, c->score);
        return policy.rank == PFlashFoldRank::Mass ? d * (double) (c->end - c->begin) : d;
    };
    std::sort(order.begin(), order.end(), [&](const auto * a, const auto * b) {
        const double ka = key(a), kb = key(b);
        if (ka != kb) return ka > kb;
        return a->ordinal < b->ordinal;
    });
    // Smallest enclosing block first.
    std::vector<PFlashTokenSpan> spans(blocks);
    std::sort(spans.begin(), spans.end(), [](const auto & a, const auto & b) {
        const int la = a.end - a.begin, lb = b.end - b.begin;
        return la != lb ? la < lb : a.begin < b.begin;
    });

    std::set<std::vector<std::pair<int, int>>> seen;
    int taken = 0;
    result.stop = PFlashSelectionStop::CandidatesExhausted;
    for (const auto * anchor : order) {
        if (taken >= policy.k) {
            result.stop = PFlashSelectionStop::TopKReached;
            break;
        }
        const int ub = anchor->begin, ue = anchor->end;
        std::vector<PFlashTokenSpan> encs;
        for (const int pos : {ub, (ub + ue) / 2, ue - 1}) {
            for (const auto & s : spans) {
                if (s.begin <= pos && pos < s.end) {
                    if (std::find(encs.begin(), encs.end(), s) == encs.end()) encs.push_back(s);
                    break;
                }
            }
        }
        std::vector<std::pair<int, int>> reg{{ub, ue}};
        bool over_cap = false;
        for (const auto & s : encs) {
            if (s.end - s.begin <= policy.cap) {
                reg.push_back({s.begin, s.end});
            } else {
                over_cap = true;
                if (policy.head > 0) reg.push_back({s.begin, s.begin + policy.head});
            }
        }
        std::sort(reg.begin(), reg.end());
        std::vector<std::pair<int, int>> merged;
        for (const auto & r : reg) {
            if (!merged.empty() && r.first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second, r.second);
            } else {
                merged.push_back(r);
            }
        }
        if (seen.count(merged)) continue;
        int add = 0;
        for (const auto & r : merged) {
            for (int p = r.first; p < r.second; ++p) add += kept[(size_t) p] ? 0 : 1;
        }
        if (add == 0) {
            seen.insert(merged);
            continue;
        }
        if (used + add > policy.token_budget) {
            ++result.skipped;
            continue;
        }
        seen.insert(merged);
        for (const auto & r : merged) std::fill(kept.begin() + r.first, kept.begin() + r.second, 1);
        used += add;
        ++taken;
        result.anchors.push_back(anchor->ordinal);
        result.capped += over_cap ? 1 : 0;
        result.contained += encs.empty() ? 0 : 1;
    }
    if (result.stop != PFlashSelectionStop::TopKReached && result.skipped > 0) {
        result.stop = PFlashSelectionStop::BudgetReached;
    }
    for (int p = 0; p < limit; ++p) {
        if (!kept[(size_t) p]) continue;
        if (!result.kept.empty() && result.kept.back().end == p) {
            ++result.kept.back().end;
        } else {
            result.kept.push_back({p, p + 1});
        }
    }
    result.retained_tokens = used;
    result.ok = true;
    return result;
}

PFlashFoldResult select_pflash_fold_for_ids(
        const luce::common::Tokenizer & vocab,
        const std::vector<int32_t> & ids,
        const std::vector<PFlashSelectionCandidate> & candidates,
        const PFlashSelectionConfig & config,
        int token_budget,
        PFlashFoldStructure * info,
        std::vector<PFlashTokenSpan> * token_blocks) {
    std::string text;
    std::vector<size_t> token_begin;
    token_begin.reserve(ids.size());
    for (const int32_t id : ids) {
        token_begin.push_back(text.size());
        text += vocab.token_text(id);
    }
    PFlashFoldStructure structure = pflash_fold_structure(text, config.fold_paragraphs);
    const auto blocks = pflash_fold_token_blocks(structure.blocks, token_begin, candidates);
    PFlashFoldPolicy policy;
    policy.token_budget = token_budget;
    policy.k = config.fold_k;
    policy.cap = config.fold_cap;
    policy.head = config.fold_head;
    policy.rank = structure.code_like ? PFlashFoldRank::Mass : PFlashFoldRank::Density;
    auto result = select_pflash_fold(candidates, blocks, policy);
    if (info) *info = std::move(structure);
    if (token_blocks) *token_blocks = blocks;
    return result;
}

} // namespace luce::pflash
