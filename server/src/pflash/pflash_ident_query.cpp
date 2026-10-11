#include "pflash_ident_query.h"

#include <algorithm>
#include <cstring>

namespace luce::pflash {

namespace {

bool is_word(unsigned char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}
bool is_alpha_(unsigned char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
bool is_letter(unsigned char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool is_lower(unsigned char c) noexcept { return c >= 'a' && c <= 'z'; }
bool is_upper(unsigned char c) noexcept { return c >= 'A' && c <= 'Z'; }
bool is_path_char(unsigned char c) noexcept { return is_word(c) || c == '.' || c == '/' || c == '-'; }
bool is_pathseg_char(unsigned char c) noexcept { return is_word(c) || c == '.' || c == '-'; }   // [\w.-]
bool is_alnum(unsigned char c) noexcept { return is_letter(c) || (c >= '0' && c <= '9'); }

// The alternatives of the identifier pattern at ``p``, in their order; each
// returns the match end or npos. The backtracking of each regular expression
// is resolved in closed form (see the header for the patterns).
struct Scanner {
    const std::string & s;
    size_t n;
    std::vector<size_t> word_end;   // end of the word run starting at i (i when s[i] is not a word char)
    std::vector<size_t> path_end;   // same for [\w./-]
    std::vector<size_t> letter_end; // same for [A-Za-z]
    std::vector<size_t> lower_end;  // same for [a-z]
    std::vector<size_t> seg_end;    // same for [\w.-]

    explicit Scanner(const std::string & text) : s(text), n(text.size()) {
        word_end.assign(n + 1, n);
        path_end.assign(n + 1, n);
        letter_end.assign(n + 1, n);
        lower_end.assign(n + 1, n);
        seg_end.assign(n + 1, n);
        for (size_t i = n; i-- > 0;) {
            const unsigned char c = (unsigned char) s[i];
            word_end[i] = is_word(c) ? word_end[i + 1] : i;
            path_end[i] = is_path_char(c) ? path_end[i + 1] : i;
            letter_end[i] = is_letter(c) ? letter_end[i + 1] : i;
            lower_end[i] = is_lower(c) ? lower_end[i + 1] : i;
            seg_end[i] = is_pathseg_char(c) ? seg_end[i + 1] : i;
        }
    }
    unsigned char at(size_t i) const noexcept { return i < n ? (unsigned char) s[i] : 0; }
    bool word_at(size_t i) const noexcept { return i < n && is_word((unsigned char) s[i]); }
    bool boundary_before(size_t p) const noexcept {   // \b at p with a word char at p
        return p == 0 || !word_at(p - 1);
    }

    // [\w./-]*[A-Za-z_]\w*\.[A-Za-z]{1,6}\b
    size_t path(size_t p) const {
        const size_t run = path_end[p] - p;
        for (size_t a = run; a-- > 0;) {
            const size_t q = p + a;
            if (!is_alpha_(at(q))) continue;
            const size_t r = word_end[q];
            if (at(r) != '.') continue;
            const size_t letters = letter_end[r + 1 <= n ? r + 1 : n] - (r + 1);
            if (letters >= 1 && letters <= 6 && !word_at(r + 1 + letters)) return r + 1 + letters;
        }
        return std::string::npos;
    }
    // \b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b
    size_t dotted(size_t p) const {
        if (!is_alpha_(at(p)) || !boundary_before(p)) return std::string::npos;
        size_t end = word_end[p];
        int parts = 0;
        while (at(end) == '.' && is_alpha_(at(end + 1))) {
            end = word_end[end + 1];
            ++parts;
        }
        return parts > 0 ? end : std::string::npos;
    }
    // \b[a-z]\w*_\w+\b
    size_t snake(size_t p) const {
        if (!is_lower(at(p)) || !boundary_before(p)) return std::string::npos;
        const size_t e = word_end[p];
        for (size_t x = p + 1; x + 1 < e; ++x) {
            if (s[x] == '_') return e;
        }
        return std::string::npos;
    }
    // \b[a-z]+[A-Z]\w*\b
    size_t camel(size_t p) const {
        if (!is_lower(at(p)) || !boundary_before(p)) return std::string::npos;
        const size_t l = lower_end[p];
        return is_upper(at(l)) ? word_end[l] : std::string::npos;
    }
    // \b[A-Z][a-z]+[A-Z]\w*\b
    size_t capwords(size_t p) const {
        if (!is_upper(at(p)) || !boundary_before(p) || !is_lower(at(p + 1))) return std::string::npos;
        const size_t l = lower_end[p + 1];
        return is_upper(at(l)) ? word_end[l] : std::string::npos;
    }
    // ident_v2 PATH: (?:\.{1,2}/|/)?[\w.-]+(?:/[\w.-]+)+/?  |  /[\w.-]+/?
    // [\w.-]+ stops only before a non-[\w.-] byte, so each run is taken
    // whole; the repeated group takes every "/run" that follows; the
    // optional prefix is tried as "../" / "./", then "/", then empty.
    size_t path_after(size_t q) const {      // [\w.-]+(?:/[\w.-]+)+/? at q
        if (q >= n || !is_pathseg_char(at(q))) return std::string::npos;
        size_t e = seg_end[q];
        int parts = 0;
        while (at(e) == '/' && is_pathseg_char(at(e + 1))) {
            e = seg_end[e + 1];
            ++parts;
        }
        if (parts == 0) return std::string::npos;
        return at(e) == '/' ? e + 1 : e;
    }
    size_t path_v2(size_t p) const {
        std::vector<size_t> prefixes;
        if (at(p) == '.' && at(p + 1) == '.' && at(p + 2) == '/') prefixes.push_back(3);
        else if (at(p) == '.' && at(p + 1) == '/') prefixes.push_back(2);
        if (at(p) == '/') prefixes.push_back(1);
        prefixes.push_back(0);
        for (const size_t k : prefixes) {
            const size_t e = path_after(p + k);
            if (e != std::string::npos) return e;
        }
        if (at(p) == '/' && is_pathseg_char(at(p + 1))) {
            const size_t e = seg_end[p + 1];
            return at(e) == '/' ? e + 1 : e;
        }
        return std::string::npos;
    }
    // ident_v2 UNDERSCORE: (?<![\w.])_{1,2}[A-Za-z0-9]\w*
    size_t underscore_v2(size_t p) const {
        if (at(p) != '_' || (p > 0 && (is_word(at(p - 1)) || at(p - 1) == '.'))) return std::string::npos;
        if (at(p + 1) == '_') return is_alnum(at(p + 2)) ? word_end[p + 2] : std::string::npos;
        return is_alnum(at(p + 1)) ? word_end[p + 1] : std::string::npos;
    }
    // ident_v2 DOTTED: \.?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\.?
    size_t dotted_v2(size_t p) const {
        const size_t q = at(p) == '.' ? p + 1 : p;
        if (!is_alpha_(at(q))) return std::string::npos;
        size_t e = word_end[q];
        int parts = 0;
        while (at(e) == '.' && is_alpha_(at(e + 1))) {
            e = word_end[e + 1];
            ++parts;
        }
        if (parts == 0) return std::string::npos;
        return at(e) == '.' ? e + 1 : e;
    }
    // \b[A-Za-z_]\w{3,}(?=\()
    size_t call(size_t p) const {
        if (!is_alpha_(at(p)) || !boundary_before(p)) return std::string::npos;
        const size_t e = word_end[p];
        return e - p >= 4 && at(e) == '(' ? e : std::string::npos;
    }
};

bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

} // namespace

const char * pflash_ident_query_name(PFlashIdentQuery mode) noexcept {
    switch (mode) {
        case PFlashIdentQuery::Heur: return "heur";
        case PFlashIdentQuery::Off: break;
    }
    return "off";
}

std::vector<std::pair<size_t, size_t>> pflash_identifier_char_spans(
        const std::string & text, PFlashIdentVersion version) {
    std::vector<std::pair<size_t, size_t>> out;
    const Scanner sc(text);
    const size_t min_chars = (size_t) kIdentMinChars;
    // re.finditer of one pattern: leftmost match, then continue at its end.
    const auto finditer = [&](auto match, auto && emit) {
        size_t p = 0;
        while (p < sc.n) {
            const size_t e = match(p);
            if (e == std::string::npos || e <= p) {
                ++p;
                continue;
            }
            emit(p, e);
            p = e;
        }
    };
    const auto keep = [&](size_t b, size_t e) {
        if (e - b >= min_chars) out.emplace_back(b, e);
    };
    finditer([&](size_t p) {
        size_t e = sc.path(p);
        if (e == std::string::npos) e = sc.dotted(p);
        if (e == std::string::npos) e = sc.snake(p);
        if (e == std::string::npos) e = sc.camel(p);
        if (e == std::string::npos) e = sc.capwords(p);
        if (e == std::string::npos) e = sc.call(p);
        return e;
    }, keep);
    if (version == PFlashIdentVersion::V2) {
        finditer([&](size_t p) { return sc.path_v2(p); }, keep);
        finditer([&](size_t p) { return sc.underscore_v2(p); }, keep);
        finditer([&](size_t p) { return sc.dotted_v2(p); }, [&](size_t b, size_t e) {
            keep(b, e);
            // every [A-Za-z_]\w* segment of the match
            size_t k = b;
            while (k < e) {
                if (is_alpha_((unsigned char) text[k])) {
                    const size_t z = std::min(sc.word_end[k], e);
                    keep(k, z);
                    k = z;
                } else {
                    ++k;
                }
            }
        });
    }
    // ^.*(?:Error|Exception|Traceback).*$ (multi-line), >= 8 characters stripped.
    size_t begin = 0;
    while (begin <= text.size()) {
        size_t end = text.find('\n', begin);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(begin, end - begin);
        if (line.find("Error") != std::string::npos || line.find("Exception") != std::string::npos ||
            line.find("Traceback") != std::string::npos) {
            size_t a = 0;
            size_t z = line.size();
            while (a < z && is_space(line[a])) ++a;
            while (z > a && is_space(line[z - 1])) --z;
            if (z - a >= (size_t) kIdentErrorLineMinChars) out.emplace_back(begin, end);
        }
        if (end == text.size()) break;
        begin = end + 1;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<int> pflash_ident_query_tokens(
        const std::vector<std::string> & token_texts,
        const std::vector<unsigned char> & markup,
        const std::vector<PFlashTokenSpan> & turns) {
    std::vector<int> tokens;
    for (const auto & turn : turns) {
        if (turn.begin < 0 || turn.end <= turn.begin || turn.end > (int) token_texts.size()) continue;
        std::string text;
        std::vector<size_t> starts;
        for (int t = turn.begin; t < turn.end; ++t) {
            starts.push_back(text.size());
            const std::string & piece = token_texts[(size_t) t];
            if (t < (int) markup.size() && markup[(size_t) t]) text.append(piece.size(), ' ');
            else text += piece;
        }
        for (const auto & [c0, c1] : pflash_identifier_char_spans(text)) {
            // Every token overlapping [c0, c1): the last start <= c0 up to the first start >= c1.
            const size_t first = (size_t) (std::upper_bound(starts.begin(), starts.end(), c0) - starts.begin()) - 1;
            const size_t last = (size_t) (std::lower_bound(starts.begin(), starts.end(), c1) - starts.begin());
            for (size_t k = first; k < last; ++k) tokens.push_back(turn.begin + (int) k);
        }
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

} // namespace luce::pflash
