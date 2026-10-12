#include "server/pflash_unified.h"

#include "common/sha1.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace pflash_unified {

namespace {

std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

bool json_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}


uint64_t fnv1a(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h == 0 ? 1 : h;
}

}  // namespace

// ── Tool list ──────────────────────────────────────────────────────────

bool ToolList::listed(const std::string & name) const {
    if (all) return true;
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool parse_tool_list(const std::string & spec, ToolList & out, std::string & error) {
    out = ToolList{};
    const std::string s = trim(spec);
    if (s.empty()) return true;
    std::vector<std::string> names;
    size_t at = 0;
    while (true) {
        const size_t comma = s.find(',', at);
        const std::string entry = trim(s.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
        if (entry.empty()) {
            error = "PFLASH_REGION_TOOLS: empty entry in '" + s + "'";
            return false;
        }
        for (unsigned char c : entry) {
            if (c < 0x21 || c == 0x7F || c == '"' || c == '[' || c == ']') {
                error = "PFLASH_REGION_TOOLS: invalid character in tool name '" + entry + "'";
                return false;
            }
        }
        names.push_back(entry);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    const bool star = std::find(names.begin(), names.end(), "*") != names.end();
    if (star && names.size() > 1) {
        error = "PFLASH_REGION_TOOLS: '*' must be the only entry";
        return false;
    }
    if (star) {
        out.all = true;
        out.identity = "*";
        return true;
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    out.names = names;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) out.identity += ',';
        out.identity += names[i];
    }
    return true;
}

std::string call_brief(const std::string & name, const nlohmann::json & arguments, size_t max_args) {
    std::string brief;
    for (unsigned char c : name) brief += (c < 0x20 || c == 0x7F) ? ' ' : (char) c;
    nlohmann::json args = arguments;
    if (args.is_string()) {
        const std::string raw = args.get<std::string>();
        nlohmann::json parsed = nlohmann::json::parse(raw, nullptr, false);
        if (!parsed.is_discarded()) args = std::move(parsed);
    }
    if (args.is_null()) return brief;
    if (args.is_object() && args.empty()) return brief;
    std::string text = args.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace);
    if (text.size() > max_args) text = text.substr(0, max_args) + "\xE2\x80\xA6";
    return brief + " " + text;
}

// ── Line structure ─────────────────────────────────────────────────────

Lines split_lines(const std::string & text, size_t begin, size_t end, bool escaped) {
    Lines out;
    out.escaped = escaped;
    out.begin = begin;
    out.end = std::min(end, text.size());
    if (out.begin >= out.end) return out;
    out.starts.push_back(out.begin);
    size_t i = out.begin;
    while (i < out.end) {
        size_t sep = std::string::npos;
        size_t next = i + 1;
        if (!escaped) {
            if (text[i] == '\n') sep = i;
        } else if (text[i] == '\\' && i + 1 < out.end) {
            if (text[i + 1] == 'n') sep = i;
            next = text[i + 1] == 'u' ? std::min(out.end, i + 6) : i + 2;
        }
        if (sep != std::string::npos) {
            next = sep + out.sep_size();
            out.ends.push_back(sep);
            if (next < out.end) out.starts.push_back(next);
        }
        i = next;
    }
    if (out.ends.size() < out.starts.size()) out.ends.push_back(out.end);
    return out;
}

std::vector<std::pair<size_t, size_t>> json_string_values(const std::string & text, size_t begin, size_t end) {
    std::vector<std::pair<size_t, size_t>> out;
    end = std::min(end, text.size());
    if (begin >= end) return out;
    if (!nlohmann::json::accept(text.begin() + (std::ptrdiff_t) begin, text.begin() + (std::ptrdiff_t) end)) {
        return out;
    }
    size_t i = begin;
    while (i < end) {
        if (text[i] != '"') {
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < end && text[j] != '"') j += text[j] == '\\' ? 2 : 1;
        if (j >= end) break;
        size_t k = j + 1;
        while (k < end && json_ws(text[k])) ++k;
        if (k >= end || text[k] != ':') out.emplace_back(i + 1, j);
        i = j + 1;
    }
    return out;
}

Lines detect_lines(const std::string & text, size_t begin, size_t end) {
    Lines best = split_lines(text, begin, end, false);
    for (const auto & v : json_string_values(text, begin, end)) {
        Lines escaped = split_lines(text, v.first, v.second, true);
        if (escaped.count() > best.count()) best = std::move(escaped);
    }
    return best;
}

// ── Region layout ───────────────────────────────────────────────────────

size_t line_cut(const Lines & lines, const std::vector<size_t> & common, int line) {
    if (line <= 0 || (size_t) line >= lines.count()) return std::string::npos;
    const size_t start = lines.starts[(size_t) line];
    if (std::binary_search(common.begin(), common.end(), start)) return start;
    const size_t sep = lines.ends[(size_t) line - 1];
    if (sep > lines.starts[(size_t) line - 1] && std::binary_search(common.begin(), common.end(), sep)) return sep;
    return std::string::npos;
}

bool layout_region(const Lines & lines, const std::vector<size_t> & common,
                   const std::vector<size_t> & proposed, Layout & out, std::string & why) {
    out = Layout{};
    const int n = (int) lines.count();
    if (n < 5) {
        why = "fewer than five lines";
        return false;
    }
    int head_line = -1;
    size_t head_cut = std::string::npos;
    for (int l = 2; l <= n - 3; ++l) {
        head_cut = line_cut(lines, common, l);
        if (head_cut != std::string::npos) {
            head_line = l;
            break;
        }
    }
    int tail_line = -1;
    size_t tail_cut = std::string::npos;
    for (int l = n - 2; l > std::max(head_line, 2); --l) {
        tail_cut = line_cut(lines, common, l);
        if (tail_cut != std::string::npos) {
            tail_line = l;
            break;
        }
    }
    if (head_line < 0 || tail_line < 0) {
        why = "no clean head or tail cut";
        return false;
    }
    std::vector<int> open;
    for (size_t p : proposed) {
        if (p <= lines.begin || p >= lines.end) continue;
        int li = (int) (std::upper_bound(lines.starts.begin(), lines.starts.end(), p) - lines.starts.begin()) - 1;
        if (li < 0) continue;
        if (p >= lines.ends[(size_t) li] && li + 1 < n) ++li;
        if (li > head_line && li < tail_line) open.push_back(li);
    }
    std::sort(open.begin(), open.end());
    open.erase(std::unique(open.begin(), open.end()), open.end());
    out.head_end = head_cut;
    out.tail_begin = tail_cut;
    out.cuts.push_back(head_cut);
    out.cut_line.push_back(head_line);
    for (int li : open) {
        const size_t c = line_cut(lines, common, li);
        if (c == std::string::npos) continue;
        out.cuts.push_back(c);
        out.cut_line.push_back(li);
    }
    out.cuts.push_back(tail_cut);
    out.cut_line.push_back(tail_line);
    return true;
}

// ── Text ────────────────────────────────────────────────────────────────

std::string json_unescape(const std::string & text, size_t begin, size_t end) {
    std::string out;
    out.reserve(end - begin);
    auto hex4 = [&](size_t at, uint32_t & v) {
        if (at + 4 > end) return false;
        v = 0;
        for (size_t k = at; k < at + 4; ++k) {
            const char c = text[k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return false;
        }
        return true;
    };
    auto put_utf8 = [&](uint32_t cp) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
    };
    size_t i = begin;
    while (i < end) {
        if (text[i] != '\\' || i + 1 >= end) {
            out += text[i++];
            continue;
        }
        const char c = text[i + 1];
        switch (c) {
            case 'n': out += '\n'; i += 2; continue;
            case 't': out += '\t'; i += 2; continue;
            case 'r': out += '\r'; i += 2; continue;
            case 'b': out += '\b'; i += 2; continue;
            case 'f': out += '\f'; i += 2; continue;
            case '"': case '\\': case '/': out += c; i += 2; continue;
            case 'u': {
                uint32_t v = 0;
                if (!hex4(i + 2, v)) break;
                size_t used = 6;
                if (v >= 0xD800 && v < 0xDC00 && i + 12 <= end && text[i + 6] == '\\' && text[i + 7] == 'u') {
                    uint32_t lo = 0;
                    if (hex4(i + 8, lo) && lo >= 0xDC00 && lo < 0xE000) {
                        v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                        used = 12;
                    }
                }
                put_utf8(v);
                i += used;
                continue;
            }
            default: break;
        }
        out += text[i++];
    }
    return out;
}


std::string json_escape(const std::string & text) {
    static const char * hex = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0xF];
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

std::string marker_text(int lines_not_shown) {
    return "[\xE2\x80\xA6 " + std::to_string(lines_not_shown) + (lines_not_shown == 1 ? " line" : " lines") +
           " not shown \xE2\x80\xA6]";
}

std::string preamble_text(RegionKind kind, int shown_lines, int total_lines) {
    const std::string counts = std::to_string(shown_lines) + " of " + std::to_string(total_lines) +
                               " lines are shown; each omitted range is marked inline.";
    if (kind == RegionKind::ToolOutput) {
        return "[Server note: this tool result was shortened: " + counts +
               " Call the tool again if you need an omitted part.]";
    }
    return "[Server note: this message was shortened: " + counts + "]";
}

std::string recall_label_text(RegionKind kind, const std::string & brief) {
    if (kind == RegionKind::ToolOutput) return "[From the earlier result of " + brief + "]";
    return "[From an earlier user message]";
}

std::string inserted_line(const std::string & text, const Lines & lines, size_t at, size_t next,
                          const std::string & body) {
    const std::string sep = lines.sep();
    const bool before = at > lines.begin && std::binary_search(lines.starts.begin(), lines.starts.end(), at);
    bool after = false;
    if (next < lines.end && next + sep.size() <= text.size() && text.compare(next, sep.size(), sep) == 0) {
        after = std::binary_search(lines.ends.begin(), lines.ends.end(), next);
    }
    std::string out;
    if (!before) out += sep;
    out += lines.escaped ? json_escape(body) : body;
    if (!after) out += sep;
    return out;
}

std::string render_region_text(const std::string & text, size_t content_begin, size_t content_end,
                               const Lines & lines, const Layout & layout, const std::vector<char> & admitted,
                               RegionKind kind) {
    const int n = layout.units();
    int dropped = 0;
    for (int u = 0; u < n; ++u) {
        if (!admitted[(size_t) u]) dropped += layout.line_count(u, u);
    }
    std::string out = text.substr(content_begin, layout.head_end - content_begin);
    out += inserted_line(text, lines, layout.head_end, layout.head_end,
                         preamble_text(kind, (int) lines.count() - dropped, (int) lines.count()));
    for (int u = 0; u < n;) {
        if (admitted[(size_t) u]) {
            out += text.substr(layout.cuts[(size_t) u], layout.cuts[(size_t) u + 1] - layout.cuts[(size_t) u]);
            ++u;
            continue;
        }
        int b = u;
        while (b + 1 < n && !admitted[(size_t) b + 1]) ++b;
        out += inserted_line(text, lines, layout.cuts[(size_t) u], layout.cuts[(size_t) b + 1],
                             marker_text(layout.line_count(u, b)));
        u = b + 1;
    }
    out += text.substr(layout.tail_begin, content_end - layout.tail_begin);
    return out;
}

// ── Admission ──────────────────────────────────────────────────────────

namespace {

// The maximal run around ``u`` of units whose ``in`` equals ``in[u]``.
std::pair<int, int> run_around(const std::vector<char> & in, int u) {
    int a = u;
    int b = u;
    while (a > 0 && in[(size_t) a - 1] == in[(size_t) u]) --a;
    while (b + 1 < (int) in.size() && in[(size_t) b + 1] == in[(size_t) u]) ++b;
    return {a, b};
}

}  // namespace

Admission admit_paging(const std::vector<int> & unit_cost, const std::vector<char> & locked,
                       const std::vector<int> & order, int fixed, int cap,
                       const std::function<int(int, int)> & marker_cost) {
    Admission out;
    const int n = (int) unit_cost.size();
    out.admitted.assign((size_t) n, 0);
    for (int u = 0; u < n; ++u) out.admitted[(size_t) u] = (u < (int) locked.size() && locked[(size_t) u]) ? 1 : 0;
    out.cost = fixed;
    for (int u = 0; u < n;) {
        if (out.admitted[(size_t) u]) {
            ++u;
            continue;
        }
        const auto run = run_around(out.admitted, u);
        out.cost += marker_cost(run.first, run.second);
        u = run.second + 1;
    }
    if (out.cost > cap) return out;
    out.ok = true;
    for (int u : order) {
        if (u < 0 || u >= n || out.admitted[(size_t) u]) continue;
        const auto gap = run_around(out.admitted, u);
        int delta = unit_cost[(size_t) u] - marker_cost(gap.first, gap.second);
        if (gap.first <= u - 1) delta += marker_cost(gap.first, u - 1);
        if (u + 1 <= gap.second) delta += marker_cost(u + 1, gap.second);
        if (out.cost + delta > cap) continue;
        out.admitted[(size_t) u] = 1;
        out.cost += delta;
        ++out.taken;
    }
    return out;
}

Admission admit_recall(const std::vector<int> & unit_cost, const std::vector<char> & held,
                       const std::vector<int> & order, int cap,
                       const std::function<int(int, int)> & label_cost) {
    Admission out;
    const int n = (int) unit_cost.size();
    out.admitted.assign((size_t) n, 0);
    out.ok = cap >= 0;
    if (!out.ok) return out;
    for (int u : order) {
        if (u < 0 || u >= n || out.admitted[(size_t) u]) continue;
        if (u < (int) held.size() && held[(size_t) u]) continue;
        const bool left = u > 0 && out.admitted[(size_t) u - 1];
        const bool right = u + 1 < n && out.admitted[(size_t) u + 1];
        int a = u;
        int b = u;
        int delta = unit_cost[(size_t) u];
        if (left) {
            const auto run = run_around(out.admitted, u - 1);
            a = run.first;
            delta -= label_cost(run.first, run.second);
        }
        if (right) {
            const auto run = run_around(out.admitted, u + 1);
            b = run.second;
            delta -= label_cost(run.first, run.second);
        }
        delta += label_cost(a, b);
        if (out.cost + delta > cap) continue;
        out.admitted[(size_t) u] = 1;
        out.cost += delta;
        ++out.taken;
    }
    return out;
}

std::vector<int> rank_units(const std::vector<double> & score, const std::vector<char> & skip) {
    std::vector<int> order;
    for (int u = 0; u < (int) score.size(); ++u) {
        if (u < (int) skip.size() && skip[(size_t) u]) continue;
        order.push_back(u);
    }
    auto key = [&](int u) {
        const double s = score[(size_t) u];
        return std::isnan(s) ? -INFINITY : s;
    };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key(a) > key(b); });
    return order;
}

// ── Re-reads ───────────────────────────────────────────────────────────

std::vector<uint64_t> line_hashes(const std::string & text, const Lines & lines) {
    std::vector<uint64_t> out;
    out.reserve(lines.count());
    for (size_t i = 0; i < lines.count(); ++i) {
        std::string line = lines.escaped ? json_unescape(text, lines.starts[i], lines.ends[i])
                                         : text.substr(lines.starts[i], lines.ends[i] - lines.starts[i]);
        while (!line.empty() && line.back() == '\r') line.pop_back();
        const bool blank = std::all_of(line.begin(), line.end(), [](unsigned char c) {
            return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
        });
        out.push_back(blank ? 0 : fnv1a(line));
    }
    return out;
}

RereadMatch match_reread(const std::vector<uint64_t> & candidate, const std::vector<uint64_t> & region,
                         int min_run, int max_repeat) {
    RereadMatch out;
    out.region_covered.assign(region.size(), 0);
    std::unordered_map<uint64_t, int> freq;
    for (uint64_t h : region) if (h) ++freq[h];
    auto informative = [&](uint64_t h) {
        if (!h) return false;
        const auto it = freq.find(h);
        return it == freq.end() || it->second <= max_repeat;
    };
    std::vector<int> reg;     // informative region lines
    for (int i = 0; i < (int) region.size(); ++i) if (informative(region[(size_t) i])) reg.push_back(i);
    std::vector<int> cand;    // informative candidate lines
    for (int j = 0; j < (int) candidate.size(); ++j) if (informative(candidate[(size_t) j])) cand.push_back(j);
    out.candidate_informative = (int) cand.size();
    std::unordered_map<uint64_t, std::vector<int>> where;
    for (int p = 0; p < (int) reg.size(); ++p) where[region[(size_t) reg[(size_t) p]]].push_back(p);
    std::vector<char> matched(cand.size(), 0);
    std::unordered_map<int, int> prev;
    for (int j = 0; j < (int) cand.size(); ++j) {
        std::unordered_map<int, int> cur;
        const auto it = where.find(candidate[(size_t) cand[(size_t) j]]);
        if (it != where.end()) {
            for (int p : it->second) {
                const auto pr = prev.find(p - 1);
                const int len = pr == prev.end() ? 1 : pr->second + 1;
                cur[p] = len;
                if (len < min_run) continue;
                const int from = len == min_run ? len - 1 : 0;
                for (int k = from; k >= 0; --k) {
                    matched[(size_t) (j - k)] = 1;
                    out.region_covered[(size_t) reg[(size_t) (p - k)]] = 1;
                }
            }
        }
        prev.swap(cur);
    }
    for (char m : matched) out.matched_candidate += m ? 1 : 0;
    // Skipped region lines between two covered informative lines are covered.
    for (size_t k = 0; k + 1 < reg.size(); ++k) {
        if (!out.region_covered[(size_t) reg[k]] || !out.region_covered[(size_t) reg[k + 1]]) continue;
        for (int i = reg[k] + 1; i < reg[k + 1]; ++i) out.region_covered[(size_t) i] = 1;
    }
    return out;
}

// ── View ───────────────────────────────────────────────────────────────

size_t View::bytes() const {
    size_t n = sizeof(View) + config.size();
    n += (raw_tokens.size() + drafter_ids.size() + view_tokens.size()) * sizeof(int32_t);
    n += pieces.size() * sizeof(Piece);
    for (const auto & p : pieces) n += p.kind.size();
    for (const auto & r : regions) {
        n += sizeof(Region) + r.brief.size() + r.tool_call_id.size() + r.content_sha.size();
        n += (r.lines.starts.size() + r.lines.ends.size() + r.layout.cuts.size()) * sizeof(size_t);
        n += r.layout.cut_line.size() * sizeof(int) + r.hashes.size() * sizeof(uint64_t);
        n += (r.unit_tb.size() + r.unit_te.size() + r.unit_db.size() + r.unit_de.size()) * sizeof(int);
        n += r.mandatory.size() + r.kept.size() + r.held.size() + r.reread.size();
    }
    return n;
}

namespace {

bool prefix_of(const std::vector<int32_t> & stored, int gen_begin, const std::vector<int32_t> & request) {
    const size_t n = gen_begin < 0 ? stored.size() : std::min(stored.size(), (size_t) gen_begin);
    return request.size() >= n && std::equal(stored.begin(), stored.begin() + (std::ptrdiff_t) n, request.begin());
}

}  // namespace

bool ViewStore::find(const std::vector<int32_t> & raw_tokens, const std::vector<int32_t> & drafter_ids,
                     const std::string & config, View & out) {
    std::lock_guard<std::mutex> lock(mutex_);
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < (int) entries_.size(); ++i) {
        const View & v = entries_[(size_t) i].view;
        if (v.config != config) continue;
        if (!prefix_of(v.raw_tokens, v.raw_gen_begin, raw_tokens)) continue;
        if (!prefix_of(v.drafter_ids, v.drafter_gen_begin, drafter_ids)) continue;
        const size_t len = v.raw_gen_begin < 0 ? v.raw_tokens.size() : (size_t) v.raw_gen_begin;
        if (best < 0 || len > best_len ||
            (len == best_len && entries_[(size_t) i].used > entries_[(size_t) best].used)) {
            best = i;
            best_len = len;
        }
    }
    if (best < 0) return false;
    entries_[(size_t) best].used = ++clock_;
    out = entries_[(size_t) best].view;
    return true;
}

void ViewStore::remember(View view) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const View & v = it->view;
        if (v.config == view.config && v.raw_tokens == view.raw_tokens && v.drafter_ids == view.drafter_ids) {
            entries_.erase(it);
            break;
        }
    }
    entries_.push_back(Entry{std::move(view), ++clock_});
    auto total = [&] {
        size_t n = 0;
        for (const auto & e : entries_) n += e.view.bytes();
        return n;
    };
    while (entries_.size() > 1 && (entries_.size() > capacity_ || total() > byte_cap_)) {
        size_t oldest = 0;
        for (size_t i = 1; i + 1 < entries_.size(); ++i) {
            if (entries_[i].used < entries_[oldest].used) oldest = i;
        }
        entries_.erase(entries_.begin() + (std::ptrdiff_t) oldest);
    }
}

size_t ViewStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

size_t ViewStore::bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = 0;
    for (const auto & e : entries_) n += e.view.bytes();
    return n;
}

std::string sha1_hex(const std::string & bytes) {
    uint8_t digest[20];
    luce::common::sha1_hash(bytes.data(), bytes.size(), digest);
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (uint8_t b : digest) {
        out += hex[b >> 4];
        out += hex[b & 0xF];
    }
    return out;
}

}  // namespace pflash_unified
