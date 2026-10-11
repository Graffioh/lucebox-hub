#include "pflash_session_offload.h"

#include "ggml-backend.h"

#include <sys/stat.h>

#include <cstring>
#include <sstream>

namespace luce::pflash {

std::vector<ggml_tensor *> pflash_owned_tensors(ggml_context * ctx) {
    std::vector<ggml_tensor *> out;
    if (!ctx) return out;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src == nullptr && t->buffer != nullptr && t->data != nullptr) out.push_back(t);
    }
    return out;
}

bool pflash_tensors_to_host(const std::vector<ggml_tensor *> & tensors, std::vector<PFlashHostTensor> & out,
                            std::string & error) {
    out.clear();
    out.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * t = tensors[i];
        if (!t || !t->buffer || !t->data || t->view_src) {
            out.clear();
            error = "tensor " + std::to_string(i) + " is not an allocated owned tensor";
            return false;
        }
        PFlashHostTensor h;
        h.name = ggml_get_name(t);
        h.type = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h.ne[d] = t->ne[d];
            h.nb[d] = t->nb[d];
        }
        h.bytes.resize(ggml_nbytes(t));
        if (!h.bytes.empty()) ggml_backend_tensor_get(t, h.bytes.data(), 0, h.bytes.size());
        out.push_back(std::move(h));
    }
    return true;
}

bool pflash_tensors_from_host(const std::vector<PFlashHostTensor> & host, const std::vector<ggml_tensor *> & tensors,
                              std::string & error) {
    if (host.size() != tensors.size()) {
        error = "tensor count " + std::to_string(tensors.size()) + " != stored " + std::to_string(host.size());
        return false;
    }
    for (size_t i = 0; i < host.size(); ++i) {
        const ggml_tensor * t = tensors[i];
        const PFlashHostTensor & h = host[i];
        const std::string where = "tensor " + std::to_string(i) + " (" + h.name + ")";
        if (!t || !t->buffer || !t->data || t->view_src) {
            error = where + " is not an allocated owned tensor";
            return false;
        }
        if (h.type != t->type) {
            error = where + " type differs";
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (h.ne[d] != t->ne[d] || h.nb[d] != t->nb[d]) {
                error = where + " shape differs";
                return false;
            }
        }
        if (h.bytes.size() != ggml_nbytes(t)) {
            error = where + " size " + std::to_string(ggml_nbytes(t)) + " != stored " +
                    std::to_string(h.bytes.size());
            return false;
        }
    }
    for (size_t i = 0; i < host.size(); ++i) {
        if (!host[i].bytes.empty()) {
            ggml_backend_tensor_set(tensors[i], host[i].bytes.data(), 0, host[i].bytes.size());
        }
    }
    return true;
}

size_t pflash_host_tensor_bytes(const std::vector<PFlashHostTensor> & host) {
    size_t total = 0;
    for (const auto & h : host) total += h.bytes.size();
    return total;
}

std::string pflash_file_identity(const std::string & path) {
    if (path.empty()) return "-";
    struct stat s {};
    if (::stat(path.c_str(), &s) != 0) return path + "|missing";
    std::ostringstream out;
    out << path << '|' << (unsigned long long) s.st_dev << ':' << (unsigned long long) s.st_ino << '|'
        << (long long) s.st_size << '|' << (long long) s.st_mtim.tv_sec * 1000000000LL + (long long) s.st_mtim.tv_nsec;
    return out.str();
}

std::string pflash_identity_mismatch(const std::string & stored, const std::string & current) {
    if (stored == current) return "";
    std::istringstream a(stored), b(current);
    std::string la, lb;
    while (true) {
        const bool ga = (bool) std::getline(a, la);
        const bool gb = (bool) std::getline(b, lb);
        if (!ga && !gb) return "identity differs";
        if (!ga) return lb.substr(0, lb.find('=')) + " added";
        if (!gb) return la.substr(0, la.find('=')) + " removed";
        if (la != lb) return la.substr(0, la.find('=')) + " differs";
    }
}

} // namespace luce::pflash
