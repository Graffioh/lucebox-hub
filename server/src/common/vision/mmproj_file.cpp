#include "mmproj_file.h"

#include "gguf.h"

#include <cstdio>
#include <memory>

namespace luce::vision {

MmprojFile::~MmprojFile() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_) ggml_free(ctx_);
    if (gguf_) gguf_free(gguf_);
}

bool MmprojFile::load(const std::string & path, ggml_backend_t backend, std::string & error) {
    error.clear();
    if (gguf_ || !backend) { error = "projector is already loaded or has no backend"; return false; }

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &ctx_;
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ || !ctx_) { error = "cannot read projector GGUF: " + path; return false; }

    const int64_t arch = gguf_find_key(gguf_, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(gguf_, arch) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(gguf_, arch)) != "clip") {
        error = "not a clip projector file (general.architecture != clip)";
        return false;
    }

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buffer_) { error = "cannot allocate projector weights on the backend"; return false; }
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weight_bytes_ = ggml_backend_buffer_get_size(buffer_);

    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file(std::fopen(path.c_str(), "rb"), std::fclose);
    if (!file) { error = "cannot open projector file: " + path; return false; }
    const size_t data_offset = gguf_get_data_offset(gguf_);
    std::vector<uint8_t> staging;
    const int64_t n_tensors = gguf_get_n_tensors(gguf_);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor * t = ggml_get_tensor(ctx_, name);
        if (!t) { error = std::string("projector tensor has no metadata: ") + name; return false; }
        const size_t size = ggml_nbytes(t);
        staging.resize(size);
        const size_t offset = data_offset + gguf_get_tensor_offset(gguf_, i);
#if defined(_WIN32)
        const int seek = _fseeki64(file.get(), (long long) offset, SEEK_SET);
#else
        const int seek = fseeko(file.get(), (off_t) offset, SEEK_SET);
#endif
        if (seek != 0 || std::fread(staging.data(), 1, size, file.get()) != size) {
            error = std::string("projector file is truncated at tensor ") + name;
            return false;
        }
        ggml_backend_tensor_set(t, staging.data(), 0, size);
    }
    return true;
}

std::string MmprojFile::projector_type() const {
    if (!gguf_) return {};
    const int64_t id = gguf_find_key(gguf_, "clip.projector_type");
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) return {};
    return gguf_get_val_str(gguf_, id);
}

bool MmprojFile::u32(const char * key, uint32_t & out) const {
    const int64_t id = gguf_ ? gguf_find_key(gguf_, key) : -1;
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_UINT32) return false;
    out = gguf_get_val_u32(gguf_, id);
    return true;
}

bool MmprojFile::f32(const char * key, float & out) const {
    const int64_t id = gguf_ ? gguf_find_key(gguf_, key) : -1;
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_FLOAT32) return false;
    out = gguf_get_val_f32(gguf_, id);
    return true;
}

bool MmprojFile::f32_array(const char * key, std::vector<float> & out) const {
    const int64_t id = gguf_ ? gguf_find_key(gguf_, key) : -1;
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf_, id) != GGUF_TYPE_FLOAT32) return false;
    const auto * values = static_cast<const float *>(gguf_get_arr_data(gguf_, id));
    out.assign(values, values + gguf_get_arr_n(gguf_, id));
    return true;
}

ggml_tensor * MmprojFile::tensor(const std::string & name) const {
    return ctx_ ? ggml_get_tensor(ctx_, name.c_str()) : nullptr;
}

bool MmprojFile::has_tensor_prefix(const std::string & prefix) const {
    const int64_t n_tensors = gguf_ ? gguf_get_n_tensors(gguf_) : 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        if (std::string(gguf_get_tensor_name(gguf_, i)).rfind(prefix, 0) == 0) return true;
    }
    return false;
}

} // namespace luce::vision
