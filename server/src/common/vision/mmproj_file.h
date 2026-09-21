// A multimodal projector file in llama.cpp's "clip" GGUF layout: the standard
// file published next to a vision-capable model. Holds the metadata and the
// tensors, loaded onto one backend. Each model's tower reads what it needs.
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace dflash::vision {

class MmprojFile {
public:
    MmprojFile() = default;
    ~MmprojFile();
    MmprojFile(const MmprojFile &) = delete;
    MmprojFile & operator=(const MmprojFile &) = delete;

    // Loads every tensor onto `backend`, which must outlive this object.
    bool load(const std::string & path, ggml_backend_t backend, std::string & error);

    // "clip.projector_type", for example "qwen3vl_merger". Empty when absent.
    std::string projector_type() const;

    // Metadata by full key. False when the key is absent or has another type.
    bool u32(const char * key, uint32_t & out) const;
    bool f32(const char * key, float & out) const;
    bool f32_array(const char * key, std::vector<float> & out) const;

    // nullptr when the file has no tensor of that name.
    ggml_tensor * tensor(const std::string & name) const;
    // True when any tensor name starts with `prefix`.
    bool has_tensor_prefix(const std::string & prefix) const;

    size_t weight_bytes() const { return weight_bytes_; }

private:
    gguf_context * gguf_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    size_t weight_bytes_ = 0;
};

} // namespace dflash::vision
