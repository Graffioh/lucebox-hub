#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// dflash27b_ggml: tree-mode ssm_conv kernel. For each new-token t, walks up
// the parent chain K-1 times via parent_ids[] to find the (K-1) ancestor slots
// in the conv input, then convolves with the kernel weights. Virtual-slot
// encoding: a non-negative parent index `p` maps to sx slot (K-1 + p). A
// parent index of -1 means "before the block" — i.e., the old conv state.
// Each successive walk beyond -1 decrements by 1, so virtual slot -k maps to
// sx slot (K-1 - k), which indexes into the old state region [0, K-1). This
// matches SGLang's causal_conv1d_triton HAS_EAGLE_TREE_CUSTOM_ATTN_MASK path.
template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_tree_f32(
        const float * __restrict__ src0,        // sx: [K-1+n_t, d_inner, n_s]
        const float * __restrict__ src1,        // c:  [K, d_inner]
        const int * __restrict__   parent_ids,  // [n_t, n_s]
        const int src0_nb0, const int src0_nb1, const int src0_nb2,
        const int src1_nb1,
        float * __restrict__ dst,               // [d_inner, n_t, n_s]
        const int dst_nb0, const int dst_nb1, const int dst_nb2,
        const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;  // sequence
    const int bidy = blockIdx.y;  // d_inner / split_d_inner

    const float * x_block = (const float *) ((const char *) src0
        + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1
        + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst
        + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    // Load kernel weights into registers.
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    const int * parent_ids_seq = parent_ids + bidx * n_t;

    for (int64_t i = 0; i < n_t; i++) {
        // Walk the parent chain K-1 times to fill the conv window.
        // ancestor_virtual[k] gives the "virtual slot" for kernel position k,
        // where the most recent slot is at k=K-1 (= token i itself) and older
        // slots are at k=K-2, K-3, ..., 0.
        //
        // ancestor_virtual[K-1] = i
        // ancestor_virtual[K-2] = parent_of(i) (or i-1 decay for negative)
        // ancestor_virtual[k  ] = parent_of(ancestor_virtual[k+1])
        int ancestors[d_conv];
        ancestors[d_conv - 1] = (int)i;
#pragma unroll
        for (int k = (int)d_conv - 2; k >= 0; k--) {
            int prev = ancestors[k + 1];
            int next;
            if (prev >= 0) {
                next = parent_ids_seq[prev];  // -1 if parent is before block
            } else {
                next = prev - 1;  // keep decaying through old state slots
            }
            ancestors[k] = next;
        }

        float sumf = 0.0f;
#pragma unroll
        for (size_t k = 0; k < d_conv; k++) {
            // Map virtual slot → sx slot: sx_slot = (K-1) + ancestors[k].
            const int sx_slot = (int)(d_conv - 1) + ancestors[k];
            const float x_val = x_block[tid * stride_x + sx_slot];
            sumf += x_val * w[k];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_tree_f32_cuda(const float * src0, const float * src1, const int * parent_ids,
                                   const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                   const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                                   const int dst_nb2, const int64_t nc, const int64_t nr,
                                   const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        ssm_conv_tree_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(
            src0, src1, parent_ids, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
            dst, dst_nb0, dst_nb1, dst_nb2, n_t);
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Tree ssm_conv only supports kernel sizes 3, 4, 5, 9.");
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                       dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

// dflash: fused conv step (see ggml_ssm_conv_step). One thread per channel
// walks the token loop with the K-1 history in registers, writes silu(conv),
// the optional rollback window and the new history in place.
template <int K>
static __global__ void ssm_conv_step_f32(const float * __restrict__ x, const int x_nb1, const int x_nb2,
                                         const float * __restrict__ w, const int w_nb1,
                                         float * state, const int st_nb1, const int st_nb2,
                                         float * __restrict__ y, const int y_nb1, const int y_nb2,
                                         float * ci, const int ci_nb1, const int ci_nb2,
                                         const int C, const int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= C) return;

    const float * xs = (const float *) ((const char *) x + (size_t) s * x_nb2) + c;
    float *       st = (float *) ((char *) state + (size_t) s * st_nb2 + (size_t) c * st_nb1);
    float *       ys = (float *) ((char *) y + (size_t) s * y_nb2) + c;
    float *       cs = ci ? (float *) ((char *) ci + (size_t) s * ci_nb2 + (size_t) c * ci_nb1) : nullptr;
    const float * wc = (const float *) ((const char *) w + (size_t) c * w_nb1);

    const int xs_stride = x_nb1 / sizeof(float);
    const int ys_stride = y_nb1 / sizeof(float);

    float wt[K];
    float win[K];   // oldest first; win[K-1] is the current input
#pragma unroll
    for (int k = 0; k < K; k++) {
        wt[k] = wc[k];
    }
#pragma unroll
    for (int j = 0; j < K - 1; j++) {
        win[j] = st[j];
        if (cs) cs[j] = win[j];
    }
    for (int t = 0; t < T; t++) {
        const float xt = xs[(size_t) t * xs_stride];
        win[K - 1] = xt;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < K; k++) {
            acc += win[k] * wt[k];
        }
        ys[(size_t) t * ys_stride] = ggml_cuda_op_silu_single(acc);
        if (cs) cs[K - 1 + t] = xt;
#pragma unroll
        for (int j = 0; j < K - 1; j++) {
            win[j] = win[j + 1];
        }
    }
#pragma unroll
    for (int j = 0; j < K - 1; j++) {
        st[j] = win[j];
    }
}

static void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * w  = dst->src[1];
    ggml_tensor *       st = dst->src[2];
    ggml_tensor *       ci = dst->src[3];

    const int K = (int) w->ne[0];
    const int C = (int) w->ne[1];
    const int T = (int) dst->ne[1];
    const int S = (int) dst->ne[2];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && st->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float));
    GGML_ASSERT(w->nb[0] == sizeof(float));
    GGML_ASSERT(st->nb[0] == sizeof(float) && st->nb[1] == (size_t) (K - 1) * sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    if (ci) {
        GGML_ASSERT(ci->type == GGML_TYPE_F32 && ci->nb[0] == sizeof(float));
        GGML_ASSERT(ci->ne[0] >= K - 1 + T);
    }

    const int threads = 256;
    const dim3 blocks((C + threads - 1) / threads, S, 1);
    cudaStream_t stream = ctx.stream();

    auto launch = [&](auto KK) {
        constexpr int kK = decltype(KK)::value;
        ssm_conv_step_f32<kK><<<blocks, threads, 0, stream>>>(
            (const float *) x->data, (int) x->nb[1], (int) x->nb[2],
            (const float *) w->data, (int) w->nb[1],
            (float *) st->data, (int) st->nb[1], (int) st->nb[2],
            (float *) dst->data, (int) dst->nb[1], (int) dst->nb[2],
            ci ? (float *) ci->data : nullptr, ci ? (int) ci->nb[1] : 0, ci ? (int) ci->nb[2] : 0,
            C, T);
    };
    switch (K) {
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 9: launch(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("ssm_conv_step only supports kernel sizes 3, 4, 5, 9.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    // dflash: fused step mode (silu already applied by the kernel)
    if (ggml_get_op_params_i32(dst, 0) == 1) {
        GGML_ASSERT(silu_dst == nullptr);
        ggml_cuda_op_ssm_conv_step(ctx, dst);
        return;
    }

    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    // dflash27b_ggml: optional src[2] = parent_ids (i32) enables tree mode
    const struct ggml_tensor * parent_ids = dst->src[2];
    const bool fuse_silu = silu_dst != nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);

    if (parent_ids != nullptr) {
        GGML_ASSERT(parent_ids->type == GGML_TYPE_I32);
        const int * parent_ids_d = (const int *) parent_ids->data;
        if (fuse_silu) {
            ssm_conv_tree_f32_cuda<true>(src0_d, src1_d, parent_ids_d,
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
                dst_d, out->nb[0], out->nb[1], out->nb[2],
                nc, nr, n_t, n_s, stream);
        } else {
            ssm_conv_tree_f32_cuda<false>(src0_d, src1_d, parent_ids_d,
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
                dst_d, out->nb[0], out->nb[1], out->nb[2],
                nc, nr, n_t, n_s, stream);
        }
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
