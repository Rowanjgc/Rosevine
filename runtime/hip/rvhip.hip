// SPDX-License-Identifier: GPL-3.0-only
// Rosevine R2 HIP kernels. See rvhip.h for the ABI and layouts. Correctness first: one simple
// kernel per operation, f32 accumulation (f64 for RMS sums, as the CPU executor and ggml do).
#include "rvhip.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>

#define QK 32
#define Q4_BLOCK 18
#define Q8_BLOCK 34

namespace {

__device__ __forceinline__ float f16(const uint8_t *p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(u));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, warpSize);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, warpSize));
    return v;
}

// Block-wide reductions for blockDim.x <= 1024; `sh` holds one value per warp.
__device__ float block_sum(float v, float *sh) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < nw; ++i) r += sh[i];
    return r;
}

__device__ float block_max(float v, float *sh) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float r = -INFINITY;
    for (int i = 0; i < nw; ++i) r = fmaxf(r, sh[i]);
    return r;
}

__device__ double block_sum_d(double v, double *sh) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, warpSize);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    double r = 0.0;
    for (int i = 0; i < nw; ++i) r += sh[i];
    return r;
}

__global__ void k_get_rows_q4(const uint8_t *w, const int32_t *ids, float *out, int k, float scale) {
    const int t = blockIdx.x;
    const uint8_t *row = w + (size_t)ids[t] * (k / QK) * Q4_BLOCK;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const uint8_t *b = row + (i / QK) * Q4_BLOCK;
        const int j = i % QK;
        const int q = j < 16 ? (b[2 + j] & 15) : (b[2 + j - 16] >> 4);
        out[(size_t)t * k + i] = (float)(q - 8) * f16(b) * scale;
    }
}

__global__ void k_rms_norm(const float *x, float *out, const float *w, int width, float eps, float post) {
    __shared__ double sh[32];
    const float *xr = x + (size_t)blockIdx.x * width;
    float *o = out + (size_t)blockIdx.x * width;
    double s = 0.0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) s += (double)xr[i] * (double)xr[i];
    s = block_sum_d(s, sh);
    const float mean = (float)(s / width);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = xr[i] * scale;
        if (post != 1.0f) v = v * post;
        if (w) v = v * w[i];
        o[i] = v;
    }
}

// One warp per output row, TT tokens per launch row-tile; lanes split the k/32 blocks.
constexpr int TT = 8;
__global__ void k_q4_mm(const uint8_t *w, const float *x, float *y, int n_out, int k, int n_in) {
    const int warps = blockDim.x / warpSize;
    const int r = blockIdx.x * warps + threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int t0 = blockIdx.y * TT;
    if (r >= n_out) return;
    const int nb = k / QK;
    const uint8_t *row = w + (size_t)r * nb * Q4_BLOCK;
    float acc[TT];
    for (int i = 0; i < TT; ++i) acc[i] = 0.f;
    for (int b = lane; b < nb; b += warpSize) {
        const uint8_t *blk = row + (size_t)b * Q4_BLOCK;
        const float d = f16(blk);
        float v[QK];
        for (int j = 0; j < 16; ++j) {
            v[j] = (float)((blk[2 + j] & 15) - 8);
            v[j + 16] = (float)((blk[2 + j] >> 4) - 8);
        }
        for (int i = 0; i < TT; ++i) {
            const int t = t0 + i;
            if (t >= n_in) break;
            const float *xs = x + (size_t)t * k + b * QK;
            float s = 0.f;
            for (int j = 0; j < QK; ++j) s += v[j] * xs[j];
            acc[i] += d * s;
        }
    }
    for (int i = 0; i < TT; ++i) {
        const int t = t0 + i;
        if (t >= n_in) break;
        const float s = warp_sum(acc[i]);
        if (lane == 0) y[(size_t)t * n_out + r] = s;
    }
}

__global__ void k_f32_mm(const float *w, const float *x, float *y, int n_out, int k, int n_in) {
    const int warps = blockDim.x / warpSize;
    const int r = blockIdx.x * warps + threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int t = blockIdx.y;
    if (r >= n_out || t >= n_in) return;
    float s = 0.f;
    for (int i = lane; i < k; i += warpSize) s += w[(size_t)r * k + i] * x[(size_t)t * k + i];
    s = warp_sum(s);
    if (lane == 0) y[(size_t)t * n_out + r] = s;
}

__global__ void k_head_norm_rope(float *x, int n_heads, int hd, const float *w, float eps, int p0, float base,
                                 const float *ff) {
    __shared__ double sh[32];
    __shared__ float buf[512];
    const int head = blockIdx.x;
    const int t = head / n_heads;
    float *h = x + (size_t)head * hd;
    double s = 0.0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) s += (double)h[i] * (double)h[i];
    s = block_sum_d(s, sh);
    const float scale = 1.0f / sqrtf((float)(s / hd) + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) buf[i] = w ? h[i] * scale * w[i] : h[i] * scale;
    __syncthreads();
    if (base > 0.f) {
        const int half = hd / 2;
        const float theta_scale = powf(base, -2.0f / (float)hd);
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            // theta_i = pos * theta_scale^i, accumulated multiplicatively as ggml does.
            float theta = (float)(p0 + t);
            for (int j = 0; j < i; ++j) theta *= theta_scale;
            const float f = ff ? ff[i] : 1.0f;
            float sn, cs;
            sincosf(theta / f, &sn, &cs);
            const float a = buf[i], b = buf[i + half];
            h[i] = a * cs - b * sn;
            h[i + half] = a * sn + b * cs;
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) h[i] = buf[i];
    }
}

// One warp-sized group of 32 threads per Q8_0 block; block (token, kv head).
__global__ void k_kv_store(const float *k, const float *v, uint8_t *kc, uint8_t *vc, int n_kv, int hd, int p0,
                           int cap) {
    const int t = blockIdx.x / n_kv, h = blockIdx.x % n_kv;
    const int row = hd / QK * Q8_BLOCK;
    const size_t slot = (size_t)((p0 + t) % cap);
    for (int which = 0; which < 2; ++which) {
        const float *src = (which ? v : k) + ((size_t)t * n_kv + h) * hd;
        uint8_t *dst = (which ? vc : kc) + (slot * n_kv + h) * row;
        for (int b = threadIdx.x / QK; b < hd / QK; b += blockDim.x / QK) {
            const int j = threadIdx.x % QK;
            const float val = src[b * QK + j];
            float amax = fabsf(val);
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            uint8_t *blk = dst + b * Q8_BLOCK;
            if (j == 0) {
                const unsigned short u = __half_as_ushort(__float2half(d));
                blk[0] = u & 0xff;
                blk[1] = u >> 8;
            }
            blk[2 + j] = (uint8_t)(int8_t)roundf(val * id);
        }
    }
}

__global__ void k_attention(const float *q, const uint8_t *kc, const uint8_t *vc, float *out, float *scores,
                            int n_head, int n_kv, int hd, int p0, int cap, int window, int max_keys) {
    __shared__ float sh[32];
    __shared__ float qs[512];
    const int i = blockIdx.x;
    const int t = i / n_head, h = i % n_head;
    const int g = h / (n_head / n_kv);
    const int pos = p0 + t;
    const int lo = window > 0 ? max(0, pos + 1 - window) : 0;
    const int len = pos - lo + 1;
    const int row = hd / QK * Q8_BLOCK;
    float *s = scores + (size_t)i * max_keys;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = q[(size_t)i * hd + d];
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const uint8_t *kr = kc + ((size_t)((lo + j) % cap) * n_kv + g) * row;
        float acc = 0.f;
        for (int b = 0; b < hd / QK; ++b) {
            const uint8_t *blk = kr + b * Q8_BLOCK;
            float sb = 0.f;
            for (int e = 0; e < QK; ++e) sb += (float)(int8_t)blk[2 + e] * qs[b * QK + e];
            acc += f16(blk) * sb;
        }
        s[j] = acc;
        mx = fmaxf(mx, acc);
    }
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float e = expf(s[j] - mx);
        s[j] = e;
        sum += e;
    }
    sum = block_sum(sum, sh);
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        const int b = d / QK, e = d % QK;
        for (int j = 0; j < len; ++j) {
            const uint8_t *blk = vc + ((size_t)((lo + j) % cap) * n_kv + g) * row + b * Q8_BLOCK;
            acc += s[j] * inv * f16(blk) * (float)(int8_t)blk[2 + e];
        }
        out[(size_t)i * hd + d] = acc;
    }
}

__device__ __forceinline__ float gelu(float x) {
    const float a = 0.044715f, c = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(c * x * (1.0f + a * x * x)));
}

__global__ void k_geglu(const float *g, const float *u, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu(g[i]) * u[i];
}

__global__ void k_geglu_split(const float *gu, float *out, int rows, int ff) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * ff) return;
    const size_t r = i / ff, j = i % ff;
    out[i] = gelu(gu[r * 2 * ff + j]) * gu[r * 2 * ff + ff + j];
}

__global__ void k_add(const float *a, const float *b, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void k_add_scale(float *x, const float *r, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (x[i] + r[i]) * s;
}

__global__ void k_gather_rows(const float *src, const int32_t *idx, float *dst, int width) {
    const float *s = src + (size_t)idx[blockIdx.x] * width;
    float *d = dst + (size_t)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) d[i] = s[i];
}

__global__ void k_moe_combine(const float *y, const int32_t *pos, const float *scale, const float *weight, float *out,
                              int used, int width) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < used; ++j) {
            const int p = t * used + j;
            const float v = y[(size_t)pos[p] * width + i] * scale[p] * weight[p];
            acc = j == 0 ? v : acc + v;
        }
        out[(size_t)t * width + i] = acc;
    }
}

__global__ void k_softcap(float *x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] * (1.0f / cap)) * cap;
}

int launched() { return (int)hipGetLastError(); }
int g_warp = 32;  // set by rvh_init
int blocks(size_t n, int per) { return (int)((n + per - 1) / per); }

}  // namespace

extern "C" {

int rvh_init(int device) {
    hipError_t e = hipSetDevice(device);
    if (e == hipSuccess) e = hipDeviceGetAttribute(&g_warp, hipDeviceAttributeWarpSize, device);
    return (int)e;
}

int rvh_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (!free_bytes || !total_bytes) return (int)hipErrorInvalidValue;
    return (int)hipMemGetInfo(free_bytes, total_bytes);
}

int rvh_alloc(size_t bytes, void **out) {
    if (!out) return (int)hipErrorInvalidValue;
    *out = nullptr;
    return (int)hipMalloc(out, bytes ? bytes : 1);
}

int rvh_free(void *ptr) { return ptr ? (int)hipFree(ptr) : 0; }

int rvh_upload(void *dst, const void *src, size_t bytes) {
    return (int)hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
}

int rvh_download(void *dst, const void *src, size_t bytes) {
    return (int)hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
}

int rvh_copy(void *dst, const void *src, size_t bytes) {
    return (int)hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
}

int rvh_sync(void) { return (int)hipDeviceSynchronize(); }

const char *rvh_error_string(int code) { return hipGetErrorString((hipError_t)code); }

#define CHECK_DIMS(cond) \
    do {                 \
        if (!(cond)) return (int)hipErrorInvalidValue; \
    } while (0)

int rvh_get_rows_q4(const void *w, const int32_t *ids, float *out, int n, int k, float scale) {
    CHECK_DIMS(w && ids && out && n > 0 && k > 0 && k % QK == 0);
    hipLaunchKernelGGL(k_get_rows_q4, dim3(n), dim3(256), 0, 0, (const uint8_t *)w, ids, out, k, scale);
    return launched();
}

int rvh_rms_norm(const float *x, float *out, const float *weight, int rows, int width, float eps, float post) {
    CHECK_DIMS(x && out && rows > 0 && width > 0);
    hipLaunchKernelGGL(k_rms_norm, dim3(rows), dim3(256), 0, 0, x, out, weight, width, eps, post);
    return launched();
}

int rvh_q4_mm(const void *w, const float *x, float *y, int n_out, int k, int n_in) {
    CHECK_DIMS(w && x && y && n_out > 0 && n_in > 0 && k > 0 && k % QK == 0);
    const int warps = 256 / g_warp;
    hipLaunchKernelGGL(k_q4_mm, dim3(blocks(n_out, warps), blocks(n_in, TT)), dim3(256), 0, 0, (const uint8_t *)w, x,
                       y, n_out, k, n_in);
    return launched();
}

int rvh_f32_mm(const float *w, const float *x, float *y, int n_out, int k, int n_in) {
    CHECK_DIMS(w && x && y && n_out > 0 && n_in > 0 && k > 0);
    hipLaunchKernelGGL(k_f32_mm, dim3(blocks(n_out, 256 / g_warp), n_in), dim3(256), 0, 0, w, x, y, n_out, k, n_in);
    return launched();
}

int rvh_head_norm_rope(float *x, int n_tok, int n_heads, int hd, const float *weight, float eps, int p0, float base,
                       const float *freq_factors) {
    CHECK_DIMS(x && n_tok > 0 && n_heads > 0 && hd > 0 && hd <= 512 && hd % 2 == 0 && p0 >= 0);
    hipLaunchKernelGGL(k_head_norm_rope, dim3(n_tok * n_heads), dim3(256), 0, 0, x, n_heads, hd, weight, eps, p0, base,
                       freq_factors);
    return launched();
}

int rvh_kv_store(const float *k, const float *v, void *kc, void *vc, int n_tok, int n_kv, int hd, int p0, int cap) {
    CHECK_DIMS(k && v && kc && vc && n_tok > 0 && n_kv > 0 && hd % QK == 0 && p0 >= 0 && cap > 0);
    hipLaunchKernelGGL(k_kv_store, dim3(n_tok * n_kv), dim3(256), 0, 0, k, v, (uint8_t *)kc, (uint8_t *)vc, n_kv, hd,
                       p0, cap);
    return launched();
}

int rvh_attention(const float *q, const void *kc, const void *vc, float *out, float *scores, int n_tok, int n_head,
                  int n_kv, int hd, int p0, int cap, int window, int max_keys) {
    CHECK_DIMS(q && kc && vc && out && scores && n_tok > 0 && n_kv > 0 && n_head % n_kv == 0 && hd % QK == 0 &&
               hd <= 512 && p0 >= 0 && cap > 0 && window >= 0);
    // Every query's key range must fit the scratch row and the cache.
    const int longest = window > 0 ? (p0 + n_tok < window ? p0 + n_tok : window) : p0 + n_tok;
    CHECK_DIMS(longest <= max_keys && longest <= cap);
    hipLaunchKernelGGL(k_attention, dim3(n_tok * n_head), dim3(256), 0, 0, q, (const uint8_t *)kc,
                       (const uint8_t *)vc, out, scores, n_head, n_kv, hd, p0, cap, window, max_keys);
    return launched();
}

int rvh_geglu(const float *g, const float *u, float *out, int n) {
    CHECK_DIMS(g && u && out && n > 0);
    hipLaunchKernelGGL(k_geglu, dim3(blocks(n, 256)), dim3(256), 0, 0, g, u, out, n);
    return launched();
}

int rvh_geglu_split(const float *gu, float *out, int rows, int ff) {
    CHECK_DIMS(gu && out && rows > 0 && ff > 0);
    hipLaunchKernelGGL(k_geglu_split, dim3(blocks((size_t)rows * ff, 256)), dim3(256), 0, 0, gu, out, rows, ff);
    return launched();
}

int rvh_add(const float *a, const float *b, float *out, int n) {
    CHECK_DIMS(a && b && out && n > 0);
    hipLaunchKernelGGL(k_add, dim3(blocks(n, 256)), dim3(256), 0, 0, a, b, out, n);
    return launched();
}

int rvh_add_scale(float *x, const float *r, float s, int n) {
    CHECK_DIMS(x && r && n > 0);
    hipLaunchKernelGGL(k_add_scale, dim3(blocks(n, 256)), dim3(256), 0, 0, x, r, s, n);
    return launched();
}

int rvh_gather_rows(const float *src, const int32_t *idx, float *dst, int n, int width) {
    CHECK_DIMS(src && idx && dst && n > 0 && width > 0);
    hipLaunchKernelGGL(k_gather_rows, dim3(n), dim3(256), 0, 0, src, idx, dst, width);
    return launched();
}

int rvh_moe_combine(const float *y, const int32_t *pos, const float *scale, const float *weight, float *out,
                    int n_tok, int used, int width) {
    CHECK_DIMS(y && pos && scale && weight && out && n_tok > 0 && used > 0 && width > 0);
    hipLaunchKernelGGL(k_moe_combine, dim3(n_tok), dim3(256), 0, 0, y, pos, scale, weight, out, used, width);
    return launched();
}

int rvh_softcap(float *x, int n, float cap) {
    CHECK_DIMS(x && n > 0 && cap != 0.0f);
    hipLaunchKernelGGL(k_softcap, dim3(blocks(n, 256)), dim3(256), 0, 0, x, n, cap);
    return launched();
}

}  // extern "C"
