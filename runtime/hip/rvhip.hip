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

// Q4_0 x f32 GEMV/GEMM (K2). One warp per output row, TT tokens per launch row-tile, as k_q4_mm.
// Each lane takes a quarter of a block per step: one 32-bit load of 4 packed bytes (elements
// 4q..4q+3 and 16+4q..16+4q+3) and two float4 activation loads, so a warp reads warpSize/4
// consecutive blocks at once. Needs 16-byte aligned activation rows (checked by the caller).
__global__ void k_q4_mv(const uint8_t *w, const float *x, float *y, int n_out, int k, int n_in) {
    const int warps = blockDim.x / warpSize;
    const int r = blockIdx.x * warps + threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int t0 = blockIdx.y * TT;
    if (r >= n_out) return;
    const int nt = min(TT, n_in - t0);
    const int nq = k / QK * 4;
    const uint8_t *row = w + (size_t)r * (k / QK) * Q4_BLOCK;
    float acc[TT];
    for (int i = 0; i < TT; ++i) acc[i] = 0.f;
    for (int i = lane; i < nq; i += warpSize) {
        const int b = i >> 2, q = i & 3;
        const uint8_t *blk = row + (size_t)b * Q4_BLOCK;
        const float d = f16(blk);
        const uint16_t *p = (const uint16_t *)(blk + 2 + 4 * q);
        const uint32_t qs = (uint32_t)p[0] | ((uint32_t)p[1] << 16);
        float lo[4], hi[4];
        for (int j = 0; j < 4; ++j) {
            lo[j] = (float)((int)((qs >> (8 * j)) & 15u) - 8);
            hi[j] = (float)((int)((qs >> (8 * j + 4)) & 15u) - 8);
        }
        for (int t = 0; t < TT; ++t) {
            if (t >= nt) break;
            const float *xs = x + (size_t)(t0 + t) * k + b * QK + 4 * q;
            const float4 a = *(const float4 *)xs;
            const float4 c = *(const float4 *)(xs + 16);
            acc[t] += d * (lo[0] * a.x + lo[1] * a.y + lo[2] * a.z + lo[3] * a.w + hi[0] * c.x + hi[1] * c.y +
                           hi[2] * c.z + hi[3] * c.w);
        }
    }
    for (int t = 0; t < nt; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)(t0 + t) * n_out + r] = s;
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

// Split-K attention (K3). The keys [lo, pos] of each query are cut into `n_chunks` contiguous
// chunks; block (query-head i, chunk c) runs an online softmax over its chunk in tiles of ATT_TILE
// keys. With one chunk it writes the normalized output; otherwise an unnormalized partial
// (max, sum, acc[hd]) for k_attn_combine. Scores are formed one warp per key with the lanes across
// the head dimension, so K rows are read contiguously; each thread then owns output dimensions
// threadIdx.x and threadIdx.x + ATT_TILE (hd <= 2 * ATT_TILE). Same scale (1.0), mask and Q8_0
// cache as k_attention; only the summation order differs.
constexpr int ATT_TILE = 256;  // keys per tile; equals blockDim.x
__global__ void k_attn_split(const float *q, const uint8_t *kc, const uint8_t *vc, float *out, float *part,
                             int n_head, int n_kv, int hd, int p0, int cap, int window, int n_chunks) {
    __shared__ float sh[32];
    __shared__ float qs[2 * ATT_TILE];
    __shared__ float p[ATT_TILE];
    const int i = blockIdx.x, c = blockIdx.y;
    const int t = i / n_head, h = i % n_head;
    const int g = h / (n_head / n_kv);
    const int pos = p0 + t;
    const int lo = window > 0 ? max(0, pos + 1 - window) : 0;
    const int len = pos - lo + 1;
    const int per = (len + n_chunks - 1) / n_chunks;
    const int k0 = lo + c * per, k1 = min(pos + 1, k0 + per);
    const int row = hd / QK * Q8_BLOCK;
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    const int d0 = threadIdx.x, d1 = threadIdx.x + ATT_TILE;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = q[(size_t)i * hd + d];
    __syncthreads();
    float acc0 = 0.f, acc1 = 0.f, m = -INFINITY, l = 0.f;
    for (int base = k0; base < k1; base += ATT_TILE) {
        const int nt = min(ATT_TILE, k1 - base);
        for (int j = wid; j < nt; j += nw) {
            const uint8_t *kr = kc + ((size_t)((base + j) % cap) * n_kv + g) * row;
            float s = 0.f;
            for (int d = lane; d < hd; d += warpSize) {
                const uint8_t *blk = kr + (d / QK) * Q8_BLOCK;
                s += f16(blk) * (float)(int8_t)blk[2 + d % QK] * qs[d];
            }
            s = warp_sum(s);
            if (lane == 0) p[j] = s;
        }
        __syncthreads();
        const float tm = block_max(threadIdx.x < nt ? p[threadIdx.x] : -INFINITY, sh);
        const float mn = fmaxf(m, tm);
        const float e = threadIdx.x < nt ? expf(p[threadIdx.x] - mn) : 0.f;
        const float ts = block_sum(e, sh);
        if (threadIdx.x < nt) p[threadIdx.x] = e;
        __syncthreads();
        const float corr = expf(m - mn);  // 0 on the first tile, where m is -inf
        l = l * corr + ts;
        acc0 *= corr;
        acc1 *= corr;
        if (d0 < hd) {
            const int bo = (d0 / QK) * Q8_BLOCK, eo = 2 + d0 % QK;
            for (int j = 0; j < nt; ++j) {
                const uint8_t *blk = vc + ((size_t)((base + j) % cap) * n_kv + g) * row + bo;
                acc0 += p[j] * f16(blk) * (float)(int8_t)blk[eo];
            }
        }
        if (d1 < hd) {
            const int bo = (d1 / QK) * Q8_BLOCK, eo = 2 + d1 % QK;
            for (int j = 0; j < nt; ++j) {
                const uint8_t *blk = vc + ((size_t)((base + j) % cap) * n_kv + g) * row + bo;
                acc1 += p[j] * f16(blk) * (float)(int8_t)blk[eo];
            }
        }
        m = mn;
        __syncthreads();  // p is rewritten by the next tile
    }
    if (n_chunks == 1) {
        const float inv = 1.0f / l;
        if (d0 < hd) out[(size_t)i * hd + d0] = acc0 * inv;
        if (d1 < hd) out[(size_t)i * hd + d1] = acc1 * inv;
        return;
    }
    float *pp = part + ((size_t)i * n_chunks + c) * (hd + 2);
    if (threadIdx.x == 0) {
        pp[0] = m;
        pp[1] = l;
    }
    if (d0 < hd) pp[2 + d0] = acc0;
    if (d1 < hd) pp[2 + d1] = acc1;
}

// Merge the per-chunk partials of each query-head: out = sum_c e^(m_c - M) acc_c / sum_c e^(m_c - M) l_c.
// Empty chunks carry m = -inf and l = 0 and contribute nothing.
__global__ void k_attn_combine(const float *part, float *out, int hd, int n_chunks) {
    const int i = blockIdx.x;
    const float *pp = part + (size_t)i * n_chunks * (hd + 2);
    float mx = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) mx = fmaxf(mx, pp[(size_t)c * (hd + 2)]);
    float den = 0.f;
    for (int c = 0; c < n_chunks; ++c) den += expf(pp[(size_t)c * (hd + 2)] - mx) * pp[(size_t)c * (hd + 2) + 1];
    const float inv = 1.0f / den;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int c = 0; c < n_chunks; ++c) {
            const float *q = pp + (size_t)c * (hd + 2);
            o += expf(q[0] - mx) * q[2 + d];
        }
        out[(size_t)i * hd + d] = o * inv;
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

int rvh_event_create(void **out) {
    if (!out) return (int)hipErrorInvalidValue;
    hipEvent_t e = nullptr;
    hipError_t r = hipEventCreate(&e);
    *out = (void *)e;
    return (int)r;
}

int rvh_event_destroy(void *ev) { return ev ? (int)hipEventDestroy((hipEvent_t)ev) : 0; }

int rvh_event_record(void *ev) {
    if (!ev) return (int)hipErrorInvalidValue;
    return (int)hipEventRecord((hipEvent_t)ev, 0);
}

int rvh_event_sync(void *ev) {
    if (!ev) return (int)hipErrorInvalidValue;
    return (int)hipEventSynchronize((hipEvent_t)ev);
}

int rvh_event_elapsed(void *start, void *end, float *ms) {
    if (!start || !end || !ms) return (int)hipErrorInvalidValue;
    return (int)hipEventElapsedTime(ms, (hipEvent_t)start, (hipEvent_t)end);
}

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

int rvh_q4_mv(const void *w, const float *x, float *y, int n_out, int k, int n_in) {
    CHECK_DIMS(w && x && y && n_out > 0 && n_in > 0 && k > 0 && k % QK == 0 && ((uintptr_t)x & 15) == 0 &&
               ((uintptr_t)w & 1) == 0);
    const int warps = 256 / g_warp;
    hipLaunchKernelGGL(k_q4_mv, dim3(blocks(n_out, warps), blocks(n_in, TT)), dim3(256), 0, 0, (const uint8_t *)w, x,
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

int rvh_attn_split(const float *q, const void *kc, const void *vc, float *out, float *part, int n_tok, int n_head,
                   int n_kv, int hd, int p0, int cap, int window, int n_chunks) {
    CHECK_DIMS(q && kc && vc && out && n_tok > 0 && n_kv > 0 && n_head % n_kv == 0 && hd % QK == 0 &&
               hd <= 2 * ATT_TILE && p0 >= 0 && cap > 0 && window >= 0 && n_chunks > 0 && (n_chunks == 1 || part));
    // Every query's key range must fit the cache ring.
    const int longest = window > 0 ? (p0 + n_tok < window ? p0 + n_tok : window) : p0 + n_tok;
    CHECK_DIMS(longest <= cap);
    hipLaunchKernelGGL(k_attn_split, dim3(n_tok * n_head, n_chunks), dim3(ATT_TILE), 0, 0, q, (const uint8_t *)kc,
                       (const uint8_t *)vc, out, part, n_head, n_kv, hd, p0, cap, window, n_chunks);
    return launched();
}

int rvh_attn_combine(const float *part, float *out, int rows, int hd, int n_chunks) {
    CHECK_DIMS(part && out && rows > 0 && hd > 0 && n_chunks > 1);
    hipLaunchKernelGGL(k_attn_combine, dim3(rows), dim3(256), 0, 0, part, out, hd, n_chunks);
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
