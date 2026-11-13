/* SPDX-License-Identifier: GPL-3.0-only
 * Rosevine HIP kernels behind a C ABI (ADR 0001). Correctness-first R2 kernels: simple,
 * one kernel per operation, no tuning. All launches go to the null stream and are ordered;
 * rvh_download and rvh_sync synchronize. Every function returns 0 or a HIP error code, and no
 * exception crosses the ABI. Device pointers come from rvh_alloc and are released by rvh_free.
 * Functions validate scalar arguments; buffer extents are the caller's contract (the Rust wrapper
 * checks every range against its allocation before calling).
 *
 * Layouts match the Rust CPU executor:
 *   Q4_0 row: k/32 blocks of {f16 d; u8 qs[16]}, value j = ((qs[j] & 15) - 8) * d,
 *             value j + 16 = ((qs[j] >> 4) - 8) * d.
 *   Q8_0 KV row: hd/32 blocks of {f16 d; i8 q[32]}; slot s, kv head h at (s * n_kv + h) * row.
 *   Activations are row-major f32 [token][width].
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

int rvh_init(int device);
int rvh_mem_info(size_t *free_bytes, size_t *total_bytes);
int rvh_alloc(size_t bytes, void **out);
int rvh_free(void *ptr);
int rvh_upload(void *dst, const void *src, size_t bytes);
int rvh_download(void *dst, const void *src, size_t bytes);
int rvh_copy(void *dst, const void *src, size_t bytes); /* device to device, ordered */
int rvh_sync(void);
const char *rvh_error_string(int code);

/* out[t] = dequant(w row ids[t]) * scale, k wide. */
int rvh_get_rows_q4(const void *w, const int32_t *ids, float *out, int n, int k, float scale);
/* out[r] = ((x[r] * rms_scale) * post) * weight (weight may be NULL), rows of `width`. In place allowed. */
int rvh_rms_norm(const float *x, float *out, const float *weight, int rows, int width, float eps, float post);
/* y[t][r] = W[r] . x[t]; W is Q4_0 n_out x k; x is n_in x k. */
int rvh_q4_mm(const void *w, const float *x, float *y, int n_out, int k, int n_in);
/* Same for an f32 matrix (the router). */
int rvh_f32_mm(const float *w, const float *x, float *y, int n_out, int k, int n_in);
/* Per head of hd values over n_tok * n_heads heads: RMS norm (weight may be NULL), then NEOX
 * RoPE at position p0 + t when base > 0 (freq factors may be NULL). */
int rvh_head_norm_rope(float *x, int n_tok, int n_heads, int hd, const float *weight, float eps, int p0,
                       float base, const float *freq_factors);
/* Quantize K and V for tokens p0.. into ring slots (pos % cap). */
int rvh_kv_store(const float *k, const float *v, void *kc, void *vc, int n_tok, int n_kv, int hd, int p0, int cap);
/* Causal attention, scale 1.0; window 0 means global, else keys p > pos - window.
 * scores: n_tok * n_head * max_keys floats of scratch. */
int rvh_attention(const float *q, const void *kc, const void *vc, float *out, float *scores, int n_tok, int n_head,
                  int n_kv, int hd, int p0, int cap, int window, int max_keys);
/* out[i] = gelu(g[i]) * u[i] */
int rvh_geglu(const float *g, const float *u, float *out, int n);
/* Rows of 2*ff (gate then up) to rows of ff. */
int rvh_geglu_split(const float *gu, float *out, int rows, int ff);
/* out[i] = a[i] + b[i] */
int rvh_add(const float *a, const float *b, float *out, int n);
/* x[i] = (x[i] + r[i]) * s */
int rvh_add_scale(float *x, const float *r, float s, int n);
/* dst[i] = src[idx[i]], rows of `width`. */
int rvh_gather_rows(const float *src, const int32_t *idx, float *dst, int n, int width);
/* out[t] = sum_j (y[pos[p]] * scale[p]) * weight[p], p = t*used+j, in slot order j. */
int rvh_moe_combine(const float *y, const int32_t *pos, const float *scale, const float *weight, float *out,
                    int n_tok, int used, int width);
/* x = tanh(x / cap) * cap */
int rvh_softcap(float *x, int n, float cap);

#ifdef __cplusplus
}
#endif
