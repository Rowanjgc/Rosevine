# Q4_0 GEMV/GEMM specialization (K2)

Date: 2026-11-13. Bead: rv-e05-7n3.2 (K2), activated by the maintainer on
2026-11-13 after K1 and K3. Parity-plan record
[G5](parity-plan.md#g5-q4_0-gemv-existing-k2-rv-e05-7n32). Status: **NO-GO** for the
`k_q4_mv` variant (2026-11-13): it is about 7% slower per decode step than
`k_q4_mm`. The criteria section is unchanged from its pre-measurement commit
(`bad85c5`). Raw captures, commands and hashes stay private.

## Baseline

Measured in K1 ([prefill and decode attribution](prefill-attribution.md)):
events-mode device time per decode token at d512, first-N = 10. Bytes are the
Q4_0 weights each span must read.

| Span | Weight bytes | Device ms | Effective GB/s |
| --- | --- | --- | --- |
| `attn` (Q/K/V projections, 30 layers) | 397 MB | 2.30 | 173 |
| `attn_out` | 227 MB | 1.11 | 204 |
| `mlp` (shared expert) | 301 MB | 1.96 | 154 |
| `gpu_experts` (20 layers × 8 experts) | 535 MB | 3.21 | 167 |
| `head` (one 262,144-row GEMV) | 415 MB | 0.81 | 514 |

The spans also contain small non-GEMV kernels, so the rates are lower bounds for
the GEMV kernel itself. The same kernel, `k_q4_mm`, reaches 514 GB/s on the
large output head and 150–205 GB/s on per-layer shapes. If the per-layer shapes
reached the head's rate, about 5.6 ms of device time per token would go. This is
an inference from these figures, not a prediction.

## Candidate

`k_q4_mv` keeps F32 activations, so only the summation order changes
(correctness class N, as for K3). It has one warp per output row; each lane takes
a quarter of a Q4_0 block per step, with one 32-bit weight load and two `float4`
activation loads. Tokens are tiled by 8, as in `k_q4_mm`. It is opt-in and can be
switched at runtime (`--q4 ref|vec`) for in-process paired comparison. A Q8_1
activation profile (`mul_mat_vec_q` style) is a separate, later variant. It
changes the numerics domain and would need its own KL evidence.

## Acceptance criteria (recorded before measurement)

Correctness:

1. Op conformance at every target shape (Q/K/V/O local and global, shared MLP,
   expert gate-up and down, output head; 1, 3, 8 and 64 tokens). Relative L2
   error below 1e-5 against the CPU ops. Relative L2 error against an exact f64
   reference no worse than twice `k_q4_mm`'s.
2. Teacher-forced KL against the frozen CPU and Vulkan-fast references
   (`kld`, 7 chunks, Q8_0 CPU experts, first-N = 10) stays inside the existing
   band (mean KL at most about 0.08, top-1 at least about 92%).

Performance (the bead's thresholds):

3. In-process ABBA pairs from the same KV state (`ab-attention --compare q4`),
   with at least 7 pairs per session over 2 sessions, at depths 512 and 16,384.
   **GO** requires the median paired step ratio (vec / ref) at depth 512 to be
   0.95 or lower, with its 95% bootstrap interval entirely below 1.0. Otherwise
   the outcome is **NO-GO**, and no optimization is claimed.
4. Prefill at P = 2048 must not be slower by more than 2% (point estimates per
   session, as for K3).
5. Report device time per span (events mode) for the changed kernels.

The attention kernel is the K3 default (split) in both arms.

## Results

Rosevine commit `1875039`; HIP, first-N = 10, Q8_0 CPU experts, 12 threads,
batch 64, split attention in both arms, P3 parity fixture. 26 supervised
single-shot processes over 2 sessions (the second in reverse order), all
successful, under the run lock and the COMPUTE window, plus one discarded smoke
run.

| # | Criterion | Result | Verdict |
| --- | --- | --- | --- |
| 1 | Op conformance below 1e-5; f64 error at most 2× `k_q4_mm`'s | All 11 shapes at 1, 3, 8 and 64 tokens pass against the CPU ops. Relative error against exact f64 is 0.94–1.38e-7 for `k_q4_mv` and 1.18–1.38e-7 for `k_q4_mm`. Batch invariance holds; unaligned inputs fall back to `k_q4_mm` bitwise. | Met |
| 2 | KL inside the band | CPU reference: 0.0784, top-1 91.82%. Vulkan-fast: 0.0687, top-1 93.05%. Reference kernel: 0.0764 / 92.44% and 0.0733 / 92.89%. PPL 12.28 → 11.81. | Met against Vulkan-fast. Against CPU, top-1 is 0.2 points under "about 92%". |
| 3 | Step ratio (vec / ref) at d512 at most 0.95, interval below 1.0 | d512: **1.070** (1.069–1.075), 25.56 → 27.33 ms. d16384: 1.063 (1.060–1.066), 31.86 → 33.89 ms. 16 ABBA pairs each over 2 sessions. | **Not met: slower** |
| 4 | Prefill at P = 2048 at most 2% slower | vec/ref 0.988 and 0.991 per session (about 99 tok/s) | Met on point estimates |
| 5 | Device time per span (events, d512, ms per token, mean of 2 sessions) | `attn` 2.31 → 2.75; `attn_out` 1.12 → 1.40; `mlp` 1.96 → 2.31; `gpu_experts` 3.22 → 3.67; `head` 0.81 → 1.17 | Reported: every GEMV span is slower |

Top-1 agreement between the kernels is 1.000 at d512 and 0.9375 at d16384,
as for K3. The PPL and KL shifts (direction depending on the reference) are
summation-order differences of the same size as K3's. They are not evidence that
either kernel is better.

## Outcome

- **NO-GO** under criterion 3. No optimization is claimed. `k_q4_mm` stays the
  default and `--q4 vec` stays available for diagnosis only.
- Likely reasons, from the code rather than measured: each lane issues two 16-bit
  weight loads plus four redundant scale loads per block, and a warp covers only
  8 consecutive blocks (144 bytes) per step, against 32 blocks (576 bytes) for
  `k_q4_mm`. Fewer bytes are in flight per warp. Even the large head, where
  `k_q4_mm` already reaches about 514 GB/s, slows down (0.81 → 1.17 ms), which
  points at the access pattern, not launch overhead.
- The per-layer gap (150–205 GB/s) remains unexplained. Untested candidates are
  multiple rows per warp to raise memory-level parallelism, and a Q8_1
  activation path (`mul_mat_vec_q` style, a different numerics domain). These
  would need new criteria and are not activated.
