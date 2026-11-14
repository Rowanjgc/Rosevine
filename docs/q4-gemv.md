# Q4_0 GEMV/GEMM specialization (K2)

Date: 2026-11-13. Bead: rv-e05-7n3.2 (K2), activated by the maintainer on
2026-11-13 after K1 and K3. Parity-plan record
[G5](parity-plan.md#g5-q4_0-gemv-existing-k2-rv-e05-7n32). Status: **criteria
recorded before any K2 measurement**. Raw captures, commands and hashes stay
private.

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
