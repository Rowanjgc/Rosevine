# Split-K decode attention (K3)

Date: 2026-11-13. Bead: rv-e05-7n3.3 (K3), activated by the maintainer on
2026-11-13 after K1 ([prefill and decode attribution](prefill-attribution.md)).
Scope: parity-plan record [G1](parity-plan.md#g1-decode-attention-at-depth-existing-k3-rv-e05-7n33).
Status: **measured; performance criteria met, two pre-recorded correctness
criteria not met** (2b and 3). A diagnostic finds no defect. The kernel stays
opt-in, and adopting it is a maintainer decision. The criteria section is
unchanged from its pre-measurement commit (`692b7df`). Raw captures, commands
and hashes stay private.

## Change

- `rvhip.hip`: `k_attn_split` runs an online softmax over 256-key tiles of one
  contiguous chunk of each query's keys. One warp forms each score, with the lanes
  across the head dimension, so K rows are read contiguously. `k_attn_combine`
  merges the per-chunk partials (max, sum, unnormalized output). With one chunk
  the first kernel writes the normalized output and the combine is skipped.
- `hip::attention_chunks`: chunks per query-head are the smaller of
  `ceil(keys / 64)` and `1024 / (tokens × heads)`. A decode step at depth gets
  64 chunks per head (1,024 blocks). A 64-token prefill microbatch gets one.
- Scratch is `max(tokens × heads, 1024) × (head_dim + 2)` floats, independent of
  `max_ctx`. The reference kernel needs `max_batch × heads × max_ctx` floats.
- Same semantics as `k_attention`: scale 1.0, causal mask, sliding window on
  local layers, Q8_0 K/V cache ring. Only the summation order differs
  (correctness class N).
- Opt-in: `HipConfig::attention` (`--attention ref|split`). The reference
  kernel stays the default until the gate review.

## Acceptance criteria (recorded before measurement)

Correctness, class N:

1. Op conformance (`split_attention_matches_cpu_and_the_reference_kernel`): relative
   L2 error below 1e-5 against the CPU ops and against `k_attention`. Cases cover a
   wrapped local window, global D = 512 decode at 5,000 keys (64 chunks), 8- and
   64-token batches, a short context and a ragged final chunk. Out-of-ring key ranges
   are rejected.
2. Teacher-forced KL against the frozen CPU and Vulkan-fast references
   (`kld`, 7 chunks, Q8_0 CPU experts, first-N = 10). Both runs must stay inside the
   existing band (mean KL at most about 0.08, top-1 at least about 92%), and mean KL
   must move by at most 0.002 from the reference kernel's.
3. Decode top-1 on the parity fixture at depths 512 and 16,384: agreement with the
   reference kernel of at least 0.95 per depth. No threshold is relaxed after results.

Performance (parity-plan G1 thresholds, set before this work):

4. **Reject** if the `attn_core` span at depth 16,384 (sync mode, all 30 layers)
   stays above 15 ms per token, or if the uninstrumented decode step improves by less
   than 30 ms per token at depth 16,384.
5. Paired design: one process per session prefills the fixture once with the
   reference kernel, then alternates 32-step decode blocks between kernels in ABBA
   order from the same KV state (`rosevine-inspect ab-attention`). At least 7 pairs
   per session and 2 sessions. Report the median paired difference with a 95%
   bootstrap interval; the interval must exclude zero.
6. No regression beyond noise elsewhere: decode at depth 512 and prefill at 2,048
   tokens must not be slower by more than 2% (upper 95% bound of the paired ratio).
7. Report scratch bytes, peak VRAM, thermal flags and CPU core-equivalents.

A kernel that fails 4 is recorded as NO-GO for K3 and stays opt-in or is removed;
the reference kernel remains correct and available either way.

## Results

Design as in criterion 5. Rosevine commit `47de6f4`; HIP, first-N = 10, Q8_0 CPU
experts, 12 threads, batch 64, P3 parity fixture. 18 supervised single-shot
processes over 2 sessions, all successful, under the run lock and the COMPUTE window.

| # | Criterion | Result | Verdict |
| --- | --- | --- | --- |
| 1 | Op conformance below 1e-5 | All six cases pass against the CPU ops and `k_attention`; out-of-ring ranges rejected | Met |
| 2a | KL inside the band | CPU reference: 0.0764, top-1 92.44%. Vulkan-fast: 0.0733, top-1 92.89%. | Met |
| 2b | Mean KL moves by at most 0.002 | +0.0055 against both references (reference kernel: 0.0709 and 0.0679). PPL 12.23 → 12.28. | **Not met** |
| 3 | Decode top-1 agreement at least 0.95 | d512: 1.000. d16384: 0.9375 (480 of 512 steps). | **Not met at d16384** |
| 4 | `attn_core` at d16384 at most 15 ms; step gain at least 30 ms | `attn_core` 7.8 ms (global layers 5.7), from 54.7 ms. Step 78.7 → 31.7 ms, a gain of 47.0 ms. | Met |
| 5 | Paired 95% interval excludes zero | d16384: gain 47.0 ms (46.7–47.1), ratio 0.403. d512: gain 4.4 ms (4.35–4.43), ratio 0.854. 16 ABBA pairs each. | Met |
| 6 | Other regressions at most 2% | d512 decode is faster. Prefill at P = 2048: split/ref 0.992 and 0.990 in the two sessions (about 100 tok/s). Two sessions are too few for an interval, so only point estimates exist. | Met on point estimates |
| 7 | Resources | Attention scratch at batch 64 is 2.1 MB, independent of context; the reference kernel needs 4 MiB per 1,024 tokens of `max_ctx` (64 MiB at 16k). Thermal flags and core-equivalents are in the private telemetry and were not analysed here. | Partly reported |

Each kernel is bitwise self-consistent across its 8 blocks per session, so the
disagreements are between the kernels, not run-to-run noise.

### Diagnostic: defect or summation order?

`attention_kernels_against_f64_in_the_peaked_regime` compares both kernels with
exact f64 attention over the same Q8_0 caches. The setup is global layer shape
(16 heads, 2 KV heads, D = 512) at depth 16,384, with query scales from
near-uniform (0.05) to sharply peaked (2.0) score distributions.

| Query scale | Reference kernel rel. error | Split kernel rel. error |
| --- | --- | --- |
| 0.05 | 2.3e-6 | 3.6e-7 |
| 0.2 | 2.4e-6 | 4.2e-7 |
| 0.5 | 1.9e-6 | 6.3e-7 |
| 1.0 | 1.1e-6 | 6.5e-7 |
| 2.0 | 1.2e-6 | 7.3e-7 |

The split kernel is 1.6–6.5× **closer** to exact attention than the reference
kernel. A likely reason, from the code rather than measured: its tiled partial sums replace the
reference kernel's single serial value accumulation over all keys. The KL shift and the d16384 top-1 flips are
therefore differences between two f32 approximations. The split kernel is the
more accurate one. For scale, engine-to-engine top-1 agreement at the same depths
in P3 was 0.92–0.99, including pinned llama.cpp HIP against Vulkan.

This explains the failures; it does not convert them into passes. Criteria 2b and 3
were set before measurement and are reported as not met.

## Outcome

- **Performance:** GO evidence. Decode at depth 16,384 is 2.5× faster per step.
  Decode at 512 is 15% faster. `attn_core` is well under the 15 ms threshold.
- **Correctness:**
  - Inside the project's KL band against both frozen references.
  - Outside the tighter kernel-to-kernel criteria recorded for K3.
  - No defect found; the new kernel is closer to exact.
- The kernel stays **opt-in** (`--attention split`). Making it the default, and
  whether criteria 2b and 3 should be judged against exact attention rather than
  the old kernel, are decisions for the maintainer at Gate 4.
