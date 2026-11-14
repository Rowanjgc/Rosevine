# Split-K decode attention (K3)

Date: 2026-11-13. Bead: rv-e05-7n3.3 (K3), activated by the maintainer on
2026-11-13 after K1 ([prefill and decode attribution](prefill-attribution.md)).
Scope: parity-plan record [G1](parity-plan.md#g1-decode-attention-at-depth-existing-k3-rv-e05-7n33).
Status: **criteria recorded before any K3 measurement**. Results follow in the
sections below once measured. Raw captures, commands and hashes stay private.

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
