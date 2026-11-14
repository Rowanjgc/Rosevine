# Prefill and decode attribution (K1)

Date: 2026-11-13. Bead: rv-e05-7n3.1 (K1). Parity-plan record
[G2](parity-plan.md#g2-prefill-bottleneck-attribution-existing-k1-rv-e05-7n31).
Status: **prefill and target-only decode attributed; MTP verification not
covered** (it waits for S1, rv-e07-ktl.1, by maintainer decision). These are
untuned-Rosevine diagnostics, not Gate 4 claims. Raw captures, commands, hashes
and per-run telemetry stay private.

## Instrumentation change

- `replay --instr-prefill MODE` gives prefill microbatches the same R3a.1 span
  accounting as decode. `--instr-per-layer` adds the per-layer table.
- The former `attn` span is split in three:
  - `attn`: norm, Q/K/V projections, head norms, RoPE.
  - `attn_core`: KV store and attention kernel, the only depth-dependent work.
  - `attn_out`: output projection, post-attention norm, residual.
- Flag-off logits are bitwise identical to `main` on the synthetic GPU fixture.
  All three modes match flag-off exactly, the hand-counted device calls still
  match, and `attn_core` holds exactly two launches per layer.
- Reducer: [`scripts/research/analyze_k1.py`](../scripts/research/analyze_k1.py).

## Design

| Item | Value |
| --- | --- |
| Rosevine | Commit `7a709ab`; HIP, 12 threads, batch 64, Q8_0 KV; first-N = 10 CPU expert layers unless stated |
| Fixture | The P3 parity fixture (16,512 real-text token IDs); prefill of the first P IDs, then D frozen IDs decoded one at a time |
| Cells | P ∈ {512, 2048} × CPU-expert activation {F32, Q8_0}; P = 16384 with Q8_0; controls at P = 2048, Q8_0: batch 256, first-N = 2, first-N = 0 (all experts on the GPU) |
| Modes | off, sync and events per cell. Sync gives exclusive, non-overlapping times; events give device time without extra syncs. |
| Sessions | 2, the second in reverse order; 40 successful supervised single-shot processes, plus 3 retained failed attempts and 2 smoke runs (one discarded: it used a stale binary). Warmup plus 1 measured repetition, except P = 16384 (1 cold repetition) |
| Supervision | `gpu_campaign` command sessions under the run lock and the COMPUTE window; production router idle and protected |

Retained failures:

- The first first-N = 2 attempt stopped on the supervisor's 1 GiB free-VRAM
  reserve. The rule audit had already disabled that stop for these sessions; the
  campaign script had copied an older 1 GiB setting. The cell was rerun with the
  supervisor reserve at 0. The runtime's own reserve and the router-growth stop
  were unchanged.
- Both first-N = 0 cells requested Q8_0 activations with no CPU expert layers,
  which Rosevine rejects. They were rerun with F32, which no CPU expert uses at N = 0.

## Acceptance evidence

| Criterion | Result |
| --- | --- |
| At least 90% of loop time in non-overlapping spans or explicit gaps | Span closure 1.0000 in every phase; closure against the harness's own loop timer 0.9998–1.0000 |
| Profiler does not change correctness | Prefill top-1 and every decode top-1 identical between off and each instrumented mode, in all 24 pairs |
| Instrumented vs uninstrumented, two sessions | Prefill overhead −0.3 to +2.0%. Decode overhead +3.8 to +10.3% in sync mode and +1.2 to +3.9% in events mode. Sessions agree within about 1% per cell. |
| Inference vs direct counters | Spans are direct timings. Events-mode device time corroborates the sync spans (`gpu_experts` 0.94 ms per prefill token in both). Byte and launch counts come from the call counters. The roofline remarks below are inference. |

## Results (sync mode, ms per token, mean of the two sessions)

Prefill rates are uninstrumented.

| Cell | Prefill tok/s | Total | CPU experts | `attn_core` (global) | GPU experts | Projections | Transfers | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prefill P = 512, F32 | 46.2 | 21.8 | 19.3 | 0.2 (0.06) | 1.0 | 1.0 | 0.25 | 0.1 |
| Prefill P = 2048, F32 | 45.0 | 22.4 | 19.4 | 0.7 (0.23) | 0.9 | 1.0 | 0.3 | 0.1 |
| Prefill P = 512, Q8_0 | 105.6 | 9.5 | 7.1 | 0.2 (0.06) | 0.9 | 1.0 | 0.2 | 0.1 |
| Prefill P = 2048, Q8_0 | 100.5 | 10.0 | 7.1 | 0.7 (0.23) | 0.9 | 1.0 | 0.2 | 0.1 |
| Prefill P = 16384, Q8_0 | 85.0 | 11.8 | 7.1 | 2.5 (1.94) | 0.9 | 1.0 | 0.2 | 0.1 |
| P = 2048, Q8_0, batch 256 | 105.4 | 9.5 | 7.0 | 0.6 (0.21) | 0.7 | 1.0 | 0.1 | 0.1 |
| P = 2048, Q8_0, first-N = 2 | 216.8 | 4.7 | 1.4 | 0.7 (0.24) | 1.3 | 1.0 | 0.1 | 0.1 |
| P = 2048, first-N = 0 | 305.1 | 3.3 | 0 | 0.7 (0.23) | 1.4 | 1.0 | 0.1 | 0.1 |

"Projections" is `attn`, `attn_out`, `mlp` and `head`. "Transfers" is `exp_in`,
`exp_out`, `exp_meta`, `router` (which includes the router GEMV) and `logits`.

| Decode (Q8_0) | Step, off (ms) | Total | CPU experts | `attn_core` (global / local) | GPU experts | Projections | Transfers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d512 | 30.2 | 33.7 | 8.6 | 6.4 (1.5 / 4.8) | 3.4 | 7.2 | 6.6 |
| d2048 | 39.3 | 42.4 | 8.4 | 15.2 (6.7 / 8.5) | 3.5 | 7.3 | 6.6 |
| d16384 | 78.5 | 81.9 | 8.6 | 54.7 (46.3 / 8.4) | 3.4 | 7.2 | 6.6 |

## Findings

**Prefill hypothesis 1 (host experts on every microbatch) is confirmed.**

- With first-N = 10, CPU experts take 71% of a Q8_0 prefill token (7.1 of
  10.0 ms) and 86% of an F32 one.
- Moving all experts to the GPU (first-N = 0) triples the Q8_0 prefill rate
  (100.5 to 305 tok/s); first-N = 2 more than doubles it.
- Batching does not amortize the CPU work:
  - The cost is about 89 µs per token-expert pair in Q8_0 prefill, against
    107 µs in decode.
  - A 4× larger microbatch changes CPU-expert time per token by 1.5%.
  - Reason (from the code, not timed separately): each expert's weight row is
    reused from cache across its group, but every (row, input) dot product
    decodes the nibbles again. Cost is therefore linear in pairs.
- At 0.7 ms per layer for 64 tokens per microbatch, the CPU path does about
  68 G multiply-adds per second (inference from the shapes). That is a small
  fraction of the six cores' integer SIMD peak.

**Prefill hypothesis 2 (per-expert unbatched `k_q4_mm`) is secondary now and
leads once host experts are gone.**

- At first-N = 10, GPU experts are 9% of prefill time.
- At first-N = 0, the GPU side is 3.3 ms per token: GPU experts 1.4,
  projections 1.0, attention 0.7. GPU experts and projections are batched Q4_0
  GEMM work.
- Pinned llama.cpp HIP prefills 1,128 tok/s at P = 2048 even with 10 host-expert
  layers (P3). That beats Rosevine's all-GPU 305 tok/s by 3.7×, so the GEMM gap is
  real, but it only matters after hypothesis 1 is addressed.

**Decode depth growth is the attention kernel in the global layers.**

- `attn_core` grows from 6.4 to 54.7 ms per token between d512 and d16384.
- The five global (D = 512) layers account for 46.3 ms of that at d16384.
- Local layers stop growing at the 1,024-token window (8.4–8.5 ms from d2048).
- The rest of the step is flat with depth (27 ms without `attn_core`).

## Ranked candidates

| Rank | Candidate | Evidence | Expected benefit | Failure threshold |
| --- | --- | --- | --- | --- |
| 1 | **K3 split-K decode attention**, global layers first (activated) | 46.3 of 54.7 ms `attn_core` at d16384 | At least 30 ms per token at d16384 | Recorded before any K3 measurement in [split-K attention](split-k-attention.md): `attn_core` above 15 ms at d16384, or less than 30 ms step gain |
| 2 | **Prefill host-expert offload** (op_offload analogue): run host-layer experts on the GPU for large microbatches, streaming weights | First-N = 0 upper bound ×3.0 prefill. Streaming 10 layers (about 4.3 GB) per microbatch costs an estimated 2.8 ms per token at batch 64, or 0.35 ms at batch 512 (assuming about 24 GB/s). | ×1.6 at batch 64 to ×2.8 at batch 512 (estimates) | Less than ×1.3 prefill at P = 2048 with no decode regression. Needs a bead and a check against ADR 0002, which is still Proposed. Not activated. |
| 3 | **Batched CPU expert kernel**: decode each weight block once for all inputs of an expert group | Linear per-pair cost and low SIMD utilization | Not estimated; bounded by the 7.1 ms per token CPU share | Less than 1.5× CPU-expert time per prefill token. Not activated. |
| 4 | **K2 batched Q4_0 GEMM** (MMQ-style) | 2.4 of 3.3 ms per token at first-N = 0; 3.7× gap to llama.cpp | Matters only after rank 2 or 3 | K2's own criteria. Stays deferred by maintainer decision. |

## Not covered

- **MTP verification widths.** These wait for S1, which defines the
  verification rows and positions.
- **Upstream in-situ prefill attribution.** llama.cpp exposes no comparable
  spans. P3 recorded only whole-prefill rates.
- **CPU thermal state.** Thermal flags are recorded per session but not analysed
  here. Both sessions agree within 1%, so they did not drive the results.
