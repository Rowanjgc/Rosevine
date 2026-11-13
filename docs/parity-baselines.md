# Parity baselines and subsystem gaps (P3)

Date: 2026-11-13. Status: measured diagnostics for the [parity track](performance-parity.md)
step P3 (`rv-tkd`), under the [parity contract](parity-contract.md). These are
untuned-Rosevine diagnostics, **not Gate 4 claims**. Raw captures, commands,
hashes and per-run telemetry stay private. Publishing this summary outside the
repository needs human review.

## Setup

| Item | Value |
| --- | --- |
| Hardware | RX 7800 XT 16 GiB, Ryzen 5 7600X, Fedora 43, ROCm 6.4; COMPUTE power window for every run |
| Model | The pinned Gemma 4 26B A4B QAT target (same SHA-256 as R2/R3a) |
| Rosevine | Commit `92bbfa9` (R3a.3 F32 CPU experts, `replay` adapter); HIP, first-N = 10 CPU expert layers, batch 64, 12 threads, Q8_0 KV |
| Reference | llama.cpp at `cd959bae`, HIP and Vulkan builds; same model and placement (`--n-cpu-moe 10` pattern), Q8_0 KV, FA on, CPU repack on, batch/ubatch 512, 6 threads |
| Fixture | 16,512 real-text token IDs: the Epic 2 corpus calibration split, then the other splits, sorted by document ID, one BOS. Same IDs in every arm (binding hash checked). |
| Design | 7 sessions × 3 arms × 5 prompt sizes = 105 supervised single-shot processes, arm order rotated per session. P ≤ 2048: warmup plus 1 measured repetition. P ∈ {8192, 16384}: 1 cold repetition in every arm. All 105 succeeded. |
| Statistics | Paired per session; ratio = Rosevine / reference (larger is better); median with 95% percentile bootstrap (10,000 resamples, seed 20261113) |

**Fixture correction.** The first smoke run used an Epic 2 long-context
workload. That workload turned out to be mostly `" x x x"` padding, and all three
arms agreed on only about 25% of top-1 tokens. The run was set aside, and the
real-text fixture replaced it. Agreement then rose to 93–95%.

**Feasibility.** The contract gates depths beyond 1,030 positions. A rule was
recorded before the runs: Rosevine must complete, top-1 agreement with llama.cpp
HIP must be at least 0.85, and it must be no more than 0.05 below the
HIP-vs-Vulkan agreement at the same depth. All three gated depths passed:

| Depth | Rosevine / HIP | HIP / Vulkan |
| --- | --- | --- |
| 2,048 | 0.945 | 0.938 |
| 8,192 | 0.992 | 0.969 |
| 16,384 | 0.922 | 0.930 |

The five gated decode and prefill cells moved to supported through recorded
matrix transitions. The TTFT cells remain unsupported, because Rosevine has no
delivery path.

## Results

Every pair is a **practical-configuration** comparison: CPU-expert activations
are F32 in Rosevine and Q8_0 in llama.cpp (D3). No cell counts toward a parity
milestone, and no cell would reach P90 anyway. The intervals are narrow because
run-to-run spread was below 1% in every arm.

| Cell | Vulkan tok/s | HIP tok/s | Rosevine tok/s | vs Vulkan (95% CI) | vs HIP (95% CI) | Top-1 vs Vk / HIP |
| --- | --- | --- | --- | --- | --- | --- |
| decode-d512 | 58.35 | 49.40 | 22.09 | 0.378 (0.378–0.380) | 0.446 (0.445–0.449) | 0.938 / 0.930 |
| decode-d2048 | 54.94 | 48.33 | 18.38 | 0.334 (0.334–0.335) | 0.380 (0.377–0.381) | 0.953 / 0.945 |
| decode-d8192 | 48.63 | 41.39 | 14.10 | 0.290 (0.290–0.290) | 0.340 (0.340–0.342) | 0.977 / 0.992 |
| decode-d16384 | 45.66 | 34.79 | 10.69 | 0.235 (0.234–0.235) | 0.307 (0.307–0.309) | 0.945 / 0.922 |
| prefill-p128 | 310 | 542 | 46.2 | 0.149 (0.147–0.150) | 0.085 (0.085–0.086) | 7/7 sessions same prefill argmax (both) |
| prefill-p512 | 818 | 1181 | 46.1 | 0.057 (0.056–0.057) | 0.039 (0.039–0.039) | 0.938 / 0.930 |
| prefill-p2048 | 805 | 1128 | 44.9 | 0.056 | 0.040 | 0.953 / 0.945 |
| prefill-p8192 | 781 | 970 | 42.9 | 0.055 | 0.044 | 0.977 / 0.992 |

Milestones over the full 12-cell denominator per baseline: P90 0/12, operational
parity 0/12, advantage 0/12. Four cells per baseline are unsupported (TTFT).

**Resources** (median per-pair ratios, Rosevine / reference, per process):

| Resource | Result |
| --- | --- |
| Peak VRAM over the idle desktop | 0.85–1.01. Rosevine uses up to 15% less at depth. |
| High-water RSS | 0.87–1.00 |
| CPU core-equivalents | 7.8–9.4 for Rosevine against 1.0–2.8 for llama.cpp: a **large regression**. R3a.3's 12 workers plus the host routing loop keep about 8 cores busy, where llama.cpp uses 6 threads at partial duty. |
| CPU temperature | About 53% of Rosevine telemetry samples crossed a thermal flag: CCD up to 100.8 °C and Tctl up to 95.5 °C. No sustained stop fired, but single CCD samples exceeded the 95 °C rated maximum. llama.cpp arms stayed at or below 93.4 °C (CCD). R3a.3 recorded the same thermal pattern. Throttling may depress Rosevine's numbers, but it cannot move ratios this far from 1. |

## Attribution (Rosevine decode, R3a.1 spans)

There are 3 sessions per mode at depths 512 and 16,384, with 128 steps each.
Sync mode synchronizes after every span and gives exclusive, non-overlapping
times with closure 1.000. It adds 4.6% (d512) and 2.1% (d16384) to the
uninstrumented campaign step of 45.2 and 93.5 ms. Event mode adds under 2% and
gives device time per span. Median ms/token in sync mode:

| Category (contract row) | d512 | d16384 | Span(s) and notes |
| --- | --- | --- | --- |
| CPU experts | 21.0 | 21.5 | `cpu_experts`, 10 layers: 2.1 ms/layer, against R3a.3's 1.5 ms objective |
| Attention | 9.7 | 57.5 | `attn`, including Q/K/V/O projections, norms, RoPE and the KV store. Device time 9.4 / 57.1 ms. Growth with depth is the attention kernel. |
| GPU GEMV/GEMM | 6.5 + part of router | 6.5 + part of router | `mlp` 2.3, `gpu_experts` 3.4, `head` 0.8; the router gate GEMV is inside `router` (2.1, which also holds the router readback) |
| H2D/D2H | ≈ 6.5 | ≈ 6.5 | `exp_out` 3.9 (host-expert output upload), `exp_meta` 1.5 (4 metadata uploads × 20 layers), `exp_in` 1.0, logits 0.12, plus the router readback inside `router` |
| Synchronization | charged to spans | charged to spans | 132 blocking calls per token. In event mode, host time waiting at the router is 16.0 / 63.8 ms; that is GPU work being waited on, not extra work. |
| Host launch | ≈ 0.8 (derived) | ≈ 0.8 (derived) | 1,129 launches per token × 0.7 µs (C3 submit cost); overlaps the spans above, not added |
| Quantization/dequantization | not separable | not separable | Fused into `k_q4_mm`, `k_attention` and the KV store; no standalone ops |
| Allocation/buffers | 0 | 0 | No allocation during evaluation (R2 invariant); counters show none |
| Other | 1.5 | 1.5 | `combine` 1.3, `select`, `embed`, `outside` |

**Upstream side.** llama.cpp exposes no comparable per-subsystem in-situ timing,
so in-situ upstream rows are **unknown**. They are not inferred by subtraction.
The available upstream evidence is:

- Whole-step times: 17.1 ms (Vulkan) and 20.2 ms (HIP) at d512; 21.9 / 28.7 ms
  at d16384.
- C1's isolated exact-shape op timings at the same pin and GPU, with per-op
  synchronization: expert gate+up+down at 1 token is 41.1 µs per layer (Vulkan)
  and 107.5 µs (HIP); the output head is 721 / 1,270 µs.
- Prefill attribution is unknown on both sides: R3a.1 instruments decode only.

## Ranked recoverable gaps

Rankings use the critical-path ms above. Only same-workload differences are
compared; no unrelated benchmarks are subtracted.

1. **Attention at depth (decode).**
   - Rosevine's step grows by 48.3 ms from d512 to d16384, and its attention
     span grows by 47.8 ms of that. Vulkan's whole step grows by 4.8 ms over the
     same depth, and HIP's by 8.5 ms.
   - At least about 40 ms per token at d16384 is recoverable, and the gap grows
     with depth. It is the largest gap at depth ≥ 8k.
   - Confidence is high: the attention span's depth growth tracks the step's
     depth growth, and the reference's own depth growth is small.
2. **Prefill (all sizes).**
   - Rosevine reaches 4–6% of the reference rate from 512 tokens up (8–15% at
     128 tokens, where the reference is far from its peak), flat at 43–46 tok/s.
   - Attribution is unknown. Two leading hypotheses need prefill spans before
     any fix is ranked. First, CPU experts run on the host for every 64-token
     microbatch; llama.cpp offloads host experts to the GPU for large batches
     (`op_offload`). Second, Rosevine's unbatched per-expert `k_q4_mm` against
     upstream MMQ.
   - The gap is the largest in relative terms, but its subsystem split is
     unproven.
3. **CPU experts (decode).**
   - 21 ms per token, which is more than llama.cpp's whole d512 step (17–20 ms),
     and llama.cpp's step includes its own 10 CPU-expert layers.
   - Lower bound of recoverable time: about 4 ms, the amount by which Rosevine's
     CPU expert span alone (21.0 ms) exceeds llama.cpp's whole Vulkan step
     (17.1 ms). Reaching the
     existing 1.5 ms/layer objective would recover about 6 ms. The Q8_0 int8 path
     (R3a.4) is the matching upstream technique.
   - This gap also drives the CPU-core and thermal regression.
4. **Host-device traffic and metadata.**
   - About 6.5 ms per token. Device-side routing (reuse audit item 2) would
     remove `exp_meta` (1.5 ms) and the GPU-layer router readbacks.
   - Host-layer input and output transfers (about 4.9 ms) are inherent to CPU
     placement. llama.cpp pays an unknown equivalent cost.
5. **GPU GEMV/GEMM.** About 6.5 ms plus the router GEMV. Upstream isolated
   expert time is 0.8 ms (Vulkan) to 2.2 ms (HIP) for 20 layers, against
   Rosevine's 3.4 ms `gpu_experts` span. That puts an upper bound of a few ms on
   this gap, which is small next to items 1–3.

## Acceptance and limits

- **Subsystem coverage:**
  - All eight subsystem rows are measured or explicitly unknown or not
    separable.
  - Rosevine closure is 1.000 (≥ 90% required), and instrumentation overhead is
    calibrated against the uninstrumented campaign.
  - Timing families are matched (engine-only replay in all arms).
  - Correctness is reported per cell, and paired intervals are computed.
- **Cold cells:** the d8192, d16384 and p8192 cells use one cold repetition per
  process in all arms (`warm_state = cold`, matched).
- **Thermal flags:** every Rosevine run carries thermal flags. There are no
  unflagged Rosevine pairs.
- **Not covered:** a bounded-contention control was not run (idle host only), so
  the contention sensitivity of these ratios is unknown. The user-visible family
  and TTFT remain unsupported.
- **No upstream in-situ subsystem attribution.** Ranking items 2–5 against
  upstream needs either upstream instrumentation or the exact-shape A/B tests
  that the [reuse audit](reuse-audit.md) proposes.
