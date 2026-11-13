# Parity gap-closure plan and scorecards (P4)

Date: 2026-11-13. Status: proposal for the [parity track](performance-parity.md)
step P4 (`rv-9zf`). It sequences existing beads and names narrowly scoped
follow-ups. It implements nothing, activates no deferred bead and authorizes no
port. Inputs: [parity baselines](parity-baselines.md) (P3) and the
[reuse audit](reuse-audit.md) (P2).

## Scorecards (per cell and baseline; never aggregated)

Targets follow the contract's milestones. P90 requires a lower 95% bound of at
least 0.90. With the P3 run-to-run spread (under 1%), that corresponds to a
median of about 0.905 × the reference. Parity is a median of at least the
reference with a lower bound of at least 0.98. Advantage is a median of at least
1.05 × the reference with a lower bound above 1.00. Current values are P3
medians. All current pairs are practical-configuration comparisons
(F32 vs Q8_0 CPU activations).

**Decode** (ms/token; lower is better):

| Cell | Baseline | Reference | Rosevine | Ratio | P90 at or below | Saving needed | Parity at or below | Advantage at or below |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d512 | Vulkan | 17.1 | 45.3 | 0.378 | 19.0 | 26.2 | 17.1 | 16.3 |
| d512 | HIP | 20.2 | 45.3 | 0.446 | 22.5 | 22.8 | 20.2 | 19.3 |
| d2048 | Vulkan | 18.2 | 54.4 | 0.334 | 20.2 | 34.2 | 18.2 | 17.3 |
| d2048 | HIP | 20.7 | 54.4 | 0.380 | 23.0 | 31.4 | 20.7 | 19.7 |
| d8192 | Vulkan | 20.6 | 70.9 | 0.290 | 22.8 | 48.1 | 20.6 | 19.6 |
| d8192 | HIP | 24.2 | 70.9 | 0.340 | 26.8 | 44.1 | 24.2 | 23.0 |
| d16384 | Vulkan | 21.9 | 93.5 | 0.235 | 24.3 | 69.2 | 21.9 | 20.9 |
| d16384 | HIP | 28.7 | 93.5 | 0.307 | 31.9 | 61.6 | 28.7 | 27.4 |

**Prefill** (tokens/s; higher is better):

| Cell | Baseline | Reference | Rosevine | Ratio | P90 at or above | Required speedup |
| --- | --- | --- | --- | --- | --- | --- |
| p128 | Vulkan / HIP | 310 / 542 | 46.2 | 0.149 / 0.085 | 279 / 488 | 6.0× / 10.6× |
| p512 | Vulkan / HIP | 818 / 1181 | 46.1 | 0.057 / 0.039 | 736 / 1063 | 16.0× / 23.1× |
| p2048 | Vulkan / HIP | 805 / 1128 | 44.9 | 0.056 / 0.040 | 724 / 1015 | 16.1× / 22.6× |
| p8192 | Vulkan / HIP | 781 / 970 | 42.9 | 0.055 / 0.044 | 703 / 873 | 16.4× / 20.3× |

The TTFT cells stay unsupported. A user-visible harness is a separate
prerequisite (G7).

**Resource scorecard.** Every cell carries the P3 resource regressions:

- 7.8–9.4 CPU core-equivalents against 1.0–2.8 for the reference.
- CPU thermal flags on every Rosevine run.
- Peak VRAM 0.85–1.01× of the reference, which is not a regression.

A speed milestone does not count while the CPU-core and thermal regression is
unexplained. Each record below must report core-equivalents and thermal flags.

## What the d512 budget implies

Vulkan P90 at d512 allows 19.0 ms per token **in total**. P3's sync attribution
at d512 is:

- CPU experts: 21.0 ms
- Attention: 9.7 ms
- Transfers: about 6.5 ms
- GPU GEMV: about 6.5 ms
- Combine and other: about 2.8 ms

Consequences:

- **The R3a.3 objective is insufficient for parity.** Its 1.5 ms per host layer
  (15 ms) alone leaves 4 ms for everything else. P90 at first-N = 10 is
  unreachable without CPU experts at about 0.8 ms per layer or less, or without
  overlapping host-expert work with GPU work. llama.cpp reaches 17.1 ms with the
  same 10 host layers. Whether its CPU-expert time overlaps GPU work is
  **unknown**; measuring that is part of G3.
- **Removing attention's depth growth alone does not reach P90 at any depth.**
  The d512 components other than attention already sum to about 36 ms.
- **The order of work is therefore CPU experts and transfers first for shallow
  cells, and attention for deep cells.** Both are needed for any decode P90.

An illustrative budget for Vulkan P90 at d512 (19 ms) is CPU experts ≤ 8,
attention ≤ 3, GEMV ≤ 3, transfers ≤ 2 and other ≤ 3. It is a planning split
for failure thresholds, not a prediction.

## Ranked optimization records

Each record carries the fields required by the parity amendment. "Actual" stays
pending until a paired P3-style run on the replay adapters measures it. Every
record must be re-measured on the frozen matrix with top-1 agreement and the
S/T/N checks, and must report CPU core-equivalents, thermal flags and peak VRAM.

### G1. Decode attention at depth (existing K3, `rv-e05-7n3.3`)

| Field | Value |
| --- | --- |
| Bottleneck | `k_attention` reads every key serially per output dimension and materializes scores. The `attn` span grows from 9.7 to 57.5 ms, d512 to d16384. |
| Upstream approach | `fattn-vec.cuh` split-K online softmax for D = 256 decode. At D = 512 the pin falls back to `fattn-tile.cuh`. Reuse audit item 4: **B** (adapt), D only with measurement. Direct port rejected for now: template matrix and spills at D = 512. |
| Baseline | P3 decode cells; attention span 9.7 / 57.5 ms |
| Expected | Recover at least 40 ms per token at d16384 (attention depth growth beyond the reference step's depth growth) and a few ms at d512. Scratch also becomes independent of `max_ctx` (F-KV). |
| Failure threshold | Attention span at d16384 above 15 ms, or less than 30 ms of step improvement at d16384, rejects the kernel |
| Correctness | Class N (summation order). Teacher-forced KL in the band; top-1 against both references within the P3 agreement band; window-boundary fixtures; at least 16k depth tested |
| Maintenance | One Rosevine kernel per head dim (256 local, 512 global); moderate |
| Activation | K3 is deferred and needs maintainer activation under its preconditions. Recommended as the first K3 scope. |

### G2. Prefill bottleneck attribution (existing K1, `rv-e05-7n3.1`)

| Field | Value |
| --- | --- |
| Bottleneck | Prefill runs at 4–6% of the reference, flat at 43–46 tok/s. Attribution is **unknown**: R3a.1 spans cover decode only. |
| Upstream approach | `op_offload`: llama.cpp copies host-resident expert weights to the GPU for large batches. Also MMQ for batched Q4_0 (reuse audit items 1–2). |
| Baseline | P3 prefill cells |
| Expected | Not estimated until attributed. The leading hypothesis is per-microbatch host experts plus per-expert unbatched `k_q4_mm`. |
| Failure threshold | K1 completes when prefill spans close at 90% or more and the leading hypotheses are confirmed or rejected |
| Correctness | Class S (instrumentation only) |
| Maintenance | Low: extend R3a.1 spans to batched evaluation |
| Activation | K1 is open and its R3a.2 dependency is closed; it can proceed as diagnostics |

### G3. CPU experts below 1 ms per layer (R3a.4 `rv-e06-k0i.6`, then a follow-up)

| Field | Value |
| --- | --- |
| Bottleneck | 21.0 ms per token (2.1 ms per layer × 10). The span alone exceeds the reference's whole d512 step. It drives 8–9 core-equivalents and the CPU thermal flags. |
| Upstream approach | `ggml_vec_dot_q4_0_q8_0` AVX-512 VNNI (reuse audit item 3: **A** for the R3a.4 flagged variant). Upstream overlap of host experts with GPU work is unknown; measure it with llama.cpp event timing or perf before adopting any scheduling change. |
| Baseline | P3 decode cells; `cpu_experts` span 21.0 ms; the scalar fallback still handles about 23% of rows |
| Expected | R3a.4 int8: the S4 prototype measured 0.75 ms per layer, which would recover about 13 ms per token. Removing the scalar fallback path is a smaller, separate gain. |
| Failure threshold | R3a.4's own: kept only if faster beyond noise with no KL-band regression. For parity planning, above 1.0 ms per layer means first-N = 10 cannot reach P90 without overlap. |
| Correctness | Class N; R3a.4 acceptance (KL against both references, layer-0, selection flips). A Q8_0 activation profile also makes P3 pairs **machinery-parity** comparisons instead of practical ones. |
| Maintenance | Low |
| Activation | R3a.4 done: opt-in `q8_0` measured 0.84 ms per layer at d512, decode ×1.48 and prefill ×2.29 against F32 inside the KL band ([Q8_0 CPU experts](cpu-experts-q8.md)). The parity cells have not been re-run with it. |

### G4. Device-side routing and indexed experts (new follow-up; needs activation)

| Field | Value |
| --- | --- |
| Bottleneck | `exp_meta` 1.5 ms (80 uploads per token); router readback on 20 GPU layers; 1,129 launches per token; 132 blocking calls |
| Upstream approach | `topk-moe.cu` fused softmax, top-k and normalize; `mul_mat_vec_q` with `ids` (reuse audit item 2: **B**) |
| Baseline | P3 decode cells; transfer row about 6.5 ms |
| Expected | 1.5–3.5 ms per token (metadata plus GPU-layer router readback and per-expert launches). Host-layer in/out (about 4.9 ms) is not removed. It enables graph capture later. |
| Failure threshold | Less than 1.5 ms per token of step improvement at d512, or any routing-selection change beyond tie flips |
| Correctness | Class T (exact selection), with selection-flip counting |
| Maintenance | Low to moderate; touches `hip_exec`'s MoE path. Check against ADR 0002 (Proposed) before any SessionPlan refactor. |
| Activation | Not yet a bead. Proposed for maintainer creation after G1 and G3 data, since its gain is smaller. |

### G5. Q4_0 GEMV (existing K2, `rv-e05-7n3.2`)

| Field | Value |
| --- | --- |
| Bottleneck | `mlp` 2.3, `gpu_experts` 3.4, `head` 0.8 ms, plus the router GEMV |
| Upstream approach | `mul_mat_vec_q` with RDNA3 parameters (reuse audit item 1: **A/B**). Exact-shape A/B in both activation profiles first. |
| Baseline | C1 isolated upstream expert GEMV is 0.8 ms (Vulkan) to 2.2 ms (HIP) for 20 layers |
| Expected | A few ms per token at most; small next to G1 and G3 |
| Failure threshold | Exact-shape kernel not faster than `k_q4_mm` at matched numerics; less than 1 ms per token of step gain |
| Correctness | Class N for Q8_1 activations; class S if F32 activations are kept |
| Maintenance | Moderate (RDNA parameter tables) |
| Activation | K2 is deferred; sequence it after G1, G3 and G4 |

### G6. Streams and asynchronous copies (F0 queue handle, D4)

Overlap of host-layer transfers with independent GPU work comes after G4, and
only with a measured positive paired interval (R3a.2 proposal 3). Graph replay
comes after G4 makes the launch sequence static. Both stay evidence-gated.

### G7. User-visible harness (prerequisite for TTFT cells)

This needs a Rosevine admission-to-delivery path with sampling. It is a runtime
feature beyond the parity track's measurement scope; propose it with R4 serving
work, not here.

## Deferred, with reasons

| Technique | Status | Reason |
| --- | --- | --- |
| LES / lookahead prefetch | Deferred | No trace shows a recoverable advantage. S2 found causal lead time weak. Transfers are about 6.5 ms, mostly metadata, not weights. |
| Dynamic expert cache | Deferred (D1) | Placement is static at first-N. S3 found popularity placement fails across domains. |
| Graph replay | Deferred | Launch sequence is data-dependent until G4; launch cost is about 0.8 ms per token (derived) |
| Adaptive scheduling | Deferred | No evidence; G3 must first show whether overlap matters |

None of these may displace G1–G5 without a reviewed trace that shows a compelling
recoverable advantage.

## Remaining unknowns

- Upstream in-situ subsystem timing, especially CPU-expert overlap. G3 needs it
  for ranking against llama.cpp.
- Prefill attribution on both sides (G2).
- Contention sensitivity: no bounded-contention control was run in P3.
- Whether the CPU thermal flags depress Rosevine's P3 numbers. G3 should also
  report a 6-worker variant with flags.
- TTFT and user-visible generation (G7).

## Architecture review boundary

Every record above stays within option 1 of the reuse audit (standalone executor,
selected technique ports) and ADR 0001. None imports GGML code. Any port that
copies upstream code needs its own bead with MIT notice handling. Adopting
selective GGML components, an upstream executor adapter or a public C ABI stream
change would each need a separate reviewed proposal or ADR.

## Sequence

1. **Ready now:** R3a.4 (G3 int8) and K1 prefill attribution (G2).
2. **On maintainer activation:** K3 attention (G1), the largest deep-context gap.
3. **Then:** G4 device routing (new bead) and K2 GEMV (G5).
4. **Then:** G6 overlap and graphs, only with evidence.
5. **After each step:** re-run the P3 campaign (the replay adapters and
   `parity_campaign.py`) and update the scorecards. Gate 4 claims still wait for
   `tuned-k`.
