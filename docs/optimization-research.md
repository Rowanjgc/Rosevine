# Model-specific optimization research

Date: 2026-11-13, with follow-up scrutiny the same day (see
[Follow-up scrutiny](#follow-up-scrutiny-2026-11-13), which **revises F1, F3 and F5**).
Status: research report for review. Nothing here is accepted architecture. It expands the [research roadmap](research-roadmap.md) and does not
replace it. [ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md) stays as
accepted, and Gate 3 (rv-e04-398.3) still needs its formal decision.

Companion documents:
- [optimization framework](optimization-framework.md): architecture boundaries;
- [ADR 0002 (Proposed)](adr/0002-expert-residency-and-scheduling.md): expert residency
  and scheduling;
- [validation plan](optimization-validation.md): benchmarks and correctness gates;
- [roadmap proposal](optimization-roadmap.md): phases and Beads mapping.

## Evidence classes

- **Measured**: produced in this investigation, on this workstation. Every
  measurement here is offline and CPU-only. No GPU run was made.
- **Recorded**: taken from an existing Rosevine record ([R2](runtime-r2-execution.md),
  [baseline](hardware-and-inference-baseline.md), [ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md)).
- **Derived**: arithmetic on measured or recorded values.
- **Hypothesis**: not yet tested.

Raw outputs, scripts and dumps are private (`.private/opt-research-v1`). This report
gives aggregates only.

## Verified model facts used throughout

These facts are recorded from the GGUF header audit and the R2 graph, and were
cross-checked against the pinned `src/models/gemma4.cpp`:

- 30 layers; embedding width 2816; 16 query heads.
- 25 sliding layers: 1024-token window, 8 KV heads of dimension 256, RoPE base 1e4.
- 5 global layers (5, 11, 17, 23, 29): 2 KV heads of dimension 512, base 1e6
  with `rope_freqs`.
- **Global layers have no V projection.** `V = rms_norm(K_raw)` with no weight
  and no RoPE, and `K = RoPE(rms_norm(K_raw) * k_norm)`. Before quantization,
  therefore, `K = RoPE(V * k_norm)`.
- 128 routed experts, top 8, expert width 704, GELU. One shared MLP of width 2112.
- The F32 router is `[2816, 128]` per layer, about 1.4 MB.
- One expert (gate_up plus down, Q4_0) is 3,345,408 bytes (3.19 MiB). One layer's
  experts are 0.399 GiB; all experts are 11.96 GiB. Non-expert weights are about
  1.31 GiB, including the 0.387 GiB tied Q4_0 output head.
- Context metadata is 262,144 for the target and 131,072 for the MTP assistant.
  The metadata limit is not evidence of long-context quality. R2 has exercised
  only about 1,030 positions on the real artifact.

## Findings from this investigation

### F1. Next-layer expert selection is predictable (measured, small sample; see S2 for lead time)

**Method.** Apply layer `l`'s real router (`gate_inp` and `gate_inp_scale`, read
from the GGUF) to the residual stream that *enters* layer `l`, which is
`l_out[l-1]`. The true router input is `rms_norm(attn_out[l])`, which is not
available until layer `l`'s attention has run. Recall is the fraction of the true
top 8 contained in the predicted top k.

**Sanity check.** Recomputing each layer's own router from its dumped `attn_out`
reproduced every dumped selection exactly (set-exact 1.0000).

**Data.** One 214-token calibration-text prompt, from the R2 HIP dump and the
llama.cpp CPU reference dump. R2 showed that decode is bitwise equal to batched
prefill, so these states are also decode states.

| Predictor (layers 1–29) | Recall of true top 8 | Candidates |
| --- | --- | --- |
| LES, next-layer router, top 8 | 0.727 (reference dump: 0.740) | 8 |
| LES, top 12 / top 16 / top 24 | 0.845 / 0.894 / 0.938 (reference: 0.860 / 0.910 / 0.952) | 12 / 16 / 24 |
| LES, two layers ahead, top 8 / top 16 | 0.556 / 0.747 | 8 / 16 |
| Same layer, previous token | 0.310 | 8 |
| Union of previous 4 tokens, same layer | 0.595 | 21.4 on average |
| Most frequent, fit on the first half, scored on the second half: top 16 / top 32 | 0.361 / 0.542 | 16 / 32 |

By layer:
- Layers 4–26 reach 0.67–0.85 at top 8 and 0.86–0.99 at top 16.
- Layers 1–3 (0.44–0.58 at top 8) and 27–29 (0.34–0.60) predict poorly. Layer 29
  is the worst.

**Interpretation.**
- The routing structure admits LES-style lookahead using state that is already
  computed.
- Lookahead clearly beats temporal and frequency predictors.
- The extra compute is one 2816×128 F32 GEMV plus a norm per layer. That is
  about 1.4 MB of reads, compared with 26.8 MB for one token's 8 experts.

**Limits.** One prompt, one domain, no held-out split across domains, no
multi-turn or code workload. Treat this as a reason to run the M1 trace study
(rv-e06-k0i.1), not as a finding about workloads in general.

### F2. Routing is flat, so prefill touches almost every expert (measured)

**Data.**
- Per layer, a median of 17 experts cover 50% of selections, and 58–59 cover 90%
  (range 45–75).
- The 214-token prompt touched a median of 106 of 128 experts per layer.
- Any 32-token window touched 65.

**Consequences.**
- Per-expert caching cannot make prefill cheap. Large batches need either every
  needed expert on the GPU or a bulk transfer.
- Decode caching needs large slot counts to reach a high hit rate.

This agrees with the external report in llama.cpp discussion
[#28248](https://github.com/ggml-org/llama.cpp/discussions/28248): a 16-slot
cache was slower than no cache, and gains depend on routing skew.

### F3. Expert-granular static placement may cut CPU expert work about 3× (in-domain only; **not supported across domains, see S3**)

**Method.** Rank all 3,840 (layer, expert) pairs by selection count over the
first 107 tokens. Place the least-used pairs on the CPU. Count CPU expert
evaluations per token over the remaining 107 tokens, and compare with first-N
placement at the same CPU expert bytes.

| Experts on CPU (bytes) | Popularity-ranked placement: CPU evals per token | First-N layers at equal bytes | Layers containing a CPU expert |
| --- | --- | --- | --- |
| 128 (0.40 GiB) | 2.4 | 8 (N = 1) | 4 |
| 256 (0.80 GiB) | 5.3 | 16 (N = 2) | 7 |
| 384 (1.20 GiB) | 8.1 | 24 (N = 3) | 10 |
| 1,024 (3.19 GiB) | 20.3 | 64 (N = 8) | 27 |

**Caveat.** More layers contain at least one CPU expert, and each such layer can
need a CPU/GPU handoff. Fewer CPU evaluations do not automatically mean less
time; the handoff cost is unmeasured. The sample is in-domain and small. A
domain shift would raise the CPU rate.

### F4. R2's CPU expert path is far from the memory roofline (measured)

**Method.** A disposable benchmark called the runtime's own `matmul_q4_0_multi`
on the real weights: 8 experts × 1 token (one decode step of one layer), median
of 25 runs after 5 warmups.

| Threads | 1 | 6 | 10 | 12 |
| --- | --- | --- | --- | --- |
| Time per layer, ms | 36.8 | 6.5 | 5.4 | 4.7 |
| Effective weight stream, GB/s | 0.7 | 4.1 | 5.0 | 5.7 |

- The same 8 experts were reused, so the working set (26.8 MB) was L3-warm. Cold
  DRAM can only be slower, so these are lower bounds.
- `par_rows` spawn and join costs about 97 µs per call, and each layer makes two
  calls.
- The DRAM ceiling is derived, not measured: dual-channel DDR5-4800 is
  76.8 GB/s theoretical, and its effective bandwidth is unmeasured (M2).
  At 60–70% of that ceiling, one layer's 8 experts would take about 0.5–0.6 ms.
- **Derived:** with N = 10, CPU experts cost at least 54 ms per decode token.
  That alone caps decode at about 18 tokens/s.

### F5. The recorded "about 4 tokens/s" is not attributed (**resolved in S1: environmental**)

Two R2 runs disagree:
- The `generate` run measured 4.09 tokens/s (about 244 ms/token), and spent
  4.9 s prefilling 40 tokens.
- A teacher-forced stepped run on the same artifact did 512 one-token steps in
  40.4 s per chunk (about 79 ms/step, including KL scoring).

F4 explains about 54 ms of the 79 ms. The remaining roughly 165 ms in `generate`
is unexplained. Candidate causes, none tested:
- page-cache eviction of the mmapped CPU expert pages: about 9 GiB of swap was in
  use on the host during planning observations;
- a `max_batch`-dependent difference;
- contention from other processes.

**Conclusion.** "CPU experts dominate" is supported for the stepped run. It is
not established for the 4.1 tokens/s figure. Attribution (K1, rv-e05-7n3.1) has
to come before any placement claim.

### F6. R2 synchronizes with the host constantly (recorded from source)

All work runs on the null stream. In every layer:
- router logits are downloaded, which blocks;
- four small blocking uploads (`idx`, `pos`, `scale`, `weight`) follow on
  GPU-expert layers;
- CPU-expert layers download their input and upload their output.

That is about 130 blocking copies and about 750 kernel launches per decode token,
followed by a 1 MiB logits download.

The cost is unmeasured. C3 measured submit at 0.7 µs per launch, but blocking
round trips were not measured.

Device-side routing would remove the per-layer readback on GPU-resident layers.
It would also make the launch sequence static, which per-token graph replay
requires.

### F7. Transfer economics favour CPU execution of decode misses (derived)

The rates per expert:

| Path | Estimate | Basis |
| --- | --- | --- |
| Upload over PCIe 4.0 ×16 | 130–170 µs | 20–26 GB/s assumed; the link is 31.5 GB/s nominal, and effective pinned bandwidth is unmeasured |
| CPU compute for one token | about 67 µs | measured with the S4 prototype, cold; today about 560 µs (F4) |
| GPU compute when resident | about 5 µs | 624 GB/s nominal |

What follows:
- **Single use.** Uploading an expert for one decode token costs about 2.5–3×
  more than computing it on a good CPU path. Upload pays only if the expert is
  reused while resident, or the copy is fully hidden while the CPU is on the
  critical path. The break-even hit rate is roughly
  `1 − PCIe_BW / RAM_BW`, about 0.65–0.75 here (the #28248 author's model,
  applied to our ratios).
- **PCIe budget at the target speed.** At about 100 tokens/s, a layer has about
  333 µs of wall time, during which PCIe moves about 2–2.5 experts. Any prefetch
  scheme, LES included, can hide only a few misses per layer per token. It
  complements a mostly-resident working set. It cannot stream one.
- **Prefill is the opposite case.** A 512-token batch puts about 32 tokens on
  each expert. Moving a whole layer of experts (0.4 GiB) takes about 16–20 ms.
  Computing the same layer on the CPU is about 49 GFLOP, which is tens of
  milliseconds even with a good kernel. Prefill should stream offloaded experts
  to the GPU; decode should compute them on the CPU. This is a hypothesis; M2
  measures the bandwidth.

### F8. VRAM budget and the KV/scratch cost of R2's allocator (derived from recorded values)

**Available VRAM.** About 13.8 GiB:
- 15.98 GiB on the device;
- minus 1.55 GiB used by the desktop at idle;
- minus about 0.64 GiB of R2 process overhead beyond its own allocations. This
  is a single-run inference: an 11.52 GiB peak, minus the 1.55 GiB baseline,
  minus the 9.33 GiB session.

**Full residency.** All weights are 13.27 GiB, so full residency leaves about
0.5 GiB for KV, scratch and headroom. A spill is unavoidable beyond short
contexts.

**R2's allocation formulas,** applied as written:

| Context | Local KV | Global KV (Q8_0, K and V) | R2 attention scores scratch (batch 512) | R2 total scratch (batch 512 / 1) |
| --- | --- | --- | --- | --- |
| 16K | 0.16 GiB | 0.17 GiB | 0.50 GiB | 1.20 / 0.002 GiB |
| 32K | 0.16 GiB | 0.33 GiB | 1.00 GiB | 1.70 / 0.003 GiB |
| 64K | 0.16 GiB | 0.66 GiB | 2.00 GiB | 2.70 / 0.005 GiB |
| 128K | 0.16 GiB | 1.33 GiB | 4.00 GiB | 4.70 / 0.009 GiB |
| 256K | 0.16 GiB | 2.66 GiB | 8.00 GiB | 8.70 / 0.017 GiB |

- Local KV is a ring of `1024 + batch` slots per layer.
- Global KV grows at 10,880 bytes per token.
- R2 materializes `batch × 16 heads × max_ctx` F32 scores, and keeps a
  `batch × 262144` logits buffer even when only the last row is needed.
- With tiled attention and last-row logits, batch-512 scratch would be about
  0.2 GiB at every context.

**Implied expert spill** (to be confirmed by Q2, rv-e08-nhh.2). Assumptions:
13.8 GiB available, 1.31 GiB non-expert weights, Q8_0 K and V, tiled-attention
scratch of 0.2 GiB, and 0.3 GiB headroom. R2's current scratch would spill far more.

| Context | Expert spill | Approximately |
| --- | --- | --- |
| 16K | about 0.3 GiB | under 1 layer |
| 64K | about 0.8 GiB | 2 layers |
| 128K | about 1.5 GiB | 4 layers |
| 256K | about 2.8 GiB | 7 layers |

These agree with the production presets (2, 3 and 8 CPU MoE layers), which also
hold the 0.25 GB MTP assistant.

### F9. Gemma-native global KV: store V once and derive K (hypothesis)

From the verified facts, `K = RoPE(V * k_norm)` before quantization. A global
layer could therefore store only V (Q8_0) and form K inside the attention kernel:
dequantize, multiply by `k_norm`, rotate.

- **Saving:** half the global KV. That is 0.33 GiB at 64K and 1.33 GiB at 256K,
  or room for about 100 and 420 more resident experts respectively.
- **Numerics:** the rounding point moves from after RoPE to before it, so the
  result is not bitwise equal to R2. It needs a fidelity test.
- **Prior art:** the pinned llama.cpp stores K and V separately.
- **Cost:** a RoPE recompute per key on every pass. That cost is small next to
  the bandwidth saved, but unmeasured.

### F10. External references (read 2026-11-13)

**OpenBMB LES**
([doc](https://github.com/OpenBMB/llama.cpp-omni/blob/main/docs/les/lookahead-expert-scheduling.md)):
- Its claims:
  - prediction steers only prefetch and scheduling, and the experts actually
    routed are always used;
  - experts that are already loaded are computed first while the rest load.
- Its numbers come from SSD-backed phones and tablets (for example 3.38 → 6.57
  tokens/s from LES alone on one device) with a MiniCPM-V 35B-A3B model.
- The doc does not describe the predictor, does not report accuracy and states
  no limitations.
- The setting differs from ours: SSD to unified memory, not RAM to a discrete
  GPU over PCIe.
- **Transferable:** the losslessness rule and the ordering that computes resident
  experts first.
- **Not transferable:** the speedup.

**llama.cpp [#28248](https://github.com/ggml-org/llama.cpp/discussions/28248)**
proposes a persistent per-tensor LRU slot pool with no prediction:
- +12% at 16 slots and +84% at 64 slots for an all-CPU-expert configuration on a
  24 GB GPU;
- a third party measured a 16-slot cache slower than no cache;
- a stale map-table bug produced wrong weight reads;
- prefill bypasses the pool when `top_k × tokens > slots`.

The pinned build's own `--moe-cache-mib` (segmented LRU, batch limit 32) remains
under audit in M1.

## Follow-up scrutiny (2026-11-13)

The maintainer asked for four findings to be checked before the ADR and the R3a
plan are finalized.

**Run conditions.**
- GPU runs were serialized under a run lock, at 100 ms sensor sampling, with the
  agreed hard stops.
- The production router was idle throughout (0.01 GiB VRAM).
- No run hit a stop. Peak junction was 66 °C, and MemAvailable stayed at or
  above 21 GiB.
- Raw outputs are private (`.private/opt-research-v1`). All harnesses are
  disposable and outside the repository. The R2 runtime was used unmodified.

### S1. Decode discrepancy: environmental, not a methodology difference (measured)

Each variant is one factor away from the recorded `generate` run, with the same
prompt, N = 10 and 12 threads:

| Variant | Decode tokens/s | Step p50 / p90, ms |
| --- | --- | --- |
| Generate as recorded (max batch 64; logits, download and argmax every step), 520 steps | **13.41** | 73.5 / 79.8 |
| Same, 200 steps | 13.61 | 72.2 / 79.2 |
| Max batch 1 | 13.77 | 71.6 / 76.8 |
| Teacher-forced, no logits | 13.99 | 70.7 / 75.9 |
| Teacher-forced, with logits and download | 13.21 | 72.8 / 88.5 |
| Generate with a concurrent 12-thread CPU competitor | 9.46 | 93.4 / 132.4 |

**Equivalence of the two harnesses.**
- `generate` and the teacher-forced loop do equivalent work to within about 3%.
- The only systematic difference is the output head plus a 1 MiB download and
  argmax: about 2 ms per step. `kld` skips them for the first half of each chunk,
  and adds file reads and KL scoring.
- Neither path has a measurement defect. Major page faults during decode were 0,
  and 78 in the very first cold run.

**The recorded 4.09 tokens/s run was contaminated.**
- Its output is token-identical to every rerun, so it did the same work.
- Its timestamps put it right after a llama.cpp CPU reference generation of the
  same prompt (that run ended at 07:41:38 and the HIP run at 07:43:59), and very
  likely overlapping it. That reference uses all cores and repacks 13.5 GiB of
  anonymous memory on a 30 GiB host.
- A partial CPU competitor alone reproduces a 30% slowdown and a 1.7× p90.
  Full-core contention plus memory pressure plausibly accounts for the rest;
  that is not reproduced, deliberately, to protect the shared host.
- R2's spawn-per-call thread model waits for its slowest thread, which makes
  it sensitive to contention.

**Corrected baseline** (`r2-ref`, N = 10): about 13.4–14.0 tokens/s, roughly
72 ms/token. The N = 10 → 20 slope attributes 5.40 ms to each CPU-expert layer
(12 threads), so about 54 ms of the 72 ms (75%) is CPU-expert layers. About
18 ms is everything else.

### S2. LES causality: a lookahead early enough to matter is weaker (measured)

**Data.** Traces over held-out text, 4,870 tokens in total:
- 2,048 prose and 2,048 code tokens from the frozen fidelity corpus holdout;
- a 560-token model-generated chat continuation;
- the 214-token calibration prompt.

The harness recorded only selections and predictions computed from states that
exist before the target layer routes.

| Predictor | State used | Available when | Recall at 8 | Recall at 16 | Wrong candidates at 16 |
| --- | --- | --- | --- | --- | --- |
| p1 (= F1) | Residual entering layer l | Start of layer l: lead is layer l's attention only | 0.70–0.79 | 0.88–0.93 | 8.8 |
| pmid | `attn_out` of layer l−1 | Mid layer l−1: lead is MoE(l−1) plus attention(l), about 1 layer | 0.60–0.73 | 0.79–0.90 | 9.3 |
| p2 | Residual entering layer l−1 | Start of layer l−1: about 1.35 layers | 0.53–0.65 | 0.71–0.83 | 9.9 |
| Previous token, same layer | | | 0.27–0.42 | | |
| Frequency top 16, fit on prose, scored on another domain | | | | 0.14–0.24 | |

Ranges are across domains. The calibration-prompt p1 value reproduces F1 (0.716
against 0.727), which validates the harness.

**By layer** (p1 at 8, code):
- weak at layers 0–3 (0.36–0.55) and 27–29 (0.55–0.68);
- 0.69–0.88 elsewhere.

**Lead time against transfer time.**
- An expert upload takes 131 µs (measured, pinned).
- A tuned decode at about 100 tokens/s leaves about 320 µs per layer.
  Attention is about 35% of a layer's weight bytes, so p1's window is about
  110 µs. That is shorter than one upload.
- **So the F1 predictor is causal but cannot stage a transfer at target
  speeds.** Only pmid and p2, which are less accurate, give one to three uploads
  per layer.

**Estimated useful savings** (upper bound: perfect overlap, free slots,
popularity plan with 256 host experts, prototype CPU kernel, fit on prose, scored
on code):

| Predictor | Hits per token | False transfers per token | PCIe per token | Saved per token |
| --- | --- | --- | --- | --- |
| pmid | 9–10 | 12.6–15.8 | 69–84 MiB | about 1.0 ms |
| p2 | 9–10 | 15.5–17.1 | 78–86 MiB | about 1.0 ms |

Against 1.4 ms of host cost per token, that is at most about 10% of a 10 ms
target token. It would need slot caches, async copies and eviction.

**Prediction for CPU-executed experts gives no transfer benefit.** The only
remaining use is resident-first ordering, which needs no prediction.

**Verdict on B3 is unchanged: NO-GO now.** Kept: B1 (log pmid and p2 predictions
after device routing exists) and a clean path (ADR 0002 rule 7).

### S3. Per-expert placement costs: no defensible gain across domains (measured and derived)

**Cross-domain placement.** Fit popularity on one domain and evaluate on another,
at equal bytes:

| Host experts (equal to) | Fit → eval | CPU evaluations per token, popularity / first-N | Host layers per token, popularity / first-N |
| --- | --- | --- | --- |
| 256 (N = 2) | prose → code | 13.0 / 16 | 8.7 / 2 |
| 256 (N = 2) | code → prose | 4.4 / 16 | 3.4 / 2 |
| 256 (N = 2) | prose → chat | 5.8 / 16 | 5.4 / 2 |
| 256 (N = 2) | code → chat | **25.9** / 16 | 15.2 / 2 |
| 1024 (N = 8) | code → chat | **94.9** / 64 | 27.9 / 8 |
| 256 (N = 2) | code first half → code second half | 0.28 / 16 | — |

- Routing is domain-specialized. In-domain popularity is excellent, but it does
  not transfer: F3's 3× was an in-domain artifact.
- Popularity placement multiplies the number of layers that need a CPU/GPU
  handoff by 2–8×.

**Handoff cost.**
- **Copies.** Measured blocking 11 KB copies take 18–31 µs. The selection
  readback and output upload add one each.
- **In situ.** The N = 10 → 20 slope exceeds the kernel microbenchmark by
  0.62–0.68 ms per host layer, at both 6 and 12 threads. So oversubscription is
  not the cause, and **each host layer currently carries at least 0.6 ms of
  unattributed overhead**. It is at least that, because GPU expert time saved
  also hides in the slope.
- **Break-even.** Popularity (256) beats first-N (N = 2) on code only if a host
  layer's handoff costs under 22–29 µs. That is below the copy latency alone.

**Cost model.** Added host time per token is
`Σ over host layers (h + c0 + c1·k_layer)`, with
`c(k) = 0.23 + 0.56k` ms for the current kernel and `0.001 + 0.067k` ms for the
prototype. At realistic `h`, popularity placement is roughly equal to or worse
than first-N except where the domains happen to match.

**Conclusion.**
- Keep the per-expert `ResidencyPlan` as the interface. It costs little, keeps
  experiments and a future cache possible, and `first_n` stays exact.
- **Do not claim a per-expert placement benefit.** The default policy remains
  first-N.
- A domain-adaptive policy belongs to M3/M4 cache research, not static
  placement.

### S4. CPU expert kernel: the 1.5 ms objective is attainable in a prototype (measured)

One decode token, 8 experts, cold weights (rotating across all 30 layers),
12 threads, median:

| Implementation | Per layer | Effective stream | Error against f64 (relative L2, gate_up) |
| --- | --- | --- | --- |
| Current `matmul_q4_0_multi` (spawn per call, dequantize to f32, scalar) | 4.72 ms | 5.7 GB/s | 1.97e-7 |
| Persistent spin-barrier pool and AVX-512 f32 activations (R2 semantics) | **0.54 ms** | 49.8 GB/s | 1.80e-7 |
| Same pool, Q8_0 activations, int8 dot (`maddubs`, unoptimized reduction) | 0.75 ms | 35.6 GB/s | 3.9e-3 (the reference-like approximation) |

**Measured limits.**
- DRAM stream-touch reaches 51 GB/s at 12 threads and 44 GB/s at 6, so the f32
  prototype is near the memory bound.
- With 6 threads the f32 prototype takes 0.77 ms (p90 1.66 ms).
- The toolchain's AVX-512 VNNI intrinsic failed to compile, so the Q8_0 variant
  used the AVX-512BW equivalent. Its slowness is an implementation artifact,
  not a verdict.

**Not yet measured.**
- **In-situ behaviour:** wake latency of a parked rather than spinning pool,
  interaction with HIP runtime threads, and the S3 residual.
- **Memory layout.** Repacking rows for wider loads is unnecessary at this
  bandwidth. Huge pages for the mapped weights are untested.

**Status.** These are disposable code paths. The objective stays an objective
until the in-situ R3a measurement.

## Technique assessments

**Verdicts:**
- **GO**: recommended once its dependencies are met and its run or
  implementation is authorized.
- **CONDITIONAL GO**: proceed only if the named evidence turns out positive.
- **NO-GO**: not now, with the reason and the condition for revisiting.

Benefits are qualitative unless a finding above supports a number. "Bottleneck"
names what the technique attacks. Existing Beads are cited by key (K1, M1, …)
from the [roadmap](research-roadmap.md).

### Summary

| # | Technique | Bottleneck | In Rosevine now | Complexity | Verdict |
| --- | --- | --- | --- | --- | --- |
| A1 | Residency plan per (layer, expert) | Placement granularity | First-N only | Low–medium | GO (interface, `first_n` default); popularity policy NO-GO (S3) |
| A2 | Device-side routing | Host sync per layer | Host top-k | Low | GO |
| A3 | Fast CPU expert kernel and persistent pool | CPU expert time (F4, S4) | Scalar, spawn per call | Medium | GO (prototype 0.54 ms per layer, S4) |
| A4 | Same-layer hybrid split, CPU overlapped with GPU | Serialized CPU/GPU | None | Medium | CONDITIONAL GO (after A1–A3) |
| A5 | Pinned host staging (bounded) | Copy efficiency | Pageable `hipMemcpy` | Low | CONDITIONAL GO (with F1) |
| A6 | Dynamic GPU expert slot cache for decode | Misses after static placement | None | High | CONDITIONAL GO, deferred to M3 |
| A7 | Prefill expert streaming | Prefill on CPU layers | None | Medium–high | CONDITIONAL GO (after M2 bandwidth) |
| A8 | Startup memory ledger and budget planner | Ad hoc budget | Partial (`device_bytes_needed`) | Low | GO |
| B1 | LES predictability study | Unknown predictability | Done here, preliminary | Low | GO: replicate held-out |
| B2 | Compute resident experts first | Idle GPU during misses | None | Low with A4 | GO, as part of A4 |
| B3 | LES speculative prefetch (async upload) | Miss latency | None | High | NO-GO now (S2: causal lead too short for p1; at most about 1 ms per token for pmid/p2); revisit after M3 |
| B4 | LES for CPU-executed experts | Page faults or cache misses | None | Low | NO-GO except as a page-fault diagnostic |
| C1 | Tiled exact attention, local and global | O(batch × ctx) scratch; attention bandwidth | Naive, full scores | Medium | GO |
| C2 | Global V-only KV, K derived (F9) | Global KV memory | None | Medium | CONDITIONAL GO (fidelity) |
| C3 | Q4 KV | KV memory | Q8_0 | Medium | CONDITIONAL GO, after C2 |
| C4 | Paged or segmented KV | Fragmentation, multi-sequence | Preallocated | High | NO-GO (single sequence) |
| C5 | KV spill to host RAM | KV memory | None | High | NO-GO (counterproductive) |
| C6 | Chunked prefill with per-phase micro-batch | Prefill scratch and latency | Batch chunking | Low | GO (formalize) |
| D1 | Grouped MoE GEMM with token grouping | Prefill expert compute | Plan grouping, then naive per-expert matmul | Medium–high | GO (K2) |
| D2 | Fused dequant GEMM / WMMA | Prefill compute | No | High | GO (K2), measured |
| D3 | Norm, activation and reduction fusion | Launches and traffic | No | Medium | CONDITIONAL GO (K3) |
| D4 | Overlap shared MLP with routed experts | Serialized independent work | No | Low with A4 | CONDITIONAL GO |
| E1 | Tuned Q4_0 GEMV for decode | Decode bandwidth | Warp-per-row, untuned | Medium | GO (K2) |
| E2 | Graph replay (hipGraph) and launch reduction | Dispatch overhead | No | Medium | CONDITIONAL GO (after A2; rocm support unverified) |
| E3 | On-device sampling and last-row logits | 1 MiB download; scratch | Host argmax | Low | GO |
| F | Async copy stream, events, double buffering | Transfer and compute serialization | Null stream only | Medium–high | CONDITIONAL GO: only with a consumer (A7 or A6) |
| G1 | MTP speculative decoding | Decode serial dependency | Not implemented (S1–S3) | High | GO via the existing S track after G3 |
| G2 | N-gram / prompt-lookup drafting | Decode on repetitive outputs | No | Low–medium | CONDITIONAL GO (shares the S2 verifier) |
| G3 | Independent draft model | Decode | No | High | NO-GO |
| G4 | Self-speculative layer skip | Decode | No | High | NO-GO |
| H1 | In-session prefix reuse | Repeated prefill | `rewind` only | Low | GO |
| H2 | Cross-request prefix snapshots | Repeated system and tool prompts | No | Medium | CONDITIONAL GO (isolation design) |
| I1 | Deterministic startup planner | Configuration effort | `--cpu-moe N` | Low | GO (with A8) |
| I2 | Between-request re-planning | Workload shift | No | Medium | Deferred |
| I3 | Online auto-tuning | — | No | High | NO-GO now |

### A. Memory residency and expert placement

**A1. Residency plan per (layer, expert).**
- **What.** An immutable map from each (layer, expert) to a location: a device
  slot or the host. First-N becomes one constructor of it.
- **Why Gemma.** 3,840 equal 3.19 MiB experts and flat routing (F2). Popularity
  still separates hot from cold experts (F3).
- **Why the RX 7800 XT.** It spills only 0.1–3 GiB of a 12 GiB expert set (F8).
  Granularity decides which bytes spill.
- **Today.** `cpu_expert_layers: usize`.
- **Study.** llama.cpp's `-ot` tensor overrides and the `--moe-cache-mib` pool.
  KTransformers and fiddler-style CPU/GPU MoE splits are references for the
  design only.
- **Depends on.** A8 (ledger). M1 held-out traces, for any popularity-based
  policy.
- **Risks.** Domain shift moves the hot set. The number of layers with a CPU
  expert grows (F3), so handoff cost can offset the savings.
- **Evidence needed.** Held-out multi-domain traces. Measured cost per handoff.
  An equal-VRAM A/B against first-N.
- **Verdict.** **GO** for the interface: it is a pure refactor, bitwise-identical
  for first-N plans. **NO-GO** for popularity placement as a static policy: it
  does not hold across domains and multiplies handoffs (S3).

**A2. Device-side routing.**
- **What.** Softmax, top 8 and renormalization on the GPU, written to a device
  selection buffer. The host copy happens only when a host consumer needs it.
- **Why.** It removes about 30 blocking readbacks per token (F6). It enables
  graph capture (E2) and device-side dispatch, and it is the place a
  next-layer prediction would be computed (B1).
- **Today.** The host runs `select_experts`, shared with the CPU executor.
- **Risks.** Tie-breaking and `p / max(sum, 2^-14)` semantics must match the host
  exactly. The selection must be identical (integer ids), not merely close.
- **Evidence needed.** Selection identity on R2 dumps and on synthetic tie
  fixtures.
- **Verdict.** **GO.**

**A3. Fast CPU expert kernel.**
- **What.**
  - A persistent worker pool that removes the per-call spawn (about 97 µs).
  - An AVX2/AVX-512 Q4_0 kernel with f32 activations, keeping R2 semantics.
  - Optionally Q8_0 activations with VNNI integer dot, which is the pinned CPU
    reference's own semantics.
- **Why.** F4: 5.4 ms against a 0.5–0.6 ms bandwidth bound. Every placement
  policy at 64K and above keeps 1–3 GiB of experts on the CPU (F8), so this pays
  off under any plan. It also remains useful if residency improves.
- **Study.** ggml-cpu `q4_0` repack (`q4_0_8x8`) and `ggml_vec_dot_q4_0_q8_0`.
  Both are MIT licensed and may be ported with attribution.
- **Risks.**
  - SIMD changes the summation order, so results are not bitwise equal to the
    current CPU path. It needs a tolerance and the KL band.
  - Repacking duplicates memory, and the CPU repack has a 13.5 GiB anonymous
    history in Epic 2. Repack only the CPU-resident experts.
- **Verdict.** **GO.** The numerics choice needs approval (see
  [decisions](#decisions)).

**A4. Same-layer hybrid split.**
- **What.** In a layer with mixed placement, the GPU runs the resident selected
  experts while the CPU runs the rest. The per-pair outputs then meet in the
  existing slot-ordered combine.
- **Why.** It hides CPU time behind GPU work. Combined with D4, CPU experts also
  overlap the shared MLP.
- **Depends on.** A1, A2, A3, and two streams or host threads.
- **Risks.** Correctness of the combine order. Uploading CPU per-pair outputs
  into the device `y` buffer keeps the sum order bitwise identical.
- **Verdict.** **CONDITIONAL GO** once attribution shows serialized CPU/GPU time.

**A5. Pinned host staging.**
- **What.** A small, bounded set of pinned staging buffers.
- **Why.** Async copies need pinned memory. Pinning all expert weights is ruled
  out by 30 GiB of RAM, a shared host and observed swap.
- **Verdict.** **CONDITIONAL GO**, alongside the first consumer of async copies.

**A6. Dynamic GPU slot cache for decode.**
- **Why it may help.** It turns repeated misses into hits.
- **Against it.** Flat routing (F2) and a 0.65–0.75 break-even hit rate (F7)
  call for large pools, but the spill is only 0.1–3 GiB. After A1, the remaining
  miss count may be about 5–10 evaluations per token at 64K.
- **Risks.** Stale-map bugs (seen in #28248), prefill pollution, VRAM taken from
  KV.
- **Verdict.** **CONDITIONAL GO**, deferred to M3 (rv-e06-k0i.3) trace replay
  against A1 plus A3 as the baseline. It is not the baseline.

**A7. Prefill expert streaming.**
- **What.** For large batches, upload each CPU-placed layer's needed experts (in
  practice nearly all, per F2) into a double-buffered device staging area, then
  run the grouped GEMM on the GPU.
- **Why.** F7: bulk PCIe beats CPU compute when each expert carries about
  32 tokens.
- **Study.** llama.cpp's offload of host-resident weights for large batches.
- **Depends on.** F (async), A5, D1, and the M2 bandwidth.
- **Verdict.** **CONDITIONAL GO** for the TTFT objective.

**A8. Memory ledger and budget.**
- **What.** One accounting of every device byte by class: weights, experts, KV,
  scratch, staging, runtime overhead and reserve. Measured high-water marks are
  checked against it.
- **Today.** The pieces exist (`device_bytes_needed`, the `max_vram` check).
- **Verdict.** **GO.**

### B. Lookahead Expert Scheduling

**Correctness rule (all B items).** Prediction may change residency, transfer
order and execution order. It never changes which experts are summed, their
weights or the combine order. Selection always comes from the real router at
the real layer.

**B1. Predictability study.**
- **Done here (F1).** Next-layer recall is about 0.73 at top 8 and about 0.89 at
  top 16; layers 4–26 are good and the ends are poor.
- **Missing.** Held-out multi-domain traces, decode-length sequences and MTP
  verification batches, all of which M1 collects.
- **Verdict.** **GO**: replicate.

**B2. Resident-first ordering.**
- **What.** Within a layer, launch resident experts before missing ones.
- **Cost.** None, once A4 exists.
- **Verdict.** **GO** as part of A4.

**B3. Speculative prefetch.**
- **Use.** Top 12 to top 16 predictions would select upload candidates for
  layer `l+1` while layer `l` runs.
- **Against it now.**
  - PCIe moves only about 2–2.5 experts per layer at the target speed (F7).
  - Each wrong prefetch costs a slot and evicts something.
  - The CPU path after A3 may already be cheaper than an exposed miss.
  - Benefit with untuned kernels is not meaningful (see the
    [validation plan](optimization-validation.md#what-can-be-measured-now)).
- **Prerequisites.** F, A5 and A6. The prediction itself (top 16 from A2's
  device-side router) is cheap and can be logged early at no scheduling risk.
- **Verdict.** **NO-GO now.** Revisit when a dynamic cache is GO and the measured
  misses per layer are both small and exposed.

**B4. Prediction for CPU-executed experts.**
- **Why it adds little.** CPU experts have no faster tier to load into. The only
  use is prefaulting mmapped pages under memory pressure, which is relevant only
  if F5 turns out to be paging.
- **Verdict.** **NO-GO** except as that diagnostic.

**Questions from the brief, answered from F1 and F7:**

| Question | Answer |
| --- | --- |
| Accuracy | 0.70–0.79 at top 8 for p1, which is too late to stage; 0.60–0.73 (pmid) and 0.53–0.65 (p2) for usable leads (S2) |
| Variation by layer | Strong: the ends are poor |
| Does the structure admit lookahead? | Yes |
| Uses available state? | Yes: `l_out[l-1]` |
| Extra router cost | One 1.4 MB GEMV per layer |
| Overlaps useful work? | Yes, if computed alongside layer `l-1`'s router |
| Enough to justify speculative transfers? | No, for now: at most about 1 ms per token saved with 13–17 false transfers per token (S2) |
| Against frequency and temporal predictors | Clearly better |
| Can wrong predictions evict needed experts? | Yes, which is why admission must be bounded |
| Expected benefit | Bounded by about 2 hidden uploads per layer per token |
| Works with CPU-executed experts? | Only for ordering (B2) |

### C. Attention and KV cache

**C1. Tiled exact attention.**
- **What.** Online-softmax tiles: separate local (hd 256, GQA 2) and global
  (hd 512, GQA 8) kernels, plus a split-K decode variant for long global
  contexts.
- **Why.** It removes the O(batch × max_ctx) score scratch (F8), and it is the
  bandwidth-critical decode path at long contexts. Global decode reads the whole
  global KV every token: 0.66 GiB at 64K, about 1.1 ms at nominal bandwidth.
- **Study.** The ggml HIP/Vulkan flash-attention kernels (C1/C2 noted HIP
  flash-attention spills at D 256/512) and Composable Kernel FMHA (an RDNA3
  reference).
- **Risks.** Register pressure at hd 512, and masking/ring correctness.
- **Verdict.** **GO.** It is a prerequisite for any context above about 32K at
  prefill batch 512.

**C2. Global V-only KV (F9).**
- **Verdict.** **CONDITIONAL GO.** It needs a teacher-forced KL check and a
  long-context retrieval check against both K/V Q8_0 and the reference.

**C3. Q4 KV.**
- **Why.** The saving lies almost entirely in global KV, because local KV is
  0.16 GiB.
- **Risk.** Long-context fidelity: quantization error accumulates across 256K
  keys.
- **Verdict.** **CONDITIONAL GO**, only after C2, and judged by long-context
  fixtures, not short KL alone.

**C4. Paged KV.** Single sequence, preallocated: there is no fragmentation
problem. **NO-GO.** Revisit with concurrent sessions.

**C5. KV spill to RAM.** Global KV is read in full every decode step: 0.66 GiB
at 64K takes about 27 ms over PCIe. Spilling cold experts costs far less (F7).
**NO-GO.** The VRAM priority order is: KV and scratch, then non-expert weights,
then hot experts, then cold experts.

**C6. Chunked prefill.**
- **Today.** It exists as batch chunking.
- **To do.** Make the micro-batch a per-phase parameter of the planner. Use the
  largest size that fits after C1, and expose the TTFT/throughput trade-off.
- **Verdict.** **GO.**

**Context limits.**
- **Memory.** C1 plus C2 make 256K memory-feasible with an expert spill of about
  1.5 GiB (derived from the F8 assumptions).
- **Prefill time.** Global attention is quadratic: 256K tokens of prefill are
  likely minutes even with good kernels. This is unmeasured.
- **Positional behaviour.** Proportional RoPE beyond tested positions is
  unverified; Q2 decides.

Memory-efficient KV does not imply correct long-context behaviour.

### D. Prefill

- **D1. Grouped MoE GEMM.**
  - `MoePlan` already groups token-expert pairs by expert.
  - At batch 512 an expert averages about 32 rows, enough for a 32×N tile. At
    batch 64 it averages about 4, which is GEMV-like.
  - One launch per (layer, projection) over all experts, instead of up to 128
    launches.
  - **Study.** ggml `mul_mat_id`, the vLLM/SGLang fused-MoE Triton kernels
    (Triton as an offline generator per ADR 0001), and CK grouped GEMM.
  - **Verdict.** **GO** under K2.
- **D2. Fused dequant GEMM with RDNA3 WMMA.**
  - gfx11 has WMMA for f16, bf16 and iu8. Q4_0 can be dequantized to f16 or i8
    in registers.
  - **Risk.** Activation precision: f16 or i8 activations change numerics, so
    tolerances must be defined first.
  - **Verdict.** **GO** under K2, measured against D1 without WMMA.
- **D3. Fusion** (norm + GEMV input, GEGLU + down input, combine + post-norm).
  - **Verdict.** **CONDITIONAL GO** under K3, once attribution shows launch- or
    traffic-bound phases.
- **D4. Overlap shared MLP with routed experts.** These two depend only on
  `attn_out`, so they can run concurrently. **CONDITIONAL GO** with A4.
- **Prefix reuse for prefill** is covered in H.

### E. Decode

The kernel deficit and the placement deficit must be separated first. The
stepped run (F5) suggests about 25 ms of non-CPU-expert time per token with N = 10.
That is an estimate from two numbers, not a measurement.

- **E1. Tuned Q4_0 GEMV.**
  - **Today.** A warp per row, with each lane reading 18-byte blocks at an 18-byte
    stride.
  - The tied output head (0.387 GiB) is the largest single read per token.
  - **Study.** ggml `mul_mat_vec_q` for HIP and Vulkan, which are faster today
    per C1/C2.
  - **Verdict.** **GO** under K2. This helps whether or not the whole model is
    resident.
- **E2. Graph replay and launch reduction.**
  - About 750 launches per token today.
  - hipGraph needs static pointers and sizes, which requires A2 plus
    device-indexed expert addressing.
  - ROCm 6.4 hipGraph on gfx1101 is unverified.
  - **Verdict.** **CONDITIONAL GO.**
- **E3. On-device sampling and last-row logits.**
  - Removes the 1 MiB download per token and 0.5 GiB of scratch at batch 512.
  - Greedy argmax first. Sampled decoding needs the S1 sampler semantics.
  - **Verdict.** **GO.**

### F. Asynchronous execution

**What.**
- One compute stream, one copy stream, events, and double-buffered staging.
- A transfer request API that both prefill streaming (A7) and any future
  prefetch (B3) consume.

**Constraint.** Layers run strictly in sequence. Overlap is available only
across independent work inside a layer (A4, D4) or for copies issued ahead of
time.

**Risks.**
- Lifetime: a slot must not be reused while a kernel reads it.
- Cancellation: drain the streams before returning.
- Instrumentation can serialize execution.

**Verdict.** **CONDITIONAL GO**, built only together with its first consumer.
The C ABI may gain a stream parameter early; this is a decision for the
maintainer.

### G. Speculative decoding

- **G1. MTP.**
  - The model ships a 4-layer assistant (0.25 GB) that shares target KV.
  - Reported production decode with MTP is 101–131 tokens/s, against 80 without,
    but the preset comparisons are confounded.
  - **Interaction with placement.** A 4-token verification touches about
    21 distinct experts per layer (F1, union of 4 tokens). On CPU-resident
    layers it therefore reads about 2.7× the expert bytes of one token. GPU
    layers are nearly flat.
  - So speculation rewards GPU residency, and its net value at N > 0 must be
    measured.
  - **Verdict.** **GO** via the existing S1 → S2 → S3 Beads after G3.
- **G2. N-gram / prompt lookup.**
  - Exact under greedy decoding. Under sampling it is exact only with standard
    rejection-sampling verification.
  - Cheap, and strong for code, RAG and tool outputs.
  - It shares the batched verifier and KV rollback (R2 has `rewind`).
  - **Verdict.** **CONDITIONAL GO**, after S2's verifier.
- **G3. Independent draft model.** No small same-tokenizer model is on hand. It
  would take VRAM from experts and KV, and MTP exists. **NO-GO.**
- **G4. Layer-skip self-speculation.** An approximate draft without training
  support, dominated by MTP. **NO-GO.**

For every exact method, the committed-token distribution must equal the
target's. Approximate variants must be labelled lossy and are out of scope.

### H. Prefix and session reuse (BlackroseAI)

**H1. In-session reuse.**
- **What.** Keep the KV across turns and evaluate only new tokens. Truncate with
  `rewind`, or replay when a rewind is beyond the ring.
- **Verdict.** **GO.**

**H2. Cross-request snapshots.**
- **What.** A snapshot at prefix P holds:
  - global KV `[0, P)`: 87 MB at P = 8K;
  - each local ring's last `1024` positions: about 111 MB, independent of P.
- **Cost.** Restore is one H2D copy, about 8 ms at 25 GB/s. Recomputing the same
  prefill costs seconds.

**Snapshot key.** The full hash of:
- the model and runtime numerics version;
- the KV format;
- the template and tokenizer hashes;
- the exact token ids;
- the placement-independent semantics version.

**Scope.** Per tenant, user and session. Never share across users unless the
prefix is explicitly marked public.

**Rule for every reuse path.** It re-runs every admission check of the path it
short-circuits: context limits, ownership, cancellation and format.

**Saved work** is measured as prefill time avoided, not tokens. Attention makes
that cost superlinear in P.

**Verdict.** **CONDITIONAL GO** after an isolation design review.

### I. Adaptive execution

**When each decision is made:**

| Decision | When |
| --- | --- |
| Residency, KV format, micro-batch sizes | Startup |
| Snapshot admission | Between requests |
| Prefill-versus-decode dispatch policy | Per batch, fixed rules |

Nothing moves weights during decode.

| Item | Verdict | Reason |
| --- | --- | --- |
| I1. Deterministic startup planner (budget, context, objective) | **GO** | |
| I2. Re-planning between requests | Deferred | Moving 0.4 GiB costs about 16 ms plus a risk of thrash |
| I3. Online auto-tuning | **NO-GO** until static policies are measured | |

## Likely counterproductive on this hardware

- KV spill to host RAM (C5).
- Uploading experts for single-token use, and small dynamic caches below the
  break-even hit rate (F7; the #28248 third-party data).
- LES prefetch beyond about 2 experts per layer per token (F7).
- Pinning all host weights.
- Independent draft models (G3).
- Paged KV for one sequence (C4).
- Any policy that moves weights during decode.

## Decisions

The maintainer decided D1–D7 on 2026-11-13. See the
[roadmap](optimization-roadmap.md#maintainer-decisions-2026-11-13).
The original questions were:
1. ADR 0002 direction.
2. The Gate 3 timing relative to refactors.
3. CPU expert numerics: f32 activations, or Q8_0 like the reference.
4. Threading a stream handle through the C ABI now.
5. Bead creation.
6. Bounded measurement runs: routing traces, PCIe bandwidth, attribution.
7. The documentation layout.
