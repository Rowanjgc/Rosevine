# Optimization validation plan

Date: 2026-11-13. Status: proposal for review. This plan **extends** the
[experiment protocol](experiment-protocol.md). It does not replace it. Run
identity, sampling counts, paired statistics, run envelope and publication rules
stay as defined there and in the [run policy](development/rule-audit.md).

This plan adds:
- correctness classes for optimizations;
- a decomposition of time and memory into separately measured quantities;
- an optimization-specific benchmark matrix;
- the rules that keep untuned-HIP results apart from the Vulkan baseline.

## Baselines

Every result names exactly one baseline. A speedup over a slow R2 path is
reported as an *internal ablation*. It is never presented as progress against
the external baseline.

| Label | What | Use |
| --- | --- | --- |
| `vk-ref` | Pinned llama.cpp Vulkan reference with the matching preset (ADR 0001 decision 2) | The only external performance baseline (Gate 4) |
| `r2-ref` | Rosevine R2 HIP, `first_n(10)`, untuned kernels, at a fixed commit: about 13.4–14.0 tokens/s decode, 72 ms/token (S1; the earlier 4.1 was contaminated) | Internal ablation baseline; correctness reference for placement and scheduling changes |
| `r2-cpu` | Rosevine CPU executor | Numerics reference for host paths |
| `tuned-k` | Rosevine after a K2/K3 kernel milestone | Baseline for overlap, cache and prefetch studies (see below) |

## Correctness classes

Every optimization declares one correctness class before any measurement.

| Class | Meaning | Check |
| --- | --- | --- |
| **S (semantic identity)** | The same operations in the same order; only location or timing changes | Bitwise logits and intermediates against the pre-change build on synthetic and real fixtures. Examples: residency plans with equal per-path numerics, device routing selection, resident-first ordering, snapshot restore |
| **T (tolerance)** | Summation order or kernel changes, same arithmetic precision | Operator relative error ≤ 1e-5 against the CPU path at exact shapes; layer-0 agreement ≤ about 1e-2 against the reference; KL band. Examples: SIMD CPU kernel, tuned GEMV, tiled attention |
| **N (numerics change)** | Precision or quantization point changes | Everything in T, plus the teacher-forced KL band against both references, the fidelity corpus, and long-context retrieval for KV changes. Requires an explicit approval of the profile. Examples: Q8_0 CPU activations, global V-only KV, Q4 KV, f16/i8 WMMA activations |
| **X (exact speculation)** | Committed tokens have the target distribution | Greedy: identical committed tokens to target-only decode. Sampled: distribution tests per S1 (forced acceptance fixtures 0–3, natural runs) |

Selection identity is mandatory in every class: the same expert ids for every
token and layer as the paired baseline. Any selection flip in class S or T is a
defect. In class N, flips are counted and reported, and their effect on the KL
band is shown.

## Quantities to separate

Each quantity has one measurement method. A span is a host-monotonic interval;
a device event is a GPU timestamp. "Counter" means counted in code, not sampled.

| Quantity | Method | Notes |
| --- | --- | --- |
| CPU expert time | Host span around each host-expert call, by layer | Also a CPU-only microbenchmark at exact shapes (as in research F4), with cold- and warm-cache variants |
| GPU expert time | Device events around expert dispatch per layer | Events serialize lightly; calibrate the overhead (on vs off) |
| Attention time | Device events per layer, local and global reported separately | Report by context length |
| Routing overhead | Device events for router plus top-k; host span for any host selection | Includes the readback when present |
| CPU/GPU transfer | Counter for bytes and count by direction; device events on copy | Application bytes are a proxy for bus traffic (protocol) |
| Synchronization | Counter of blocking calls; host span of time blocked | Distinguishes waiting from work |
| Dispatch overhead | Counter of launches; host span of submission | Compare with the C3 figure of 0.7 µs per submit |
| Allocation | Ledger planned versus allocated; device free before and after | Evaluation must perform no allocation (R2 invariant) |
| Cache hit and miss cost | Counters of hits, misses, bytes and evictions; span per miss | Only if a cache exists |
| Prediction quality | Logged predictions against the actual selection: recall at top k by layer | Offline from traces first |
| Prediction overhead | Device events for the extra router GEMV | Zero when disabled |
| Actual overlap | `1 − elapsed / Σ(non-overlapped spans)` per layer, from events on both streams | Report achieved overlap, never theoretical |
| KV memory | Ledger KV class against analytic formula; measured device usage | By capacity and occupancy |
| Prefill, decode, TTFT | Protocol metric definitions | Cold and warm reported separately |
| Peak VRAM and RAM | Protocol telemetry (100 ms sampling); ledger high-water | RSS and file-backed pages split; swap in and out |

**Closure rule** (from K1): at least 90% of elapsed loop time must be accounted
for by non-overlapping spans or named gaps before an attribution is used to rank
work.

## What can be measured now

These results are valid with untuned HIP kernels, because they do not depend on
kernel speed, or the dependence is explicit.

- **Routing statistics.**
  - Per-layer recall of predictors, locality, unique experts per batch and
    popularity drift across domains.
  - These are model properties, collected from CPU or HIP traces.
- **CPU expert path cost and its bandwidth bound.** CPU-only.
- **Effective PCIe bandwidth and latency.**
  - Pageable and pinned, at expert-sized and layer-sized transfers.
  - Copy-only microbenchmarks.
- **Host overhead.** Synchronization and launch counts, time blocked, allocation
  and ledger accuracy.
- **Correctness of every class S change.**
- **Fraction-of-time attribution.** Absolute GPU numbers will shrink later, so
  report fractions *and* absolutes.
- **MTP acceptance.** A model property, measurable on `vk-ref` (S1).
- **Prefix reuse correctness**, and the prefill time saved when both arms use
  the same kernels. Report it as relative, with untuned kernels named.

## What must wait

- **Wait for `tuned-k`.**
  - Any end-to-end tokens/s or TTFT comparison against `vk-ref`.
  - Overlap, prefetch and dynamic-cache benefit. Slow kernels inflate the GPU
    time that can hide transfers, so benefits measured on R2 overstate what a
    tuned runtime would see.
  - Graph-replay benefit. Launch overhead relative to kernel time changes with
    tuning.
- **Wait for the kernels and C1.** Context feasibility above about 32K at prefill
  batch 512: R2's score scratch prevents it (research F8).
- **Wait for the Q2 fixtures.** Long-context quality claims.

## Benchmark matrix

**Workloads** come from the protocol suite: short 1900, medium 29000, the prefill
sweep and occupied-context points. Two additions:
- a **multi-turn assistant workload**: system prompt plus tool schema plus 3–5
  turns, synthetic and redistribution-safe, for prefix reuse;
- a **code-heavy workload** for n-gram drafting.

**Factors.** Each study varies one factor around a fixed control. The full
factorial is not run.

| Study | Varied | Held fixed | Primary metric | Cold/warm |
| --- | --- | --- | --- | --- |
| Attribution (K1) | Context 1900 / 29000; phase | `r2-ref` | Span breakdown (closure ≥ 90%) | Both |
| CPU kernel (A3) | Threads 6/10/12; numerics profile | Placement `first_n(10)` | CPU expert ms per layer; decode tokens/s (ablation) | Warm; cold-page variant |
| Placement (A1) | `first_n(N)` against popularity at equal bytes: 0.4 / 0.8 / 1.6 / 3.2 GiB | Kernels, context | Decode tokens/s; CPU evaluations per token; handoffs per token | Warm |
| Device routing (A2) | On / off | All else | Blocking calls per token; decode tokens/s | Warm |
| PCIe (M2) | Pageable / pinned; 3.19 MiB and 0.4 GiB; H2D and D2H | Idle GPU | GB/s, latency p50/p95 | n/a |
| Prefill streaming (A7) | Off / on; micro-batch 64 / 256 / 512 / 1024 | Placement | Prefill tokens/s, TTFT | Cold and warm |
| Tiled attention (C1) | Naive / tiled; contexts 4K–64K | Placement | Attention ms; scratch bytes; feasibility | Warm |
| Global V-only KV (C2) | Off / on | Everything | KL band; retrieval accuracy; KV bytes | n/a |
| Prefix reuse (H) | Off / in-session / snapshot | Workload, kernels | TTFT and prefill time saved per turn | Cold first turn; warm later |
| Speculation (G) | MTP off / max 3; n-gram on / off; placement N 0 / 2 / 8 | Workload | Committed tokens/s; acceptance; verification cost | Warm |
| Cache/LES (A6/B3), only if M3 is GO | Slots 0 / 64 / 128 / 256; prefetch off / top 12 / top 16 | `tuned-k`, equal VRAM | Decode tokens/s; hit rate; wasted bytes | Cold and warm |

## Controls and contention

**Single GPU user.**
- Only one agent runs GPU work at a time.
- A GPU study first checks that the production router is idle and that no other
  Rosevine run is active, then holds a run lock (a lock file under the private
  run directory) for the whole window.
- The existing COMPUTE power window (`scripts/gpu_power_window.py`) and the
  router-coexistence rules from Epic 2 apply unchanged.

**Router contention.**
- Record router GPU activity and VRAM before and after each run.
- If the router becomes active during a run, cancel the trial, keep it, and mark
  it contaminated.

**CPU contention.**
- Record load average and CPU frequency.
- CPU expert studies need an idle host. Report thread counts and SMT use
  explicitly.

**Memory state.**
- Record swap use, MemAvailable and page-fault counts for every CPU expert run.
- Report cold-page and warm-page variants separately (research F5).

**Order.** Run A/B pairs in randomized or ABBA order, with a second-session
reproduction before any gate claim (protocol).

**Concurrent workloads.** Never run a CPU reference or another model process
during a Rosevine timing run. S1 traced a 3× slowdown to exactly that, so
record the process list in the private manifest.

## Acceptance and gates

**Correctness gate.** Every change merges only with its correctness class
satisfied.

**Ablation claim.** Report the paired interval against `r2-ref`. An ablation
result is labelled *internal* and does not count as Gate 4 evidence.

**Gate 4 claim.** Unchanged:
- at least 5% median benefit in a named primary metric, with a paired 95%
  interval above zero;
- the comparison is against `vk-ref` or the defined unchanged control;
- reproduced in a second session;
- every regression reported.

**Negative results.**
- A technique that fails its benefit criterion is removed from the default plan
  and recorded as a finding.
- Its flag may stay only if it is still tested.
