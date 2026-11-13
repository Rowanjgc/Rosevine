# 0002: Expert residency and scheduling

Status: Proposed
Date: 2026-11-13
Human acceptance: pending. On 2026-11-13 the maintainer gave conditional approval
of the direction, with the ADR to stay Proposed until its assumptions,
synchronization costs and correctness implications are reviewed.
Supersedes: none

## Context and problem

[ADR 0001](0001-primary-backend-and-runtime-strategy.md) left expert memory
placement undecided. R2 implements **static first-N placement**: the routed
experts of layers `0..N` stay in host memory and run on the CPU
([R2](../runtime-r2-execution.md)).

Constraints:
- **Spill is unavoidable.** On the RX 7800 XT, all target weights (13.27 GiB) do
  not fit beside KV and scratch. The derived spill is about 0.3 GiB at 16K and
  about 2.8 GiB at 256K
  ([research F8](../optimization-research.md#f8-vram-budget-and-the-kvscratch-cost-of-r2s-allocator-derived-from-recorded-values)).
- **Granularity is unchosen.** We have to decide which granularity of residency
  the runtime supports, where misses execute, and whether caching and Lookahead
  Expert Scheduling (LES) belong in the design. The design must not commit to
  unmeasured dynamic machinery, and must keep a clean path to it.

Correctness requirements:
- Placement and scheduling never change which experts contribute, their routing
  weights or the combine order.
- R2 showed that placement does not change semantics. N = 10 and N = 20 produce
  identical selections, and the
  [follow-up](../optimization-research.md#follow-up-scrutiny-2026-11-13) found
  identical 200-token greedy outputs. That property must hold for every
  residency plan.

## Options considered

| Option | Summary |
| --- | --- |
| A. Keep first-N layer placement as an integer | R2 as is |
| B. Static residency plan per (layer, expert), first-N by default | Immutable plan per session; first-N is one constructor; the CPU executes host-resident experts |
| B′. B with a popularity-ranked per-expert policy as the default | Least-used experts on the CPU |
| C. B plus a dynamic GPU slot cache | LRU/SLRU slots fed by uploads on miss, as in pinned llama.cpp `--moe-cache-mib` or llama.cpp discussion #28248 |
| D. C plus LES prefetch | Router prediction drives async uploads |
| E. Stream everything to the GPU | Upload every needed expert for every batch |

## Evidence

Source: [optimization research](../optimization-research.md). The S-items are
the follow-up scrutiny, measured on this workstation with serialized and
supervised GPU runs.

- **S1.**
  - Corrected `r2-ref` decode at N = 10 is about 13.4–14.0 tokens/s. The recorded
    4.09 tokens/s was contaminated by contention.
  - CPU-expert layers cost 5.40 ms each in situ, about 75% of token time.
- **S2.** Lookahead from the residual entering the target layer (p1) has 0.70–0.79
  recall at 8, but its lead time (the target layer's attention, about 110 µs at
  target speed) is shorter than one 131 µs upload.
  - Predictors with usable lead times (pmid, p2) have 0.53–0.73 recall at 8.
  - Their upper-bound saving is about 1 ms per token, with 13–17 false transfers.
- **S3.**
  - Popularity placement does not transfer across domains. Fitting on code and
    scoring on chat gives 25.9 CPU evaluations per token against 16 for first-N.
  - It multiplies handoff layers 2–8×.
  - Each host layer currently carries at least 0.6 ms of in-situ overhead beyond
    its kernel (unattributed).
  - Break-even needs a handoff under about 25 µs.
- **S4.** A persistent-pool AVX-512 f32 kernel does a decode layer's 8 experts in
  0.54 ms (cold), against 4.72 ms today, with unchanged f32 semantics.
- **F7, with measured PCIe.**
  - Pinned PCIe moves 25.5 GB/s per expert (131 µs) and 28.4 GB/s for a layer
    (14.7 ms).
  - A one-token decode miss is about 2× cheaper to compute on the S4 CPU path
    (67 µs) than to upload.
  - Prefill batches favour bulk upload.
- **External.**
  - In #28248, small slot caches lost to no cache, and a stale slot map caused
    wrong weight reads.
  - The OpenBMB LES numbers come from SSD-backed devices and do not transfer to
    this platform.

**Hypotheses, not findings.**
- That the S3 per-host-layer residual is reducible.
- That a domain-adaptive cache (C) could exploit the strong in-domain locality S3
  shows.
- That prefill streaming (E for prefill only) improves TTFT.

## Tradeoffs

- **A.** Simplest and fewest handoffs. But it bakes an integer into every
  interface, so any later cache or experiment needs a refactor.
- **B.**
  - **Cost:** a plan type and per-expert dispatch.
  - **Reach:** `first_n` is exactly A, so there is no regression risk.
  - **Value:** it keeps C, D and experiments possible without a later refactor.
- **B′.** Rejected as a default by S3: fragile across domains, with more
  handoffs.
- **C.**
  - **For:** could exploit in-domain locality.
  - **Against:** high complexity (async copies, slot lifetimes, stale-map
    hazards), VRAM taken from KV, and an unproven benefit here.
- **D.** All of C's machinery for at most about 1 ms per token on current
  evidence (S2).
- **E.** Right only for large prefill batches.

## Decision

Proposed. Acceptance requires human review.

1. **Adopt B with first-N as the default policy.**
   - A `ResidencyPlan` maps each (layer, expert) to `Device(slot)` or `Host`. It
     is immutable for the session and checked against a `MemoryLedger` before any
     allocation.
   - Constructors: `first_n(n)` (reproduces R2 bitwise; the default),
     `all_device()`, and `explicit(list)` for experiments only.
   - A non-layer-contiguous policy may become a default only with held-out,
     cross-domain evidence and measured handoff costs that beat first-N.
2. **Host-resident experts execute on the CPU during decode and verification.**
   - Selection and inputs move to the host once per such layer, in one copy.
   - Per-pair outputs return to the device `y` buffer.
   - The existing slot-ordered combine is the only merge point. This keeps the
     result independent of where each expert ran.
3. **Large-batch prefill may stream host-resident experts to the GPU** as a
   separate, phase-specific policy. It is enabled only after an A/B run shows a
   TTFT benefit. Until then, prefill uses rule 2.
4. **Routing moves to the device.** The host receives selections only when host
   experts or observers need them. Selection must be identical (integer ids) to
   `select_experts`.
5. **Resident-first ordering** applies within a layer once concurrent CPU/GPU
   execution exists. Ordering never alters the combine order.
6. **No dynamic slot cache and no LES prefetch are built now.**
   - pmid and p2 predictions may be computed and logged, never acted on, to gather
     held-out accuracy and miss statistics.
   - A cache or prefetch requires an amending ADR backed by M3 (rv-e06-k0i.3)
     trace replay. The comparison baseline is B with the S4-class CPU kernel at
     equal VRAM, measured on tuned kernels.
7. **A clean path to C and D**, preserved now:
   - expert addresses are resolved through a per-layer slot table owned by
     `residency`, never baked into callers;
   - the table carries a generation counter checked at dispatch;
   - any future slot is reused only after the events of every reading kernel have
     completed;
   - prefill never admits to decode slots;
   - errors and cancellation invalidate the whole table.
8. **Streams without a public ABI change** (maintainer decision D4):
   - all device work goes through an executor-owned queue handle on the Rust side,
     which wraps the null stream today;
   - when a consumer for asynchronous transfer exists, stream-taking entry points
     are *added* to the C ABI alongside the existing ones;
   - the existing signatures stay unchanged.

## Consequences

**Code.**
- `HipConfig.cpu_expert_layers` becomes a `ResidencyPlan` (default `first_n`),
  and `DevLayer.experts` becomes a slot table.
- The R2 tests remain valid through `first_n`.

**New tests.**
- Plan permutations are bitwise equal on synthetic weights when per-path
  numerics are equal.
- Device-routing selection is identical to `select_experts`, including tie
  fixtures.
- Ledger planned versus allocated.

**Benchmarks.** Manifests record the plan, not N, and comparisons hold VRAM bytes
equal.

**Beads.**
- Informs M4 (rv-e06-k0i.4), which already requires an ADR.
- M3 keeps the authority to propose C or D.
- R3a (CPU kernel and attribution) is independent of this ADR's acceptance.

**Out of scope.** Learned routing, expert pruning or any precision change.
Q1 (rv-e08-nhh.1) owns those.

## Revisit conditions

- The S3 per-host-layer residual is attributed and reduced below about 50 µs, and
  held-out cross-domain traces still show non-contiguous placement beating
  first-N at equal bytes.
- M3 shows a dynamic cache or LES prefetch beating B plus the fast CPU kernel,
  with a paired 95% interval above zero under the
  [protocol](../experiment-protocol.md).
- A tuned decode leaves a lead window long enough for the p1 predictor to stage
  transfers.
- Context or KV changes (for example global V-only KV) shrink the spill below one
  layer. Placement then matters less than kernels.
