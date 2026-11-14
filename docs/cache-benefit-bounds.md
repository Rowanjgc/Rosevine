# Cache benefit bounds and transfer overlap (M3)

Date: 2026-11-13. Bead: rv-e06-k0i.3 (RV-06 M3). Status: decision proposal.
**Recommendation: NO-GO now for a custom dynamic expert cache. Static first-N
placement remains the policy.** This document builds on the
[MoE cache audit](moe-cache-audit.md) (M1) and the
[placement economics](placement-economics.md) (M2). It implements nothing.
Raw traces and captures stay private.

## Method

1. **Cost model, calibrated on live pinned-cache cells.** The decode step at
   equal placement, with 10 host layers on Vulkan, is modeled as
   `T = c + 10·a + m·u`.
   - `c = 8.95 ms` is the GPU-side intercept from the M2 fit.
   - `m` is the number of misses per token.
   - Two M1 cells (1 GiB and 2 GiB) give `a = 0.137 ms` per cached layer per
     token (ID readback synchronization plus GPU expert work) and `u = 0.332 ms`
     per miss.
2. **Prediction error on cells not used in the fit:**

   | Cell | Predicted | Measured | Error |
   | --- | --- | --- | --- |
   | 2 GiB, cold only (1 repetition) | 14.73 ms | 14.72 ms | 0.05% |
   | 2 GiB, ubatch 32 | 13.67 ms | 13.69 ms | 0.1% |
   | 2 GiB, P = 2048 (intercept from that cell's cache-off control) | 13.16 ms | 13.01 ms | 1.2% |
   | HIP, 2 GiB (HIP intercept, Vulkan `a` and `u`) | 17.39 ms | 16.66 ms | 4.4% |

   The simulated hit rates also fall inside the live values (M1). The 4-token
   chunk cells were not modeled, because their CPU cost per chunk is not
   separable.
3. **Policies on held-out traces.** The traces are the S2 traces: prose and code
   holdout of 2,048 tokens each, 560 tokens of generated chat and the 214-token
   calibration prompt, all 30 layers. Policies run over the layers 0–9 host
   group with shared slots, at equal budgets. Counting starts after a warmup of
   512 tokens (or half the trace).
   - **Deployable:** LRU, SLRU (mirrors the pin), LFU, and a static
     expert-granular plan **fitted on a different domain** (prose → others,
     code → prose). No future routing enters these.
   - **Oracle:** Belady (farthest next use). It is an upper bound only and is
     never a deployable result.
4. **Overlap probe.** One supervised 0.75 s HIP run on the RX 7800 XT
   concurrently ran a 590 GB/s bandwidth-bound read kernel (a GEMV stand-in)
   and uploads of eight 3.19 MiB experts (25.5 MiB) on separate streams.

   | Host memory | Copy alone | Kernel slowdown when concurrent | Copy slowdown when concurrent |
   | --- | --- | --- | --- |
   | Pinned | 28.3 GB/s | +3.2% | +9.7% |
   | Pageable | 27.5 GB/s | +3.5% | +11.8% |

   - **Hardware overlap:** uploads overlap well with bandwidth-bound GPU work.
     A raw demand transfer of one expert takes 0.118 ms.
   - **Pinned cache:** its 0.332 ms per miss is mostly serialization. It
     synchronizes on the IDs, then uploads on the critical path.

## Policy results (steady state, misses per token / hit rate, range across 4 domains)

| Budget | LRU | SLRU | LFU | Static plan from another domain | Belady oracle |
| --- | --- | --- | --- | --- | --- |
| 1 GiB (319 slots) | 20.9–26.2 / 0.67–0.74 | 19.4–25.1 / 0.69–0.76 | 18.3–27.8 / 0.65–0.77 | 24.2–50.3 / 0.37–0.70 | 8.6–12.7 / 0.84–0.89 |
| 2 GiB (640) | 3.4–9.3 / 0.88–0.96 | 3.4–8.4 / 0.90–0.96 | 3.7–9.6 / 0.88–0.95 | 7.8–29.5 / 0.63–0.90 | 1.4–3.3 / 0.96–0.98 |
| 3 GiB (962) | 0.2–1.6 / 0.98–1.00 | 0.3–1.6 / 0.98–1.00 | 0.3–1.6 / 0.98–1.00 | 0.9–16.8 / 0.79–0.99 | 0.2–1.4 / 0.98–1.00 |

- **4-token verification batches** need about 2.5× more distinct experts per
  layer. Their hit rates fall by up to 13 points, while misses per token barely
  change.
- **Domain shift** (warm on prose, then 256 code tokens): LRU and SLRU stay at
  0.75–0.93. LFU collapses to 0.41–0.65. Frequency-based retention, static or
  LFU, does not survive a domain change.

## Synchronization-adjusted decode bounds (ms per token, Vulkan, 10 host layers)

The ranges below span the four domains.

| Budget | Static first-N (equal VRAM) | Pinned cache, SLRU | Same, half the bandwidth | Ideal demand fetch, SLRU | Hybrid: misses run on the CPU, SLRU | Oracle, CPU misses | Oracle without per-layer sync |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 GiB | **15.09** | 16.76–18.67 | 19.06–21.64 | 12.61–13.29 | 12.31–12.90 | 11.20–11.62 | 9.83–10.25 |
| 2 GiB | **13.04** | 11.44–13.10 | 11.83–14.09 | 10.72–11.31 | 10.66–11.18 | 10.46–10.66 | 9.09–9.29 |
| 3 GiB | **10.98** | 10.41–10.86 | 10.44–11.05 | 10.35–10.51 | 10.35–10.49 | 10.34–10.46 | 8.97–9.09 |
| 4 GiB (all experts fit) | **8.95** | 10.33–10.79 | 10.33–10.96 | 10.32–10.49 | 10.32–10.46 | 10.32–10.46 | 8.95–9.09 |

- **Ideal demand fetch:** uploads at the probe's raw 28.3 GB/s with no extra
  serialization.
- **Hybrid:** misses run on the CPU at 0.1025 ms per expert (the M2 slope / 8);
  hits run on the GPU; uploads happen in the background (+3% GPU slowdown,
  ignored here).
- **Uncertainty:** model error is 0.05–1.2% on Vulkan and 4.4% across backends,
  plus the domain range shown. The bounds are steady state; the live short
  session (M1) was slower, because it includes the cold fill.

## Decision

1. **Pinned llama.cpp cache: NO-GO as a default or as a comparator setting.**
   - It loses to static placement at 1 GiB, in cold or short sessions, and at
     full budget.
   - Its best case is about 12% faster than static at 2 GiB in a warm,
     locality-rich steady state.
   - Most of its miss cost is serialization overhead, not bandwidth.
2. **Custom Rosevine dynamic cache (hybrid or ideal demand fetch): NO-GO now,
   to revisit after the GPU intercept work.**
   - The deployable gain bound is 1.9–2.4 ms per token at 2 GiB, at most
     0.6 ms at 3 GiB, and *negative* once all experts fit.
   - The oracle adds only 0.2–0.5 ms beyond SLRU at 2 GiB. Better prediction
     cannot buy much more.
   - Rosevine's step at first-10 is 30.6 ms, and 13.4 ms of that is the
     GPU-side intercept gap to llama.cpp ([placement economics](placement-economics.md)).
     So a cache would recover at most about 6–8% of today's step, for slot
     banks, async uploads, eviction, a per-layer sync and new failure modes.
   - It also cannot help prefill-dominated work beyond the read-only copy path
     M1 measured.
   - The decision should be re-run when the intercept has shrunk. The relative
     value grows as the rest of the step gets faster.
   - Trigger: Rosevine decode within 20% of llama.cpp Vulkan at the same
     placement.
3. **Static-placement fallback (always in force):**
   - First-N host experts executed on the CPU (Q8_0 opt-in, 6 threads for
     efficiency).
   - N is the smallest count that fits the VRAM budget with the hard reserve.
   - Each CPU layer frees 408.4 MiB and costs 0.82 ms per token.
   - No cache, no prediction and no per-expert placement.
4. **Kept for later:**
   - The SLRU and Belady simulators and the calibrated cost model, as a private
     trace-replay tool.
   - The finding that hardware upload overlap is about 3%. Any future design
     must take uploads off the critical path to earn its gain.

## Limits

- **Single calibration point.** The model is fit on one host, one backend and
  one placement. Its HIP error is 4.4%.
- **Hybrid model not built.** Its CPU-miss cost assumes a linear per-expert
  share of the M2 slope, with no fixed per-layer overhead.
- **MTP and verification.** Real MTP acceptance was not run. Verification
  batches are fixed-token approximations.
- **Traces are from one engine.** They come from Rosevine routing. The pinned
  engine's routing agrees on 88–98% of expert sets in layers 0–9 (R2), and live
  hit rates match the simulated range.
