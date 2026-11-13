# Performance parity and upstream reuse amendment

Date: 2026-11-13. Status: proposed implementation track; strategic direction
requested by the maintainer. No runtime replacement, dependency addition, kernel
port, GPU experiment or change to R3a is authorized by this document.

Performance parity is an engineering milestone before advanced optimization.
Specialization must earn measurable advantages; generic kernels may already be
best. Close the largest proven gaps first. This amends the external-comparison
restriction in the optimization roadmap and validation plan: early comparisons
are permitted as diagnostics, with untuned kernels identified. They are not
Gate 4 success claims. Existing R3a criteria, including internal-only attribution,
remain unchanged. ADR 0001 remains accepted; ADR 0002 remains Proposed.

## Workloads and milestones

Freeze a matrix before runs, separately for pinned llama.cpp Vulkan and HIP on
the same RX 7800 XT. Keep the historical v0.9.0 source pin
`cd959bae99ff4bed22bef2604f45241c09bbb8c8` for both. Newer revisions are a
separate audit/control series, never a silent baseline update. Pin Rosevine too.

Initial proposed cells: decode 128 target-only tokens at occupied depths 512,
2,048, 8,192 and 16,384; prefill 128, 512, 2,048 and 8,192 prompt tokens from an
empty cache. Measure TTFT on those prompts. Extend depth to 29K/32K only after
budget and attention feasibility checks. Freeze exact token IDs, BOS/template,
evaluation rows, microbatch, thread count and affinity, warmup and sampling in
the manifest. Use the experiment protocol's paired repetitions and uncertainty
analysis. Unsupported, unsafe or unmatched cells remain explicit, not omitted
from a milestone denominator after seeing results.

Use identical hashed GGUF artifacts and Q4_0 weights, matching K/V cache precision,
capacity, FlashAttention policy, target-only mode and equivalent actual expert
placement. Flag equality is insufficient: record tensor homes, resident bytes,
CPU-expert activation precision/repacking and CPU worker policy. If exact
placement, cache formats or numerical paths cannot match, report a separate
practical configuration comparison and the precise difference; do not call it
an execution-machinery parity cell. Include an equal-memory-budget comparison
when backend overhead prevents identical placement. No production router edits.

Keep two timing families:

- Engine-only prefill/decode: matching evaluation boundaries, synchronized GPU
  completion, same logits/readback policy, sampling excluded in both arms.
  Audit `tools/llama-bench/llama-bench.cpp` at the pinned revision before using it.
- User-visible generation and TTFT: request admission to first delivered token,
  including tokenization, prefill, logits, sampling and delivery in both arms.
  Separate model load/startup (cold) from loaded-session (warm) results. Use a
  matching generation harness; sampler-inclusive Rosevine time must never be
  divided by sampler-excluded llama-bench time.

For each workload/backend/timing family, publish throughput ratio
`Rosevine / reference` and latency ratio `reference / Rosevine`, so larger is
better. Keep raw values, units, median, p95 and paired confidence intervals.
Never average workloads or backends into one parity score. The relevant baseline
is specified before a run: HIP diagnoses same-backend machinery; Vulkan remains
ADR 0001's Gate 4 comparator. A Vulkan deficit cannot be hidden by beating HIP.

Proposed milestones apply to every predeclared supported cell independently:

1. **P90:** lower 95% confidence bound of the speed ratio ≥ 0.90.
2. **Parity:** point estimate ≥ 1.00 and lower bound ≥ 0.98 (predeclared 2%
   noninferiority margin). This means operational parity within that margin;
   strict statistical evidence of ≥ 1.00 is reported separately.
3. **Specialization advantage:** lower bound > 1.00 and a predeclared practical
   improvement (initial proposal: 5%) on the named workloads, with regressions
   elsewhere disclosed. No blanket claim from a single favorable cell.

Correctness must pass the existing S/T/N rules, exact-shape tests, layer checks,
teacher-forced KL and routing checks before performance acceptance. Do not demand
Vulkan token identity where its reference is nondeterministic. Record peak VRAM
(ledger plus device telemetry), peak RAM (RSS, mapped/file-backed and private
pages), sampling interval and missed-peak limitations. Report CPU process time,
core-equivalent utilization, threads, idle-host versus bounded-contention controls,
GPU pressure and paging. Resource metrics get per-cell ratios in their natural
units and explicit budgets; speed does not excuse reserve violations or unexplained
memory/CPU regressions. Existing run locks, live-condition checks, reserves,
sensor stops and private capture rules apply. No new experiments run here.

## Source inspection and reuse register

Source tree and selected implementation inspection used the frozen revision above
and upstream snapshot `9daa0aa22b8aa0143344d7549dee1f0e32f048bb` (v1.0.0-era;
258 commits ahead by the upstream compare API). Snapshot selection is not a
recommendation to upgrade. Full audit and gfx1101 measurements remain work.
All paths below are relative to [llama.cpp at the frozen pin](https://github.com/ggml-org/llama.cpp/tree/cd959bae99ff4bed22bef2604f45241c09bbb8c8).
A = reuse/direct port; B = adapt techniques; C = retain existing; D = specialize
only with a defensible measured benefit. These are provisional dispositions,
not findings that a port will win.

| Order / subsystem | Exact source paths at frozen pin | Initial disposition and decision evidence needed |
| --- | --- | --- |
| 1. HIP Q4_0 GEMV/GEMM and dispatch | `ggml/src/ggml-cuda/mmvq.cu`, `mmvq.cuh`, `mmq.cu`, `mmq.cuh`, `mmq-config-rdna3.cuh`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-hip/CMakeLists.txt` | A/B: isolate Q4_0 kernels and `ggml_cuda_should_use_mmvq/mmq`; measure exact shapes, batch crossover and activation quantization cost. HIP builds share CUDA sources; audit portability macros and transitive headers. Avoid importing an entire backend for one kernel. |
| 2. MoE grouping/execution | `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `ggml/src/ggml-backend.cpp`, `src/llama-moe-cache.cpp` | B: inspect indexed matmul, expert bounds and row scatter/gather. Preserve Rosevine routing/combine invariants; cache policy remains deferred. |
| 3. CPU SIMD quantized matmul | `ggml/src/ggml-cpu/arch/x86/quants.c`, `repack.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c` | A/B: Q4_0/Q8_0 vector dot and repack paths; existing f32 R3a path stays C until paired evidence. Upstream quantized activations are a numerical-profile difference, not a drop-in f32 replacement. |
| 4. Attention / FlashAttention | `ggml/src/ggml-cuda/fattn.cu`, `fattn-tile.cu`, `fattn-vec.cuh`, `fattn-common.cuh` | A/B: test local/global masks, ring boundaries, Q8 KV and D128/256, spills and scratch. Compare tiled generic kernels before D. |
| 5. Launch, graphs, synchronization | `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`, `ggml/src/ggml-hip/CMakeLists.txt` | B/C: retain executor/C ABI; inspect pooling, streams, graph capture/reuse and HIP guards. Graph support on gfx1101/ROCm 6.4 must be verified; graph replay stays evidence-gated. |
| 6. Allocation and buffer reuse | `ggml/src/ggml-alloc.c`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/common.cuh`, `ggml-cuda.cu` | B/C: adapt scratch lifetimes and workspace reuse; retain no-evaluation-allocation invariant and memory ledger. Full GGML allocator coupling needs separate review. |
| Control: Vulkan kernels | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-vulkan/vulkan-shaders/` | B: explain fastest-reference behavior and shape choices; no Vulkan runtime integration under ADR 0001. |

The first priorities combine existing CPU critical-path evidence (R3a) with the
known HIP MoE deficit in C1. They are inspection order, not a proven ranking of
Rosevine-versus-upstream gaps.

Relevant newer changes, pinned for follow-up inspection:

- [MoE MMQ/MMF layout](https://github.com/ggml-org/llama.cpp/commit/69ee2ecc0150ba7d1204a7f7393dfd651569db8c): grouping and locality.
- [Few-row MMQ, batches 2–24](https://github.com/ggml-org/llama.cpp/commit/0fb8f157702702f33f4e018de8701b2569a1bc28): dispatch crossover; snapshot adds `ggml/src/ggml-cuda/mmq-few.cu` and `.cuh`.
- [Fixed HIP BLAS workspace](https://github.com/ggml-org/llama.cpp/commit/432433637b6d06ece7d134d9d779c307969fd75b): repeated allocation overhead.
- [CPU execution of missed experts](https://github.com/ggml-org/llama.cpp/commit/e95426f16982054c379ca7bdc8186996dd22f0fe) and [avoid CPU waiting for cached experts](https://github.com/ggml-org/llama.cpp/commit/cf34d311332e362198f0a410c466d1e940287d02): scheduling techniques; no cache adoption implied.
- [RDNA3/4 Vulkan split-K](https://github.com/ggml-org/llama.cpp/commit/da3416d8077c55d68e45ce15babd999ac7ba1c49): separate newer Vulkan control.

Commit subjects/tree inspection identify candidates only; audit patches, compile
closure and actual gfx1101 applicability before selecting any import. NVIDIA-only
or CDNA-only improvements do not establish RX 7800 XT benefit.

For each candidate record A/B/C/D, full source commit, source files/symbols,
transitive dependencies, license notices, adaptation diff, build footprint,
correctness risks, measured benefit, owner and upstream-update cost. Preserve
upstream copyright and MIT permission text with copied/substantial adapted code;
check file-specific and third-party terms rather than assuming repository-wide
MIT covers every dependency. Rosevine stays GPL-3.0. No imported code in this change.

## Actionable comparison and optimization gates

The comparison report must contain one row per subsystem and workload, with
Rosevine ms/token or ms/prompt, HIP and Vulkan equivalents, instrument overhead,
absolute delta, uncertainty, overlap/critical-path contribution and unexplained
remainder. Required rows: CPU experts; GPU GEMV/GEMM; attention; host launch;
synchronization; H2D/D2H; quantization/dequantization; allocation/buffer management.
Separate routing, combine and output-head costs where needed. Unknown upstream
attribution is marked unknown, never filled by subtracting unrelated benchmarks.

Use R3a.2's spans/counters as input, plus comparable upstream instrumentation or
bounded isolated exact-shape tests. Calibrate instrumentation against uninstrumented
runs; require ≥90% nonoverlapping elapsed-time closure before ranking fixes.
Do not sum overlapping CPU and device times. Report wait as wait rather than
charging it again to kernels/transfers. Rank by recoverable critical-path ms,
confidence and integration cost. Existing evidence is insufficient for a complete
paired subsystem gap table today; collecting it is an explicit deliverable.

Every major optimization proposal and completion record must state:

- Target bottleneck and relevant pinned upstream approach, including reuse rejected.
- Matched baseline values, workload, correctness profile and noise bounds.
- Expected recoverable time and failure threshold, followed by actual paired result.
- Correctness impact and S/T/N/X verification; resource and other-workload regressions.
- Complexity, dependency footprint and ongoing maintenance cost versus measured gain.

Accept only benefits beyond predeclared noise/practical thresholds with correctness
and budgets intact. Negative results are valid evidence; disable/revert ineffective
features. LES, dynamic expert caches, graph replay and adaptive scheduling cannot
displace parity work unless a reviewed trace/control demonstrates a compelling
recoverable advantage and the foundational opportunity cost is explicit.

## Architecture review boundary

Compare three options in the reuse audit: current standalone specialized executor
with selected kernel ports; selective GGML components behind the existing C ABI;
and a limited upstream executor adapter as an isolated evaluation option. Compare
exact-shape and end-to-end speed, memory, dependencies, ownership/cancellation,
numerics, attribution and update burden. Current HIP/Rust executor remains the
working architecture. Runtime replacement, major dependencies, a backend change
or altered ADR commitments require a separate reviewed proposal/ADR. Independence
of model policy and execution ownership does not require independent low-level math.

## Proposed track and dependencies

Local Beads are open proposals, not an instruction to preempt R3a. No dependency
is added to an existing R3a Bead. Read-only source review may proceed independently;
benchmark scheduling waits for R3a.2 and R3a.3 to preserve the active sequence.

| Step | Deliverable | Prerequisites |
| --- | --- | --- |
| P1 (`rv-two`) | Frozen workload/metric contract and Vulkan/HIP baseline adapters; fixtures that reject mismatched boundaries/profiles. Done: [parity contract](parity-contract.md) | Existing benchmark manifest/metrics infrastructure; no R3a code edits |
| P2 (`rv-aa8`) | Full pinned reuse audit, A/B/C/D register and limited-integration comparison. Done: [reuse audit](reuse-audit.md) | None; consumes R3a findings when available |
| P3 (`rv-tkd`) | Paired baseline collection, numerical checks, resource/CPU controls and subsystem gap report | P1, P2, R3a.2, R3a.3; extend existing K1 rather than duplicate it |
| P4 (`rv-9zf`) | Ranked gap-closure plan and per-workload P90 → parity → advantage scorecards | P3; each implementation separately scoped, existing K2/K3 reused |

P1 must review existing private benchmark schema before extending it. Raw outputs
and profiler data stay private; the present public exporter lacks this matrix.
Do not bypass it or claim machine validation is human publication review. Design
reviewed public summaries separately if a schema change proves necessary.
