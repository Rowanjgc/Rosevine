# 0001: Primary backend and runtime strategy

Status: Accepted
Date: 2026-11-13
Human acceptance: maintainer, 2026-11-13, at Gate 2 review (option "HIP + Rust host").
PR: #3 (https://github.com/Rowanjgc/Rosevine/pull/3)
Supersedes: none

## Context and problem

Gate 3 needs a minimal Gemma-specific reference runtime (rv-e04-398). It must run
Gemma 4 26B A4B QAT (Q4_0) target-only text on one RX 7800 XT (gfx1101, 16 GiB)
with experts placed across the GPU and the CPU. It must be checked layer by layer
against the frozen references ([Gate 1](../gate-1-review.md),
[divergence review](../reference-divergence-review.md)).

Before anything is built, two choices have to be made:
- the primary GPU execution path;
- how host code, kernels and automation divide across languages.

Constraints:
- the production llama.cpp router stays untouched;
- GPL-3.0 is preserved;
- correctness comes before speed, and no performance win is required to choose a
  reference path;
- one primary backend, unless a hybrid's costs are justified by measurement.

## Options considered

| Option | Summary |
| --- | --- |
| A. HIP primary | HIP C++ kernels compiled ahead of time for gfx1101; ROCm runtime; Vulkan llama.cpp kept as the performance baseline |
| B. Vulkan primary | GLSL/SPIR-V compute pipelines through RADV; the backend with today's fastest decode |
| C. Hybrid HIP + Vulkan | Use each backend where it is faster (Vulkan decode, HIP prefill) |
| D. Triton-centred | Triton-generated kernels as the main kernel source |
| E. NO-GO / PIVOT | No custom runtime yet; keep pinned llama.cpp as the experimental platform |

Host language, applicable to A to D: C++ throughout, or a Rust host calling
kernels through a narrow C ABI. Python is considered for automation only.

## Evidence

Sources: [C1/C2](../backend-evaluation-c1-c2.md) and
[C3](../backend-evaluation-c3.md). The protocol is the Gate 1 frozen run rules
under the COMPUTE power profile. HIP builds were made from the same pinned
llama.cpp v0.9.0 source as the Vulkan reference.

| Criterion | HIP | Vulkan | Source |
| --- | --- | --- | --- |
| Builds and runs the full model, including MTP | yes | yes | C1, C2 |
| Exact-shape op conformance (105 model ops) | 103/105; the 2 harness faults pass the standard suites (GET_ROWS 216/216, SET_ROWS 159/159) | 105/105 | C1, C3 |
| Layer-0 agreement with the CPU reference | 7.4e-3 | 7.7e-3 | C1 |
| Teacher-forced KL vs CPU (mean / top-1) | 0.080 / 92.7% | 0.056 / 93.4% | C1, band about 0.08 |
| Greedy output identity across 8 sessions | deterministic | 3–8 variants per fixture | C1 |
| Target-only decode (llama.cpp kernels) | 0.73–0.76× Vulkan | baseline | C1 |
| MTP decode | 0.96× | baseline | C1 |
| Prefill at 16k depth / MTP-path prefill | 1.03–1.05× / 1.07× | baseline | C1 |
| Single-token MoE expert ops (llama.cpp kernels) | 2.5–2.7× slower | fastest | C1 |
| Same MoE down op, first Triton kernel on HIP | 27.1 µs, 2.4× faster than llama.cpp HIP | 23.8 µs | C3 |
| Register spills | flash attention at D 128/256 spills heavily | none in the model's pipelines | C1, C2 |
| VRAM (llama.cpp) | about +0.85 GiB | baseline | C1 |
| Startup | about 15 s | about 5 s | C1 |
| Kernel language and compilation | single-source C++, ahead of time; Triton output loads the same way | GLSL to SPIR-V, compiled and cached at runtime | C1–C3 |
| Profiling available here | timestamps only; rocprof not installed | timestamps, pipeline statistics | C1, C2 |
| Host submit cost per launch | 0.7 µs (C++ and Rust equal) | not measured | C3 |

Host-language evidence (C3):
- C++ direct calls, C++ through a C ABI and Rust through the same ABI are
  equal within noise for submit cost, round trips, copies and device time.
- Python ctypes adds 0.3–0.45 µs per call.
- Triton's launcher adds 4–8 µs per launch.

**Hypotheses, not findings:**
- that custom HIP kernels can close the decode gap on dense GEMVs;
- that HIP's determinism will hold for a custom runtime;
- that the 0.85 GiB VRAM overhead belongs to llama.cpp's HIP backend rather than
  the platform.

## Tradeoffs

**HIP (A):**
- **Correctness tooling.** Deterministic greedy decoding makes output identity a
  usable regression control on top of the KL band and layer-0 checks. Vulkan
  cannot offer that.
- **Kernels.** HIP C++, hipcc and Triton output all produce code objects launched
  the same way, so kernel experiments share one host path. llama.cpp's MIT ggml
  HIP kernels can serve as reference implementations.
- **Costs.** Today's decode with llama.cpp kernels is 24–27% slower, and the gap
  must be earned back with custom kernels. Startup is slower and VRAM use is
  higher (in llama.cpp). Fedora ROCm sits outside AMD's support matrix.
- **Profiling.** There is no hardware profiler yet; rocprof availability is
  unverified on gfx1101.

**Vulkan (B):**
- **Speed.** The fastest decode and kernels without spills exist today.
- **Costs.** There is no output-identity control. Host code carries descriptor,
  synchronization and pipeline complexity. Kernels are written in a separate
  shader language with runtime compilation. Triton output cannot be used.

**Hybrid (C):** two runtimes, two kernel sets, and transfers or synchronization at
every boundary. The measured advantages (HIP prefill +3–7%, Vulkan decode) are
smaller than the complexity, so it is not justified now.

**Triton-centred (D):** viable as a generator, but no kernel beat the best
existing implementation, and Python in the loop is too expensive.

**NO-GO (E):** zero cost, but it gives up the research question that Gates 3 to 5
exist to answer. The correctness and coverage evidence does not force it.

**Rust host versus C++ host:**
- **Rust:**
  - Ownership and error contracts are compiler-checked for the state machines
    that later work needs: GGUF/mmap validation, KV rollback and MTP.
  - The FFI cost is zero as measured.
  - Costs: two toolchains, hand-written HIP bindings, and build glue (hipcc
    invoked from the build) that has to be maintained.
- **C++:** one toolchain and direct reuse of llama.cpp code, but no compiler help
  with lifetimes across a growing host runtime.

## Decision

Accepted by the maintainer on 2026-11-13 as proposed, including the Rust host.

1. **Primary backend: HIP (option A)** on Fedora ROCm 6.4. Kernels are compiled
   ahead of time for native gfx1101. This is the reference path for Gate 3.
2. **Performance baseline and fallback:** the pinned llama.cpp Vulkan reference.
   Gate 4 benefit is measured against it. No Vulkan code goes into the new
   runtime, and no hybrid is built.
3. **Host language: Rust**, calling HIP C++ kernels through a narrow C ABI that
   follows the C3 contract:
   - opaque handles, each with one owner and one release;
   - integer error codes with an always-available message;
   - bounds checked at the ABI before launch;
   - no exceptions or panics crossing the boundary.

   C++ host code is the stated alternative if the maintainer prefers a single
   toolchain. The evidence does not distinguish the two on cost.
4. **Triton** may be used offline as a kernel generator. A Triton kernel enters the
   runtime only as a code object, with a measured advantage beyond noise over the
   best alternative, and with its launch contract documented. Python never runs in
   the runtime loop.
5. **Python** is for automation, fixtures and analysis only.

Accepting this ADR does not authorize implementation. R1 (rv-e04-398.1) still
needs its own go-ahead.

## Consequences

- **R1/R2** build a Rust (or C++) host with HIP kernels. The first kernels may port
  llama.cpp's MIT ggml HIP kernels to make Gate 3 correctness reachable before any
  specialization.
- **Gate 3 checks** add HIP greedy output identity to the tolerances from the
  divergence review: the KL band at or below about 0.08, layer-0 agreement at or
  below about 1e-2, and exact-shape conformance.
- **Build and CI** need hipcc and ROCm devel packages (installed 2026-11-12) and a
  Rust toolchain (present). `./scripts/check.sh` will need build and test steps
  for these once code exists.
- **K1 (critical-path attribution)** should first check whether rocprofv3 works on
  gfx1101; installing it needs approval.
- **MoE expert kernels** are the first optimization target: C1's largest HIP
  deficit, and C3 showed it is a kernel-quality gap.

## Revisit conditions

- After K2, custom HIP decode kernels for the core GEMV and MoE ops stay more than
  10% slower than the Vulkan reference at equal correctness. Re-evaluate Vulkan
  as the primary backend.
- The HIP runtime's determinism or correctness regresses: output identity lost,
  KL above the band, or layer-0 agreement outside tolerance without a kernel
  explanation.
- Fedora ROCm updates break gfx1101 builds, or VRAM overhead blocks the target
  placement or context sizes required by Q2.
- Rust–HIP build glue or bindings cost more maintenance than their safety
  benefit. Switch the host to C++ under a superseding ADR.
- A Triton kernel shows a reproducible advantage beyond noise. Consider adding
  it under point 4.
