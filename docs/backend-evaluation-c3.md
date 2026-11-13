# Backend evaluation: Triton feasibility and host-language/FFI costs (C3)

Date: 2026-11-13
Bead: rv-e03-mmf.3 (C3); input to Gate 2 (rv-e03-mmf.4)
Status: Complete. Triton is viable on gfx1101 as a kernel generator. Host-language
FFI cost does not separate C++ from Rust. No language is selected here.

## Setup

| Item | Value |
| --- | --- |
| GPU | RX 7800 XT, gfx1101, wave32, same machine as [C1/C2](backend-evaluation-c1-c2.md) |
| Triton | 3.8.0 PyPI wheel (MIT), CPython 3.13, isolated venv. Bundled AMD backend and LLVM. Maintainer-approved download 2026-11-13. |
| Other Python | numpy 2.3.5 (BSD-3-Clause), for fixtures and references only |
| HIP | Fedora ROCm 6.4 (HIP 6.4.2), `hipcc --offload-arch=gfx1101` |
| Rust | rustc 1.98.1 (Fedora), edition 2024, std only, no crates |
| C++ | hipcc (clang) for the device and host benchmark |
| Power | COMPUTE profile for every timed run ([rv-19d](gate-1-review.md#power-profile-decision-rv-19d)) |

The venv uses about 0.9 GiB, almost all of it Triton's bundled compiler. Nothing
was installed globally. PyTorch was not installed, because Triton 3.8's AMD driver
imports torch only to look up the current device and stream. A 15-line subclass
supplies device 0 and the null stream from HIP directly; compile and launch paths
are unchanged. Raw fixtures, logs, compiled code objects and scripts stay private.

## Triton feasibility

Triton compiles and runs on native gfx1101 with no architecture override. It
reports `GPUTarget('hip', 'gfx1101', 32)`, and its AMD compiler recognises the
gfx11 family (WMMA and true16 paths).

### Exact-shape kernels

Three Q4_0 kernels were written from the GGUF block layout (f16 scale plus 16
nibble bytes) at the target model's shapes, with float32 activations:
- a decode GEMV;
- the same GEMV with expert indirection (`MUL_MAT_ID`);
- a 512-token prefill GEMM that dequantizes to f16 and multiplies with `tl.dot`.

Each was checked against a float64 numpy reference on random weights and inputs.
A small manual sweep (block sizes and warps) picked the configurations below.
Device time is the median of 5 event-bracketed loops after warmup.

| Case (shape) | Triton µs | Vulkan µs | HIP µs | Triton / Vulkan | Max relative error |
| --- | --- | --- | --- | --- | --- |
| Q projection GEMV, 2816 → 4096 | 29.1 | 9.2 | 18.8 | 3.2 | 1.1e-7 |
| K projection GEMV, 2816 → 2048 | 14.4 | 5.3 | 10.9 | 2.7 | 9.8e-8 |
| Attention output GEMV, 4096 → 2816 | 30.1 | 8.7 | 15.5 | 3.5 | 1.6e-7 |
| Expert gate+up `MUL_MAT_ID`, 1 token, 8 of 128 | 52.2 | 17.3 | 42.6 | 3.0 | 1.3e-7 |
| Expert down `MUL_MAT_ID`, 1 token, 8 of 128 | 27.1 | 23.8 | 64.5 | 1.14 | 9.8e-8 |
| Q projection GEMM, 512 tokens | 517 | 437 | 333 | 1.18 | 2.8e-4 (f16 WMMA) |
| FFN gate GEMM, 512 tokens | 289 | 281 | 224–230 | 1.03 | 2.8e-4 (f16 WMMA) |

The Vulkan and HIP columns are the C1 `test-backend-ops perf` medians for the same
shapes. Those include per-op synchronization, while the Triton loop overlaps
launches, so the comparison slightly favours Triton.

**Findings:**
- **Correctness:** every Triton kernel matched the reference. The GEMM error is
  the expected f16 rounding of `tl.dot` inputs.
- **Expert down projection:** Triton is within 14% of Vulkan, and 2.4× faster than
  llama.cpp's HIP kernel, on the same HIP platform. llama.cpp's HIP deficit on this
  op (C1) is therefore a kernel-quality gap, not a limit of the HIP platform.
- **Batched GEMMs:** within 3–18% of Vulkan, but behind llama.cpp's HIP MMQ.
- **Dense decode GEMVs:** 2.7–3.5× slower than Vulkan. The ISA shows why: nibbles
  are unpacked with `v_bfe`, converted to float32 and multiplied one element at a
  time. llama.cpp quantizes activations to q8_1 and uses packed integer dot
  products, which this first kernel does not.

### ISA, registers and compilation

| Kernel | VGPRs | Spills / scratch | Notable instructions |
| --- | --- | --- | --- |
| GEMV and MoE GEMV (best) | 49–50 | none | `v_bfe`, `v_cvt_f32_i32`, DPP reductions; no `v_dot` |
| Prefill GEMM, gather layout | 256 | 80 spilled, 320 B scratch at 4 warps | 32 `v_wmma` |
| Prefill GEMM, per-block layout | 253 | none | 32 `v_wmma_f32_16x16x16_f16` |

Data layout decides whether Triton spills. The first GEMM gathered scales and
nibbles per element and spilled. Loading each Q4_0 block contiguously and feeding
its low and high nibbles to two K = 16 WMMA products removed the spills and made
the best configuration 1.7–1.8× faster.

**Compile cost**, from 3 fresh processes with an empty cache, then the same cache
warm:

| Phase | Cold | Warm cache |
| --- | --- | --- |
| `import triton` | 0.29 s | 0.09–0.10 s |
| First GEMV call (includes HIP runtime init) | 0.35–0.36 s | 0.26 s |
| Each further kernel's compile | 0.08–0.25 s | ≤ 0.003 s |
| Whole process | 1.26–1.28 s | 0.65–0.67 s |

Compilation is fast and cacheable. Triton's output is an ordinary HIP code object
(`.hsaco`), which the FFI measurement below loads and launches with no Python.

## Host-language and FFI cost

**Method.** One native shared library (`librvk.so`, built with hipcc) exposes a
C ABI:
- opaque context and buffer handles, each with exactly one create and one
  destroy;
- upload, download and synchronization calls;
- a hipcc-compiled empty kernel;
- a launcher for the Triton-compiled Q4 GEMV code object above.

Every call returns 0 or a HIP error code, and an error string is always available.
The launcher checks buffer bounds, because the kernel itself does not. Five host
paths drive the same object, kernel, fixture and buffers:

| Host path | How it calls |
| --- | --- |
| C++ direct | HIP runtime calls inline, no wrapper (baseline) |
| C++ through the C ABI | calls `librvk` |
| Rust through the C ABI | `extern "C"` over `librvk`; RAII types whose `Drop` frees each handle; codes mapped to `Result`; every hot-loop call checked |
| Python ctypes through the C ABI | calls `librvk` (as automation) |
| Triton's Python launcher | the same kernel source and configuration, through Triton's own JIT dispatch |

There were three alternating rounds (C++, Rust, Python, then reversed) under
COMPUTE. Medians across rounds are shown with ranges.

| Host path | Submit, empty kernel | Submit, GEMV | Launch+sync, empty | Launch+sync, GEMV | 11 KiB up + 16 KiB down | GEMV device time |
| --- | --- | --- | --- | --- | --- | --- |
| C++ direct HIP | 0.70 (0.69–0.71) | 0.76 (0.75–0.78) | 16.1 (15.9–17.2) | 39.3 (39.3–44.0) | 51.0 (49.6–54.3) | n/a |
| C++ via C ABI | 0.70 (0.69–0.71) | 0.76 (0.75–0.77) | 16.5 (16.4–18.1) | 39.4 (39.2–44.4) | 50.9 (49.6–53.8) | 26.8 (26.8–30.2) |
| Rust via C ABI | 0.68 (0.68–0.71) | 0.75 (0.74–0.76) | 16.3 (16.3–16.6) | 39.2 (39.2–39.4) | 49.8 (49.4–51.2) | 26.8 (26.7–26.9) |
| Python ctypes via C ABI | 0.97 (0.96–0.99) | 1.20 (1.20–1.21) | 16.7 (16.7–16.7) | 40.1 (40.1–40.3) | 57.3 (57.0–57.4) | n/a |
| Triton Python launcher | 4.70 (4.64–4.77) | 8.53 (8.46–8.59) | 21.3 (21.3–21.4) | 48.4 (48.4–48.5) | n/a | 25.9 (25.9–26.0) |

All values are µs. "Submit" is host time per launch in 32-launch bursts after an
untimed synchronize, so it measures host overhead without queue backpressure.
"Launch+sync" is the full round trip. Device time is event-bracketed.

**Findings:**
- **Kernel and host costs separate cleanly.** The same code object has the same
  device time whichever host launches it (26.8 µs from C++ and Rust). Host submit
  cost is 0.7 µs. Launch-plus-sync latency (16 µs for an empty kernel) is
  dominated by HIP runtime synchronization and is the same for every native host.
- **C++ versus Rust: no difference beyond noise.** Rust over the C ABI matched
  C++ direct calls in every column, including with per-call error checks and RAII
  ownership. Choosing between them cannot rest on dispatch cost.
- **The C ABI layer is free.** C++ through the C ABI matched direct HIP calls.
- **Python is automation-only.** ctypes adds 0.3–0.45 µs per call and about 6 µs
  per copy pair. Triton's JIT launcher adds 4–8 µs per launch. A decode step runs
  hundreds of kernels; at 600 launches the Triton launcher alone would add about
  3–5 ms per token, against about 11.6 ms per token for the Vulkan baseline at
  86 tokens/s. Python stays out of any decode loop.
- **Triton kernels can ship without Python.** A Triton-generated code object runs
  from C++ and Rust through `hipModuleLaunchKernel`, given its argument layout
  (user arguments plus two scratch pointers) and its dynamic shared-memory size.
  Its compile-time specializations (here 16-byte divisibility of pointers and of
  some integers, and constexpr block sizes) become a launch contract the host
  must honour.
- **The error contract holds in every language.** A bounds-violating launch
  returns "invalid argument". A missing code object returns "file not found" and
  leaves the output handle null.

## Unrelated C1 follow-up: HIP GET_ROWS and SET_ROWS

C1 traced two HIP exact-shape faults to the test-file harness. llama.cpp's own
standard suites for those ops pass on the HIP build: `GET_ROWS` 216/216 and
`SET_ROWS` 159/159. This supports the harness-artifact explanation. The two
exact model shapes remain unverified through the harness, but the model ran
correctly in every C1 HIP run.

## Outcomes

| Candidate | Outcome | Evidence |
| --- | --- | --- |
| Triton 3.8 on gfx1101 as a kernel generator | **Viable** | Compiles and runs natively; all fixtures correct; WMMA used without spills once the layout was fixed; fast cached compiles; output launches from native hosts |
| Triton as a runtime dependency (Python in the loop) | **Rejected** | 4–8 µs per launch, a 0.9 GiB toolchain, and a torch-coupled driver needing a shim |
| Triton kernels in a hybrid runtime now | **Not justified** | No kernel beats the best existing implementation for its shape. The only notable result (MoE down, 2.4× over llama.cpp HIP) is still 14% behind Vulkan. Inclusion needs a measured advantage beyond noise plus the documented launch contract above. |
| C++ host | **Viable** | Baseline |
| Rust host over a C ABI | **Viable** | Same cost as C++ within noise; ownership and error contract demonstrated |
| Python | **Automation only** | Adds measurable per-call cost |

The decode-GEMV gap is a hypothesis for later kernel work (rv-e05-7n3), not a
Triton limit: q8_1 activations with integer `tl.dot` (RDNA3 has int8 WMMA) were
not tried.

## Limits

- **Kernel coverage.** Three kernel families at 7 shapes. Attention, norms,
  routing and Gemma-specific fusions were not ported. Best-of-sweep
  configurations, not autotuned.
- **Activation precision.** Triton kernels use float32 activations. llama.cpp
  uses q8_1. The data volumes and accumulation differ, so the timings compare
  implementations, not equal algorithms.
- **Workload.** One fixture per shape with random data. Weights fit in the 64 MiB
  Infinity Cache, as in the C1 per-op timings.
- **FFI scope.** One GPU, one process, the default stream, no multi-threading, no
  HIP graphs, no async error capture. Rust bindings were written by hand; no
  binding generator or crate was evaluated.
- **Not measured.** Vulkan host-language cost (equivalent C ABI wrapping over a
  Vulkan dispatcher). Triton's Vulkan or SPIR-V path does not exist upstream.
- **Unexplained.** Device time differed by about 3% between Triton's launcher
  (25.9 µs) and the identical code object launched natively (26.8 µs).
