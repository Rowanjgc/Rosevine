# Pinned llama.cpp reuse audit (P2)

Date: 2026-11-13. Status: source audit for the [parity track](performance-parity.md),
step P2 (`rv-aa8`). Read-only inspection: no code was imported, ported or
benchmarked, and no recommendation here is a finding that a port will win.
Every expected benefit is an upper bound or unmeasured until P3 pairs it. A
generic kernel is not assumed slower than a specialized one, or the reverse.

## Sources

| Tree | Revision | Use |
| --- | --- | --- |
| Frozen reference | `cd959bae99ff4bed22bef2604f45241c09bbb8c8` | All paths and symbols below unless marked |
| Newer snapshot | `9daa0aa22b8aa0143344d7549dee1f0e32f048bb` | 258 commits ahead; the frozen pin is its ancestor. Separate control series only. |
| Rosevine | this branch (R3a.3 included) | `runtime/hip/rvhip.hip` (469 lines), `runtime/src/hip_exec.rs`, `runtime/src/expert_kernel.rs` |

Paths are relative to llama.cpp's root. The shared CUDA sources build for HIP
through `ggml/src/ggml-cuda/vendors/hip.h` and `ggml/src/ggml-hip/CMakeLists.txt`.

**License closure.** The frozen tree's root `LICENSE` is MIT ("Copyright (c)
2023-2026 The ggml authors"). Every file audited below sits under `ggml/src`
or `src` with no file-specific notice. The only other bundled license file in
the tree is `licenses/LICENSE-jsonhpp`, which none of these files include.
Copying or substantially adapting any of this code into GPL-3.0 Rosevine is
permitted. It must keep the MIT copyright and permission notice in the copied
file or in a third-party notice, and it needs its own scoped bead. ROCm headers
and libraries (HIP, hipBLAS, rocBLAS) are already Rosevine build dependencies
under ADR 0001. A port of the kernels below adds no new library.

## Disposition key

- **A:** reuse or directly port the kernel.
- **B:** adapt the technique in Rosevine's own code.
- **C:** retain Rosevine's existing implementation.
- **D:** specialize only if there is a defensible measured benefit.

A disposition records what P3 and P4 should measure first. It is not an
integration decision.

## Subsystem register

### 1. HIP Q4_0 GEMV/GEMM and dispatch

| Item | Detail |
| --- | --- |
| Upstream | `ggml/src/ggml-cuda/mmvq.cu`: `mul_mat_vec_q<type,…>` (with `ids` for MoE and fused gate/up/GLU/bias via `ggml_cuda_mm_fusion_args_device`), `ggml_cuda_should_use_mmvq` (batch ≤ `MMVQ_MAX_BATCH_SIZE` = 8), `MMVQ_PARAMETERS_RDNA3_0` and `get_mmvq_mmid_max_batch_rdna3`; `vecdotq.cuh`: `vec_dot_q4_0_q8_1`; `quantize.cu`: Q8_1 activation quantization; `mmq.cu`/`mmq.cuh`/`mmq-config-rdna3.cuh`: `ggml_cuda_should_use_mmq` (no MFMA on RDNA3, so the DP4A path) |
| Rosevine today | `k_q4_mm`: one wavefront per output row, F32 activations, scalar dequantize-and-FMA, `TT` tokens per block; no fusion, no indexed experts |
| Numerics | Upstream quantizes activations to Q8_1 and uses int8 dot products, so a direct port is a class-N change. It must pass teacher-forced KL, routing checks and layer agreement, and it is a different profile from the F32 reference. A port that keeps F32 activations would keep the memory-access and tiling design but lose the int8 dot. |
| Footprint | Kernel and dot product are about 400–700 lines after removing `ggml_tensor` host plumbing. `vecdotq.cuh` (1,443 lines) and `quantize.cu` (696) contain mostly unneeded types. Host launch logic is coupled to `ggml_tensor` strides. |
| Expected benefit | Unmeasured. C1 found MoE expert kernels to be HIP's largest deficit against Vulkan, so even upstream's HIP GEMV is not the fastest reference here. R3a.2's GPU expert and shared MLP spans are smaller than attention and CPU experts. Upper bound: those spans' critical-path share, to be measured in P3. |
| Maintenance | Moderate: RDNA parameter tables move upstream (`MMVQ_PARAMETERS_*`). Pin a revision and re-audit on update. |
| Disposition | **A/B.** First isolate `mul_mat_vec_q` for Q4_0 at exact shapes (2816×1408 gate/up, 704×2816 down, shared MLP and head) against `k_q4_mm` in a standalone HIP test, for both Q8_1 and F32 activation variants. Port only if it wins at matched numerics, or if a Q8 activation profile is accepted under D3. |

### 2. MoE routing, grouping and execution

| Item | Detail |
| --- | --- |
| Upstream | `ggml/src/ggml-cuda/topk-moe.cu`: `ggml_cuda_should_use_topk_moe` and the fused softmax, top-k and normalize kernel (power-of-two expert counts include 128); `mmvq.cu` with `ids` (one launch covers all selected experts for batch ≤ the RDNA3 `mmid` limit); `mmid.cu` (179 lines) for the batched gather/scatter; `ggml-backend.cpp` scheduler copies for host-resident experts |
| Rosevine today | Router logits are downloaded each layer (blocking). Top-k runs on the host. Four blocking uploads (`idx`, `pos`, `scale`, `weight`) follow. Then `gather_rows`, one `k_q4_mm` per selected expert for gate/up and again for down, and `moe_combine`. F6 counted about 130 blocking copies and about 750 launches per decode token. R3a.2 measured the router barrier at about 21–22 ms/token, which includes waiting for preceding GPU work. |
| Numerics | Device-side softmax/top-k/normalize is class T if the selection matches exactly. Ties and float order in softmax can flip near-equal selections, so selection flips must be counted. |
| Footprint | `topk-moe.cu` is 516 lines with many template instances. The technique for 128 experts and top 8 is about 100 lines. |
| Expected benefit | Removes the per-layer router readback and uploads on GPU-expert layers and makes the launch sequence static. The barrier span is an upper bound, because part of it is real GPU work being waited on. Host-expert layers still need their input downloaded. |
| Maintenance | Low for a Rosevine-owned routing kernel. Gemma 4's gate scale and normalization are model-specific anyway. |
| Disposition | **B.** Adapt device-side routing and a single indexed expert launch, keeping Rosevine's routing/combine invariants. Do not adopt `llama-moe-cache.cpp` (656 lines at the pin, 510 changed lines by the snapshot). Expert caching stays deferred under D1. |

### 3. CPU SIMD quantized matmul

| Item | Detail |
| --- | --- |
| Upstream | `ggml/src/ggml-cpu/arch/x86/quants.c`: `ggml_vec_dot_q4_0_q8_0` (AVX-512, AVX-512 VNNI with VL, AVX-VNNI and AVX2 branches); `ggml/src/ggml-cpu/repack.cpp` and `arch/x86/repack.cpp`: `ggml_gemv_q4_0_8x8_q8_0` / `ggml_gemm_q4_0_8x8_q8_0` (interleaved CPU_REPACK layout); `ggml-cpu.c` threadpool |
| Rosevine today | R3a.3: persistent workers with scalar, AVX2 and AVX-512 F32-activation Q4_0 kernels. The 1.5 ms per host layer objective was missed (2.10 ms at 12 workers). Scalar fallback handled 21.9% of rows. |
| Numerics | Upstream quantizes activations to Q8_0, which is the reference CPU backend's own approximation. Adopting it is exactly R3a.4's experimental class-N variant (D3). It is not a drop-in F32 replacement. |
| Footprint | The Q4_0×Q8_0 AVX-512 and VNNI dot is about 100–150 lines. The 8×8 repack adds a layout transform at load. Epic 2 recorded that CPU_REPACK load churn consumed 58 s of a 60 s load budget. |
| Expected benefit | The S4 prototype measured 0.75 ms for an int8 path whose VNNI intrinsic failed on the current toolchain. Upstream's VNNI branch shows the instruction path compiles under GCC with `__AVX512VNNI__ && __AVX512VL__`. Unmeasured in Rosevine. |
| Maintenance | Low for a dot-product port. Repack adds load-time and memory-layout ownership. |
| Disposition | **C for F32, A for R3a.4.** Keep the R3a.3 F32 path as the reference. Port `ggml_vec_dot_q4_0_q8_0`'s VNNI technique into the R3a.4 flagged variant. Repack is **B**, only after a measured load-time budget. |

### 4. Attention / FlashAttention

| Item | Detail |
| --- | --- |
| Upstream | `ggml/src/ggml-cuda/fattn.cu`: `ggml_cuda_get_best_fattn_kernel`. On RDNA3 (`amd_wmma_available`), decode (`Q->ne[1]` = 1) with a quantized K/V cache selects the vector kernel (`fattn-vec.cuh`) when head dim ≤ 256. The 512-dim global layers fail `can_use_vector_kernel`, so they fall back to the tile kernel (`fattn-tile.cuh`). WMMA MMA-F16 is used only for wide GQA tiles with head dim ≤ 256. |
| Rosevine today | `k_attention`: one block per (token, head). It writes every score to a global F32 scratch of `batch × heads × max_ctx`, then each thread runs a serial loop over all keys for its output dimension. Work is linear in context per output element, and scratch grows with capacity (F8). |
| Numerics | Online-softmax tiling changes summation order (class N, small). The Q8_0 K/V format is the same. |
| Footprint | `fattn-vec.cuh` (613 lines) plus `fattn-common.cuh` (1,327, which includes `vecdotq.cuh`). The tile kernel is 1,357 lines. C1 recorded heavy register spills in HIP FA kernels at head dims 256 and 512. |
| Expected benefit | R3a.2 sync attribution put about 16.4 ms/token in attention at 1,900 tokens. That is an upper bound, and it grows with depth. This is the most likely large gap for the depth cells. Unmeasured against upstream. |
| Maintenance | High for a direct port (template instance matrix, head-dim cases). Low to moderate for a Rosevine split-K/online-softmax kernel. |
| Disposition | **B, with D available.** Adapt the vector kernel's split-K online-softmax design for decode. Measure upstream vec and tile at D=256 and D=512 at exact shapes first. Spills at D=512 make a Rosevine-specific kernel defensible only if measured. |

### 5. Launch, graphs and synchronization

| Item | Detail |
| --- | --- |
| Upstream | `ggml/src/ggml-cuda/common.cuh`: `USE_CUDA_GRAPH` when `GGML_HIP_GRAPHS` (ON in the pinned HIP build); `ggml-cuda.cu`: `ggml_cuda_graph_evaluate_and_capture`, `ggml_cuda_graph_update_required`, `ggml_cuda_graph_set_enabled`. Graphs are disabled for NVIDIA before Volta, with no AMD-specific gate at the pin. Work runs on context streams. |
| Rosevine today | Null stream, synchronous `hipMemcpy` in `rvh_upload`/`rvh_download`, about 750 launches per token. C3 measured about 0.7 µs submit per launch. ADR 0001/D4: no public stream change; an internal queue handle is allowed. |
| Numerics | None (class S). |
| Footprint | Graph capture needs a static launch sequence. Rosevine's host routing (item 2) makes it data-dependent today. |
| Expected benefit | Submit cost is about 0.5 ms/token at 750 launches × 0.7 µs. Blocking copies cost more but are not attributed separately. Graphs only pay off after item 2. Graph support and stability on gfx1101 with ROCm 6.4 are unverified. |
| Disposition | **B/C.** Keep the executor and C ABI. Adopt an internal stream and async copies first (D4). Graph replay stays evidence-gated, after device routing. |

### 6. Allocation and buffer reuse

| Item | Detail |
| --- | --- |
| Upstream | `ggml/src/ggml-alloc.c` (`ggml_gallocr`, liveness-based tensor reuse); `ggml-cuda.cu`: `ggml_cuda_pool_leg` (the pinned HIP build sets `GGML_HIP_NO_VMM=ON`, so this legacy pool is used; `ggml_cuda_pool_vmm` is not) |
| Rosevine today | Allocation happens at session construction (no allocation during evaluation), with a memory ledger. The logits scratch is `batch × vocab`, and attention scores are `batch × heads × max_ctx` (F8). |
| Expected benefit | Not a decode-time cost: Rosevine already avoids per-evaluation allocation. The gain is capacity, not speed. F8 estimates batch-512 scratch would fall to about 0.2 GiB at every context with tiled attention and last-row logits. That matters for the depth cells' feasibility. |
| Disposition | **C** for the allocator. **B** for scratch sizing, which follows from item 4 and last-row logits. Do not couple to `ggml_gallocr`. |

### Control: Vulkan

`ggml/src/ggml-vulkan/ggml-vulkan.cpp` and `vulkan-shaders/` are kept as the
explanation for the fastest reference (C2: no spills, best MoE and single-token
kernels). Disposition **B** only: read shader tiling choices for items 1, 2 and 4.
There is no Vulkan runtime integration under ADR 0001.

## Newer-snapshot candidates (inspected patches)

| Commit | Change | Applicability to RX 7800 XT / ROCm 6.4 |
| --- | --- | --- |
| [69ee2ec](https://github.com/ggml-org/llama.cpp/commit/69ee2ecc0150ba7d1204a7f7393dfd651569db8c) | Linearized MoE layout for MMQ/MMF (`mmid.cu`, `mmq.cuh`, `mmf.cuh`; 230+/231−) | Shared CUDA/HIP code, batched (prefill) path only. Possible prefill relevance; measure in the newer-control series. |
| [0fb8f15](https://github.com/ggml-org/llama.cpp/commit/0fb8f157702702f33f4e018de8701b2569a1bc28) | Few-row MMQ for batch 2–24 (`mmq-few.cu`, 876 lines) | **Not applicable.** It is guarded `!defined(GGML_USE_HIP)` and aborts on HIP, and it needs NVIDIA Ampere+ int8 MMA. Its supported types exclude Q4_0. |
| [4324336](https://github.com/ggml-org/llama.cpp/commit/432433637b6d06ece7d134d9d779c307969fd75b) | Fixed hipBLAS workspace | **Not applicable on this toolchain.** Gated on `HIP_VERSION >= 70000000`; Fedora ROCm here is 6.4. |
| [e95426f](https://github.com/ggml-org/llama.cpp/commit/e95426f16982054c379ca7bdc8186996dd22f0fe), [cf34d31](https://github.com/ggml-org/llama.cpp/commit/cf34d311332e362198f0a410c466d1e940287d02) | MoE cache: run missed experts on CPU without waiting for cached experts | Scheduling technique only. The cache stays deferred under D1. The "do not wait" overlap idea is already R3a.2 proposal 3. |
| [da3416d](https://github.com/ggml-org/llama.cpp/commit/da3416d8077c55d68e45ce15babd999ac7ba1c49) | RDNA3/4 Vulkan large-tile split-K | Vulkan prefill matmul. It changes the newer-control Vulkan reference only, and is reported separately from the frozen baseline. |

## Integration options (no option adopted)

| Criterion | 1. Standalone executor + selected ports | 2. Selective GGML components behind the C ABI | 3. Limited upstream executor adapter (evaluation only) |
| --- | --- | --- | --- |
| Exact-shape speed | Equals the best port per kernel | Same kernels as upstream | Same as upstream HIP; still not Vulkan |
| End-to-end speed | Rosevine scheduling: routing, overlap and CPU workers | Mixed: GGML ops need `ggml_tensor` views and stream sharing | Upstream scheduling; Rosevine's CPU-expert pool cannot be used |
| Memory | Rosevine ledger, last-row logits | GGML pools inside a Rosevine-owned ledger: double accounting risk | Upstream allocator; C1 measured about 0.85 GiB more VRAM for HIP than Vulkan |
| Dependencies | ROCm only, as today | libggml-base and libggml-hip (MIT) as a build dependency | Full libllama and libggml; runtime replacement territory |
| Ownership and cancellation | Unchanged (Rust host, bounded cancellation) | Kernel calls stay synchronous to Rosevine. Cancellation is unchanged if GGML ops are leaf calls. | Upstream abort callback (CPU-only at the pin per `llama.h`); weaker |
| Numerics | Chosen per port (F32 or Q8 profiles) | Upstream profiles (Q8_1 GPU activations, Q8_0 CPU) | Upstream profiles only |
| Attribution | R3a.1 spans unchanged | Spans wrap GGML calls | Upstream perf counters only |
| Update burden | Re-audit pinned files per update | Track the GGML ABI and internal headers (`common.cuh` is not public API) | Track the whole upstream |
| ADR impact | Within ADR 0001 | Needs a reviewed proposal (major dependency) | Needs a reviewed proposal or ADR (runtime replacement) |

Option 1 remains the working architecture. Option 2 is plausible for one or two
kernels only if a standalone port proves impractical. Option 3 is useful as a
measurement control, which `parity_replay.cpp` already provides, and not as an
integration path.

## Recommended measurement order for P3/P4

These rankings are by evidence available today, not proven gaps:

1. **Attention at depth (item 4).** It has the largest attributed span, and it
   gates the depth cells' feasibility.
2. **Device routing plus indexed experts (item 2).** It removes the measured
   barrier pattern and enables item 5.
3. **Q4_0 GEMV (item 1).** Use exact-shape A/B against `mul_mat_vec_q` at both
   activation profiles.
4. **Q8_0 CPU expert dot (item 3)** under R3a.4.
5. **Streams and async copies (item 5).** Graph replay comes only after item 2.

Each step needs its own scoped bead with the optimization record fields from the
parity amendment: bottleneck, upstream approach, baseline, expected and actual
result, correctness and maintenance.
