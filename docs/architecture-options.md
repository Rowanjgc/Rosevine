# Architecture options

Status: proposed, 2026-11-11. No accepted ADR chooses a language, backend,
compiler, memory policy or model representation. Local evidence is in the
[baseline](hardware-and-inference-baseline.md); decisions require the
[roadmap gates](research-roadmap.md) and [protocol](experiment-protocol.md).

## Candidate execution paths

| Candidate | Verified starting point | Hypothesis worth testing | Cost / falsification |
| --- | --- | --- | --- |
| C++ / HIP | Fedora hipcc enumerates gfx1101; rocBLAS artifacts exist | Direct access to streams, events, exact-shape kernels and AMD ISA may reduce overhead | Compile/runtime/library failures, numerical errors or poor end-to-end performance can reject it; Fedora is outside the cited upstream OS matrix |
| C++ / Vulkan | Existing reference uses Vulkan libraries and RADV; shaders and device feature queries available | Preserve working driver path; specialized compute pipelines may suit Q4_0 GEMV and small verification batches | Descriptor, synchronization and pipeline complexity; enumerate features and benchmark actual shader variants |
| Triton | Upstream describes Linux/AMD support with ROCm 6.2+ | Faster kernel iteration and fusion may justify a limited kernel role | Generic AMD support does not certify gfx1101, this Python version or Fedora packages; compile, execute, inspect ISA and time exact shapes before inclusion |
| Rust orchestration with native kernels | rustc/cargo present | Ownership and experiment-state handling may justify host runtime use | Compare direct C++ and a minimal C ABI for identical native kernels; count FFI calls, copies, errors and maintenance burden |
| Python experiment orchestration | System Python present | Convenient reproducible workload, analysis and plotting tooling outside the token loop | Pin an isolated environment; avoid inherited project environments; time token-loop dispatch separately if ever proposed |
| Hybrid backend runtime | No prototype exists | Different prefill/decode winners might repay interoperability costs | Require measured savings greater than duplicate allocations, synchronization, transfers, build and debugging costs; default to one backend for the first correct runtime |

The [Triton compatibility statement](https://github.com/triton-lang/triton#compatibility)
is broad, so local execution is a prerequisite. The
[LLVM AMDGPU guide](https://llvm.org/docs/AMDGPUUsage.html) distinguishes processor
targets and supported instructions. Measure generated ISA, registers, occupancy,
spills, vectorization and memory transactions; language names do not establish
kernel speed. A compiler accepting gfx1101 is weaker evidence than a correct
executing kernel, which is weaker than a complete-model improvement.

[RADV documentation](https://docs.mesa3d.org/drivers/radv.html) describes its ACO
compiler path. GLSL/SPIR-V shape specialization and HIP/LLVM are different routes
to GPU instructions. Keep pipeline creation/JIT, shader cache hits, cold load,
warm execution and ISA inspection distinct. Vulkan
[subgroup guidance](https://docs.vulkan.org/guide/latest/subgroups.html) requires
feature queries; do not hardcode wave32 from the HSA result into all shaders.
Cooperative matrix enumeration alone does not demonstrate a profitable matrix
instruction path for small quantized products.

## Proposed runtime boundaries

Begin with an offline, single-request, target-only text executor. Reuse a pinned
GGML operation/backend layer if that is the fastest route to an independently
checked graph; compare against separately captured reference outputs. Describe
shared components honestly and include an independent dequantization/logit oracle
where shared code could conceal a common bug. A more independent executor is an
option after evidence, not a prerequisite to generating research value.

Keep these boundaries explicit without building a general plugin framework:

1. **Artifact reader and model contract:** read-only mmap, validated bounds and
   tensor types, tokenizer/template identity, exact Gemma graph and named shapes.
2. **Numerical reference and executor:** one-token and prefill operation semantics,
   attention positions/masks, shared/routed FFNs, routing and logits.
3. **Memory owner:** host/device placement, scratch lifetime, KV allocation,
   cancellation and resource accounting. A static placement is the first policy.
4. **Backend operations:** GEMV/GEMM, attention, normalization/activation and copy
   interfaces with defined shape/precision/stream preconditions.
5. **Decode policy:** target-only first; later an independently tested MTP state
   machine with verification, sampling correction and KV rollback.
6. **Experiment runner:** immutable configuration, metric timestamps and artifact
   references. Keep private inputs and captures separate from publishable reports.

These boundaries should permit a later projector to deliver embedding sequences,
positions and modality masks. Multimodal local attention may have bidirectional
segments; preserving an explicit mask contract prevents a text-only causal
assumption from becoming an irreversible API choice. Do not implement the image
path until text correctness passes.

## Prioritized specialization hypotheses

| Hypothesis | Required control | Evidence and decision |
| --- | --- | --- |
| Q4_0 unpack/GEMV dominates short decode | Exact descriptor-derived matrix families; same precision/placement, no MTP | Profile before optimizing; compare fused unpack-dot with pinned GGML and independent scalar dequantization |
| Verification/prefill benefits from another tile or GEMM path | Batch widths 1, 2, 4, 8, 32, 128 and reference microbatch widths up to 1024 | Measure useful output rows, compute and launch overhead separately; verify actual MTP batch width rather than assuming it equals 3 |
| Local/global attention deserves separate kernels | Local window 1024, global occupied length sweep; Q8_0 K/V | Test masking, RoPE, Q/K normalization, precision and SWA ring wrap before timing; preserve absent-V behavior |
| Fusion and graph reuse save launch/memory overhead | Norm/GELU/router/FFN intermediates and exact operation ordering | Measure launch count, bytes, scratch and end-to-end latency; floating-point reordering needs tolerance checks |
| CPU SIMD and noncontiguous layer placement improve heterogeneous execution | Same tensor bytes, occupied length, host thread budget and speculation policy | Compare first-N, noncontiguous and static per-expert placement; account for RAM bandwidth and synchronization |
| Expert cache locality repays transfer overhead | Existing cache disabled/enabled on a pinned backend, plus static placement | Trace real routing; simulate held-out traces; measure miss cost and memory displaced from KV/scratch |
| Repacking/mixed precision improves residency | Immutable original target and reversible derivative with transformation manifest | Byte-preserving layout changes and lossy re-encoding have different fidelity gates; a smaller file alone is no speed result |

The exact local artifact is Q4_0/F32. Q4_K kernels are out of the initial scope
unless artifact provenance establishes a different intended reference. A change
of reference requires an explicit decision and separately reproduced baselines.

## MoE and transfer economics

Start with the pinned llama.cpp implementation. Its cache already has segmented
LRU, shape/capacity eligibility and asynchronous miss upload; its router-ID read
also synchronizes the backend. Audit backend support and packed tensor handling,
then benchmark cache disabled/enabled at equal memory budgets. Newer upstream
features are a separate experiment with their own commit and baseline.

Collect per-layer selection counts, entropy, unique experts per verification
batch, reuse distance, burst length, domain shift and token-step autocorrelation.
Split training and held-out prompts before choosing a cache policy. Include cold
cache and prefill pollution; global expert popularity can hide layer-local misses.

For each layer, estimate `miss_bytes / measured_transfer_bandwidth + setup +
route_readback + synchronization`, and compare against measured CPU execution and
GPU hit cost. Sum only costs on the critical path. PCIe 4.0 ×16 theoretical
payload capacity is an upper bound, not measured bandwidth. Test pageable mmap,
bounded pinned staging and CPU/GPU overlap as separately named variants. Pinning
or duplicating weights may exhaust 32 GB RAM even when VRAM fits.

Simulate static placement, LRU/SLRU/LFU and an offline oracle at the same total
memory budget. An oracle is an upper bound, never deployable routing prediction.
Reject a cache if realistic transfers/synchronization consume its benefit, even
when hit rate is high. Preserve static placement as a fallback. Never alter
router decisions or skip selected experts to manufacture speed; such approximate
model changes require a separate quality track.

## Proposed decision procedure

Gate 2 compares pinned HIP and Vulkan microbenchmarks and full-model controls at
matched artifact, placement, sampler, cache, context and MTP settings. Include
target-only, prefill and MTP verification phases; document unsupported cells.
Triton and Rust remain candidates only if their small feasibility experiments
show concrete value. A backend may win one workload and lose another.

Propose an ADR containing the measured matrix, failures, supported operations,
debug/profiling limitations, build complexity and fallback. Select a primary
simple path after human review. If no backend can execute correctly, pivot to
toolchain/correctness investigation; if no specialization opportunity survives,
retain llama.cpp as the practical winner and report the research result.

Detailed counters are optional evidence. The
[rocprofv3 documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
does not prove all collection modes work on gfx1101. Use supported timestamps,
application spans and validated telemetry when counters are unavailable, and
report the resulting limits on causal attribution.
