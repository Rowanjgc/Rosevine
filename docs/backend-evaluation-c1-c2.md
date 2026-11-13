# Backend evaluation: HIP/rocBLAS (C1) and Vulkan (C2)

Date: 2026-11-13
Beads: rv-e03-mmf.1 (C1), rv-e03-mmf.2 (C2); input to Gate 2 (rv-e03-mmf.4)
Status: Both evaluations complete. Both backends are viable. No backend is
chosen here; Gate 2 also needs C3 and a human-reviewed ADR.

## Setup

| Item | Value |
| --- | --- |
| GPU | RX 7800 XT, gfx1101, 60 CUs, wave32 native, 16 GiB |
| Source | Pinned llama.cpp v0.9.0 (cd959bae9), unmodified |
| Vulkan build | Gate 1 installed reference; RADV, Mesa 25.3.6, Vulkan 1.4.328 |
| HIP build | Separate out-of-tree build of the same source tree: `GGML_HIP=ON`, `GPU_TARGETS=gfx1101`, Release, shared, native CPU. Fedora ROCm 6.4 (HIP 6.4.2, rocBLAS 6.4.4, hipBLAS 6.4.1). No architecture override. |
| Packages added | `rocm-hip-devel`, `rocblas-devel`, `hipblas-devel` (+`hipblas-common-devel`), maintainer-approved 2026-11-12 |
| Power | COMPUTE profile for every timed run ([rv-19d](gate-1-review.md#power-profile-decision-rv-19d)) |
| Placement | Fast-preset model and MTP assistant at n-cpu-moe 4, flash attention on, q8_0 KV, 6 threads. Identical for both backends. |

The HIP build took 4 min wall time (22 min CPU) with 6 jobs. The only warnings
were 24 `loop not unrolled` notices. Every session used the
[gpu_campaign](../scripts/gpu_campaign.py) supervisor with the Gate 1 stop rules.
Each session required and verified the COMPUTE profile. No production setting,
Vulkan install or router state was changed.

## Capability report (C2)

**Vulkan device features (RADV, gfx1101):**
- fp16 and int8/int16 shader arithmetic;
- 8- and 16-bit storage;
- accelerated integer dot product (4×8-bit packed, signed and mixed);
- `VK_KHR_cooperative_matrix` (ggml uses `KHR_coopmat`; bf16 is not exposed);
- subgroup size control (32 and 64) with full subgroups;
- 64 KiB shared memory;
- `VK_KHR_pipeline_executable_properties`;
- calibrated timestamps with a 10 ns period and 64 valid bits.

RADV's default compute subgroup size is 64.

**HIP:** all 7,309 compiled kernels are wave32.

## Exact-shape correctness

The model's 105 unique graph operations, exported by `test-export-graph-ops` for
512-token batches and single-token decode, were compared with the CPU backend
using `test-backend-ops --test-file`.

- **Vulkan:** 105 of 105 pass ([divergence review](reference-divergence-review.md)).
- **HIP:** 103 of 105 pass. The other two are reproducible GPU memory-access
  faults, and in both the evidence points at the test harness rather than at HIP:
  - **`GET_ROWS` with a strided index view** faults intermittently. The cause is
    in the harness: the generic test-file path writes random row IDs
    contiguously from offset 0, so a strided view's other index slots are left
    uninitialized. HIP device memory is not zeroed, so the kernel reads out of
    range. The same case passes when run alone, and the model graph never
    produces garbage IDs.
  - **In-place f32 `SET_ROWS` into a one-row view** faults every time (both batch
    sizes). The harness rebuilds in-place outputs as a view of the wrong source.
    This is not established as a HIP kernel defect.

  Neither fault appeared in any HIP model run below: 8 benchmark sessions,
  8 server sessions, 3 KL runs and 1 capture, all of which ran the real graph.

## Exact-shape op timings

Each figure is the median of two `test-backend-ops perf` runs under COMPUTE.
Single-op timing includes launch and synchronization, which graph execution
partly hides.

| Op (shape) | Vulkan µs | HIP µs | HIP / Vulkan |
| --- | --- | --- | --- |
| Expert gate+up `MUL_MAT_ID`, 1 token | 17.3 | 42.6 | 2.46 |
| Expert down `MUL_MAT_ID`, 1 token | 23.8 | 64.9 | 2.73 |
| Q projection GEMV (4096) | 9.2 | 19.2 | 2.08 |
| Output head GEMV (262144) | 721 | 1270 | 1.76 |
| Expert gate+up `MUL_MAT_ID`, 512 tokens | 1003 | 1775 | 1.77 |
| Expert down `MUL_MAT_ID`, 512 tokens | 586 | 1117 | 1.91 |
| Router logits, 512 tokens | 40 | 188 | 4.70 |
| Dense projections, 512 tokens (range) | 135–791 | 123–657 | 0.67–0.91 |
| Flash attention, D 256 / D 512, 512 tokens | 1046 / 5811 | 828 / 6125 | 0.79 / 1.05 |
| Small elementwise/norm ops, 1 token | 1.3–1.6 | 2.7–3.3 | about 2 |

Taken one op at a time, Vulkan leads every single-token and MoE op, and HIP leads
the dense batched GEMMs.

## Registers, spills and compilation

**HIP (code-object metadata):**
- **Flash attention:** 109 of 1,620 kernels spill and 466 use scratch. The
  largest spills are the 128- and 256-dimension f16 kernels, at 613–695 VGPRs
  spilled. This model uses head dimensions 256 and 512.
- **MMQ:** 33 of 1,616 kernels spill.
- **GEMV and the routing kernels** (top-k MoE, argsort, softmax): no spills.
- **Compilation:** kernels are compiled ahead of time and ship in the library
  (58 MiB fatbin).

**Vulkan (RADV shader statistics):** the 17 matmul and flash-attention pipelines
this model uses compile with no VGPR/SGPR spills, no scratch, and at most 192
VGPRs. Pipelines compile at runtime and are cached by Mesa.

**Startup** (process wall time for an identical short benchmark, 3 runs each):

| Startup | Wall time |
| --- | --- |
| Vulkan, warm shader cache | 4.5 s |
| Vulkan, cache disabled (cold pipelines) | 5.5–7.5 s |
| HIP | 13.9–14.6 s |

Server load time to healthy was 5.5 s for Vulkan and 15.6 s for HIP (medians of
8 sessions). Vulkan's cold pipeline compilation costs about 1–3 s per process.
The HIP runtime and code-object loading cost about 10 s.

**Subgroup choice (C2).** Four ABBA pairs compared RADV with forced compute
wave32 (`RADV_PERFTEST=cswave32`) against the default. Paired ratios were
0.993–1.002 on every phase, so subgroup size has no measurable effect on this
workload. Wave32 is not assumed: the default wave64 was tested against it.

## Full-model phase matrix

There were 8 independent sessions per backend in ABBA order, each a separate
`llama-bench` process with one repetition per cell. Ratios are paired HIP/Vulkan
values within ABBA blocks. Rates are tokens/s.

| Phase | Vulkan median | HIP median | Paired HIP/Vulkan (range) |
| --- | --- | --- | --- |
| Prefill 512, empty context | 1558 | 1547 | 0.990 (0.987–1.078) |
| Prefill 2048, empty context | 1528 | 1517 | 0.990 (0.985–0.996) |
| Prefill 512 at 16k depth | 1100 | 1152 | **1.049** (1.047–1.060) |
| Prefill 2048 at 16k depth | 1092 | 1122 | **1.026** (1.022–1.030) |
| Target-only decode, empty context | 85.9 | 63.0 | **0.734** (0.698–0.738) |
| Target-only decode at 16k depth | 74.3 | 56.5 | **0.760** (0.720–0.764) |

**MTP verification.** Eight `llama-server` sessions per backend, ABBA, with MTP on
(draft up to 3) and 8 measured greedy 1900-token requests each:

| Metric | Vulkan | HIP |
| --- | --- | --- |
| Decode tokens/s (median, range) | 109.7 (108.9–111.5) | 105.4 (103.5–105.5), −4% |
| Prefill tokens/s | 2184 | 2334, +7% |
| MTP acceptance | 0.746–0.765 | 0.749 in every session |
| Minimum free VRAM | 1.21 GiB | 0.36 GiB (HIP uses about 0.85 GiB more) |
| Requests completed | 80/80 | 80/80 |

No request crossed a thermal flag. With MTP, the verification batch (up to 4
tokens) brings HIP's decode deficit down from 27% to 4%.

## Correctness

| Check | Vulkan | HIP |
| --- | --- | --- |
| Teacher-forced KL vs CPU reference (1792 positions) | 0.0558 / 93.4% top-1 | 0.0798 / 92.7% |
| KL vs Vulkan reference | 0 (repeat floor) | 0.0690 / 92.8% |
| KL repeat determinism | bit-exact | bit-exact |
| Layer-0 output vs CPU (relative) | 7.7e-3 | 7.4e-3 |
| Greedy output identity across 8 sessions | 3–8 distinct outputs per fixture | **1 per fixture (deterministic)** |

- **HIP is inside the band against Vulkan.** Its divergence from the Vulkan
  reference falls within the implementation band from the divergence review.
- **HIP is slightly above the band against CPU.** Divergence from the CPU
  reference (0.080) sits just above the band's previous top (0.073). Its layer-0
  agreement is within the ≤1e-2 tolerance, so this reads as the same routing
  amplification rather than a defect signal. The band's upper estimate should
  move to about 0.08.
- **Routing flips match.** HIP's visible routing-change rate against CPU
  (58/175 sampled token-layer positions) matches the other pairs.
- **Greedy decoding is reproducible on HIP.** HIP's greedy MTP decoding was
  bit-reproducible across 8 independent sessions. Vulkan's was not, as Gate 1
  also found. This makes output identity usable as a correctness control on HIP
  only.
- **Attention dump anomaly.** In the per-tensor capture, both GPU backends' layer-0
  attention dumps disagree with the CPU's token-0 value, but agree with each
  other to about 1e-3, and both match the CPU downstream. The anomaly comes from
  the GPU capture path and is not Vulkan-specific. It remains unexplained.

## Outcomes

**C1, HIP/rocBLAS: viable, correct, mixed performance.**
- **Builds and runs:** HIP builds from the pinned source on Fedora ROCm 6.4 for
  native gfx1101 and runs the whole model, including MTP, deterministically.
  The only two failing exact-shape cases trace to the test harness.
- **Faster:** prefill at depth (+3–5%) and MTP-path prefill (+7%).
- **Slower:** target-only decode (−24 to −27%), MTP decode (−4%) and the MoE
  expert kernels.
- **Costs:** about 0.85 GiB more VRAM and about 10 s slower startup.
- **Register pressure:** flash-attention kernels at this model's head dimensions
  spill heavily.

**C2, Vulkan: viable, fastest decode, weaker reproducibility.**
- **Speed:** best single-token and MoE kernels, and best target-only and MTP
  decode.
- **Kernels:** pipelines compile without spills; runtime pipeline compilation
  costs about 1–3 s cold.
- **Profiling:** timestamps and pipeline statistics are available.
- **Subgroups:** subgroup size has no measurable effect.
- **Reproducibility:** greedy output is not reproducible across sessions.

**For Gate 2.** Neither backend dominates:
- Vulkan leads decode, which dominates interactive use.
- HIP leads deep-context prefill and gives deterministic greedy decoding.
- The MoE expert kernels are the largest HIP deficit and the main specialization
  target for either path.

C3 (Triton, host language and FFI) remains open.

## Limits

- **Placement coverage:** one placement (n-cpu-moe 4) and one model. Prefill at
  this placement includes CPU-to-GPU expert transfers for 4 layers, which may
  compress backend differences.
- **Repetitions:** each `llama-bench` cell ran once per session (8 sessions per
  backend). There were 4 wave32 pairs.
- **Op timings:** these use random data and per-op synchronization.
- **rocBLAS:** not timed separately. llama.cpp's HIP backend uses its own MMQ and
  MMVQ kernels for quantized weights, and rocBLAS only for some float GEMMs. A
  standalone rocBLAS shape study is not covered.
- **No independent profilers:** no rocprof or RGA was available, so no hardware
  counters were collected for either backend.
- **KL coverage:** HIP KL used 1792 positions, and the layer-0 check used one
  214-token capture.
