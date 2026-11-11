# Open research questions

Planning snapshot: 2026-11-11. No performance hypothesis below has been demonstrated.
Task IDs and installed blockers are maintained in the [roadmap](research-roadmap.md).
Evidence is distinguished in the [baseline](hardware-and-inference-baseline.md).

| Question | Why it matters / current evidence | Investigating Bead |
| --- | --- | --- |
| Is the named UD-Q4_K_XL file the intended reference, and how was it produced? | All inspected target tensor types are Q4_0/F32; file size/descriptor arithmetic agrees. The internal QAT-lossless label is unverified; full hashes and independent reader/provenance are outstanding. | H2 — rv-e01-898.2 |
| Do configured, embedded and canonical templates render identical token streams? | Configured and embedded hashes differ. Byte differences alone do not establish behavioral differences; reasoning/image markers may matter. | H2 — rv-e01-898.2 |
| What are current memory timings and effective bandwidth? | DMI reports two 16 GB single-rank DDR5 DIMMs on channels A/B at configured 4800 MT/s. Current timings remain unknown; do not equate configured rate to measured bandwidth. | H1 — rv-e01-898.1; bandwidth in M2 — rv-e06-k0i.2 |
| Can all reference binaries and loaded libraries be pinned to a reproducible build? | Active server reports cd959bae9; system bench is b5904. Original compiler flags and source cleanliness unknown. Adjacent tools need verification. | H2 — rv-e01-898.2 |
| Can the four historical presets be reproduced under explicit timing definitions? | Reported 131/47 tokens/s anchors lack raw variance, exact prompts and acceptance counters. Context, placement and MTP are confounded. | B3 — rv-e02-22u.3 |
| Which numerical tolerances distinguish implementation drift from quality change? | Proposed protocol limits require calibration on independent oracles and repeated reference runs, before candidate results. | B2 — rv-e02-22u.2; G1 — rv-e02-22u.4 |
| Does Fedora's installed HIP stack execute required gfx1101 operations correctly? | Enumeration, compiler and gfx1101 rocBLAS code objects exist. This is not full-model correctness or upstream Fedora certification. | C1 — rv-e03-mmf.1 |
| Which backend wins prefill, single-token decode, verification and MoE? | Existing Vulkan execution is reported; no controlled local HIP/Vulkan comparison performed. Different phases may favor different paths. | C1/C2 — rv-e03-mmf.1 / rv-e03-mmf.2; G2 — rv-e03-mmf.4 |
| Are subgroup choices, integer-dot or cooperative-matrix paths useful on these shapes? | Vulkan features are enumerated, but generated ISA, supported matrix types/shapes and performance are untested. HSA wave32 differs from Vulkan default subgroup64. | C2 — rv-e03-mmf.2; K2 — rv-e05-7n3.2 |
| Can Triton compile and run correctly on this exact GPU/toolchain? | Generic upstream AMD support does not settle gfx1101/Fedora/Python compatibility. No Rosevine environment was established. | C3 — rv-e03-mmf.3 |
| Is Rust orchestration or a hybrid backend worth the extra boundary? | Host tools exist; no FFI/copy/lifetime measurements or hybrid allocation costs. GPU instruction quality is a separate question. | C3 — rv-e03-mmf.3; G2 — rv-e03-mmf.4 |
| Which profiling modes and counters are usable without disrupting display workloads? | Basic sysfs/SMI tools exist; detailed tools absent from PATH; current upstream profiler lists do not establish this card's support. | H1 — rv-e01-898.1; K1 — rv-e05-7n3.1 |
| Is short decode limited by unpack, memory, CPU experts, synchronization or launches? | Exact shapes and quant blocks are known, but no attribution experiment has run. | K1 — rv-e05-7n3.1 |
| Can exact-shape quantized kernels or reversible packing beat pinned GGML? | Q4_0 payload dominates, but unpack/packing and residency benefits remain hypotheses. Must include prefill and MTP shapes. | K2 — rv-e05-7n3.2 |
| Do specialized attention, GELU/norm fusion, graphs or scratch reuse pay off? | Local/global heads differ; exact graph order matters. No launch/traffic savings measured. | K3 — rv-e05-7n3.3 |
| Does current llama.cpp expert caching already solve the useful problem? | Pinned source exposes SLRU cache with <=32-token and slot-capacity conditions. Newer upstream CPU-share documentation describes a different revision. Backend/MTP behavior needs actual tests. | M1 — rv-e06-k0i.1 |
| Are routing selections predictable enough for useful caches across domains and MTP batches? | No traces collected. Per-layer frequency, temporal locality, unique experts/batch, cold cache and prefill pollution all matter. | M1 — rv-e06-k0i.1; M3 — rv-e06-k0i.3 |
| Is CPU SIMD/static/noncontiguous placement better than transferring expert misses? | PCIe reports Gen4 ×16, but effective pageable/pinned bandwidth and CPU expert costs are unmeasured. RAM timings are unknown. | M2 — rv-e06-k0i.2 |
| Can uploads overlap useful GPU/CPU work without route-readback synchronization erasing savings? | Pinned cache code has synchronized routing-ID retrieval and async uploads. Async API calls do not prove actual overlap. | M3 — rv-e06-k0i.3; conditional M4 — rv-e06-k0i.4 |
| How does the separate assistant bind shared KV/features and reject drafts correctly? | Assistant has 4 layers, width 1024, four shared-KV layers and 131072 context metadata. Exact binding, sampling correction and rollback need source/runtime validation. | S1 — rv-e07-ktl.1; S2 — rv-e07-ktl.2 |
| What are natural MTP acceptance, verification work and net speed by context/domain? | Max draft3 is configured; no acceptance or phase timings collected. Acceptance alone is not end-to-end speed. | S1 — rv-e07-ktl.1; S3 — rv-e07-ktl.3 |
| Which tensors/experts tolerate precision changes, and do derivatives improve performance? | Original file must remain immutable. Re-encoding could change router behavior, duplicate memory or lose QAT properties. | Q1 — rv-e08-nhh.1 |
| Do compact KV formulas predict actual allocation at 64k, 128k and 256k? | Ideal global Q8_0 slope is 10880 bytes/token with ~106.25 MiB local windows. Allocator, SWA policy, checkpoint and scratch overhead remain unknown. | Q2 — rv-e08-nhh.2 |
| Are near-limit occupied contexts practical and correct with 16 GB VRAM/32 GB RAM? | Historical presets configure capacities but the reported measurements occupy only ~1.9k/~29k tokens. No near-capacity prefill/decode evidence here. | Q2 — rv-e08-nhh.2 |
| Can board/CPU/whole-system energy be measured reliably? | GPU board-power readings exist; cadence, accuracy, CPU energy access and wall-power instrumentation unverified. Do not invent missing counters. | H1 — rv-e01-898.1; B1 — rv-e02-22u.1; Q3 — rv-e08-nhh.3 |
| What survives sustained use and defines a true Pareto frontier? | Thermal drift, failure rates, variance and quality/resource tradeoffs are unmeasured. Slower high-fidelity configurations must remain visible. | Q3 — rv-e08-nhh.3; G5 — rv-e08-nhh.4 |
| Which projector preprocessing, image masks and embedding boundaries must text architecture preserve? | Projector metadata is F32, 27 vision blocks, projection width 2816. Nominal image size does not specify all crop/token behavior. | V1 — rv-e09-0s4.1; V2 — rv-e09-0s4.2 |
| What research packaging or integration is actually useful? | No production replacement is intended. Reproducibility can be valuable even if specialization loses. Redistribution, publication and router integration remain separately reviewed. | V3 — rv-e09-0s4.3 |

The next useful evidence is exact artifact/template/build identity, a safe
measurement envelope and a frozen workload/oracle suite. Every later question
has an owner and can resolve negatively without rewriting a hypothesis as a fact.
