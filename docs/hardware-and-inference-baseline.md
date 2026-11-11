# Hardware and inference baseline

Read-only planning observations, 2026-11-11. No model inference or stress test
was run. Raw selected metadata and diagnostic captures are private. This is a
manually authored technical summary, not a public raw benchmark export.

## Evidence classes and collection

**Observed** means read from local CPU/sysfs/package data, device enumeration,
ELF dependencies, binary help/version, or GGUF headers. **Reported** means the
user supplied it. **Inferred** means derived analytically or interpreted from
pinned upstream source. These classes do not establish runtime correctness.

Methods: `lscpu`, `uname -r`, Fedora release file, `free`, `df`, package queries,
readable DRM/sysfs attributes, `hipcc --version`, filtered `rocminfo`,
`vulkaninfo`, targeted configuration parsing, and bounded GGUF header parsing.
No unrelated configuration, process command lines or environment dumps were
collected. Unprivileged `dmidecode --type 17` failed; a noninteractive read-only
DMI query then succeeded, with only capacity/type/speed/channel fields retained
in this summary. No system configuration was changed. Header parsing checked
tensor offsets and payload sizes against F32 and Q4_0 block sizes for all three
GGUFs, with zero mismatches.
Independent-reader and full-file identity checks remain outstanding.

## Workstation

| Item | Evidence | Finding and limit |
| --- | --- | --- |
| CPU | Observed | AMD Ryzen 5 7600X, 6 cores / 12 threads, one NUMA node; AVX2, FMA, F16C, AVX-512 including VNNI and BF16 advertised; sustained SIMD throughput untested |
| CPU cache | Observed | 6 MiB aggregate L2; 32 MiB L3 |
| RAM | Observed | Two 16 GB single-rank DDR5 DIMMs, one on each DMI-reported channel A/B; configured and rated speed 4800 MT/s, configured voltage 1.1 V; OS exposes about 30.55 GiB usable; current timings unknown |
| Memory pressure | Observed snapshot | About 20 GiB available, about 9.4 GiB swap used out of 15 GiB; this is a shared-machine snapshot, not an idle benchmark condition |
| GPU | Observed | Radeon RX 7800 XT, gfx1101, 60 compute units, HSA wavefront size 32; sysfs exposes about 15.98 GiB VRAM |
| Other devices | Observed | Integrated Raphael GPU and llvmpipe also enumerate; tests must select the discrete GPU explicitly |
| PCIe | Observed | Discrete GPU current and maximum link: 16.0 GT/s, width 16 (PCIe 4.0 ×16); effective host-device bandwidth unmeasured |
| Kernel / display | Observed | Kernel 7.2.9-101.fc43.x86_64; Fedora 43; amdgpu; X11 session; bspwm reported |
| Vulkan | Observed | RADV NAVI32, Mesa 25.3.6; device API 1.4.328; loader package 1.4.341.0-1.fc43; libdrm 2.4.134-1.fc43 |
| Disk | Observed snapshot | About 496 GiB free on the checkout filesystem; no large artifacts proposed without a fresh budget |

DRM exposes board power, power cap and temperature readings on the discrete GPU.
Their cadence, accuracy and attribution are unvalidated. A sampled power value
does not establish whole-system energy. GPU activity was nonzero during inspection;
no current workload was displaced. PCIe negotiated bandwidth is not an observed
transfer rate. CPU RAM use must distinguish mmap file-backed pages, RSS/PSS,
anonymous allocations, system pressure and swap activity.

## Toolchain inventory

| Tool or package | Observed availability | Remaining validation |
| --- | --- | --- |
| HIP | rocm-hip 6.4.2-2.fc43; hipcc 19-14.rocm6.4.2.fc43; compiler reports HIP 6.4.43484-9999 / clang 19.0.0git | Compile, link, allocation, launch, error propagation and repeated numerical checks on explicit gfx1101 |
| HSA / BLAS | rocm-runtime 6.4.2-3.fc43; rocBLAS 6.4.4-1.fc43; gfx1101 HSACO/Tensile artifacts present; rocminfo enumerates GPU | Exact-shape rocBLAS correctness and performance; library/compiler/runtime compatibility |
| Vulkan shaders | glslc, glslangValidator, spirv-dis present | Pin versions and build flags; inspect generated ISA and pipeline-cache behavior |
| Host tools | GCC 15.3.1; Rust/Cargo 1.98.1; system Python 3.14.7; CMake 3.31.11; Ninja 1.13.0.git.kitware.jobserver-pipe-1 | Isolated reproducible builds; dependency compatibility, especially Python/Triton |
| Basic GPU monitoring | amd-smi, rocm-smi, radeontop present; readable DRM/sysfs | Validate per-device semantics, measurement overhead and sampling gaps |
| Detailed profiling | perf, rocprof, rocprofv3, rocprofiler-compute, rocgdb and gdb absent from PATH | Permission/support audit and approval for any install; installed libraries may exist separately |
| Triton | No Rosevine environment established or tested | Identify compatible pinned release/toolchain and execute a gfx1101 smoke test later |

The first discrete-device Vulkan block advertises FP16, int8, integer dot product,
cooperative-matrix extension, subgroup sizes 32–64, default subgroup size 64,
compute timestamps and a 10 ns timestamp period. Extension enumeration does not
prove useful matrix shapes, efficient instructions or a working kernel. HSA wave32
and Vulkan's reported default subgroup64 are different API observations.

AMD's versioned [ROCm 6.4.3 requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.3/reference/system-requirements.html)
list RX 7800 XT/gfx1101 with specific Ubuntu and RHEL restrictions; Fedora 43 is
not listed. These are adjacent-version upstream requirements, not certification
of the installed Fedora 6.4.2 packaging. Enumeration and installed code objects
justify a test, not a support or stability claim. The current
[Compute Profiler compatibility list](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/reference/compatible-accelerators.html)
does not establish RX 7800 XT counter support. Probe actual capabilities before
promising counters. Tool and web observations are dated and must be rechecked.

## Reference executable and semantics

The running server and PATH server report `0.9.0-dev`, build 171, GNU 15.3.1,
commit `cd959bae9`. Upstream resolves that short ID to
[`cd959bae99ff4bed22bef2604f45241c09bbb8c8`](https://github.com/ggml-org/llama.cpp/commit/cd959bae99ff4bed22bef2604f45241c09bbb8c8).
ELF dependencies include the Vulkan and CPU ggml libraries. The server is not
RPM-managed; source-tree cleanliness and original CMake flags were not found.
The executable hash is privately retained, but shared-library hashes and build
provenance must also be recorded before reproduction.

The system `llama-bench` belongs to `llama-cpp-b5904-2.fc43`, a different build.
Matching bench, perplexity, callback and backend-test binaries exist beside the
active server; validate their revision and library resolution before using them.
The system bench rejected `--version` and printed usage; that invocation was not
a performance test. Do not pair it with the new server as an equivalent reference.

Pinned [argument source](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/common/arg.cpp)
and local help agree that `--n-cpu-moe N` keeps expert weights in the **first N
layers** on CPU. It is not a count of individual experts. `--load-mode mmap`
explicitly maps the model. `--spec-draft-n-max 3` caps drafted tokens; it does
not guarantee three accepted tokens per step. Inspect effective startup settings
later because router defaults, environment overrides and request parameters can
still change execution.

The pinned build exposes `--moe-cache-mib`, default disabled. Its
[cache source](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/llama-moe-cache.cpp)
uses segmented LRU and a batch limit of 32, plus a slot-capacity condition. It
reads routing IDs with backend synchronization and uploads misses asynchronously.
Backend support and end-to-end behavior still need tests. The configured presets
contain no explicit cache option; do not assume an enabled cache. Current
[upstream cache documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md)
also describes CPU-share controls absent from the inspected pinned argument
source. Treat that as a separate, revision-pinned comparison, not a flag available
in the reference. The doc path itself did not exist at the pinned revision.

## Exact local GGUF descriptors

GGUF v3, tensor dimensions in GGML storage order (input extent first).

| Artifact | Header and size findings |
| --- | --- |
| Target | Filename UD-Q4_K_XL; 14,249,047,104 bytes; architecture gemma4; 658 tensors: 266 Q4_0 and 392 F32; general.file_type 2 |
| Assistant | Separate `mtp-gemma-4-26B-A4B-it-Q4_0.gguf`; 251,939,328 bytes; gemma4-assistant; 4 layers, width 1024, FFN 8192, 49 tensors (23 Q4_0, 26 F32), output width 2816 |
| CPU image projector | `mmproj-F32.gguf`; 2,291,200,480 bytes; clip/gemma4v; 356 F32 tensors; projection width 2816, 27 vision blocks, width 1152, nominal image 224 and patch 16 |

The [pinned GGML enum](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/include/ggml.h)
maps types 0 and 2 to F32 and Q4_0; Q4_K is 12. The
[block definitions](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-common.h)
specify 32 weights per 18-byte Q4_0 block and 32 values per 34-byte Q8_0 block.
The target's internal name calls it “smart Q4_0, QAT-lossless”; that is an artifact
claim, not a demonstrated fidelity result. The Unsloth
[model card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)
is a provenance lead; neither its filename nor its current contents prove this
local file's identity. No weights were downloaded or transformed.

Target metadata: 30 blocks, embedding 2816, vocabulary extent 262144, dense/shared
FFN width 2112, 128 routed experts, 8 selected, expert width 704. All 30 blocks
have routing tensors. Twenty-five layers use 1024-token sliding attention,
8 KV heads and 256-dimensional heads; layers 5, 11, 17, 23, 29 (zero-based) use
global attention, 2 KV heads and 512-dimensional heads. Query head count is 16.
Context metadata is 262144. RMS epsilon is approximately 1e-6; final logit cap 30;
RoPE bases are 10000 local and 1000000 global. Target shared-KV-layer count is 0.

| Operation / tensor | Representative storage shape |
| --- | --- |
| Token embedding (also reference output fallback) | [2816, 262144], Q4_0; no output.weight descriptor |
| Local Q / K / V | [2816, 4096] / [2816, 2048] / [2816, 2048], Q4_0 |
| Global Q / K | [2816, 8192] / [2816, 1024], Q4_0; no separate global V descriptor |
| Shared FFN gate/up and down | [2816, 2112] and [2112, 2816], Q4_0 |
| Fused expert gate/up | [2816, 1408, 128], Q4_0 |
| Expert down | [704, 2816, 128], Q4_0 |
| Router | [2816, 128], F32, plus F32 input and per-expert scales |

The pinned [Gemma graph](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/models/gemma4.cpp)
uses K projection output as V input when V weights are absent, then applies
normalization/RoPE at specific points. Shared and routed FFNs use GELU, with
separate normalization paths and routing scales. Do not substitute a generic
SwiGLU/MoE graph. The assistant metadata declares four shared KV layers and
context 131072; its [graph](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/models/gemma4-assistant.cpp)
and [speculation code](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/common/speculative.cpp)
must establish actual target/assistant KV binding, positions and rollback.
Do not infer 256k assistant correctness from target context metadata.

## Memory estimates, not allocation measurements

Descriptor-derived expert weights occupy about 11.964 GiB: 408.375 MiB per
layer, 3.1904 MiB per expert per layer, excluding tiny scale tensors. Actual
VRAM buffers may add packing, alignment and temporary copies. Static CPU
placement moves hundreds of MiB per layer; it is not equivalent to streaming
all active expert bytes every token.

An ideal Q8_0 cache with both K and V and compact sliding buffers has global
slope `5 × 2 × 2 × 512 × 34/32 = 10880` bytes per occupied token, plus about
106.25 MiB for 25 local windows of 1024 tokens. At 65536 / 131072 / 262144 tokens,
global payload alone is 0.664 / 1.328 / 2.656 GiB. This excludes allocator padding,
SWA policy, checkpoints, verification, masks, scratch and fragmentation. Compare
with actual allocations before treating it as a budget or feasibility result.
Equal K/V projection weights do not imply equal post-transform cache values.

## Historical performance anchors

All measurements below are **user-reported**, not reproduced here. Common stated
settings: same target, F32 CPU projector, Q8_0 K/V, mmap, canonical template and
recommended sampler. Canonical-template equivalence now needs verification.

| Preset | Capacity | First CPU MoE layers | MTP | Decode near 1.9k / 29k occupied tokens | Reported peak VRAM |
| --- | --- | --- | --- | --- | --- |
| gemma-4-26b-a4b-qat | 65536 | 2 | max 3 | 101 / 82 tokens/s | 15.14 GiB |
| gemma-4-26b-a4b-qat-fast | 65536 | 1 | max 3 | 131 / 92 tokens/s | 15.54 GiB |
| gemma-4-26b-a4b-qat-128k | 131072 | 3 | off | 80 / 69 tokens/s | 15.04 GiB |
| gemma-4-26b-a4b-qat-256k | 262144 | 8 | off | 47 / 40 tokens/s | 14.80 GiB |

Observed configured common values include six CPU threads, six batch threads,
one sequence, batch 2048, microbatch 1024, flash attention on, unified KV,
32 context checkpoints and all non-overridden layers on Vulkan0. The two 64k
presets request draft-mtp, max 3, minimum 0 and probability minimum 0, with the
assistant on GPU. These are interpreted settings, not a copied configuration.

The configured template has 18631 bytes and SHA-256
`d95156fde7b832bbe118b466eccc8add3eae78425e01ecf82ed0b10a1af31387`;
the embedded target/assistant template is 18922 characters with UTF-8 SHA-256
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`.
Different bytes do not by themselves establish different rendered prompts.
Full hashes, template token parity, effective request sampler, MTP acceptance,
warmup, variance, TTFT, prefill, RAM, energy and long occupied-context feasibility
are open work. Preset differences cannot establish causality.
