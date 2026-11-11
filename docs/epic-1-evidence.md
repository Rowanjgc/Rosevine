# Epic 1 characterization evidence

Characterization complete with NO-GO, 2026-11-11. Tracks H1 (`rv-e01-898.1`), H2
(`rv-e01-898.2`) and Gate 0 (`rv-e01-898.3`). Gate 0 records a reviewed negative outcome; no inference
experiment has been completed. Read alongside the
[baseline](hardware-and-inference-baseline.md) and
[protocol](experiment-protocol.md).

## Independent artifact audit

`scripts/gguf_audit.py` implements a separate standard-library GGUF v3 reader.
It imports no llama.cpp or gguf-py code. It checks unique keys/tensor names,
rank, positive dimensions, row block divisibility, alignment, nonoverlapping
ranges and file bounds. Unknown tensor types fail; this narrow implementation
supports the reference F32 and Q4_0 layouts only. Metadata reading is bounded
at 128 MiB, 100000 descriptors and two million array entries. Optional full
SHA-256 reads stream in 8 MiB chunks outside measurement windows. File metadata
is checked for changes during inspection; this is not an adversarial filesystem
snapshot guarantee. Raw metadata, tokenizer strings and templates in the output
are private evidence and must never be copied into public logs.

The layout sources are the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
[reference enum](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/include/ggml.h)
and [block structures](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-common.h).
Synthetic tests cover Q4_0/F32 byte counts, exact hash, truncation,
misalignment, invalid block rows, unsupported types and overlap:

```sh
python3 -m unittest discover -s tests/research -v
```

All three local artifacts passed descriptor validation and full-file hashing.
These are local identities, not proof of upstream provenance or fidelity:

| Role | Tensors | SHA-256 |
| --- | --- | --- |
| Target | 658 | a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891 |
| Assistant | 49 | 7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a |
| Projector | 356 | ef269e294502d6ee3722cbf129681b2586c2e6ceb79d0507963c92146e058cd4 |

Private manifests are under the ignored mode-0700 `.private/epic1` directory,
created with umask 077. Four relevant preset sections were captured separately outside the checkout;
no credentials or unrelated router sections were retained. The private index
records their external location; router configuration is not kept in Rosevine. Originals were read
only. The active PATH server installation includes adjacent bench and tokenize
tools; their installed library resolution and reported revision were verified
as described below. Full source-build equivalence remains unproven.

## Environment recheck

Read-only CPU, package and DRM probes confirm the baseline CPU SIMD flags,
Fedora 43/kernel version, HIP 6.4.2 and clang 19, Mesa 25.3.6 and Vulkan loader
1.4.341. Discrete PCI device ID is `0x747e`, distinct from integrated `0x164e`;
the discrete device reports 17163091968 bytes VRAM and PCIe 16 GT/s width 16.
DRM card numbering is session-local and must not be hardcoded as API ordinals.
Correlate the PCI identity with Vulkan/HIP enumeration before any allocation.
GPU-busy, VRAM-used and amdgpu hwmon nodes are readable. Enumeration and file
access do not demonstrate calibrated counters, HIP kernel support or timing
accuracy. The profiler SDK package and rocprof/rocprofv3 executables are absent;
no installation was attempted. RAM timings and effective bandwidth remain unknown.

## Current acceptance status

H1 is closed with its inventory, support matrix, private evidence and explicit
measurement-access gaps recorded. H2 has completed the local identity, range,
shape, tool-selection, template/token and sampling-policy investigation described
below. Its negative findings bound equivalence claims; they do not certify the
candidate reference. The [Gate 0 decision](gate-0-review.md) links each remaining
unknown to its owner and impact. The agent evidence review records NO-GO; no human acceptance is claimed. No downstream experiment
or architecture decision is activated by this evidence.

## Matching tools and source-build evidence

The two running server executable paths match the PATH server. The server,
adjacent bench and adjacent tokenizer report version 0.9.0-dev, build 171,
commit prefix cd959bae9 and GNU 15.3.1. Their ELF resolutions select the same
installed llama/common/ggml CPU/Vulkan libraries. SHA-256 hashes of each tool
and all resolved libraries are retained privately in `build.json`. Use those
adjacent tools for reference work; the system bench remains excluded.

An installation-prefix source checkout is clean at the full reference commit
cd959bae99ff4bed22bef2604f45241c09bbb8c8. Its CMake cache and compile commands
were captured privately with hashes. The cache specifies Release, C++ release
flags `-O3 -DNDEBUG`, Vulkan enabled, HIP disabled and native CPU compilation.
These are retained build records, not yet proven production build flags:
installed server/bench/tokenizer/template-test file hashes differ from their
build-directory counterparts. Stripping, relinking or another build could
explain this; cause has not been established. Do not claim source-to-installed
reproducibility from version strings alone.

## Native template and tokenizer comparison

The installed native `test-chat-template` rendered five synthetic fixtures in
its direct Jinja mode (`--no-common`), with metadata-derived BOS/EOS and explicit
`add_generation_prompt=true`. The adjacent tokenizer processed each exact output
with `--ids --no-bos --offline`, parsing special tokens. Inputs, rendered bytes,
IDs, logs and the comparison driver are retained privately. This executes no
inference and sends no request to the production router.

| Fixture | Exact byte and token equality | Embedded / configured token count |
| --- | --- | --- |
| Plain user, thinking omitted | No | 19 / 22 |
| System plus user, thinking omitted | No | 27 / 25 |
| User, thinking explicitly true | Yes | 22 / 22 |
| Prior assistant reasoning plus follow-up, thinking omitted | No | 33 / 36 |
| Synthetic image marker, thinking omitted | No | 18 / 21 |

For the four omitted-setting fixtures, the configured template emits a thinking
system marker; the embedded template instead emits a closed thought-channel
prefix at generation start. This is an observed default-setting mismatch,
not a claim about effective router requests. Explicit true/false controls,
canonical upstream template comparison and common/server preprocessing still
need checking. The image fixture checks a rendered marker only, not image
preprocessing or multimodal embedding correctness. Historical canonical-template
equivalence remains unverified.

API correlation now independently confirms Vulkan device ID 0x747e as the
RX 7800 XT and HSA gfx1101 as the same named GPU; integrated Vulkan ID 0x164e
and HSA gfx1036 are distinct. Adjacent llama bench enumeration identifies the
discrete card as Vulkan0 for this session. Recheck at each session; ordinals
are not permanent identities. Raw enumeration is in the private evidence bundle.

## Explicit thinking controls and executable comparison

Ten additional native direct-Jinja/tokenizer comparisons cover each of the five
fixtures with `enable_thinking=true` and `false`. All five true cases match exact
bytes and IDs. All five false cases differ: the embedded template emits a closed
thought-channel generation prefix; the configured template omits it (four tokens
fewer in each fixture). Private `template-controls.json` records counts/equality
and the exact prompt/token pairs remain available. This does not test server
preprocessing or establish canonical upstream equivalence.

The build-directory and installed `.text` sections match for server, bench,
tokenizer and template test. ELF inspection shows different RUNPATHs: the build
includes its build directory while the installation uses only `$ORIGIN`.
Stripping a private copy with `--strip-unneeded` does not reproduce installed
file hashes. No original binary was modified. Executable-code agreement narrows
the gap; data sections, library build provenance and full reproducibility remain
unverified. The retained installed binary/library hashes define the experimental
reference independently of the source-build equivalence claim.

H1 now has a dedicated [environment/support report](environment-characterization.md),
private repeated-probe manifest and executable preliminary telemetry-read check.
Its proposed smoke budget and validation steps await review and implementation
in B1; no GPU workload or performance result is claimed.

## Complete descriptor shape contract

`scripts/gemma4_shapes.py` checks exact names, shapes and types for all 658 target,
49 assistant and 356 projector tensors. The private parsed manifests pass with
zero missing, extra, duplicate, shape or type mismatches. The target contract
follows the pinned Gemma loader's local/global head dimensions, optional global
V, shared FFN and fused expert tensors. The assistant contract checks four
shared-KV layers (Q/output without independent K/V), pre/post projections and
1024-wide hidden state. The projector contract checks all 27 repeated vision
blocks plus patch/position/input projections and standardization vectors. This
checks descriptor semantics; it does not execute any tensor operation or certify
image preprocessing. Failure tests reject missing/extra/duplicate descriptors,
wrong types/shapes and check local/global/shared-KV boundaries.

The semantic reference is the pinned
[target loader](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/models/gemma4.cpp),
[assistant loader](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/src/models/gemma4-assistant.cpp)
and [vision loader](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/tools/mtmd/clip.cpp).
The contract is deliberately specific to these local artifacts, not generic model
support. Q4_0 block storage and F32 types resolve the misleading filename as a
local descriptor fact; they do not establish the conversion's QAT fidelity.

## Canonical upstream comparison and sampling audit

The upstream Google template was retrieved from revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; its UTF-8 SHA-256 is
`ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`.
The [pinned template](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/chat_template.jinja)
has different bytes from both local templates. Ten direct-native-render/tokenizer
fixtures compare canonical to configured: all five thinking=true cases match
bytes and IDs; all five false cases differ by the four-token closed thought
prefix. Results, exact IDs/bytes and source provenance are retained in the private
`canonical-comparison` directory, separate from embedded comparisons. Original
embedded controls were repeated after canonical capture to retain distinct pairs.
This is finite fixture evidence; no universal equivalence claim follows.

The [Google model guidance](https://huggingface.co/google/gemma-4-26B-A4B-it#best-practices)
and [distributor guidance](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF#best-practices)
recommend temperature 1.0, top-p 0.95 and top-k 64. All four captured local presets
set those values, with min-p 0 and repetition penalty 1. They also specify
reasoning on. This supports explicit-thinking fixture relevance, but effective
request overrides, sampler order, seed and stopping policy are not yet captured.
The primary guidance removes historical reasoning from ordinary follow-up turns;
the synthetic history fixture checks that behavior. Image marker rendering does
not cover image loading or embeddings.

Pinned argument-source inspection confirms `n-cpu-moe=N` keeps expert weights
from the first N layers on CPU, not all weights from those layers. `load-mode=mmap`
selects explicit memory mapping; auto may fall back if a device does not support
it. Draft maximum/minimum and probability options populate separate speculative
parameters; probability minimum is documented for greedy drafting and does not
prove natural acceptance or sampled correctness. The runtime behavior remains
subject to later correctness experiments. See the
[pinned parser](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/common/arg.cpp).

Current H2 gaps: common/server request preprocessing, effective sampler policy,
upstream weight conversion lineage and installed data/library build equivalence.
Their owner is reference-identity follow-up. They block broad upstream/reference
equivalence and a Gate 0 GO claim. Exact installed binary/library and local-model
hashes permit a narrower experimental-reference proposal, subject to review.

## Common-library preprocessing and reference proposal

A private CPU-only C++ helper was compiled against exact-revision headers and
linked to the installed common/llama shared libraries. It calls
`common_chat_msgs_parse_oaicompat`, `common_chat_templates_init` and
`common_chat_templates_apply`, setting Jinja, generation prompt, thinking boolean
and DeepSeek reasoning format explicitly. Compilation command/result, source,
helper hash and per-case render/tokenizer outputs are private. No model context,
GPU allocation or server endpoint is used. The upstream template test's common
mode does not expose these thinking controls, so its defaults alone would not
prove the required comparisons.

Eight text cases (four fixtures, each thinking true/false) completed through that
common-library path for all three templates. All true cases match exact bytes
and IDs. All false cases preserve the configured-versus-embedded/canonical
four-token difference. Both synthetic image cases fail common rendering before
tokenization for all three templates; these failed results/logs are preserved.
The helper does not implement the server's media preprocessing. The direct-Jinja
image-marker comparisons remain valid at their stated boundary; common/server
image equivalence is unverified.

Pinned server source derives thinking from reasoning configuration and template
capabilities, then permits request `chat_template_kwargs.enable_thinking` and
`reasoning_effort` overrides (`none` disables thinking). Thus a preset saying
reasoning on does not establish every request's effective mode. The source's
default sampler sequence is penalties, DRY, top-n-sigma, top-k, typical-p, top-p,
min-p, XTC, temperature. Disabled samplers may be no-ops; order alone does not
establish their effective settings. The default seed is the random-seed sentinel.
Neither historical requests nor their seeds/overrides were available; no personal
prompt or raw historical log was collected to infer them. B1/B2 must freeze exact
sampler values, order, seed, stops and rendered bytes/IDs for new reference runs.

Reference proposal for Gate 0: retain the hashed installed tools/libraries and
three exact local GGUFs as a candidate experimental reference, with explicit
thinking=true for the supported text fixtures. Do not substitute a source rebuild,
system bench, different template, derivative weight file or disabled-thinking run
as equivalent. Do not call it upstream-weight-equivalent, fully reproducibly built,
or a reproduction of historical requests. Preserve originals and production
configuration. Common/server media parity, historical request policy, upstream
conversion lineage and full installed data/library build equivalence are bounded
unverified items; they prohibit the corresponding claims. A bounded diagnostic
follow-up can resolve them without beginning performance comparisons.
