# Proposed reference correctness capture and tolerance contract

Version: proposed v1, 2026-11-11. B2 remains in progress. This is a concrete capture
plan for review, not a full captured oracle or run authorization. A two-position
CPU smoke subset has now been captured twice as recorded below. It adopts no runtime,
backend or model-representation ADR. [Gate 0](gate-0-review.md) remains NO-GO for
performance experiments. The [experiment protocol](experiment-protocol.md) owns
the comparison definitions and proposed numerical limits.

## Reference identity and shared-code boundary

The candidate is the exact local Q4_0/F32 target and installed tools/libraries
identified in [Epic 1](epic-1-evidence.md). The source mapping below follows revision
`cd959bae99ff4bed22bef2604f45241c09bbb8c8`, `src/models/gemma4.cpp` and
`src/llama-graph.cpp`. A callback's presence in source is not evidence that the
installed binary exposes a usable capture interface. An isolated instrumented
build needs its own binary/library/source/build identities and equivalence check;
it must not silently replace the installed candidate reference.

Three layers of independence are required: the independently implemented scalar
Q4_0/math helpers, the pinned target-only graph, and that reference's complete
decoding policy. Callback captures share the model graph, kernels, tokenizer and
weights with the native reference. They detect candidate deviations but cannot
exclude defects common to both. The native scalar decoder cross-check shares the
format contract but not decoding implementation. Full-precision original weights
are unavailable as an assumed oracle; quality remains relative to the immutable
local quantized target.

## Operation coverage and proposed limits

All captured tensors must have finite values, declared dtype/shape/strides, token
positions, layer index, native node name and reference artifact identity. Shape,
position, expert IDs and masking errors fail regardless of numerical agreement.
Each numerical mismatch records max absolute/relative error, normalized RMSE,
cosine and affected rows/layers; zero norms use the absolute-error rule.

Limit A: elementwise `abs(error) <= 1e-5 + 1e-4*abs(reference)` for independent
small FP32 operation checks. Limit M: normalized RMSE <= 1e-3 and cosine >= 0.9999
for mixed/quantized intermediate outputs, with max absolute error <= 1e-5 for
zero-norm fixtures. Limit L: final-position normalized RMSE <= 2e-3, cosine >=
0.9999, mean full-vocabulary reference-to-candidate KL <= 1e-4 nats and top-1
agreement >= 99.9% over at least 4096 frozen positions. These are initial protocol
proposals, not calibrated or reviewed thresholds. Repeated reference variability
must be measured before freezing per-operation limits, without consulting a
favored candidate's errors. No threshold is loosened after candidate failure.

[`reference_variability.py`](../scripts/reference_variability.py) compares every
pair of at least two reference captures after enforcing the same run identity,
vocabulary, input/context identity, logit stage and ordered position labels. It reports
directional and symmetric KL plus worst-direction normalized RMSE, cosine and
absolute error. Rows remain streamed from private shards. The command also
requires each capture's retained supervisor bundle; finalization writes a
private binding over the index and terminal result. Analysis verifies successful
cleanup, distinct child process identities and a common executable hash. This
establishes separate supervised child executions, not independent sessions,
thermal states or run windows. Its result describes observed repeat variation
only; it does not approve these starting limits or calibrate a tolerance from an
undersized capture.

| Target operation | Native capture/reference | Independent check and failure condition |
| --- | --- | --- |
| GGUF reader, tensor layout and Q4_0 decoding | Exact artifact bytes and descriptor contract | Exact byte/range/shape/type and decoded FP32 bit equality; malformed/nonfinite scales fail. Native synthetic decoder cross-check is verified; deterministic real-target block sampling is now verified as described below. |
| Tokenization, special tokens and template | Embedded template, explicit thinking=true and native IDs/bytes | Exact frozen bytes/IDs/special-token mapping; no character-count occupancy estimate; server preprocessing parity remains unverified |
| Embedding lookup and sqrt-width scaling | `inp_scaled`, all selected input rows | `correctness_oracle.embedding_lookup` independently gathers explicit rows/IDs and applies the frozen scale; A for FP32 fixtures and M for graph output. The exact target descriptor has no per-layer embedding tensors |
| Input, Q/K/V and post-attention RMS norms | `attn_norm`, `Qcur_normed`, `Kcur_normed`, `Vcur_normed`, `attn_post_norm` | Independent small RMS fixtures with explicit epsilon/weight conventions, A; full graph M. Missing V projection uses pre-normalized K as its source, not post-RoPE K |
| Q/K/V projections | `Qcur`, `Kcur`, `Vcur`; fused `wqkv` only if actually active | Independent scalar dequant/matvec on small fixtures; M on full tensors. Reused KV layers must refer to their producer; do not fabricate missing K/V nodes |
| RoPE, proportional frequencies and positions | `Qcur_pos`, `Kcur_pos`, frozen position/frequency inputs | `correctness_oracle.rope_rotate` independently applies explicit disjoint pairs/angles, with unpaired dimensions preserved; small rotations use A and full tensors use M. Frequency derivation remains a separately bound target-policy input. Exact position, local/global mode and producer mapping; wrong offsets fail even if aggregate error is small |
| Attention masks, score scaling, softmax and KV reads/writes | Instrumented attention inputs and attention internals before output projection; node coverage must be established for active flash/unfused path | Small scalar attention reference, A; full output M; causal/local/global masks, cache indices and KV state exact. A fused opaque output alone cannot establish internal correctness |
| Attention output projection and residual | Attention output before/after `attn_post_norm`, then `attn_out` | Independent small projection and `correctness_oracle.vector_add` residual fixture, A; full graph M; compare inputs and residual independently |
| Shared FFN norms, gate/up, GELU gate product, down projection | `ffn_norm_1`, helper `ffn_gate`, `ffn_up`, `ffn_geglu`/`ffn_gelu`, `ffn_down`, then `ffn_mlp` | Small matrix/activation/gating checks, A; full outputs M. Capture actual GELU branch and operation order; do not substitute SiLU |
| Router RMS/scaling and projection | `ffn_moe_logits` plus added hooks for router RMS, inverse-sqrt scaling and learned input scale | `correctness_oracle.router_projection` independently checks the explicit RMSNorm → inverse-sqrt width scale → learned elementwise scale → projection chain, A; full logits M. Inputs and learned scale must be bound to the exact artifact; router input is `attn_out`, not the normalized expert FFN input |
| Router probabilities, selection and mixture weights | `ffn_moe_probs`, `ffn_moe_topk`, `ffn_moe_weights` and actual active normalized/scaled weight nodes | `correctness_oracle.router_selected_weights` independently computes full-vector softmax and selected-weight renormalization with the pinned F16-minimum denominator clamp, using explicit captured IDs; A for small fixtures. Numeric weights M; selected IDs exact away from near ties. Retain k/k+1 logit margin and all differences. A near tie requires investigation and downstream comparison; it is not an automatic exemption |
| Routed expert gate/up, GELU product, down projection and reduction | `ffn_norm_2`, `ffn_moe_gate_up`, actual gate/up/activation/down/reduction nodes, `ffn_moe` | Scalar selected-expert small fixtures and `correctness_oracle.gemma4_router_mixture` compose the router stages with explicit selected expert outputs, A; full expert/weighted outputs M; expert identities, expert-specific scales, reduction order and mixture weights recorded |
| Shared+routed addition, final FFN norm, residual and output scaling | `ffn_moe_combined`, `ffn_post_norm`, added pre-scale residual hook, `out_scaled`, `l_out` | `correctness_oracle.vector_add` plus norm/scale fixtures, A; full graph M; capture both branches before combination so cancellation cannot hide an error |
| Final RMS norm, tied output projection and softcap | `result_norm`, added pre-softcap logit hook, `result_output` | Independent small matvec and `correctness_oracle.softcap` fixtures, A; full logits L. Record consistently whether comparisons are before or after softcap |
| Greedy/sampled decoding and MTP state | Exact policy manifest, committed IDs, draft/verify/correction and KV/position snapshots | Greedy token IDs exact except reviewed numerical ties; at least 20 seeds for preregistered stochastic tests. Forced acceptance/rejection 0–3 must leave equivalent target-only state, including eos/cancel/resume |

### Real-target Q4_0 block sample

A read-only CPU cross-check sampled all 266 Q4_0 tensors in the local target.
For each ordinary tensor it decoded first/last and up to two deterministic
interior blocks; for each routed expert it applied that policy per expert. The
independent scalar decoder and pinned native CPU library matched bit for bit on
all 31,544 sampled blocks (1,009,408 FP32 values), including 60 routed-expert
tensors. The model artifact was identity-checked against its private frozen
descriptor before sampling; the private report retains hashes and per-tensor
coverage. No model was loaded and no GPU was initialized. This is broad sampled
block-layout/decoding evidence, not exhaustive weight coverage, quantizer
provenance, matmul/graph equivalence, or model correctness. The checker is
`scripts/native_q4_model_blocks.py`.

The target descriptor contract has no per-layer embedding tensors; the optional
per-layer embedding path in generic Gemma 4 source is outside this exact artifact.
Any newly advertised artifact/path needs its own coverage. The F32 CPU projector,
image preprocessing and image-conditioned logits belong to the later multimodal
contract; omitting them does not establish full-preset equivalence. The assistant
requires separate instrumentation/shape coverage before MTP correctness claims.

## Bounded intermediate and logit capture plan

1. Before any execution, validate the candidate-reference choice, sensor mapping,
   display/thermal/resource limits, private disk budget, cancellation and concrete
   invocation in an approved idle window. Begin with a short target-only smoke
   check; neither this document nor a passed unit test authorizes model execution.
2. In an isolated instrumented reference, enumerate actual active nodes and map
   them to the operation table. Preserve both the original and instrumented build
   identities. Reject incomplete callback coverage; request additional hooks
   rather than reporting unobserved intermediates as checked.
3. For intermediate references, use at least one tiny fixture from each of prose,
   code, arithmetic and structured instructions. Select a bounded set of token
   rows but capture every layer and every applicable operation, expert IDs and
   routing margins. Record which rows are captured and which are not. Include
   zero/extreme activation fixtures in independent operator tests.
4. Plan eight deterministic synthetic teacher-forced sequences, each with 513 exact
   token IDs and 512 scored next-token positions (4096 total). The current candidate
   uses overlapping calibration-token windows, not independent quality samples.
   Reset KV between
   sequences. Logits at input position j predict token j+1; omit no scored token,
   include explicit BOS/template handling, and freeze the position manifest before
   candidate testing. No generation is needed for this teacher-forced phase.
   These short sequences do not establish long-context behavior or quality.
5. Retain full-vocabulary FP32 logits in shards of at most 32 scored positions.
   With vocabulary 262144, one shard is 32 MiB and 4096 positions are exactly
   4 GiB. Two reference repetitions use 8 GiB before metadata/intermediates. Plan
   <=1 GiB for intermediate captures and <=1 GiB for metadata/margin, within a
   separately checked <=10 GiB pilot campaign. Do not start if the predicted
   intermediate footprint exceeds that bound; reduce selected intermediate rows
   before capture or seek an explicitly reviewed budget change. Never delete
   earlier evidence automatically to make room.
6. Record every shard's position range, shape, dtype, hash and finite-value check.
   Compute float64 errors/KL streaming by shard; do not allocate all vocabulary
   rows from all runs simultaneously. Repeat reference captures to calibrate
   numerical variability and compare router near-tie distributions. Additional
   repetitions may require another reviewed campaign/budget; do not silently
   exceed the pilot cap.
7. Separately exercise local-window positions 1023/1024/1025, global retrieval at
   beginning/middle/end, repeated keys, continuation, chunked prefill/decode,
   capacity boundaries and resume. Record explicit unsupported configurations.
   Long-context tests require their own run-window approval.

The calibration-only [context position planner](../scripts/context_positions.py)
prepares three 1,027-token direct-ID sequences with exact logits rows at input
positions 1023, 1024 and 1025. This is only a bounded boundary-probe input
manifest. The proposed 1,024-token window is confirmed against the active target:
the private target
descriptor specifies a 1,024-token sliding window and a 30-layer pattern with 25
local and 5 global layers; the clean pinned Gemma 4 loader maps that metadata to
`hparams.n_swa` and `hparams.is_swa_impl`, which feed the graph's SWA mask. The
audit is bound to the private descriptor/source hashes and this plan. The native
adapter accepts the selected-row format; its pinned-header API-double check and
compiled parser `--validate-only` accepted the exact private manifest. No logits
or model run has occurred. Protocol scope review: direct token IDs are suitable
for the planned teacher-forced final-logit numerical comparison because they
freeze model inputs and remove rendering/tokenization variation from that
comparison. This is limited to graph/logit comparison; it does not validate
rendering, tokenizer behavior, chat templates, independent quality samples, or
user-visible workloads. The overlapping/circular calibration windows are
correlated and are not quality evidence. Rendered prompt bytes are therefore
not applicable to this particular input protocol, but remain mandatory for
workload runs. This scope decision does not authorize capture: a separate run
window and resource review is still required. The boundary probe also does not
test semantic global retrieval, continuation, chunked prefill, capacity edges
or resume.

[reference_positions.py](../scripts/reference_positions.py) has now generated
and independently audited eight exact 513-token direct-input streams with 4096
explicit scored positions, from calibration data only. The audit checked every
input against calibration data and verified source spans, vocabulary bounds,
BOS handling and target alignment. The private plan records hashes, reset policy
and the 8 GiB two-repetition logit budget. The inputs use deterministic overlapping/circular windows,
synthetic sequence boundaries and reset KV; they are not natural-document NLL or
chat workloads. Protocol scope review accepts direct-token input only for
teacher-forced numerical logit comparison: fixed token IDs isolate graph outputs
from rendering/tokenization changes. Rendered prompt bytes are not applicable
to this capture protocol, and this plan does not satisfy the rendered-text
workload contract or establish quality. Its 4096 rows are correlated numerical
coverage, not 4096 independent quality observations. A separate approved run
window/resource envelope remains required. The complete 4096-position plan
and native intermediates remain uncaptured; a tiny two-position prefix now has
actual logit captures. The 16 performance manifests are separate
inputs and evidence. After adding the streamed NLL mode, the refreshed strict
native-linked helper accepted this exact 4096-position input through
`--validate-only`. A hash-bound private audit reconfirmed 4 GiB of FP32 logits per
repetition and 8 GiB for two; this parser-only check initialized no backend, loaded
no model and captured no rows.

## Prepared native logit adapter

[capture_logits.cpp](../research/capture_logits.cpp) is an isolated research
helper against the pinned C API. It is not an adopted production runtime or an
approved reference replacement. Its `--validate-only INPUT` mode reads a private
binary input without initializing backends or loading weights. The helper has
been compiled with C++17, strict warnings, fast math disabled and FP contraction
disabled against the existing pinned headers/libraries. The actual 4096-position
input passed native framing validation; seven malformed inputs were rejected.
Build command, source/binary/library hashes and validation records stay private.
Its `--run` path has since executed two tiny CPU captures; numerical equivalence
to the installed completion tool's full decoding policy remains unestablished.
Link/build success alone does not establish model execution correctness.

The execution path fixes CPU placement, two threads, batch/microbatch one,
causal attention, F16 K/V, flash attention off, mmap loading, lazy loading off
and MTP off. Before registry access it sets the pinned process-local Vulkan
disable switch, then rejects registered non-CPU devices. It does not call backend
discovery. Capacities must be multiples of 256, at most 4096, and the realized
context/batch dimensions must match the request. Current `RVLOG003` inputs bind
selected-row mode and `use_extra_bufts` in explicit flags. Preparation freezes
the full native policy; finalization checks the binary against it, and the v4
index records a policy digest. The setting must match the approved reference
invocation because controlled captures found it changes logits. Legacy
`RVLOG001` inputs are at most eight sequences of 513 IDs and retain every output
row. Legacy selected-row `RVLOG002` inputs can carry sequences up to the 4096-token context bound, decode
all causal positions through N-2, and retain only predeclared, sorted output
positions; at most 96 full-vocabulary rows are allowed. A full KV clear separates
sequences. No sampling, generation, chat rendering or context shifting occurs.
Each output row at input index j predicts frozen ID j+1. The final target is not
an additional decoded/scored position.
Pinned Gemma 4's public logits follow its final softcap; this helper captures
post-softcap logits only and provides no intermediate tensor hooks.

The pinned upstream [`llama-bench` source](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/tools/llama-bench/llama-bench.cpp)
exposes `--fit-target` and `--repack` as separate options; its repack default
comes from `llama_model_default_params().use_extra_bufts`. This establishes that
the fit-target setting is not itself a repack declaration in that tool. It does
not establish which value the installed completion executable passes, so `--fit
off` is not used as evidence for the capture policy. Runtime tool-policy
equivalence remains open.

The same isolated CPU helper now accepts a separate `RVNLL001` protocol for
corpus evaluation. It teacher-forces each exact document stream with KV reset,
normalizes the full post-softcap vocabulary row in long double, and emits one
FP64 target-NLL value for input position `j` and frozen target `j+1`. The protocol
is bounded to 1024 documents, 200,000 targets and a 2 MiB input; it stores no full
logit rows. [nll_capture.py](../scripts/nll_capture.py) binds split/corpus/model/
tokenizer identities and finalizes only exact, finite, complete outputs from
successful reaped supervision. Synthetic API-double reduction and target-order
checks passed. The updated helper was strictly rebuilt against the previously
verified pinned llama/ggml libraries; existing library hashes were rechecked and
11 resolved loader libraries were recorded under an explicit environment. Both
private corpus split inputs passed `--validate-only` through this native-linked
helper, which does not register a backend or load a model. This path has not
produced corpus NLL values; actual supervised run identity, run-time library
provenance, corpus results and reference variability remain outstanding.

[logit_capture.py](../scripts/logit_capture.py) prepares the canonical shared
input manifest and exact binary input, without launching anything. Its separate
finalization step requires a successful, reaped supervisor result and native
completion receipt. It verifies frozen input bytes, shard counts/sizes/modes,
hashes, finite rows and sequence/position alignment before creating a v4
policy-bound index. Partial/invalid captures remain on disk without an index. A native marker
alone cannot turn a timed-out or failed supervisor result into a valid capture.
Synthetic tests exercise a shard crossing a sequence boundary, the final short
shard, missing receipt, corrupt/nonfinite rows, unexpected files and changed input.

The input preparation record is not run approval. The next execution must begin
with a separately reviewed tiny capture smoke invocation under the external
supervisor and resource guard, then establish equivalence to the pinned tool.
The existing completion-tool pilot does not automatically authorize this helper.
Keep the shard directory under the same bounded private campaign as supervisor
logs; include its predicted bytes and retained failures in disk admission and
monitoring. The helper caps each shard at 32 rows and total raw output at 4 GiB,
but relies on external supervision for wall time, thermal/device health, memory
and campaign/disk reserves. Finalization is a separate monitored operation too.
Independently verify the supervisor command, frozen input and actual loaded build,
library/model/tokenizer identities; accepting a supplied digest is not that audit.
Actual helper cancellation during loading/decode and broad reference
repeatability remain unverified. The two tiny executions passed the runtime
CPU-only registration check and repeated the two selected rows exactly. Full-model capture
conditions and the native operator precision failures remain unresolved.

### First real capture smoke and exact-input replay

Two completion-tool diagnostics now use `--binary-file` and verbose input-ID
logging. Both evaluated all 15 frozen IDs exactly, including the final newline,
and produced identical generation bytes. The earlier `--file` preprocessing
deviation remains retained. The [native diagnostic reader](../scripts/native_completion.py)
keeps phase counters separate from output-delivery rates; no paired performance
or TTFT finding follows from these replays.

For the native helper, freeze the first three IDs of the calibration plan's first
sequence and score input positions 0 and 1, with reset KV and the original 4096
capacity. The last ID is a target only. Each of two independently loaded helper
processes completed under the bounded CPU diagnostic profile and wrote two finite
FP32 full-vocabulary rows (2 MiB per repetition). Finalization checked the exact
input bytes, successful supervision and cleanup, receipt, labels, shape, modes,
all finite values and complete shard hashes. Corresponding shards were bit
identical; streaming comparison reported zero KL/error over these two positions.
Only two unique positions are covered, not four or 4096. Both captures share
weights, graph and kernels, so they do not exclude shared implementation defects
or calibrate all-operation tolerances. They establish a working native row-capture
path and tiny repeatability check. Private records remain under
`.private/epic2-exact-replay-v3`; earlier monitor-timeout attempts are also retained.

A subsequent bounded CPU helper run captured all 15 positions of the exact
completion prompt, appending one dummy target solely to obtain the final row.
The 15 full-vocabulary rows passed shape, finiteness, checksum and cleanup checks.
The final-row greedy token did not match the first token obtained by retokenizing
the completion's printed output. Retokenization can change token boundaries and
the original sampled IDs were not captured, so this is an unresolved discrepancy,
not proof of a model error or equivalence. Direct sampled-ID capture and controlled
configuration comparison are needed before drawing a conclusion. Evidence remains
private in `.private/epic2-prompt-logits-v1`.

Follow-up used the pinned native vocabulary API to decode the helper's argmax
token directly, without retokenizing the completion output. Its six-byte token
piece did not occur in the 13-byte completion output; the helper's top-two
logit margin was approximately 0.393. This strengthens the discrepancy signal
but does not identify its cause: sampled IDs were not logged, and the helper and
completion use different batch and extra-buffer settings. The vocabulary-only
probe, source/build records and byte comparisons are retained privately. Match
execution settings and capture the sampled IDs before changing tolerances or
classifying this as a numerical defect.

The apparent sampler-history debug log is commented out in the pinned source;
the earlier description of it as an available sampled-ID path was incorrect.
The first bounded `--log-verbose` CPU attempt stopped during loading at the 90 C
CCD limit. A subsequent one-thread attempt completed, with the same output bytes
as the earlier two-thread run, but produced no sampled-ID messages. Both attempts
and their bounded logs remain private; no thermal limit was increased.

A vocabulary-only native probe enumerated token pieces that prefix the printed
output. Two token IDs are candidates; neither is the helper's argmax. Their
helper-logit ranks were 75185 and 7, with gaps of approximately 12.53 and 1.50
from the argmax. This is stronger diagnostic evidence but still depends on output
framing and is not a capture of the original sampled IDs. Different prompt-batch
and extra-buffer settings remain confounders. The native probe and analysis are
private in `.private/epic2-prompt-logits-v1`; direct sampling instrumentation and
matched configurations remain necessary before attributing a defect.

A follow-up captured both sampled IDs directly from the completion tool using a
process-local interposer on its exported sampler function. A synthetic test first
verified that the interposer forwards the original return value unchanged. The
one-thread CPU completion succeeded, its output bytes matched the earlier
completion, and the first sampled ID differed from the helper's final-row argmax.
This confirms a token-level difference for these executions. The helper used
batch-one incremental evaluation, two threads and different extra-buffer settings;
the completion used prompt batching and one thread. Those differences must be
isolated before attributing the result to a numerical or implementation defect.
The probe, identity, logs, checksums and comparison remain private in the
`.private/epic2-sampler-interpose-v1` and `-v2` directories. The initial probe
setup was rejected before launch by private-file mode checks and retained.

An isolated one-factor control then rebuilt the helper with only
`use_extra_bufts` changed from false to true; source, compiler flags, libraries,
prompt IDs, batch-one evaluation and two-thread settings otherwise matched the
previous helper capture. The supervised run completed and all 15 full-vocabulary
rows passed finite/shape/hash checks. Top-1 agreed at 14 of 15 positions, and
the final-position argmax now matched the completion's directly captured first
sampled ID. Across these 15 positions, mean reference-to-candidate KL was about
0.0927 nats, maximum normalized RMSE 0.331, minimum cosine 0.944, and maximum
absolute error 6.64. These exceed the proposed initial logit limits; no tolerance
was changed. Each setting has since been captured twice on this prompt: each
same-setting repeat was bit-identical across all 15 rows, with zero KL. The
cross-setting comparison against the extra-buffer repeat produced the same
0.0927-nat mean KL and 14/15 top-1 agreement. This supports repeatability within
this session, not a general cross-session variability envelope. Completion
still differs in batching and thread count. The experiment identifies a strong
extra-buffer-dependent difference on this fixture, not its underlying kernel or
full-model cause. Source difference, build identities, captures, telemetry and
independent audit remain private in `.private/epic2-extra-buffer-control-v1`.

Runtime logs identify `CPU_REPACK` as an allocated model buffer in the enabled
capture and show no such buffer in the disabled capture. The enabled run also
reports fallback to ordinary CPU buffers for some tensors, including the token
embedding, so not every tensor used the extra buffer. Source maps the CPU repack
buffer to specialized Q4_0 packing and matrix kernels, but the exact responsible
operation has not been identified; installed binary/source-build equivalence is
still a separate limitation.

Per-position analysis finds the discrepancy already at position 0: normalized
RMSE about 0.0565 and KL about 0.00297 nats. At position 14 they are about 0.331
and 0.240 nats. Position 0 has no earlier token history, although errors can still
accumulate across the model's layers. This narrows the investigation toward an
independent small-tensor comparison of ordinary and repacked CPU paths; it does
not by itself prove which path or operation is incorrect. The private
`position-analysis.json` retains all 15 position results and validated shard
hashes remain unchanged.

The first model-free operator probe used a synthetic 256-by-16 Q4_0 matrix,
one integer-valued activation vector and one CPU thread. The enabled case
allocated `CPU_REPACK` and initialized its repack tensor traits; the other used
the ordinary CPU buffer. Both produced identical output values on this fixture.
Against the independent scalar Q4_0 decoder/matvec, each had maximum absolute
error approximately 0.013855. This is a measured difference from the mathematical
reference; its source has not been established or excused. This one matrix-vector
case does not reproduce the full-model discrepancy and does not cover routed
`MUL_MAT_ID`, other shapes, activations, or graph interactions. The private
source/build/native outputs and scalar comparison are in
`.private/epic2-q4-repack-synthetic-v1`.

A separate model-free routed `MUL_MAT_ID` probe used a 256-by-16-by-4 Q4_0
expert tensor, one activation and distinct selected experts 0 and 3. Both CPU
paths produced identical 32-element outputs, with maximum absolute difference
from the independent scalar result approximately 0.138977. The selected experts'
scalar outputs differ, so the fixture can detect a wrong expert selection. An
earlier fixture accidentally repeated expert values and is retained as invalid
for expert-identity verification. This single routed shape/activation does not
reproduce the full-model buffer difference or establish the scalar discrepancy's
cause. Evidence is private in `.private/epic2-q4-routed-synthetic-v1` and `-v2`.

The pinned CPU Q4_0 dot-product trait selects Q8_0 activations. For the two
integer-vector fixtures, an independent reconstruction quantized each 32-value
activation block using its magnitude, rounded the scale to FP16 and reconstructed
the Q8_0 values before the scalar dot product. The maximum difference from either
native path was approximately 1.96e-8 for the ordinary matrix and 4.93e-8 for
the routed case. Thus their larger errors against unquantized mathematical input
are consistent with the activation quantization on these fixtures. This does not
explain the full-model extra-buffer mismatch, establish every native dispatch or
approve a precision exception. The diagnostic uses binary64 sums and the printed
native values, and remains private in `.private/epic2-q8-activation-analysis-v1`.

A subsequent target-shaped routed probe used the descriptor's Q4_0 down-expert
axes `[704, 2816]` with four deterministic synthetic experts and one input row.
Ordinary CPU and initialized `q4_0_8x8` CPU_REPACK produced bitwise-identical
values for selected expert IDs 0 and 3 over all 5,632 outputs. This rules out a
simple shape-triggered difference for that synthetic `MUL_MAT_ID` case only; it
does not reproduce the model's real weights, activations or upstream routing and
does not resolve the model-level extra-buffer finding. See
[native CPU findings](native-operator-findings.md#target-shaped-routed-q4_0-probe).

To prepare the next discriminating check, an isolated C API helper now observes
named MoE nodes through the pinned scheduler evaluation callback at prompt
position zero. It records router logits/probabilities, selected expert IDs,
mixture weights, and available routed gate/up/activation/down/reduction nodes.
It requires a one-sequence frozen input, CPU-only registration, batch/microbatch
one, two threads, F16 KV, flash attention/MTP off, and an explicit ordinary or
extra-buffer mode. Per-node and total capture bytes are bounded at 16 MiB and
100 MiB. The offline adapter binds preparation to exact input/model hashes,
requires successful supervised cleanup and finite shape-checked records, then
compares the matched node sets and reports the proposed M metrics plus exact
selected-ID agreement. Its routing summary also reports the ordinary/repack
logit gap between the kth and (k+1)th experts for every captured MoE layer; it
records the margin without treating any fixed margin as an automatic pass.
Each finalized index retains the full native policy. Comparison rejects changes
to threads, batch, microbatch, KV format, flash attention, MTP state or backend;
only the declared `extra_buffers` switch may differ. Finalization binds each
index to the exact successful supervisor result; comparison can verify both run
bundles, match their run-identity hashes and require distinct reaped child
processes with one executable hash. Without those bundles the tensor comparison
remains descriptive and reports run provenance as unverified.

An audit of the concrete paired command plan caught that its mode-specific
prepared directories did not yet contain the `inputs.bin` paths used by the
native invocation. Preparation now copies both the exact manifest and binary
into each private mode directory, and finalization revalidates their encoding,
model/tokenizer identities and hashes. The corrected v2 private plan verifies
both command input paths exist; no model command has been launched.

Finalization requires router logits, probabilities, IDs and weights plus the
routed gate/up, GELU, down, output, post-normalization and combined nodes to align
across every MoE layer observed. Missing stage/layer coverage remains incomplete.
The offline parser opens `nodes.bin` without following symlinks, validates its
private owner/mode/size, and compares slices from the same validated byte snapshot
instead of reopening the path during analysis. This prevents a replaced capture
path from changing the bytes between checksum and reduction.

The helper compiled with strict warnings against the pinned API and installed
libraries. A model-free GGML backend-scheduler smoke graph exercised the actual
`ask`/observe callback lifecycle and captured synthetic logits, expert IDs and
mixture weights through the same observer. It verified exact values, record
names/shapes, nonfinite rejection and position filtering. The test exposed that
the pinned CPU scheduler can provide synchronized host-backed nodes with no
backend buffer handle; the observer now copies `tensor->data` only in that
CPU-only path and rejects nodes with neither readable storage form. Offline
framing/finalizer/comparison tests pass. It has not loaded the model or executed
the callback against a real inference graph. Capturing named
nodes forces scheduler synchronization/materialization and can affect graph
partitioning; therefore this is a diagnostic instrument, not a performance run
or a pristine baseline. It observes position zero only and does not yet cover the
router's internal RMS/scaling nodes, other positions, quality or final-logit
effect. A real paired capture remains outside the
authorized first-pilot invocation and requires a reviewed run condition. No
tolerance or gate has changed.

The model-free check is reproducible with
[`check_moe_capture_adapter.py`](../scripts/check_moe_capture_adapter.py). It
requires an explicitly selected clean pinned source checkout, existing compiler
and installed llama/ggml libraries, plus new `.tools` binary paths and a new
private report directory. It compiles both the adapter and synthetic scheduler
harness, then checks exact captured values. It is optional and is not part of the
default test command; it installs nothing and does not load weights.

An optional [native API-double check](../scripts/check_capture_adapter.py) now
compiles the same producer against pinned headers and a
[test double](../tests/research/native_capture_double.cpp), linking no llama/ggml
libraries. It requires explicit source, new binary and private report paths; it
does not install a compiler or enter the default model-free check command.
The earlier 15-scenario version passed. The API-double plan now also checks the
extra-buffer policy and a 600-token selected context, which exercises the current
RVLOG003 selected-row flag above the legacy 513-token limit; these additions have
not yet been rerun against pinned headers because one unique matching source tree
was not available. The earlier successful case checked 33 full-vocabulary synthetic
rows, exact teacher-forced input/position order, two KV clears, a sequence boundary
inside the first 32-row shard, the final one-row shard, and context-before-model
cleanup. A selected-context case evaluated every sequence position but emitted
only its nine frozen boundary rows. Failure cases covered model/context creation, vocabulary/capacity
mismatches, non-CPU/missing devices, decode warning/abort/error, missing/nonfinite
logits and TERM-ignoring load/decode hangs. Failed producers retained their valid
first row when available and produced no completion marker/index. The external
supervisor killed and reaped both simulated hangs; the finalizer rejected every
failed result. These are deterministic API/control-flow checks: the double does
not open a weight file, implement kernels, initialize a backend or exercise the
deployed ABI. They establish no real-model numerical, resource or cancellation
finding and do not resolve the run conditions above.

The llama-independent framing decoder now lives in
[`capture_input_protocol.h`](../research/capture_input_protocol.h), and its
standalone C++ regression runs from the default research test suite when the
existing `g++` is available. It directly accepts the 600-token RVLOG003 sparse
input with positions 2/512/598 and the extra-buffer bit, and rejects unknown
flags, an out-of-range row, trailing bytes and a legacy 514-token full-row input.
It also checks the shared RVNLL001 framing. This verifies the framing logic used
by `capture_logits.cpp`; it does not compile or validate the llama API adapter,
deployed ABI, model run or cancellation path.
The optional pinned-header API-double build identity hashes this decoder header
alongside the adapter translation unit and test double.

[Native CPU findings](native-operator-findings.md) also require resolution before
freezing operation precision: output flags were necessary to materialize a fused
RMSNorm intermediate, and the installed GELU/GEGLU FP16 lookup path fails the
unchanged mathematical FP32 criterion on selected synthetic fixtures. No threshold
or precision exception has been approved from that result.

The independent CPU-specific GELU diagnostic now matches the installed table
bit-for-bit over all finite binary16 inputs. This gives a precise second reference
for explaining the pinned CPU lookup behavior, alongside the original mathematical
GELU reference and its retained failures. It does not calibrate a candidate
tolerance or establish full-model precision; those decisions remain open.

## Held-out quality and statistical review

The independent fidelity corpus must freeze at least 100k actual teacher-forced
prose/code tokens, licenses/provenance, document-level train/calibration/holdout
separation and tokenization hashes. The scored suite must freeze 100 tasks,
category balance, inputs, answer/scorer definitions and licenses. Performance
padding fixtures and operator calibration examples are not a substitute for
this held-out corpus. The [quality inputs](quality-corpus.md) provide 100 original
scored tasks plus 100128 native-token held-out positions from pinned
problem-prose/Python sources, with recorded license notices and exact split
hashes. A scoped protocol review accepts them for fixed, descriptive diagnostics
only. The task suite cannot establish the proposed 3-point noninferiority margin
and is template-clustered; the corpus does not support broad domain or
pretraining-exclusion claims. Model quality capture and any broader acceptance
suite remain outstanding. This capture plan does not automatically download or
approve another dataset.

Preregister NLL change <=0.01 nats/token with paired 95% upper bound <=0.02, and
accuracy drop <=1 percentage point with paired 95% upper bound <=3 points, as
proposed in the protocol. Resample documents/tasks rather than treating tokens
as independent observations. Report inability to establish noninferiority when
intervals are too wide; 100 tasks do not automatically have enough power.
Document stochastic tests, at least 20 seeds and context scoring before running
candidates. A proposed tolerance table and position budget alone do not close B2
or satisfy Gate 1.
