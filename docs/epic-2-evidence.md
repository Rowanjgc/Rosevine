# Epic 2 implementation and evidence

Date: 2026-11-11. Status: in progress; no baseline experiment performed.
[Protocol](experiment-protocol.md) remains proposed. The
[Gate 0 NO-GO](gate-0-review.md) still applies to performance execution.
No runtime/backend ADR is accepted by this work. Python here extends existing
research tooling; it chooses no inference implementation language.

## Implemented offline infrastructure

The full-vocabulary target-NLL reduction now has an independent C++ long-double
cross-check against the Python binary64 implementation. On four already retained
rows (two from each of two captures, vocabulary 262144), the maximum arithmetic
difference stayed within the declared 2e-12-nat cross-check bound. The raw rows,
targets and per-row values remain private. This validates only the scalar reduction
path on those rows; it is not native corpus capture, a corpus NLL result, model
quality evidence or an acceptance threshold. No model was run for this check.

[benchmark_core.py](../scripts/benchmark_core.py) provides exact rendered-token
occupancy construction through an injected reference adapter, replay checks,
private workload identities, paired randomized schedules, delivery timing
calculations, descriptive statistics and paired percentile bootstrap analysis.
It distinguishes configured capacity from input occupancy and reserves 512
outputs plus 512 control tokens. An unreachable token length fails explicitly;
there is no character-count estimate or truncation of a rendered template.
The bounded padding search is not guaranteed to find every reachable length.
A new fixture version or padding strategy may be needed for the actual tokenizer.

[short-text-v1.json](../research/fixtures/short-text-v1.json) defines eight original
synthetic fixtures in the four protocol categories, with provenance and a GPL-3.0
license declaration. These are workload construction inputs, not a reviewed frozen target
manifest. The tests replay all eight at 1900 and 29000 synthetic tokens and two
capacities. Separately, [native_workloads.py](../scripts/native_workloads.py)
constructed all 16 actual Gemma manifests using the Epic 1 adjacent native tools,
the embedded template, explicit thinking=true, direct-Jinja text mode and ` x`
padding. The JSON fixture's longer suggested padding was not used for this run.
The builder completed successfully; each manifest reached exactly 1900 or 29000
tokens at capacity 65536, and each passed its built-in token/render replay.
A separate verification checked all 16 file/manifest/render/token hashes, token
counts, capacity reserves and private file modes, then independently rerendered
and retokenized prose-summary-1900 with the current adapter: bytes and IDs match.
These are CPU-only construction/replay results, not inference or server parity.
Greedy sampling in these manifests is a proposed control, not the historical
sampler policy or a reviewed final choice. Actual repeated runs at other configured
capacities remain outstanding. The new [capacity control builder](../scripts/capacity_controls.py)
created a second private manifest set from the 16 frozen rendered workloads,
rebinding 1900-token prompts to capacity 4096 and 29000-token prompts to capacity
32768. All controls retain at least 512 output and 512 reserve tokens. An
independent persisted-artifact audit verified all prompt/render/token bytes and
hashes against the capacity-65536 originals; only `capacity_tokens` and the
manifest checksum changed. This is exact-input manifest evidence only: no
retokenization, model context allocation, inference, fit or performance behavior
was measured. Private evidence is in `.private/epic2-capacity-controls-v1`.

The [private-format reference](private-benchmark-format.md) documents capture,
identity, telemetry, paired-trial, quality-task and shard contracts and their
limits. No public export schema is extended.

The native builder takes explicit model/renderer/tokenizer paths, audits and
hashes the complete model before construction, extracts its embedded template,
and hashes tokenizer metadata and both tools. It creates a private output bundle
with incomplete/successful/failed outcome and per-workload identities. Its native
adapter retains failed scratch inputs/logs; successful scratch is removed after
rendered bytes and IDs are returned. The CLI sets umask 077. Loaded-library and software/build identities still need to be populated for an actual run; the new private identity schema inventories those fields.
Private native workload evidence is in `.private/epic2-workloads-v1`; decoder
cross-check evidence is in `.private/epic2-correctness`. Neither is published.

The schedule labels two warmups and seven measured trials for every configuration
and workload, sharing a seed within a pair and randomizing configuration order.
Cold-start trials and larger sweeps are separate campaigns. The analysis helper
uses median paired relative percent changes with 10000 seeded bootstrap resamples
by default. Callers must match prompt/seed identities and retain excluded runs
with reasons; the helper cannot establish pairing from anonymous arrays. No GO
is inferred from statistics. Fewer than 30 samples receive a p95 caveat.

[paired_analysis.py](../scripts/paired_analysis.py) now validates actual structured
trial records against a frozen schedule, configuration identities, prompt/token
hashes, seeds and units. Exactly one named capacity/first-N/MTP factor may differ;
confounded preset rows cannot become a causal comparison. It retains missing,
failed and invalid runs, excludes whole preregistered pairs, checks two successful
warmups per configuration and at least seven eligible measured pairs, and blocks
intervals on unregistered failures/missing metrics. Zero reference metrics cannot
produce relative improvement. Separate private CLI validation recovered a known
synthetic 10% paired change and retained a synthetic allocation failure with no
comparison interval. This is serialization/analysis evidence, not measured
performance.

The local child supervisor captures private stdout/stderr, command identity,
status, monotonic elapsed time and checksums in a new 0700 directory with 0600
files. It retains nonzero exits, timeout and cancellation results, refuses
replacement of an existing bundle and terminates the child process group.
Pipes are drained with bounded capture (8 MiB by default); telemetry has a separate
1 MiB default cap. Reaching a cap with additional data cancels as incomplete,
rather than reporting success with silently truncated evidence. Explicit disk
reserves are checked before launch and during supervision. These caps cover
captured pipes/telemetry, not arbitrary files the child writes. Optional campaign accounting now includes all existing private captures and
projects capture growth before launch, with periodic budget checks. It assumes
one sequential writer and is not a filesystem quota; arbitrary child writes or
final failure metadata can overshoot between checks. Raw tensor capture budgets
still need explicit planning.

An optional telemetry callback runs before launch and periodically (100 ms by
default); a preflight rejection launches nothing. Samples include monotonic times,
collection cost, readings and rejection reason. Missing metric adapters remain
null with reasons. Exceptions retain failed results. SIGTERM is followed by
SIGKILL after the grace period; a synthetic child ignoring SIGTERM verifies this
path. Children must not daemonize out of the supervised process group. Callbacks
must be bounded; a hung callback cannot be interrupted by this synchronous API.
The child receives a private working directory and umask 077. Child elapsed time
includes supervision/cleanup and is not decode, TTFT or a GPU timestamp.

[collector_worker.py](../scripts/collector_worker.py) now isolates potentially
blocking acquisition in an explicit owned process with bounded JSON-pipe requests,
responses and diagnostics. The response deadline is capped by remaining workload
time; preflight stalls launch nothing and running stalls cancel the owned workload.
Fresh request IDs and monotonic acquisition/receipt intervals are checked and
retained. Duplicate/malformed data, stale IDs, oversized output, collector exit
and callback exceptions reject safely. Workload cleanup precedes collector cleanup;
both owned-leader waits are bounded and pending cleanup is reported as incomplete,
without claiming group/device quiescence. The generic synchronous callback API
still requires a bounded caller. No sandbox, sensor accuracy or model-run approval
is implied.

Twelve process-worker tests cover real local pipes, hangs, ignored termination,
output caps, malformed/stale responses, startup/exit failures, successful
transport, workload deadlines and a known owned descendant. A blocked leader reap
is simulated rather than creating an unkillable task. A separate private synthetic
ResourceGuard integration in `.private/epic2-process-collector-v1` retained three
fresh samples, verified all capture checksums and confirmed both owned leaders
reaped. No actual safety readings or model execution were used in that integration.
A second synthetic campaign in `.private/epic2-process-collector-v2` additionally
archives the selected framework/worker source, verifies that current source stayed
unchanged during capture, and checks the response time bindings and capture-file
hashes. This is selected-source provenance for a synthetic test, not complete
model/build identity or permission to run another trial.

A separate private v1 synthetic integration campaign in
`.private/epic2-supervision-v1` exercised ResourceGuard plus the supervisor: a
synthetic swap increase cancelled the sleeping child, retaining preflight and
both running samples, source hashes, run identity and verification. No workstation
readings or model execution were used. This is not a calibrated GPU runner.
There is no operational endpoint or production adapter.

[benchmark_resources.py](../scripts/benchmark_resources.py) parses explicit Linux
memory/swap snapshots with byte units and evaluates externally supplied resource
limits. Missing/invalid applicable readings, known unhealthy devices,
reserve breaches and time/temperature/capture limits reject a trial. The current
v2 paging policy records small events/reset warnings and rejects sustained
significant paging under low headroom, rather than any global counter change. Importing it
collects nothing. It does not establish telemetry accuracy, device selection,
sensor-specific thermal limits, overhead or sufficient allocation headroom.
The snapshot adapter now handles selected process status/PSS/fault/core-time
fields and explicit GPU samples with byte, fraction, Celsius, Hz and watt units.
It discards process identity strings and requires supplied page size/tick rate
for the relevant counters. Missing/invalid sources stay null with reasons; a
power cap cannot be used as a power sample. Trapezoidal board-energy integration
reports actual time coverage and maximum sample gap, without extrapolation or
whole-system claims. ResourceGuard connects explicit safety snapshots to the
supervisor and checks swap deltas between successive samples. Absent required
health/thermal signals reject preflight; CPU-only profiles may explicitly report
unknown GPU health without claiming it healthy. Readable files alone are not a health signal.
No public command automatically collects workstation data.

A separate 100-sample live read-only collector probe completed but failed the
proposed 1 ms p95 host-cycle cost bound. Grouped phase profiling identified GPU
attribute reads as dominant, without isolating an individual sensor or proving
model-workload impact. Results remain private in `.private/epic2-collector-cost-v1`.
No bound was relaxed. A per-attribute 100-sample probe compared reopening files
with retained-descriptor pread; individual reads stayed below the proposed total
cycle bound, which does not prove combined-cycle compliance. A revised 100-sample
full collector reused descriptors and parsed swap once, but its p95 cycle still
missed the bound. All failed and revised results remain private; the overhead
issue is unresolved. A further 100-sample read-only probe through the isolated
process collector completed with confirmed owned-leader cleanup but also missed
the unchanged 1 ms p95 host-cycle bound. Its snapshots explicitly reject workload
authorization; it tests transport/host cost only. Results remain private in
`.private/epic2-process-collector-cost-v1`. Isolation bounds failure handling and
does not demonstrate low instrumentation overhead. Actual workload-time
perturbation, counter accuracy and GPU
health remain unvalidated. A read-only preflight found sufficient proposed initial
RAM/display headroom but nonzero GPU activity; it established no idle window.
The [pilot packet](epic-2-pilot-review.md) records the concrete CPU-only command
and these unmet conditions. The maintainer requested continued work within safe
spec; no model run or positive gate was inferred from that instruction.

A preregistered randomized read-only comparison now also tested overlapping the
GPU and proc/SMAPS acquisition groups. Both methods retained the same signals,
100 measured samples and two warmups each, at 100 ms cadence, with per-source
acquisition intervals. Overlap had a lower observed p95, but both methods still
missed the unchanged 1 ms host-cost bound. An independent audit reconstructed all
204 request timechains, schedule assignments, sample counts and percentiles and
confirmed collector cleanup. Raw data and source/configuration snapshots remain
private in `.private/epic2-collector-overlap-v1`; the unsuccessful optimization
was not promoted into a public adapter. This narrows the available acquisition
options without resolving the run-approval, accuracy or workload-overhead gaps.

The maintainer later approved one immediate CPU-only diagnostic with conservative
sensor stops, the candidate reference/resource envelope/limited monitoring scope,
and a diagnostic-only 3 ms p95 collector allowance. Fresh identity/resource/thermal
and no-paging preflight passed, as did the fresh 100-sample collector check and
synthetic threshold checks of the actual private monitor. The exact completion
command launched once, then stopped when system-wide swap-in increased by one
4096-byte page. The child exited on SIGTERM; workload/collector cleanup and raw
checksums were independently verified. No native output/timing was captured,
so model loading, token evaluation and execution placement remain unestablished.
The paging source is unknown; a system-wide counter cannot attribute it to the
child. The incomplete attempt and source snapshots are retained privately in
`.private/epic2-authorized-pilot-v1`. There was no automatic retry. See the
[pilot outcome](epic-2-pilot-review.md) for the approval scope and next conditions.
The original benchmark collector bound, performance experiments, actual tensor/
logit/quality results and Gate 1 remain outstanding.

The maintainer then requested a project-wide sensitivity audit and another run.
The [rule audit](development/rule-audit.md) separates physical stops, measurement
acceptance and publication controls. Protocol v2 replaces any-global-paging abort
with sustained significant paging plus low-headroom detection, makes optional
health requirements scope-specific, distinguishes collector startup from running
deadlines and carries authorization across bounded follow-ups. Secret/privacy
scans, hard reserves, finite-value checks, evidence integrity and scientific
correctness limits remain enforced. New regressions cover isolated paging/bursts,
true sustained pressure, reset/gap recovery, immediate hard reserves, explicit
unknown-health scope and distinct startup/running deadlines.

The first follow-up recorded paging without a false stop but reached 81.125 C
against the original 80 C CPU cutoff. That record is retained. After verifying
the CPU's 95 C manufacturer maximum, the procedural cutoff was corrected to
90 C with a 5 C margin. Cooling/preflight preceded one further bounded attempt.
It exited successfully, produced native output and native prompt/decode timing,
and confirmed cleanup. All 135 retained samples passed the safety profile;
paging warnings did not establish low-memory pressure. The audit also resolved
a prompt preprocessing discrepancy: `--file` removed a trailing LF, so 14 actual
input IDs were evaluated rather than the planned 15. Independent vocabulary-only
replay verified the effective bytes/IDs, and the original frozen record remains
unchanged. Future exact-byte completion invocations must use a verbatim mode or
record the processing explicitly. See the [pilot outcome](epic-2-pilot-review.md).
This is successful bounded loading/evaluation/cleanup evidence, not a paired
benchmark, correctness oracle, reference repeatability or positive gate.

A private audit of the successful run found its recorded full telemetry-cycle
p95 within the approved 3 ms diagnostic allowance but above the original 1 ms
benchmark-quality target. One timestamp gap exceeded one second even though no
individual collector response exceeded its deadline; the cause is unknown.
This records cadence/scheduling uncertainty, not workload perturbation or a
collector timeout. The earlier result retained the p95 and count but not the
individual full-cycle durations, so that p95 cannot be independently
reconstructed from its telemetry. Future supervisor results now retain each
duration, bounded by the telemetry capture budget.

[benchmark_manifest.py](../scripts/benchmark_manifest.py) defines a private
run-identity schema covering all protocol identity groups: experiment, software,
artifacts, request, execution, MTP, environment and measurement. It inventories
unknowns as null with reasons, validates core hashes/types/artifact sizes,
checksums the identity and reports missing comparison prerequisites. This
structural inventory is not experimental approval; a known value or arbitrary
review-reference string does not prove review occurred. An actual run still
needs complete, verified identities and run-window/resource approval.

## Correctness infrastructure

[correctness_oracle.py](../scripts/correctness_oracle.py) independently decodes
18-byte Q4_0 blocks with FP16 scales and low/high nibble ordering, computes scalar
embedding row lookup/scaling, matrix/vector products, tensor errors, stable full-vocabulary log-softmax/KL,
top-1 agreement and router boundary margins. Python binary64 reductions are
analysis references; this is not an FP32 execution model or a full model oracle.
It also provides explicit-pair RoPE arithmetic over frozen angle inputs, scalar
`cap * tanh(x / cap)` softcap, elementwise residual addition, weighted vector
reduction with already-selected weights, and an explicit router RMSNorm → inverse-
sqrt-width scale → learned elementwise scale → projection chain. Router dimensions
and scales remain caller-supplied; none are inferred from model data. The helpers
preserve unpaired dimensions, reject overlapping/invalid pairs and mismatched
vector shapes, and require finite inputs. `router_selected_weights` also computes
stable softmax probabilities and renormalizes explicitly selected probabilities
with the pinned minimum denominator; expert IDs remain caller-supplied, so this
does not emulate ggml argsort ties. `gemma4_router_mixture` composes those stages
with explicit selected-expert output vectors in the same supplied order. These
verify scalar transforms only; they do not derive Gemma frequency policy or
establish full graph equivalence. The latest oracle additions are untested, as
recorded below.
The router helper's deterministic tie order is diagnostic, not a claim about
llama.cpp's tie policy. Zero norms have no cosine; acceptance must use the absolute
error rule. No helper currently decides model acceptance.

Synthetic tests cover all nibble values, zero/negative/subnormal/maximum finite
FP16 scales, invalid scales and block lengths, cancellation in scalar reductions,
zero vectors, nonfinite rejection, analytic KL and router ties. [native_q4_verify.py](../scripts/native_q4_verify.py) subsequently compared
190464 synthetic blocks with the installed `dequantize_row_q4_0` C function:
63488 finite FP16 scale bit patterns times three payload patterns. The patterns
cover every nibble value in both halves, all-zero and all-15 nibbles. All decoded
FP32 bits match, including signed zero. A separate audit confirmed the native
library hash matches Epic 1's resolved-library provenance. This CPU-only check
loads no model and does not establish tensor, quantizer or graph equivalence.
No native model intermediate tensor capture has occurred. The independent scalar
work is now complemented by the two-position target-logit smoke captures below.

The independent scalar decoder was first checked against 30,720 blocks from the
local model's routed expert tensors. The checker was then expanded to sample all
266 Q4_0 tensors: first/last plus up to two deterministic interior blocks per
ordinary tensor, and the same selection per expert for routed tensors. All 31,544
sampled blocks (1,009,408 decoded FP32 values), across all 266 tensors including
60 routed-expert tensors, matched the pinned native CPU decoder bit for bit. The
full model hash and GGUF descriptor were verified before reading blocks. This
was read-only: no model load, inference or GPU initialization. The first report
is under `.private/epic2-q4-model-blocks-v1`; the broader audit is under
`.private/epic2-q4-model-blocks-v2`. Sampling does not establish every weight,
quantization provenance, matrix multiplication or graph correctness; B2 remains
in progress.

Additional scalar helpers cover RMSNorm, tanh GELU, softmax, small GELU-gated FFNs
and one-head masked attention. [Native CPU findings](native-operator-findings.md)
record bounded one-thread synthetic cross-checks against hashed installed native
CPU/base libraries. The initial capture failed to materialize a fused RMSNorm
intermediate; marking saved tensors as outputs corrected it while preserving the
terminal weighted result on this fixture. Corrected RMS/softmax/projection and
analytic attention outputs meet the existing small-FP32 criterion. GELU and four
dependent outputs, including target-helper GEGLU_SPLIT, fail it: pinned source
and a separate rounded diagnostic show FP16 lookup semantics on the fixtures.
The strict mathematical failures are retained; no tolerance or precision exception
changed. This is a reference-contract finding, not full model or GPU correctness.

[reference-correctness-plan.md](reference-correctness-plan.md) maps target
operations to exact-revision callback names, independent checks, proposed limits
and failure conditions. It budgets eight teacher-forced 513-token sequences
(4096 scored positions), full-vocabulary FP32 shards of at most 32 positions,
two reference repetitions totaling 8 GiB of logits and at most 2 GiB for other
pilot evidence. Active-node coverage, additional hooks, actual position streams
and reference variability remain required. Direct-token scope is reviewed below
for numerical logits only. Candidate corpus/task selection and native token replay
have since completed as described below; the corpus/task review accepts only the
narrow diagnostic scope described below, while broad quality acceptance remains
open.
The table and disk arithmetic do not constitute a captured or calibrated oracle.
[reference_positions.py](../scripts/reference_positions.py) has since generated
an exact calibration-only direct-token plan: eight 513-token streams, 4096
explicit next-token targets, source spans/hashes/reset policy and an 8 GiB
full-vocabulary budget for two repetitions. An independent audit verified every
stream/position against the calibration pool. Overlapping/circular windows and
direct-input scope has been reviewed for this numerical comparison only: frozen
IDs remove rendering/tokenization variation, while the resulting rows remain
correlated calibration coverage rather than independent quality samples. Rendered
bytes are inapplicable here, but remain required for workload evidence. The
complete plan remains uncaptured and no held-out quality data entered this
calibration plan. A separately reviewed run window/resource envelope is still
required.
The refreshed native-linked helper now accepts the exact input plan through
`--validate-only`; the hash-bound audit confirms 4 GiB of logits per repetition
and 8 GiB for two. This parser check did not initialize a backend or load a model,
and no logits were captured.

Added [context_positions.py](../scripts/context_positions.py) to prepare a
calibration-only boundary probe: three 1,027-token sequences with nine selected
logit rows at positions 1023/1024/1025. The private plan is in
`.private/epic2-context-boundaries-v1`; its footprint is 9 MiB per repetition.
Tests verify exact target alignment, capacity/BOS checks, unique streams and
calibration-only source selection. The target GGUF descriptor confirms the
1,024-token sliding window and a 30-layer pattern containing 25 local and 5
global layers. Pinned Gemma 4 source maps these values into the hparams consumed
by its SWA graph mask; exact descriptor/source hashes are bound in the private
target-window audit. This is static configuration evidence, not runtime mask
validation. The native adapter now accepts
selected-row protocol RVLOG003: it evaluates each full context but writes only
the preregistered rows, capped at 96 full-FP32 rows. A pinned-header API-double
control-flow check covers sparse output rows, and the compiled parser accepted
the exact private manifest in `--validate-only` mode. This input plan does not
test semantic retrieval or other long-context behavior and has no captured logits.

[logit_compare.py](../scripts/logit_compare.py) streams explicitly supplied private
little-endian FP32 shards and checks index identity, unique/aligned positions,
full vocabulary width, file sizes/hashes, private shard modes and finite values.
Both streams must reach EOF so the last shard's hash is verified. It reports
float64 error/KL, top-1 mismatches and whether 4096 positions are present, without
asserting correctness acceptance. Synthetic tests cover analytic KL, insufficient
coverage, changed final-shard bytes, mismatched positions, invalid paths/shapes
and nonfinite values. Evidence-bound comparison now verifies retained supervisor
bundles, exact index/result hashes, successful cleanup and distinct child processes;
an integration test rejects a tampered binding. Actual two-position repeats have since been compared below;
the full position and tolerance coverage remains outstanding.
The version 3 index binds the exact ordered position inventory and count to its
plan hash; bundle-attested comparison also binds those fields to the supervisor
run identity. It requires matching shared input/context digests and
pre/post-softcap stage before reading rows. Equal sequence labels alone do not
establish equal inputs. Private index validation rejects symlinks, public modes,
nonregular/unowned files, duplicate JSON keys and indexes over 16 MiB. Synthetic
regressions verify that changed input identity or stage fails before shard access.
Legacy v2 captures remain numerically comparable with provenance marked but cannot
claim a plan-bound 4096-position inventory; v1 remains rejected.
Capture provenance still requires independent validation of the bound manifest.

An isolated [native producer](../research/capture_logits.cpp) and
[offline preparation/finalization adapter](../scripts/logit_capture.py) now bridge
the planned input to version 2 logit shards. The producer compiled against pinned
headers/libraries with strict warnings and FP contraction/fast math disabled.
Native input-only validation accepted the actual 4096-position plan and rejected
seven malformed binary inputs, without model/backend initialization. Synthetic
finalization tests validate sequence-to-shard alignment, exact inputs, complete
supervision, modes, finite values and retained partial failures. The initial
validation did not execute the CPU model path. Two real tiny captures have since
exercised loading/decode and CPU registration; actual helper cancellation and
broader numerical coverage remain unverified. See the [capture plan](reference-correctness-plan.md) for the
fixed policy and required tiny smoke/equivalence checks before reference adoption.
The separate optional native API-double check also passed 14 control-flow
scenarios against the same producer and pinned header signatures. It verified
exact token/position/reset order, shard boundaries and retained failures, including
forced cleanup of simulated TERM-ignoring load/decode hangs. It links no native
inference libraries and opens no model. This strengthens adapter validation, not
real execution, ABI, numerical or cancellation evidence by itself.

Two exact completion replays now use the native verbatim-file mode and verbose
prompt logging. Every one of the 15 frozen input IDs matched in each replay, and
generation bytes matched across the traced and untraced runs. The new
[native diagnostic reader](../scripts/native_completion.py) parses native host
phase times, verifies executed IDs and derives a prefill rate only when the phase
covers the full frozen prompt. Evaluation counts are not substituted for
committed output/delivery counts; decode throughput, TTFT and output rate remain
explicitly unavailable without the appropriate boundaries.

The real capture helper then completed twice on a fixed calibration-only
three-ID prefix, scoring two positions with reset KV. Each run emitted 2 MiB of
finite FP32 logits at vocabulary width 262144. Independent finalization checked
successful supervision/cleanup, exact inputs, labels, shape, modes and complete
checksums. Corresponding shards were bit identical; streaming comparison reported
zero KL/error on these two unique positions. This verifies the native capture
path and tiny repeatability, not 4096-position coverage, all-layer correctness,
tool-policy equivalence or tolerance calibration. Raw captures, identities and
comparisons remain private in `.private/epic2-exact-replay-v3`.

A later bounded helper run captured all 15 positions of the exact completion
prompt, with a dummy final target, and passed finite/shape/hash/cleanup checks.
Its final-row argmax did not match the first token obtained by retokenizing the
completion's printed output. Original sampled IDs were not captured, and
retokenization can change token boundaries; direct token-ID capture is needed to
determine whether this reflects a real discrepancy. Both outcomes remain private
in `.private/epic2-prompt-logits-v1`; no equivalence claim is made.
Native vocabulary decoding subsequently showed that the helper's six-byte
argmax piece is absent from the 13-byte completion output. Its top-two logit
margin was approximately 0.393. This adds byte-level evidence but does not
isolate the cause: sampled IDs and matched batch/buffer settings are still needed.
Correction: the apparent sampler-history debug statement is commented out in the
pinned source. The first bounded debug attempt reached the 90 C CCD stop during
loading; a one-thread follow-up completed with output bytes matching the earlier
two-thread run but emitted no sampled IDs. A vocabulary-only native probe found
two possible first-piece IDs, neither equal to the helper argmax; their helper
logit ranks were 75185 and 7. Output framing, batch and extra-buffer differences
still prevent definitive attribution. The attempts and probe remain private; no
thermal limit or numerical tolerance was increased.
Direct sampler interception has since captured both actual completion token IDs.
The probe was synthetically verified to forward the sampler return unchanged;
the bounded one-thread completion succeeded and reproduced prior output bytes.
Its first sampled ID differs from the helper argmax. Batch, thread and buffer
settings differ, so the cause is not established. Private probe/source/capture
evidence is in `.private/epic2-sampler-interpose-v1` and `-v2`.

An extra-buffer-only helper control reproduced a substantial change on the same
15 prompt positions. Enabling extra buffers changed one top-1 token; the final
argmax then matched the completion's directly captured first sampled ID. Mean
KL was about 0.0927 nats, maximum normalized RMSE 0.331 and minimum cosine 0.944.
The source differed by exactly the buffer-setting line; helper batch/thread
settings, inputs, compiler flags and libraries matched. Both settings have now
been captured twice; each same-setting repeat was bit-identical across all 15
positions, and the cross-setting result was unchanged. This limited same-session
repeat does not isolate a specific kernel or explain full-model behavior.
No proposed tolerance was relaxed. Private evidence and the independent source/
capture audit are in `.private/epic2-extra-buffer-control-v1`.
Runtime logs show an actual `CPU_REPACK` allocation only in the enabled capture;
some tensors fell back to the ordinary CPU buffer. The output discrepancy is
already present at position 0 (normalized RMSE approximately 0.0565, KL 0.00297
nats) and grows on the final position of this fixture. This rules out a dependency
on earlier prompt-token history for the initial discrepancy, while leaving
layerwise propagation and the responsible operation unresolved. Per-position
analysis remains private alongside the original captures.
A model-free synthetic 256-by-16 Q4_0 matrix-vector probe exercised both the
ordinary CPU and initialized `CPU_REPACK` paths. Their outputs were identical
on this fixture; each differed from an independent scalar Q4_0/matvec result by
up to approximately 0.013855. This case does not reproduce the model-level
discrepancy or establish the error's source. Routed `MUL_MAT_ID`, other shapes
and activation distributions remain untested. Private evidence is in
`.private/epic2-q4-repack-synthetic-v1`.
A distinct-expert synthetic routed Q4_0 `MUL_MAT_ID` check also produced identical
ordinary/repacked outputs on its single shape, with maximum absolute error about
0.138977 versus independent scalar math. The first attempted fixture had identical
expert values and is retained as inadequate for expert-ID verification; the
corrected fixture selected distinguishable experts. Neither fixture reproduces
or explains the full-model difference. Private records are in
`.private/epic2-q4-routed-synthetic-v1` and `-v2`.
Pinned source selects Q8_0 activation vectors for the Q4_0 CPU dot product. An
independent Q8_0 reconstruction for these integer fixtures, including FP16 scale,
matches both native paths within approximately 5e-8. This explains the observed
fixture errors as consistent with activation quantization; the original strict
mathematical comparisons and failures remain recorded. It does not explain the
full-model extra-buffer difference or establish a new tolerance.

Earlier exact-replay attempts stopped on collector response deadlines. Per-source
tracing observed pauses between individually fast reads; the underlying cause
was not established. Per-read regular-file trace I/O was removed from the critical
worker, collector-only syscall tracing used a private RAM spool for one run,
and the untraced repeat also passed. The loaded CPU profile explicitly uses a
1 s hard response deadline with nominal 100 ms cadence and retained slow-cycle
warnings. Failed attempts and their cleanup evidence remain preserved in the v1
and v2 exact-replay directories; no physical resource limit was increased.

## Candidate quality inputs

[quality-corpus.md](quality-corpus.md) records original task definitions, pinned
primary-source license evidence, selection/split policy and limitations. The
100 held-out scored tasks have 20 separate calibration tasks, with frozen public
recipe hashes and strict visible-JSON scoring. All 120 prompts passed native
render/token replay; an independent audit verified every record. A fresh complete
model hash matched the prior reference. Conservative paired task-accuracy bounds
show that even 100 perfectly matching task pairs leave a 3.62-point upper bound.
The iid best case needs at least 122 pairs with zero discordances to fall under
the proposed 3-point margin; shared-template dependence means this is only a
planning lower bound. Broader task families or cluster-aware analysis need review.

[fidelity_corpus.py](../scripts/fidelity_corpus.py) built 550 whole documents with
100128 held-out scored tokens (50085 problem prose, 50043 Python code), plus
2021/2037 calibration tokens respectively. Each selected document passed exact
native replay. Independent checks verified source/corpus/text/token hashes,
source-derived document IDs and splits, exact text/code deduplication, next-token
counts, context ceiling and private modes. Pinned source text/license evidence,
selected documents/IDs and verification remain private. Public recipes contain
provenance/definitions/hashes rather than raw external data. The corpus covers
math-problem prose and Python, with no broad-language or pretraining-exclusion
claim. Scoped corpus/scorer review accepts these fixed inputs for descriptive
diagnostics, not broad quality acceptance. Actual NLL/task evaluation, stochastic
and context checks remain required; no quality result follows from tokenization.

Added a native streamed NLL path to `capture_logits.cpp` and the offline
[NLL capture adapter](../scripts/nll_capture.py). `RVNLL001` carries exact corpus
token streams; the CPU helper reduces each post-softcap full-vocabulary row to a
stable long-double target NLL and writes one FP64 scalar per target. The adapter
prepares/finalizes private corpus-aligned captures, checking token hashes, exact
target counts, receipts, finite values and successful reaped supervision. Its
calibration/holdout bundles contain 4,058/100,128 targets (32,464/801,024 raw
bytes). A pinned-header API-double build passed 16 scenarios, including synthetic
NLL values against the Python binary64 reducer. The updated helper was then built
with strict flags against the previously verified pinned llama/ggml libraries;
unchanged source-library hashes were confirmed, and an explicit loader audit
recorded 11 resolved shared-library hashes. The native-linked helper accepted
both full corpus bundles in `--validate-only` mode, which does not call backend
registration or load a model. The four existing real-logit rows independently
confirm reduction arithmetic within 2e-12 nats. No model corpus NLL run occurred. Private
plans are in `.private/epic2-nll-capture-prep-v2`; the API-double report is in
`.private/epic2-nll-capture-adapter-v1`. This does not establish loaded-library
provenance, corpus quality, reference variability or acceptance.

[nll_compare.py](../scripts/nll_compare.py) implements stable full-vocabulary
target NLL and complete document-paired analysis, with corpus/run/token/position
and softcap-stage validation. It uses a seeded document bootstrap within strata,
fixed token-weighted stratum mixture, exact target denominators and explicit
coverage/perplexity-overflow reporting. Analytic cases, mismatch/failure cases
and a separate private CLI fixture pass. No actual model NLL is captured or
compared, and no noninferiority result is asserted.

## Acceptance audit and remaining work

| Child | Current evidence | Work still required |
| --- | --- | --- |
| B1 | 16 workload manifests, 1900-token exact-input controls at capacity 4096/65536 and 29000-token controls at 32768/65536, two exact 15-ID native replays, native phase reader, explicit MTP event reducer contract, resource/supervisor tests, bounded CPU diagnostics, ResourceGuard reduction factory, conflict-checked resource/native/delivery inventory merge, synthetic end-to-end private result persistence, hash-bound paired schedules with fixed safety monitoring, run-bundle-bound paired trial analysis, opt-in child/shared-library identity capture and prelaunch declared-artifact hash checks; an additional opt-in supervisor audit now binds declared file mappings in `/proc/<pid>/maps` to preflight device/inode identities and reports coverage/stability; synthetic tests verify observed mappings, `not_observed` semantics, and rejection of rehashed identity tampering; the verifier exposes these provenance findings to logit/NLL bindings | Controlled collector/workload measurement, validation of the new mapping audit on the approved reference invocation, validated native MTP/device-event producers, appropriate device-health/thermal calibration, reviewed workload/quality freeze and invocation-specific synchronized timing, steady-state, runtime-attested device metric producers |
| B2 | Independent scalar/native decoder checks, operation plan and two bit-identical native repeats covering two full-vocabulary positions; prepared and parser-validated streamed NLL inputs for the full calibration/holdout corpus; streamed pairwise reference-variability analyzer with verified supervisor-bundle/index bindings, distinct child identities and common executable checks; reference-repeat, NLL and logit reports retain validated optional mapped-artifact provenance, with reference repeats requiring a consistent audit setting; intermediate MoE captures can require an explicit exact layer inventory and bind it to both placement runs; stochastic plan/finalizer retains all planned outcomes and emits analyzer input only from complete receipts; process-local sampler observer forwards native token IDs unchanged and its event parser binds IDs to checksummed stderr, sample policy and observer artifact; synthetic forwarding test passes and installed completion `--help` resolves observer ABI without a model or sampler call; distribution analyzer binds >=20 seeded token receipts to checksummed successful supervisor bundles and prelaunch model/observer hashes; native logit capture now explicitly freezes `extra_buffers` in RVLOG003/RV4 and compares the policy hash through supervisor identity | Full 4096-position/all-layer capture and adequately sized repeated references, reviewed tolerance calibration, session-level independence and equivalence of the selected policy to the installed reference tool; actual sampled-output captures, context/MTP execution and NLL/task evaluation |
| B3 | Gate 0 NO-GO remains authoritative; no runs | Approved candidate reference and safe idle window, B1/B2 completion, full-preset reproduction, paired one-factor/control matrix with all failure records and historical uncertainty comparisons |
| Gate 1 | Incomplete; private bundle verifier checks internal hashes, byte totals, retained-cycle p95, child executable identity fields, expected-artifact prelaunch checks and independently rederives supported ResourceGuard summaries from raw rows; it reports legacy gaps. Prelaunch hashes alone do not attest runtime-loaded model identity | Independently replay short actual manifest, audit recorded executable against the manifest, obtain runtime-attested model/library identity and correctness, recalculate native/device summaries, review/freeze comparison and tolerance standards; record evidence-backed GO/NO-GO/PIVOT |

No child is closed by these offline tests. The unchanged benchmark exporter is
not used to publish richer private bundles. The public diff includes no raw capture, model, copied router configuration,
raw external corpus or engineering-memory text.
No performance, model-fidelity or reference repeatability result is claimed.

## Verification

Run `./scripts/check.sh`; its existing research discovery includes these tests.
The synthetic tests require no GPU, model, endpoint, download or package install.
The public diff must also be manually inspected. The latest `./scripts/check.sh`
passed 179 tests (18 guardrail and 161 research), along with formatting, syntax,
links, worktree/history privacy scans and the dependency-cycle check. This covers
the `router_projection` and `router_selected_weights` fixtures. The subsequently
added `gemma4_router_mixture` composition helper has no direct fixture coverage
yet. Native
workload construction, hash/count/mode verification, one separate native workload
replay and the Q4_0 native cross-check also passed. All 120 quality-task prompts
and all 550 selected corpus documents passed native token replay and separate
hash/count/split/mode audits. The collector-cost probe completed with an unmet
host-cost bound. Controlled telemetry workload impact, full intermediate/logit
captures, sampled-output captures, actual NLL/task evaluation and GPU performance
experiments remain unrun. Two-position
native logit smoke captures and exact-ID completion replays are verified above. One subsequent maintainer-requested bounded CPU replay
timed out at its unchanged 60-second limit after 576 supervisor telemetry cycles;
termination and collector cleanup were confirmed. This incomplete attempt is retained
privately and does not alter the earlier exact-replay evidence or establish completed
evaluation. A bounded post-run audit found all 15 frozen prompt IDs in the captured
native log but no prompt-evaluation or decode-evaluation phase timing. This confirms
the exact input was logged by the process; it does not establish that model evaluation
completed, identify which evaluation stage consumed the remaining time, or explain the
timeout. No timeout increase or automatic retry follows from this observation.
The bounded native CPU synthetic operator experiment ran with five retained
mathematical-FP32 failures as documented above; green repository tests do not
supersede that negative numerical result.

The initial final scan stopped on two generic credential-pattern matches in
optional public artifact-digest metadata. Both values were independently
recomputed from the reference tokenizer metadata and native binary. The optional
public duplicates were removed while complete private identity stayed intact;
no scanner, policy or allowlist changed. The subsequent worktree/history scans
and full check pass. No credential material was identified by that investigation.

## GPU baseline campaign and Gate 1 (2026-11-12)

The campaign that closes Epic 2 is recorded in the [Gate 1 review](gate-1-review.md),
with the updated [Gate 0 outcome](gate-0-review.md) and the revised
[run rules](development/rule-audit.md#gpu-baseline-revision-2026-11-12).
Tooling: [gpu_campaign.py](../scripts/gpu_campaign.py) with synthetic tests in
`tests/research/test_gpu_campaign.py`. The earlier CPU exact-replay timeout was
diagnosed from retained evidence as model-load swap churn. The `CPU_REPACK`
allocation is 13.5 GiB anonymous; the load phase swapped out 5.9 GiB and took
58 s of the 60 s budget. It was not an evaluation hang. Server sessions now have
a separate model-load deadline.

Acceptance closure:

| Child | Outcome |
| --- | --- |
| B1 | Met. Frozen 1900/29000-token manifests replayed with verified `prompt_n` across 31 server sessions; capacity controls separate capacity from occupancy; failed and stopped attempts persisted; no operational endpoint used (private UNIX socket); collector overhead measured end to end; metric inventory in Gate 1 |
| B2 | Met with recorded limits. The operation plan and proposed limits are unchanged; calibrated noise floors come from deterministic CPU and Vulkan teacher-forced repeats. 1792 rather than 4096 positions were captured because the calibration split holds 4082 tokens; 4096 remain planned. The unexplained Vulkan–CPU divergence is carried forward |
| B3 | Met under the maintainer's minimal scope: four presets × three sessions plus seven one-factor controls; interaction cells deferred |
| Gate 1 | GO (restricted), with conditions in the review |
