# Private benchmark and analysis formats

Version: proposed v1. These formats support the [experiment protocol](experiment-protocol.md);
they are not public export schemas or run approvals. All captures, exact commands,
model paths, request bytes/IDs and result records remain private under the
[privacy policy](development/privacy-policy.md). No endpoint is configured by
these modules. The narrow public benchmark exporter remains unchanged.

## Capture bundle

Use a new capture directory inside a separately created mode-0700 private campaign.
[benchmark_core.py](../scripts/benchmark_core.py) exposes `supervise(command,
directory, timeout_s=..., ...)` for explicit local children. It creates 0600 files,
uses that private directory as child cwd and sets child umask 077. Specify a child
`env` map to avoid inherited native configuration overrides; inherited mode is
marked unfrozen. Exact explicit environment values belong in private run identity;
the supervisor stores only their canonical SHA-256, not the caller's environment.

| File | Meaning |
| --- | --- |
| manifest.json | Immutable initial command, limits, optional run/trial identities, environment policy/digest, safety telemetry policy, optional measurement-collector state and metric inventory; initial status remains incomplete |
| stdout.txt / stderr.txt | Bounded private raw child output; never a structured metric record by itself |
| telemetry.jsonl | Optional timestamped preflight/running snapshots, collection cost and rejection reason; not a public JSONL artifact |
| result.json | Authoritative terminal status, reason, launch/exit information, pinned child executable identity, captured bytes, elapsed supervision time, and every full telemetry-cycle duration used for the reported p95 |
| collector-stderr.txt / collector-invalid-response.txt | Optional bounded private collector diagnostics; invalid-response prefix exists only on failure |
| collector-result.json | Optional collector transport and owned-leader cleanup outcome |
| checksums.json | SHA-256 of known capture/collector files after they close; excludes child-created arbitrary artifacts |

[`verify_capture_bundle.py`](../scripts/verify_capture_bundle.py) performs an
offline consistency check with
`/usr/bin/python3 scripts/verify_capture_bundle.py --directory "$PRIVATE_RUN"`.
It rehashes indexed private files, checks bounded file modes/ownership, verifies
telemetry row ordering and byte totals, recomputes cycle p95 from retained
durations, and validates the persisted metric inventory/provenance contract.
When a result carries ResourceGuard coverage metadata, it independently
recomputes the supported RSS/PSS/anonymous/file/high-water and VRAM peaks,
MemAvailable minimum, process and swap deltas, temperature endpoints, memory
growth and eligible GPU-busy mean from the raw running rows. It rejects a
rehashed result whose sourced values disagree.
Older bundles are reported with explicit legacy gaps when they lack cycle
durations, terminal metrics, units or child executable identity. Unindexed child files require manual
review. The checksum index is not an external authenticity anchor, and this
check does not prove that run identity matched the executed process or recalculate
native synchronized phases, output delivery, energy, PCIe or device-allocation
metrics or loaded shared-library/model identities from raw sources. The
supervisor pins `/proc/<pid>/exe` immediately after spawn, verifies the child
start time remains stable across capture, and hashes the pinned executable only
after cleanup; that post-run hashing is excluded from `elapsed_s`. The optional
`audit_loaded_libraries=True` scans `/proc/<pid>/maps` at the configured cadence,
pins at most 128 shared-object files with paths up to 1024 characters and 4 GiB total, and hashes those files
after cleanup. It excludes model mappings and static code, records scan time
separately, and never stops a workload solely because its inventory is
incomplete. Enabling it adds perturbation and must be held fixed within a
comparison; these hashes do not identify model-file content.

Callers may opt into `expected_artifacts`, a bounded list of role/path/SHA-256/size
records. The supervisor securely opens and streams each file before starting the
timed interval, records matched/mismatch/unavailable outcomes, and blocks child
launch unless every declared artifact matches. The manifest and result retain
private paths. This closes a prelaunch identity check only: a file can change
after the check, and the check does not prove that the runtime loaded the declared
model or tokenizer bytes. It is opt-in and hashing time is excluded from
`elapsed_s`.

`audit_mapped_artifacts=True` additionally samples `/proc/<pid>/maps` for the
declared artifact device/inode identities. It holds each checked file open and
verifies its device, inode, size, modification time and change time stay stable
from audit setup through cleanup. The result distinguishes observed mappings
from roles that were never observed, and reports scan durations and incomplete
coverage. This establishes that a file with the preflight-checked inode was
mapped during the child lifetime; it does not prove every byte was read or used,
and non-mmap loading may correctly report `not_observed`. Keep the audit switch
fixed within paired comparisons because `/proc` sampling adds overhead. Audit
time is recorded separately from the workload metric boundary.

The bundle verifier exposes this sanitized mapping summary to logit and NLL
capture/run-binding reports and to reference-repeat analysis. Reference repeats
require a consistent audit-enabled setting across their supervisor bundles;
their per-run mapping summaries remain available in the process evidence.
These reports carry mapping status and per-role
observation/stability while leaving numerical comparison and acceptance rules
unchanged. `not_requested`, `not_observed` and incomplete audit status remain
visible provenance gaps; they are not silently converted to a positive finding.

A missing terminal result means incomplete evidence. A successful child is not a
correct model or a valid benchmark. Output caps retain the prefix and mark the
run incomplete when data is discarded. Timeout, resource rejection and nonzero
exit are retained, including preflight rejections that launch no child. There is
no overwrite, evidence deletion, retry or automatic publication.

Older result bundles may retain only the cycle count and p95 summary. Their
telemetry `collection_ns` measures callback/response work before telemetry JSONL
serialization and flush, so it cannot reconstruct the full-cycle p95. New
supervisor results retain all full-cycle durations in `result.json` for exact
recalculation.

Paired collector perturbation keeps the safety telemetry channel active in both
arms. The optional measurement worker has separate `measurement-telemetry.jsonl`
and collector artifacts, and the plan-bound p95 includes only that worker plus
serialization/write/flush. Safety-monitor cost is excluded from this p95 but stays
inside the end-to-end workload perturbation. A disabled arm reports no optional
collector p95 rather than a fabricated zero-cost measurement. Both JSONL streams
share the run's single `max_telemetry_bytes` cap, and each collector request sees
the combined retained-byte count.

Elapsed supervision time includes cleanup and is not a model metric. A bounded
post-kill leader wait is separately configured by `cleanup_timeout_s` (default 1 s).
A kernel-blocked task can outlive SIGKILL; pending cleanup has an owned PID and
null exit code, forces incomplete status and must stop further trials pending
state/resource verification. Direct-child reaping and group signals do not prove
device memory release. Cleanup latency is additional to the request timeout and
termination grace; it must be included in a reviewed run envelope. TTFT and
steady delivery rate require the explicit monotonic delivery timestamps accepted
by `delivered_metrics`; the latter uses N-1 inter-token intervals. Synchronized
prefill/decode spans and MTP denominators need native instrumentation. The metric
inventory keeps unsupported data null with a reason and explicit unit; zero does
not mean missing. A shared unit table covers every metric key, so a later adapter
can fill a value without changing its interpretation.
`linux_metrics` adapts supplied snapshots, while `ResourceGuard` supplies safety
rejections. Importing either module probes no devices or processes.

[`benchmark_metrics.py`](../scripts/benchmark_metrics.py) reduces explicit
`linux_metrics` samples or captured ResourceGuard telemetry rows to the run-level
resource inventory without reading hardware. ResourceGuard preflight rows are
excluded; only `running` stage rows contribute. It reports sampled memory
peaks/minima, process counter deltas only
when the process start identity is stable, and system-wide swap deltas with
unknown attribution. Memory growth is maximum sampled RSS minus the first sampled
RSS for the same process identity. Idle VRAM and steady VRAM percentiles require
separately supplied idle samples and a post-warmup boundary. Time-weighted GPU busy/clock
means require a caller-declared maximum sample gap; otherwise they remain null.
Board energy integrates only between supplied power samples and reports coverage
and maximum gap. Missing, reset, incomplete or unvalidated sources remain null
with a reason. Event, PCIe and per-process device-allocation values remain null
until their respective adapters are supplied.

The same module adapts explicit committed-token delivery timestamps and merges
complete metric inventories from resource, native-phase and delivery producers.
Every field must match the shared unit/missing-reason schema; absent values may
be filled by another producer, while conflicting measured values reject the
merge. Native prompt-evaluation throughput carries its `tokens/s` unit and is
not treated as target decode rate. Delivery TTFT/end-to-end/steady rate remain
separate from synchronized device phase timings.

With `campaign_root` and `campaign_limit_bytes`, preflight counts existing logical
file sizes (including failed captures), projects both capture caps, a bounded
result-duration array (half the telemetry cap) and 64 KiB of other result
metadata, then checks projected disk reserve. It checks campaign growth
again on cadence. Accounting requires 0700 directories and regular 0600 files,
rejects symlinks/inaccessible trees and never reads their contents. It assumes
one sequential writer, not concurrent reservations. Periodic checks are not a
filesystem quota: child writes, growth between checks and final failure metadata
can overshoot a very small limit. Choose a realistic campaign budget before
starting, retain any breach and do not delete previous evidence to regain space.

`telemetry(pid_or_none, elapsed_s, captured_bytes)` returns a JSON object with
optional `rejection` (nonempty reason or null). It is called with null PID before
launch; a rejection prevents launch. Callbacks must be bounded; this synchronous
API cannot interrupt a hung callback. Collection cost includes callback work;
collector-cycle cost additionally includes serialization/write/flush, not the
entire supervisor or workload-time perturbation. Keep source cadence/units and
power integration coverage explicit. Readable attributes do not prove device
health, counter accuracy or process attribution.

## Isolated collector protocol

For potentially blocking source acquisition, pass `telemetry_command` instead of
a Python `telemetry` callback; supplying both rejects the capture. The explicit
trusted collector runs in its own process group, private cwd and the same selected
child environment. [collector_worker.py](../scripts/collector_worker.py) provides
`serve(callback)` for an explicitly configured collector. It imports no hardware
inventory and exposes no endpoint. Worker responses are JSON data, never pickles,
commands or evaluated source. This is process/lifecycle separation, not a sandbox
for arbitrary code.

Each stdin request is a single newline-terminated JSON object with `request_id`,
`pid` (null for preflight), `elapsed_s` and `captured_bytes`. Requests are bounded
at 4096 bytes and IDs increase by one. A response is one JSON line containing the
same integer ID, monotonic `started_ns` / `completed_ns`, and a snapshot object.
The acquisition interval must lie between the parent's request and receipt times.
The parent rejects duplicate keys, malformed/nonfinite JSON, extra frames,
incorrect/stale IDs and invalid acquisition times. Successful transport does not
validate snapshot accuracy, hardware freshness, health or resource limits.

`telemetry_timeout_s` defaults to 250 ms, additionally capped by remaining workload
time. A response timeout, dead collector or malformed response becomes a retained
rejection: preflight launches no workload, and a running failure stops the owned
workload. This response deadline is a liveness bound, not a relaxation of the
proposed 1 ms p95 collector-cost criterion. Valid samples retain request/acquisition/
receipt times under `collector_transport` for a separate audit. In process mode,
`collection_ns` includes the pipe round trip and response validation; the separate
worker interval bounds provider work plus its validation. Neither is GPU event
time or worker CPU time. Source updates or cached hardware values still require
counter calibration.

The [loaded CPU profile](development/rule-audit.md) explicitly selects a 1 s
response deadline after retained loading-time pauses exceeded the earlier 250 ms
limit. The nominal cadence remains 100 ms, with actual gaps/slow cycles retained.
This bounds diagnostic liveness; it is not a benchmark-overhead acceptance.

`telemetry_startup_timeout_s` can separately allow bounded initialization before
launch (the current CPU profile uses 1 s). Subsequent responses retain the running
deadline, and both phases remain capped by total request time. Collector host-cost
targets are measurement criteria; small overruns are recorded diagnostic warnings.
See the [rule audit](development/rule-audit.md) for current paging and profile rules.

The supervisor accepts an optional post-cleanup `metric_adapter(result, rows)`.
It returns named complete inventories that pass the shared unit/missing-reason
schema and conflict checks; accepted metrics and per-field source provenance are
stored in private `result.json`. An adapter exception preserves the child run
status, records only the exception class, and leaves all metric values null with
reasons. The adapter is opt-in, runs after process/collector cleanup, and does
not add hardware probing or a default inference metric source. The
`resource_guard_metric_adapter` factory summarizes only captured `running`
rows, preserves sample/energy coverage metadata, and can merge caller supplied
native-phase or delivery inventories.

Responses are capped at the smaller of 64 KiB and the telemetry cap; collector
stderr is capped at the smaller of 64 KiB and the workload output cap. Campaign
projection includes both extra bounds. Invalid response bytes are retained only
as a bounded private prefix. Worker stdout is reserved for protocol frames;
callback logging there invalidates framing. Callback exceptions return a failed
snapshot with an exception category, without printing private exception details.

The supervisor stops its workload before waiting for collector cleanup. The
collector gets SIGTERM and a 50 ms grace before SIGKILL, followed by the bounded
leader wait. An unexpected worker exit is retained even if the workload already
completed. A closed collector transport is not a successful benchmark. Kernel-
blocked tasks and detached sessions cannot be promised reaped by this mechanism;
collectors must not daemonize, and pending cleanup must prevent another trial.
OS scheduling and synchronous filesystem/caller callbacks are not real-time
bounded by the pipe deadline.

## Immutable run identity

[benchmark_manifest.py](../scripts/benchmark_manifest.py) inventories every
protocol identity group and validates core fields. `make_identity` accepts
`provided[group][field]` values and fills all omissions with null plus the supplied
reason. `annotate_missing` adds precise nonapplicability explanations; it cannot
replace a known fact. `validate_identity` checks checksum/schema/values and returns
missing comparison prerequisites. A complete schema or review-reference string
cannot prove that a run was reviewed or that recorded facts are true.

Artifact identities use full SHA-256 and byte size. Workload files retain exact
rendered bytes, integer token IDs, their hashes, policy and capacity separately.
A shared prompt at different capacities must retain the same render/token hashes;
the whole workload manifest hash changes with capacity. Raw samples, loaded
libraries, source/build identity and instrumentation overhead still need actual
capture; a schema is not a substitute for them.

The offline [capacity control builder](../scripts/capacity_controls.py) verifies
these invariants when rebinding a manifest: source checksum, prompt/render/token
identity and the 512-token output plus 512-token control reserve. It changes only
configured capacity and the manifest checksum. This prepares exact-input controls;
it does not establish that a runtime can allocate or execute at that capacity.

## Short paired analysis

[paired_analysis.py](../scripts/paired_analysis.py) accepts a private JSON object
with `plan` and `runs` and writes a new private summary:

```sh
/usr/bin/python3 scripts/paired_analysis.py "$PRIVATE_RECORDS" --output "$PRIVATE_SUMMARY"
```

Before execution, freeze a validated plan and its randomized order with
`/usr/bin/python3 scripts/paired_analysis.py --prepare-plan "$PRIVATE_PLAN" --output "$PRIVATE_PACKAGE"`.
The private package binds the plan, each schedule ordinal and the exact run
count with SHA-256 digests; no workload is launched. For final analysis, provide
`{"package": ..., "runs": [...]}` so the package is revalidated and its
schedule hash is retained in the summary. Collector-on/off packages require an
identical enabled safety-monitor profile in both arms.

The plan schema is `rosevine-paired-plan-v1`. Required fields are `factor`,
`configurations`, `workloads`, `metric`, `seed`, `warmups`, `trials`,
`bootstrap_resamples` and `invalid_conditions`. Configurations are named reference
and candidate; exactly one named capacity/first-N/MTP/collector-enabled factor may
differ. `collector_enabled` is an explicit boolean for the optional measurement
collector. Collector-on/off plans must also carry the same identified
`safety_monitor` (`enabled: true`, with a profile hash) in both configurations;
safety monitoring cannot be toggled off as part of the instrumentation factor.
Collector perturbation plans must compare lower-is-better `end_to_end_s` in
seconds. The summary reports collector-on paired median workload-time increase
and its 95% interval; it marks the strict +/-1% equivalence target only when
every submitted run is bound to a verified supervisor bundle and the complete
interval lies inside that range. A host collector-cycle p95 remains a separate
measurement-quality result. Bundle-bound collector analyses also pool retained
running-cycle durations by configuration and report p95 against the 1 ms target;
these are specifically `measurement_collector_cycle_durations_ns`: the optional
measurement collector's request/response, serialization, write and flush cycle.
The required safety monitor remains enabled in both arms and has separate
`collector_cycle_durations_ns`. The end-to-end paired comparison captures the
combined runtime impact of enabling the optional collector; neither metric alone
establishes that all observer effects are below the target.
Model, build, sampler and all other configuration values must match. Artifact omissions
need explicit reasons and a separately named text-only control; MTP requires an
assistant identity. These structural checks do not validate effective settings
against a native process.

Each workload supplies ID, rendered/token SHA-256 and occupied input length
(1900 or 29000). Configured capacity reserves 512 outputs plus 512 control tokens.
The primary metric specifies name, unit and benefit direction. At least two
warmups and seven measured pairs are required. The seeded schedule is exactly
`paired_schedule`; its ordinal, prompt, pair, configuration and seed identify
each planned observation. Preserve that order during execution; record input
array order can differ because analysis joins by ordinal.

Every run contains `run_id`, `ordinal`, `phase`, `pair`, `workload`,
`configuration`, `seed`, `configuration_sha256`, `tokens_sha256`,
`rendered_sha256`, terminal `status`, `invalid_condition` and `metric`.
Metric is exactly `{value, reason, unit}`: finite nonnegative value with null
reason, or null with a nonempty missing reason. Units must equal the plan's unit.
Statuses are successful/failed/incomplete. Invalid-condition IDs must be
preregistered; unknown exclusions, duplicate/unplanned ordinals, mismatched seeds,
wrong prompt hashes, changed configurations or conflicting units reject analysis.

For evidence-bound analysis, also supply a `run_bundles` mapping from every
`run_id` to its retained supervisor bundle directory, and pass the same trial
identity to `benchmark_core.supervise(..., trial_identity=...)`. The analyzer
verifies each bundle, matches the complete scheduled trial identity and terminal
status, and compares the primary metric value/reason/unit against the persisted
result metric. Bundle paths stay in the private input and are not copied to the
summary. Without this mapping the analyzer still describes the submitted records
but sets `run_bundle_provenance_verified` false; such a summary is not bound to
the captured execution. Even verified bundles do not independently prove every
effective runtime setting or loaded artifact beyond the fields and checks the
bundle actually records.

Exclusions remove whole measured pairs. Failed and missing runs remain listed.
A missing planned run, unregistered failure/missing metric, fewer than two valid
warmups per configuration or fewer than seven eligible pairs prevents a paired
interval. Preregistered invalid trials can be excluded if at least seven pairs
remain, with counts and pair IDs retained. Zero reference values have no relative
improvement interval. Summaries include median/range/IQR/p95/CV; fewer than 30
samples carry a descriptive-p95 caveat. The default seeded percentile bootstrap
resamples paired prompt/seed trials, never tokens. No statistical output is a GO
without correctness, resource acceptance and reviewed gate criteria.

## Intermediate MoE capture coverage

`intermediate_capture.py prepare` accepts an optional explicit
`--expected-moe-layers` inventory. When supplied, finalization requires the
captured router/routed-FFN stage set to cover that exact layer list; missing or
extra layers prevent an index from being written. Without it, the capture remains
a diagnostic subset and reports `all_expected_moe_layers_covered` false. A
comparison reports complete declared-inventory coverage only when both placement
captures share the inventory and their retained run bundles bind the exact
prepared-manifest hash and layer list. The tool covers the named MoE stages only;
this flag does not claim attention, shared-FFN, residual or full graph coverage.

## Quality and full-vocabulary data

[quality_tasks.py](../scripts/quality_tasks.py) creates the original synthetic
20-task calibration / 100-task holdout manifest. The public
[recipe](../research/fixtures/quality-tasks-v1.json) freezes IDs, seeds, category
counts, license and canonical split hashes. Score visible content only, preserving
reasoning separately. The scorer requires exactly the requested JSON types/keys;
it rejects duplicate keys, markdown wrappers, nonfinite values and extra text.
Do not extract a convenient final JSON object from an otherwise invalid answer.
Shared templates limit generality; these tasks do not replace held-out prose/code
NLL or establish model-pretraining exclusion.

Paired task accuracy diagnostics retain wins/losses and use conservative
Clopper-Pearson discordance bounds, with explicit independence assumptions. Even
zero observed regressions over 100 tasks yields a conservative upper bound above
the proposed 3-point limit. Additional independent tasks/cluster-aware review may
be needed; do not replace an inconclusive bound with a zero bootstrap interval.

[fidelity_corpus.py](../scripts/fidelity_corpus.py) selects and replays whole
prose/code documents from supplied local source records. It never executes code.
Scored positions exclude token index zero; KV resets between documents. Exact
text/code dedup and source/holdout splits precede token-quota selection. A shortage
or unreachable context limit fails instead of padding/repeating/truncating input.

[logit_compare.py](../scripts/logit_compare.py) defines the private
`rosevine-logits-v4` shard index in its module documentation. V4 binds the exact
native execution policy and its digest alongside the position inventory. V2/v3
remain readable as legacy records but do not prove that policy. Each v4 index binds
its ordered position labels to a frozen plan hash and expected count. Both row streams must
reach validated EOF; a last-shard hash mismatch fails. The comparison reports
position identity/coverage and errors/KL/top-1, with no acceptance assertion.
For evidence-bound comparisons, supply both `--reference-run-bundle` and
`--candidate-run-bundle`. The analyzer verifies each private bundle, the capture
index/result binding, successful child cleanup, identity agreement and distinct
child processes. Without bundles, it still reports data comparison but marks
`run_bundle_provenance_verified` false.
The owned regular index must be private, at most 16 MiB, with unique JSON keys.
Version 4 requires equal `inputs_sha256` and explicit equal `logit_stage`
(`pre-softcap` or `post-softcap`) before reading rows. Its supervisor identity
must also bind the input digest, plan hash, expected position-label digest/count
and native policy digest; preparation stores these as
`supervisor_run_identity_fields` so callers can add the independently resolved
`identity_sha256` without rebuilding position hashes. The shared input digest
must bind the model and tokenizer identities, full ordered input IDs and sequence
labels, context/position/reset policy, and vocabulary order. Keep that canonical
manifest private alongside the capture and verify its digest independently.
Backend/build identities remain separate so different implementations can be
compared. Matching supplied digests alone cannot prove capture provenance.
Legacy v2/v3 indexes remain usable for numerical comparison and can retain bundle
binding, but the summary sets `position_inventory_verified` and
`minimum_4096_planned_positions_met` false because v2 did not bind the plan or
expected label inventory. For v3/v4, the inventory-integrity field reflects its
internal digest/count check; `position_inventory_verified` additionally requires
verified run bundles binding the plan and label digest to the supervisor manifest.
v1 indexes are rejected; do not fabricate missing provenance to upgrade an
existing capture.
See the [capture plan](reference-correctness-plan.md) for operation coverage,
limits, position budget and the distinction between planned and captured oracles.

The offline [capture adapter](../scripts/logit_capture.py) can prepare the native
helper input from the calibration position plan, or finalize terminated native
outputs. Preparation freezes a canonical `rosevine-logit-inputs-v1` manifest
(model/tokenizer identity, vocabulary, context capacity, reset/position policy and
ordered sequence labels/IDs). Its canonical digest is the shared `inputs_sha256`.
The current `RVLOG003` binary starts with eight magic bytes, then little-endian
uint32 vocabulary, capacity, sequence count and policy flags; bit 0 selects
context rows and bit 1 sets `extra_buffers=true`. Other bits are rejected.
The policy flag is checked against the prepared record, whose complete native
policy is itself digest-bound; the same policy and digest are stored in the v4 index.
`RVLOG001` and `RVLOG002` are still accepted by the native helper as legacy
formats and default `extra_buffers` to false. Each sequence is a uint32 ID count
followed by that many uint32 token IDs. Sequence labels live in the manifest;
the native helper preserves manifest sequence order. Strict validation excludes
trailing data, invalid IDs, unsupported dimensions and nonprivate inputs.
For bounded context-boundary work, the legacy `RVLOG002` keeps its three-field
header and token
encoding, then appends to each sequence a uint32 selected-row count and sorted
uint32 input positions. The native helper still evaluates every causal input
position through N-2, but requests/writes logits only at those selected rows.
Selected positions must be unique, in range and declared for every sequence;
the full-vocabulary output is capped at 96 rows (96 MiB at vocabulary 262144).
This leaves the sequence context intact while bounding retained logits. It does
not bound model compute or replace the external time/resource guard.
The native output directory contains a `started` marker, numbered
`logits-000000.f32` shards of at most 32 rows and a `complete` marker containing
the total row count. A receipt is not supervisor success or provenance proof.
Finalization checks both, verifies every shard and writes create-only `index.json`;
it retains failed inputs/outputs and never launches a model. Supervisor logs
live separately under the same monitored campaign. A successful finalization
also writes `capture-run-binding.json`, hashing the finalized index and exact
supervisor result bytes and recording digests of the child process identity and
executable. Repeated-reference analysis requires the corresponding retained run
bundles, verifies each bundle and binding, and requires distinct child identities
and one executable hash. This does not establish separate sessions or thermal
states, nor is the local checksum chain an external authenticity anchor. See the capture plan for
fixed native policy, untested execution paths and outstanding run requirements.

[native_completion.py](../scripts/native_completion.py) validates the executed
verbose-prompt IDs against the frozen token stream and reads named native host
phase times with explicit units. Completion's `--file` removes one trailing LF;
use its supported `--binary-file` for exact-byte replays. A native single-row
evaluation counter is not a committed-output count. Missing phase/delivery
measurements remain null with reasons; native console decode rate is not silently
used as target decode rate, TTFT or steady output rate. A prefill rate is derived
only when the native phase covers the complete verified multi-token prompt.

[`native_mtp_metrics.py`](../scripts/native_mtp_metrics.py) reduces an explicit
`rosevine-native-mtp-events-v1` record. Each verification event must include
unique identity, proposed/accepted prefix counts, committed-token count, target
evaluation rows, rollback operations and monotonic draft/verify/correction
intervals. Intervals must not overlap before their durations are summed. The
reducer computes per-position acceptance from opportunities and accepted-prefix
counts and the aggregate accepted/proposed fraction; it does not infer events
from output text. The shared metric inventory names that aggregate
`mtp_acceptance_rate` with unit `fraction`. This is an adapter contract
only: no pinned native MTP event producer is currently connected or validated,
so actual MTP metrics remain unavailable until that producer exists and is
checked against a known workload.


## NLL captures and direct-token position plans

[nll_compare.py](../scripts/nll_compare.py) accepts corpus, reference and candidate
private JSON files, plus output path, split, seed and resample count. Capture schema
`rosevine-nll-v1` requires successful status, corpus/identity SHA-256, explicit split,
pre/post-softcap logit stage and documents `{id, tokens_sha256, nll}`. Each `nll`
array contains finite nonnegative nats in exact target order for every token after
context token zero. Both captures must include every selected corpus document.
No subset comparison, missing target, changed token stream or mixed softcap stage
is accepted. Analysis can complete on a small synthetic fixture while the explicit
100k holdout-coverage flag stays false; complete computation is not model acceptance.

The bootstrap resamples whole paired documents within each stratum and preserves
the frozen corpus token-weighted stratum mixture, with declared seed/resamples.
The fixed corpus and its two domains are reviewed for narrow diagnostic use only;
they do not support broad quality claims. Tokens are not IID resampling units.
[nll_capture.py](../scripts/nll_capture.py) prepares `RVNLL001`
inputs from a private corpus split and finalizes only a complete supervised native
capture. The isolated native helper calculates stable long-double full-vocabulary
log-sum-exp for target `j+1` after each teacher-forced input row and writes one
little-endian FP64 scalar in document/target order. The capture schema records the
post-softcap stage, exact token hashes and caller-supplied run identity digest.
Finalization also writes a private `capture-run-binding.json` linking the exact
capture bytes, supervisor-result digest, child process instance and executable.
For bundle-attested analysis, pass both `--reference-run-bundle` and
`--candidate-run-bundle` to `nll_compare.py`; each verified bundle must carry the
matching identity digest, plan/input/corpus hashes, split, target count and logit
stage in `run_identity`, and must use distinct bundles and child processes. This
check verifies internal checksums and run binding; it
does not independently attest that the manifest truthfully describes the invoked
model or establish runtime-loaded library identity.
Programmatic `compare()` results explicitly set
`run_bundle_provenance_verified` false and `run_bundle_provenance` null because
that API receives capture objects without paths to verify; the CLI populates them
only when both retained bundles validate.
Four retained full-logit rows cross-check the arithmetic against Python; the
native mode has also passed a pinned-header API-double test. Full corpus execution,
real loaded-library/build provenance audit, reference variability and quality
acceptance remain outstanding. Perplexity overflow is reported as null plus reason
while finite NLL remains available.

[reference_positions.py](../scripts/reference_positions.py) creates a separate
`rosevine-reference-positions-v1` plan from calibration documents only, an explicit
BOS ID and vocabulary/capacity bounds. It stores direct token streams, source pool
spans, scored input/target indices, reset policy, hashes and exact FP32 logit-byte
budgets. Circular windows may overlap; this is graph/logit coverage input, not
natural-document NLL, independent task samples or a rendered chat prompt. Rendered
bytes are null because this numerical graph-coverage protocol starts from frozen
token IDs. This reviewed scope is limited to logits; it does not validate
rendering/tokenization or replace rendered workload evidence. The calibration
windows may overlap and are not independent quality samples. Planned position
counts must never be reported as captured or checked logits.
Planned position counts must never be reported as captured or checked logits.
