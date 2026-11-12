# First model pilot review packet

Status: the first attempt stopped on a single paging event; after the requested
rule audit, a bounded CPU completion succeeded. All attempts remain retained.
This packet is limited to a
CPU-only allocation and cancellation diagnostic. It does not activate B3 or establish Gate 0 GO.
The current [Gate 0 NO-GO](gate-0-review.md) remains in effect for performance
experiments. No backend/runtime ADR, production change or publication is proposed.

## Concrete candidate

The private `.private/epic2-pilot-review-v1/plan.json` contains the expanded exact
command, explicit child environment, executable/tokenizer hashes, recorded target
hash/size, prompt/token hashes and pending review fields. Native help was captured
privately and confirms every proposed flag is available. Pinned source was checked
for raw-prompt tokenization, explicit CPU selection, load mode and sampler options.
The vocabulary-only tokenizer produced exactly 15 input tokens; no model weights
were evaluated. Rehash the model and verify current tool/library identities before
any approved execution, outside the timed interval.

Sanitized equivalent command (review only; exact paths stay private):

```sh
llama-completion --model "$MODEL" --file "$PROMPT" \
  --offline --no-escape --no-conversation --no-display-prompt \
  --no-context-shift --no-warmup --device none --gpu-layers 0 --fit off \
  --load-mode mmap --lazy-mode off --ctx-size 4096 \
  --batch-size 32 --ubatch-size 32 --threads 2 --threads-batch 2 \
  --flash-attn off --cache-type-k f16 --cache-type-v f16 \
  --n-predict 2 --samplers temperature --temp 0 --seed 20261111
```

This is plain synthetic completion, with native automatic BOS handling, rather
than a rendered chat request. Its purpose is to test controlled model loading,
resource monitoring, cleanup and failure retention. No chat-template parity,
historical preset equivalence, TTFT, throughput or quality claim follows from
success. Eos before two output tokens must remain explicit. No assistant or
projector is loaded; this is not a full-preset comparison.

The proposed child environment contains only fixed PATH, C locale, dumb terminal
and OMP_NUM_THREADS=2. It inherits no LLAMA_ARG overrides or library-path variables.
The supervisor now supports explicit child environments and records their digest;
it does not dump the calling environment. The original model is opened read-only;
no weight derivative, cache flush, service, power or clock change is proposed.

## Original v1 limits (historical)

- One process/request, 15 input tokens, at most two outputs, context 4096.
- 60 s request timeout, SIGTERM then SIGKILL after 5 s workload grace;
  100 ms telemetry cadence. Isolated collection adds a 50 ms collector termination
  grace and up to two 1 s owned-leader reap waits. Include these controlled cleanup
  waits in the final reviewed envelope; scheduling/filesystem latency is not a
  hard real-time guarantee, and pending cleanup stops further trials.
- Initial RAM must cover the complete target file plus 2 GiB reserve and a
  reviewed KV/scratch allowance (proposed 512 MiB). During execution, reject
  MemAvailable below 2 GiB and any increase/reset of swap-in/out counters.
- Preserve at least 1 GiB free device memory for display/other clients even though
  this CPU-only command requests no GPU offload. Do not evict another client.
- At most 100 MiB captured evidence for this pilot; at most 10 GiB cumulative
  campaign evidence. Projected growth includes failed earlier captures. No
  automatic deletion. Periodic accounting is not a filesystem quota.
- CPU sensor identity and temperature ceiling, applicable GPU sensor ceilings,
  display reserve, diagnostic health-monitor coverage and an idle window require
  review. Do not invent a thermal limit from a sensor's critical maximum.

The diagnostic monitor must report its actual coverage: RAM/swap, selected thermal
readings, device headroom, native error/exit status and bounded liveness. Unsupported
GPU reset/power/PCIe counters remain unknown; CPU-only execution cannot calibrate
those counters or establish GPU model support. A generic ResourceGuard with an
unknown device-health reading rejects launch; do not fill that reading with true
to make a pilot pass. A CPU-only diagnostic monitor needs an explicitly reviewed
scope before it can replace that generic guard for this invocation.

## Review needed and next actions

The [environment characterization](environment-characterization.md) explicitly
requires: “No workload may start until a reviewer approves sensor-specific thermal
limits, display reserve, the run window and the concrete invocation.” This is the
reason an implementation request alone has not triggered a model run.

The maintainer must identify the idle window and acceptable temperature limits,
and review the exact installed target/tool identity as a bounded diagnostic
reference despite the documented source-build/upstream-lineage gaps. The proposed
CPU-only monitoring scope and resource envelope above also need review. Review
may approve, revise or reject this packet; it cannot retroactively turn unrun
experiments into evidence. Record the response in the private plan and public
status without copying personal details.

After those inputs, finish selecting/validating the exact sensor adapters, check
capture/collector overhead and preflight resources, and execute only if all
approved constraints hold. A rejection or failed preflight is retained as such.
The pilot is not an authorization for the subsequent intermediate/logit captures,
long-context trials or complete preset matrix; those require their applicable
conditions and run-window review. No positive gate outcome is asserted here.

## Earlier safe-scope follow-up

The maintainer requested continuing within safe spec. A read-only preflight
confirmed the proposed initial RAM and display-headroom checks at that instant,
but GPU activity was nonzero. A single sample does not establish an idle window.
Sensor-specific stopping limits and diagnostic health coverage remain unvalidated;
no specific limits or window were inferred from the broad instruction. No model
run started and no gate was activated.

A 100-sample read-only collector probe completed, but its p95 host-cycle cost
missed the proposed 1 ms bound. Phase attribution identified GPU attribute reads
as the dominant grouped phase; it did not identify an individual sensor or prove
workload-time impact. Both results are retained privately. Before a model pilot,
resolve collector overhead and monitoring coverage in the approved envelope;
do not silently relax the bound or treat successful sensor reads as calibration.


The supervisor now supports an isolated collector process with bounded framed
JSON responses and owned-leader cleanup waits. Synthetic tests verify that a
stalled collector blocks preflight or cancels a running workload, with incomplete
status if cleanup cannot be confirmed. A read-only 100-request process-collector
probe completed but still missed the unchanged host-cost criterion. This improves
failure handling without resolving the remaining thermal, health, overhead,
reference precision or run-window conditions. No model pilot was launched during
those probes.

A further preregistered read-only comparison retained two warmups and 100 measured
samples per method at 100 ms cadence, with randomized method order. It compared
serial acquisition with overlapping the GPU-read group and proc/SMAPS-read group,
using the same retained descriptors, fields, parsers, framed transport and write
boundary. Every source retained acquisition intervals; no stale cache, reduced
cadence or signal omission was introduced. Overlap had a lower observed p95, but
both methods still failed the unchanged 1 ms host-cost bound. Independent replay
verified all 204 request timechains, method assignments, sample counts, calculated
percentiles and confirmed collector cleanup. This is another negative readiness
result, not a model-workload perturbation or accuracy experiment. The failed
parallel acquisition remains a private diagnostic rather than a promoted adapter.
The raw comparison and source/configuration snapshots stay private in
`.private/epic2-collector-overlap-v1`.

## Authorized first attempt

The maintainer subsequently approved the immediate idle window, conservative
sensor-limit preparation, the diagnostic-only 3 ms p95 collector allowance, the
candidate reference, limited CPU monitoring scope and proposed resource envelope.
This approval covers one small diagnostic; the original benchmark requirements,
GPU experiments and subsequent capture invocations retain their own conditions.

The frozen private configuration selected procedural stops of 80 C for both CPU
control/CCD readings, 75 C GPU edge, and 85 C GPU junction and memory. CPU identity,
manufacturer guidance, kernel sensor mapping and exact paths stay in private
review evidence. These are conservative diagnostic stops, not calibrated sensor
accuracy or claimed manufacturer GPU operating maxima. All five readings are
required. The child environment additionally disables the pinned Vulkan backend
within that process. The generic ResourceGuard and unknown GPU health counters
were not reclassified as healthy; a separate explicitly approved CPU diagnostic
monitor checked RAM/swap, thermal readings, display headroom and bounded liveness.

Fresh preflight verified the full model, executable, prompt and candidate-library
identities, RAM/display/disk reserves, no observed paging, and the selected
temperatures. It retained two warmups and 100 collector samples and passed the
approved 3 ms p95 diagnostic limit. Synthetic checks exercised the actual private
monitor decision function at its resource, paging, temperature, time, capture and
window boundaries before launch. The original 1 ms benchmark criterion remains
unmet; this exception does not establish less than 1% workload perturbation.

The exact prepared completion command launched once. The third retained telemetry
sample observed a 4096-byte increase in system-wide swap-in, triggering the
unchanged zero-paging stop. The supervisor sent termination, recorded SIGTERM
exit and reaped the workload and collector; the owned workload PID was absent
when independently checked. No thermal stop occurred in the retained samples.
No native output or timing was captured before cancellation, so even successful
model loading, token evaluation and CPU/GPU placement are not established by
this attempt. System-wide paging counters do not identify its cause.

The attempt is retained as incomplete, with no automatic retry. Private evidence
is in `.private/epic2-authorized-pilot-v1`, including the approval/scope record,
frozen invocation, source snapshots, raw telemetry, checksums and independent
cleanup audit. This failure led to the maintainer-requested
[rule audit](development/rule-audit.md), which replaces the over-sensitive
single-page rule with explicit sustained-pressure detection and authorizes a
bounded follow-up. The original failed attempt remains unchanged.
It provides actual process/monitor cancellation
evidence, not a throughput, full-model correctness or positive gate finding.

## Proportionate-policy follow-up

The maintainer requested an audit of all project rules for disproportionate
development friction and another run. The [audit](development/rule-audit.md)
records the policy coverage, changes and retained protections. Research protocol
v2 records isolated system-wide paging as a warning, rebases counter resets and
requires sustained significant paging with low RAM headroom before cancellation.
Hard reserves remain immediate stops. Collector startup and running deadlines
are separate; optional process telemetry is not allowed to turn an exit race
into a safety failure. The private follow-up monitor uses the shared tested
PagingMonitor, not an untested private replacement rule.

The first v2 attempt progressed through paging observations, then reached an
81.125 C CPU CCD reading against the original 80 C procedural cutoff. It exited
on termination with confirmed cleanup; its failed record is preserved. Verified
manufacturer guidance permits a 95 C maximum for this CPU. The overly conservative
80 C cutoff was corrected to 90 C with a 5 C margin, retaining the GPU limits and
RAM/display/disk reserves. After cooling below 65 C and repeating live preflight,
one further bounded attempt was permitted with no further threshold escalation.

That attempt completed with exit code zero, nonempty native output and confirmed
workload/collector cleanup. All 135 retained telemetry samples passed the v2
safety rules. Small/global paging events were recorded; no sustained low-headroom
paging stop occurred. Raw output, native timing and exact hardware readings stay
private; this establishes a small CPU-requested loading/evaluation/cleanup
diagnostic, not a performance baseline or model-fidelity result.

The native output audit found a reproducibility deviation: completion's `--file`
handler removes one trailing newline, whereas the tokenizer reads file bytes
verbatim. The effective prompt was therefore 14 tokens rather than the planned
15. A separate vocabulary-only replay of the processed bytes verified all 14
IDs and agrees with the native prompt-evaluation count. The original frozen
manifest stays unchanged; a private effective-input record binds the actual bytes
and IDs. Future exact-byte completion invocations should use the supported
`--binary-file` mode or freeze the explicit processing step. This discrepancy
limits the original input-count claim and does not erase successful bounded
execution. The printed sampler seed is inactive for the observed greedy chain;
no stochastic reproducibility claim is made.

The two follow-ups are preserved in `.private/epic2-revised-pilot-v1` and
`.private/epic2-revised-pilot-v2`. Their campaign accounting includes the original
failed CPU attempt. No GPU compute experiment, production setting change,
publication, full-preset comparison or positive gate is implied.

## Exact-input replay and native capture follow-up

The next bounded CPU session switched to supported `--binary-file` mode and
verbose prompt-ID logging. Two successful replays evaluated all 15 frozen IDs
exactly and produced identical generation bytes. This resolves the one-newline
input discrepancy for the new runs without changing the earlier record. Native
host phase times are retained; evaluation counts are not used as committed-token
decode throughput or delivery timestamps.

Initial exact-replay attempts stopped on collector deadlines despite healthy
last readings. Per-source tracing showed a roughly 646 ms interval between
individually submillisecond reads; the cause was not established. A subsequent
per-read file-traced attempt also exceeded 1 s. Per-read regular-file trace I/O
was removed from the critical worker, with one collector-only syscall trace
written to a private RAM spool and copied to durable private storage. That replay
and an untraced repeat passed with a 1 s hard response deadline and nominal 100 ms
cadence. No thermal, RAM, disk or total-request limit was increased. All earlier
deadline failures remain retained, and trace overhead is not benchmark evidence.

The native C API helper also completed two supervised three-ID/two-position
teacher-forced captures under the CPU diagnostic profile. Each produced 2 MiB of
finite full-vocabulary FP32 logits; input/shape/label/hash checks and cleanup passed.
The two native captures were bit identical over these two unique positions. This
establishes the tiny capture path, not full 4096-position, all-layer or quality
coverage. See the [capture plan](reference-correctness-plan.md). The working
session is private in `.private/epic2-exact-replay-v3`; predecessor failed sessions
remain in the v1/v2 exact-replay directories and count toward its campaign budget.

## Additional post-audit replay

After the maintainer asked for one more run following the project-wide rule audit,
a fresh 100-sample CPU preflight passed the approved RAM, disk, display-headroom,
thermal, paging and cleanup checks. Current model, executable and exact-input hashes
matched the approved reference; installed candidate-library files were hash-pinned
before and after the attempt. The one extra replay launched under the same 60-second,
two-thread, 4096-context profile, then reached the request timeout and was terminated
with confirmed workload and collector cleanup. It retained 576 telemetry cycles and
2,865 bytes of output. No automatic retry followed. The incomplete attempt is private
in `.private/epic2-exact-replay-v3/completion-3`; it establishes neither completed
evaluation nor a correctness/performance result. A campaign-accounting setup rejection
and two temporary preflight harness errors in `completion-2` did not launch the model;
those records are retained as setup failures.
