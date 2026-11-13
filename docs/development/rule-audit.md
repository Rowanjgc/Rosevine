# Proportionate research rules and audit

Updated 2026-11-12 at the maintainer's request to remove disproportionate
development friction, then retry the bounded CPU diagnostic. This is a revision
of research execution policy, not a waiver of privacy, publication or correctness
requirements. Earlier run records retain the rules actually used at the time.

## Audit coverage and decisions

Reviewed the agent/contribution instructions, privacy/security and development
policies, Beads policy, charter, roadmap/ADR gates, environment/pilot/gate records,
experiment/correctness/quality protocols and executable repository/resource/
collector checks. The audit covers the current documented rules and implemented
guards; it is not a guarantee against future false positives or hardware faults.

| Area | Assessment and current rule |
| --- | --- |
| System-wide swap counters | Any increment was too sensitive and had no process attribution. Record deltas; stop on sustained significant paging with actual low memory headroom, or an immediate hard RAM-reserve breach. |
| Counter reset or sample gap | Rebase paging trend and warn. Do not interpret the discontinuity as measured pressure or fabricate continuity. Missing required current safety readings still stop. |
| Collector cost | A 1 ms p95 host target and a 95% paired end-to-end time interval strictly within +/-1% are benchmark-quality criteria. Record diagnostic overhead, including the earlier 3 ms target, without treating small overruns as hardware failures. A separate bounded response deadline protects liveness. |
| Collector startup | Interpreter/import startup occurs before model launch. Give startup its own bounded allowance; never extend running response deadlines to hide a stuck sensor. |
| Optional counters and device health | Missing PSS, power, PCIe/reset telemetry stays null with reason. Require only the safety signals applicable to the selected run. CPU-only diagnostics may explicitly allow unknown GPU compute health; known device failure still stops. GPU trials require their applicable validated safety profile. |
| Thermal and resource reserves | Keep mapped, hardware-informed thermal stops and hard reserves. The initial 80 C CPU cutoff stopped an 81.125 C CCD observation on a CPU rated to 95 C; correct this procedural stop to 90 C with 5 C margin. Preserve the first attempt, GPU limits and RAM/display/disk reserves. Do not repeatedly raise limits to force success. |
| Background activity | Record it. Desktop GPU activity alone does not cancel a CPU allocation diagnostic. It can invalidate a controlled performance comparison; unknown contention is not proof of an idle benchmark environment. |
| Identity and provenance | Pin actual local artifacts. Hash large artifacts at session entry or after identity-affecting changes, outside timing; retain stable file metadata if reusing a session hash. Unknown upstream lineage/source-build equivalence limits those claims, not every diagnostic of a named local artifact. |
| Approval and ADRs | Reuse authorization for the approved scope, including routine code/tests and bounded follow-ups. Recheck conditions at execution, not by inventing repeated permission gates or expiring approval during code preparation. Production architecture adoption, larger resource scope, external writes/publication and other explicit restrictions still need their applicable authorization. Disposable research tools do not adopt a production language/backend. |
| Scientific gates and sample counts | Keep exact workload alignment, honest missing metrics, uncertainty and reviewed correctness/performance criteria for claims. Two warmups/seven pairs, 4096 logit positions and quality-corpus counts do not apply to a one-request loading/cleanup smoke test. Numerical failures remain failures; no candidate-driven tolerance relaxation. |
| Campaign and capture bounds | Keep explicit per-run/campaign byte limits, free-space reserve and failure retention. A campaign cap applies to its named campaign, not every past/future result in the project. Do not rotate campaign names merely to evade a bound. |
| Repository/security/publication checks | Keep secret/privacy scans, private file modes, immutable dependency pins, narrow public schemas, no production/global edits and explicit remote/publication authorization. False positives require investigation and the existing exact-content review process; do not disable scanners or add broad ignores. |
| Workflow and phase history | Use targeted tests while iterating and the full check before handoff. Historical Project Zero/planning restrictions do not override later direct implementation authorization. Tracker adoption is complete; routine updates do not need repeat approval. |

## Current memory policy

`PagingMonitor` in [benchmark_resources.py](../../scripts/benchmark_resources.py)
implements the v2 diagnostic pressure rule. Record every observed swap-in/out
delta as system-wide evidence with unknown attribution. Warn on small activity.
Cancel when combined paging is at least 16 MiB/s in consecutive observation
windows of at least one second, totaling at least three seconds, while
MemAvailable stays at or below twice the hard RAM reserve throughout those
windows. The hard reserve breach still cancels immediately. These explicit
operational choices are not kernel-certified universal thresholds.

A counter reset or observation gap greater than one second resets the pressure
trend and records a warning. Negative/malformed readings are invalid. An isolated
large burst cannot accumulate duration through quiet windows; healthy headroom
breaks the pressure trend. Sustained paging can contaminate benchmark results
even when it does not require terminating a diagnostic. Preserve these warnings
for review before accepting any paired performance comparison.

The kernel describes MemAvailable as an estimate, and pressure telemetry as a
measure of resource contention. Existing swap occupancy or a single restored page
does not establish thrashing. See the primary
[proc documentation](https://docs.kernel.org/filesystems/proc.html) and
[pressure-stall documentation](https://docs.kernel.org/accounting/psi.html).
No swap, VM, power, driver, service or router setting is changed by this policy.

## Run profiles and escalation

For the current CPU smoke profile, preserve the existing two-thread/4096-context,
15-input/two-output, 60 s request and 5 s termination-grace envelope, five
sensor-specific stops, 1 GiB display reserve, 2 GiB disk reserve, 100 MiB run
capture and 10 GiB campaign cap. Require current RAM/swap/thermal/headroom
readings, explicit CPU placement, error/exit reporting and confirmed cleanup.
GPU compute health is reported unknown, not filled with a fictitious healthy flag.
The current thermal profile uses 90 C CPU control/CCD, 75 C GPU edge and 85 C
GPU junction/memory stops. The CPU's verified manufacturer maximum is 95 C;
the exact hardware identity and primary product reference stay in the private
review record. The earlier 80 C stop was an unnecessarily low procedural cutoff,
not evidence that the hardware had reached its rated maximum. GPU values remain
conservative diagnostic settings, not claimed manufacturer maxima.

Use a 1 s collector startup allowance before launch and a 1 s hard running response
deadline for the loaded CPU profile, with the requested 100 ms cadence. The
original 250 ms running limit stopped an otherwise healthy loading diagnostic;
source tracing observed a roughly 646 ms gap between individually submillisecond
reads. The scheduling/reclaim/writeback cause is not established. Per-read trace
writes were removed from the critical worker; one collector-only syscall trace
to a private RAM spool and an untraced exact-input repeat passed under the 1 s
profile. Slow cycles remain visible rather than being treated as timely samples.
The low-level API's default is still 250 ms; loaded CPU runs select this explicit
profile. Both startup and running deadlines are capped by the remaining
request deadline. Record actual acquisition intervals/cadence; scheduling and
filesystem latency are not hard real-time guarantees. A stalled collector or
unconfirmed process cleanup still prevents another trial.

Fresh preflight applies after preparation, immediately before execution. The
user's request to run after an audit authorizes that follow-up; the audit's elapsed
time alone does not revoke it. A changed resource profile, GPU allocation, longer
trial or material environment change requires reassessment. Bound follow-ups,
preserve every attempt and avoid blind retry loops.

Gate 0 continues to limit baseline performance experiments. Diagnostics needed
to resolve it may proceed within their authorized profile. Neither successful
loading nor this policy revision supplies correctness, repeatability, source
equivalence or an optimization GO. Never replace those evidence requirements with
a successful smoke test.

## GPU baseline revision (2026-11-12)

Revised at the maintainer's request after live GPU sessions showed the earlier
stops cancelling routine work that the production router performs daily without
any guardrails. This supersedes the GPU thermal values and display reserve above
for reference-server sessions run by [gpu_campaign.py](../../scripts/gpu_campaign.py).

| Rule | Previous behaviour | Current rule and reason |
| --- | --- | --- |
| GPU/CPU thermal | Any single sample at 75/85 °C (GPU) or 90 °C (CPU) cancelled the run | Hard stop only near firmware limits and only when sustained for 0.5 s: GPU junction 105, memory 103, edge 95 °C (driver critical 110/108/100); CPU Tctl and CCD 100 °C. The CPU stop indicates failed regulation: this Zen 4 CPU boosts until it reaches its 95 °C limit and holds there under sustained multi-core load (observed within 3–5 s at 4 and 6 threads), so 95 °C is its normal loaded operating point and is recorded as a flag. The old values become per-request data-quality flags (CPU flag 95 °C). Single-sample die-sensor spikes (a 91.9 °C and a 95.6 °C CCD reading while Tctl stayed near 72–76 °C) stopped two preset sessions without indicating sustained heat. |
| Thermal data quality | Implicit in the stop | Each measured request records maximum temperatures, sustained time over its flag value and median shader clock. Analyses report results with and without flagged requests instead of discarding them silently. |
| Free-VRAM reserve | Stop when free VRAM fell below 1 GiB, later 256 MiB | Disabled for reference-server sessions. The fast/default/128k presets run in production with about 0.3 GiB free. Allocation failure ends the session in the server, and growth of any foreign `llama-server` process (production router or its sleeping model children) still cancels the session immediately. |
| Collector cycle cost | 1 ms p95 host-cycle target | Informational. A paired end-to-end test measured the collector's effect on decode rate at +0.07% (95% CI −0.14% to +0.13%), which is the quantity the target was meant to protect. |

Unchanged: router protection, RAM reserve and sustained-paging stop, disk reserve,
confirmed process cleanup, one attempt per session with retained failures, and
all privacy, publication and remote-write controls. Throttling or thermal flags
limit the claims made from affected requests; they do not invalidate other data.
