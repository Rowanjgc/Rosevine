# Decode critical-path attribution (R3a.2)

Bead: rv-e05-7n3.6. Baseline: `ea6c4e4888ec90d655773aad6df88939300563d4`
(R3a.1), unchanged execution code.
Status: attribution complete with the limitations below; no optimization adopted.

## Reproduction and boundaries

Use the [existing run policy](development/rule-audit.md),
[experiment protocol](experiment-protocol.md) and
[instrumentation boundary definitions](decode-instrumentation.md).
Raw captures, token fixtures, model identity, executable hashes, helper source snapshots,
session manifests, telemetry and reductions belong in the private campaign directory.
The report contains manually selected aggregates; it is not an approved benchmark export.

Build the baseline helpers without registry access:

```sh
cargo build --manifest-path runtime/Cargo.toml --offline --locked --release --example cpu_attribution
cargo build --manifest-path runtime/Cargo.toml --offline --locked --release --features hip --example hip_attribution
hipcc --offload-arch=gfx1101 -O3 -Wall -Werror scripts/research/upload_probe.hip -o PRIVATE_PROBE
```

These executables are workloads, **not safety supervisors**. Invoke them using
`scripts/gpu_campaign.py` command sessions inside the existing COMPUTE power window,
holding the shared private GPU lock before entering that window. Never run them
alongside another GPU benchmark or an active production model. Retain failures;
confirm cleanup before starting another session. Do not modify production configuration.

- CPU workload: `cpu_attribution MODEL`. Uses the production scalar
  `matmul_q4_0_multi` and `par_rows`, exact target dimensions (2816, 704, eight experts),
  layer-0 experts 0–7 and deterministic F32 sinusoidal activations. Gate/up,
  serial GELU, down and slot-order combine are timed separately. This is a proxy
  for the routed runtime workload, with distinct activation values and expert selection.
  It does not replace the in-situ spans.
- Two warmups and 40 measured samples per CPU condition/thread count; 200 empty
  spawn/join samples after two warmups. Warm means repeated access to the same
  26.8 MB of weights. Evicted means a process-local 128 MiB write sweep before
  each sample, outside its timer. This is a cache-eviction intervention, not a
  claim of cold filesystem pages or guaranteed eviction of every cache line.
- Contention uses one additional thread streaming writes over 64 MiB, with a
  stop check at each 64 KiB chunk and a joined shutdown. It perturbs CPU scheduling,
  memory traffic and frequency together; it does not isolate DRAM bandwidth.
- Upload workload: `PRIVATE_PROBE [--no-events]`. Copies 11,264 bytes and verifies
  every result byte. Two warmups and 40 samples per cell; pageable/pinned and
  blocking/async variants are interleaved, reversing order on alternating repetitions.
  GPU-idle intervals are 0, 0.5, 5 and 20 ms. Each starts after a synchronized marker.
  A second control launches and synchronizes a marker after the idle interval.
  Account for that marker's time when interpreting the faster following copy.
- Enqueue time wraps only the copy API. Completion wait and total host duration
  are separate. Event intervals include scheduling and command boundaries; they
  are **not isolated DMA-engine occupancy**. Events can perturb wake behavior,
  hence the event-free control. Async enqueue time alone is not completion time.
- In-situ workload: `hip_attribution MODEL TOKEN_FILE FIRST_N THREADS ORDER`.
  Uses the frozen 1,900-token prose-continuation fixture, batch limit 64 and
  32-token greedy continuations; context capacity is 1,965. One uninstrumented
  warmup continuation precedes off/host/host/off/events/sync; the second session
  reverses this order. Each pass rewinds 32 positions using the existing bounded
  KV retention and replays identical work. Every logit is compared bitwise with
  the warmup. No MTP, routing, placement, combine or activation changes.
- Per-token wall samples wrap `eval`; sampling, full-logit comparison and copying
  span records occur outside that timer. The instrumentation's `outside` span
  records caller work. Cumulative span/counter records are differenced by
  `scripts/research/analyze_attribution.py`; its output also stays private.
- Two sessions per cell; 6 and 12 workers with affinity allowing all 12 logical
  CPUs (six physical cores with SMT). Six workers are not pinned one per core.
  Preserve the exact affinity and topology in the private manifest. Samples
  within a session are correlated; do not treat them as independent sessions.
- The initial session timeout is 240 seconds, with 2 GiB RAM/disk and 1 GiB VRAM reserves,
  100 ms safety collection, sustained thermal stops and production-server growth
  protection. Individual command and telemetry captures are capped at 20 MiB.
  Clock polling is disabled to avoid additional sensor queries during attribution.
  Record thermal flags and system-wide paging warnings, even when reserves remain healthy.
  The N=20/six-worker attempt timed out during prefill and was reaped; it is retained
  as incomplete. A separately named follow-up uses 360 seconds based on the
  observed prefill cost, with unchanged resource and thermal stops. This duration
  revision does not turn the original timeout into a successful experiment.

The actual artifact hash equals the frozen baseline. Its filename says Q4_K_XL,
but the runtime's tensor/shape/type contract validates it; names are not dtype evidence.
Rust release builds use the repository defaults; HIP uses gfx1101, `-O3`, and the
runtime's `-ffp-contract=off`. No compiler, dependency or model was installed.

The host is the Ryzen 5 7600X / RX 7800 XT (16 GiB) target, running Fedora 43,
Rust 1.98.1 and HIP 6.4.43484 (Clang 19).
Exact compiler versions, source/binary/library digests and model file identity
are pinned in the private manifests. The model SHA-256 is
`a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`.

## CPU controls

Initial session medians, milliseconds per eight-expert layer:

| Workers | Warm | Cache-evicted | One streaming competitor | Empty spawn/join, one call |
| --- | --- | --- | --- | --- |
| 6 | 7.216 / 7.318 | 7.496 / 7.256 | 8.864 / 8.807 | 0.068 / 0.068 |
| 12 | 4.748 / 4.810 | 5.096 / 5.075 | 6.939 / 7.072 | 0.119 / 0.120 |

Slash-separated values are the two sessions, not confidence bounds. The CPU
expert path makes two matrix-dispatch calls per layer. Empty-call wall time
includes creation, scheduling and joining; it does not measure kernel execution
or establish the removable cost under load. Both the isolated warm benchmark
and the in-situ path already pay this cost. It cannot by itself explain their difference.

The corresponding p95 values (milliseconds, session 1 / 2) are:

| Workers | Warm | Cache-evicted | Streaming competitor | Empty dispatch |
| --- | --- | --- | --- | --- |
| 6 | 9.244 / 8.188 | 8.779 / 8.071 | 9.652 / 9.725 | 0.080 / 0.082 |
| 12 | 5.703 / 6.717 | 6.719 / 7.025 | 8.228 / 8.639 | 0.163 / 0.160 |

This tail variability limits small-difference claims. Condition blocks use
warm/evicted/contended order in both sessions; frequency and thermal drift are
not randomized away. A repeat with an unwind-safe competitor shutdown guard
checks the final helper's normal path; the initial helper snapshots are retained.

At 12 workers the cache intervention raises the median by about 0.26–0.35 ms.
The 6-worker result does not reproduce that direction in both sessions. The
streaming competitor makes both worker counts slower. The experiment establishes
contention sensitivity, not which resource causes it.

## Upload controls

Event-enabled, blocking API medians in microseconds:

| Idle | Pageable, session 1 / 2 | Pinned, session 1 / 2 |
| --- | --- | --- |
| 0 ms | 30.6 / 27.5 | 28.1 / 27.6 |
| 0.5 ms | 210.5 / 215.9 | 208.7 / 213.1 |
| 5 ms | 470.3 / 471.9 | 466.6 / 469.1 |
| 20 ms | 475.8 / 476.0 | 473.3 / 517.6 |

Blocking API p95 values in microseconds, in the same session order:

| Idle | Pageable | Pinned |
| --- | --- | --- |
| 0 ms | 31.2 / 28.1 | 32.2 / 28.3 |
| 0.5 ms | 212.0 / 217.1 | 212.1 / 216.2 |
| 5 ms | 520.4 / 488.2 | 520.8 / 512.3 |
| 20 ms | 587.3 / 582.1 | 534.9 / 572.6 |

After 5 ms idle, a synchronized pre-copy marker takes about 0.45–0.47 ms itself;
the following blocking copy falls to about 0.03 ms with either staging type.
This moves the cost to the marker; it is not a demonstrated speedup.

At 5 ms idle, pinned async enqueue takes about 4 microseconds, but completion
wait takes about 0.46 ms. The event interval is about 13 microseconds. Blocking
copy event intervals are about 30–33 microseconds. These boundaries strongly
separate short enqueue/device intervals from long host-visible completion.
They do not identify a particular firmware power state or a driver-internal mechanism.

The event-free reproductions give 462 / 467 microseconds for pageable blocking
enqueue and 457 / 460 microseconds for pinned blocking enqueue after 5 ms idle.
Pinned async enqueue is 5.6 microseconds, followed by 452 microseconds of completion
wait. The extra blocking completion sync is about one microsecond. Thus timing
events are not necessary for the large idle-associated delay to appear.

## In-situ attribution and closure

The completed 12-worker cells give the following mean decode-loop times,
milliseconds/token (caller work included):

| N | Session | Off | Host | Events | Sync |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 83.331 | 83.276 | 83.414 | 85.349 |
| 10 | 2 | 82.766 | 82.951 | 83.465 | 84.594 |
| 20 | 1 | 136.509 | 138.571 | 138.729 | 138.789 |
| 20 | 2 | 137.549 | 136.594 | 138.414 | 141.711 |

Host-mode differences change sign between sessions. They do not demonstrate an
execution improvement or regression. Events and especially synchronization
perturb the measured path; use host mode for the ordinary serial path and the
other modes to interpret asynchronous work. The earlier approximately 71.6 ms
baseline uses a different short prompt. These 1,900-token results are not a
regression comparison against it.

The following pools 128 measured evaluations per mode/cell across the two sessions
(two measured off passes and two host passes per session). These are correlated
token samples, not 128 independent sessions. Units are milliseconds/evaluation;
sampling/comparison work is outside this timer.

| N | Workers | Off median / p95 | Host median / p95 |
| --- | --- | --- | --- |
| 10 | 12 | 81.038 / 92.815 | 81.565 / 88.991 |
| 20 | 12 | 134.529 / 149.433 | 134.831 / 152.482 |
| 10 | 6 | 99.677 / 110.316 | 99.932 / 107.761 |
| 20 | 6 | 170.662 / 182.810 | 170.765 / 180.969 |

The ordinary host-mode breakdown, averaging both passes in both sessions:

| N / workers | Total including caller, ms/token | CPU experts, ms (%) | Expert upload, ms (%) | Router barrier, ms (%) |
| --- | --- | --- | --- | --- |
| 10 / 12 | 83.113 | 51.840 (62.37%) | 4.058 (4.88%) | 22.307 (26.84%) |
| 20 / 12 | 137.583 | 104.376 (75.86%) | 7.899 (5.74%) | 20.870 (15.17%) |
| 10 / 6 | 100.983 | 69.477 (68.80%) | 4.259 (4.22%) | 22.331 (22.11%) |
| 20 / 6 | 172.339 | 138.586 (80.41%) | 8.451 (4.90%) | 20.883 (12.12%) |

For N=10/12, the remaining named host spans are expert-input handoff 0.495 ms
(0.60%), GPU-expert metadata 1.690 ms (2.03%), final logits barrier 1.067 ms
(1.28%), other submissions/host selection 1.368 ms (1.65%), and caller work
0.288 ms (0.35%). Rounding accounts for small table-sum differences.
The router barrier includes earlier queued GPU work; it is not 22 ms of router math.
Sync mode places attention at about 16.44 ms/token, shared MLP at 2.26 ms,
GPU experts at 3.37 ms, router at 2.05 ms, and output head at 0.81 ms for N=10/12.
Those are instrumented boundary costs, not additional costs to add to host mode.

**Closure and correctness.** Minimum measured decode-loop closure is 0.999878;
span closure is 1.0000. All eight successful placement/thread/session cells pass
full-logit bitwise equality across the warmup and all six measured passes. The
timeout is excluded. No arithmetic or execution-path code changed, and the
normal unit/integration suites pass. Independent-reference KL was not rerun for
this diagnostic-only change; it remains required for R3a.3's arithmetic change.

The hand-count formulas per evaluated token are:

- Launches: `1319 - 19*N`; H2D calls: `121 - 3*N`; D2H calls: `31 + N`; D2D: 5.
- H2D bytes: `4 + 11264*N + 128*(30-N)`.
- D2H bytes: `1048576 + 30*512 + 11264*N`.

The reducer rejects a mismatch with these formulas, a counter discontinuity,
incomplete token samples or different greedy continuations. Its unit tests cover
per-token differencing and rejection of partial/mismatched evidence.

## Residual and remaining uncertainty

At 12 workers, changing N from 10 to 20 adds 54.470 ms/token, or 5.447 ms per
additional host layer. The CPU-expert span accounts for 5.254 ms/layer of this
increment and output upload for 0.384 ms/layer. Other named spans have a net
change of minus 0.191 ms/layer, including avoided GPU expert work and metadata.
Against the original warm isolated control's mean-of-session medians (4.779 ms),
the approximately 0.668 ms residual is therefore 0.475 ms of CPU-span excess plus
0.384 ms upload minus 0.191 ms elsewhere. This is a measured accounting identity,
not a causal sum of independent optimizations.

Cache-evicted isolated CPU medians are about 5.085 ms, leaving approximately
0.17 ms/layer between that proxy and the CPU-span placement increment. This
remaining average mismatch is not a confidence bound. Input/selected-expert
differences, cache/frequency/scheduling effects, per-thread row allocation and
host combine bookkeeping remain unresolved within the named CPU span. Empty
dispatch is already present in both paths. The broad tails and coupled controls
do not support declaring one of these the root cause. Six-worker in-situ CPU
means are about 6.93–6.95 ms/layer, below the isolated warm medians; there is no
universal fixed CPU-runtime surcharge established by these measurements.

The executor's stream has no useful queued work during CPU experts after the
expert-input download completes. Its idle interval and the upload's large
host-visible completion delay are exposed on the critical path. The controlled
idle and marker interventions reproduce that delay without a preceding useful
GPU workload. They support an inactivity-associated readiness/completion cost;
the division among firmware wake, runtime command processing and synchronization
remains unexplained. A pinned allocation and a fast async enqueue do not remove it.

## Measurement quality and deviations

- The original N=20/six-worker attempt is incomplete (240-second timeout,
  successful cleanup). The separately named 360-second follow-up completes in
  about 287 seconds; its second-session reproduction completes in about 289 seconds.
- Observed minimum MemAvailable is above 21 GiB; peak total device use is about
  11.67 GiB. No reserve, sustained pressure, thermal or router-growth stop fired.
  Small system-wide paging events are retained; attribution to a process is unknown.
- All GPU temperatures remain below their quality flags. CPU CCD excursions
  above the 95 C quality flag occur in seven of the eight successful in-situ
  sessions. Both N=20/12 sessions also flag Tctl. Their peaks are 99.375 C CCD
  and 96.125 C Tctl, below the sustained hard stops. These flags include prefill;
  no per-evaluation absolute timestamps were recorded to filter individual tokens.
- Conservatively excluding every thermally flagged session leaves only the
  first N=20/six-worker follow-up: off median 171.297 ms, host median 171.425 ms,
  versus pooled medians 170.662 / 170.765 ms with all sessions. The other cells
  have **no entirely unflagged session** under that conservative filter. Do not
  present an unflagged two-session performance result for them.
- Offline reductions ran intermittently during the in-situ campaign. Their
  possible CPU/memory contention was not timestamped per evaluation. This limits
  small timing comparisons and tail interpretation. Subsequent performance
  comparisons must serialize analysis after timed workloads finish.
- One supplemental CPU process-resource control overlapped an offline reduction
  and is explicitly marked contaminated, retained and excluded. Its standalone
  replacement and the other clean supplemental control report zero major faults,
  about 70,500 minor faults and approximately 269 MiB peak RSS for the entire
  process, including buffer initialization, eviction and the competitor.
  Their 12-worker warm medians are 4.858 / 4.863 ms, evicted 5.075 / 5.231 ms,
  and contended 7.023 / 7.077 ms. These corroborate the initial controls;
  they do not identify faults inside an individual timed kernel invocation.
- The analysis retains flags and failures rather than silently selecting the
  fastest clean-looking samples. No performance improvement is claimed by this bead.

## Optimization proposals and failure thresholds

These are ranked proposals, not integrated results. Estimates are planning bounds
for N=10 at this context and cannot be added as independent savings.

| Order | Proposal | Expected opportunity | Failure threshold / decision |
| --- | --- | --- | --- |
| 1 | R3a.3 persistent workers plus SIMD Q4_0 with F32 activations | About 52 ms/token of CPU expert work is exposed. Reaching the existing 1.5 ms/layer objective would save roughly 37 ms/token before secondary effects. Removing empty dispatch alone has a much smaller envelope, about 2.4 ms/token. | Keep the existing numerical requirements and 1.5 ms objective. Report an objective miss and its cause; reject numerical failures. Require matched end-to-end measurements with uncertainty before claiming a gain. |
| 2 | Profile/tune the existing GPU attention and GEMV kernels under their own beads | Sync attribution at 1,900 tokens places about 16.4 ms/token in attention; this is an upper bound, not an expected full saving. Shared MLP, GPU experts and the output head add smaller opportunities. | A kernel must preserve its correctness class and improve a matched metric beyond uncertainty. Re-run attribution after tuning; do not extrapolate this context to 29k. |
| 3 | Re-measure upload after CPU acceleration; consider bounded scheduling overlap in a separate proposal | Current idle-associated host excess is roughly 0.35–0.45 ms per host layer, an opportunity of about 3.5–4.5 ms/token if it can actually be hidden. Faster CPU work changes the idle interval and this estimate. | Reject a purported gain that only moves the wait to a marker/sync. Reject a control with no positive paired end-to-end interval, numerical failure or unacceptable responsiveness/power cost. No persistent artificial GPU workload is proposed. |
| 4 | Pinned staging alone | Idle and warm controls suggest only a few microseconds per copy, around 0.01–0.05 ms/token at N=10. | Deprioritize unless a matched runtime experiment reproduces a gain beyond noise. Pinning alone has not removed the roughly 0.4 ms delay. |

The existing executor serializes expert-input download, CPU expert computation,
expert-output upload and its dependent combine. These costs are on the current
critical path. Enqueue work can overlap later host submissions, but the GPU cannot
combine expert output before the data arrives. An asynchronous API does not
remove this dependency. Existing attention/shared-MLP work has already passed the
router-readback barrier before CPU experts start. Moving independent work into
that interval would be a scheduling change requiring separate validation.

The CPU cache intervention explains neither a removable scheduling cost nor an
implementable residency scheme. Dynamic caching, LES, placement redesign,
speculation and major attention rewrites remain deferred. No optimization is
implemented by this attribution bead.

## Limitations

- The 29,000-token cells are unrun: 1,900-token N=10 prefill alone takes about
  71 seconds. Extending those runs would exceed this campaign's bounded profile.
  No long-context correctness or performance claim follows from this report.
- No GPU hardware profiler was used. Event intervals cannot determine pure bus
  service time or distinguish firmware wake from runtime command processing.
- CPU cache state, frequency and scheduling are coupled. The eviction control
  is evidence for a hypothesis, not a full attribution of the in-situ CPU gap.
- The original CPU/upload control sessions lack per-session CPU-frequency
  snapshots; in-situ sessions record them. The safety collector records memory,
  paging and temperatures throughout. No missing reading is filled with a guess.
- The GPU operators are untuned: warp-per-row Q4 GEMV, simple F32 GEMV,
  untiled attention, separate normalization/activation/combine kernels and a
  tied Q4 output head. Attribution changes when these change. No comparison
  against `vk-ref` or claim of parity is made.
