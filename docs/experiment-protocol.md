# Experiment protocol

Version: proposed v1, 2026-11-11. Gate 1 must freeze the workload manifest,
tolerance tables and stopping rules before optimization results are collected.
Changes create a new protocol version and rerun affected comparisons. No new
benchmark was executed during planning.

## Authorization and run envelope

Run offline against an isolated executable and endpoint in an agreed idle
window. Do not evict another workload or edit the router. Before each session,
check available disk/RAM/VRAM, display reserve, activity and swap pressure.
Record the current resource budget and cancellation behavior. Begin with bounded
smoke tests; long-context and sustained runs require a separately approved window.

Proposed initial limits: one request/sequence; 512 output tokens per short trial;
at most 20 minutes for a sustained trial after short trials pass; at least 2 GiB
system MemAvailable and 0.5 GiB device headroom. Validate the headroom policy for
display and allocation spikes before enabling it. The historical 15.54 GiB peak
may violate that proposed device reserve: report that conflict and seek a reviewed
run budget rather than silently lowering the reserve or claiming the preset fits.
Cancel on allocation failure, GPU reset/hang, growing swap-in/out, resource-budget
breach or an agreed thermal limit derived from device documentation. Preserve
partial evidence as failed/incomplete. Never change power limits or clocks for a
trial without specific authorization.

## Immutable run identity

Every run's private manifest records:

- Run ID, protocol/workload version, hypothesis, primary metric, rejection rule,
  planned sample count, software revision/dirty diff, binary and loaded-library
  hashes, build commands/flags and dependency/toolchain versions.
- Model, assistant and projector full SHA-256 and sizes; per-tensor type/shape
  manifest; any derivative's parent hashes, conversion command and output hash.
  Schedule full hashing separately to avoid competing I/O during timing.
- Exact rendered prompt bytes and token IDs; template and tokenizer hashes;
  special tokens, reasoning mode, sampler order/parameters, seed and stop rules.
- Capacity and occupied tokens before/after prefill and decode; local window,
  KV formats/placement/allocation, checkpoints, context shifting, batch/microbatch,
  thread/affinity settings, graph/pipeline caches and expert placement/cache budget.
- MTP on/off, assistant revision, effective draft limits, acceptance policy,
  target verification batch width and KV sharing/rollback settings.
- Hardware class, kernel, driver/runtime/library versions, selected device,
  power policy, telemetry cadence, start/end temperatures and background activity.
- Timing boundaries, warmup/cold-cache state, raw samples, failure reason,
  instrumentation overhead and analysis version. Missing data is null plus reason,
  never zero. Keep exact local paths only in private manifests.

## Workload suite

Create and freeze actual prompt fixtures in the benchmark epic; this plan does
not pretend those fixtures already exist. Use synthetic or redistribution-safe
inputs, pinned licenses and held-out prompts for policy/precision evaluation.

| Suite | Proposed fixed workload | Purpose |
| --- | --- | --- |
| Shape microbench | Descriptor-derived Q4_0/F32 matrices; activation widths 1, 2, 4, 8, 32, 128, 512, 1024 where feasible; local/global attention | Operator attribution; match logical work and dtype; separate JIT/packing from warm execution |
| Short text | 8 prompts covering prose, code, arithmetic, structured instructions; exactly 1900 occupied input tokens after formatting, 512 output tokens | Primary historical-anchor reproduction; eos-limited runs retained separately |
| Medium text | The same categories expanded to exactly 29000 occupied input tokens, 512 output tokens | Historical long-prefix anchor and acceptance dependence |
| Prefill sweep | 512, 1900, 8192, 29000, 65536, 131072, 262144 total input tokens only where output/control reserve fits | Throughput/TTFT scaling; maximum capacity cannot also reserve output beyond its limit |
| Capacity controls | 65536 / 131072 / 262144 configured capacity at fixed 1900 and 29000 occupied tokens | Separate allocation effects from attention work |
| Occupied-context sweep | Within each capacity: 1900, 29000 and near-limit input `capacity - 1024`; reserve 512 generated tokens and special-token/control margin | Practical context, allocation pressure, decode trajectory and context correctness |
| Fidelity corpus | At least 100k held-out teacher-forced tokens across prose/code and 100 fixed scored instruction/reasoning tasks | NLL/distributional and task regression checks; avoid seed-dependent exact-string judgments |
| Context correctness | Retrieval at beginning/middle/end, repeated keys, distractors and continuation across local-window boundaries | Correct positions, global reach, SWA wrapping and capacity/occupancy distinctions |
| MTP fixtures | Forced acceptance lengths 0–3 for state tests, natural greedy and sampled generation, eos/cancellation/resume cases | Rejection correction, verification rows and KV consistency; synthetic acceptance is never a valid speed/quality result |
| Multimodal later | Licensed fixed images across sizes/aspect ratios plus text-only controls | Projector embeddings, image tokens/masks and image-conditioned correctness |

Tokenizer-produced lengths must be validated rather than inferred from character
count. A 262144-token prefill at capacity 262144 is a prefill-only boundary test;
it cannot be compared to a run producing 512 tokens at that same capacity.

First reproduce four complete historical presets as descriptive controls. Then
perform a feasible controlled matrix: capacity × occupied length × first-N CPU
MoE layers (1, 2, 3, 8) × MTP off/max3, holding other variables fixed. Start with
one-factor controls around the fast 64k preset, then selected interactions.
Infeasible/OOM cells are findings. Do not sweep all large cells before short
controls establish safety. Separate cache-enabled runs from historical presets.

Preserve the configured F32 CPU projector in full-preset reproduction and record
whether it is actually resident during text requests. A minimal text runtime
that omits projector loading must also use a separately named, equivalent
text-only llama.cpp control for memory/load comparisons. Retain the full-preset
comparison and explicitly report the missing image capability; do not credit
omitted projector memory as an optimization of equivalent functionality.

## Metric definitions

Use a monotonic host clock for user-visible latency and supported device events
or timestamps for GPU spans. Synchronize at defined boundaries. Do not sum
overlapping spans as elapsed time. Verify counters against a known small workload.

| Metric | Definition and required caveat |
| --- | --- |
| Prefill tokens/s | Newly evaluated prompt tokens divided by synchronized prefill wall time; report tokenization, cached prefixes, allocation and load separately |
| Target-only decode tokens/s | Committed output tokens divided by target-only decode-loop time including sampling/synchronization; identify first-step handling |
| Speculative decode tokens/s | Committed output tokens divided by all draft + verification + correction + scheduling wall time; rejected proposals are not output tokens |
| Steady output rate | For N committed outputs, `(N-1)/(last_token_time-first_token_time)` when N > 1; distinct from whole-loop throughput |
| TTFT | Request admitted to first committed output delivered; report queue, rendering/tokenization, prefill and first decode components; cold-load TTFT separately |
| End-to-end latency | Admission to final committed output/cancellation, including host overhead; report reasoning and visible-token accounting explicitly |
| MTP | Proposed and accepted counts, accepted/proposed, per-position conditional acceptance denominators, mean committed tokens/verification, target evaluated rows, draft/verify/correction time and rollback work |
| VRAM | Total device used at idle, sampled run peak, steady p50/p95 and process allocations if available; report sampling period and instrumented allocation high-water mark separately |
| System RAM | Process RSS/PSS, anonymous/file-backed split and high-water RSS; MemAvailable, page faults and swap-in/out; do not double-count mapped pages and page cache |
| Utilization | GPU busy and CPU core-time normalized with stated convention (one core or whole system); sampled, not a proof of bottleneck |
| PCIe | Measured H2D/D2H bytes and direction/cadence if validated counters exist; application-submitted copy bytes are a labeled proxy, not bus traffic |
| Power / energy | Integrate validated board power over elapsed time; J/committed token and prefill/response energy separately; board-only versus whole-system boundaries explicit |
| Stability | Errors/resets, first-versus-last throughput windows, temperatures/clocks, memory growth and thermal trend over approved sustained trials |

## Sampling and comparisons

Use two warmups per configuration and at least seven measured paired runs per
short workload/configuration, in randomized A/B or balanced ABBA order. Record
separate cold-start trials (at least three) without global page-cache drops on
the shared workstation. Warmup includes model residency, pipeline/graph compilation
and cache fill; report how readiness was established. Long trials may use a
predeclared smaller count with explicit uncertainty and no equivalent-confidence
claim. The paired prompt/seed is the resampling unit; tokens within one run are
not independent samples.

Report median, range, IQR, p95 and 95% paired bootstrap intervals with method and
resample seed. With seven runs, p95 is descriptive and weakly estimated; acquire
at least 30 samples when tail latency drives a decision. If coefficient of
variation exceeds 5% or paired interval crosses zero, investigate contention or
increase the preregistered count before claiming improvement. Keep all runs;
exclude only preregistered invalid conditions and report reasons/counts.

A proposed optimization GO requires at least 5% median benefit in its named
primary metric and a paired 95% improvement interval above zero, plus correctness
and resource acceptance. Smaller benefits remain reported findings. No automatic
GO follows from an isolated microbenchmark: measure affected end-to-end phases
and unchanged controls. Reproduce on a second independent session before Gate 4.

## Correctness and quality gates

Use three references: independent scalar/block dequantization and small FP32
operations; the pinned llama.cpp target-only graph; and its complete decoding
policy. Record shared code so agreement cannot conceal an identical defect.
Numerical envelopes below are proposed starting limits, to be calibrated on
repeated reference runs and reviewed before use. Never loosen them after seeing
a favored optimization fail.

Define normalized RMSE as `sqrt(mean((candidate-reference)^2)) /
max(sqrt(mean(reference^2)), 1e-12)` over the specified tensor/position. Use
float64 analysis reductions. Cosine is the normalized dot product; zero-norm
fixtures use the absolute-error rule instead. Compute full-vocabulary KL as
`sum(p_reference * (log(p_reference)-log(p_candidate)))` in nats using stable
log-softmax, then average over frozen teacher-forced positions. Record the
reference/candidate precision and compare before or after logit softcapping
consistently. Exact block equality is applied before matrix accumulation.

| Layer of verification | Proposed acceptance |
| --- | --- |
| Byte-preserving reader/repack | Tensor ranges/shapes/types validated; inverse conversion recovers every original byte; special token IDs and fixture token streams exactly equal |
| Q4_0 block decode | Exact decoded FP32 values for sampled blocks plus synthetic all-nibble/extreme-scale fixtures; no added quantization error |
| Small FP32 operations | All finite; elementwise `abs(error) <= 1e-5 + 1e-4*abs(reference)`; document accumulation/order effects |
| Mixed/quantized operator outputs | All finite; normalized RMSE <= 1e-3 and cosine >= 0.9999 initially; record max absolute/relative errors and failing shapes; per-op table needed before implementation |
| Intermediate graph | Compare embeddings, Q/K/V before/after transforms, attention, router logits/top-k, shared/routed FFNs and residuals on every layer for short fixtures; near router ties require logit-margin analysis, not arbitrary tolerance |
| Final target logits | Across >=4096 teacher-forced positions: normalized RMSE <= 2e-3, cosine >= 0.9999, mean reference-to-candidate KL <= 1e-4 nats and greedy top-1 agreement >=99.9%; all mismatches investigated |
| Controlled generation | Fixed greedy fixtures reproduce reference token IDs except documented numerical near ties; stochastic variants compare distributions across at least 20 seeds rather than demanding identical random trajectories |
| Lossy weight changes | Against immutable exact-file reference: mean NLL increase <=0.01 nats/token and paired 95% upper bound <=0.02; fixed task accuracy drop <=1 percentage point with paired 95% upper bound <=3 points; otherwise reject or label separate degraded-quality stratum |
| Context behavior | Match reference at local window 1023/1024/1025 boundaries, global positions, repeated prefill/decode, near capacity and resume; define task scores per fixture and fail unexplained retrieval regressions |
| MTP | Greedy sequence equivalence to target-only on fixtures; forced accept/reject paths leave equivalent KV/positions/logits; sampled rejection/correction preserves target distribution within preregistered statistical tests |

Quantization analysis includes tensor/expert-specific error, routing changes,
outlier layers, clipping, dequantization/requantization rounding and scale fidelity.
Perplexity is `exp(mean NLL)` on fixed tokens; record both to avoid scale confusion.
Quality claims for derivatives are relative to the local quantized target unless
a separately licensed and approved higher-precision reference is available. Do
not claim preservation of an unavailable full-precision model. Confidence bounds
may require expanding 100 tasks; inconclusive noninferiority is not a pass.

Before Gate 3, freeze operator-specific thresholds and small/text fixtures. Before
Gate 4, freeze the full logit/quality envelopes and validate all affected paths.
Before Gate 5, complete long-context and MTP checks for every advertised supported
configuration. Unsupported capacities are reported explicitly.

## Results and preservation

Private run directories use mode 0700 and files 0600, with a manifest, raw timing
samples, telemetry, correctness outputs, command/configuration records and analysis
artifacts. Preserve failed runs and conversion lineage. Set a per-campaign disk
budget before captures; default to <=10 GiB per pilot campaign, checking free space
again before traces or weight derivatives. No automatic deletion of evidence.
Keep a manifest checksum index; backups require a reviewed private destination.

Each report states question, prior expectation, design, actual conditions, results,
uncertainty, correctness/quality, deviations, negative results, GO/NO-GO/PIVOT and
next discriminating experiment. Pareto analysis keeps every metric separate and
does not assert dominance when uncertainty or workload/fidelity differs.

All raw Beads, logs, prompts, model files and profiler data remain private under
the [privacy policy](development/privacy-policy.md). The existing benchmark
exporter permits only its narrow reviewed schema. Rich private result bundles
remain private; do not change that schema or pass `--reviewed` as an agent claim
of human approval. A richer public schema or Beads export needs explicit human
review of a concrete proposal. Local synchronization is not publication.
