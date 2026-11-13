# Persistent CPU experts with F32 activations (R3a.3)

Bead: rv-e06-k0i.5. Parent: `cbb9257` (R3a.2). Status: implementation verified;
the performance objective is missed, with measured results retained below.
This is an internal ablation of the R2 reference runtime. No comparison with the
Vulkan runtime's performance, production-router change or broad upstream port is included.

## Implementation and ownership

- `runtime/src/expert_pool.rs`: one persistent worker pool per CPU/HIP session
  using host experts. GPU-only placement creates no CPU pool. Validated jobs
  borrow immutable weight/input views and disjoint output regions. A completion
  guard drains every worker before those borrows expire, including cancellation
  or caller unwind. Worker panics cancel/drain the job and return a fixed error;
  workers remain reusable. Drop wakes and joins parked workers. Partial outputs
  are not committed by the executor.
- Work uses bounded row chunks. The caller polls cancellation while waiting;
  workers check cancellation between rows. Cancellation is cooperative, not a
  hard real-time scheduling guarantee. The external supervisor still enforces
  request and cleanup deadlines.
- `runtime/src/expert_kernel.rs`: scalar, AVX2 and AVX-512F paths, selected by
  runtime CPU features. Standard Q4_0 blocks remain in place; scales and
  activations stay F32. There is no Q8_0 activation path or FMA contraction.
  SIMD reductions change summation association (correctness class T).
- Cancellation-sensitive sums recompute in the original scalar order when
  `abs(result) <= 0.02 * sum(abs(products))`, or when vector values are nonfinite.
  This is an empirical compatibility guard, not a universal floating-point error
  proof. Unsupported forced direct dot calls use scalar; pool construction
  rejects an unavailable explicitly selected ISA.
- Worker stacks are configured at 256 KiB. Persistent row scratch is bounded by
  the larger expert/input width; bounded descriptors and stack reservations are
  included in the session allocation budget. HIP also checks its host scratch
  plus workers against `max_bytes`. These are configured reservations, not RSS
  estimates or promises about allocator/OS overhead.
- The original scalar operators remain unchanged as the reference. Both the CPU
  executor and HIP host-expert path use the pool; routing, expert placement and
  slot-order combination are unchanged.

The default wait policy is **sleep**. Hybrid polls for at most 20 microseconds
after a job, then parks; the experimental spin control polls for at most
100 microseconds, then parks. There is no indefinitely spinning mode. Idle
parking, shutdown, panic recovery, caller unwind, cancellation, odd shapes,
multiple inputs, thread counts and invalid/overlapping output ranges have tests.

Rust 1.85 remains the declared minimum. Stable AVX-512 intrinsics require 1.89;
the build script enables that module only on sufficiently new compilers. Older
compilers use scalar/AVX2. The locally tested compiler is 1.98.1; an actual 1.85
toolchain build was not run. No dependency or toolchain was installed.

## Upstream study and layout decision

The local llama.cpp source was verified at
`cd959bae99ff4bed22bef2604f45241c09bbb8c8`. Its
[x86 Q4_0 dot](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cpu/arch/x86/quants.c)
uses Q8_0 activations, so it does not directly satisfy this bead's F32 contract.
The standard low/high-nibble unpacking and the
[persistent worker/poll/park lifecycle](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-cpu/ggml-cpu.c)
were studied. This change imports no GGML subsystem or upstream source files;
Rosevine remains GPL-3.0.

Weights retain their existing row-major, 18-byte Q4_0 blocks, with exact unaligned
SIMD loads and no overread. Persistent F32 row scratch supports scalar fallback.
There is no measured load-bound finding that justifies the bead's conditional
row-blocked repacking or huge-page experiments. Neither was adopted. The
compatibility fallback and dispatch costs remain more immediate questions.

## Numerical gates

The fixed gates are per-row relative error at most `1e-5` versus the original
scalar dot at exact shapes, F64 relative L2 at most `1e-5`, bitwise batch/step
agreement, and the existing full-runtime band against both frozen references
(mean KL at most about 0.08, top-1 at least about 92%). No threshold was relaxed.

The first vector candidate, with a weaker cancellation guard, failed the
pointwise scalar gate despite small aggregate F64 error. It was retained
privately and rejected before performance claims. The strengthened guard passes
9,732,096 row comparisons: every row of all 128 experts in layers 0, 14 and 29,
gate/up and down matrices, three distinct deterministic F32 activation patterns,
and both vector ISAs. Maximum scalar-relative error is `7.33e-6`; maximum F64
relative L2 is `4.78e-7`. Synthetic tests also cover odd dimensions, extreme
scales, zeros, nonfinite values and unsupported-ISA fallback.

| Frozen reference | Scored rows | Mean KL | Top-1 agreement |
| --- | --- | --- | --- |
| CPU | 1,785 | 0.064073 | 92.61% |
| Vulkan fast | 1,785 | 0.070023 | 92.94% |

Both full-runtime checks pass the existing band. These values do not establish
better model quality. Full-model KL was run on the target's default AVX-512 path;
AVX2 has row-level/model-weight and synthetic checks, rather than its own full KL run.

The preserved baseline reproduces the historical 520-token fixture exactly.
The candidate first differs at token **143** (one-based); 145 positions out of
520 match. Identity is not required under class T. The seven primary generation
pairs use 200 steps, so their continuations differ after that point. A separately
identified, post hoc 100-step control tests the common continuation prefix.
Real-model execution at batch 64 and batch 1 is bitwise identical across all
seven chunks' scored-logit digests (1,785 full vocabulary rows). Both reference
files also produce identical candidate digests. This complements the synthetic
batched/stepped, rewind and cancellation/resume checks; it is not a whole-model
identity claim against the pre-change scalar build.

## Controlled performance method

The target is Ryzen 5 7600X, RX 7800 XT 16 GiB, Fedora 43, Rust 1.98.1 and HIP 6.4.
The model is the same hash pinned by [R3a.2](decode-attribution.md); F32 activations,
Q8_0 KV, first-N=10, batch limit 64 and the historical 40-token prompt are fixed.
All 12 logical CPUs are allowed by affinity. Six workers are not pinned one per
physical core. No manual frequency settings, router configuration, placement or
model weights changed; the agreed COMPUTE window is applied and restored.

The GPU run lock is held before entering the existing COMPUTE power window.
Every workload uses `scripts/gpu_campaign.py` command supervision, live preflight,
100 ms safety collection, 1 GiB device and 2 GiB RAM/disk reserves, existing thermal
and sustained-paging stops, production-server growth protection and confirmed cleanup.
Raw logs, exact commands, source/binary/library/model/reference hashes, fixtures,
resource counters and failed attempts remain private. Analysis runs after timed
workloads finish. No competing GPU benchmark or other model was launched.

The pre-change executable and its HIP library were preserved before implementing
the pool. Seven matched 200-step pairs span two sessions, with two warmups per
variant per session and alternating AB/BA order. Rates use the driver's own
decode-loop timer (199 evaluations), excluding model load and prefill. The existing
`benchmark_core.paired_improvement` computes the median paired relative change and
95% percentile bootstrap interval (10,000 resamples, seed 20261113). Individual
tokens are not counted as independent trials.

A 60 Hz scheduler-deadline observer runs in the private command wrapper for both
variants. It measures CPU scheduling lateness over load/prefill/decode, not GUI
input/render latency. Process resource counters include the wrapper and child.

## Results

| Primary 200-step generation | Result |
| --- | --- |
| Baseline median | 13.99 tokens/s |
| Candidate median | 24.51 tokens/s |
| Paired median benefit | 75.3% |
| Paired 95% interval | 74.9% to 77.2% |
| Entirely unflagged matched pairs | 0; no unflagged estimate |

Every primary pair has a thermal quality flag in at least one phase. The result
is a **flagged internal campaign result**, with no unflagged-subset confidence
interval. It is not a clean thermal replication or a claim of parity with Vulkan.

The host spans give 2.098 ms per CPU expert layer at 12 workers and 2.695 ms at
six workers. The **at-most-1.5 ms objective is missed**. The scalar-pool ablation
is 4.894 ms/layer at 12 workers. Expert-output upload remains about 0.39 ms/layer.
The default path recomputes 21.9% of dot rows in scalar order in this decode.
The exact contribution of those rows versus vector, dispatch and cache costs
was not independently timed; the unchecked 0.54 ms prototype is not an applicable
integrated result under the pointwise compatibility gate.

Initial proxy controls use two warmups and 80 measurements per condition, exact
eight-expert target shapes, the same scheduler observer, and reversed configuration
order in the second session. Twelve-worker cache-evicted medians are about 5.1 ms
legacy, 5.1 ms persistent scalar, 3.07 ms AVX2 and 3.0 ms AVX-512. Six-worker
AVX-512 medians are about 3.85–3.89 ms. A single 64 MiB streaming-write competitor
raises the twelve-worker AVX-512 median to about 3.13 ms, versus about 7.05 ms
legacy. It perturbs scheduling and memory together; this is not pure DRAM isolation.

Sleep/hybrid/bounded-spin proxy differences are small and inconsistent between
sessions; scheduler-latency tails also vary. The default stays sleep. All policies
park when idle; the benchmark asserts this after its work. Neither these short
probes nor the 60 Hz observer establish perceptual desktop responsiveness.

The post hoc 100-step control keeps **all decoded tokens identical** in every
baseline/candidate pair. It has its own two-session, seven-pair schedule and two
warmups per variant/session: 14.04 to 24.80 tokens/s median, paired benefit
76.4% (95% interval 75.0–77.0%). This supports a speedup with a common trajectory;
it does not replace or retroactively preregister the primary 200-step experiment.
It also has zero entirely unflagged pairs under the conservative whole-command
thermal filter.

The reducer `scripts/research/analyze_cpu_experts.py` checks successful cleanup,
matched options, declared prompt/output lengths, continuation identity for the
100-step control, seven-pair coverage, the frozen KL band and batch/step digests.
It retains thermal flags and refuses to invent an interval for fewer than seven
unflagged pairs. Its output remains private.

Final-build proxy repeats confirm the result with fallback accounting enabled:

| Workers / condition | Legacy median / p95, ms (session 3 / 4) | AVX-512 median / p95, ms (session 3 / 4) |
| --- | --- | --- |
| 6, evicted | 7.118 / 7.915; 7.150 / 8.607 | 3.801 / 4.142; 3.891 / 4.264 |
| 12, evicted | 5.134 / 6.918; 5.147 / 6.127 | 2.981 / 3.217; 2.956 / 3.222 |
| 12, contended | 7.207 / 8.592; 7.197 / 8.462 | 3.151 / 3.832; 3.062 / 3.622 |

The proxy recomputes 28.6% of rows, versus 21.9% in the actual decode: input and
expert selection matter. It should not be substituted for in-situ attribution.
The actual-generation hybrid/spin controls give 24.20 / 24.29 tokens/s, against
the default sleep campaign's 24.51 median. Each alternate policy has only one
generation control, so this does not establish a statistically significant
regression. Neither demonstrates a benefit sufficient to change the default.

Across the seven primary trials, each baseline run's observer p95 is about
0.98–1.09 ms and each candidate's about 0.61–0.84 ms. Hybrid and bounded-spin
controls give 0.70 and 0.81 ms. The first matched pair uses about 126 versus
59 CPU-seconds over the entire command; hybrid and spin use about 60 and
62 CPU-seconds. These are workload/resource observations, not isolated policy
costs or a promise about interactive rendering. Peak process RSS in that pair
is about 12.48 GiB for both builds; file-backed model pages dominate it.

All successful R3a.3 runs retain more than 21 GiB MemAvailable; peak total VRAM
use is about 11.52 GiB. No hard reserve, sustained thermal, paging-pressure or
production-server growth stop fired. GPU maxima are 61 C edge, 69 C junction and
74 C memory. CPU maxima across the complete campaign are 98.5 C CCD and
95.375 C Tctl. Primary paired trials' flags are CCD excursions; the whole-command
filter includes loading and prefill. There is no decode-only temperature or
continuous CPU-frequency trace. Primary generation trials record initial CPU
frequency/load snapshots; the matched-prefix follow-up lacks those snapshots.
System-wide paging deltas are retained with unknown process attribution.

## Verification and limitations

- The normal CPU suite, per-ISA numerical tests and pool failure/lifecycle tests
  pass. The GPU operator and synthetic executor suites pass, including
  instrumentation, batch/step, cancellation/resume and resource checks.
- `scripts/check.sh` passed: privacy/history, formatting, Python/shell, Rust tests,
  rustfmt and clippy with warnings as errors. All 42 CPU Rust tests and eight
  separately supervised GPU operator/executor regression tests passed.
- The two numerical reducers have rejection tests for partial samples, altered
  workload lengths, missing KL chunks and band failures. They reuse the existing
  pairing/percentile implementations rather than introduce a parallel metric definition.
- Long-context fidelity, AVX2 full-model KL, an actual Rust 1.85 build and GUI
  input/render latency are unrun. The frozen corpus covers 1,785 scored positions;
  this does not complete the deferred full-quality corpus.
- The confidence intervals describe these paired windows. Thermal flags and
  shared-machine conditions limit extrapolation to another day or workload.
- The compatibility guard is empirical, and the per-row branch cost has not been
  isolated. Future numerical or layout work must keep the current acceptance
  criteria or obtain a separately reviewed contract. No target was retuned to fit.

Changed files: `runtime/src/expert_kernel.rs`, `runtime/src/expert_pool.rs`,
`runtime/src/exec.rs`, `runtime/src/hip_exec.rs`, `runtime/src/error.rs`,
`runtime/src/lib.rs`, `runtime/src/bin/rosevine-inspect.rs`, `runtime/build.rs`,
`runtime/examples/expert_validate.rs`, `runtime/examples/worker_bench.rs`,
`scripts/research/analyze_cpu_experts.py`, `tests/research/test_cpu_experts.py`,
`runtime/README.md` and this report. Kernel/pool unit tests live in their modules.

## Reproduction

Build without registry access:

```sh
cargo build --manifest-path runtime/Cargo.toml --offline --locked --release --features hip
cargo build --manifest-path runtime/Cargo.toml --offline --locked --release --examples
```

Run the workloads inside the supervisor/power/lock envelope above:

- `expert_validate MODEL`: model-weight row checks against scalar and F64.
- `worker_bench MODEL legacy|scalar|avx2|avx512 sleep|hybrid|spin`: layer proxy,
  cache eviction, idle/wakeup, bounded streaming competitor and scheduler observer.
- `rosevine-inspect generate MODEL --backend hip --cpu-moe 10 --threads 12 --n 200`:
  feed the private historical token fixture on stdin; pass the explicit allocation
  limits, batch size and safety session used by the recorded campaign.
- Add `--instr host` for attribution. `--expert-kernel scalar|avx2|avx512` forces
  an available CPU path; default is auto. `--expert-wait sleep|hybrid|spin` selects
  the bounded wait policy. `spin` always parks after its 100 microsecond budget.
- Use `kld --base PRIVATE_REFERENCE --chunks 7 --batch 64` and `--batch 1` for
  the frozen calibration and batch/step comparison, with matching backend/placement.

These are workloads, not autonomous safety harnesses. Keep their output private;
benchmark publication still follows the repository's human-review policy.
