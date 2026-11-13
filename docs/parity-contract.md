# Parity contract (P1)

Date: 2026-11-13. Status: implemented contract and adapters for the
[parity track](performance-parity.md) step P1 (`rv-two`). No parity benchmark
has run. This document freezes what P3 measures and how it is compared; it is
not a result and authorizes no run outside the existing GPU run rules.

Code: [parity_contract.py](../scripts/parity_contract.py) (matrix, arm
identity, pairing rules, paired ratios, milestones, report),
[parity_replay.cpp](../research/parity_replay.cpp) (llama.cpp adapter) and the
`replay` command of `rosevine-inspect` (Rosevine adapter). Tests:
[test_parity_contract.py](../tests/research/test_parity_contract.py).

## Existing infrastructure reviewed

- **Kept:** `benchmark_core` supervision, private capture bundles, `paired_schedule`,
  `summarize` and percentile bootstrap; `benchmark_metrics`/ResourceGuard
  resource reduction; `gpu_campaign` run lock, preflight, 100 ms safety telemetry,
  thermal/VRAM/RAM stops and router checks; the private run identity in
  `benchmark_manifest`.
- **Not reused as is:** `paired_analysis` validates one-factor llama-server
  controls with a fixed 512-token output and llama.cpp-only settings. A
  cross-engine comparison varies the engine itself, so P1 adds a separate pairing
  layer rather than loosening that validator.
- **Public exporter unchanged.** Parity rows and adapter output are private
  captures. A public summary needs its own reviewed schema.

## Finding: llama-bench cannot supply matched fixtures

At the frozen pin, `test_gen` and `test_prompt` in
`tools/llama-bench/llama-bench.cpp` feed `std::rand() % n_vocab` token IDs (BOS
first). Rosevine decodes real text. In a MoE model the token stream decides which
experts run, and with first-N CPU expert placement it decides how much work lands
on the host. llama-bench numbers therefore measure a different workload.

- llama-bench output is parsed (`parse_llama_bench_json`) only as a
  `random-token` diagnostic. It can be compared with other llama-bench runs, such
  as HIP against Vulkan, but `classify_pair` rejects pairing it with a frozen fixture.
- Matched cells use two replay adapters with one output schema
  (`rosevine-parity-replay-v1`). Both read the same frozen token IDs, prefill
  the first `P`, then evaluate the next `D` one at a time.

## Timing families and boundaries

| Family | Boundary (both arms) | Adapter |
| --- | --- | --- |
| `engine` | Synchronized completion of each evaluation; last-position logits on the host; no sampling, tokenization or delivery; model load excluded; warm process | `replay` / `parity_replay` |
| `user_visible` | Request admission to first delivered token, including tokenization, prefill, logits, sampling and delivery; cold and warm reported separately | None yet: Rosevine has no admission/delivery path |

The llama.cpp adapter requests logits only for the last position. llama.cpp copies
them into its host output buffer within `llama_decode`, and the adapter then calls
`llama_synchronize`. Rosevine's `Logits::Last` returns a host slice, so both arms
include that readback. An arm's boundary must equal its family's declared
boundary exactly, or `validate_arm` rejects it.

Each repetition clears the cache, times the prefill of `P` ids (batched
`n_batch`, logits on the last position) and then times each decode step
separately. Per-repetition rates are `P / prefill_s` and `D / sum(step_s)`. A pair
is one matched repetition or session, never one token.

## Matrix v1 (frozen denominator)

`matrix_v1()` returns the cells and a SHA-256 over them. A report built from an
edited matrix is rejected, and cells cannot be dropped after results are seen.
Every cell is compared separately against `vulkan` (ADR 0001's Gate 4
comparator) and `hip` (same-backend diagnostic).

| Cells | Family | Workload | Status |
| --- | --- | --- | --- |
| `decode-d512` | engine | 512-token prompt, 128 decode tokens | supported |
| `decode-d2048`, `-d8192`, `-d16384` | engine | depth prompt, 128 decode tokens | feasibility check required (all passed in P3; see [parity baselines](parity-baselines.md)) |
| `prefill-p128`, `-p512` | engine | prompt from an empty cache | supported |
| `prefill-p2048`, `-p8192` | engine | prompt from an empty cache | feasibility check required (passed in P3) |
| `ttft-warm-p128` … `-p8192` | user_visible | warm TTFT on the same prompts | unsupported (no Rosevine delivery path) |

"Feasibility check required" covers positions beyond the roughly 1,030 that R2
verified on the real artifact. Before timing, P3 must check device budget and
attention correctness at that depth. A failed check records the cell as
unsupported, with the reason.

## Arm identity and pairing rules

Each arm is a complete `rosevine-parity-arm-v1` record; no field may be null.
`arm_from_replay` takes the adapter's printed facts and rejects any declared
value that contradicts them.

| Class | Fields | Rule |
| --- | --- | --- |
| Hard | model and fixture SHA-256, token binding hash, `P`, `D`, timing family, boundary, warm/cold, hardware class, K/V type, power policy | Any difference rejects the pair |
| State | CPU-expert layer count, tensor homes, CPU-expert activation precision, capacity | A difference downgrades the pair to `practical_configuration`, with the differences listed; such a pair never counts toward parity milestones |
| Machinery | microbatch, threads, affinity, attention policy, CPU repack, expert kernel, wait policy, resident bytes | Must be declared; recorded, not matched (this is the machinery under test) |
| Identity | engine, backend, source revision, binary SHA-256, `untuned`, series | The reference must be llama.cpp at `cd959bae…` (`frozen`) or `9daa0aa2…` (`newer-control`, reported separately). The candidate must be Rosevine HIP. |

**Current state difference.** llama.cpp quantizes CPU-expert activations to Q8_0,
while Rosevine's reference path is F32 (D3). Until the R3a.4 Q8_0 variant exists,
every Rosevine/llama.cpp pair at matched placement is a
`practical_configuration` comparison. This is reported, not hidden.

**Capacity alignment.** llama.cpp may pad `n_ctx`. Choose a capacity that is a
multiple of 256 and pass it to both arms with `--ctx`. The adapters print the
effective value.

## Statistics and milestones

- `paired_ratio` computes per-pair ratios: throughput as Rosevine / reference,
  latency as reference / Rosevine, so larger is always better. It reports the
  median and a 95% percentile-bootstrap interval, needs at least 7 pairs and
  defaults to 10,000 resamples. Raw summaries of both arms are kept.
- `milestones`:
  - **P90:** lower bound ≥ 0.90.
  - **Operational parity:** point ≥ 1.00 and lower bound ≥ 0.98.
  - **Strict parity:** lower bound ≥ 1.00.
  - **Advantage:** lower bound > 1.00 and point ≥ 1.05.
- `build_report` writes one row per cell and baseline: `measured`, `not_run`,
  `unsupported` or `correctness_failed`. It counts milestones per baseline over
  the full predeclared denominator, never averages across cells or baselines, and
  never lets HIP results offset Vulkan. A row whose correctness checks failed
  carries no speed outcome.
- `pair_schedule` reuses the experiment protocol's alternating schedule (2 warmups
  and 7 trials by default).

## Resource, CPU and contention controls

- **Memory:** peak VRAM comes from device telemetry at 100 ms plus each engine's
  own allocation report (llama.cpp buffer-size log lines; Rosevine
  `device_bytes`). Peak RSS, PSS, anonymous and file-backed bytes come from
  ResourceGuard rows reduced by `benchmark_metrics`. The sampling interval and the
  possibility of missed peaks are reported.
- **CPU:** process CPU time is taken from `/proc` deltas over the timed span,
  converted to core-equivalents, with threads and affinity recorded.
  `resource_ratios` reports candidate / reference per cell in natural units and
  flags budget breaches even when speed improves. A missing value stays missing,
  with a reason.
- **Primary condition is an idle host.** The run lock is held, the production
  router is checked idle before each arm, no other model or CPU reference process
  runs, and existing thermal, VRAM, RAM and paging stops apply. A
  bounded-contention control (a predeclared fixed background CPU load) is a
  separate labelled condition, never mixed into idle pairs.
- **Correctness first:** a speed row counts only when the existing S/T/N checks
  pass for that build (exact-shape tests, teacher-forced KL against the frozen
  band, routing checks). Vulkan token identity is not required.

## Adapter usage

Build the llama.cpp adapter against each pinned build tree (no install, no
source change):

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$SRC/include" -I"$SRC/ggml/include" \
  research/parity_replay.cpp -L"$BUILD/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$BUILD/bin" -o "$PRIVATE/parity_replay"
"$PRIVATE/parity_replay" --validate-only TOKENS P D        # no backend initialized
"$PRIVATE/parity_replay" --run MODEL TOKENS P D REPS CTX BATCH UBATCH THREADS CPU_MOE q8_0 on on
rosevine-inspect replay MODEL --backend hip --n-prompt P --n-decode D --reps R --ctx CTX \
  --cpu-moe 10 --threads T < TOKENS
```

`CPU_MOE` uses the same per-layer expert pattern as `--n-cpu-moe` at the pin.
`--run` and the Rosevine command load weights. Run them only under
`gpu_campaign` supervision, inside a reviewed run window.

## Verification performed

- `parity_replay.cpp` compiles with `-Werror` against both the pinned Vulkan
  build and the private pinned HIP build. Validate-only mode accepts a fixture and
  rejects a short one.
- The Rosevine `replay` command ran a 5-token, 2-repetition CPU-only smoke test
  on the real artifact (no GPU, no timing claim). It produced the schema above.
- Both native adapters and `tokens_fnv1a64` produce the same binding hash for
  the same ids. A unit test anchors that value.
- 20 unit tests cover the frozen matrix, adapter parsing, incomplete or foreign
  output, every rejection class, practical versus machinery classification, ratio
  direction, milestones, resource breaches and report denominators.

## Limits

- No GPU replay has run on either engine. The adapters' timing overhead against
  llama-bench and against `generate` is a P3 calibration item.
- The `user_visible` family has no Rosevine harness, so it stays unsupported.
- Tensor homes are derived from the declared first-N rule, not read back from
  either engine's allocator. The buffer-size lines and `device_bytes` are the
  available cross-check.
- Activation precision makes every current pair a practical-configuration
  comparison (see above).
