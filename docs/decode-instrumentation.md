# Decode instrumentation (R3a.1)

Date: 2026-11-13. Bead: rv-e05-7n3.5. Status: implemented; numbers below are `r2-ref`
(HIP, `first_n(10)`, 12 threads, batch limit 64, 200-step greedy decode, idle router and host).

## Design

- `runtime/src/instr.rs`: contiguous lap spans per layer and module, plus `outside` for the
  caller's time between evaluations. Pure bookkeeping, unit-tested without a GPU.
- `hip.rs`: per-thread counters of launches, copies (count and bytes per direction) and syncs,
  always on (a thread-local add). The probe's own syncs and events are counted apart.
- Additive C ABI: `rvh_event_create/destroy/record/sync/elapsed` (no existing signature changed).
- `HipSession::set_instrumentation(None | Some(Host | Sync | Events))`; off by default, so the
  disabled path is one branch per boundary. Enabling allocates everything; evaluation allocates
  nothing. Aborted evaluations (error, cancellation) are dropped and counted.
- `rosevine-inspect generate --instr host|sync|events` prints the summary to stderr.

Span boundaries and blocking points are documented in `hip_exec.rs`. In `Host` mode, device work
is charged to the next blocking span (mostly `router`); use `Sync` or `Events` for per-module
device time.

## Acceptance evidence

| Criterion | Result |
| --- | --- |
| Flag off, synthetic | Logits SHA-256 equal to the pre-change build; all three modes bitwise equal to off |
| Flag off, real artifact | 520-token greedy fixture identical (520/520) |
| Flag off, timing | 7 ABBA pairs against the pre-change build: mean −0.21 ms/token (95% bootstrap −0.68 to +0.21) on about 71.6 ms; no regression |
| Closure | 1.0000 in every mode, against the harness's own decode-loop timer |
| Counters vs hand count | Exact: 1,129 launches, 91 H2D, 41 D2H, 5 D2D per token (132 blocking calls); synthetic test asserts the hand-count formula |

## Mode calibration (ms/token, one run each)

| Mode | ms/token | Probe calls/token | Note |
| --- | --- | --- | --- |
| Off (ABBA mean) | 71.6 | 0 | |
| Host | 72.7 | 0 | Within run-to-run range |
| Events | 73.6 | 245 | Device time per span without extra syncs |
| Sync | 75.1 | 243 | About 14 µs per probe sync |

## First observations (input to R3a.2, not yet conclusions)

- CPU expert kernel in situ: 5.28 ms per host layer (host span), against 4.72 ms in the earlier
  microbenchmark.
- The host-expert output upload (11 KB, pageable) takes 0.39 ms of host time per layer, but only
  0.035 ms on the device timeline; four similar uploads on GPU-expert layers take 0.085 ms in
  total. Candidate causes: GPU idle-state wake after about 5 ms without work, or runtime staging.
- Host-layer minus GPU-layer total is about 5.47 ms, consistent with the earlier 5.40 ms slope.
