# Verification-width attribution (K1b)

Date: 2026-11-13. Bead: rv-e05-7n3.7 (K1b, split from K1). Status: **attributed**.
These are untuned-Rosevine diagnostics, not Gate 4 claims. Rosevine implements
no MTP: this measures the target-side cost of the verification batches the pinned
reference issues ([S1](mtp-reference-semantics.md)). Raw captures, commands and
hashes stay private.

## Shape (from S1)

A verification evaluates `1 + k` rows: the last committed token plus `k ≤ 3`
drafts, at consecutive positions, all with logits. It keeps `a + 1` rows and
removes the rest from the KV. K1b measures widths 1–4 with full acceptance
(`w2a1`, `w3a2`, `w4a3`), the all-rejected case `w4a0`, and ordinary decode
(`w1a0`).

## Instrumentation change

- `replay --verify-width W --verify-accept A` runs verification steps.
  `verify-sweep` prefills once, then runs every shape per repetition from the
  same KV state, in a rotating order, instrumenting each block separately.
- Rejected rows use Rosevine's bounded rewind, which only moves the position.
- `verification_rows_and_rollback_match_sequential_decode` checks the shape on
  the synthetic GPU model. Each row's logits match sequential decode, and
  decoding resumes correctly after rejecting rows.
- Reducer: [`scripts/research/analyze_k1b.py`](../scripts/research/analyze_k1b.py).

## Design

| Item | Value |
| --- | --- |
| Rosevine | Commit `69c1e28`; HIP, first-N = 10, Q8_0 CPU experts, split attention, 12 threads, batch 64 |
| Depths | 1,900 (P3 parity fixture) and 28,935 (the 29,000-token prose-summary workload); 65 frozen ids after the prompt |
| Cells | 2 depths × {off, sync}; 4 repetitions × 5 shapes per process; 16–65 verification steps per block |
| Sessions | 2, the second in reverse order; 8 supervised single-shot processes, all successful, plus 1 smoke run; run lock and COMPUTE window |

## Acceptance evidence

| Criterion | Result |
| --- | --- |
| At least 90% of verification loop time attributed | Span closure 1.0000 in every block |
| Instrumented against uninstrumented, 2 sessions, identical top-1 | 3,136 of 3,136 steps identical. Sync-mode overhead +3.0% (width 4) to +11.8% (width 1). Sessions agree within 0.7% per shape. |
| Ranked hypotheses with benefit and failure threshold | Below |

## Results (uninstrumented step; sync spans per verification, ms)

| Shape | Step | Per committed token | CPU experts | `attn_core` global | `attn_core` local | GPU experts | Other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d1900 w1a0 | 26.9 | 26.9 | 8.6 | 0.9 | 2.1 | 3.4 | 15.4 |
| d1900 w2a1 | 41.0 | 20.5 | 16.1 | 1.4 | 2.9 | 5.6 | 18.5 |
| d1900 w3a2 | 53.0 | 17.7 | 23.3 | 1.9 | 3.6 | 7.4 | 20.0 |
| d1900 w4a3 | 64.9 | 16.2 | 30.8 | 2.4 | 4.3 | 9.1 | 21.3 |
| d1900 w4a0 | 64.7 | 64.7 | 30.7 | 2.4 | 4.3 | 9.1 | 21.4 |
| d29000 w1a0 | 36.0 | 36.0 | 8.5 | 9.6 | 2.1 | 3.5 | 15.4 |
| d29000 w2a1 | 56.2 | 28.1 | 15.5 | 17.6 | 3.0 | 4.7 | 18.6 |
| d29000 w3a2 | 74.9 | 25.0 | 22.4 | 26.1 | 3.7 | 6.0 | 20.1 |
| d29000 w4a3 | 92.7 | 23.2 | 29.5 | 33.8 | 4.5 | 7.1 | 21.8 |
| d29000 w4a0 | 92.7 | 92.7 | 29.3 | 33.9 | 4.5 | 7.0 | 21.9 |

"Other" is projections, router, transfers and head; it is the sync total minus
the named columns. Sync totals exceed the uninstrumented step by the overhead
above.

## Findings

- **Rejecting rows costs nothing.** `w4a0` and `w4a3` take the same time. The
  cost is set by width, not acceptance, as in the reference (S1).
- **CPU experts are linear in rows.** About 7.0–7.4 ms per extra row,
  close to the 8.6 ms of a single row. Each row routes to its own 8 experts in
  each of the 10 host layers, and the cost is linear in token-expert pairs, as
  K1 found for prefill. At d1900 width 4 they are 47% of the step.
- **Global-layer attention is linear in rows at depth.** At d29000 it is
  9.6 → 17.6 → 26.1 → 33.8 ms for widths 1–4: 36% of the width-4 step.
  - Reason, from the code: `k_attn_split` gives every (row, query head) pair its
    own blocks, which dequantize and read their key range independently. The
    same KV is therefore read once per row and per query head of a group.
  - Width 4 also cuts the chunks per head from 64 to 16
    (`hip::attention_chunks`).
- **GPU experts grow sublinearly:** 3.4 → 9.1 ms. More rows reach more distinct
  experts, but rows sharing an expert are batched.
- **Projections, router and head grow slowly:** 15.4 → 21.3 ms. These are
  weight-bandwidth bound, and more rows reuse the same weights.

**Projection (inference, not a measurement).** Assume the reference's natural
greedy acceptance (3.17 committed per verification at 1,900, 3.34 at 29,000)
and its 4 ms draft cost. Rosevine's target-side verification would then give
about 21.7 ms per token at d1900 and 29.0 at d29000, against 26.9 and 36.0
target-only: about 1.24× at both depths. The reference measured 1.41× and 1.29×
under its own placement.

## Ranked verification hypotheses

| Rank | Candidate | Evidence | Expected benefit | Failure threshold |
| --- | --- | --- | --- | --- |
| 1 | **Row- and group-shared decode attention**: one block reads each key range once for all verification rows and all query heads of its KV group | Global `attn_core` 33.8 ms at d29000 width 4, against 9.6 ms at width 1 | Up to about 24 ms per width-4 verification at d29000 (26% of the step); small at d1900 | Global `attn_core` above 15 ms at d29000 width 4, or a step gain below 15 ms there. Class T; KL band and f64 comparison as in K3. |
| 2 | **Batched CPU expert kernel** (existing deferred bead rv-e05-7n3.9) | CPU experts 30.8 of 64.9 ms at d1900 width 4, linear in rows | Bounded by about 22 ms per width-4 verification. Gains only where rows share experts or decode amortizes. | That bead's threshold: less than 1.5× CPU-expert time |
| 3 | Fewer host-expert layers for verification (placement) | Same linear CPU cost | Depends on VRAM; first-N = 2 more than doubled prefill (K1) | Needs an ADR 0002 decision; not a kernel candidate |

None of these is activated. MTP implementation (S2) remains a separate decision.
