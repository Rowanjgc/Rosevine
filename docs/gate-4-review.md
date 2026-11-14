# Gate 4 specialization review

Date: 2026-11-13
Bead: rv-e05-7n3.4
Status: Reviewed
Outcome: **NO-GO** for a Gate 4 optimized-path claim. One internal improvement
(K3 split-K attention) is retained as the runtime default. The correct reference
kernels stay available.
Reviewer: agent evidence review, at the maintainer's direction (2026-11-13) to
work RV-05 through Gate 4 under the existing rules. No criterion was changed
after its evidence was collected.

## Criteria and when they were frozen

The Gate 4 claim rule ([optimization validation](optimization-validation.md#acceptance-and-gates),
[research roadmap](research-roadmap.md)) predates every K2 and K3 measurement:

| # | Criterion |
| --- | --- |
| C1 | At least 5% median benefit in a named primary metric, with a paired 95% interval above zero |
| C2 | Comparison against `vk-ref` or the defined unchanged control. A paired interval against `r2-ref` is an *internal ablation* and "does not count as Gate 4 evidence". |
| C3 | Reproduced in a second session |
| C4 | No unacceptable numerical or quality regression; every regression reported |

Each investigation also recorded its own criteria before measurement:
[K3](split-k-attention.md#acceptance-criteria-recorded-before-measurement) and
[K2](q4-gemv.md#acceptance-criteria-recorded-before-measurement).

## Candidates

| Candidate | Investigation outcome | C1 | C2 | C3 | C4 |
| --- | --- | --- | --- | --- | --- |
| K3 split-K decode attention (`--attention split`, default since `4ab375d`) | GO on performance. Two pre-recorded correctness criteria (2b, 3) not met; adopted by maintainer decision. | Met internally: d16384 step −47.0 ms (ratio 0.403, 95% CI excludes 1); d512 −4.4 ms (0.854) | **Not met:** paired against the R2 reference kernel (`r2-ref` lineage), an internal ablation | Met: 2 sessions | KL inside the band against both references (0.0764 / 0.0733). Kernel-to-kernel KL shift +0.0055 and d16384 top-1 agreement 0.9375 reported. The split kernel is closer to exact f64 attention. |
| K2 quarter-block Q4_0 GEMV (`--q4 vec`) | **NO-GO**: 7% slower | Not met: d512 ratio 1.070 (1.069–1.075) | Internal | Met (2 sessions) | KL inside the band; slower in every GEMV span |

## Against `vk-ref`

There is no paired `vk-ref` comparison of the current runtime. The latest paired
parity record ([P3](parity-baselines.md)) predates K3 and used F32 CPU experts.
For scale only (unpaired, different sessions and activation profile):

| Depth | `vk-ref` (P3) | Rosevine now (K2 campaign, reference arm, split attention, Q8_0 CPU experts) |
| --- | --- | --- |
| 512 | 58.4 tok/s (17.1 ms) | 25.6 ms per step (about 39 tok/s) |
| 16,384 | 45.7 tok/s (21.9 ms) | 31.9 ms per step (about 31 tok/s) |

Rosevine is still well behind the external baseline in decode and much further
behind in prefill (K1: about 100 tok/s against about 800 at P = 2,048). No
candidate can pass C2 against `vk-ref` today, and none was measured that way.

## Outcome

- **NO-GO:** no path meets the Gate 4 claim rule. This is a negative result under
  the frozen criteria, not a closed research direction.
- **Retained:**
  - K3 split-K attention as the default. It is an internal improvement inside
    the KL band, and the `tuned-k` milestone for attention.
  - The R2 reference attention kernel and `k_q4_mm` as correct fallbacks
    (`--attention ref`, `--q4 ref`).
- **Rejected:** `k_q4_mv`. It stays only as a tested diagnostic flag.
- **Supported workloads:** target-only text decode and prefill with first-N
  static placement, as in Gate 3. MTP is not implemented. Its reference
  semantics are now recorded ([S1](mtp-reference-semantics.md)).
- **What the evidence says to do next** (each needs its own activation):
  1. Prefill is dominated by host experts: 71% at first-N = 10 (K1). The
     deferred prefill expert offload (rv-e05-7n3.8) and batched CPU expert
     kernel (rv-e05-7n3.9) are the ranked candidates.
  2. Decode is now flat in depth outside attention: the remaining 27 ms per
     token is CPU experts, transfers and projections (K1). The device routing
     follow-up (parity-plan G4) and any GEMV work should raise memory-level
     parallelism, not narrow per-lane work (K2).
  3. MTP: verification-width costs are attributed in K1b
     ([verification attribution](verification-attribution.md)). Implementation
     (S2) needs its own activation.
- The final frontier study (rv-e08-nhh.3) can proceed with the unoptimized
  fallback. It must state that no Gate-4-validated optimized path exists.

## Interpretation note

C2 allows "the defined unchanged control", but the same document states that a
paired interval against `r2-ref` does not count. This review reads the unchanged
control as the matching external configuration, not Rosevine's own earlier
kernels. If the maintainer intends K3's paired result against its reference
kernel to count, K3 would meet C1–C3, and C4 would rest on the accepted 2b/3
deviation. That would be a **GO, scoped** to depth-dependent decode attention.
The evidence is in place for either reading; the decision is the maintainer's.
