# Gate 3 target-only reference correctness review

Date: 2026-11-13
Bead: rv-e04-398.3
Status: Reviewed
Outcome: **GO, scoped** to target-only text execution in the supported matrix below
Reviewer: agent evidence review, at the maintainer's direction (2026-11-13) to
complete Gate 3 under the existing rules. No criterion was changed after the
evidence was collected.

## Criteria and when they were frozen

Gate 3 GO requires that every required target-only check passes without
unexplained mismatches, against protocol tolerances frozen in advance. There is
no speed threshold. The bead's scope is per-layer differences, routing, logit
tolerances, deterministic fixtures, resource ownership, shared-code blind spots,
and an explicit statement of supported contexts.

**Source documents.** The tolerances come from the
[divergence review](reference-divergence-review.md#updated-tolerances-for-gate-3-and-later-candidates):
- committed 2026-11-12 (68e802a);
- band upper estimate moved to about 0.08 by C1 on 2026-11-13 (dba441b);
- [ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md#consequences) adds
  HIP greedy output identity.

All of these predate the R2 runs (2026-11-13, 07:10–07:56) and this review.

| # | Frozen criterion |
| --- | --- |
| T1 | Repeat checks are bit-exact on the same backend and settings |
| T2 | Teacher-forced divergence from **both** references inside the band: mean KLD at most about 0.08 and top-1 at least about 92% |
| T3 | Exact-shape op conformance against the CPU path |
| T4 | Layer-0 intermediate agreement within about 1e-2 relative |
| T5 | HIP greedy output identity (ADR 0001) |
| T6 | No fidelity claim "better than" a reference from a lower KLD inside the band |

## Check results

Evidence is in [R2](runtime-r2-execution.md#verification), unless marked
*follow-up*. Follow-up items come from the
[optimization research follow-up](optimization-research.md#follow-up-scrutiny-2026-11-13),
which ran the unmodified R2 runtime.

| Check | Result | Evidence |
| --- | --- | --- |
| T1 repeat | **Pass** | 1,024 stepped HIP decode steps give logits SHA-256 equal to the 512-token batched run. *Follow-up:* six further sessions on the same commit reproduce the 520-token greedy continuation exactly or as a prefix. They cover max batch 1 and 64, N = 10 and 20, and a CPU-contended run. |
| T2 band | **Pass** for both executors against both references | CPU: 0.062 / 93.4% and 0.067 / 94.0%. HIP (N = 10): 0.066 / 92.3% and 0.074 / 92.9%. 1,785 positions |
| T3 op conformance | **Pass** | Every HIP kernel at target shapes is within 1e-5 relative of the CPU path, or bitwise; out-of-allocation ranges are refused |
| T4 layer 0 | **Pass** | 6.1e-3 for both executors against the reference (tolerance about 1e-2); input embedding bitwise |
| T5 greedy identity | **Pass** | As T1 follow-up: identical across sessions, batch limits and placements |
| T6 | Respected | R2 makes no "better than reference" claim |
| Per-layer differences | Explained | Smooth growth with depth from routing flips (all 30 layers dumped); no step at the placement boundary; HIP and CPU agree to 2.5e-5 at layer 0 |
| Routing | Pass | 97.7% of tokens have identical expert sets at layer 0 against the reference. Selection is identical at N = 10 and 20, and the expert-output difference is 7.2e-7 |
| Window and state | Pass | Synthetic exact boundary at 1023/1024/1025 with a wrapping ring. Real-artifact wrap at 1,030 positions with a window-disabled control. Cancel and resume, rewind and ring limits |
| Resource ownership | Pass | Allocation is up front and checked against `max_bytes` / `max_vram` and the reserve. Held bytes are invariant across evaluations. Oversize and invalid inputs are rejected without side effects. VRAM returned to baseline after every run |
| Shared-code blind spots | Bounded | The CPU and HIP executors share `select_experts`, weight binding, the tokenizer and Q8_0 KV quantization, so a defect there would be common-mode. The independent llama.cpp references (T2, T4, the routing agreement) and R1's tokenizer parity cover this at band and layer-0 resolution |

## Mismatches and explanations

**HIP top-1 near the floor.** HIP's top-1 against the CPU reference (92.3%) sits
near the band floor.
- Per-chunk KL is highest on chunk 5 for every executor and both references,
  which points to a content property.
- The layer-0 gap is consistent with the reference's Q8_0 activation
  quantization (7.5e-3 from that step alone). That consistency is not proof of
  cause.
- The band was frozen to accommodate exactly this class of legitimate
  implementation difference. **No unexplained mismatch remains.**

**Decode-rate discrepancy (not a correctness item).** The recorded 4.09 tokens/s
was traced to run contention.
- Same-work reruns run at 13.4–14.0 tokens/s with identical tokens.
- R2 has a dated correction note.

## Supported matrix (frozen for later regression)

| Dimension | Verified | Allowed but unverified | Not supported |
| --- | --- | --- | --- |
| Model | H2-hashed Gemma 4 26B A4B QAT target, text only | — | MTP assistant, image path (R1 boundary only) |
| Executors | CPU; HIP on gfx1101 | — | Other GPUs or backends |
| Placement | HIP `first_n` with N = 10 and N = 20 | Other N within the VRAM budget | — |
| Batch | 1, 64, 214, 512 tokens | Other sizes up to `max_batch` | — |
| Occupied context | Up to 1,030 positions on the real artifact; sliding-window boundary exact (synthetic) | Up to `max_ctx`, which is bounded by the model's 262,144 and by memory. R2's attention scratch scales with `max_batch × max_ctx` | Any long-context quality claim: Q2 (rv-e08-nhh.2) |
| Decoding | Greedy (argmax) | — | Sampled decoding (S1), speculative verification (S2) |
| KV | Q8_0, ring buffer for sliding layers, linear for global layers | — | Other formats |

**Regression fixtures:**
- the committed synthetic tests (`./scripts/check.sh`);
- the GPU conformance tests (`hip_ops`, ignored by default);
- these private real-artifact controls: KL chunk hashes, the 520-token greedy
  continuation, and layer dumps for the 214-token prompt.

## Outcome

**GO, scoped** to the verified column above.

**What GO makes eligible.** Work that names Gate 3 as a blocker becomes eligible:
K1, S2, M4, Q1 and Q2. Each still needs its own authorization, and MTP and
optimized paths remain deferred until then. Closing the gate activates nothing.

**Limitations.**
- KL coverage is one text domain (1,785 positions).
- Real-artifact context coverage ends at about 1,030 positions.
- Agreement inside the band is not, by itself, proof of correctness. The op-level
  and layer-0 checks carry that weight.

**Next discriminating actions.**
- R3a: attribution and the CPU expert kernel ([roadmap](optimization-roadmap.md)).
- Q2 for long-context correctness.
- S1 for the sampling and verification semantics.
