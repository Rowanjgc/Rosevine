# Gate 0 environment and reference review

Date: 2026-11-11
Bead: rv-e01-898.3
Status: Reviewed
Outcome: NO-GO for baseline performance experiments
Reviewer: epic1-research (agent evidence/criteria review; not human approval)
Downstream activation: none

## Evidence and criteria

| Requirement | Current evidence | Assessment |
| --- | --- | --- |
| Hardware inventory, device selection and access gaps | [H1 report](environment-characterization.md); private environment/API/access manifests | Characterization complete; no backend or calibrated-counter claim |
| Every descriptor and byte range accounted for | [H2 evidence](epic-1-evidence.md); independent reader plus exact shape/type contract | 658 target, 49 assistant, 356 projector; zero range/contract mismatches |
| Exact local artifact identity | Full SHA-256 and sizes in private manifests; public hashes in H2 | Established for local files; upstream conversion lineage remains unknown |
| Reference tool mismatch resolved | Adjacent tools report same revision/build, resolve hashed installed libraries | Old system bench excluded; full rebuild equivalence unproven |
| Template/token policy | Direct and common-library text fixtures and direct image-marker fixtures | True-mode finite text parity observed; false-mode mismatch and common media rejection retained |
| Sampler policy | Presets match primary guidance; pinned source documents defaults/overrides | Historical effective request seeds/overrides unknown; new manifests must freeze them |
| Safe short-test budget | H1 proposes one sequence, 4096 context, 32 outputs, 60 s timeout, 5 s grace, 2 GiB RAM/1 GiB VRAM reserves | Concrete proposal exists; supervisor, thermal/display limits and idle window unvalidated |
| No unsupported success claim | No inference, benchmark, model conversion or production changes | No throughput, correctness, upstream-equivalence or backend-support conclusion |

## Decision and restrictions

Record NO-GO: characterization can finish with explicit negative findings,
but the available evidence does not support starting baseline performance runs.
The local identities and thinking-enabled text fixtures provide a candidate
reference, not certification of every historical or multimodal request. This
proposal accepts no runtime language, backend, representation or placement ADR.
No dependent Bead is automatically activated by closing Gate 0.

Do not modify router settings, install dependencies, change driver/service/power
settings, download weights, run long GPU trials, publish captures or commit/push
as part of this gate. Local evidence and Beads remain private. This negative research decision does
not accept an ADR, approve an unvalidated run budget or authorize a workload.

## Unknown owners, impact and next actions

| Unknown | Owner | Impact / next discriminating action |
| --- | --- | --- |
| Installed data/library versus source build equivalence | Reference-identity follow-up | Blocks reproducible-source-build claim; compare remaining ELF sections/library provenance or define a separately named reviewed rebuilt reference |
| Upstream conversion/QAT lineage | Reference-identity follow-up | Blocks upstream fidelity claim; obtain pinned distributor conversion records and hashes; preserve exact local reference meanwhile |
| Request overrides/seed/stop policy | B1/B2 | Blocks historical-request reproduction; define new synthetic request manifests with full effective policy rather than guessing history |
| Server media preprocessing | Later image contract study | Blocks server-image equivalence; use actual licensed image fixture and isolated media path after required review |
| Thinking-disabled template mismatch | Reference-identity follow-up | Blocks claiming template equivalence in that mode; retain distinct reference strata or propose a reviewed correction without editing production |
| Telemetry calibration, overhead, cancellation | B1/backend evaluation | Blocks reliable experimental interpretation; follow H1 synthetic-supervisor tests and paired collector on/off validation |
| Thermal/display reserves and idle window | Maintainer | Blocks GPU pilot; review H1 budget with sensor-specific limits before a concrete isolated invocation |
| RAM timings/bandwidth and GPU counter support | Environment/backend follow-up | Blocks timing/bandwidth/counter conclusions; explicit unknowns do not invalidate descriptor inspection |

## When to re-evaluate

Revisit this NO-GO when the evidence changes; there is no calendar wait. The later
[bounded CPU diagnostic](epic-2-pilot-review.md) and two-position logit smoke capture
show that a small CPU-requested run can load/evaluate tokens and that the capture
path works. They do not satisfy the performance gate: they are too small to measure
the target workloads, do not exercise the GPU, and do not validate observer effect.

Before proposing GO for baseline performance work, B1 should show the collector's
p95 cycle cost below 1 ms and a paired collector-off/on end-to-end change whose
95% interval lies within +/-1%, with cancellation and cleanup evidence. The run
manifest must freeze exact input token IDs, sampler/stop policy, executable and
resolved library identities; any unresolved upstream or source-build lineage must
be called out as a scope limit rather than treated as equivalence. The selected
device, sensors, resource reserves and run envelope must be validated for the
specific GPU experiment, with an approved idle window and concrete invocation.
Unsupported counters may remain unknown only when the planned claims do not depend
on them. Then update this gate with the new evidence, restrictions and downstream
activation decision before starting B3.

This is a readiness checklist, not a GO decision or run authorization. Upstream
conversion lineage, image preprocessing and other unrelated fidelity gaps need
only be resolved before making the corresponding claims; a restricted text-only
baseline can state those limits explicitly.

## Review record

Reviewed by epic1-research on 2026-11-11 against H1/H2 criteria, the baseline,
experiment protocol and current Beads dependencies. No human acceptance is claimed.
The roadmap requires a reviewed gate record; its explicit human-acceptance rule
applies to ADRs. This NO-GO adopts no architecture and enables no workload.
The earlier request for human approval of the negative record was an unnecessary
interpretation, not a repository requirement.

Review checked the independent parser and shape contract against the documented
reference dimensions, the private artifact/build/template results against public
claims, and failed media cases against the stated limitations. All 382 indexed
private evidence files (including the selected-preset capture outside the checkout) match recorded hashes/sizes and restrictive file modes.
The standard `./scripts/check.sh` now includes the six research tests alongside
18 guardrail tests; all 24 pass. Formatting, syntax, links, worktree/history
privacy scans and dependency-cycle checks pass. The public changes contain no
raw captures or copied private router configuration; exact paths remain private.
Existing uncommitted Project Zero files were preserved. No commit/push is made.

Acceptance audit: H1 inventory/access criteria and H2 artifact/tool/template
investigation criteria are met with explicit gaps. Gate 0 maps every remaining
unknown to an owner and impact, records a concrete conservative short-test budget
proposal and chooses NO-GO because that proposal is not validated. The decision
and downstream activation (none) are explicit. Negative findings satisfy the
characterization outcome; they do not satisfy the preconditions for performance
experiments. The only directly dependent experiment, B3, must remain deferred.
Future GO requires updated evidence, a validated run envelope and the applicable
human run-window approval. The candidate reference proposal remains unaccepted
for performance execution.

## Updated outcome 2026-11-12: GO (restricted)

Outcome: GO for text-only Vulkan reference baselines (B3) within the envelope below.
Reviewer: agent evidence review; GPU run window, VRAM reserve and thermal stops
approved by the maintainer on 2026-11-12. No ADR, backend or runtime is adopted.

| Readiness item | Evidence | Assessment |
| --- | --- | --- |
| Run tooling | [gpu_campaign.py](../scripts/gpu_campaign.py): private `llama-server` per session on a UNIX socket (no TCP port, production router untouched), token-exact greedy `/completion` requests, separate model-load deadline, synthetic tests | Ready |
| Identity | Target, MTP assistant, projector, server binary and Vulkan/llama libraries re-hashed at campaign start; target/assistant/projector match Epic 1 | Local identities fixed; upstream lineage and source-build equivalence remain scope limits |
| GPU smoke | Fast preset, 1900-token prompt: load 7.0 s, prefill 2256 tokens/s, 32-token decode 142.6 tokens/s, MTP 22/25 accepted, minimum free VRAM 0.52 GiB, confirmed cleanup | Allocation and execution work |
| Cancellation | One run stopped on the VRAM reserve and one on the GPU-junction stop; both terminated the server and confirmed cleanup | Stops are live, not just tested synthetically |
| Collector effect | Three ABBA pairs, 24 matched 512-token requests: decode rate +0.07% (95% CI −0.14% to +0.13%), wall time +0.01% (−0.11% to +0.08%) with 100 ms versus 1 s collection | Meets the ±1% paired criterion. Request-level bootstrap over only three independent session pairs |
| Collector cycle cost | p95 round trip 9.8 ms (1484 cycles) | The 1 ms host-cycle target is still unmet. It is retained as a quality observation because the end-to-end effect it protects is measured directly above |

Envelope approved by the maintainer after live findings (the thermal and VRAM
items below were further revised the same day; see the
[GPU baseline revision](development/rule-audit.md#gpu-baseline-revision-2026-11-12)):

- Free-VRAM stop 256 MiB. The fast, default and 128k presets cannot keep 1 GiB
  free: the installed fit tool needs ≥5 CPU MoE layers for a 1 GiB margin, and
  explicit `--n-gpu-layers all` disables fitting.
- GPU stops: junction 100 °C, memory 98 °C, edge 85 °C. The driver reports
  110/108/100 °C critical. Normal decode drives junction to 82–85 °C within a
  minute, so the earlier 85 °C stop cancelled routine sessions. CPU stops stay at 90 °C.
- Always sample at 100 ms: 1 s sampling under-reported junction peaks by about
  7 °C relative to 100 ms sessions.
- RAM reserve 2 GiB; sustained-paging rule unchanged; disk reserve 2 GiB.
- Any growth over 512 MiB in a foreign `llama-server` process (production router
  or its sleeping model children) cancels the session. Sessions start only when
  the GPU is idle and no foreign server holds a model.
- One attempt per session, no automatic retry; failed attempts are retained.

Scope limits carried into B3: text-only token-exact requests (chat rendering,
reasoning parsing and router-only keys are not exercised); the configured
projector is loaded as in the presets, plus one named text-only control;
greedy decoding only; historical sampled-request overrides remain unknown.
Downstream activation: B3 (rv-e02-22u.3) under the minimal Gate 1 scope.
