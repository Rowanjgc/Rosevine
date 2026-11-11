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
