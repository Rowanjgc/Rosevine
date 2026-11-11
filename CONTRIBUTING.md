# Contributing to Rosevine

Run `./scripts/bootstrap.sh` once, then `./scripts/check.sh` before review.
Use feature branches and a pull request to main. See the
[workflow](docs/development/engineering-workflow.md) and
[privacy policy](docs/development/privacy-policy.md). Contributions retain GPL-3.0.

Keep code clear, small and owned by explicit modules. Prefer direct solutions;
avoid speculative abstractions, unnecessary dependencies and undocumented coupling.
Document invariants and non-obvious behavior. Handle errors at boundaries with
useful context and no private data; never silently accept partial results.
Make allocation, ownership, lifetimes, cleanup and device resource limits explicit.
Unsafe operations and low-level optimizations need documented preconditions,
correctness checks and review proportional to their risk.

**Performance optimizations must preserve explicitly defined correctness and must
be independently measurable.** Define tolerances, reference behavior and failure
conditions before optimization. Report warmup, sample count, units, model and
software revisions, hardware class, compiler flags and methodology. Label failed
and incomplete experiments accurately. Explain regressions and variability;
avoid claims from a single unrepresentative run. Full GPU benchmarks are not
required for trivial documentation changes.

Keep builds reproducible: pin tools/dependencies, document provenance and build
flags, separate downloads from build steps, review licenses and security impact.
Add dependencies only for a demonstrated need. Tool upgrades require checksum/
immutable-pin review and guardrail tests. Do not silently install global packages.
Language-specific formatters, linters, compiler checks, unit tests, sanitizers and
build verification will be added after an accepted language/backend decision.

Test observable behavior and important failure paths. Match verification to risk.
Update affected documentation, examples and acceptance criteria. Consequential
architecture changes require a human-accepted [ADR](docs/adr/README.md).
Security, hooks, workflow permissions, allowlists, dependencies and benchmark
publication require explicit human review; agents cannot self-approve exceptions.
A single maintainer may review and merge their own PR after checks and deliberate
review. No unavailable second reviewer is required. Never auto-merge.

## Definition of Done

- Acceptance criteria met; relevant tests and security checks pass.
- Formatting/static checks pass where applicable; documentation updated.
- Public diff and metadata reviewed; no known unintended privacy disclosure.
- Accepted ADRs respected or explicitly revised with approval.
- Performance claims include reproducible benchmark and correctness evidence.
- No unexplained regression or concealed critical warning.
- Bead status accurately reflects evidence when a tracker is approved.
- Deviations, unrun checks and residual risks are reported explicitly.
