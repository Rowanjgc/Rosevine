# Project Zero hardening report

Assessment date: 2026-11-11. Scope: Rosevine Project Zero governance and local
controls, not an inference engine, production router audit or research roadmap.

## Initial state and preservation

The checkout was clean on main, with one reachable commit (`952152a`) containing
README.md and LICENSE. The only remote was the intended public GitHub repository;
only main and its origin tracking refs existed. No project agent instructions,
CI, active hooks or Beads workspace existed. Default hook samples were present.
No configured hooksPath was found. Git, Python, Go, GitHub CLI, br 0.7.6 and emem
were available; Gitleaks and pre-commit were absent. Advisory engineering memory
was recalled without copying its contents into the repository.

Work is on local feature branch `chore/project-zero-guardrails`. Existing README
content is retained with development links added; GPL-3.0 LICENSE is unchanged.
No Rosevine commit, push, merge, history rewrite, credential revocation or remote
setting change was performed. No unrelated project or global Git config changed.

All reachable file trees and commit messages passed credential and project privacy
scans. No tracked sensitive file was found. **Existing public commit metadata
contains a personal email address.** Its value is deliberately omitted. The owner
must review whether that identity was intended to be public and decide whether any
remediation is needed. Removing it from future commits cannot remove existing
copies. History rewriting needs separate authorization. New local commits require
a GitHub noreply identity; bootstrap does not choose or change the owner's identity.

## Implemented controls

- Ignore rules and independent local excludes for private storage, tools and Beads.
- Active repository-local pre-commit and pre-push hooks via `.githooks`.
- Checksum-pinned local Gitleaks 8.30.1 with default credential rules, decoding,
  full redaction and suppressed raw diagnostics. No inline scanner allow comments
  or local secret ignore files are honored. Missing/broken tooling fails closed.
- Full index snapshots at commit; all reachable trees/messages at push, including
  intermediate deleted secrets and annotated tag messages. Direct main updates
  and deletion blocked locally. Unsupported/non-commit tag targets fail closed.
- Independent forbidden-path/artifact, private-path/network/device/email,
  binary/encoding, symlink/submodule and 1 MiB size controls.
- Privacy exceptions limited to exact path/content hash/category with a reviewed
  reason; no credential/path/artifact suppression. Initial allowlist is empty.
- Structured benchmark validator with exact fields, finite numeric checks and
  explicit human review; arbitrary log redaction is intentionally unsupported.
- Engineering standards and Definition of Done, concise agent instructions,
  privacy and Beads policies, ADR template/process, disclosure policy and PR checklist.
- Language-independent CI with read-only permissions, immutable checkout pin,
  full history, no persisted credentials, no privileged PR trigger and no artifacts.
  Monthly Dependabot PR configuration for GitHub Actions; no automatic merge.

`bootstrap.sh` was run successfully and rerun for idempotence; local hooksPath is
`.githooks`, both hooks executable, and local excludes retained. No global packages
were installed. `.tools` is ignored and untracked. No private capture directory or
Beads workspace was created. Bootstrap supports Linux x86_64; other platforms need
a reviewed release checksum. Network failure blocks scanner installation.

## Validation evidence

`./scripts/check.sh` ran successfully: worktree and complete reachable-history
scans, formatting/syntax, local Markdown file references, numbered ADR structure,
agent sections, workflow policy invariants and **18 passing automated tests**.
An initial fixture-root error was corrected before the passing run.

The tests use synthetic data in disposable repositories and a clean Git environment:

- Normal public hardware/source change committed and pushed to a local bare fixture.
- Fake token and private key, auth filename, private path/network/device identifier,
  raw benchmark directory, model artifact, oversized file and symlink rejected.
- Index containing private data rejected even when worktree content was benign.
- Bad ancestor created only in a disposable fixture; deleting its fake token in a
  later commit did not allow push. Receiving bare repository remained unchanged.
- Annotated tag containing a fake token rejected; direct main push rejected.
- New private email identity rejected without printing the identity.
- Synthetic sensitive values absent from scanner output; diagnostics stay generic.
- Missing scanner and scanner runtime failure blocked with actionable guidance.
- Structured benchmark accepts public synthetic record, rejects unknown fields,
  private provenance and non-finite numbers.
- Documentation checker accepts the valid tree, rejects malformed ADR and broken link.

Deliberately bad commits never entered Rosevine history. Benchmark examples are
synthetic test inputs, not performance results. CI-equivalent checks ran locally;
no GitHub Actions run is claimed because nothing was published. Required checks
cannot be confirmed from a workflow file alone.

## GitHub state confirmed by read-only API

| Control | Observed state |
| --- | --- |
| Visibility / default branch | Public / main |
| CLI authentication / repository access | Authenticated, admin capability; broad repo scope |
| Native secret scanning | Enabled |
| Native secret push protection | Enabled |
| main branch protection | Absent (API: branch not protected) |
| Repository rulesets | Empty |
| Dependabot vulnerability alerts | Disabled |
| Dependabot security-update PRs | Disabled |
| Non-provider patterns / validity checks | Disabled |
| Private vulnerability reporting | Disabled |
| Actions enabled | Yes; all actions allowed; SHA pinning not required remotely |
| Default Actions token | Read-only; cannot approve PRs |
| Required guardrails check | Not configured; workflow has not run remotely |
| Outside-contributor workflow approval | Unverified; owner must inspect UI |

The CLI scope permits consequential operations; it was used only for read-only
queries and never to print tokens or authentication identity. Available connector
tools did not expose repository-administration inspection, so the CLI was used.
Remote protection is incomplete despite admin capability: the user explicitly
withheld authorization for remote changes. Follow
[owner setup and verification](github-protections.md) after publication is authorized.
Require PRs and `guardrails` with zero mandatory second reviewers; prevent force
pushes/deletion and ordinary admin bypass; enable alerts/private reporting and
restrict Actions. Confirm exact status context after a real workflow run.

## Remaining risks and handoff

Local hooks can be disabled, replaced or bypassed; CI executes branch code and
cannot prevent initial public disclosure. Human review of security-control changes
is essential. Regexes and recognized-secret rules cannot detect every private
fact or novel credential. Existing public metadata needs owner review. Remote
branch enforcement and several security settings still require explicit owner
action. Native secret scanning protects only supported patterns. Full-history
scans are intentionally conservative and may need measured optimization later.
Documentation checks validate file references, not URL availability or anchors;
workflow checks are invariants, not a general YAML security proof.

No new private content was detected in the resulting source/docs/config changes;
this is evidence from scans and review, not proof of absence. No raw inventories,
logs, traces, downloaded models or task exports were added. The private Beads
storage/export policy is ready; the sanitized task exporter and research roadmap
are explicitly deferred. Architecture choices remain undecided.

Project Zero's local controls and governance are ready for the subsequent local
Beads research-planning handoff. Public publication remains conditional on explicit
authorization, review of existing metadata and the documented remote protections.
No roadmap or inference implementation was started.

Routine commands:

```sh
./scripts/bootstrap.sh
./scripts/check.sh
git diff --check
git diff --cached
```

Before an authorized commit, use a repo-local GitHub noreply identity as described
in the [workflow](engineering-workflow.md). Read the public diff and metadata;
commit only with authorization, and obtain separate authorization to push or merge.
