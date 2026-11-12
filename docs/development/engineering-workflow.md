# Development workflow

Prerequisites: Git, `/usr/bin/python3` (3.10+), POSIX sh and Linux x86_64. Fedora
owners may install missing packages with `sudo dnf install git python3`; setup
never runs a system package manager. Bootstrap requires HTTPS access to the pinned
Gitleaks release and checks its SHA-256. No virtualenv or pre-commit package needed.
Other platforms need a reviewed installer/checksum change before publication.

```sh
./scripts/bootstrap.sh
./scripts/check.sh
git switch -c topic/short-description
# Edit, inspect the public diff and commit metadata; stage selected paths.
git diff --check
git diff --cached
./scripts/check.sh
# Commit only when authorized; push/publish/merge require explicit authorization.
```

Bootstrap is idempotent, installs only `.tools/gitleaks`, configures this repo's
`core.hooksPath=.githooks` and adds local excludes without overwriting existing
hooks or excludes. It refuses conflicting hooks. Re-run after cloning; hooks are
not automatically enabled by Git clones. Verify with `git config --local --get
core.hooksPath` and `/usr/bin/python3 scripts/guardrails.py tool`.
No global Git configuration changes. Before committing, configure a GitHub noreply
identity with `git config --local user.email YOUR_GITHUB_NOREPLY_ADDRESS`; hooks
reject other real email domains for new commits (example domains are fixture-only).
Use the exact address shown in GitHub email settings; do not guess it. A normal commit invokes staged checks and
push invokes history checks; direct updates/deletion of main are blocked locally.
`check.sh` scans the worktree and reachable history, checks text formatting,
Python/shell syntax, relative Markdown file links, ADR structure, agent headings,
workflow invariants and synthetic guardrail tests. It needs no GPU or downloaded
model. Ignored private files are not opened by worktree checks.

Use the [standards and Definition of Done](../../CONTRIBUTING.md). Identify criteria
and accepted ADRs before implementation. Track evidence and honest status. Ask for
human approval of consequential decisions, guardrail changes and publication.
Existing user authorization covers routine implementation and bounded research
within its approved resource scope; do not request it again for each diagnostic
command. See the [rule audit](rule-audit.md) for diagnostic versus benchmark gates.
Do not weaken checks for convenience. Tool failure blocks publication; bootstrap
or investigate locally. No security bypass flag exists in project commands.

## Tool updates

Gitleaks 8.30.1 is pinned in the installer and scanner version check. Review release
notes and official checksums, change both pins, verify the downloaded archive's
checksum, and rerun synthetic tests before approval. The installer checksum is
fixed in source; it does not trust a downloaded checksum at each installation.
The locally cached binary hash is an idempotence check, not a tamper-proof trust
root. Review `.tools` integrity if the workstation is compromised.
Checkout is pinned to the immutable v4.2.2 commit; Dependabot proposes monthly
GitHub Actions updates. Review upstream changes and new immutable SHAs; do not
auto-merge. See [GitHub setup](github-protections.md) for remote controls and official
references. pre-commit was evaluated but adds no necessary layer for two direct
hooks; maintained Gitleaks handles credential detection while Python adds project
privacy/artifact policy. New implementation languages extend, not replace, CI.
