# Public and private research boundary

Public after review: code, build instructions, ADRs, reproducible experiments,
public references, synthetic fixtures, sanitized tasks and benchmark summaries.
RX 7800 XT, VRAM capacity, Fedora version and the target model are intentionally
public. Publish other hardware facts only when necessary for reproducibility.

Default private: credentials, auth files, local configs, identifying inventories,
unnecessary personal names/emails/paths/hostnames, serial numbers, MAC addresses,
private networks/Wi-Fi/endpoints, environment/shell/process dumps, logs/crashes,
model weights/GGUF/mmproj, personal prompts/generations, raw profiler captures,
benchmark outputs and all unreviewed Beads state. Do not inspect unrelated home
files, browser profiles, credential stores or other projects. Do not copy router
configuration into Rosevine or change the production router.

Prefer a separately created directory outside the checkout with mode 0700 and
files mode 0600 (`umask 077`). No project command collects workstation data.
An optional `.private/` inside the checkout must also have mode 0700; create it
with `umask 077; mkdir -m 700 .private` only if needed. Bootstrap does not create
it. Git ignores and independent snapshot/path checks block it even if force-added.
Do not store private data under an innocuous filename to circumvent policy.

## Capture, review, publish

1. Capture raw measurements outside Git; keep exact data private and access limited.
2. Manually construct a structured candidate containing only the fields below.
   Do not pipe arbitrary terminal output into a sanitizer. Confirm no unique
   identifiers, private paths, prompts or unrelated machine details remain.
3. Validate with `/usr/bin/python3 scripts/benchmark_export.py PRIVATE_CANDIDATE --reviewed`.
   This emits only the validated record; it never changes or reads a raw capture.
   `--reviewed` asserts completed human review, not automated approval.
4. After review, save the output under `docs/benchmarks/NAME.json` (create that
   directory only when needed), review the diff and run `./scripts/check.sh`.
   No command automatically stages, commits or publishes a report.

Required fields (no additional fields): text `hardware_class`, `os_version`,
`software_revision` (full 40-character commit ID), `compiler_flags`, `model`,
`quantization`, `methodology`, `status` (successful/failed/incomplete),
`timing_unit` (ms/s/tokens/s); numeric `vram_gib`, integer `warmup_runs`, integer
`measured_runs`, `median`, `p95`. Values must be finite, nonnegative; measured runs
and capacity positive. Describe sample selection, warmup, timing boundaries,
correctness reference/tolerance and variability in methodology and linked public
experiment definitions. Preserve units and numerical accuracy. Report status
honestly; schema validity does not demonstrate scientific validity.

## Enforcement and exceptions

Hooks scan full staged snapshots and all reachable commit trees/messages before
push, including transient deleted files and tags resolving to commits; annotated tag
messages are scanned too. New commits require a GitHub noreply identity; synthetic
tests use reserved example-domain identities. Existing author metadata still needs
human review. Unknown
non-commit tags, symlinks, submodules, non-UTF-8, binary and >1 MiB files fail closed.
Paths, private IPv4/IPv6 ranges, device markers and personal emails receive extra
checks. Gitleaks scans with default rules, decoding, full redaction, no inline
allow comments or local ignore file. No raw scanner output is printed.

A violation reports only category. Inspect the relevant staged diff privately;
unstage with `git restore --staged -- PATH` (or `git rm --cached -- PATH` for a new
file), move private data outside Git, and rescan. If data entered a local commit,
stop the push and ask the owner for a reviewed remediation; never automatically
rewrite history. For real exposed credentials follow [SECURITY](../../SECURITY.md).

False positives require human review of the exact public content. Privacy-only
exceptions in `scripts/privacy-allowlist.json` bind one path, SHA-256, rule and
written reason; content changes invalidate them. They cannot exempt credentials,
private keys, forbidden paths or binaries. Gitleaks findings need investigation,
not automatic suppression. Broader policy changes need explicit human review.

Hooks can be bypassed by configuration, command flags, another clone or changing
the hook/tool itself. CI and remote protection provide independent layers; none
prove publication is safe. Regexes miss novel, encoded or context-dependent private
data. Review all changes and public commit author/committer identities and messages.
Use an approved public Git identity (such as GitHub's noreply address) via repo-local
configuration; do not copy private identity details into reports. History metadata
is not automatically rewritten or silently exempted.
