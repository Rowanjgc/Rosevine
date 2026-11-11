# Beads integration policy

Project Zero inspects tooling only: no `.beads` initialization, issues, enrollment,
research roadmap or GitHub synchronization. Installed `br` was verified as 0.7.6.
Before later adoption, recheck CLI help and get approval for the storage workflow.

Beads Rust uses `.beads/` with SQLite working data, journals/WAL/SHM, backups,
configuration, and portable `issues.jsonl` exports. Mutations normally auto-flush
JSONL; commands can auto-import changed exports. Even JSONL is not safe by default:
titles, descriptions, comments, actor/assignee identities, timestamps, references,
audit fields and free text may reveal personal data. Machine-local database
metadata can contain workstation paths. Inspect schema and actual export privately
before approving publication. Never commit the database, journal or raw export.

Rosevine ignores and independently rejects the entire `.beads/` tree and generic
JSONL artifacts. Keep later local storage private (0700 directory, 0600 files,
`umask 077`). Do not assume upstream ignore rules protect metadata. Do not move an
unreviewed export outside `.beads` to evade checks. Project policy overrides the
general habit of committing `.beads/issues.jsonl`.

When adoption is authorized: read the local br instructions/skill, use `br where
--json` to verify scope, then `br ready --json`, `br show ID --json`, and inspect
dependencies before claiming with `br update ID --claim --actor PUBLIC_ROLE --json`.
Respect other agents' claims. `br dep add A B` means A depends on B; parent-child
hierarchy is separate. Keep the dependency graph acyclic. Later tasks should link
accepted ADRs and include acceptance criteria, verification evidence and honest
status. Close significant work with Root cause / Resolution / Verification /
Engineering takeaway. Do not invent a lesson or close unmet criteria.

`br sync --status --json` inspects sync; `br sync --flush-only --json` exports
locally and `--import-only` imports. Neither is a publication approval. Do not run
bare sync, force migrations/recovery, or synchronize to public GitHub issues.
Do not initialize a home-wide tracker or modify another project's state.

A future public export needs a reviewed allowlisted schema (public task ID, title,
criteria, status, dependency IDs and ADR links), manual review of every field,
privacy/secret scans, and a narrowly approved file policy change. Until that exists,
all raw/local task data stays private. The benchmark exporter is not a Beads exporter.
