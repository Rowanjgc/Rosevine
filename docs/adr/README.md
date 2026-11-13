# Architecture decisions

Copy [template.md](template.md) to `NNNN-short-topic.md` with the next unused
four-digit number. Statuses: Proposed, Accepted, Superseded, Rejected.
Proposals record options and evidence with links, uncertainty and tradeoffs.
Only explicit human review can mark an ADR Accepted. Record the reviewer role,
date and PR reference without unnecessary personal identifiers. Rejected proposals
retain their reasoning. Superseding decisions link both records; do not silently
rewrite an accepted decision's meaning. Review changes through a PR.

[0001](0001-primary-backend-and-runtime-strategy.md) (Accepted): HIP primary backend,
Rust host over a C ABI. Kernel specialization, model representation and expert
memory placement remain undecided.
`quality.py` checks numbered ADR status and required sections, plus local file
links. It cannot validate scientific evidence, acceptance authority or URL health.
