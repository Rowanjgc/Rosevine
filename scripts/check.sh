#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 scripts/guardrails.py worktree
/usr/bin/python3 scripts/guardrails.py history
/usr/bin/python3 scripts/quality.py
/usr/bin/python3 -m unittest discover -s tests/guardrails -v
/usr/bin/python3 -m unittest discover -s tests/research -v
# Rust runtime (ADR 0001): format, lints as errors, tests. No registry access: the crate has no
# dependencies and the lockfile is committed.
cargo fmt --manifest-path runtime/Cargo.toml --check
cargo clippy --manifest-path runtime/Cargo.toml --offline --locked --all-targets -- -D warnings
cargo test --manifest-path runtime/Cargo.toml --offline --locked
