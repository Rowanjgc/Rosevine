#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 scripts/guardrails.py worktree
/usr/bin/python3 scripts/guardrails.py history
/usr/bin/python3 scripts/quality.py
/usr/bin/python3 -m unittest discover -s tests/guardrails -v
/usr/bin/python3 -m unittest discover -s tests/research -v
