#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
command -v git >/dev/null || { echo 'Install Git using your package manager.' >&2; exit 1; }
test -x /usr/bin/python3 || { echo 'Install python3 using your package manager.' >&2; exit 1; }
existing=$(git config --get core.hooksPath || :)
case "$existing" in ''|.githooks) ;; *) echo 'Existing hooksPath found; review integration manually. No changes made.' >&2; exit 1 ;; esac
# Preserve any existing active default hooks.
for hook in pre-commit pre-push; do
    path=$(git rev-parse --git-path "hooks/$hook")
    if test -f "$path" && test "$existing" != .githooks; then
        echo 'Existing active hook found; review integration manually. No changes made.' >&2; exit 1
    fi
done
/usr/bin/python3 scripts/install-gitleaks.py
/usr/bin/python3 scripts/guardrails.py tool
chmod +x .githooks/pre-commit .githooks/pre-push
git config --local core.hooksPath .githooks
exclude=$(git rev-parse --git-path info/exclude)
/usr/bin/python3 - "$exclude" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text() if p.exists() else ''
for line in ['/.private/', '/.tools/', '/.beads/']:
    if line not in s.splitlines():
        s += '\n' + line + '\n'
p.write_text(s)
PY
printf '%s\n' 'Local commit/push hooks active. Run ./scripts/check.sh before review.'
