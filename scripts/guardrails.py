#!/usr/bin/env python3
"""Fail-closed Git snapshot scanner. Never print filenames or matched contents."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1024 * 1024
DENIED_DIRS = {'.beads', '.private', '.tools', '.ssh', '.aws', '.config', 'models',
               'weights', 'artifacts', 'outputs', 'logs', 'traces', 'profiles', 'raw',
               '.kube', '.docker', '.gnupg'}
DENIED_SUFFIXES = {'.gguf', '.safetensors', '.pt', '.pth', '.bin', '.onnx', '.mmproj',
                   '.log', '.dump', '.core', '.trace', '.db', '.sqlite', '.pem', '.key',
                   '.p12', '.pfx', '.prof', '.perf', '.nsys-rep', '.csv', '.jsonl', '.zip',
                   '.gz', '.tar', '.pcap', '.har'}
PATTERNS = {
    'personal-path': r'(?:/(?:home|Users)/[^\s/"\']+|[A-Za-z]:[\\/]Users[\\/][^\s\\/]+)',
    'private-network': r'\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|(?:fc|fd)[0-9a-f]{2}:[0-9a-f:]+|fe80:[0-9a-f:]+)\b',
    'device-identifier': r'(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b|\b(?:serial(?:_number)?|hostname|ssid|username)\s*[:=]\s*["\']?[\w.-]+',
    'personal-email': r'\b[A-Za-z0-9_.+-]+@(?!example\.(?:com|org|net)\b|users\.noreply\.github\.com\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'private-key': r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----',
    'credential-assignment': r'(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["\']?[A-Za-z0-9_+/=-]{12,}',
}


class Violation(Exception):
    pass


def git(*args):
    p = subprocess.run(['git', *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise Violation('Git inspection failed; check repository/object availability. No scan was skipped.')
    return p.stdout


def scanner():
    executable = ROOT / '.tools/gitleaks'
    if not executable.is_file():
        raise Violation('Gitleaks missing. Run ./scripts/bootstrap.sh; publication is blocked.')
    p = subprocess.run([str(executable), 'version'], capture_output=True)
    if p.returncode or p.stdout.strip() != b'8.30.1':
        raise Violation('Gitleaks failed or version differs. Re-run reviewed bootstrap; do not bypass.')
    return str(executable)


def exceptions():
    # Exact content hash + path + rule only; never suppress Gitleaks or private keys.
    data = json.loads((ROOT / 'scripts/privacy-allowlist.json').read_text())
    if not isinstance(data, list):
        raise Violation('Malformed privacy allowlist.')
    for row in data:
        if set(row) != {'path', 'sha256', 'rule', 'reason'} or not row['reason'].strip() or row['rule'] not in {'personal-email', 'personal-path', 'private-network', 'device-identifier'}:
            raise Violation('Malformed or overbroad privacy exception.')
    return data


def inspect(path, data, mode='100644', allow=()):
    p = PurePosixPath(path)
    lower = path.lower()
    if mode not in ('100644', '100755'):
        raise Violation('Symlink, submodule or unsupported file type rejected.')
    if p.is_absolute() or '..' in p.parts or '\\' in path:
        raise Violation('Unsafe repository path rejected.')
    if p.name.lower() in {'.netrc', '.npmrc', '.pypirc', '.git-credentials', 'auth.json',
                           '.bash_history', '.zsh_history', '.python_history'}:
        raise Violation('Authentication/configuration/history filename rejected. Keep it outside Git.')
    if any(x.lower() in DENIED_DIRS for x in p.parts) or p.suffix.lower() in DENIED_SUFFIXES or re.search(r'(?i)(?:^|/)(?:\.env[^/]*|id_(?:rsa|ed25519)[^/]*|credentials[^/]*|.*\.local(?:\.[^/]*)?|.*(?:inventory|crash|benchmark[-_]raw|raw[-_]benchmark|mmproj|rocprof)[^/]*)$', path):
        raise Violation('Private filename or generated artifact rejected. Move raw data outside the checkout; publish only reviewed structured summaries.')
    if len(data) > LIMIT or b'\0' in data:
        raise Violation('Large or binary artifact rejected (1 MiB maximum). Store outside Git.')
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError:
        raise Violation('Non-UTF-8 content rejected; review artifact storage.') from None
    for rule, pattern in PATTERNS.items():
        if re.search(pattern, content) or re.search(pattern, path):
            digest = hashlib.sha256(data).hexdigest()
            if any(r['path'] == path and r['sha256'] == digest and r['rule'] == rule for r in allow):
                continue
            raise Violation('Privacy check rejected category ' + rule + '. Remove private data; inspect locally without copying it into reports.')
    return content


def scan(entries):
    executable = scanner()
    allow = exceptions()
    with tempfile.TemporaryDirectory(prefix='rosevine-scan-') as temp:
        base = Path(temp)
        source = base / 'source'
        source.mkdir(mode=0o700)
        config = base / 'rules.toml'
        config.write_text('[extend]\nuseDefault = true\n')
        ignore = base / 'empty-ignore'
        ignore.write_text('')
        for i, (path, data, mode) in enumerate(entries):
            inspect(path, data, mode, allow)
            # Numeric names prevent malicious filenames from appearing in scanner diagnostics.
            (source / str(i)).write_bytes(data)
        p = subprocess.run([executable, 'dir', str(source), '--config', str(config),
                            '--gitleaks-ignore-path', str(ignore), '--ignore-gitleaks-allow',
                            '--redact=100', '--no-banner', '--exit-code', '42'], capture_output=True)
        if p.returncode == 42:
            raise Violation('Gitleaks detected a credential. Remove it and review affected commits locally. If real/exposed, rotate it; do not bypass.')
        if p.returncode:
            raise Violation('Gitleaks execution failed. Run bootstrap and verify the pinned scanner; publication blocked.')


def tree(ref):
    for record in git('ls-tree', '-rz', ref).split(b'\0'):
        if not record:
            continue
        meta, name = record.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        data = git('cat-file', '-p', oid) if kind == 'blob' else b''
        yield name.decode('utf-8'), data, mode


def history(refs=None):
    if (Path(git('rev-parse', '--git-path', 'shallow').decode().strip())).exists():
        raise Violation('Shallow history rejected. Fetch complete history before scanning.')
    # Annotated tag messages are public independently of their target trees.
    for row in git('for-each-ref', '--format=%(objecttype) %(objectname)', 'refs/tags').decode().splitlines():
        kind, oid = row.split()
        if kind == 'tag':
            message = git('cat-file', '-p', oid).split(b'\n\n', 1)
            scan([('tag-message', message[-1], '100644')])
    commits = git('rev-list', *(refs or ['--all'])).decode().splitlines()
    for commit in commits:
        scan(tree(commit))
        # Messages are public too. Author identities require a separate human review.
        scan([('commit-message', git('show', '-s', '--format=%B', commit), '100644')])


def main():
    os.chdir(git('rev-parse', '--show-toplevel').decode().strip())
    mode = sys.argv[1]
    if mode == 'tool':
        scanner()
    elif mode == 'staged':
        for role in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
            identity = git('var', role).decode()
            email = re.search(r'<([^>]+)>', identity)
            if not email or not re.fullmatch(r'[A-Za-z0-9_.+-]+@(?:users\.noreply\.github\.com|example\.(?:com|org|net))', email[1]):
                raise Violation('Commit identity requires an approved public email. Configure a GitHub noreply email with git config --local user.email; no identity value was printed.')
        entries = []
        for row in git('ls-files', '--stage', '-z').split(b'\0'):
            if row:
                meta, name = row.split(b'\t', 1)
                perm, oid, stage = meta.decode().split()
                if stage != '0':
                    raise Violation('Unmerged index rejected; resolve conflicts first.')
                entries.append((name.decode(), git('cat-file', '-p', oid), perm))
        scan(entries)
    elif mode == 'worktree':
        entries = []
        for name in git('ls-files', '--cached', '--others', '--exclude-standard', '-z').split(b'\0'):
            if name:
                path = Path(os.fsdecode(name))
                if path.is_symlink():
                    raise Violation('Worktree symlink rejected.')
                if path.exists():
                    entries.append((str(path), path.read_bytes(), '100644'))
        scan(entries)
    elif mode == 'history':
        history()
    elif mode == 'push':
        refs = []
        for row in sys.stdin:
            fields = row.split()
            if len(fields) != 4:
                raise Violation('Malformed pre-push input; publication blocked.')
            if fields[2] == 'refs/heads/main':
                raise Violation('Direct updates/deletion of main blocked. Publish a feature branch and use a reviewed PR.')
            if set(fields[1]) != {'0'}:
                # Resolve annotated tags to commits; reject non-commit tags rather than skip.
                refs.append(git('rev-parse', '--verify', fields[1] + '^{commit}').decode().strip())
        history(['--all', *refs])
    else:
        raise Violation('Unknown scan mode; no checks skipped.')
    print('Guardrails: ' + mode + ' passed.')


if __name__ == '__main__':
    try:
        main()
    except (Violation, OSError, ValueError, subprocess.SubprocessError):
        # Exception details from libraries can include secret paths or tool output.
        error = sys.exc_info()[1]
        print('Guardrails: ' + (str(error) if isinstance(error, Violation) else 'Tool/configuration failure. Inspect setup locally; run bootstrap. No checks skipped.'), file=sys.stderr)
        sys.exit(1)
