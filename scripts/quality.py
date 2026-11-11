#!/usr/bin/env python3
"""Language-independent, mechanically checkable repository conventions."""
import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['AGENTS.md', 'CONTRIBUTING.md', 'SECURITY.md', 'LICENSE',
            'docs/development/privacy-policy.md', 'docs/development/beads-policy.md',
            'docs/development/engineering-workflow.md', 'docs/development/hardening-report.md']
ADR_HEADINGS = ['Context and problem', 'Options considered', 'Evidence', 'Tradeoffs',
                'Decision', 'Consequences', 'Revisit conditions']


def main():
    errors = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append('Missing required governance document.')
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).split(b'\0')
    for raw in files:
        if not raw:
            continue
        path = ROOT / raw.decode()
        if not path.is_file():
            continue
        data = path.read_text()
        if not data.endswith('\n') or any(line.rstrip() != line for line in data.splitlines()) or '\r' in data:
            errors.append('Text formatting: use LF, final newline, no trailing whitespace.')
        if path.suffix == '.py':
            ast.parse(data)
        if path.suffix == '.sh' or path.parent.name == '.githooks':
            if subprocess.run(['sh', '-n', str(path)], capture_output=True).returncode:
                errors.append('Shell syntax invalid.')
        if path.suffix == '.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', data):
                if re.match(r'^[a-z]+:', target) or target.startswith('#'):
                    continue
                target = target.split('#')[0]
                if not (path.parent / target).exists():
                    errors.append('Broken relative documentation link.')
        if path.parent == ROOT / 'docs/adr' and re.match(r'\d{4}-.+\.md$', path.name):
            if not re.search(r'^Status: (Proposed|Accepted|Superseded|Rejected)$', data, re.M):
                errors.append('ADR status invalid.')
            for heading in ADR_HEADINGS:
                if '## ' + heading not in data:
                    errors.append('ADR section missing.')
        if path.parent == ROOT / 'docs/benchmarks' and path.suffix == '.json':
            from benchmark_export import validate
            import json
            validate(json.loads(data))
    agent = (ROOT / 'AGENTS.md').read_text()
    for heading in ['Identity', 'Authority', 'Workflow', 'Restrictions']:
        if '## ' + heading not in agent:
            errors.append('Agent instruction structure invalid.')
    for path in (ROOT / '.github/workflows').glob('*.yml'):
        text = path.read_text()
        if 'permissions:\n  contents: read\n' not in text or 'persist-credentials: false' not in text or 'fetch-depth: 0' not in text:
            errors.append('Workflow security invariant missing.')
        if any(term in text for term in ['pull_request_target', 'secrets.', 'write-all', 'upload-artifact', 'continue-on-error']):
            errors.append('Workflow requires security review: prohibited elevated/unsafe pattern.')
        for action in re.findall(r'uses:\s*(\S+)', text):
            if not re.fullmatch(r'[^@]+@[0-9a-f]{40}', action):
                errors.append('Action must be pinned to an immutable commit.')
    if errors:
        raise ValueError('; '.join(sorted(set(errors))))
    print('Quality: formatting, syntax, local file links, ADRs, agent structure and workflow invariants passed.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, SyntaxError, subprocess.SubprocessError):
        print('Quality validation failed. Check formatting, required documents, ADR sections, links and workflow policy locally.', file=sys.stderr)
        sys.exit(1)
