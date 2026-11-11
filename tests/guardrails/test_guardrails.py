"""Synthetic-only tests; deliberate bad commits live exclusively in temporary repos."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rosevine-test-')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'repo'
        self.repo.mkdir()
        self.env = {'PATH': '/usr/bin:/bin', 'HOME': self.temp.name,
                    'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
                    'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}
        shutil.copytree(ROOT / 'scripts', self.repo / 'scripts')
        shutil.copytree(ROOT / '.githooks', self.repo / '.githooks')
        (self.repo / '.tools').mkdir()
        shutil.copy2(ROOT / '.tools/gitleaks', self.repo / '.tools/gitleaks')
        self.git('init', '-b', 'test-feature')
        self.git('config', 'user.name', 'Synthetic Tester')
        self.git('config', 'user.email', 'tester@example.com')
        self.git('config', 'core.hooksPath', '.githooks')

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.repo, env=self.env, capture_output=True)

    def stage(self, name, content):
        p = self.repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.assertEqual(self.git('add', '--', name).returncode, 0)

    def rejected(self, name, content):
        self.stage(name, content)
        result = self.git('commit', '-m', 'Synthetic rejection test')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(content.strip().encode(), result.stdout + result.stderr)
        if 'gh' + 'p_' in content:
            self.assertNotIn(('gh' + 'p_' + 'AbCdEf0123456789' * 3).encode(), result.stdout + result.stderr)
        self.assertIn(b'Guardrails:', result.stderr)

    def test_normal_commit_and_push(self):
        self.stage('sample.py', 'print("RX 7800 XT, 16 GiB VRAM; Fedora; Gemma 4 26B A4B QAT")\n')
        self.assertEqual(self.git('commit', '-m', 'Synthetic safe change').returncode, 0)
        remote = str(Path(self.temp.name) / 'remote.git')
        self.git('init', '--bare', remote)
        result = self.git('push', remote, 'HEAD:refs/heads/test-feature')
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        result = self.git('push', remote, 'HEAD:refs/heads/main')
        self.assertNotEqual(result.returncode, 0)

    def test_fake_token(self):
        self.rejected('sample.py', 'value = "' + 'gh' + 'p_' + 'AbCdEf0123456789' * 3 + '"\n')

    def test_fake_key(self):
        self.rejected('sample.txt', '-----BEGIN ' + 'PRIVATE KEY-----\n' + 'synthetic-key-material\n')

    def test_sensitive_filename(self):
        self.rejected('.env.local', 'SYNTHETIC_ONLY=true\n')

    def test_private_path(self):
        self.rejected('sample.md', '/ho' + 'me/synthetic-user/research\n')

    def test_raw_benchmark(self):
        self.rejected('outputs/benchmark.json', json.dumps({'host' + 'name': 'synthetic-device'}))

    def test_identifiers_in_ordinary_file(self):
        self.rejected('sample.md', 'host' + 'name=synthetic-device\n')

    def test_private_network(self):
        self.rejected('sample.md', 'endpoint: ' + '192' + '.168.23.45\n')

    def test_model_artifact(self):
        self.rejected('model.gguf', 'SYNTHETIC_MODEL_ONLY\n')

    def test_large_artifact(self):
        self.stage('large.txt', 'x' * (1024 * 1024 + 1))
        self.assertNotEqual(self.git('commit', '-m', 'Synthetic size').returncode, 0)

    def test_staged_snapshot_not_worktree(self):
        content = '/ho' + 'me/synthetic-user/private\n'
        self.stage('sample.md', content)
        (self.repo / 'sample.md').write_text('Public hardware information\n')
        self.assertNotEqual(self.git('commit', '-m', 'Synthetic index test').returncode, 0)

    def test_outgoing_deleted_secret(self):
        content = 'value = "' + 'gh' + 'p_' + 'AbCdEf0123456789' * 3 + '"\n'
        self.stage('sample.py', content)
        # Deliberate bypass ONLY in this throwaway synthetic repository.
        self.assertEqual(self.git('-c', 'core.hooksPath=/dev/null', 'commit', '-m', 'Synthetic bad ancestor').returncode, 0)
        self.stage('sample.py', 'print("public")\n')
        self.assertEqual(self.git('commit', '-m', 'Synthetic deletion').returncode, 0)
        remote = str(Path(self.temp.name) / 'remote.git')
        self.git('init', '--bare', remote)
        result = self.git('push', remote, 'HEAD:refs/heads/test-feature')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(content.encode(), result.stdout + result.stderr)
        self.assertIn(b'Gitleaks detected', result.stderr)
        self.assertFalse(self.git('--git-dir', remote, 'show-ref').stdout)

    def test_missing_and_broken_scanner_fail_closed(self):
        scanner = self.repo / '.tools/gitleaks'
        scanner.unlink()
        self.stage('sample.md', 'Public information\n')
        result = self.git('commit', '-m', 'Synthetic missing tool')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'bootstrap', result.stderr)
        scanner.write_text('#!/bin/sh\nif [ "$1" = version ]; then echo 8.30.1; else exit 2; fi\n')
        scanner.chmod(0o700)
        result = self.git('commit', '-m', 'Synthetic failed tool')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'execution failed', result.stderr)

    def test_symlink(self):
        (self.repo / 'sample').symlink_to('nonexistent')
        self.git('add', 'sample')
        self.assertNotEqual(self.git('commit', '-m', 'Synthetic symlink').returncode, 0)

    def test_annotated_tag_message(self):
        self.stage('sample.py', 'print("public")\n')
        self.assertEqual(self.git('commit', '-m', 'Synthetic safe change').returncode, 0)
        token = 'gh' + 'p_' + 'AbCdEf0123456789' * 3
        self.assertEqual(self.git('tag', '-a', 'synthetic-tag', '-m', token).returncode, 0)
        remote = str(Path(self.temp.name) / 'remote.git')
        self.git('init', '--bare', remote)
        result = self.git('push', remote, 'refs/tags/synthetic-tag')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(token.encode(), result.stdout + result.stderr)

    def test_private_commit_identity(self):
        self.stage('sample.py', 'print("public")\n')
        email = 'synthetic@' + 'private.invalid'
        self.git('config', 'user.email', email)
        result = self.git('commit', '-m', 'Synthetic private identity')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'public email', result.stderr)
        self.assertNotIn(email.encode(), result.stdout + result.stderr)


class QualityTests(unittest.TestCase):
    def test_bad_link_and_malformed_adr(self):
        with tempfile.TemporaryDirectory(prefix='rosevine-doc-test-') as temp:
            repo = Path(temp) / 'repo'
            shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns('.git', '.tools', '__pycache__'))
            env = {'PATH': '/usr/bin:/bin', 'HOME': temp,
                   'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
            subprocess.run(['git', 'init', '-b', 'synthetic'], cwd=repo, env=env, capture_output=True, check=True)
            command = ['/usr/bin/python3', 'scripts/quality.py']
            self.assertEqual(subprocess.run(command, cwd=repo, env=env, capture_output=True).returncode, 0)
            (repo / 'docs/adr/0001-synthetic.md').write_text('# Invalid ADR\nStatus: Unknown\n')
            self.assertNotEqual(subprocess.run(command, cwd=repo, env=env, capture_output=True).returncode, 0)
            (repo / 'docs/adr/0001-synthetic.md').unlink()
            (repo / 'broken.md').write_text('[synthetic](missing-document.md)\n')
            self.assertNotEqual(subprocess.run(command, cwd=repo, env=env, capture_output=True).returncode, 0)


class BenchmarkTests(unittest.TestCase):
    def test_schema_and_privacy(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from benchmark_export import validate
        record = dict(hardware_class='RX 7800 XT', os_version='Fedora',
                      software_revision='a' * 40, compiler_flags='-O3',
                      model='Gemma 4 26B A4B QAT', quantization='QAT',
                      methodology='Synthetic illustration only, not a performance claim',
                      status='incomplete', timing_unit='ms', vram_gib=16,
                      warmup_runs=1, measured_runs=3, median=1.5, p95=2)
        self.assertEqual(validate(record), record)
        with self.assertRaises(ValueError):
            validate(dict(record, extra='private'))
        from guardrails import Violation
        with self.assertRaises(Violation):
            validate(dict(record, compiler_flags='-I/ho' + 'me/synthetic-user/include'))
        with self.assertRaises(ValueError):
            validate(dict(record, median=float('nan')))


if __name__ == '__main__':
    unittest.main()
