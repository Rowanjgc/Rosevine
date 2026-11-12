"""Real local pipe workers: deadlines, malformed data, flood caps and cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_core import supervise, terminate_group
from collector_worker import ProcessCollector


def command(body):
    return [sys.executable, '-u', '-c',
            'import sys; sys.path.insert(0, ' + repr(str(ROOT / 'scripts')) + '); ' + body]


def serving(body):
    return command('from collector_worker import serve; ' + body)


class CollectorWorkerTests(unittest.TestCase):
    def test_startup_allowance_does_not_extend_running_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'startup'
            collector = serving('import time; time.sleep(.15); serve(lambda *args: {})')
            result = supervise([sys.executable, '-c', 'import time; time.sleep(.12)'],
                               path, timeout_s=2, telemetry_command=collector,
                               telemetry_timeout_s=.05, telemetry_startup_timeout_s=.7)
            self.assertEqual(result['status'], 'successful')
            path = Path(temp) / 'running'
            collector = serving('import time; serve(lambda pid, *args: (time.sleep(.2) if pid else {}))')
            result = supervise([sys.executable, '-c', 'import time; time.sleep(2)'],
                               path, timeout_s=2, grace_s=.01, cadence_s=.01,
                               telemetry_command=collector, telemetry_timeout_s=.05,
                               telemetry_startup_timeout_s=.7)
            self.assertEqual(result['reason'], 'collector response timeout')
            self.assertFalse(result['workload_cleanup']['pending'])

    def test_valid_preflight_and_periodic_rejection_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trial'
            collector = serving('serve(lambda pid, elapsed, size: {"rejection": "synthetic budget breach" if pid else None})')
            result = supervise([sys.executable, '-c', 'import time; time.sleep(5)'], path,
                               timeout_s=2, grace_s=.01, cadence_s=.01,
                               telemetry_command=collector, telemetry_timeout_s=.5)
            self.assertTrue(result['launched'])
            self.assertEqual(result['reason'], 'synthetic budget breach')
            rows = [json.loads(line) for line in (path / 'telemetry.jsonl').read_text().splitlines()]
            self.assertEqual([r['stage'] for r in rows], ['preflight', 'running'])
            self.assertEqual([r['collector_transport']['request_id'] for r in rows], [1, 2])
            for row in rows:
                t = row['collector_transport']
                self.assertLessEqual(t['requested_ns'], t['acquisition_started_ns'])
                self.assertLessEqual(t['acquisition_started_ns'], t['acquisition_completed_ns'])
                self.assertLessEqual(t['acquisition_completed_ns'], t['received_ns'])
            self.assertFalse(result['collector']['cleanup']['pending'])
            index = json.loads((path / 'checksums.json').read_text())
            self.assertIn('collector-result.json', index)
            self.assertIn('collector-stderr.txt', index)
            for file in path.iterdir():
                self.assertEqual(file.stat().st_mode & 0o777, 0o600)

    def test_measurement_channel_is_separate_from_invariant_safety_channel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'requires an active safety'):
                supervise([sys.executable, '-c', 'pass'], root / 'invalid', timeout_s=1,
                          measurement_telemetry_command=serving('serve(lambda *_: {"x": 1})'))
            path = root / 'both'
            result = supervise([sys.executable, '-c', 'import time; time.sleep(.03)'], path,
                               timeout_s=1, cadence_s=.005,
                               telemetry=lambda *_: {'safety': 'active'},
                               measurement_telemetry_command=serving('serve(lambda *_: {"sample": 1})'),
                               measurement_telemetry_timeout_s=.5)
            self.assertEqual(result['status'], 'successful')
            self.assertTrue(result['measurement_telemetry_enabled'])
            self.assertGreater(result['telemetry_bytes'], 0)
            self.assertTrue(result['measurement_collector'])
            self.assertTrue((path / 'telemetry.jsonl').is_file())
            self.assertTrue((path / 'measurement-telemetry.jsonl').is_file())
            from verify_capture_bundle import verify_bundle
            verified = verify_bundle(path)
            self.assertEqual(verified['status'], 'internally_consistent')
            index = json.loads((path / 'checksums.json').read_text())
            self.assertIn('measurement-collector-result.json', index)

    def test_safety_and_measurement_records_share_one_telemetry_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'shared-cap'
            limit = 512
            result = supervise([sys.executable, '-c', 'import time; time.sleep(.1)'], path,
                               timeout_s=1, cadence_s=.001,
                               max_telemetry_bytes=limit,
                               telemetry=lambda *_: {'fixed': 1},
                               measurement_telemetry_command=serving('serve(lambda *_: {"sample": 1})'),
                               measurement_telemetry_timeout_s=.5)
            self.assertIn(result['reason'], ('telemetry capture limit reached',
                                              'measurement telemetry capture limit reached'))
            self.assertLessEqual(result['telemetry_bytes'] + result['measurement_telemetry_bytes'], limit)
            self.assertFalse(result['workload_cleanup']['pending'])

    def test_hung_preflight_never_launches_workload_and_ignoring_term_is_killed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'hung'
            body = ('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                    'sys.stderr.write("collector-ready\\n"); sys.stderr.flush(); '
                    'serve(lambda *args: time.sleep(5))')
            before = time.monotonic()
            result = supervise([sys.executable, '-c', 'print("must not run")'], path,
                               timeout_s=2, telemetry_command=serving(body), telemetry_timeout_s=.5)
            self.assertLess(time.monotonic() - before, 1.5)
            self.assertFalse(result['launched'])
            self.assertEqual(result['reason'], 'collector response timeout')
            self.assertIn('collector-ready', (path / 'collector-stderr.txt').read_text())
            self.assertEqual(result['collector']['cleanup']['returncode'], -signal.SIGKILL)
            self.assertFalse(result['collector']['cleanup']['pending'])

    def test_hung_running_sample_cancels_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'running-hang'
            collector = serving('import time; serve(lambda pid, elapsed, size: (time.sleep(5) if pid else {}))')
            before = time.monotonic()
            result = supervise([sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(5)'],
                               path, timeout_s=2, grace_s=.01, cadence_s=.05,
                               telemetry_command=collector, telemetry_timeout_s=.3)
            self.assertLess(time.monotonic() - before, 1.5)
            self.assertTrue(result['launched'])
            self.assertEqual(result['reason'], 'collector response timeout')
            self.assertIn('started', (path / 'stdout.txt').read_text())
            self.assertFalse(result['workload_cleanup']['pending'])
            self.assertFalse(result['collector']['cleanup']['pending'])

    def test_malformed_and_stale_responses_fail_closed(self):
        bodies = [
            'sys.stdin.readline(); print("not-json", flush=True); import time; time.sleep(5)',
            'import json,time; r=json.loads(sys.stdin.readline()); t=time.monotonic_ns(); '
            'print(json.dumps(dict(request_id=r["request_id"]-1, started_ns=t, completed_ns=t, snapshot={})), flush=True); time.sleep(5)',
            'import json,time; r=json.loads(sys.stdin.readline()); '
            'print(json.dumps(dict(request_id=r["request_id"], started_ns=0, completed_ns=1, snapshot={})), flush=True); time.sleep(5)',
        ]
        reasons = ['collector response JSON invalid', 'collector response request mismatch', 'collector response timestamps invalid']
        with tempfile.TemporaryDirectory() as temp:
            for i, (body, reason) in enumerate(zip(bodies, reasons)):
                with self.subTest(reason=reason):
                    path = Path(temp) / str(i)
                    result = supervise([sys.executable, '-c', 'pass'], path, timeout_s=2,
                                       telemetry_command=command(body), telemetry_timeout_s=.5)
                    self.assertFalse(result['launched'])
                    self.assertEqual(result['reason'], reason)
                    self.assertTrue((path / 'collector-invalid-response.txt').is_file())

    def test_worker_stdout_and_stderr_are_capped(self):
        with tempfile.TemporaryDirectory() as temp:
            for stream, reason, name in [('stdout', 'collector response limit reached', 'collector-invalid-response.txt'),
                                          ('stderr', 'collector stderr limit reached', 'collector-stderr.txt')]:
                path = Path(temp) / stream
                path.mkdir(mode=0o700)
                body = 'sys.stdin.readline(); sys.' + stream + '.write("x"*1000000); sys.' + stream + '.flush()'
                with ProcessCollector(command(body), path, timeout_s=.5,
                                      max_response_bytes=1024, max_stderr_bytes=1024) as worker:
                    snapshot = worker.sample(None, 0, 0)
                    self.assertEqual(snapshot['rejection'], reason)
                self.assertEqual((path / name).stat().st_size, 1024)
                self.assertFalse(worker.summary['cleanup']['pending'])

    def test_callback_exception_is_a_failed_snapshot_not_a_parent_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'exception'
            result = supervise([sys.executable, '-c', 'pass'], path, timeout_s=2,
                               telemetry_command=serving('serve(lambda *args: 1/0)'), telemetry_timeout_s=.5)
            self.assertFalse(result['launched'])
            self.assertEqual(result['reason'], 'collector callback failed')
            row = json.loads((path / 'telemetry.jsonl').read_text())
            self.assertEqual(row['snapshot']['failure_category'], 'ZeroDivisionError')

    def test_owned_leader_reap_timeout_is_explicit(self):
        # Simulate a kernel-blocked reap without creating an unkillable task.
        fake = unittest.mock.Mock(pid=12345)
        fake.wait.side_effect = subprocess.TimeoutExpired('synthetic', .01)
        with patch('benchmark_core.os.killpg'), patch('benchmark_core.time.sleep'):
            result = terminate_group(fake, .01, .01)
        self.assertTrue(result['pending'])
        self.assertIsNone(result['returncode'])
        self.assertEqual(result['group_signals'], ['SIGTERM', 'SIGKILL'])
        fake.wait.assert_called_once_with(timeout=.01)

    def test_workload_deadline_bounds_a_collector_with_a_longer_default(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'whole-deadline'
            before = time.monotonic()
            result = supervise([sys.executable, '-c', 'pass'], path, timeout_s=.15,
                               telemetry_command=serving('import time; serve(lambda *args: time.sleep(5))'),
                               telemetry_timeout_s=10, grace_s=.01)
            self.assertFalse(result['launched'])
            self.assertLess(time.monotonic() - before, 1)
            self.assertEqual(result['reason'], 'collector response timeout')

    def test_collector_spawn_failure_retains_both_results(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'missing'
            with self.assertRaises(FileNotFoundError):
                supervise([sys.executable, '-c', 'pass'], path, timeout_s=1,
                          telemetry_command=[str(Path(temp) / 'missing-worker')])
            result = json.loads((path / 'result.json').read_text())
            self.assertFalse(result['launched'])
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['collector']['status'], 'incomplete')
            self.assertIn('collector startup failed', result['collector']['failure'])

    def test_unexpected_worker_exit_after_response_is_not_silent_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'exit'
            path.mkdir(mode=0o700)
            body = ('import json,time; r=json.loads(sys.stdin.readline()); t=time.monotonic_ns(); '
                    'print(json.dumps(dict(request_id=r["request_id"], started_ns=t, completed_ns=t, snapshot={})), flush=True)')
            with ProcessCollector(command(body), path, timeout_s=.5) as worker:
                self.assertEqual(worker.sample(None, 0, 0), {})
                worker.process.wait(timeout=1)
            self.assertEqual(worker.summary['failure'], 'collector exited unexpectedly')

    def test_normal_worker_with_no_rejection_can_complete_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'normal'
            result = supervise([sys.executable, '-c', 'print("synthetic success")'], path, timeout_s=2,
                               telemetry_command=serving('serve(lambda *args: {})'), telemetry_timeout_s=.5)
            self.assertEqual(result['status'], 'successful')
            self.assertEqual(result['returncode'], 0)
            self.assertEqual(result['collector']['status'], 'closed')

    def test_worker_group_cleanup_reaches_a_term_ignoring_descendant(self):
        child = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); time.sleep(5)'
        body = ('import subprocess,time; from collector_worker import serve; '
                'p=subprocess.Popen([sys.executable,"-u","-c",' + repr(child) + '],stdout=subprocess.PIPE); '
                'assert p.stdout.readline().strip()==b"ready"; '
                'serve(lambda *args: {"owned_child":p.pid})')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'group'
            path.mkdir(mode=0o700)
            with ProcessCollector(command(body), path, timeout_s=1) as worker:
                pid = worker.sample(None, 0, 0)['owned_child']
                initial = Path('/proc') / str(pid) / 'stat'
                identity = initial.read_text().rpartition(')')[2].split()[19]

            def owned_state():
                try:
                    fields = initial.read_text().rpartition(')')[2].split()
                except FileNotFoundError:
                    return None
                return fields[0] if fields[19] == identity else None

            try:
                deadline = time.monotonic() + 1
                while owned_state() not in (None, 'Z') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIn(owned_state(), (None, 'Z'))
            finally:
                # Only touch the known owned instance if the assertion failed.
                if owned_state() not in (None, 'Z'):
                    os.kill(pid, signal.SIGKILL)
