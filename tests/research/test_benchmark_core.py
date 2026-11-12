"""Synthetic replay, denominators, pairing, and supervisor failure paths."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('benchmark_core', ROOT / 'scripts/benchmark_core.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class BenchmarkTests(unittest.TestCase):
    def test_all_workloads_exact_replay_and_capacity(self):
        suite = json.loads((ROOT / 'research/fixtures/short-text-v1.json').read_text())
        identity = {key: 'a' * 64 for key in ['model_sha256', 'template_sha256', 'tokenizer_sha256']}
        policy = {'thinking': True, 'seed': 7, 'sampler_order': ['temperature'],
                  'sampler_parameters': {'temperature': 0}, 'stops': []}
        # Each Unicode character is one synthetic token; this is not model parity.
        adapter = lambda prompt, policy: ('<' + prompt + '>', [ord(c) for c in '<' + prompt + '>'])
        for fixture in suite['fixtures']:
            for occupied in (1900, 29000):
                # Unit-width padding guarantees reachable occupancy for this adapter.
                a = B.make_workload(fixture, 'x', occupied, 65536, adapter, identity, policy)
                b = B.make_workload(fixture, 'x', occupied, 131072, adapter, identity, policy)
                self.assertEqual(len(a['token_ids']), occupied)
                self.assertEqual(a['tokens_sha256'], b['tokens_sha256'])
                self.assertNotEqual(a['manifest_sha256'], b['manifest_sha256'])
        with self.assertRaises(ValueError):
            B.make_workload(suite['fixtures'][0], 'x', 1900, 2000, adapter, identity, policy)
        with self.assertRaises(ValueError):
            B.make_workload(suite['fixtures'][0], 'x', 1900, 65536,
                            lambda p, s: ('a', [1, 2]), identity, policy, max_attempts=4)

    def test_capacity_rebinding_preserves_exact_request_and_checks_reserve(self):
        fixture = {'id': 'synthetic', 'prompt': 'answer'}
        identity = {key: 'a' * 64 for key in ['model_sha256', 'template_sha256', 'tokenizer_sha256']}
        policy = {'thinking': True, 'seed': 7, 'sampler_order': ['temperature'],
                  'sampler_parameters': {'temperature': 0}, 'stops': []}
        source = B.make_workload(fixture, 'x', 1900, 65536,
                                 lambda prompt, _: (prompt, [ord(c) for c in prompt]),
                                 identity, policy)
        rebound = B.rebind_workload_capacity(source, 4096)
        self.assertEqual(rebound['capacity_tokens'], 4096)
        for key in ('prompt', 'rendered', 'token_ids', 'tokens_sha256', 'rendered_sha256'):
            self.assertEqual(rebound[key], source[key])
        self.assertNotEqual(rebound['manifest_sha256'], source['manifest_sha256'])
        self.assertEqual(rebound['manifest_sha256'], B.digest({k: v for k, v in rebound.items()
                                                               if k != 'manifest_sha256'}))
        with self.assertRaises(ValueError):
            B.rebind_workload_capacity(source, 2911)
        tampered = dict(source, rendered=source['rendered'] + '!')
        tampered['manifest_sha256'] = B.digest({k: v for k, v in tampered.items()
                                                 if k != 'manifest_sha256'})
        with self.assertRaises(ValueError):
            B.rebind_workload_capacity(tampered, 4096)

    def test_pairing_warmups_and_seed(self):
        schedule = B.paired_schedule(['short', 'medium'], ['A', 'B'], 17)
        self.assertEqual(schedule, B.paired_schedule(['short', 'medium'], ['A', 'B'], 17))
        self.assertEqual(len(schedule), 36)
        for phase, count in [('warmup', 2), ('measured', 7)]:
            for trial in range(count):
                for workload in ['short', 'medium']:
                    rows = [r for r in schedule if (r['phase'], r['pair'], r['workload']) == (phase, trial, workload)]
                    self.assertEqual({r['configuration'] for r in rows}, {'A', 'B'})
                    self.assertEqual(len({r['seed'] for r in rows}), 1)

    def test_timing_boundaries_and_missing_counters(self):
        metrics = B.delivered_metrics(1000000000, 4000000000, [2000000000, 2500000000, 3000000000])
        self.assertEqual(metrics, {'ttft_s': 1, 'end_to_end_s': 3, 'steady_output_tokens_per_s': 2})
        self.assertIsNone(B.delivered_metrics(0, 1, [])['ttft_s'])
        with self.assertRaises(ValueError):
            B.delivered_metrics(3, 5, [2])
        self.assertTrue(all(v['value'] is None and v['reason'] for v in B.missing_metrics('unsupported').values()))
        self.assertTrue(all(v['unit'] for v in B.missing_metrics('unsupported').values()))

    def test_bootstrap_and_summary(self):
        result = B.paired_improvement([100] * 7, [90] * 7, higher_is_better=False, seed=42, resamples=1000)
        self.assertEqual(result['ci95_percent'], [10, 10])
        self.assertEqual(result['median_benefit_percent'], 10)
        self.assertEqual(B.summarize([1, 2, 3, 4])['median'], 2.5)
        with self.assertRaises(ValueError):
            B.summarize([float('nan')])
        with self.assertRaises(ValueError):
            B.paired_improvement([1] * 7, [1] * 6, higher_is_better=True, seed=0)

    def test_supervision_retains_success_failure_timeout_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = [('success', 'print(42)', 'successful', None),
                     ('failure', 'raise SystemExit(3)', 'failed', None),
                     ('timeout', 'import time; time.sleep(5)', 'incomplete', None),
                     ('cancel', 'import time; time.sleep(5)', 'incomplete', lambda: 'budget breach')]
            for name, code, status, cancel in cases:
                path = root / name
                result = B.supervise([sys.executable, '-c', code], path,
                                     timeout_s=.15, grace_s=.01, cancel=cancel)
                self.assertEqual(result['status'], status)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                for file in path.iterdir():
                    self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
                index = json.loads((path / 'checksums.json').read_text())
                self.assertEqual(set(index), {'manifest.json', 'stdout.txt', 'stderr.txt', 'result.json'})
            with self.assertRaises(FileExistsError):
                B.supervise([sys.executable, '-c', 'pass'], root / 'success', timeout_s=1)

    def test_supervisor_persists_validated_metric_inventory_without_changing_run_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metric-success'
            inventory = B.missing_metrics('not produced by synthetic adapter')
            inventory['end_to_end_s'] = {'value': .25, 'reason': None, 'unit': 's'}

            def adapter(result, records):
                self.assertEqual(result['status'], 'successful')
                self.assertGreaterEqual(len(records), 2)
                self.assertEqual(records[0]['stage'], 'preflight')
                self.assertIn('running', {row['stage'] for row in records})
                return {'synthetic delivery': inventory}

            result = B.supervise([sys.executable, '-c', 'import time; time.sleep(.03)'], path,
                                 timeout_s=2, cadence_s=.005,
                                 telemetry=lambda *_: {'counter': 1}, metric_adapter=adapter)
            persisted = json.loads((path / 'result.json').read_text())
            self.assertEqual(result['status'], 'successful')
            self.assertEqual(persisted['metrics']['end_to_end_s']['value'], .25)
            self.assertEqual(persisted['metric_sources']['end_to_end_s'], ['synthetic delivery'])
            self.assertTrue(json.loads((path / 'manifest.json').read_text())['metric_adapter_enabled'])

    def test_metric_adapter_failure_preserves_run_and_marks_all_values_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metric-adapter-failure'

            def adapter(_result, _records):
                raise RuntimeError('sensitive details must not be copied')

            result = B.supervise([sys.executable, '-c', 'pass'], path,
                                 timeout_s=2, metric_adapter=adapter)
            self.assertEqual(result['status'], 'successful')
            self.assertEqual(result['metric_adapter_error'], 'RuntimeError')
            self.assertTrue(all(metric['value'] is None and metric['reason']
                                for metric in result['metrics'].values()))
            self.assertNotIn('sensitive details', (path / 'result.json').read_text())

    def test_result_retains_recomputable_telemetry_cycle_durations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cycle-audit'
            result = B.supervise([sys.executable, '-c', 'import time; time.sleep(.04)'], path,
                                 timeout_s=1, cadence_s=.005,
                                 telemetry=lambda *_: {'rejection': None})
            durations = result['collector_cycle_durations_ns']
            self.assertEqual(result['status'], 'successful')
            self.assertEqual(len(durations), result['collector_cycles'])
            self.assertGreaterEqual(len(durations), 2)
            self.assertTrue(all(type(v) is int and v > 0 for v in durations))
            self.assertEqual(result['collector_cycle_p95_ns'], B.percentile(durations, .95))
            persisted = json.loads((path / 'result.json').read_text())
            self.assertEqual(persisted['collector_cycle_durations_ns'], durations)

    def test_spawn_failure_retains_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'spawn-failure'
            with self.assertRaises(FileNotFoundError):
                B.supervise([str(Path(temp) / 'missing-program')], path, timeout_s=1)
            result = json.loads((path / 'result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['reason'], 'FileNotFoundError')
            self.assertTrue((path / 'checksums.json').exists())

    def test_expected_artifact_preflight_matches_and_blocks_mismatch_or_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / 'synthetic-model.bin'
            artifact.write_bytes(b'known bytes')
            expected = {'role': 'model', 'path': str(artifact),
                        'sha256': __import__('hashlib').sha256(b'known bytes').hexdigest(),
                        'size_bytes': len(b'known bytes')}
            passed = B.supervise([sys.executable, '-c', 'print("ran")'], root / 'matched',
                                 timeout_s=2, expected_artifacts=[expected])
            self.assertTrue(passed['launched'])
            self.assertEqual(passed['artifact_checks'][0]['status'], 'matched')

            mismatch = dict(expected, sha256='0' * 64)
            blocked = B.supervise([sys.executable, '-c', 'raise SystemExit(91)'], root / 'mismatch',
                                  timeout_s=2, expected_artifacts=[mismatch])
            self.assertFalse(blocked['launched'])
            self.assertEqual(blocked['status'], 'incomplete')
            self.assertEqual(blocked['reason'], 'artifact_identity_check_failed')
            self.assertEqual(blocked['artifact_checks'][0]['status'], 'mismatch')

            missing = dict(expected, path=str(root / 'missing'))
            unavailable = B.supervise([sys.executable, '-c', 'raise SystemExit(92)'], root / 'missing-run',
                                      timeout_s=2, expected_artifacts=[missing])
            self.assertFalse(unavailable['launched'])
            self.assertEqual(unavailable['artifact_checks'][0]['status'], 'unavailable')

    def test_private_writer_rejects_public_parent_and_existing_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o755)
            with self.assertRaises(ValueError):
                B.write_private_json(root / 'candidate.json', {})
            os.chmod(root, 0o700)
            B.write_private_json(root / 'candidate.json', {})
            with self.assertRaises(FileExistsError):
                B.write_private_json(root / 'candidate.json', {})
            (root / 'link.json').symlink_to(root / 'candidate.json')
            with self.assertRaises(FileExistsError):
                B.write_private_json(root / 'link.json', {})

    def test_output_flood_is_bounded_and_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'flood'
            result = B.supervise([sys.executable, '-c',
                                  'import os; os.write(1, b"x" * 1000000); os.write(2, b"y" * 1000000)'],
                                 path, timeout_s=2, grace_s=.01, max_output_bytes=1000)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['reason'], 'output capture limit reached')
            self.assertEqual(sum((path / name).stat().st_size for name in ['stdout.txt', 'stderr.txt']), 1000)
            self.assertEqual(result['captured_output_bytes'], 1000)

    def test_telemetry_preflight_blocks_launch_and_preserves_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'preflight'
            result = B.supervise([sys.executable, '-c', 'raise RuntimeError()'], path,
                                 timeout_s=1, telemetry=lambda pid, elapsed, size:
                                 {'rejection': 'VRAM reserve breach', 'pid': pid})
            self.assertFalse(result['launched'])
            self.assertEqual(result['status'], 'incomplete')
            rows = [json.loads(line) for line in (path / 'telemetry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['stage'], 'preflight')
            self.assertEqual(rows[0]['snapshot']['rejection'], 'VRAM reserve breach')

    def test_telemetry_periodic_rejection_and_capture_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'periodic'
            result = B.supervise([sys.executable, '-c', 'import time; time.sleep(5)'], path,
                                 timeout_s=1, grace_s=.01, cadence_s=.01,
                                 telemetry=lambda pid, elapsed, size:
                                 {'rejection': 'swap activity' if pid else None})
            self.assertTrue(result['launched'])
            self.assertEqual(result['reason'], 'swap activity')
            self.assertEqual(result['collector_cycles'], 2)
            result = B.supervise([sys.executable, '-c', 'pass'], Path(temp) / 'large-sample',
                                 timeout_s=1, max_telemetry_bytes=10,
                                 telemetry=lambda *args: {'counter': 123})
            self.assertFalse(result['launched'])
            self.assertEqual(result['reason'], 'telemetry capture limit reached')
            self.assertEqual(result['telemetry_bytes'], 0)

    def test_telemetry_failure_and_disk_rejection_do_not_launch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad-sample'
            with self.assertRaises(ValueError):
                B.supervise([sys.executable, '-c', 'pass'], path, timeout_s=1,
                            telemetry=lambda *args: {'counter': float('nan')})
            self.assertEqual(json.loads((path / 'result.json').read_text())['reason'], 'ValueError')
            with patch.object(B.shutil, 'disk_usage') as usage:
                usage.return_value.free = 99
                result = B.supervise([sys.executable, '-c', 'pass'], Path(temp) / 'disk',
                                     timeout_s=1, disk_reserve_bytes=100)
            self.assertFalse(result['launched'])
            self.assertEqual(result['reason'], 'disk reserve breach')

    def test_ignoring_sigterm_is_killed_after_grace(self):
        import signal
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ignore-term'
            code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready", flush=True); time.sleep(10)'
            result = B.supervise([sys.executable, '-c', code], path,
                                 timeout_s=.3, grace_s=.05)
            self.assertEqual(result['reason'], 'timeout')
            self.assertIn('ready', (path / 'stdout.txt').read_text())
            self.assertEqual(result['returncode'], -signal.SIGKILL)
            self.assertGreaterEqual(result['elapsed_s'], .35)
            self.assertLess(result['elapsed_s'], 2)

    def test_campaign_accounts_for_existing_failed_evidence_and_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'previous').mkdir(mode=0o700)
            B.write_private_json(root / 'previous/result.json', {'status': 'failed'})
            used = B.campaign_usage(root)
            self.assertEqual(used, (root / 'previous/result.json').stat().st_size)
            self.assertIsNone(B.campaign_rejection(used, 10, used + 10, 110, 100))
            self.assertEqual(B.campaign_rejection(used, 11, used + 10, 111, 100), 'campaign byte budget exceeded')
            self.assertEqual(B.campaign_rejection(used, 10, used + 10, 109, 100), 'projected disk reserve breach')
            result = B.supervise([sys.executable, '-c', 'print("must not run")'], root / 'blocked',
                                 timeout_s=1, campaign_root=root, campaign_limit_bytes=1024)
            self.assertFalse(result['launched'])
            self.assertEqual(result['reason'], 'campaign byte budget exceeded')
            self.assertEqual((root / 'blocked/stdout.txt').read_bytes(), b'')
            self.assertTrue((root / 'previous/result.json').exists())

    def test_campaign_rejects_symlinks_and_public_capture_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'bad'
            path.symlink_to(root)
            with self.assertRaises(ValueError):
                B.campaign_usage(root)
            path.unlink()
            path.write_text('synthetic')
            os.chmod(path, 0o644)
            with self.assertRaises(ValueError):
                B.campaign_usage(root)

    def test_explicit_environment_prevents_inherited_native_option_overrides(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {'LLAMA_ARG_MODEL': 'synthetic-inherited-value'}):
                result = B.supervise([sys.executable, '-c', 'import os; print(os.environ.get("LLAMA_ARG_MODEL", "absent"))'],
                                     root / 'env', timeout_s=1, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
            self.assertEqual(result['status'], 'successful')
            self.assertEqual((root / 'env/stdout.txt').read_text().strip(), 'absent')
            manifest = json.loads((root / 'env/manifest.json').read_text())
            self.assertEqual(manifest['environment_mode'], 'explicit')
            self.assertIsNotNone(manifest['environment_sha256'])

    def test_runtime_mapped_artifact_audit_binds_child_and_reports_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / 'synthetic-model.bin'
            artifact.write_bytes(b'model fixture bytes')
            expected = {'role': 'model', 'path': str(artifact),
                        'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        'size_bytes': artifact.stat().st_size}
            code = (f"import mmap,time; f=open({str(artifact)!r}, 'rb'); "
                    'm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); time.sleep(.15)')
            run_dir = root / 'mapped-audit'
            result = B.supervise([sys.executable, '-c', code], run_dir,
                                 timeout_s=2, cadence_s=.005,
                                 expected_artifacts=[expected], audit_mapped_artifacts=True)
            self.assertEqual(result['status'], 'successful')
            identity = result['mapped_artifact_identity']
            self.assertEqual(identity['status'], 'complete')
            self.assertEqual(identity['artifacts'][0]['role'], 'model')
            self.assertEqual(identity['artifacts'][0]['status'], 'observed')
            self.assertTrue(identity['artifacts'][0]['stable_file_identity'])
            import verify_capture_bundle as verifier
            verified = verifier.verify_bundle(run_dir)
            self.assertEqual(verified['mapped_artifact_audit_status'], 'complete')

    def test_runtime_mapped_artifact_audit_preserves_not_observed_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / 'synthetic-model.bin'
            artifact.write_bytes(b'model fixture bytes')
            expected = {'role': 'model', 'path': str(artifact),
                        'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        'size_bytes': artifact.stat().st_size}
            result = B.supervise([sys.executable, '-c', 'import time; time.sleep(.04)'],
                                 root / 'unmapped-audit', timeout_s=2, cadence_s=.005,
                                 expected_artifacts=[expected], audit_mapped_artifacts=True)
            self.assertEqual(result['status'], 'successful')
            self.assertEqual(result['mapped_artifact_identity']['status'], 'complete')
            self.assertEqual(result['mapped_artifact_identity']['artifacts'][0]['status'], 'not_observed')
