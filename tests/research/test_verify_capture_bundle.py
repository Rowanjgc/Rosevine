"""Private bundle verifier recomputes checksums, cycles and metric contracts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_core as B
import benchmark_metrics as M
import verify_capture_bundle as V


class CaptureBundleVerificationTests(unittest.TestCase):
    def make_run(self, root, code='pass', **kwargs):
        path = root / 'run'
        result = B.supervise([sys.executable, '-c', code], path, timeout_s=2, **kwargs)
        return path, result

    @staticmethod
    def rewrite_result_and_hash(run_dir, mutate):
        result_path = run_dir / 'result.json'
        result = json.loads(result_path.read_text())
        mutate(result)
        result_path.write_text(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
        index_path = run_dir / 'checksums.json'
        index = json.loads(index_path.read_text())
        index['result.json'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        index_path.write_text(json.dumps(index, sort_keys=True, separators=(',', ':')) + '\n')

    def test_recomputes_supervisor_byte_totals_cycle_p95_and_metric_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir, result = self.make_run(
                root, 'import time; time.sleep(.02)', cadence_s=.003,
                telemetry=lambda *_: {'rejection': None})
            summary = V.verify_bundle(run_dir)
            self.assertEqual(summary['status'], 'internally_consistent')
            self.assertEqual(summary['run_status'], result['status'])
            self.assertGreaterEqual(summary['telemetry_rows'], 2)
            self.assertEqual(summary['collector_cycles'], result['collector_cycles'])
            self.assertEqual(summary['metric_count'], len(B.METRICS))
            self.assertTrue(summary['child_executable_identity_available'])
            self.assertEqual(result['child_process_identity']['pid'],
                             result['workload_cleanup']['pid'])
            self.assertEqual(len(result['child_process_identity']['sha256']), 64)
            self.assertFalse(summary['manual_review_required'])

    def test_legacy_bundle_without_measurement_channel_is_accepted_with_explicit_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp))
            manifest_path = run_dir / 'manifest.json'
            result_path = run_dir / 'result.json'
            manifest = json.loads(manifest_path.read_text())
            result = json.loads(result_path.read_text())
            manifest.pop('measurement_telemetry_enabled')
            for key in ('measurement_telemetry_enabled', 'measurement_telemetry_bytes',
                        'measurement_collector', 'measurement_collector_cycles',
                        'measurement_collector_cycle_durations_ns', 'measurement_collector_cycle_p95_ns'):
                result.pop(key)
            for path, value in ((manifest_path, manifest), (result_path, result)):
                path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
            index_path = run_dir / 'checksums.json'
            index = json.loads(index_path.read_text())
            index['manifest.json'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            index['result.json'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index, sort_keys=True, separators=(',', ':')) + '\n')
            summary = V.verify_bundle(run_dir)
            self.assertEqual(summary['status'], 'internally_consistent')
            self.assertIn('bundle predates separate optional measurement collector accounting',
                          summary['legacy_exceptions'])

    def test_partial_measurement_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp))
            self.rewrite_result_and_hash(run_dir, lambda result:
                                         result.pop('measurement_collector_cycles'))
            with self.assertRaisesRegex(ValueError, 'measurement collector state differs|totals are missing'):
                V.verify_bundle(run_dir)

    def test_checksum_tampering_and_rehashed_metric_schema_mismatch_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp))
            with (run_dir / 'stdout.txt').open('a') as stream:
                stream.write('tampered')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                V.verify_bundle(run_dir)
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp))
            self.rewrite_result_and_hash(
                run_dir, lambda result: result['metrics']['rss_bytes'].update(unit='kilobytes'))
            with self.assertRaisesRegex(ValueError, 'schema or unit mismatch'):
                V.verify_bundle(run_dir)

    def test_expected_artifact_identity_is_verified_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / 'tool'
            artifact.write_bytes(b'tool')
            expected = {'role': 'tool', 'path': str(artifact),
                        'sha256': hashlib.sha256(b'tool').hexdigest(), 'size_bytes': 4}
            run_dir, _ = self.make_run(root, expected_artifacts=[expected])
            self.assertEqual(V.verify_bundle(run_dir)['status'], 'internally_consistent')
            self.rewrite_result_and_hash(
                run_dir, lambda result: result['artifact_checks'][0].update(actual_sha256='0' * 64))
            with self.assertRaisesRegex(ValueError, 'matched artifact check disagrees'):
                V.verify_bundle(run_dir)

    def test_mapped_artifact_audit_is_bound_to_prelaunch_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / 'model'
            artifact.write_bytes(b'model')
            expected = {'role': 'model', 'path': str(artifact),
                        'sha256': hashlib.sha256(b'model').hexdigest(), 'size_bytes': 5}
            code = (f"import mmap,time; f=open({str(artifact)!r}, 'rb'); "
                    'm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); time.sleep(.1)')
            run_dir, _ = self.make_run(root, code, expected_artifacts=[expected],
                                       audit_mapped_artifacts=True, cadence_s=.003)
            self.assertEqual(V.verify_bundle(run_dir)['mapped_artifact_audit_status'], 'complete')
            self.rewrite_result_and_hash(
                run_dir, lambda result: result['mapped_artifact_identity']['artifacts'][0].update(
                    device=result['mapped_artifact_identity']['artifacts'][0]['device'] + 1))
            with self.assertRaisesRegex(ValueError, 'disagrees with prelaunch artifact check'):
                V.verify_bundle(run_dir)

    def test_rehashed_p95_tamper_rejected_and_unindexed_child_file_is_flagged(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp), 'open("child.bin", "wb").write(b"x")')
            summary = V.verify_bundle(run_dir)
            self.assertEqual(summary['unindexed_regular_files'], 1)
            self.assertTrue(summary['manual_review_required'])
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp), 'import time; time.sleep(.02)', cadence_s=.003,
                                       telemetry=lambda *_: {'rejection': None})
            self.rewrite_result_and_hash(run_dir, lambda result:
                                         result.update(collector_cycle_p95_ns=999999999))
            with self.assertRaisesRegex(ValueError, 'p95 does not match'):
                V.verify_bundle(run_dir)

    def test_rehashed_resource_metric_tamper_rejected_against_raw_telemetry(self):
        def telemetry(_pid, _elapsed, _size):
            return {
                'memory': {'MemAvailable': 4096}, 'swap': {'pswpin': 8192, 'pswpout': 4096},
                'process': {
                    'memory': {'rss_bytes': 128, 'anonymous_bytes': 64,
                               'file_backed_bytes': 64, 'high_water_rss_bytes': 160},
                    'pss_bytes': {'value': 96, 'reason': None},
                    'counters': {'page_faults': 3, 'cpu_core_seconds': .25,
                                 'process_start_ticks': 123},
                },
                'vram_used_bytes': 512, 'gpu_busy_percent': 10,
                'temperatures_c': {'private test sensor': 45.0},
            }

        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp), 'import time; time.sleep(.02)',
                                       cadence_s=.003, telemetry=telemetry,
                                       metric_adapter=M.resource_guard_metric_adapter())
            verified = V.verify_bundle(run_dir)
            self.assertTrue(verified['resource_aggregates_recomputed'])
            self.rewrite_result_and_hash(run_dir, lambda result:
                                         result['metrics']['rss_bytes'].update(value=129))
            with self.assertRaisesRegex(ValueError, 'disagrees with raw telemetry'):
                V.verify_bundle(run_dir)

    def test_opt_in_loaded_library_identity_is_bounded_and_verifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _ = self.make_run(Path(temp), 'import time; time.sleep(.05)',
                                       cadence_s=.005, audit_loaded_libraries=True)
            summary = V.verify_bundle(run_dir)
            result = json.loads((run_dir / 'result.json').read_text())
        libraries = result['loaded_library_identity']
        self.assertIn(libraries['status'], ('complete', 'incomplete'))
        self.assertGreater(libraries['sample_count'], 0)
        self.assertGreater(libraries['unique_library_count'], 0)
        self.assertEqual(summary['loaded_library_audit_status'], libraries['status'])
        self.assertEqual(summary['loaded_library_count'], libraries['unique_library_count'])
        self.assertTrue(all(len(entry['sha256']) == 64 for entry in libraries['libraries']))


if __name__ == '__main__':
    unittest.main()
