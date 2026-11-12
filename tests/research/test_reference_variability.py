"""Repeated-reference variability remains descriptive, aligned and private."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests/research'))
import benchmark_core as B
SPEC = importlib.util.spec_from_file_location(
    'reference_variability', ROOT / 'scripts/reference_variability.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
from test_logit_compare import fixture


class ReferenceVariabilityTests(unittest.TestCase):
    def test_references_with_different_extra_buffer_policy_are_not_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [fixture(root / name, [[0, 1]]) for name in ('policy-false', 'policy-true')]
            base = {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                    'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                    'mtp': False, 'logit_stage': 'post-softcap', 'max_shard_rows': 32}
            for i, path in enumerate(paths):
                record = json.loads(path.read_text())
                record['schema'] = 'rosevine-logits-v4'
                record['native_policy'] = {**base, 'extra_buffers': bool(i)}
                record['native_policy_sha256'] = B.digest(record['native_policy'])
                path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'same native capture policy'):
                V.analyze_reference_repeats(paths)

    def test_all_repeat_pairs_are_summarized_without_tolerance_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indexes = [
                fixture(root / 'repeat-a', [[0, 1], [10000, 10001]]),
                fixture(root / 'repeat-b', [[0, 1], [10000, 10001]]),
                fixture(root / 'repeat-c', [[.1, 1], [10000, 10001]]),
            ]
            result = V.analyze_reference_repeats(indexes)
        self.assertEqual(result['reference_captures'], 3)
        self.assertEqual(result['pairwise_comparisons'], 3)
        self.assertEqual(len(result['pairs']), 3)
        self.assertGreater(result['pairwise_observed']['max_position_normalized_rmse_max'], 0)
        self.assertEqual(result['pairwise_observed']['symmetric_mean_kl_nats_max'],
                         max(pair['symmetric_mean_kl_nats'] for pair in result['pairs']))
        self.assertGreaterEqual(result['pairwise_observed']['max_directional_mean_kl_nats'],
                                result['pairwise_observed']['symmetric_mean_kl_nats_max'])
        self.assertIn('not an approved tolerance', result['interpretation'])
        self.assertEqual(result['reference_identity_sha256'], 'a' * 64)
        self.assertFalse(result['minimum_4096_positions_met_for_all_pairs'])
        self.assertFalse(result['position_inventory_verified_for_all_pairs'])
        self.assertFalse(result['minimum_4096_planned_positions_met_for_all_pairs'])
        self.assertFalse(result['distinct_supervised_processes_verified'])

    def test_retained_run_bundles_verify_distinct_child_processes_and_capture_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indexes, bundles = [], []
            for number in range(2):
                index = fixture(root / f'repeat-{number}', [[0, 1], [10000, 10001]])
                index_record = json.loads(index.read_text())
                policy = {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                          'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                          'mtp': False, 'extra_buffers': False,
                          'logit_stage': 'post-softcap', 'max_shard_rows': 32}
                index_record['schema'] = 'rosevine-logits-v4'
                index_record['native_policy'] = policy
                index_record['native_policy_sha256'] = B.digest(policy)
                index.write_text(json.dumps(index_record))
                run_identity = {key: index_record[key] for key in (
                    'identity_sha256', 'inputs_sha256', 'position_plan_sha256',
                    'expected_positions_sha256', 'expected_position_count')}
                run_identity['native_policy_sha256'] = index_record['native_policy_sha256']
                run = root / f'run-{number}'
                artifact = root / f'model-{number}'
                artifact.write_bytes(b'synthetic mapped model')
                expected_artifact = {'role': 'model', 'path': str(artifact),
                                     'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                     'size_bytes': artifact.stat().st_size}
                code = (f"import mmap,time; f=open({str(artifact)!r}, 'rb'); "
                        'm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); time.sleep(.05)')
                B.supervise([sys.executable, '-c', code], run, timeout_s=2, cadence_s=.003,
                            run_identity=run_identity, expected_artifacts=[expected_artifact],
                            audit_mapped_artifacts=True)
                result_bytes = (run / 'result.json').read_bytes()
                result = json.loads(result_bytes)
                child = result['child_process_identity']
                binding = {'schema': 'rosevine-logit-capture-run-v1',
                           'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
                           'supervisor_result_sha256': hashlib.sha256(result_bytes).hexdigest(),
                           'process_instance_sha256': hashlib.sha256(
                               f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest(),
                           'executable_sha256': child['sha256']}
                path = index.parent / 'capture-run-binding.json'
                path.write_text(json.dumps(binding, sort_keys=True, separators=(',', ':')) + '\n')
                os.chmod(path, 0o600)
                indexes.append(index)
                bundles.append(run)
            result = V.analyze_reference_repeats(indexes, bundles)
            self.assertTrue(result['distinct_supervised_processes_verified'])
            self.assertEqual(len(result['process_evidence']['process_instance_sha256']), 2)
            self.assertTrue(result['process_evidence']['mapped_artifact_audit_enabled'])
            self.assertEqual(len(result['process_evidence']['mapped_artifact_identity']), 2)
            for mapping in result['process_evidence']['mapped_artifact_identity']:
                self.assertEqual(mapping['status'], 'complete')
                self.assertEqual(mapping['artifacts'][0]['status'], 'observed')
            self.assertTrue(result['position_inventory_verified_for_all_pairs'])
            self.assertFalse(result['minimum_4096_planned_positions_met_for_all_pairs'])
            self.assertIn('separate sessions', result['independence_note'])
            binding_path = indexes[0].parent / 'capture-run-binding.json'
            binding = json.loads(binding_path.read_text())
            binding['index_sha256'] = '0' * 64
            binding_path.write_text(json.dumps(binding, sort_keys=True, separators=(',', ':')) + '\n')
            with self.assertRaisesRegex(ValueError, 'binding does not match'):
                V.analyze_reference_repeats(indexes, bundles)

    def test_mismatched_context_or_stage_rejects_before_pair_summary(self):
        for field, value, expected in [
                ('inputs_sha256', 'c' * 64, 'input/context identity mismatch'),
                ('logit_stage', 'pre-softcap', 'logit stage mismatch'),
                ('identity_sha256', 'c' * 64, 'run identity mismatch')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                first = fixture(root / 'first', [[0, 1]])
                second = fixture(root / 'second', [[0, 1]])
                import json
                index = json.loads(second.read_text())
                index[field] = value
                second.write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError, expected):
                    V.analyze_reference_repeats([first, second])

    def test_single_reference_is_not_a_repeatability_sample(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            V.analyze_reference_repeats(['one-index'])

    def test_duplicate_index_path_is_not_counted_as_two_captures(self):
        with tempfile.TemporaryDirectory() as temp:
            index = fixture(Path(temp) / 'repeat', [[0, 1]])
            with self.assertRaisesRegex(ValueError, 'distinct index path'):
                V.analyze_reference_repeats([index, index])


if __name__ == '__main__':
    unittest.main()
