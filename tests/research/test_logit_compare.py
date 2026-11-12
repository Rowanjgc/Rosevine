"""Shard alignment, exact bytes, final hash verification, and analytic logit errors."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_core as B
SPEC = importlib.util.spec_from_file_location('logit_compare', ROOT / 'scripts/logit_compare.py')
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


def fixture(root, rows, positions=None):
    root.mkdir(mode=0o700)
    raw = b''.join(struct.pack('<' + 'f' * len(row), *row) for row in rows)
    path = root / 'logits.f32'
    path.write_bytes(raw)
    os.chmod(path, 0o600)
    labels = positions or [['synthetic', i] for i in range(len(rows))]
    record = {'schema': 'rosevine-logits-v3', 'identity_sha256': 'a' * 64,
              'inputs_sha256': 'b' * 64, 'logit_stage': 'post-softcap',
              'position_plan_sha256': 'c' * 64,
              'expected_positions_sha256': B.digest(labels),
              'expected_position_count': len(labels),
              'vocabulary': len(rows[0]), 'shards': [{'file': 'logits.f32',
              'sha256': hashlib.sha256(raw).hexdigest(), 'positions': labels}]}
    path = root / 'index.json'
    path.write_text(json.dumps(record))
    os.chmod(path, 0o600)
    return path


class LogitShardTests(unittest.TestCase):
    def test_policy_bound_capture_reports_extra_buffer_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [fixture(root / name, [[0, 1]]) for name in ('false', 'true')]
            policy = {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                      'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                      'mtp': False, 'extra_buffers': False, 'logit_stage': 'post-softcap',
                      'max_shard_rows': 32}
            for i, path in enumerate(paths):
                record = json.loads(path.read_text())
                record['schema'] = 'rosevine-logits-v4'
                record['native_policy'] = {**policy, 'extra_buffers': bool(i)}
                record['native_policy_sha256'] = B.digest(record['native_policy'])
                path.write_text(json.dumps(record))
            result = L.compare(*paths)
            self.assertTrue(result['native_policy_comparison']['extra_buffers_varied'])
            self.assertFalse(result['native_policy_comparison']['same'])
            record = json.loads(paths[0].read_text())
            for fault in ('digest', 'thread-policy'):
                corrupted = dict(record)
                if fault == 'digest':
                    corrupted['native_policy_sha256'] = 'f' * 64
                else:
                    corrupted['native_policy'] = {**corrupted['native_policy'], 'threads': 4}
                    corrupted['native_policy_sha256'] = B.digest(corrupted['native_policy'])
                paths[0].write_text(json.dumps(corrupted))
                with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'native capture policy'):
                    L.load_index(paths[0])

    def test_same_labels_do_not_permit_different_context_or_stage(self):
        for field, value in [('inputs_sha256', 'c' * 64),
                             ('logit_stage', 'pre-softcap')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                reference = fixture(root / 'ref', [[0, 1]])
                candidate = fixture(root / 'cand', [[0, 1]])
                record = json.loads(candidate.read_text())
                record[field] = value
                candidate.write_text(json.dumps(record))
                # Fail before attempting to open either shard.
                (reference.parent / 'logits.f32').unlink()
                with self.assertRaisesRegex(ValueError, 'identity mismatch|stage mismatch'):
                    L.compare(reference, candidate)

    def test_private_regular_index_and_unique_keys_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = fixture(root / 'ref', [[0, 1]])
            original = path.read_text()
            os.chmod(path, 0o644)
            with self.assertRaises(ValueError):
                L.load_index(path)
            os.chmod(path, 0o600)
            path.write_text(original[:-1] + ', "vocabulary": 2}')
            with self.assertRaisesRegex(ValueError, 'duplicate index key'):
                L.load_index(path)
            path.write_text(original)
            link = path.parent / 'link.json'
            link.symlink_to(path.name)
            with self.assertRaises(OSError):
                L.load_index(link)

    def test_analytic_comparison_and_insufficient_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture(root / 'ref', [[0, 0], [10000, 10001]])
            candidate = fixture(root / 'cand', [[0, 1], [10000, 10001]])
            result = L.compare(reference, candidate)
            self.assertAlmostEqual(result['mean_kl_nats'], .1201145069582775 / 2)
            self.assertEqual(result['top1_agreement'], .5)
            self.assertFalse(result['minimum_4096_positions_met'])
            self.assertEqual(result['top1_mismatches'], [('synthetic', 0)])

    def test_legacy_v2_capture_remains_comparable_without_plan_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [fixture(root / name, [[0, 1]]) for name in ('old-ref', 'old-cand')]
            for path in paths:
                record = json.loads(path.read_text())
                record['schema'] = 'rosevine-logits-v2'
                for key in ('position_plan_sha256', 'expected_positions_sha256',
                            'expected_position_count'):
                    record.pop(key)
                path.write_text(json.dumps(record))
            result = L.compare(*paths)
            self.assertEqual(result['positions'], 1)
            self.assertFalse(result['position_inventory_verified'])
            self.assertFalse(result['minimum_4096_planned_positions_met'])

    def test_run_bundles_bind_indexes_and_require_distinct_children(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indexes, bundles = [], []
            for label in ('reference', 'candidate'):
                index = fixture(root / label, [[0, 1]])
                run = root / (label + '-run')
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
                runtime = run.parent / (label + '-mapped-runtime')
                runtime.write_bytes(b'synthetic mapped runtime fixture')
                expected_runtime = {'role': 'runtime', 'path': str(runtime),
                                    'sha256': hashlib.sha256(runtime.read_bytes()).hexdigest(),
                                    'size_bytes': runtime.stat().st_size}
                code = (f"import mmap,time; f=open({str(runtime)!r}, 'rb'); "
                        'm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); time.sleep(.05)')
                result = B.supervise([sys.executable, '-c', code], run,
                                     timeout_s=2, cadence_s=.003, run_identity=run_identity,
                                     expected_artifacts=[expected_runtime],
                                     audit_mapped_artifacts=True)
                child = result['child_process_identity']
                binding = {'schema': 'rosevine-logit-capture-run-v1',
                           'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
                           'supervisor_result_sha256': hashlib.sha256(
                               (run / 'result.json').read_bytes()).hexdigest(),
                           'process_instance_sha256': hashlib.sha256(
                               f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest(),
                           'executable_sha256': child['sha256']}
                path = index.parent / 'capture-run-binding.json'
                path.write_text(json.dumps(binding, sort_keys=True, separators=(',', ':')) + '\n')
                os.chmod(path, 0o600)
                indexes.append(index)
                bundles.append(run)
            result = L.compare(*indexes, run_bundles=bundles)
            self.assertTrue(result['run_bundle_provenance_verified'])
            self.assertTrue(result['native_policy_comparison']['same'])
            self.assertEqual(len(result['run_bundle_provenance']), 2)
            for evidence in result['run_bundle_provenance']:
                self.assertEqual(evidence['mapped_artifact_identity']['status'], 'complete')
                self.assertEqual(evidence['mapped_artifact_identity']['artifacts'][0]['status'], 'observed')
            binding_path = indexes[0].parent / 'capture-run-binding.json'
            binding = json.loads(binding_path.read_text())
            binding['index_sha256'] = '0' * 64
            binding_path.write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError, 'does not match index'):
                L.compare(*indexes, run_bundles=bundles)

    def test_last_shard_hash_must_be_verified_on_both_streams(self):
        for side in ['ref', 'cand']:
            with self.subTest(side=side), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                reference = fixture(root / 'ref', [[0, 1]])
                candidate = fixture(root / 'cand', [[0, 1]])
                # Same byte length and finite values; only the checksum catches this.
                (root / side / 'logits.f32').write_bytes(struct.pack('<ff', 0, 2))
                with self.assertRaises(ValueError):
                    L.compare(reference, candidate)

    def test_positions_shapes_nonfinite_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture(root / 'ref', [[0, 1]])
            candidate = fixture(root / 'cand', [[0, 1]], [['different', 0]])
            with self.assertRaises(ValueError):
                L.compare(reference, candidate)
            record = json.loads(candidate.read_text())
            record['shards'][0]['positions'] = [['synthetic', 0]]
            for filename in ['../outside.f32', '/outside.f32']:
                record['shards'][0]['file'] = filename
                candidate.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    L.load_index(candidate)
            nan_index = fixture(root / 'nan', [[float('nan'), 1]])
            with self.assertRaises(ValueError):
                L.compare(reference, nan_index)
            (reference.parent / 'logits.f32').write_bytes(b'')
            with self.assertRaises(ValueError):
                list(L.rows(reference, L.load_index(reference)))
