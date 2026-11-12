"""Private identity coverage, unknowns, tamper detection and artifact evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('benchmark_manifest', ROOT / 'scripts/benchmark_manifest.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class IdentityTests(unittest.TestCase):
    def test_all_protocol_fields_accounted_for_without_inventing_values(self):
        record = M.make_identity({'experiment': {'run_id': 'synthetic-1'},
                                  'software': {'revision': 'a' * 40},
                                  'artifacts': {'target': {'sha256': 'b' * 64, 'size_bytes': 18}}},
                                 missing_reason='not collected in synthetic run')
        gaps = M.validate_identity(record)
        self.assertIn('environment.run_window_review', gaps)
        self.assertNotIn('experiment.run_id', gaps)
        self.assertNotIn('artifacts.target', gaps)
        record = M.annotate_missing(record, {'artifacts': {'projector': 'not loaded by this text-only control'}})
        self.assertEqual(record['fields']['artifacts']['projector']['reason'], 'not loaded by this text-only control')
        M.validate_identity(record)
        with self.assertRaises(ValueError):
            M.annotate_missing(record, {'experiment': {'run_id': 'unknown'}})

    def test_tampering_and_invalid_known_fields_fail(self):
        record = M.make_identity({}, missing_reason='not collected')
        record['fields']['experiment']['run_id']['value'] = 'changed'
        with self.assertRaises(ValueError):
            M.validate_identity(record)
        for provided in [{'unknown': {}}, {'software': {'revision': 'short'}},
                         {'request': {'thinking': 1}}, {'execution': {'batch': 0}},
                         {'artifacts': {'target': {'sha256': 'a' * 64, 'size_bytes': 0}}},
                         {'software': {'loaded_library_hashes': {}}},
                         {'request': {'sampler_parameters': {'temperature': float('nan')}}}]:
            with self.subTest(provided=provided):
                with self.assertRaises(ValueError):
                    M.make_identity(provided, missing_reason='not collected')
