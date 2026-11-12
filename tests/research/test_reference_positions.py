"""Calibration-only source spans, target alignment, diversity and byte budgets."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('reference_positions', ROOT / 'scripts/reference_positions.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def corpus():
    docs = []
    for name, tokens, split in [('a', [1, 2, 3, 4, 5], 'calibration'),
                                ('b', [1, 6, 7, 8, 9], 'calibration'),
                                ('holdout-only', [1, 99, 100], 'holdout')]:
        docs.append({'id': name, 'split': split, 'stratum': 'synthetic', 'token_ids': tokens,
                     'tokens_sha256': R.digest(tokens), 'scored_tokens': len(tokens) - 1,
                     'first_scored_target_index': 1})
    record = {'version': 'rosevine-fidelity-corpus-v1', 'documents': docs}
    record['corpus_sha256'] = R.digest(record)
    return record


class ReferencePositionTests(unittest.TestCase):
    def test_windows_are_explicit_calibration_tokens_not_quality_holdout(self):
        result = R.make_plan(corpus(), 1, streams=2, positions=3, capacity=4, vocabulary=128)
        self.assertEqual([s['token_ids'] for s in result['sequences']], [[1, 2, 3, 4], [1, 6, 7, 8]])
        self.assertEqual(result['scored_positions'], 6)
        self.assertEqual(result['two_repetitions_bytes'], 6 * 128 * 8)
        self.assertFalse(result['minimum_4096_positions_planned'])
        self.assertEqual(result['sequences'][0]['scored_input_positions'], [0, 2])
        self.assertEqual(result['sequences'][0]['scored_target_indices'], [1, 3])
        self.assertNotIn('holdout-only', {s['document_id'] for s in result['pool_spans']})
        self.assertEqual(result['protocol_scope']['decision'],
                         'direct-token teacher-forced input is applicable to final-logit numerical comparison')
        self.assertEqual(result['protocol_scope']['execution_status'],
                         'planned only; separate run-window and resource review still required')
        self.assertIn('overlapping/circular calibration windows are correlated and are not quality samples',
                      result['protocol_scope']['limitations'])

    def test_bos_capacity_shortage_and_duplicate_streams_reject(self):
        for kwargs in [dict(bos_id=2, positions=3, capacity=4),
                       dict(bos_id=1, positions=4, capacity=4),
                       dict(bos_id=1, positions=9, capacity=10),
                       dict(bos_id=1, streams=20, positions=2, capacity=4)]:
            with self.assertRaises(ValueError):
                R.make_plan(corpus(), **kwargs)
