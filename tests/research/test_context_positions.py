"""Exact calibration-only local-window boundary position planning."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('context_positions', ROOT / 'scripts/context_positions.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def corpus():
    docs = []
    for name, tokens in [('a', [1, 2, 3, 4, 5, 6]),
                         ('b', [1, 7, 8, 9, 10, 11]),
                         ('c', [1, 12, 13, 14, 15, 16]),
                         ('held-out', [1, 99, 100, 101, 102, 103])]:
        docs.append({'id': name, 'split': 'holdout' if name == 'held-out' else 'calibration',
                     'stratum': 'synthetic', 'token_ids': tokens,
                     'tokens_sha256': C.digest(tokens), 'scored_tokens': len(tokens) - 1,
                     'first_scored_target_index': 1})
    record = {'version': 'rosevine-fidelity-corpus-v1', 'documents': docs}
    record['corpus_sha256'] = C.digest(record)
    return record


class ContextPositionTests(unittest.TestCase):
    def test_boundary_rows_and_target_alignment_are_exact(self):
        result = C.make_plan(corpus(), 1, local_window=3, capacity=256, vocabulary=128)
        self.assertEqual(result['captured_input_positions'], [2, 3, 4])
        self.assertEqual(result['capture_rows'], 9)
        self.assertEqual(result['two_repetitions_bytes'], 9 * 128 * 8)
        for stream in result['streams']:
            self.assertEqual(len(stream['token_ids']), 6)
            self.assertEqual(stream['scored_target_indices'], [3, 4, 5])
            self.assertEqual(stream['target_token_ids'], [stream['token_ids'][p] for p in [3, 4, 5]])
            self.assertTrue(stream['reset_kv_before_sequence'])
        self.assertNotIn('held-out', {span['document_id'] for span in result['pool_spans']})
        self.assertIn('no logits captured', result['status'])

    def test_capacity_bos_and_pool_shortages_fail(self):
        for args in [dict(bos_id=2, local_window=3, capacity=256, vocabulary=128),
                     dict(bos_id=1, local_window=254, capacity=256, vocabulary=128),
                     dict(bos_id=1, local_window=3, capacity=255, vocabulary=128),
                     dict(bos_id=1, local_window=16, capacity=256, vocabulary=128)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                C.make_plan(corpus(), **args)


if __name__ == '__main__':
    unittest.main()
