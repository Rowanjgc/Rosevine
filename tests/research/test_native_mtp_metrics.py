"""Explicit MTP event reduction preserves denominators and timing boundaries."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from native_mtp_metrics import reduce_events


def record():
    return {'schema': 'rosevine-native-mtp-events-v1', 'mtp_enabled': True,
            'max_draft_tokens': 3, 'verifications': [
                {'verification_id': 'v1', 'proposed_tokens': 3, 'accepted_tokens': 2,
                 'committed_tokens': 3, 'target_evaluated_rows': 4,
                 'phase_intervals_ns': {'draft': [0, 10], 'verify': [11, 30],
                                        'correction': [31, 35]},
                 'rollback_operations': 2},
                {'verification_id': 'v2', 'proposed_tokens': 1, 'accepted_tokens': 0,
                 'committed_tokens': 1, 'target_evaluated_rows': 2,
                 'phase_intervals_ns': {'draft': [40, 45], 'verify': [46, 56],
                                        'correction': None},
                 'rollback_operations': 0}]}


class NativeMTPMetricTests(unittest.TestCase):
    def test_counts_denominators_and_disjoint_phase_durations(self):
        metrics = reduce_events(record())
        self.assertEqual(metrics['mtp_proposed']['value'], 4)
        self.assertEqual(metrics['mtp_accepted']['value'], 2)
        self.assertEqual(metrics['mtp_acceptance_rate']['value'], .5)
        self.assertEqual(metrics['mtp_acceptance_by_position']['value'], [
            {'position': 1, 'accepted': 1, 'proposed': 2, 'fraction': .5},
            {'position': 2, 'accepted': 1, 'proposed': 1, 'fraction': 1.0},
            {'position': 3, 'accepted': 0, 'proposed': 1, 'fraction': 0.0}])
        self.assertEqual(metrics['mtp_committed_per_verification']['value'], 2)
        self.assertEqual(metrics['target_evaluated_rows']['value'], 6)
        self.assertEqual(metrics['draft_s']['value'], 15e-9)
        self.assertEqual(metrics['verify_s']['value'], 29e-9)
        self.assertEqual(metrics['correction_s']['value'], 4e-9)
        self.assertEqual(metrics['rollback_work']['value'], 2)
        self.assertEqual(metrics['prefill_tokens_per_s']['value'], None)

    def test_disabled_policy_and_invalid_or_overlapping_events(self):
        disabled = {'schema': 'rosevine-native-mtp-events-v1', 'mtp_enabled': False,
                    'max_draft_tokens': 3, 'verifications': []}
        result = reduce_events(disabled)
        self.assertEqual(result['mtp_proposed']['value'], 0)
        self.assertIsNone(result['mtp_acceptance_rate']['value'])
        self.assertEqual(result['draft_s']['value'], 0)
        self.assertIsNone(result['verify_s']['value'])
        faults = []
        invalid_count = record()
        invalid_count['verifications'][0]['accepted_tokens'] = 4
        faults.append(invalid_count)
        overlap = record()
        overlap['verifications'][1]['phase_intervals_ns']['draft'] = [29, 45]
        faults.append(overlap)
        duplicate = record()
        duplicate['verifications'][1]['verification_id'] = 'v1'
        faults.append(duplicate)
        for invalid in faults:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                reduce_events(invalid)
        disabled['verifications'].append(record()['verifications'][0])
        with self.assertRaisesRegex(ValueError, 'disabled record'):
            reduce_events(disabled)


if __name__ == '__main__':
    unittest.main()
