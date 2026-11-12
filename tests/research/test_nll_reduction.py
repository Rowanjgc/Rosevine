"""Target alignment for the independent retained-logit reduction checker."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import check_nll_reduction as C


class ReductionPlanTests(unittest.TestCase):
    def test_expected_targets_follow_selected_input_positions(self):
        manifest = {'sequences': [
            {'id': 'first', 'token_ids': [10, 11, 12, 13], 'capture_input_positions': [0, 2]},
            {'id': 'second', 'token_ids': [20, 21, 22]},
        ]}
        self.assertEqual(C._expected_rows(manifest), {
            ('first', 0): 11, ('first', 2): 13,
            ('second', 0): 21, ('second', 1): 22,
        })

    def test_default_positions_cover_each_next_token_target(self):
        manifest = {'sequences': [{'id': 'sequence', 'token_ids': [3, 5, 8, 13]}]}
        self.assertEqual(C._expected_rows(manifest), {
            ('sequence', 0): 5, ('sequence', 1): 8, ('sequence', 2): 13,
        })


if __name__ == '__main__':
    unittest.main()
