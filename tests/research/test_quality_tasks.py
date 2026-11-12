"""Frozen task split, strict visible JSON scoring and noninferiority uncertainty."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('quality_tasks', ROOT / 'scripts/quality_tasks.py')
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


class QualityTaskTests(unittest.TestCase):
    def test_fixed_balanced_disjoint_splits_and_all_correct_answers(self):
        suite = Q.suite()
        self.assertEqual(len(suite['calibration']), 20)
        self.assertEqual(len(suite['holdout']), 100)
        self.assertEqual(suite, Q.suite())
        calibration = {t['prompt'] for t in suite['calibration']}
        self.assertFalse(calibration & {t['prompt'] for t in suite['holdout']})
        for category in Q.CATEGORIES:
            self.assertEqual(sum(t['category'] == category for t in suite['holdout']), 20)
        for task in suite['calibration'] + suite['holdout']:
            self.assertTrue(Q.score(task, json.dumps(task['expected'])))
            self.assertFalse(Q.score(task, json.dumps(dict(task['expected'], extra='unrequested'))))

    def test_strict_json_not_greedy_json_extraction_or_boolean_integer_equality(self):
        task = {'expected': {'result': 1}}
        for response in ['{"result":true}', '{"result":1,"result":1}', '```json\n{"result":1}\n```',
                         'reasoning then {"result":1}', '{"result":NaN}', '{"result":1.0}', '{}']:
            self.assertFalse(Q.score(task, response))
        self.assertTrue(Q.score(task, ' \n {"result": 1} \n'))
        with self.assertRaises(ValueError):
            Q.score({'expected': {'result': float('nan')}}, '{"result":1}')
        with self.assertRaises(ValueError):
            Q.score({}, '{}')

    def test_zero_discordance_does_not_imply_zero_population_uncertainty(self):
        for n in [100, 125]:
            reference = {str(i): True for i in range(n)}
            result = Q.paired_accuracy(reference, reference)
            self.assertAlmostEqual(result['conservative_95_upper_drop'], 1 - .025 ** (1 / n), places=12)
            self.assertEqual(result['accuracy_drop'], 0)
            if n == 100:
                self.assertGreater(result['conservative_95_upper_drop'], .03)
            else:
                self.assertLess(result['conservative_95_upper_drop'], .03)

    def test_best_case_task_count_for_three_point_margin(self):
        plan = Q.minimum_zero_discordance_tasks(.03)
        self.assertEqual(plan['minimum_iid_tasks'], 122)
        self.assertLessEqual(plan['upper_bound_at_minimum'], .03)
        self.assertGreater(plan['upper_bound_at_previous_n'], .03)
        self.assertEqual(plan['one_sided_alpha'], .025)
        for margin, alpha in ((0, .025), (1, .025), (.03, 0), (.03, 1), (True, .025), (.03, False)):
            with self.subTest(margin=margin, alpha=alpha):
                with self.assertRaises(ValueError):
                    Q.minimum_zero_discordance_tasks(margin, alpha)

    def test_discordant_pairing_and_invalid_score_types(self):
        result = Q.paired_accuracy({'a': True, 'b': False}, {'a': False, 'b': True})
        self.assertEqual((result['losses'], result['wins'], result['accuracy_drop']), (1, 1, 0))
        with self.assertRaises(ValueError):
            Q.paired_accuracy({'a': True}, {'b': True})
        with self.assertRaises(ValueError):
            Q.paired_accuracy({'a': True}, {'a': 1})

    def test_recipe_hashes_and_manually_checked_examples_are_frozen(self):
        recipe = json.loads((ROOT / 'research/fixtures/quality-tasks-v1.json').read_text())
        suite = Q.suite()
        self.assertEqual(recipe['calibration']['sha256'], suite['calibration_sha256'])
        self.assertEqual(recipe['holdout']['sha256'], suite['holdout_sha256'])
        examples = {task['category']: task for task in suite['holdout'] if task['id'].endswith('-000')}
        expected = {'arithmetic': {'result': 345},
                    'structured-transform': {'values': [38, 68, 92, 98]},
                    'algorithm-trace': {'total': 185},
                    'text-instructions': {'words': ['BIRCH', 'CEDAR', 'GRANITE']},
                    'retrieval': {'value': 991}}
        self.assertEqual({key: task['expected'] for key, task in examples.items()}, expected)
