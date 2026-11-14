"""S1 reductions: per-verification acceptance from trace logs, task-to-request mapping, divergence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('analyze_s1', ROOT / 'scripts/research/analyze_s1.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def log(tasks):
    lines = []
    for task, steps in tasks:
        lines.append(f'0.01.000.000 I slot launch_slot_: id  0 | task {task} | processing task, is_child = 0')
        lines += [f'0.01.000.001 I slot update_slots: id  0 | task {task} | accepted {a:2d}/{n:2d} draft tokens'
                  for a, n in steps]
    lines.append('0.02.000.000 I spec print_stats: statistics        draft-mtp: #calls(b,g,a) =    2     5      5, '
                 '#gen drafts =      5, #acc drafts =     4, #gen tokens =     15, #acc tokens =     9, '
                 '#mean acc len = 2.80, #acc rate/pos = (0.800, 0.600, 0.400), dur(b,g,a) = 0.010, 20.000, 0.500 ms')
    return '\n'.join(lines) + '\n'


def request(rid, sent, tokens, n_verif):
    return {'id': rid, 'client_sent_ns': sent, 'ok': True, 'output_token_ids': tokens,
            'timings': {'predicted_n': len(tokens), 'predicted_ms': 10.0 * n_verif, 'prompt_n': 1900,
                        'predicted_per_second': 100.0, 'draft_n': 3 * n_verif, 'draft_n_accepted': 0}}


def write(root, label, tasks, rows):
    d = root / label
    d.mkdir()
    (d / 'run-summary.json').write_text(json.dumps({'status': 'successful'}))
    (d / 'server.log').write_text(log(tasks))
    (d / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


class S1Tests(unittest.TestCase):
    def test_histogram_positions_and_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [(3, 3), (0, 3), (2, 3), (1, 3), (3, 3)]
            write(root, 'nat-mtp', [(0, []), (7, steps)],
                  [request('warm0', 1, [1], 0), request('prose-summary-1900-r0', 2, [1, 2, 3, 4], 5)])
            write(root, 'nat-off', [(0, []), (3, [])],
                  [request('warm0', 1, [1], 0), request('prose-summary-1900-r0', 2, [1, 2, 9, 4], 0)])
            report = S.reduce(root)
        r = report['cells']['nat-mtp']['requests']['prose-summary-1900-r0']
        self.assertEqual(r['verifications'], 5)
        self.assertEqual(r['accepted_hist'], {'0': 1, '1': 1, '2': 1, '3': 2})
        self.assertEqual(r['accept_rate_per_pos'], [0.8, 0.6, 0.4])
        self.assertAlmostEqual(r['committed_per_verification'], 14 / 5)
        self.assertEqual(r['ms_per_verification'], 10.0)
        self.assertEqual(report['cells']['nat-mtp']['draft_stats_final']['dur_draft_ms'], 20.0)
        self.assertEqual(report['mtp_vs_off_first_difference']['prose-summary-1900-r0'], 2)

    def test_task_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, 'nat-mtp', [(0, [])], [request('warm0', 1, [1], 0), request('x-1900-r0', 2, [1], 0)])
            with self.assertRaises(ValueError):
                S.reduce(root)


if __name__ == '__main__':
    unittest.main()
