"""Parity P3 reducer on a synthetic private campaign layout."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests/research'))
import parity_campaign as C
import parity_contract as P
from test_parity_contract import replay_text

IDENTITY = {'binaries': {'rv': 'a' * 64, 'hip': 'b' * 64, 'vk': 'c' * 64}, 'rosevine_revision': 'd' * 40,
            'model_sha256': 'e' * 64, 'fixture_sha256': 'f' * 64, 'hardware_class': 'rx7800xt-7600x',
            'power_policy': 'compute-window', 'affinity': 'all-12'}


def write_case(root, label, engine, step_ns, *, status='successful', top1=5):
    case = root / label
    case.mkdir()
    (case / 'run-summary.json').write_text(json.dumps({'status': status, 'reason': None}))
    session = {'sensors': [{'label': 'GPU edge', 'flag_c': 85}]}
    (case / 'session.json').write_text(json.dumps(session))
    extra = {'cpu_moe_layers': 10, 'n_ctx': 768, 'cpu_expert_activation': 'f32' if engine == 'rosevine' else 'q8_0'}
    text = replay_text(engine, reps=2, n_prompt=512, n_decode=128, step_ns=step_ns, extra=extra)
    if top1 != 5:
        text = text.replace('"top1": [' + ', '.join(['5'] * 128), '"top1": [' + ', '.join([str(top1)] * 128))
    (case / 'replay.jsonl').write_text(text)
    (case / 'resources.txt').write_text('\tMaximum resident set size (kbytes): 1000\n'
                                        '\tUser time (seconds): 4.0\n\tSystem time (seconds): 1.0\n')
    telemetry = [{'stage': 'preflight', 'snapshot': {'vram_used_bytes': 100}},
                 {'stage': 'running', 'snapshot': {'vram_used_bytes': 600, 'temperatures_c': {'GPU edge': 70}}}]
    (case / 'supervision').mkdir()
    (case / 'supervision' / 'telemetry.jsonl').write_text('\n'.join(json.dumps(t) for t in telemetry) + '\n')


class Reduce(unittest.TestCase):
    def build(self, root, sessions=7, rv_top1=5):
        for s in range(1, sessions + 1):
            write_case(root, 's%d-rv-p512' % s, 'rosevine', 40_000_000 + s, top1=rv_top1)
            write_case(root, 's%d-hip-p512' % s, 'llama.cpp', 20_000_000)
            write_case(root, 's%d-vk-p512' % s, 'llama.cpp', 16_000_000)

    def test_pairs_ratios_and_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root)
            report = C.reduce(root, IDENTITY, seed=1, agreement_floor=0.85)
        rows = {(r['cell'], r['baseline']): r for r in report['rows']}
        vk = rows[('decode-d512', 'vulkan')]
        self.assertEqual(vk['outcome'], 'measured')
        self.assertEqual(vk['comparison_class'], 'practical_configuration')
        self.assertAlmostEqual(vk['ratio']['ratio_median'], 0.4, places=3)
        self.assertFalse(vk['milestones']['p90'])
        self.assertEqual(rows[('prefill-p512', 'hip')]['outcome'], 'measured')
        self.assertEqual(rows[('decode-d2048', 'hip')]['outcome'], 'not_run')
        self.assertEqual(len(report['rows']), 2 * len(P.matrix_v1()['cells']))

    def test_correctness_floor_blocks_speed_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, rv_top1=9)
            report = C.reduce(root, IDENTITY, seed=1, agreement_floor=0.85)
        rows = {(r['cell'], r['baseline']): r for r in report['rows']}
        self.assertEqual(rows[('decode-d512', 'vulkan')]['outcome'], 'correctness_failed')

    def test_too_few_pairs_is_not_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, sessions=3)
            report = C.reduce(root, IDENTITY, seed=1, agreement_floor=0.85)
        self.assertEqual(report['details']['decode-d512|vulkan']['pairs'], 3)
        rows = {(r['cell'], r['baseline']): r for r in report['rows']}
        self.assertEqual(rows[('decode-d512', 'vulkan')]['outcome'], 'not_run')

    def test_failed_case_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root)
            (root / 's9-rv-p512').mkdir()
            (root / 's9-rv-p512' / 'run-summary.json').write_text(json.dumps({'status': 'rejected', 'reason': 'thermal'}))
            report = C.reduce(root, IDENTITY, seed=1, agreement_floor=0.85)
        self.assertEqual(report['failures'], [{'session': 9, 'arm': 'rv', 'p': 512, 'failed': 'thermal'}])

    def test_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_case(root, 's1-rv-p512', 'rosevine', 1)
            out = C.resources(root / 's1-rv-p512')
        self.assertEqual(out['vram_peak_bytes'], 500)
        self.assertEqual(out['cpu_core_seconds'], 5.0)
        self.assertEqual(out['high_water_rss_bytes'], 1024000)


if __name__ == '__main__':
    unittest.main()
