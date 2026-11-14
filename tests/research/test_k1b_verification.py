"""K1b reductions: per-shape step and committed-token costs, sync spans, overhead and top-1 identity."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/research'))
SPEC = importlib.util.spec_from_file_location('analyze_k1b', ROOT / 'scripts/research/analyze_k1b.py')
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)
from test_k1_attribution import summary  # noqa: E402

SHAPES = [(1, 0), (4, 3), (4, 0)]
C = (',', ':')


def case(root, label, ms, flip=False):
    d = root / label
    d.mkdir()
    (d / 'run-summary.json').write_text(json.dumps({'status': 'successful'}))
    rows = [{'schema': 'rosevine-verify-sweep-v1', 'kind': 'config'}, {'kind': 'prefill', 'prefill_ns': 1}]
    err = []
    for rep in range(2):
        for w, a in SHAPES:
            top1 = [5, 6, 9 if flip else 7]
            rows.append({'kind': 'block', 'rep': rep, 'width': w, 'accept': a,
                         'step_ns': [int(ms * w * 1e6)] * 3, 'top1': top1})
            err += summary(f'phase=w{w}a{a} tokens=3 loop_ms=30.000 loop_closure=1.0000', 3, 4.0 * w, 2.0)
    (d / 'out.txt').write_text('\n'.join(json.dumps(r, separators=C) for r in rows) + '\n')
    (d / 'err.txt').write_text('\n'.join(err) + '\n' if label.endswith('sync') else '')


class K1bTests(unittest.TestCase):
    def test_shapes_overhead_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case(root, 's1-d1900-off', 10.0)
            case(root, 's1-d1900-sync', 11.0, flip=True)
            report = K.reduce(root)
        w4 = report['shapes']['d1900-w4a3']
        self.assertEqual(w4['blocks'], 2)
        self.assertAlmostEqual(w4['step_ms'], 40.0)
        self.assertAlmostEqual(w4['ms_per_committed_token'], 10.0)
        self.assertAlmostEqual(report['shapes']['d1900-w4a0']['ms_per_committed_token'], 40.0)
        self.assertAlmostEqual(w4['sync_overhead'], 0.1)
        self.assertAlmostEqual(w4['spans_ms_per_verification']['cpu_experts/host'], 16.0)
        self.assertEqual(report['top1_compared'], 18)
        self.assertEqual(report['top1_mismatches'], 6)


if __name__ == '__main__':
    unittest.main()
