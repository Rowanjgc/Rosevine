"""K2 reductions: paired ABBA step ratio, top-1 agreement, prefill ratio and KL parsing."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
SPEC = importlib.util.spec_from_file_location('analyze_k2', ROOT / 'scripts/research/analyze_k2.py')
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)
C = (',', ':')


def write(root, label, out, err=''):
    d = root / label
    d.mkdir()
    (d / 'run-summary.json').write_text(json.dumps({'status': 'successful'}))
    (d / 'out.txt').write_text(out)
    (d / 'err.txt').write_text(err)


def ab_text(ref_ms, vec_ms, flip=False, compare='q4'):
    lines = [json.dumps({'kind': 'config', 'compare': compare}, separators=C)]
    for pair in range(4):
        for arm in (['ref', 'vec'] if pair % 2 == 0 else ['vec', 'ref']):
            ms = ref_ms if arm == 'ref' else vec_ms
            top1 = [1, 2, 3, 4] if arm == 'ref' or not flip else [1, 2, 3, 9]
            lines.append(json.dumps({'kind': 'block', 'pair': pair, 'q4': arm,
                                     'decode_ns': [int(ms * 1e6)] * 4, 'top1': top1}, separators=C))
    return '\n'.join(lines) + '\n'


def replay_text(prefill_ns):
    config = {'kind': 'config', 'n_prompt': 1000, 'n_decode': 1, 'reps': 1}
    rep = {'kind': 'rep', 'rep': 0, 'prefill_ns': prefill_ns, 'decode_ns': [1], 'prefill_top1': 1, 'top1': [1]}
    return json.dumps(config, separators=C) + '\n' + json.dumps(rep, separators=C) + '\n'


class K2Tests(unittest.TestCase):
    def test_ratio_agreement_prefill_and_kl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, 's1-ab-p512', ab_text(25.0, 20.0))
            write(root, 's2-ab-p512', ab_text(25.0, 20.0, flip=True))
            write(root, 's1-prefill-p2048-ref-a', replay_text(10_000_000_000))
            write(root, 's1-prefill-p2048-vec-a', replay_text(8_000_000_000))
            write(root, 's1-kl-cpu-vec', 'kld rows=1785 mean=0.070100 se=0.006 median=0.006 p99=1.0 p99.9=3.0 '
                                          'max=5.0 same_top=0.9250 ppl=12.2300 ppl_base=12.0276 logits_sha256=x\n')
            report = K.reduce(root)
        ab = report['ab']['ab-p512']
        self.assertEqual(ab['pairs'], 8)
        self.assertAlmostEqual(ab['ratio']['median'], 0.8)
        self.assertEqual(ab['diff_ms']['ci95'], [5.0, 5.0])
        self.assertAlmostEqual(ab['top1_agreement'], 28 / 32)
        self.assertAlmostEqual(report['prefill']['s1-p2048']['vec_over_ref'], 1.25)
        self.assertEqual(report['kl']['kl-cpu-vec']['mean_kl'], 0.0701)

    def test_incomplete_pair_and_wrong_comparison_are_rejected(self):
        with self.assertRaises(ValueError):
            K.ab_pairs('\n'.join(ab_text(1.0, 1.0).splitlines()[:-1]))
        with self.assertRaises(ValueError):
            K.ab_pairs(ab_text(1.0, 1.0, compare='attention'))


if __name__ == '__main__':
    unittest.main()
