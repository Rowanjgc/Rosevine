"""K1 reductions: per-token normalization, local/global split, overhead and top-1 checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('analyze_k1', ROOT / 'scripts/research/analyze_k1.py')
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)

HEADER = ('span         group   layers   ms/eval    share    dev_ms   max_ms  launch   h2d   d2h   d2d  sync')


def summary(phase_line, evals, cpu_ms, core_ms):
    layers = ['layer attn attn_core attn_out']
    # Six layers: layer 5 is global and holds 3 of every 4 attn_core ms.
    for layer in range(6):
        core = core_ms * (0.75 if layer == 5 else 0.05)
        layers.append(f'{layer} 0.1000/0.0000 {core:.4f}/0.0000 0.1000/0.0000')
    return [phase_line,
            f'mode=Sync evals={evals} aborted=0 wall_ms=10.0 span_ms=10.0 closure=1.0000 per_eval_ms=1.0',
            'per_eval launches=10.0 h2d=1.0 h2d_B=4 d2h=1.0 d2h_B=4 d2d=0.0 syncs=0.0 blocking=2.0 probe=0.0',
            HEADER,
            f'cpu_experts  host         2 {cpu_ms:9.4f}   0.5000    0.0000   1.0000     0.0   0.0   0.0   0.0   0.0',
            f'attn_core    gpu          4 {core_ms:9.4f}   0.1000    0.0000   1.0000     2.0   0.0   0.0   0.0   0.0',
            *layers]


def case(root, label, *, top1=7, prefill_ns=2_000_000_000, status='successful'):
    d = root / label
    d.mkdir()
    (d / 'run-summary.json').write_text(json.dumps({'status': status, 'reason': None}))
    config = {'kind': 'config', 'n_prompt': 128, 'n_decode': 2, 'reps': 2, 'n_batch': 64, 'cpu_moe_layers': 2,
              'cpu_expert_activation': 'q8_0'}
    reps = [{'kind': 'rep', 'rep': r, 'prefill_ns': prefill_ns, 'decode_ns': [10_000_000, 12_000_000],
             'prefill_top1': 3, 'top1': [top1, 8]} for r in (0, 1)]
    (d / 'out.txt').write_text('\n'.join(json.dumps(x, separators=(',', ':')) for x in [config, *reps]) + '\n')
    err = []
    if not label.endswith('-off'):
        for _ in range(2):
            err += summary('phase=prefill tokens=128 loop_ms=2000.000 loop_closure=1.0000', 2, 40.0, 4.0)
            err += summary('decode_loop_ms=22.000 loop_closure=0.9990', 2, 4.0, 2.0)
    (d / 'err.txt').write_text('\n'.join(err) + '\n')


class K1Tests(unittest.TestCase):
    def reduce(self, build):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build(root)
            return K.reduce(root)

    def test_normalizes_per_token_and_splits_global_layers(self):
        def build(root):
            case(root, 's1-p128-off')
            case(root, 's1-p128-sync', prefill_ns=2_500_000_000)
        report = self.reduce(build)
        sync = report['cells']['p128']['sessions'][1]['sync']
        prefill = sync['phases']['prefill']['per_token']
        # 2 evaluations over 128 tokens: 40 ms per evaluation is 0.625 ms per token.
        self.assertAlmostEqual(prefill['cpu_experts/host']['host_ms'], 0.625)
        self.assertAlmostEqual(prefill['attn_core/global']['host_ms'], 3.0 * 2 / 128)
        self.assertAlmostEqual(prefill['attn_core/local']['host_ms'], 1.0 * 2 / 128)
        decode = sync['phases']['decode']['per_token']
        self.assertAlmostEqual(decode['cpu_experts/host']['host_ms'], 4.0)
        self.assertAlmostEqual(sync['prefill_overhead'], 0.25)
        self.assertTrue(sync['top1_identical_to_off'])

    def test_reports_failures_and_changed_tokens(self):
        def build(root):
            case(root, 's1-p128-off')
            case(root, 's1-p128-events', top1=9)
            case(root, 's2-p128-off', status='failed')
        report = self.reduce(build)
        self.assertEqual(report['failed'], {'s2-p128-off': 'failed'})
        self.assertFalse(report['cells']['p128']['sessions'][1]['events']['top1_identical_to_off'])

    def test_global_layer_pattern(self):
        self.assertEqual([layer for layer in range(30) if K.is_global(layer)], [5, 11, 17, 23, 29])


if __name__ == '__main__':
    unittest.main()
