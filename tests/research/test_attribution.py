"""R3a.2 reductions must use per-evaluation deltas and reject partial evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('attribution', ROOT / 'scripts/research/analyze_attribution.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def capture():
    lines = []
    for round_id in (0, 1):
        lines.append(f'round={round_id} mode=Some(Host) wall_s=0.032')
        for step in range(32):
            lines.append(f'token,{step},1000000,{step+1}')
        for step in range(32):
            n = step + 1
            lines.append(f'span,{step},0,cpu_experts,{n},{n*1000000},0,{n*2},{n},4,0,0,0')
        lines.append('loop_closure=1.000000')
    return '\n'.join(lines)


class AttributionTests(unittest.TestCase):
    def reduce(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.txt'
            path.write_text(text)
            return A.reduce_hip(path)

    def test_differences_cumulative_spans_and_counters(self):
        rows = self.reduce(capture())
        self.assertEqual(rows[1]['stage_ms']['cpu_experts']['mean'], 1)
        self.assertEqual(rows[1]['spans']['0/cpu_experts']['host_ms']['p95'], 1)
        self.assertEqual(rows[1]['calls_per_token']['launches'], 2)
        self.assertEqual(rows[0]['token_ms']['n'], 32)

    def test_rejects_partial_and_different_continuation(self):
        for text in (capture().replace('token,31,1000000,32', '', 1),
                     capture().replace('token,0,1000000,1', 'token,0,1000000,9', 1)):
            with self.assertRaises(ValueError):
                self.reduce(text)

    def test_rejects_counter_discontinuity(self):
        text = capture().replace('cpu_experts,2,2000000', 'cpu_experts,2,1', 1)
        with self.assertRaises(ValueError):
            self.reduce(text)


if __name__ == '__main__':
    unittest.main()
