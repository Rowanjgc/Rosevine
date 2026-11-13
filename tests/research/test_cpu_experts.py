"""Reject incomplete/malformed numerical evidence; tolerances stay fixed."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('cpu_experts', ROOT / 'scripts/research/analyze_cpu_experts.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class CpuExpertsAnalysisTests(unittest.TestCase):
    def test_generation_requires_full_declared_output(self):
        text = 'generated 1 2\nprompt=40 prefill_s=1.00 steps=2 decode_s=0.20 decode_tok_s=5.00\n'
        self.assertEqual(A.generation(text)['tokens'], [1, 2])
        for invalid in (text.replace('steps=2', 'steps=3'), text.replace('decode_s=0.20', 'decode_s=0.00'), text+text):
            with self.assertRaises(ValueError):
                A.generation(invalid)

    def test_kl_gate_requires_coverage_and_both_metrics(self):
        digest = 'a' * 64
        text = ''.join(f'chunk {i} logits_sha256={digest}\n' for i in range(7))
        text += f'kld rows=1785 mean=0.064 same_top=0.926 logits_sha256={digest}\n'
        self.assertEqual(A.kld(text)['rows'], 1785)
        for invalid in (text.replace('mean=0.064', 'mean=0.081'), text.replace('same_top=0.926', 'same_top=0.919'),
                        text.replace('rows=1785', 'rows=255'), text.replace(f'chunk 0 logits_sha256={digest}\n', '')):
            with self.assertRaises(ValueError):
                A.kld(invalid)


if __name__ == '__main__':
    unittest.main()
