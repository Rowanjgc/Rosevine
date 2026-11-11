"""Contract failure-path tests without real weights."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('shapes', Path(__file__).resolve().parents[2] / 'scripts/gemma4_shapes.py')
SHAPES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHAPES)


class ContractTests(unittest.TestCase):
    def test_reject_corruptions(self):
        tensors = [dict(name=k, shape=v[0], type=v[1]) for k, v in SHAPES.expected('assistant').items()]
        variants = [tensors[:-1], tensors + [tensors[0]], tensors + [dict(name='extra', shape=[1], type='F32')]]
        for field, value in [('shape', [1]), ('type', 'F32')]:
            copy = [dict(t) for t in tensors]
            copy[0][field] = value
            variants.append(copy)
        for variant in variants:
            with self.assertRaises(ValueError):
                SHAPES.validate('assistant', variant)

    def test_attention_and_shared_kv_boundaries(self):
        target = SHAPES.expected('target')
        self.assertEqual(target['blk.5.attn_q.weight'][0], [2816, 8192])
        self.assertNotIn('blk.5.attn_v.weight', target)
        self.assertEqual(target['blk.6.attn_q.weight'][0], [2816, 4096])
        assistant = SHAPES.expected('assistant')
        self.assertFalse(any('.attn_k.weight' in name for name in assistant))


if __name__ == '__main__':
    unittest.main()
