import unittest

from native_q4_model_blocks import sample_positions, sample_tensor_positions


class ModelBlockSamplingTests(unittest.TestCase):
    def test_sampling_is_stable_and_covers_each_expert_boundaries(self):
        first = sample_positions('blk.0.ffn_down_exps.weight', 128, 2000)
        second = sample_positions('blk.0.ffn_down_exps.weight', 128, 2000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 128 * 4)
        for expert in range(128):
            group = first[expert * 4:(expert + 1) * 4]
            base = expert * 2000
            self.assertIn(base, group)
            self.assertIn(base + 1999, group)
            self.assertEqual(len(set(group)), 4)
            self.assertTrue(all(base < position < base + 1999 for position in group if position not in (base, base + 1999)))

    def test_sampling_rejects_invalid_dimensions(self):
        for experts, blocks in ((0, 10), (2, 0), (2, 1), (True, 10), (2, 10.0)):
            with self.subTest(experts=experts, blocks=blocks):
                with self.assertRaises(ValueError):
                    sample_positions('tensor', experts, blocks)

    def test_tensor_sampling_handles_small_tensors_and_keeps_stable_boundaries(self):
        for count in (1, 2, 3):
            self.assertEqual(sample_tensor_positions('tensor', count), list(range(count)))
        first = sample_tensor_positions('blk.0.attn_q.weight', 8192)
        self.assertEqual(first, sample_tensor_positions('blk.0.attn_q.weight', 8192))
        self.assertEqual(len(first), 4)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 8191)
        self.assertTrue(all(0 < position < 8191 for position in first[1:-1]))


if __name__ == '__main__':
    unittest.main()
