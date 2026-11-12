"""Independent analytic fixtures, including packed nibbles and near ties."""
import importlib.util
from pathlib import Path
import struct
import unittest

SPEC = importlib.util.spec_from_file_location('correctness_oracle', Path(__file__).resolve().parents[2] / 'scripts/correctness_oracle.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class OracleTests(unittest.TestCase):
    def test_q4_all_nibbles_and_scale_edges(self):
        packed = bytes((15 - i) * 16 + i for i in range(16))
        for scale in [0.0, -0.0, 1.0, -2.0, 2 ** -24, 65504.0]:
            actual = C.q4_0_decode(struct.pack('<e', scale) + packed)
            expected = [scale * i for i in range(-8, 8)] + [scale * i for i in range(7, -9, -1)]
            self.assertEqual(actual, expected)
        for scale in [float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                C.q4_0_decode(struct.pack('<e', scale) + packed)
        with self.assertRaises(ValueError):
            C.q4_0_decode(bytes(17))

    def test_scalar_matvec_and_cancellation(self):
        self.assertEqual(C.matvec([[1e16, 1, -1e16], [1, 2, 3]], [1, 1, 1]), [1, 6])
        self.assertEqual(C.vector_add([1e16, 2], [-1e16, 3]), [0, 5])
        self.assertEqual(C.weighted_reduce([[1, 2], [3, 4]], [.25, .75]), [2.5, 3.5])
        with self.assertRaises(ValueError):
            C.matvec([[1]], [1, 2])
        with self.assertRaises(ValueError):
            C.vector_add([1], [1, 2])
        with self.assertRaises(ValueError):
            C.weighted_reduce([[1], [1, 2]], [1, 1])
        with self.assertRaises(ValueError):
            C.weighted_reduce([[1]], [float('nan')])

    def test_embedding_lookup_and_explicit_scaling(self):
        table = [[1, 2], [-3, 4]]
        self.assertEqual(C.embedding_lookup(table, [1, 0, 1], 2),
                         [[-6, 8], [2, 4], [-6, 8]])
        self.assertEqual(C.embedding_lookup(table, [0], 0), [[0, 0]])
        for ids, scale in [([2], 1), ([-1], 1), ([True], 1), ([0], float('inf'))]:
            with self.assertRaises(ValueError):
                C.embedding_lookup(table, ids, scale)
        with self.assertRaises(ValueError):
            C.embedding_lookup([[1], [2, 3]], [0], 1)

    def test_tensor_metrics_zero_and_nonfinite(self):
        self.assertAlmostEqual(C.tensor_errors([1, 2], [1, 2])['cosine'], 1)
        self.assertEqual(C.tensor_errors([0, 0], [0, 0])['normalized_rmse'], 0)
        self.assertIsNone(C.tensor_errors([0], [1])['cosine'])
        for candidate in [[], [float('nan')], [1, 2]]:
            with self.assertRaises(ValueError):
                C.tensor_errors([1], candidate)

    def test_stable_kl_and_top1(self):
        self.assertEqual(C.logit_errors([[10000, 10001]], [[10000, 10001]])['mean_kl_nats'], 0)
        result = C.logit_errors([[0, 0]], [[0, 1]])
        self.assertAlmostEqual(result['mean_kl_nats'], .1201145069582775)
        self.assertEqual(result['top1_agreement'], 0)
        self.assertFalse(result['sufficient_positions'])
        with self.assertRaises(ValueError):
            C.logit_errors([[0, 1], [0]], [[0, 1], [0]])

    def test_router_boundary_ties(self):
        result = C.router_margin([3, 2, 2, 1], [0, 1], 2)
        self.assertEqual(result['boundary_margin'], 0)
        self.assertTrue(result['selected_set_matches'])
        self.assertFalse(C.router_margin([3, 2, 1], [0, 2], 2)['selected_set_matches'])

    def test_router_selected_weights_use_softmax_and_clamped_renormalization(self):
        import math
        result = C.router_selected_weights([0, math.log(3), 0], [1, 0])
        for actual, expected in zip(result['probabilities'], [.2, .6, .2]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result['selected_probabilities'], [.6, .2]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(result['weights'], [.75, .25]):
            self.assertAlmostEqual(actual, expected)
        underflow = C.router_selected_weights([0, -1000], [1])
        self.assertEqual(underflow['selected_sum'], 0)
        self.assertEqual(underflow['normalization_denominator'], 6.103515625e-5)
        self.assertEqual(underflow['weights'], [0])
        for ids in [[], [0, 0], [2], [True]]:
            with self.assertRaises(ValueError):
                C.router_selected_weights([0, 1], ids)

    def test_rms_norm_zero_epsilon_weights_and_shape(self):
        import math
        result = C.rms_norm([3, 4], .5, [2, -1])
        self.assertAlmostEqual(result[0], 6 / math.sqrt(13))
        self.assertAlmostEqual(result[1], -4 / math.sqrt(13))
        self.assertEqual(C.rms_norm([0, 0], 1e-6), [0, 0])
        for epsilon in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                C.rms_norm([1], epsilon)
        with self.assertRaises(ValueError):
            C.rms_norm([1, 2], 1e-6, [1])

    def test_router_projection_preserves_explicit_source_order(self):
        import math
        result = C.router_projection([3, 4], .5, .25, [2, -1],
                                     [[1, 2], [-1, 1]])
        self.assertEqual(result['rms_norm'], [3 / math.sqrt(13), 4 / math.sqrt(13)])
        self.assertEqual(result['width_scaled'], [3 / (4 * math.sqrt(13)),
                                                   1 / math.sqrt(13)])
        self.assertEqual(result['input_scaled'], [3 / (2 * math.sqrt(13)),
                                                  -1 / math.sqrt(13)])
        self.assertEqual(result['logits'], [-1 / (2 * math.sqrt(13)),
                                            -5 / (2 * math.sqrt(13))])
        for scale, inverse in [([1], .5), ([1, float('inf')], .5), ([1, 1], 0)]:
            with self.assertRaises(ValueError):
                C.router_projection([1, 2], 1e-6, inverse, scale, [[1, 1]])

    def test_gelu_symmetry_and_large_limits(self):
        result = C.gelu_tanh([-10, -1, 0, 1, 10])
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 10)
        self.assertEqual(result[2], 0)
        self.assertAlmostEqual(result[3] - result[1], 1)

    def test_explicit_rope_pair_rotation_and_softcap_edges(self):
        import math
        rotated = C.rope_rotate([1, 0, 3], [(0, 1)], [math.pi / 2])
        self.assertAlmostEqual(rotated[0], 0)
        self.assertAlmostEqual(rotated[1], 1)
        self.assertEqual(rotated[2], 3)
        self.assertEqual(C.rope_rotate([1, 2, 3], [], []), [1, 2, 3])
        for pairs, angles in [([(0, 1), (1, 2)], [0, 0]), ([(0, 3)], [0]), ([(0, 1)], [float('nan')])]:
            with self.assertRaises(ValueError):
                C.rope_rotate([1, 2, 3], pairs, angles)
        capped = C.softcap([-100, 0, 100], 2)
        self.assertAlmostEqual(capped[0], -2)
        self.assertEqual(capped[1], 0)
        self.assertAlmostEqual(capped[2], 2)
        with self.assertRaises(ValueError):
            C.softcap([1], 0)

    def test_cpu_gelu_fp16_precision_stratum_and_cutoff(self):
        values = [-10.0, -9.999, -1.0, 0.0, 1.0, 9.999, 10.0]
        result = C.gelu_cpu_fp16_lookup(values)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[-1], 10.0)
        self.assertEqual(result[3], 0.0)
        # The source checks the original FP32 input at +/-10, then uses an
        # FP16-indexed table for interior values (including values rounding to
        # the cutoff when converted to FP16).
        self.assertNotEqual(result[2], C.gelu_tanh([values[2]])[0])
        self.assertNotEqual(result[4], C.gelu_tanh([values[4]])[0])
        self.assertEqual(result, C.gelu_cpu_fp16_lookup(values))
        with self.assertRaises(ValueError):
            C.gelu_cpu_fp16_lookup([float('nan')])
        with self.assertRaises(ValueError):
            C.gelu_cpu_fp16_lookup([1e100])

    def test_attention_analytic_weights_masks_and_invalid_rows(self):
        import math
        result = C.attention([1], [[0], [math.log(3)]], [[2], [6]], [True, True], 1)
        self.assertAlmostEqual(result['weights'][0], .25)
        self.assertAlmostEqual(result['output'][0], 5)
        masked = C.attention([1], [[0], [1000]], [[2], [999]], [True, False], 1)
        self.assertEqual(masked, {'weights': [1, 0], 'output': [2]})
        with self.assertRaises(ValueError):
            C.attention([1], [[0]], [[2]], [False], 1)
        with self.assertRaises(ValueError):
            C.attention([1, 2], [[0]], [[2]], [True], 1)

    def test_gated_ffn_uses_gelu_gate_and_correct_matrix_orientation(self):
        result = C.gated_ffn([2, -1], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[3, 4]])
        self.assertEqual(result['up'], [2, -1])
        self.assertEqual(result['gate'], [-1, 2])
        self.assertAlmostEqual(result['output'][0], 6 * C.gelu_tanh([-1])[0] - 4 * C.gelu_tanh([2])[0])
        with self.assertRaises(ValueError):
            C.gated_ffn([1], [[1]], [[1], [2]], [[1]])
