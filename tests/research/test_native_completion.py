"""Exact executed IDs, native phase units, missing data and non-inferred output rates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import native_completion as N


def log(ids=(2, 7, 9), phase_count=3, duration='30.00'):
    return ('0.001 I main: number of tokens in prompt = 3\n'
            + ''.join(f'0.001 I {token:6d} -> \'synthetic\'\n' for token in ids)
            + f'0.010 I common_perf_print: prompt eval time = {duration} ms / {phase_count} tokens (irrelevant rate)\n'
            + '0.020 I common_perf_print:        eval time = 10.00 ms / 1 runs (irrelevant rate)\n')


class NativeCompletionTests(unittest.TestCase):
    def test_exact_replay_phase_conversion_and_explicit_missing_delivery_metrics(self):
        result = N.observations(log(), [2, 7, 9])
        self.assertEqual(result['prompt_replay']['status'], 'exact')
        self.assertEqual(result['native_prompt_phase']['value'], {'duration_s': .03, 'count': 3})
        self.assertEqual(result['metrics']['prefill_tokens_per_s']['value'], 100)
        self.assertEqual(result['metrics']['prefill_tokens_per_s']['unit'], 'tokens/s')
        for key in ['target_decode_tokens_per_s', 'steady_output_tokens_per_s', 'ttft_s', 'end_to_end_s']:
            self.assertIsNone(result['metrics'][key]['value'])
            self.assertTrue(result['metrics'][key]['reason'])

    def test_prompt_id_or_count_mismatch_cannot_be_hidden_by_valid_timings(self):
        for text in [log((2, 7)), log((2, 7, 10)), log() + '0.030 I main: number of tokens in prompt = 3\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                N.observations(text, [2, 7, 9])

    def test_incomplete_prompt_phase_does_not_invent_whole_prompt_throughput(self):
        result = N.observations(log(phase_count=2), [2, 7, 9])
        self.assertEqual(result['prompt_replay']['status'], 'exact')
        self.assertIsNone(result['metrics']['prefill_tokens_per_s']['value'])
        result = N.observations(log().split('0.010')[0], [2, 7, 9])
        self.assertIsNone(result['native_prompt_phase']['value'])

    def test_invalid_or_ambiguous_timings_fail_and_zero_duration_stays_unknown(self):
        for duration in ['nan', 'inf', '-1']:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                N.observations(log(duration=duration), [2, 7, 9])
        with self.assertRaises(ValueError):
            N.observations(log() + '0.030 I common_perf_print: prompt eval time = 10 ms / 3 tokens\n', [2, 7, 9])
        self.assertIsNone(N.observations(log(duration='0'), [2, 7, 9])['metrics']['prefill_tokens_per_s']['value'])
