"""Parity P1 contract: frozen matrix, adapter parsing, mismatch rejection and per-cell ratios."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import parity_contract as P


def replay_text(engine, *, reps=2, n_prompt=4, n_decode=3, step_ns=50_000_000, extra=None):
    config = {'schema': P.REPLAY_SCHEMA, 'kind': 'config', 'engine': engine, 'n_prompt': n_prompt,
              'n_decode': n_decode, 'reps': reps, 'n_ctx': 512, 'n_batch': 64, 'n_ubatch': 64,
              'threads': 6, 'cpu_moe_layers': 10, 'kv_type': 'q8_0',
              'cpu_expert_activation': 'f32' if engine == 'rosevine' else 'q8_0',
              'tokens_fnv1a64': '91f01bff94253cd7'}
    if engine == 'rosevine':
        config.update(backend='hip', expert_kernel='auto', expert_wait='sleep')
    else:
        config.update(flash_attn='on', cpu_repack='on', swa_full=False, op_offload=True, buffers=[])
    config.update(extra or {})
    rows = [config] + [{'schema': P.REPLAY_SCHEMA, 'kind': 'rep', 'rep': i, 'prefill_ns': 400_000_000,
                        'decode_ns': [step_ns] * n_decode} for i in range(reps)]
    return '\n'.join(json.dumps(r) for r in rows) + '\n'


def declared(backend, engine):
    return dict(backend=backend, source_revision=P.FROZEN_PIN if engine == 'llama.cpp' else 'f' * 40,
                binary_sha256='b' * 64, model_sha256='a' * 64, fixture_sha256='c' * 64,
                timing_family='engine', boundary=dict(P.ENGINE_BOUNDARY), warm_state='warm',
                hardware_class='rx7800xt-7600x', power_policy='compute-window', affinity='all-12',
                resident_bytes=10 << 30, untuned=engine == 'rosevine', series='frozen')


def arms(cand_extra=None, ref_extra=None, ref_backend='vulkan'):
    ref = P.arm_from_replay(P.parse_replay(replay_text('llama.cpp', extra=ref_extra), engine='llama.cpp'),
                            **declared(ref_backend, 'llama.cpp'))
    cand = P.arm_from_replay(P.parse_replay(replay_text('rosevine', extra=cand_extra), engine='rosevine'),
                             **declared('hip', 'rosevine'))
    return ref, cand


CELL = {'id': 'x', 'family': 'engine', 'n_prompt': 4, 'n_decode': 3}


class Matrix(unittest.TestCase):
    def test_frozen_denominator_and_explicit_statuses(self):
        m = P.validate_matrix(P.matrix_v1())
        ids = [c['id'] for c in m['cells']]
        self.assertIn('decode-d16384', ids)
        self.assertIn('prefill-p128', ids)
        statuses = {c['id']: c['status'] for c in m['cells']}
        self.assertEqual(statuses['decode-d512'], 'supported')
        self.assertEqual(statuses['decode-d2048'], 'feasibility_check_required')
        self.assertEqual(statuses['ttft-warm-p512'], 'unsupported')
        self.assertEqual(P.matrix_v1()['matrix_sha256'], m['matrix_sha256'])

    def test_edited_matrix_is_rejected(self):
        m = P.matrix_v1()
        m['cells'] = [c for c in m['cells'] if c['id'] != 'decode-d16384']
        with self.assertRaises(ValueError):
            P.validate_matrix(m)


class Adapters(unittest.TestCase):
    def test_replay_rates(self):
        r = P.parse_replay(replay_text('rosevine'), engine='rosevine')
        self.assertEqual(len(r['repetitions']), 2)
        self.assertAlmostEqual(r['repetitions'][0]['decode_tokens_per_s'], 20.0)
        self.assertAlmostEqual(r['repetitions'][0]['prefill_tokens_per_s'], 10.0)

    def test_binding_hash_matches_both_native_adapters(self):
        # Value printed by `rosevine-inspect replay` and `parity_replay --validate-only`.
        self.assertEqual(P.tokens_fnv1a64([2, 818, 5279, 236761, 1003]), '91f01bff94253cd7')

    def test_incomplete_or_foreign_replay_rejected(self):
        text = replay_text('rosevine')
        with self.assertRaises(ValueError):
            P.parse_replay(text.rsplit('\n', 2)[0], engine='rosevine')  # one rep missing
        with self.assertRaises(ValueError):
            P.parse_replay(text, engine='llama.cpp')
        with self.assertRaises(ValueError):
            P.parse_replay(text.replace('[50000000, 50000000, 50000000]', '[50000000]', 1), engine='rosevine')
        with self.assertRaises(ValueError):
            P.parse_replay(text.replace('400000000', 'NaN', 1), engine='rosevine')

    def test_llama_bench_is_random_token_only(self):
        rows = P.parse_llama_bench_json(json.dumps([{'n_prompt': 0, 'n_gen': 128, 'n_depth': 512,
                                                      'samples_ts': [80.0, 81.0], 'backends': 'Vulkan',
                                                      'build_commit': 'cd959ba'}]))
        self.assertEqual(rows[0]['fixture'], 'random-token')
        self.assertEqual(rows[0]['phase'], 'decode')
        ref, cand = arms()
        ref['fixture_sha256'] = 'random-token'
        with self.assertRaises(ValueError):
            P.classify_pair(ref, cand, CELL, 'vulkan')


class Pairing(unittest.TestCase):
    def test_practical_when_activation_precision_differs(self):
        ref, cand = arms()
        result = P.classify_pair(ref, cand, CELL, 'vulkan')
        self.assertEqual(result['class'], 'practical_configuration')
        self.assertIn('cpu_expert_activation', result['state_differences'])

    def test_machinery_parity_when_state_matches(self):
        ref, cand = arms(cand_extra={'cpu_expert_activation': 'q8_0'})
        self.assertEqual(P.classify_pair(ref, cand, CELL, 'vulkan')['class'], 'machinery_parity')

    def test_hard_mismatches_rejected(self):
        for field, value in [('model_sha256', 'd' * 64), ('fixture_sha256', 'e' * 64),
                             ('kv_type', 'f16'), ('warm_state', 'cold'), ('power_policy', 'auto')]:
            ref, cand = arms(cand_extra={'cpu_expert_activation': 'q8_0'})
            cand[field] = value
            with self.assertRaises(ValueError, msg=field):
                P.classify_pair(ref, cand, CELL, 'vulkan')
        ref, cand = arms(cand_extra={'tokens_fnv1a64': '0000000000000001'})
        with self.assertRaises(ValueError):
            P.classify_pair(ref, cand, CELL, 'vulkan')

    def test_boundary_family_and_baseline_rules(self):
        ref, cand = arms()
        bad = copy.deepcopy(cand)
        bad['boundary'] = dict(P.ENGINE_BOUNDARY, sampling=True)
        with self.assertRaises(ValueError):
            P.validate_arm(bad)
        with self.assertRaises(ValueError):
            P.classify_pair(ref, cand, CELL, 'hip')  # reference is Vulkan
        with self.assertRaises(ValueError):
            P.classify_pair(ref, cand, dict(CELL, n_prompt=512), 'vulkan')
        with self.assertRaises(ValueError):
            P.classify_pair(cand, ref, CELL, 'vulkan')

    def test_reference_pin_must_match_series(self):
        ref, _ = arms()
        ref['series'] = 'newer-control'
        with self.assertRaises(ValueError):
            P.validate_arm(ref)
        ref['source_revision'] = P.NEWER_CONTROL_PIN
        P.validate_arm(ref)

    def test_declared_fact_contradicting_adapter_rejected(self):
        replay = P.parse_replay(replay_text('rosevine'), engine='rosevine')
        with self.assertRaises(ValueError):
            P.arm_from_replay(replay, **declared('hip', 'rosevine'), kv_type='f16')

    def test_missing_field_rejected(self):
        ref, _ = arms()
        del ref['affinity']
        with self.assertRaises(ValueError):
            P.validate_arm(ref)


class Statistics(unittest.TestCase):
    def test_ratio_directions(self):
        up = P.paired_ratio([100.0] * 7, [90.0] * 7, higher_is_better=True, seed=1, resamples=1000)
        self.assertAlmostEqual(up['ratio_median'], 0.9)
        down = P.paired_ratio([1.0] * 7, [2.0] * 7, higher_is_better=False, seed=1, resamples=1000)
        self.assertAlmostEqual(down['ratio_median'], 0.5)
        with self.assertRaises(ValueError):
            P.paired_ratio([1.0] * 6, [1.0] * 6, higher_is_better=True, seed=1)

    def test_milestones(self):
        flag = lambda point, lo: P.milestones({'ratio_median': point, 'ci95': [lo, point + .01]})
        self.assertEqual(flag(0.95, 0.91), {'p90': True, 'operational_parity': False,
                                             'strict_parity': False, 'specialization_advantage': False})
        self.assertTrue(flag(1.00, 0.985)['operational_parity'])
        self.assertFalse(flag(0.999, 0.985)['operational_parity'])
        self.assertTrue(flag(1.06, 1.01)['specialization_advantage'])
        self.assertFalse(flag(1.03, 1.01)['specialization_advantage'])

    def test_resource_budget_breach_visible(self):
        out = P.resource_ratios({'vram_peak_bytes': 10, 'cpu_core_seconds': 4},
                                {'vram_peak_bytes': 12, 'cpu_core_seconds': 8},
                                {'vram_peak_bytes': 11})
        self.assertTrue(out['vram_peak_bytes']['breach'])
        self.assertEqual(out['cpu_core_seconds']['ratio'], 2.0)
        self.assertIsNone(out['high_water_rss_bytes']['ratio'])

    def test_schedule_alternates(self):
        schedule = P.pair_schedule(['decode-d512'], seed=3)
        self.assertTrue(schedule)


class Report(unittest.TestCase):
    def test_report_keeps_denominator_and_never_counts_practical(self):
        m = P.matrix_v1()
        ratio = P.paired_ratio([100.0] * 7, [99.0] * 7, higher_is_better=True, seed=1, resamples=1000)
        ref, cand = arms()
        practical = P.classify_pair(ref, cand, CELL, 'vulkan')
        ref2, cand2 = arms(cand_extra={'cpu_expert_activation': 'q8_0'}, ref_backend='hip')
        parity = P.classify_pair(ref2, cand2, CELL, 'hip')
        report = P.build_report(m, {('decode-d512', 'vulkan'): {'pairing': practical, 'ratio': ratio,
                                                                 'correctness_pass': True},
                                    ('decode-d512', 'hip'): {'pairing': parity, 'ratio': ratio,
                                                              'correctness_pass': True}})
        self.assertEqual(len(report['rows']), 2 * len(m['cells']))
        counts = report['milestone_counts_by_baseline']
        self.assertEqual(counts['vulkan']['p90'], 0)
        self.assertEqual(counts['hip']['p90'], 1)
        self.assertEqual(counts['hip']['cells'], len(m['cells']))
        outcomes = {(r['cell'], r['baseline']): r['outcome'] for r in report['rows']}
        self.assertEqual(outcomes[('ttft-warm-p128', 'vulkan')], 'unsupported')
        self.assertEqual(outcomes[('prefill-p128', 'hip')], 'not_run')
        self.assertNotIn('aggregate', json.dumps(report))

    def test_correctness_failure_is_not_a_speed_result(self):
        m = P.matrix_v1()
        ratio = P.paired_ratio([100.0] * 7, [120.0] * 7, higher_is_better=True, seed=1, resamples=1000)
        ref, cand = arms(cand_extra={'cpu_expert_activation': 'q8_0'})
        report = P.build_report(m, {('decode-d512', 'vulkan'): {
            'pairing': P.classify_pair(ref, cand, CELL, 'vulkan'), 'ratio': ratio, 'correctness_pass': False}})
        self.assertEqual(report['rows'][0]['outcome'], 'correctness_failed')
        self.assertEqual(report['milestone_counts_by_baseline']['vulkan']['specialization_advantage'], 0)

    def test_comparison_outside_matrix_rejected(self):
        with self.assertRaises(ValueError):
            P.build_report(P.matrix_v1(), {('decode-d999', 'vulkan'): {}})


if __name__ == '__main__':
    unittest.main()
