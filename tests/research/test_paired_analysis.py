"""Preregistered pairing, confounding, exclusions and invalid/missing outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_core as B
SPEC = importlib.util.spec_from_file_location('paired_analysis', ROOT / 'scripts/paired_analysis.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def fixture(trials=7):
    config = dict(reference_build_sha256='a' * 64, target_sha256='b' * 64,
                  assistant_identity={'not_loaded_reason': 'target-only control'},
                  projector_identity={'not_loaded_reason': 'separately named text-only control'},
                  capacity_tokens=65536, cpu_moe_layers=1, mtp=False, kv_policy='q8_0 CPU',
                  batch=512, microbatch=128, threads=2, output_tokens=512,
                  cache_policy='no reused prompt/KV cache', sampler_policy_sha256='c' * 64,
                  thinking=True, collector_enabled=False)
    plan = dict(schema='rosevine-paired-plan-v1', factor='capacity_tokens',
                configurations={'reference': config, 'candidate': dict(config, capacity_tokens=131072)},
                workloads=[dict(id='synthetic-short', rendered_sha256='d' * 64,
                                tokens_sha256='e' * 64, occupied_input_tokens=1900)],
                metric=dict(name='end_to_end_s', unit='s', higher_is_better=False),
                seed=42, warmups=2, trials=trials, bootstrap_resamples=1000,
                invalid_conditions=['contention', 'resource_budget'])
    schedule = P.validate_plan(plan)
    runs = []
    for i, row in enumerate(schedule):
        run = dict(row, ordinal=i, run_id='synthetic-' + str(i),
                   configuration_sha256=P.digest(plan['configurations'][row['configuration']]),
                   rendered_sha256='d' * 64, tokens_sha256='e' * 64, status='successful',
                   invalid_condition=None, metric=dict(value=100 if row['configuration'] == 'reference' else 90,
                                                      reason=None, unit='s'))
        runs.append(run)
    return plan, runs


class PairedAnalysisTests(unittest.TestCase):
    def test_complete_known_benefit_and_record_order_independence(self):
        plan, runs = fixture()
        result = P.analyze(plan, list(reversed(runs)))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['per_workload']['synthetic-short']['eligible_pairs'], 7)
        self.assertEqual(result['per_workload']['synthetic-short']['comparison']['ci95_percent'], [10, 10])
        self.assertEqual(result['missing_trial_ordinals'], [])
        self.assertFalse(result['run_bundle_provenance_verified'])

    def test_run_bundles_bind_each_trial_identity_status_and_primary_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, runs = fixture()
            bundles = {}
            for run in runs:
                value = run['metric']['value']

                def adapter(_result, _samples, metric_value=value):
                    inventory = B.missing_metrics('not emitted by paired bundle fixture')
                    inventory[plan['metric']['name']] = {
                        'value': metric_value, 'reason': None, 'unit': plan['metric']['unit']}
                    return {'paired synthetic source': inventory}

                trial = {key: run[key] for key in (
                    'run_id', 'ordinal', 'phase', 'pair', 'workload', 'configuration',
                    'seed', 'configuration_sha256', 'tokens_sha256', 'rendered_sha256')}
                bundle = root / f"bundle-{run['ordinal']}"
                B.supervise([sys.executable, '-c', 'pass'], bundle, timeout_s=2,
                            run_identity={'identity_sha256': run['configuration_sha256']},
                            trial_identity=trial, metric_adapter=adapter)
                bundles[run['run_id']] = bundle
            verified = P.analyze(plan, runs, run_bundles=bundles)
            self.assertTrue(verified['run_bundle_provenance_verified'])
            self.assertEqual(verified['verified_trial_bundles'], len(runs))
            tampered = [dict(run) for run in runs]
            tampered[0] = dict(tampered[0], metric=dict(tampered[0]['metric'], value=999))
            with self.assertRaisesRegex(ValueError, 'primary metric differs'):
                P.analyze(plan, tampered, run_bundles=bundles)

    def test_missing_and_failed_trials_are_retained_without_comparison(self):
        plan, runs = fixture()
        failed = next(r for r in runs if r['phase'] == 'measured')
        failed.update(status='failed', metric=dict(value=None, reason='allocation failure', unit='s'))
        result = P.analyze(plan, runs)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['invalid_or_failed_runs'][0]['run_id'], failed['run_id'])
        self.assertIsNone(result['per_workload']['synthetic-short']['comparison'])
        result = P.analyze(plan, runs[:-1])
        self.assertEqual(result['missing_trial_ordinals'], [len(runs) - 1])

    def test_preregistered_exclusion_removes_whole_pair_and_requires_seven_remaining(self):
        plan, runs = fixture(trials=8)
        selected = next(r for r in runs if r['phase'] == 'measured' and r['pair'] == 0)
        selected['invalid_condition'] = 'contention'
        result = P.analyze(plan, runs)
        summary = result['per_workload']['synthetic-short']
        self.assertEqual(summary['eligible_pairs'], 7)
        self.assertEqual(summary['excluded_pair_ids'], [0])
        self.assertEqual(summary['comparison']['ci95_percent'], [10, 10])
        # A failed warmup is not evidence of two successful warmups.
        warmup = next(r for r in runs if r['phase'] == 'warmup')
        warmup.update(status='failed', invalid_condition='resource_budget')
        self.assertFalse(P.analyze(plan, runs)['per_workload']['synthetic-short']['warmups_ready'])
        self.assertIsNone(P.analyze(plan, runs)['per_workload']['synthetic-short']['comparison'])

    def test_configuration_confounding_is_rejected(self):
        plan, _ = fixture()
        plan['configurations']['candidate']['threads'] = 4
        with self.assertRaises(ValueError):
            P.validate_plan(plan)

    def test_collector_perturbation_is_a_supported_single_factor(self):
        plan, runs = fixture()
        plan['factor'] = 'collector_enabled'
        plan['configurations']['candidate']['capacity_tokens'] = 65536
        plan['configurations']['candidate']['collector_enabled'] = True
        safety_monitor = {'enabled': True, 'profile_sha256': 'f' * 64}
        plan['configurations']['reference']['safety_monitor'] = safety_monitor
        plan['configurations']['candidate']['safety_monitor'] = safety_monitor
        for run in runs:
            config = run['configuration']
            run['configuration_sha256'] = P.digest(plan['configurations'][config])
            run['metric']['value'] = 100 if config == 'reference' else 101
        P.validate_plan(plan)
        result = P.analyze(plan, runs)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['named_factor'], 'collector_enabled')
        self.assertEqual(result['per_workload']['synthetic-short']['comparison']['median_benefit_percent'], -1)
        perturbation = result['per_workload']['synthetic-short']['collector_workload_perturbation']
        self.assertEqual(perturbation['paired_median_workload_time_increase_percent'], 1)
        self.assertEqual(perturbation['paired_ci95_workload_time_increase_percent'], [1, 1])
        self.assertFalse(perturbation['run_bundle_provenance_verified'])
        self.assertIsNone(perturbation['within_target'])
        plan['configurations']['candidate']['threads'] = 4
        with self.assertRaises(ValueError):
            P.validate_plan(plan)

    def test_collector_bundle_analysis_pools_running_cycle_p95_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, runs = fixture()
            plan['factor'] = 'collector_enabled'
            plan['configurations']['candidate']['capacity_tokens'] = 65536
            plan['configurations']['candidate']['collector_enabled'] = True
            safety = {'enabled': True, 'profile_sha256': 'f' * 64}
            plan['configurations']['reference']['safety_monitor'] = safety
            plan['configurations']['candidate']['safety_monitor'] = safety
            bundles = {}
            for run in runs:
                config = run['configuration']
                run['configuration_sha256'] = P.digest(plan['configurations'][config])
                value = run['metric']['value']

                def adapter(_result, _samples, measured=value):
                    inventory = B.missing_metrics('not emitted by paired bundle fixture')
                    inventory['end_to_end_s'] = {'value': measured, 'reason': None, 'unit': 's'}
                    return {'paired synthetic source': inventory}

                trial = {key: run[key] for key in (
                    'run_id', 'ordinal', 'phase', 'pair', 'workload', 'configuration',
                    'seed', 'configuration_sha256', 'tokens_sha256', 'rendered_sha256')}
                bundle = root / ('bundle-' + str(run['ordinal']))
                measurement_command = None
                if plan['configurations'][config]['collector_enabled']:
                    measurement_command = [sys.executable, '-u', '-c',
                        'import sys; sys.path.insert(0, ' + repr(str(ROOT / 'scripts')) +
                        '); from collector_worker import serve; serve(lambda *_: {"sample": 1})']
                B.supervise([sys.executable, '-c', 'import time; time.sleep(.025)'], bundle,
                            timeout_s=1, cadence_s=.005, telemetry=lambda *_: {'fixed': 1},
                            measurement_telemetry_command=measurement_command,
                            measurement_telemetry_timeout_s=.5,
                            run_identity={'identity_sha256': run['configuration_sha256']},
                            trial_identity=trial, metric_adapter=adapter)
                bundles[run['run_id']] = bundle
            result = P.analyze(plan, runs, run_bundles=bundles)
            summary = result['per_workload']['synthetic-short']
            cycle_summary = summary['collector_telemetry_cycle_cost']
            self.assertTrue(cycle_summary['verified'])
            self.assertFalse(cycle_summary['per_configuration']['reference']['enabled'])
            self.assertTrue(cycle_summary['per_configuration']['candidate']['enabled'])
            self.assertIsNone(cycle_summary['per_configuration']['reference']['pooled_running_p95_ns'])
            for config in ('candidate',):
                samples = []
                for run in runs:
                    if run['phase'] != 'measured' or run['configuration'] != config:
                        continue
                    bundle = bundles[run['run_id']]
                    durations = json.loads((bundle / 'result.json').read_text())['measurement_collector_cycle_durations_ns']
                    rows = [json.loads(line) for line in (bundle / 'measurement-telemetry.jsonl').read_text().splitlines()]
                    samples.extend(duration for row, duration in zip(rows, durations)
                                   if row['stage'] == 'running')
                expected = B.percentile(samples, .95)
                self.assertEqual(cycle_summary['per_configuration'][config]['pooled_running_p95_ns'],
                                 expected)

    def test_collector_factor_requires_explicit_boolean_state(self):
        plan, _ = fixture()
        plan['configurations']['reference']['collector_enabled'] = 'enabled'
        with self.assertRaises(ValueError):
            P.validate_plan(plan)

    def test_collector_factor_requires_end_to_end_time_metric(self):
        plan, _ = fixture()
        plan['factor'] = 'collector_enabled'
        plan['configurations']['candidate']['capacity_tokens'] = 65536
        plan['configurations']['candidate']['collector_enabled'] = True
        safety = {'enabled': True, 'profile_sha256': 'f' * 64}
        plan['configurations']['reference']['safety_monitor'] = safety
        plan['configurations']['candidate']['safety_monitor'] = safety
        plan['metric'] = {'name': 'errors', 'unit': 'count', 'higher_is_better': False}
        with self.assertRaisesRegex(ValueError, 'end-to-end wall time'):
            P.validate_plan(plan)

    def test_collector_perturbation_requires_same_enabled_safety_monitor(self):
        plan, _ = fixture()
        plan['factor'] = 'collector_enabled'
        plan['configurations']['candidate']['capacity_tokens'] = 65536
        plan['configurations']['candidate']['collector_enabled'] = True
        enabled = {'enabled': True, 'profile_sha256': 'f' * 64}
        plan['configurations']['reference']['safety_monitor'] = enabled
        plan['configurations']['candidate']['safety_monitor'] = enabled
        P.validate_plan(plan)
        plan['configurations']['candidate']['safety_monitor'] = {
            'enabled': False, 'profile_sha256': 'f' * 64}
        with self.assertRaisesRegex(ValueError, 'invariant safety monitor'):
            P.validate_plan(plan)
        plan['configurations']['candidate']['safety_monitor'] = {
            'enabled': True, 'profile_sha256': 'e' * 64}
        with self.assertRaisesRegex(ValueError, 'exactly the named factor'):
            P.validate_plan(plan)

    def test_prepared_collector_schedule_is_reproducible_and_hash_bound(self):
        plan, _ = fixture()
        plan['factor'] = 'collector_enabled'
        plan['configurations']['candidate']['capacity_tokens'] = 65536
        plan['configurations']['candidate']['collector_enabled'] = True
        monitor = {'enabled': True, 'profile_sha256': 'f' * 64}
        plan['configurations']['reference']['safety_monitor'] = monitor
        plan['configurations']['candidate']['safety_monitor'] = monitor
        package = P.prepare_paired_package(plan)
        self.assertEqual(package, P.prepare_paired_package(plan))
        self.assertEqual(package['counts']['scheduled_records'], 18)
        self.assertEqual(package['schedule_sha256'], P.digest(package['schedule']))
        self.assertEqual(P.verify_paired_package(package), package)
        for phase, pair_count in [('warmup', plan['warmups']), ('measured', plan['trials'])]:
            for pair in range(pair_count):
                arms = [r for r in package['schedule'] if r['phase'] == phase and r['pair'] == pair]
                self.assertEqual({r['configuration'] for r in arms}, {'reference', 'candidate'})
                self.assertEqual({r['seed'] for r in arms}, {plan['seed'] + pair})
        tampered = copy.deepcopy(package)
        original = tampered['schedule'][0]['configuration']
        tampered['schedule'][0]['configuration'] = 'candidate' if original == 'reference' else 'reference'
        with self.assertRaisesRegex(ValueError, 'do not verify'):
            P.verify_paired_package(tampered)

    def test_legacy_factor_plans_remain_valid_without_collector_field(self):
        plan, _ = fixture()
        for config in plan['configurations'].values():
            del config['collector_enabled']
        P.validate_plan(plan)

    def test_identity_order_seed_units_and_unregistered_exclusions_rejected(self):
        plan, runs = fixture()
        for field, value in [('tokens_sha256', 'f' * 64), ('configuration_sha256', 'f' * 64),
                             ('seed', 100), ('pair', True), ('invalid_condition', 'posthoc-excuse')]:
            bad = copy.deepcopy(runs)
            bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                P.analyze(plan, bad)
        bad = copy.deepcopy(runs)
        bad[0]['metric']['unit'] = 'ms'
        with self.assertRaises(ValueError):
            P.analyze(plan, bad)
        with self.assertRaises(ValueError):
            P.analyze(plan, runs + [runs[0]])

    def test_zero_reference_is_not_a_fabricated_relative_improvement(self):
        plan, runs = fixture()
        for run in runs:
            run['metric']['value'] = 0
        result = P.analyze(plan, runs)['per_workload']['synthetic-short']
        self.assertIsNone(result['comparison'])
        self.assertIn('zero reference', result['comparison_missing_reason'])

    def test_missing_metric_without_preregistered_condition_blocks_inference(self):
        plan, runs = fixture(trials=8)
        missing = next(r for r in runs if r['phase'] == 'measured')
        missing['metric'] = dict(value=None, reason='counter failed unexpectedly', unit='s')
        result = P.analyze(plan, runs)['per_workload']['synthetic-short']
        self.assertEqual(result['eligible_pairs'], 7)
        self.assertIsNone(result['comparison'])
