#!/usr/bin/env python3
"""Analyze private short-trial records against a frozen paired schedule.

The caller supplies a plan and structured records, never arbitrary native logs.
Confounded configurations reject causal comparison. Unknown failures and missing
trials stay visible and prevent a complete comparison; no GO is inferred.
"""
import argparse
import json
import math
from pathlib import Path
import sys

from benchmark_core import canonical, digest, paired_improvement, paired_schedule, percentile, summarize, write_private_json
from verify_capture_bundle import secure_bytes, secure_json, verify_bundle

FACTORS = {'capacity_tokens', 'cpu_moe_layers', 'mtp', 'collector_enabled'}
SETTINGS = {'reference_build_sha256', 'target_sha256', 'assistant_identity',
            'projector_identity', 'capacity_tokens', 'cpu_moe_layers', 'mtp',
            'kv_policy', 'batch', 'microbatch', 'threads', 'output_tokens',
            'cache_policy', 'sampler_policy_sha256', 'thinking', 'collector_enabled',
            'safety_monitor'}
LEGACY_SETTINGS = SETTINGS - {'collector_enabled', 'safety_monitor'}


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_plan(plan):
    expected = {'schema', 'factor', 'configurations', 'workloads', 'metric',
                'seed', 'warmups', 'trials', 'bootstrap_resamples', 'invalid_conditions'}
    if not isinstance(plan, dict) or set(plan) != expected or plan['schema'] != 'rosevine-paired-plan-v1':
        raise ValueError('invalid paired plan schema')
    if plan['factor'] not in FACTORS or set(plan['configurations']) != {'reference', 'candidate'}:
        raise ValueError('named one-factor comparison required')
    if type(plan['seed']) is not int or plan['seed'] < 0 or type(plan['bootstrap_resamples']) is not int or plan['bootstrap_resamples'] < 1000:
        raise ValueError('seed and bootstrap count required')
    if not isinstance(plan['invalid_conditions'], list) or any(not isinstance(v, str) or not v.strip() for v in plan['invalid_conditions']) or len(set(plan['invalid_conditions'])) != len(plan['invalid_conditions']):
        raise ValueError('explicit unique invalid-condition IDs required')
    configs = plan['configurations']
    required_settings = SETTINGS if plan['factor'] == 'collector_enabled' else LEGACY_SETTINGS
    for config in configs.values():
        if (not isinstance(config, dict) or not required_settings <= set(config) or
                set(config) - SETTINGS or any(v is None for v in config.values())):
            raise ValueError('complete control settings required; unknown is not fixed')
        for name in ['reference_build_sha256', 'target_sha256', 'sampler_policy_sha256']:
            if not is_hash(config[name]):
                raise ValueError('invalid control identity hash')
        for name in ['capacity_tokens', 'batch', 'microbatch', 'threads', 'output_tokens']:
            if type(config[name]) is not int or config[name] <= 0:
                raise ValueError('positive integer control count required')
        if config['output_tokens'] != 512 or type(config['cpu_moe_layers']) is not int or not 0 <= config['cpu_moe_layers'] <= 30:
            raise ValueError('short protocol output count or first-N setting invalid')
        if any(type(config[name]) is not bool for name in ('mtp', 'thinking')):
            raise ValueError('explicit boolean control settings required')
        if 'collector_enabled' in config and type(config['collector_enabled']) is not bool:
            raise ValueError('collector state must be an explicit boolean')
        if plan['factor'] == 'collector_enabled':
            monitor = config['safety_monitor']
            if (not isinstance(monitor, dict) or set(monitor) != {'enabled', 'profile_sha256'} or
                    monitor['enabled'] is not True or not is_hash(monitor['profile_sha256'])):
                raise ValueError('collector perturbation requires an enabled, identified invariant safety monitor')
        for name in ['assistant_identity', 'projector_identity']:
            artifact = config[name]
            if not isinstance(artifact, dict):
                raise ValueError('explicit artifact identity or omission reason required')
            if set(artifact) == {'not_loaded_reason'}:
                if not isinstance(artifact['not_loaded_reason'], str) or not artifact['not_loaded_reason'].strip():
                    raise ValueError('artifact omission reason required')
                if name == 'assistant_identity' and config['mtp']:
                    raise ValueError('MTP control requires assistant artifact identity')
            elif set(artifact) != {'sha256', 'size_bytes'} or not is_hash(artifact.get('sha256')) or type(artifact.get('size_bytes')) is not int or artifact['size_bytes'] <= 0:
                raise ValueError('artifact hash and size required')
        canonical(config)
    ref, cand = configs['reference'], configs['candidate']
    differences = {key for key in set(ref) | set(cand) if canonical(ref.get(key)) != canonical(cand.get(key))}
    if differences != {plan['factor']}:
        raise ValueError('confounded or unchanged comparison; exactly the named factor may vary')
    metric = plan['metric']
    if set(metric) != {'name', 'unit', 'higher_is_better'} or any(not isinstance(metric[k], str) or not metric[k] for k in ['name', 'unit']) or type(metric['higher_is_better']) is not bool:
        raise ValueError('primary metric, unit and direction required')
    if (plan['factor'] == 'collector_enabled'
            and metric != {'name': 'end_to_end_s', 'unit': 's', 'higher_is_better': False}):
        raise ValueError('collector perturbation requires paired end-to-end wall time in seconds')
    if not isinstance(plan['workloads'], list) or not plan['workloads']:
        raise ValueError('frozen workload list required')
    ids = set()
    for workload in plan['workloads']:
        if set(workload) != {'id', 'rendered_sha256', 'tokens_sha256', 'occupied_input_tokens'} or not isinstance(workload['id'], str) or not workload['id'] or workload['id'] in ids:
            raise ValueError('invalid or duplicate workload')
        ids.add(workload['id'])
        if any(not is_hash(workload[k]) for k in ['rendered_sha256', 'tokens_sha256']):
            raise ValueError('workload token/render identities required')
        if workload['occupied_input_tokens'] not in (1900, 29000) or type(workload['occupied_input_tokens']) is not int:
            raise ValueError('short workload occupancy must be 1900 or 29000')
        if any(workload['occupied_input_tokens'] + 1024 > c['capacity_tokens'] for c in configs.values()):
            raise ValueError('control capacity does not reserve output and margin')
    # Also validates >=2 warmups and >=7 measured trials.
    return paired_schedule([w['id'] for w in plan['workloads']], ['reference', 'candidate'],
                           plan['seed'], plan['warmups'], plan['trials'])


def prepare_paired_package(plan):
    """Freeze the validated randomized schedule and bind it to the plan."""
    schedule = validate_plan(plan)
    schedule = [dict(row, ordinal=ordinal) for ordinal, row in enumerate(schedule)]
    counts = {'workloads': len(plan['workloads']), 'warmups_per_configuration': plan['warmups'],
              'measured_pairs_per_workload': plan['trials'], 'scheduled_records': len(schedule)}
    package = {'schema': 'rosevine-paired-package-v1', 'plan': plan,
               'plan_sha256': digest(plan), 'schedule': schedule,
               'schedule_sha256': digest(schedule), 'counts': counts}
    return package


def verify_paired_package(package):
    """Recompute plan and schedule identities before execution or analysis."""
    if not isinstance(package, dict) or set(package) != {
            'schema', 'plan', 'plan_sha256', 'schedule', 'schedule_sha256', 'counts'} or package['schema'] != 'rosevine-paired-package-v1':
        raise ValueError('invalid paired package schema')
    expected = prepare_paired_package(package['plan'])
    if package != expected:
        raise ValueError('paired package plan, schedule, counts or hashes do not verify')
    return expected


def verify_run_bundles(plan, records, run_bundles):
    if not isinstance(run_bundles, dict) or set(run_bundles) != {run['run_id'] for run in records}:
        raise ValueError('one supervisor bundle path is required for every submitted trial')
    paths = [Path(run_bundles[run['run_id']]).absolute() for run in records]
    if len(set(paths)) != len(paths):
        raise ValueError('each paired trial must have a distinct supervisor bundle')
    cycle_data = {}
    for run in records:
        path = Path(run_bundles[run['run_id']])
        summary = verify_bundle(path)
        manifest = secure_json(path / 'manifest.json')
        expected_trial = {key: run[key] for key in (
            'run_id', 'ordinal', 'phase', 'pair', 'workload', 'configuration', 'seed',
            'configuration_sha256', 'tokens_sha256', 'rendered_sha256')}
        if manifest.get('trial_identity') != expected_trial:
            raise ValueError('supervisor bundle trial identity differs from paired record')
        result = secure_json(path / 'result.json')
        if summary['run_status'] != run['status']:
            raise ValueError('supervisor result status differs from paired record')
        measured = result.get('metrics', {}).get(plan['metric']['name'])
        if (not isinstance(measured, dict) or set(measured) != {'value', 'reason', 'unit'}
                or measured != run['metric']):
            raise ValueError('paired primary metric differs from supervisor result')
        if plan['factor'] == 'collector_enabled':
            expected_enabled = plan['configurations'][run['configuration']]['collector_enabled']
            if manifest.get('measurement_telemetry_enabled') is not expected_enabled:
                raise ValueError('supervisor measurement collector state differs from paired configuration')
            durations = result.get('measurement_collector_cycle_durations_ns')
            if not expected_enabled:
                if durations or result.get('measurement_telemetry_bytes') != 0:
                    raise ValueError('collector-disabled run retained optional measurement samples')
                cycle_data[run['run_id']] = {'enabled': False, 'preflight': [], 'running': []}
            elif not isinstance(durations, list):
                cycle_data[run['run_id']] = None
            else:
                raw_rows = secure_bytes(path / 'measurement-telemetry.jsonl', 32 * 1024 * 1024)
                telemetry_rows = [json.loads(line) for line in raw_rows.splitlines()]
                if len(telemetry_rows) != len(durations):
                    raise ValueError('measurement collector cycle and row counts differ')
                by_stage = {'preflight': [], 'running': []}
                for row, duration in zip(telemetry_rows, durations):
                    if row.get('stage') not in by_stage:
                        raise ValueError('unexpected measurement collector telemetry stage')
                    by_stage[row['stage']].append(duration)
                cycle_data[run['run_id']] = dict(enabled=True, **by_stage)
    return {'verified': True, 'verified_trial_bundles': len(records),
            'collector_cycle_data': cycle_data}


def analyze(plan, records, *, run_bundles=None):
    schedule = validate_plan(plan)
    if not isinstance(records, list):
        raise ValueError('run record list required')
    workloads = {w['id']: w for w in plan['workloads']}
    runs, run_ids = {}, set()
    config_hashes = {key: digest(value) for key, value in plan['configurations'].items()}
    for run in records:
        expected = {'run_id', 'ordinal', 'phase', 'pair', 'workload', 'configuration', 'seed',
                    'configuration_sha256', 'tokens_sha256', 'rendered_sha256',
                    'status', 'invalid_condition', 'metric'}
        if not isinstance(run, dict) or set(run) != expected:
            raise ValueError('invalid private trial schema')
        ordinal = run['ordinal']
        if type(ordinal) is not int or not 0 <= ordinal < len(schedule) or ordinal in runs:
            raise ValueError('unplanned or duplicate trial ordinal')
        if not isinstance(run['run_id'], str) or not run['run_id'] or run['run_id'] in run_ids:
            raise ValueError('unique run IDs required')
        run_ids.add(run['run_id'])
        if type(run['pair']) is not int or type(run['seed']) is not int:
            raise ValueError('integer trial pair and seed required')
        planned = schedule[ordinal]
        if any(run[key] != planned[key] for key in planned):
            raise ValueError('trial does not match preregistered order/pair/seed')
        if run['configuration_sha256'] != config_hashes[run['configuration']]:
            raise ValueError('configuration identity mismatch')
        if any(run[key] != workloads[run['workload']][key] for key in ['tokens_sha256', 'rendered_sha256']):
            raise ValueError('paired prompt/token identity mismatch')
        if run['status'] not in ('successful', 'failed', 'incomplete'):
            raise ValueError('terminal trial status required')
        if run['invalid_condition'] is not None and run['invalid_condition'] not in plan['invalid_conditions']:
            raise ValueError('unregistered invalid condition')
        metric = run['metric']
        if not isinstance(metric, dict) or set(metric) != {'value', 'reason', 'unit'} or metric['unit'] != plan['metric']['unit']:
            raise ValueError('metric units/schema mismatch')
        if metric['value'] is None:
            if not isinstance(metric['reason'], str) or not metric['reason'].strip():
                raise ValueError('missing metric reason required')
        elif metric['reason'] is not None or type(metric['value']) not in (int, float) or not math.isfinite(metric['value']) or metric['value'] < 0:
            raise ValueError('finite nonnegative metric or explicit missing reason required')
        runs[ordinal] = run
    provenance = (verify_run_bundles(plan, records, run_bundles)
                  if run_bundles is not None else
                  {'verified': False, 'verified_trial_bundles': 0})
    missing = [i for i in range(len(schedule)) if i not in runs]
    invalid = [{'ordinal': i, 'run_id': r['run_id'], 'status': r['status'],
                'invalid_condition': r['invalid_condition'], 'metric_reason': r['metric']['reason']}
               for i, r in sorted(runs.items()) if r['status'] != 'successful' or
               r['invalid_condition'] is not None or r['metric']['value'] is None]
    unplanned_failure = any((r['status'] != 'successful' or r['metric']['value'] is None) and
                            r['invalid_condition'] is None for r in runs.values())
    results = {}
    for workload in workloads:
        eligible, eligible_runs, excluded = [], [], []
        for pair in range(plan['trials']):
            selected = {r['configuration']: r for r in runs.values()
                        if r['phase'] == 'measured' and r['workload'] == workload and r['pair'] == pair}
            if len(selected) != 2 or any(r['status'] != 'successful' or r['invalid_condition'] is not None or
                                         r['metric']['value'] is None for r in selected.values()):
                excluded.append(pair)
            else:
                eligible.append([selected[k]['metric']['value'] for k in ('reference', 'candidate')])
                eligible_runs.append([selected[k] for k in ('reference', 'candidate')])
        ref = [pair[0] for pair in eligible]
        cand = [pair[1] for pair in eligible]
        # Missing warmups/trials or an unregistered failure prevents inference.
        warmups_ready = all(sum(1 for r in runs.values() if r['phase'] == 'warmup' and
                                  r['workload'] == workload and r['configuration'] == config and
                                  r['status'] == 'successful' and r['invalid_condition'] is None) >= 2
                            for config in ('reference', 'candidate'))
        complete = not missing and not unplanned_failure and warmups_ready and len(eligible) >= 7
        comparison = None
        reason = None
        if not complete:
            reason = 'missing trials, unregistered failure, insufficient warmups or fewer than seven eligible pairs'
        elif any(value <= 0 for value in ref):
            reason = 'relative improvement undefined for zero reference metric'
        else:
            comparison = paired_improvement(ref, cand, higher_is_better=plan['metric']['higher_is_better'],
                                            seed=plan['seed'], resamples=plan['bootstrap_resamples'])
        workload_result = {'eligible_pairs': len(eligible), 'warmups_ready': warmups_ready,
                           'excluded_pair_ids': excluded,
                           'reference': summarize(ref) if ref else None,
                           'candidate': summarize(cand) if cand else None,
                           'comparison': comparison, 'comparison_missing_reason': reason}
        if plan['factor'] == 'collector_enabled':
            if comparison is None:
                perturbation = None
            else:
                # paired_improvement uses positive values for faster candidate;
                # collector overhead is the signed time increase instead.
                benefit_ci = comparison['ci95_percent']
                overhead_ci = [-benefit_ci[1], -benefit_ci[0]]
                perturbation = {
                    'paired_median_workload_time_increase_percent': -comparison['median_benefit_percent'],
                    'paired_ci95_workload_time_increase_percent': overhead_ci,
                    'equivalence_target_abs_percent': 1.0,
                    'within_target': (None if not provenance['verified'] else
                                      overhead_ci[0] > -1.0 and overhead_ci[1] < 1.0),
                    'run_bundle_provenance_verified': provenance['verified'],
                    'interpretation': 'candidate is collector enabled; positive is slowdown; strict +/-1% equivalence interval required'}
            workload_result['collector_workload_perturbation'] = perturbation
            cycle_data = provenance.get('collector_cycle_data', {})
            telemetry_cycle_cost = {'verified': False, 'target_p95_ns': 1_000_000,
                                    'per_configuration': {},
                                    'scope': 'optional measurement collector worker plus serialization, write and flush; invariant safety telemetry is not included; separate from end-to-end perturbation'}
            if provenance['verified']:
                for configuration in ('reference', 'candidate'):
                    selected_runs = [pair[0 if configuration == 'reference' else 1]
                                     for pair in eligible_runs]
                    enabled = plan['configurations'][configuration]['collector_enabled']
                    if not enabled:
                        telemetry_cycle_cost['per_configuration'][configuration] = {
                            'enabled': False, 'cycle_count': 0, 'eligible_runs': len(selected_runs),
                            'pooled_running_p95_ns': None, 'within_target': None,
                            'reason': 'optional measurement collector disabled by plan'}
                        continue
                    samples, missing_cycle_runs = [], []
                    for selected_run in selected_runs:
                        stage_data = cycle_data.get(selected_run['run_id'])
                        if not isinstance(stage_data, dict) or not stage_data.get('enabled') or not stage_data['running']:
                            missing_cycle_runs.append(selected_run['run_id'])
                        else:
                            samples.extend(stage_data['running'])
                    if missing_cycle_runs:
                        telemetry_cycle_cost['per_configuration'][configuration] = {
                            'enabled': True,
                            'cycle_count': len(samples), 'eligible_runs': len(selected_runs),
                            'missing_cycle_run_ids': missing_cycle_runs,
                            'pooled_running_p95_ns': None, 'within_target': None,
                            'reason': 'one or more eligible runs lack retained running collector cycles'}
                    else:
                        p95 = percentile(samples, .95) if samples else None
                        telemetry_cycle_cost['per_configuration'][configuration] = {
                            'enabled': True,
                            'cycle_count': len(samples), 'eligible_runs': len(selected_runs),
                            'missing_cycle_run_ids': [], 'pooled_running_p95_ns': p95,
                            'within_target': p95 is not None and p95 < 1_000_000,
                            'reason': None if p95 is not None else 'no eligible running collector cycles'}
                telemetry_cycle_cost['verified'] = all(
                    not record['enabled'] or
                    record['pooled_running_p95_ns'] is not None
                    for record in telemetry_cycle_cost['per_configuration'].values())
            else:
                telemetry_cycle_cost['reason'] = 'verified supervisor bundles are required'
            workload_result['collector_telemetry_cycle_cost'] = telemetry_cycle_cost
        results[workload] = workload_result
    return {'schema': 'rosevine-paired-analysis-v1', 'plan_sha256': digest(plan),
            'run_bundle_provenance_verified': provenance['verified'],
            'verified_trial_bundles': provenance['verified_trial_bundles'],
            'status': 'complete' if not missing and not invalid else 'incomplete',
            'metric': plan['metric'], 'named_factor': plan['factor'],
            'missing_trial_ordinals': missing, 'invalid_or_failed_runs': invalid,
            'per_workload': results, 'acceptance': 'not decided; requires correctness, resources and reviewed gate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?', help='private plan or analysis-input JSON file')
    parser.add_argument('--prepare-plan', action='store_true',
                        help='validate a plan and freeze its reproducible randomized schedule')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input is None:
        parser.error('an input file is required')
    data = json.loads(args.input.read_text())
    if args.prepare_plan:
        write_private_json(args.output, prepare_paired_package(data))
        print('Private paired schedule prepared and hash-bound; no workload launched.')
    else:
        if set(data) in ({'package', 'runs'}, {'package', 'runs', 'run_bundles'}):
            package = verify_paired_package(data['package'])
            summary = analyze(package['plan'], data['runs'], run_bundles=data.get('run_bundles'))
            summary['schedule_sha256'] = package['schedule_sha256']
        elif set(data) in ({'plan', 'runs'}, {'plan', 'runs', 'run_bundles'}):
            summary = analyze(data['plan'], data['runs'], run_bundles=data.get('run_bundles'))
        else:
            raise ValueError('private analysis input requires plan/runs or package/runs')
        write_private_json(args.output, summary)
        print('Private paired analysis complete; no optimization GO asserted.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Paired analysis failed; inspect private plan/records locally.', file=sys.stderr)
        sys.exit(1)
