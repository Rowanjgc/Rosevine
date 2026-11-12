#!/usr/bin/env python3
"""Freeze stochastic seed plans and finalize only complete supervised captures.

This module never launches a model. It binds each planned prompt/seed to a
successful supervisor bundle and leaves an explicit private run inventory when
any planned sample is missing or failed.
"""
import argparse
import hashlib
from pathlib import Path
import sys

from benchmark_core import canonical, digest, write_private_json
import stochastic_distribution as distribution
from stochastic_sampler_events import parse as parse_sampler_events
from verify_capture_bundle import secure_bytes, secure_json, verify_bundle


PLAN_FIELDS = {'schema', 'model_sha256', 'sampler_policy_sha256', 'sampler_observer_sha256', 'vocabulary_size',
               'eos_token_id', 'max_output_tokens', 'seed_schedule', 'prompts'}


def prepare_plan(value):
    if not isinstance(value, dict) or set(value) != PLAN_FIELDS or value['schema'] != 'rosevine-stochastic-plan-v2':
        raise ValueError('stochastic run plan schema mismatch')
    if any(not distribution.is_hash(value[key]) for key in
           ('model_sha256', 'sampler_policy_sha256', 'sampler_observer_sha256')):
        raise ValueError('stochastic model, sampler-policy and observer hashes are required')
    vocab, eos, maximum = value['vocabulary_size'], value['eos_token_id'], value['max_output_tokens']
    if (type(vocab) is not int or vocab <= 1 or type(eos) is not int or not 0 <= eos < vocab
            or type(maximum) is not int or maximum <= 0):
        raise ValueError('invalid stochastic vocabulary, EOS or output bound')
    seeds = value['seed_schedule']
    if (not isinstance(seeds, list) or len(seeds) < 20
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('at least 20 distinct nonnegative seeds are required')
    prompts = value['prompts']
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('stochastic plan requires prompts')
    ids, prompt_hashes = set(), set()
    for prompt in prompts:
        if (not isinstance(prompt, dict) or set(prompt) != {'prompt_id', 'prompt_sha256'}
                or not isinstance(prompt['prompt_id'], str) or not prompt['prompt_id']
                or prompt['prompt_id'] in ids or not distribution.is_hash(prompt['prompt_sha256'])
                or prompt['prompt_sha256'] in prompt_hashes):
            raise ValueError('invalid or duplicate stochastic prompt identity')
        ids.add(prompt['prompt_id'])
        prompt_hashes.add(prompt['prompt_sha256'])
    frozen = dict(value)
    frozen['plan_sha256'] = digest(value)
    return frozen


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS | {'plan_sha256'}:
        raise ValueError('frozen stochastic plan schema mismatch')
    body = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if plan != prepare_plan(body):
        raise ValueError('stochastic plan hash or normalized content mismatch')
    return plan


def _sample_identity(plan, prompt, seed):
    return {'schema': 'rosevine-stochastic-sample-v2',
            'model_sha256': plan['model_sha256'],
            'sampler_policy_sha256': plan['sampler_policy_sha256'],
            'sampler_observer_sha256': plan['sampler_observer_sha256'],
            'prompt_id': prompt['prompt_id'], 'prompt_sha256': prompt['prompt_sha256'],
            'seed': seed, 'vocabulary_size': plan['vocabulary_size'],
            'eos_token_id': plan['eos_token_id'],
            'max_output_tokens': plan['max_output_tokens']}


def finalize_capture(plan, bundle_map, bundle_root):
    validate_plan(plan)
    if (not isinstance(bundle_map, dict) or set(bundle_map) != {'schema', 'plan_sha256', 'runs'}
            or bundle_map['schema'] != 'rosevine-stochastic-bundle-map-v1'
            or bundle_map['plan_sha256'] != plan['plan_sha256']
            or not isinstance(bundle_map['runs'], list)):
        raise ValueError('stochastic bundle map does not match frozen plan')
    prompts = {prompt['prompt_id']: prompt for prompt in plan['prompts']}
    expected_keys = {(prompt_id, seed) for prompt_id in prompts for seed in plan['seed_schedule']}
    run_map, duplicate_keys = {}, set()
    for row in bundle_map['runs']:
        if (not isinstance(row, dict) or set(row) != {'prompt_id', 'seed', 'run_bundle'}
                or row['prompt_id'] not in prompts or type(row['seed']) is not int
                or row['seed'] not in plan['seed_schedule']
                or not isinstance(row['run_bundle'], str) or not row['run_bundle']):
            raise ValueError('invalid stochastic run bundle mapping')
        key = (row['prompt_id'], row['seed'])
        if key in run_map:
            duplicate_keys.add(key)
        run_map[key] = row['run_bundle']
    path_counts = {}
    for raw_path in run_map.values():
        path_counts[raw_path] = path_counts.get(raw_path, 0) + 1
    duplicate_paths = {path for path, count in path_counts.items() if count > 1}
    unknown_keys = set(run_map) - expected_keys
    unknown_rows = [row for row in bundle_map['runs']
                    if (row['prompt_id'], row['seed']) in unknown_keys]
    failures = []
    capture_prompts = []
    run_records = []
    root = Path(bundle_root).absolute()
    for prompt_id in sorted(prompts):
        prompt = prompts[prompt_id]
        samples = []
        for seed in plan['seed_schedule']:
            key = (prompt_id, seed)
            raw_path = run_map.get(key)
            record = {'prompt_id': prompt_id, 'seed': seed, 'run_bundle': raw_path,
                      'status': 'missing', 'reason_code': 'bundle_not_provided',
                      'checksums_index_sha256': None}
            if raw_path is not None:
                if key in duplicate_keys or raw_path in duplicate_paths:
                    record.update(status='invalid', reason_code='duplicate_bundle_mapping')
                else:
                    try:
                        path = distribution.resolve_bundle_path(root, raw_path)
                        summary = verify_bundle(path)
                        record['checksums_index_sha256'] = summary['checksum_index_sha256']
                        if summary['run_status'] != 'successful':
                            record.update(status=summary['run_status'], reason_code='supervised_run_not_successful')
                        else:
                            identity = _sample_identity(plan, prompt, seed)
                            stderr = secure_bytes(path / 'stderr.txt', 8 * 1024 * 1024)
                            receipt = parse_sampler_events(stderr,
                                vocabulary_size=plan['vocabulary_size'],
                                eos_token_id=plan['eos_token_id'],
                                max_output_tokens=plan['max_output_tokens'])
                            samples.append({'seed': seed, 'status': 'successful',
                                            'finish_reason': receipt['finish_reason'],
                                            'token_ids': receipt['token_ids'],
                                            'run_bundle': raw_path})
                            record.update(status='successful', reason_code=None)
                    except Exception as error:
                        record.update(status='invalid', reason_code=type(error).__name__)
            if record['status'] != 'successful':
                failures.append(key)
            run_records.append(record)
        if len(samples) == len(plan['seed_schedule']):
            capture_prompts.append({'prompt_id': prompt_id,
                                    'prompt_sha256': prompt['prompt_sha256'],
                                    'samples': samples})
    if unknown_keys:
        failures.extend(sorted(unknown_keys))
        run_records.extend({'prompt_id': row['prompt_id'], 'seed': row['seed'],
                            'run_bundle': row['run_bundle'], 'status': 'invalid',
                            'reason_code': 'unplanned_bundle_mapping',
                            'checksums_index_sha256': None}
                           for row in unknown_rows)
    planned_records = [row for row in run_records
                       if (row['prompt_id'], row['seed']) in expected_keys]
    capture = None
    finalization_error = None
    if not failures and len(capture_prompts) == len(prompts):
        capture = {'schema': 'rosevine-stochastic-output-v3',
                   'model_sha256': plan['model_sha256'],
                   'sampler_policy_sha256': plan['sampler_policy_sha256'],
                   'sampler_observer_sha256': plan['sampler_observer_sha256'],
                   'vocabulary_size': plan['vocabulary_size'],
                   'eos_token_id': plan['eos_token_id'],
                   'max_output_tokens': plan['max_output_tokens'],
                   'seed_schedule': plan['seed_schedule'], 'prompts': capture_prompts}
        try:
            parsed = distribution.parse_capture(capture,
                hashlib.sha256(canonical(capture) + b'\n').hexdigest())
            distribution.verify_run_bindings(parsed, root)
            record_status = 'ready'
        except Exception as error:
            capture = None
            record_status = 'incomplete'
            finalization_error = type(error).__name__
    else:
        record_status = 'incomplete'
    record = {'schema': 'rosevine-stochastic-capture-record-v1',
              'status': record_status, 'plan_sha256': plan['plan_sha256'],
              'planned_sample_count': len(expected_keys),
              'successful_sample_count': sum(item['status'] == 'successful' for item in planned_records),
              'failed_or_missing_sample_count': len(planned_records) - sum(
                  item['status'] == 'successful' for item in planned_records),
              'unknown_mapping_count': len(unknown_rows),
              'run_records': run_records,
              'finalization_error': finalization_error,
              'capture_sha256': (hashlib.sha256(canonical(capture) + b'\n').hexdigest()
                                 if capture is not None else None),
              'acceptance': 'not decided; finalization only validates complete supervised token receipts'}
    return record, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-plan', action='store_true')
    parser.add_argument('--input', type=Path, required=True,
                        help='private input plan or frozen plan, depending on mode')
    parser.add_argument('--bundle-map', type=Path,
                        help='private mapping of each prompt/seed to its supervisor directory')
    parser.add_argument('--record', type=Path, required=True,
                        help='new private plan or capture-inventory record')
    parser.add_argument('--capture-output', type=Path,
                        help='new private v3 capture file; emitted only for a complete verified plan')
    args = parser.parse_args()
    source = secure_json(args.input)
    if args.prepare_plan:
        write_private_json(args.record, prepare_plan(source))
        print('Private stochastic seed plan frozen; no model launched.')
        return
    if args.bundle_map is None or args.capture_output is None:
        parser.error('finalization requires --bundle-map and --capture-output')
    plan = validate_plan(source)
    if args.record.parent.absolute() != args.input.parent.absolute() or args.capture_output.parent.absolute() != args.input.parent.absolute():
        raise ValueError('plan, capture and finalization record must share one private campaign directory')
    if args.bundle_map.parent.absolute() != args.input.parent.absolute():
        raise ValueError('bundle map must be inside the private campaign directory')
    if any(path.exists() or path.is_symlink()
           for path in (args.record, args.capture_output)):
        raise ValueError('finalization destinations must be new private files')
    bundle_map = secure_json(args.bundle_map)
    record, capture = finalize_capture(plan, bundle_map, args.input.parent)
    if capture is not None:
        write_private_json(args.capture_output, capture)
    write_private_json(args.record, record)
    print('Stochastic capture finalized: ' + record['status'] + '; no model launched.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Stochastic capture finalization failed; no acceptance inferred.', file=sys.stderr)
        sys.exit(1)
