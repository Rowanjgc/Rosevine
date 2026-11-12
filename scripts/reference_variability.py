#!/usr/bin/env python3
"""Measure observed pairwise variation among repeated private reference captures.

This summarizes repeatability only. It never turns observed variation into an
acceptance tolerance, and all logit rows remain streamed from private shards.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import stat
import sys

from benchmark_core import write_private_json
from logit_compare import compare, load_index
from verify_capture_bundle import secure_json, verify_bundle


def private_sha256(path, limit=16 * 1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > limit):
            raise ValueError('bounded private regular file required')
        hasher = hashlib.sha256()
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            total += len(block)
            if total > limit:
                raise ValueError('private file byte limit exceeded')
            hasher.update(block)
        return hasher.hexdigest()


def verify_run_bindings(index_paths, run_bundle_paths, indexes):
    if not isinstance(run_bundle_paths, list) or len(run_bundle_paths) != len(index_paths):
        raise ValueError('one retained supervisor run bundle is required per reference')
    if len({Path(path).absolute() for path in run_bundle_paths}) != len(run_bundle_paths):
        raise ValueError('each reference requires a distinct supervisor run bundle')
    processes, run_hashes, mapping_modes, mapping_evidence = [], [], [], []
    for index_path, run_path, index in zip(index_paths, run_bundle_paths, indexes):
        index_path, run_path = Path(index_path), Path(run_path)
        summary = verify_bundle(run_path)
        if summary['run_status'] != 'successful':
            raise ValueError('reference supervisor run was not successful')
        manifest = secure_json(run_path / 'manifest.json')
        mapping_modes.append(manifest.get('audit_mapped_artifacts', False))
        mapping_evidence.append(summary['mapped_artifact_identity'])
        run_identity = manifest.get('run_identity')
        expected_identity = {'identity_sha256': index['identity_sha256']}
        if index['_position_inventory_verified']:
            expected_identity.update({
                'inputs_sha256': index['inputs_sha256'],
                'position_plan_sha256': index['position_plan_sha256'],
                'expected_positions_sha256': index['expected_positions_sha256'],
                'expected_position_count': index['expected_position_count']})
        if index.get('schema') == 'rosevine-logits-v4':
            expected_identity['native_policy_sha256'] = index['native_policy_sha256']
        if (not isinstance(run_identity, dict)
                or any(run_identity.get(key) != value for key, value in expected_identity.items())):
            raise ValueError('reference index identity does not match its supervisor manifest')
        binding = secure_json(index_path.parent / 'capture-run-binding.json', 16384)
        if (not isinstance(binding, dict) or set(binding) !=
                {'schema', 'index_sha256', 'supervisor_result_sha256',
                 'process_instance_sha256', 'executable_sha256'}
                or binding['schema'] != 'rosevine-logit-capture-run-v1'):
            raise ValueError('reference capture run binding is missing or invalid')
        result_path = run_path / 'result.json'
        result_hash = private_sha256(result_path)
        index_hash = private_sha256(index_path)
        if binding['index_sha256'] != index_hash or binding['supervisor_result_sha256'] != result_hash:
            raise ValueError('capture binding does not match its index and run result')
        result = secure_json(result_path)
        child = result.get('child_process_identity')
        cleanup = result.get('workload_cleanup')
        if (result.get('status') != 'successful' or result.get('launched') is not True
                or result.get('returncode') != 0 or not isinstance(child, dict)
                or child.get('reason') is not None or type(child.get('pid')) is not int
                or type(child.get('start_ticks')) is not int or not isinstance(child.get('sha256'), str)
                or not isinstance(cleanup, dict) or cleanup.get('pending') is not False
                or cleanup.get('pid') != child['pid'] or cleanup.get('returncode') != 0):
            raise ValueError('reference run lacks successful reaped child identity')
        instance_hash = hashlib.sha256(f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest()
        if (binding['process_instance_sha256'] != instance_hash
                or binding['executable_sha256'] != child['sha256']):
            raise ValueError('capture binding disagrees with supervisor child identity')
        processes.append((instance_hash, child['sha256']))
        run_hashes.append(result_hash)
    if len(set(mapping_modes)) != 1:
        raise ValueError('reference runs must use the same mapped-artifact audit setting')
    if len({item[0] for item in processes}) != len(processes):
        raise ValueError('references reuse a supervised child process identity')
    if len({item[1] for item in processes}) != 1:
        raise ValueError('reference executable identity differs across captures')
    return {'distinct_successful_child_processes_verified': True,
            'common_executable_sha256': processes[0][1],
            'process_instance_sha256': [item[0] for item in processes],
            'supervisor_result_sha256': run_hashes,
            'mapped_artifact_audit_enabled': mapping_modes[0],
            'mapped_artifact_identity': mapping_evidence}


def analyze_reference_repeats(index_paths, run_bundle_paths=None):
    if not isinstance(index_paths, list) or len(index_paths) < 2:
        raise ValueError('at least two independent reference capture indexes required')
    paths = [Path(path).absolute() for path in index_paths]
    if len(set(paths)) != len(paths):
        raise ValueError('each reference capture must use a distinct index path')
    indexes = [load_index(path) for path in index_paths]
    process_evidence = (verify_run_bindings(index_paths, run_bundle_paths, indexes)
                        if run_bundle_paths is not None else None)
    anchor = indexes[0]
    for index in indexes[1:]:
        if index['identity_sha256'] != anchor['identity_sha256']:
            raise ValueError('reference repeat run identity mismatch')
        if index['vocabulary'] != anchor['vocabulary']:
            raise ValueError('reference repeat vocabulary mismatch')
        if index['inputs_sha256'] != anchor['inputs_sha256']:
            raise ValueError('reference repeat input/context identity mismatch')
        if index['logit_stage'] != anchor['logit_stage']:
            raise ValueError('reference repeat logit stage mismatch')
        if index.get('native_policy') != anchor.get('native_policy'):
            raise ValueError('reference repeats must use the same native capture policy')
        labels = [tuple(p) for shard in anchor['shards'] for p in shard['positions']]
        if [tuple(p) for shard in index['shards'] for p in shard['positions']] != labels:
            raise ValueError('reference repeat position identity/order mismatch')

    pairs = []
    for left in range(len(indexes)):
        for right in range(left + 1, len(indexes)):
            pair_bundles = ([run_bundle_paths[left], run_bundle_paths[right]]
                            if run_bundle_paths is not None else None)
            result = compare(index_paths[left], index_paths[right], run_bundles=pair_bundles)
            reverse_bundles = list(reversed(pair_bundles)) if pair_bundles is not None else None
            reverse = compare(index_paths[right], index_paths[left], run_bundles=reverse_bundles)
            pairs.append({'left_capture': left, 'right_capture': right,
                          'left_identity_sha256': result['reference_identity_sha256'],
                          'right_identity_sha256': result['candidate_identity_sha256'],
                          'positions': result['positions'], 'vocabulary': result['vocabulary'],
                          'left_to_right_mean_kl_nats': result['mean_kl_nats'],
                          'right_to_left_mean_kl_nats': reverse['mean_kl_nats'],
                          'symmetric_mean_kl_nats': math.fsum((result['mean_kl_nats'],
                                                              reverse['mean_kl_nats'])) / 2,
                          'max_directional_mean_kl_nats': max(result['mean_kl_nats'],
                                                              reverse['mean_kl_nats']),
                          'top1_agreement': result['top1_agreement'],
                          'max_directional_position_normalized_rmse': max(
                              result['max_position_normalized_rmse'],
                              reverse['max_position_normalized_rmse']),
                          'minimum_position_cosine': min(
                              value for value in (result['minimum_position_cosine'],
                                                  reverse['minimum_position_cosine'])
                              if value is not None) if result['minimum_position_cosine'] is not None or reverse['minimum_position_cosine'] is not None else None,
                          'max_absolute_error': result['max_absolute_error'],
                          'top1_mismatches': len(result['top1_mismatches']),
                          'minimum_4096_positions_met': result['minimum_4096_positions_met'],
                          'position_inventory_verified': result['position_inventory_verified'],
                          'minimum_4096_planned_positions_met': result['minimum_4096_planned_positions_met']})

    def finite_values(key):
        values = [pair[key] for pair in pairs if pair[key] is not None]
        if any(not math.isfinite(value) for value in values):
            raise ValueError('nonfinite repeat summary')
        return values

    symmetric_kl = finite_values('symmetric_mean_kl_nats')
    directional_kl = finite_values('max_directional_mean_kl_nats')
    cosine = finite_values('minimum_position_cosine')
    rmse = finite_values('max_directional_position_normalized_rmse')
    absolute = finite_values('max_absolute_error')
    agreement = finite_values('top1_agreement')
    return {
        'schema': 'rosevine-reference-variability-v1',
        'status': 'complete',
        'reference_identity_sha256': anchor['identity_sha256'],
        'inputs_sha256': anchor['inputs_sha256'],
        'logit_stage': anchor['logit_stage'],
        'native_policy': anchor.get('native_policy'),
        'native_policy_sha256': anchor.get('native_policy_sha256'),
        'reference_captures': len(indexes),
        'distinct_index_paths': True,
        'distinct_supervised_processes_verified': bool(process_evidence),
        'process_evidence': process_evidence,
        'independence_note': ('distinct successfully supervised child processes verified; separate sessions, thermal states, or other session-level independence are not established'
                              if process_evidence else 'separate index paths alone do not prove separate child executions or sessions; provide retained supervisor run bundles'),
        'pairwise_comparisons': len(pairs),
        'positions_per_pair': pairs[0]['positions'],
        'minimum_4096_positions_met_for_all_pairs': all(p['minimum_4096_positions_met'] for p in pairs),
        'position_inventory_verified_for_all_pairs': all(p['position_inventory_verified'] for p in pairs),
        'minimum_4096_planned_positions_met_for_all_pairs': all(
            p['minimum_4096_planned_positions_met'] for p in pairs),
        'pairwise_observed': {
            'symmetric_mean_kl_nats_max': max(symmetric_kl),
            'max_directional_mean_kl_nats': max(directional_kl),
            'top1_agreement_min': min(agreement),
            'max_position_normalized_rmse_max': max(rmse),
            'minimum_position_cosine_min': min(cosine) if cosine else None,
            'max_absolute_error_max': max(absolute),
        },
        'pairs': pairs,
        'interpretation': 'observed reference repeat variation only; not an approved tolerance or correctness acceptance',
        'acceptance': 'not decided; requires reviewed tolerance calibration and adequate position coverage',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, nargs='+', required=True,
                        help='two or more private reference index.json files')
    parser.add_argument('--run-bundles', type=Path, nargs='+', required=True,
                        help='one retained, verified supervisor run directory per reference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_private_json(args.output, analyze_reference_repeats(args.references, args.run_bundles))
    print('Private reference variability summary written; no tolerance acceptance asserted.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Reference variability analysis failed; no calibrated tolerance recorded.', file=sys.stderr)
        sys.exit(1)
