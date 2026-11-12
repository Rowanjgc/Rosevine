#!/usr/bin/env python3
"""Stream private FP32 logit shards; report errors without approving tolerances.

Index schemas v2-v4: v4 additionally binds the complete native policy and its
digest. identity_sha256, inputs_sha256, a position-plan hash, and a digest/count
for the exact ordered position labels. inputs_sha256 binds the shared model/tokenizer,
ordered full input IDs, sequence IDs, position/reset policy, and vocabulary order;
it excludes backend/build identity, which belongs in identity_sha256.
Each shard has file, sha256 and positions (unique [sequence_id, input_position]
labels). Raw values are contiguous little-endian FP32 rows in full vocabulary
order. These files are private reference/candidate evidence, never a public export.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

from benchmark_core import digest, write_private_json
from correctness_oracle import logit_errors
from verify_capture_bundle import secure_bytes, secure_json, verify_bundle


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate index key')
        result[key] = value
    return result


def load_index(path):
    path = Path(path)
    if path.parent.is_symlink() or stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise ValueError('private index directory required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > 16 * 1024 * 1024):
            raise ValueError('bounded owned private index required')
        raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError('index byte limit exceeded')
    record = json.loads(raw, object_pairs_hook=unique_object)
    legacy_keys = {'schema', 'identity_sha256', 'inputs_sha256', 'logit_stage',
                   'vocabulary', 'shards'}
    current_keys = {'schema', 'identity_sha256', 'inputs_sha256', 'position_plan_sha256',
                    'expected_positions_sha256', 'expected_position_count', 'logit_stage',
                    'vocabulary', 'shards'}
    policy_keys = current_keys | {'native_policy', 'native_policy_sha256'}
    legacy = isinstance(record, dict) and record.get('schema') == 'rosevine-logits-v2' and set(record) == legacy_keys
    current = isinstance(record, dict) and record.get('schema') == 'rosevine-logits-v3' and set(record) == current_keys
    policy_current = (isinstance(record, dict) and record.get('schema') == 'rosevine-logits-v4'
                      and set(record) == policy_keys)
    if not (legacy or current or policy_current):
        raise ValueError('invalid logit index schema')
    identity_fields = ('identity_sha256', 'inputs_sha256')
    if current or policy_current:
        identity_fields += ('position_plan_sha256', 'expected_positions_sha256')
    if policy_current:
        identity_fields += ('native_policy_sha256',)
    for field in identity_fields:
        if not isinstance(record[field], str) or not re.fullmatch('[0-9a-f]{64}', record[field]):
            raise ValueError('invalid identity hash')
    if record['logit_stage'] not in ('pre-softcap', 'post-softcap'):
        raise ValueError('explicit logit stage required')
    if policy_current:
        policy = record['native_policy']
        required = {'backend', 'threads', 'batch', 'microbatch', 'flash_attention',
                    'kv', 'mtp', 'extra_buffers', 'logit_stage', 'max_shard_rows'}
        extra_buffers = policy.get('extra_buffers') if isinstance(policy, dict) else None
        expected_policy = {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                           'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                           'mtp': False, 'extra_buffers': extra_buffers,
                           'logit_stage': 'post-softcap', 'max_shard_rows': 32}
        if (not isinstance(policy, dict) or set(policy) != required
                or type(policy['extra_buffers']) is not bool
                or policy != expected_policy or record['logit_stage'] != policy['logit_stage']
                or digest(policy) != record['native_policy_sha256']):
            raise ValueError('invalid or unbound native capture policy')
    vocabulary = record['vocabulary']
    if type(vocabulary) is not int or not 0 < vocabulary <= 262144:
        raise ValueError('unsupported vocabulary width')
    if not isinstance(record['shards'], list) or not record['shards']:
        raise ValueError('nonempty shard list required')
    if (current or policy_current) and (type(record['expected_position_count']) is not int
                    or record['expected_position_count'] <= 0):
        raise ValueError('positive expected position count required')
    labels, files = set(), set()
    for shard in record['shards']:
        if set(shard) != {'file', 'sha256', 'positions'}:
            raise ValueError('invalid logit shard')
        if not isinstance(shard['file'], str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*[.]f32', shard['file']) or shard['file'] in files:
            raise ValueError('invalid or duplicate shard filename')
        files.add(shard['file'])
        if not isinstance(shard['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', shard['sha256']):
            raise ValueError('invalid shard hash')
        if not isinstance(shard['positions'], list) or not 0 < len(shard['positions']) <= 32:
            raise ValueError('shard requires 1..32 scored positions')
        for label in shard['positions']:
            if not isinstance(label, list) or len(label) != 2 or not isinstance(label[0], str) or not label[0] or type(label[1]) is not int or label[1] < 0:
                raise ValueError('invalid scored position label')
            label = tuple(label)
            if label in labels:
                raise ValueError('duplicate scored position')
            labels.add(label)
    ordered_labels = [label for shard in record['shards'] for label in shard['positions']]
    if (current or policy_current) and (len(ordered_labels) != record['expected_position_count']
                    or digest(ordered_labels) != record['expected_positions_sha256']):
        raise ValueError('captured positions differ from frozen ordered position inventory')
    record['_position_inventory_verified'] = current or policy_current
    return record


def rows(path, record):
    parent = Path(path).parent
    width = record['vocabulary']
    for shard in record['shards']:
        fd = os.open(parent / shard['file'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != len(shard['positions']) * width * 4:
                raise ValueError('shard shape/size mismatch')
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('private shard file mode required')
            hasher = hashlib.sha256()
            for label in shard['positions']:
                raw = stream.read(width * 4)
                if len(raw) != width * 4:
                    raise ValueError('truncated logit row')
                hasher.update(raw)
                values = array.array('f')
                if values.itemsize != 4:
                    raise ValueError('unsupported native float size')
                values.frombytes(raw)
                if sys.byteorder != 'little':
                    values.byteswap()
                yield tuple(label), values
            if stream.read(1) or hasher.hexdigest() != shard['sha256']:
                raise ValueError('shard hash/length mismatch')


def verify_run_bindings(index_paths, run_bundle_paths, indexes):
    if not isinstance(run_bundle_paths, (list, tuple)) or len(run_bundle_paths) != 2:
        raise ValueError('reference and candidate supervisor bundles are both required')
    if len({Path(path).absolute() for path in run_bundle_paths}) != 2:
        raise ValueError('reference and candidate require distinct supervisor bundles')
    processes, evidence = [], []
    for index_path, bundle_path, index in zip(index_paths, run_bundle_paths, indexes):
        index_path, bundle_path = Path(index_path), Path(bundle_path)
        bundle_summary = verify_bundle(bundle_path)
        if bundle_summary['run_status'] != 'successful':
            raise ValueError('logit supervisor bundle did not complete successfully')
        manifest = secure_json(bundle_path / 'manifest.json')
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
        if (not isinstance(run_identity, dict) or any(
                run_identity.get(key) != value for key, value in expected_identity.items())):
            raise ValueError('logit index identity differs from supervisor manifest')
        binding = secure_json(index_path.parent / 'capture-run-binding.json', 16384)
        if (not isinstance(binding, dict) or set(binding) != {
                'schema', 'index_sha256', 'supervisor_result_sha256',
                'process_instance_sha256', 'executable_sha256'}
                or binding['schema'] != 'rosevine-logit-capture-run-v1'):
            raise ValueError('logit capture run binding is missing or invalid')
        index_raw = secure_bytes(index_path, 16 * 1024 * 1024)
        result_raw = secure_bytes(bundle_path / 'result.json')
        if (hashlib.sha256(index_raw).hexdigest() != binding['index_sha256']
                or hashlib.sha256(result_raw).hexdigest() != binding['supervisor_result_sha256']):
            raise ValueError('logit capture binding does not match index and supervisor result')
        result = secure_json(bundle_path / 'result.json')
        child, cleanup = result.get('child_process_identity'), result.get('workload_cleanup')
        if (result.get('status') != 'successful' or result.get('launched') is not True
                or result.get('returncode') != 0 or not isinstance(child, dict)
                or child.get('reason') is not None or type(child.get('pid')) is not int
                or type(child.get('start_ticks')) is not int
                or child.get('sha256') != binding['executable_sha256']
                or not isinstance(cleanup, dict) or cleanup.get('pending') is not False
                or cleanup.get('pid') != child['pid'] or cleanup.get('returncode') != 0):
            raise ValueError('logit run bundle lacks successful reaped child identity')
        process = hashlib.sha256(f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest()
        if process != binding['process_instance_sha256']:
            raise ValueError('logit capture binding disagrees with supervised process identity')
        processes.append(process)
        evidence.append({'supervisor_result_sha256': binding['supervisor_result_sha256'],
                         'process_instance_sha256': process,
                         'executable_sha256': binding['executable_sha256'],
                         'mapped_artifact_identity': bundle_summary['mapped_artifact_identity']})
    if processes[0] == processes[1]:
        raise ValueError('reference and candidate must use distinct supervised child processes')
    return evidence


def compare(reference_path, candidate_path, *, run_bundles=None):
    reference, candidate = load_index(reference_path), load_index(candidate_path)
    if reference['vocabulary'] != candidate['vocabulary']:
        raise ValueError('vocabulary mismatch')
    if reference['inputs_sha256'] != candidate['inputs_sha256']:
        raise ValueError('input/context identity mismatch')
    if reference['logit_stage'] != candidate['logit_stage']:
        raise ValueError('logit stage mismatch')
    policy_comparison = None
    if reference.get('schema') == 'rosevine-logits-v4' or candidate.get('schema') == 'rosevine-logits-v4':
        if reference.get('schema') != 'rosevine-logits-v4' or candidate.get('schema') != 'rosevine-logits-v4':
            raise ValueError('cannot compare policy-bound v4 capture with an unbound legacy capture')
        ref_policy, cand_policy = reference['native_policy'], candidate['native_policy']
        if {k: v for k, v in ref_policy.items() if k != 'extra_buffers'} != {
                k: v for k, v in cand_policy.items() if k != 'extra_buffers'}:
            raise ValueError('native capture policies differ beyond extra_buffers')
        policy_comparison = {'reference': ref_policy, 'candidate': cand_policy,
                             'same': ref_policy == cand_policy,
                             'extra_buffers_varied': ref_policy['extra_buffers'] != cand_policy['extra_buffers']}
    if (reference['_position_inventory_verified'] and candidate['_position_inventory_verified']
            and reference['position_plan_sha256'] != candidate['position_plan_sha256']):
        raise ValueError('position plan mismatch')
    # Validate equal label sets before reading; iterator comparison also enforces order.
    ref_labels = [tuple(p) for shard in reference['shards'] for p in shard['positions']]
    cand_labels = [tuple(p) for shard in candidate['shards'] for p in shard['positions']]
    if ref_labels != cand_labels:
        raise ValueError('scored position identity/order mismatch')
    run_evidence = (verify_run_bindings((reference_path, candidate_path), run_bundles,
                                        (reference, candidate))
                    if run_bundles is not None else None)
    inventory_integrity = (reference['_position_inventory_verified']
                           and candidate['_position_inventory_verified'])
    inventory_bound = inventory_integrity and run_evidence is not None
    ref_rows, cand_rows = rows(reference_path, reference), rows(candidate_path, candidate)
    count, agreement, kl_sum = 0, 0, 0.0
    max_rmse, min_cosine, max_abs = 0.0, 1.0, 0.0
    zero_norm_positions, top1_mismatches = [], []
    # Never zip generators: both must reach EOF and validate their final shard hash.
    for label in ref_labels:
        ref_label, ref_values = next(ref_rows)
        cand_label, cand_values = next(cand_rows)
        if ref_label != label or cand_label != label:
            raise ValueError('position mismatch')
        result = logit_errors([ref_values], [cand_values])
        error = result['per_position_errors'][0]
        count += 1
        agreement += result['top1_agreement']
        kl_sum = math.fsum([kl_sum, result['mean_kl_nats']])
        max_rmse = max(max_rmse, error['normalized_rmse'])
        max_abs = max(max_abs, error['max_absolute_error'])
        if error['cosine'] is None:
            zero_norm_positions.append(label)
        else:
            min_cosine = min(min_cosine, error['cosine'])
        if result['top1_agreement'] != 1:
            top1_mismatches.append(label)
    for iterator in (ref_rows, cand_rows):
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError('unaccounted logit row')
    return {'status': 'complete', 'reference_identity_sha256': reference['identity_sha256'],
            'candidate_identity_sha256': candidate['identity_sha256'],
            'inputs_sha256': reference['inputs_sha256'], 'logit_stage': reference['logit_stage'],
            'positions': count, 'vocabulary': reference['vocabulary'],
            'minimum_4096_positions_met': count >= 4096,
            'position_inventory_integrity_verified': inventory_integrity,
            'position_inventory_verified': inventory_bound,
            'minimum_4096_planned_positions_met': count >= 4096 and inventory_bound,
            'mean_kl_nats': kl_sum / count, 'top1_agreement': agreement / count,
            'max_position_normalized_rmse': max_rmse,
            'minimum_position_cosine': min_cosine if count > len(zero_norm_positions) else None,
            'max_absolute_error': max_abs, 'zero_norm_positions': zero_norm_positions,
            'top1_mismatches': top1_mismatches,
            'acceptance': 'not decided; requires frozen reviewed tolerances and complete coverage',
            'native_policy_comparison': policy_comparison,
            'run_bundle_provenance_verified': run_evidence is not None,
            'run_bundle_provenance': run_evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-run-bundle', type=Path)
    parser.add_argument('--candidate-run-bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.reference_run_bundle is None) != (args.candidate_run_bundle is None):
        raise ValueError('supply both reference and candidate supervisor bundles')
    bundles = None
    if args.reference_run_bundle is not None:
        bundles = [args.reference_run_bundle, args.candidate_run_bundle]
    result = compare(args.reference, args.candidate, run_bundles=bundles)
    write_private_json(args.output, result)
    print('Private logit comparison complete; no correctness acceptance asserted.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, StopIteration, OverflowError):
        print('Logit comparison failed; no valid comparison recorded.', file=sys.stderr)
        sys.exit(1)
