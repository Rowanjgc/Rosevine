#!/usr/bin/env python3
"""Offline input encoding and shard finalization for the native logit helper.

Neither operation launches a process or loads a model. Native execution requires
the applicable authorized resource profile and external supervision. Selected
context probes use protocol RVLOG003, decode every context input, and retain only
predeclared rows (at most 96 full-FP32 outputs). Failed/partial outputs are
retained; only complete, finite, correctly sized rows receive an index.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

from benchmark_core import canonical, digest, open_private, write_private_json
from logit_compare import rows, unique_object

STARTED = b'rosevine-native-logits-v1\n'


def private_read(path, limit):
    path = Path(path)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700 or parent.st_uid != os.getuid():
        raise ValueError('private owned parent required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid() or info.st_size > limit:
            raise ValueError('bounded owned private file required')
        result = stream.read(limit + 1)
        if len(result) > limit:
            raise ValueError('file byte limit exceeded')
        return result


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def encode_inputs(manifest, *, extra_buffers):
    if type(extra_buffers) is not bool:
        raise ValueError('explicit extra-buffer policy required')
    if not isinstance(manifest, dict) or set(manifest) != {'schema', 'model_sha256', 'tokenizer_sha256', 'vocabulary', 'capacity_tokens', 'reset_kv_before_sequence', 'position_policy', 'sequences'}:
        raise ValueError('invalid native input manifest')
    if manifest['schema'] != 'rosevine-logit-inputs-v1' or not all(sha(manifest[k]) for k in ('model_sha256', 'tokenizer_sha256')):
        raise ValueError('explicit model/tokenizer identity required')
    vocabulary, capacity = manifest['vocabulary'], manifest['capacity_tokens']
    if type(vocabulary) is not int or not 1 <= vocabulary <= 262144 or type(capacity) is not int or not 1 <= capacity <= 4096 or capacity % 256:
        raise ValueError('native dimensions outside envelope')
    if manifest['reset_kv_before_sequence'] is not True or manifest['position_policy'] != 'zero-based causal; row j predicts input ID j+1':
        raise ValueError('unsupported KV/position policy')
    sequences = manifest['sequences']
    if not isinstance(sequences, list) or not 1 <= len(sequences) <= 8:
        raise ValueError('requires 1..8 sequences')
    selective = [isinstance(sequence, dict) and 'capture_input_positions' in sequence
                 for sequence in sequences]
    if any(selective) and not all(selective):
        raise ValueError('selected-row mode must be declared for every sequence')
    selected_rows = all(selective)
    flags = (1 if selected_rows else 0) | (2 if extra_buffers else 0)
    result = bytearray(b'RVLOG003' + struct.pack('<IIII', vocabulary, capacity,
                                                 len(sequences), flags))
    seen = set()
    total_selected = 0
    for sequence in sequences:
        expected_keys = {'id', 'token_ids', 'capture_input_positions'} if selected_rows else {'id', 'token_ids'}
        if not isinstance(sequence, dict) or set(sequence) != expected_keys:
            raise ValueError('invalid input sequence')
        label, tokens = sequence['id'], sequence['token_ids']
        if not isinstance(label, str) or not label or label in seen:
            raise ValueError('unique sequence labels required')
        seen.add(label)
        maximum = min(capacity, 4096 if selected_rows else 513)
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= maximum or any(type(t) is not int or not 0 <= t < vocabulary for t in tokens):
            raise ValueError('invalid input IDs or sequence length')
        result.extend(struct.pack('<I', len(tokens)))
        result.extend(struct.pack('<' + 'I' * len(tokens), *tokens))
        if selected_rows:
            positions = sequence['capture_input_positions']
            if (not isinstance(positions, list) or not positions
                    or any(type(p) is not int or not 0 <= p < len(tokens) - 1 for p in positions)
                    or positions != sorted(set(positions))):
                raise ValueError('selected output positions must be sorted, unique input rows')
            total_selected += len(positions)
            if total_selected > 96:
                raise ValueError('selected-row capture exceeds the 96 MiB full-FP32 envelope')
            result.extend(struct.pack('<I', len(positions)))
            result.extend(struct.pack('<' + 'I' * len(positions), *positions))
    return bytes(result)


def capture_run_identity_fields(manifest, position_plan_sha256, native_policy):
    labels = [(sequence['id'], position)
              for sequence in manifest['sequences']
              for position in sequence.get('capture_input_positions',
                                            range(len(sequence['token_ids']) - 1))]
    return {'inputs_sha256': digest(manifest),
            'position_plan_sha256': position_plan_sha256,
            'expected_positions_sha256': digest([list(label) for label in labels]),
            'expected_position_count': len(labels),
            'native_policy_sha256': digest(native_policy)}


def from_plan(plan, model_sha256, tokenizer_sha256):
    if (plan.get('schema') not in ('rosevine-reference-positions-v1', 'rosevine-context-positions-v1')
            or digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan.get('plan_sha256')):
        raise ValueError('position plan checksum/schema mismatch')
    context_plan = plan['schema'] == 'rosevine-context-positions-v1'
    sequences = plan['streams'] if context_plan else plan['sequences']
    for sequence in sequences:
        ids = sequence['token_ids']
        if sequence['tokens_sha256'] != digest(ids) or sequence['reset_kv_before_sequence'] is not True:
            raise ValueError('plan token/position/reset mismatch')
        if context_plan:
            positions = sequence.get('capture_input_positions')
            if (not isinstance(positions, list) or not positions
                    or any(type(p) is not int or not 0 <= p < len(ids) - 1 for p in positions)
                    or positions != sorted(set(positions))
                    or positions != plan.get('captured_input_positions')
                    or sequence.get('scored_target_indices') != [p + 1 for p in positions]
                    or sequence.get('target_token_ids') != [ids[p + 1] for p in positions]):
                raise ValueError('context boundary row/target alignment mismatch')
        elif sequence.get('scored_input_positions') != [0, len(ids) - 2] or sequence.get('scored_target_indices') != [1, len(ids) - 1]:
            raise ValueError('teacher-forced row/target alignment mismatch')
    count = (sum(len(s['capture_input_positions']) for s in sequences) if context_plan
             else sum(len(s['token_ids']) - 1 for s in sequences))
    if ((context_plan and (count != plan['capture_rows'] or plan.get('source_split') != 'calibration'))
            or (not context_plan and (count != plan['scored_positions'] or plan['source_split'] != 'calibration'))):
        raise ValueError('plan coverage/split mismatch')
    manifest = {'schema': 'rosevine-logit-inputs-v1', 'model_sha256': model_sha256,
                'tokenizer_sha256': tokenizer_sha256, 'vocabulary': plan['vocabulary'],
                'capacity_tokens': plan['capacity_tokens'], 'reset_kv_before_sequence': True,
                'position_policy': 'zero-based causal; row j predicts input ID j+1',
                'sequences': [{'id': s['id'], 'token_ids': s['token_ids'],
                               **({'capture_input_positions': s['capture_input_positions']} if context_plan else {})}
                              for s in sequences]}
    # This validation call does not prepare an execution artifact; pass the
    # policy value explicitly so callers cannot inherit an implicit default.
    encode_inputs(manifest, extra_buffers=False)
    return manifest


def prepare(plan, model_sha256, tokenizer_sha256, directory, *, extra_buffers):
    manifest = from_plan(plan, model_sha256, tokenizer_sha256)
    if type(extra_buffers) is not bool:
        raise ValueError('explicit extra-buffer policy required')
    raw = encode_inputs(manifest, extra_buffers=extra_buffers)
    native_policy = {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                     'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                     'mtp': False, 'extra_buffers': extra_buffers,
                     'logit_stage': 'post-softcap', 'max_shard_rows': 32}
    run_identity_fields = capture_run_identity_fields(manifest, plan['plan_sha256'], native_policy)
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_private_json(directory / 'inputs.json', manifest)
    with open_private(directory / 'inputs.bin') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    write_private_json(directory / 'prepared.json', {
        'status': 'prepared; no model loaded', 'inputs_sha256': digest(manifest),
        'binary_input_sha256': hashlib.sha256(raw).hexdigest(),
        'position_plan_sha256': plan['plan_sha256'],
        'native_policy': native_policy,
        'native_policy_sha256': digest(native_policy),
        'supervisor_run_identity_fields': run_identity_fields,
        'authorization': 'preparation grants no run authority; execution records the applicable authorized profile and supervised invocation'})
    return manifest


def finalize(directory, prepared_directory, identity_sha256, supervision):
    """Validate a terminated successful producer and index its exact native rows.

    supervision is the authoritative supervisor result, not a native marker.
    Caller must independently audit its command, hashes and frozen run identity.
    This function proves row integrity/alignment, not that a build used the right
    model. No automatic retry, deletion or model execution is performed.
    """
    if (supervision.get('status') != 'successful' or supervision.get('returncode') != 0
            or supervision.get('launched') is not True
            or not isinstance(supervision.get('workload_cleanup'), dict)
            or supervision['workload_cleanup'].get('pending') is not False
            or supervision['workload_cleanup'].get('returncode') != 0):
        raise ValueError('successful supervised and reaped producer required')
    child = supervision.get('child_process_identity')
    if (not isinstance(child, dict) or child.get('reason') is not None
            or type(child.get('pid')) is not int or child['pid'] <= 0
            or type(child.get('start_ticks')) is not int or child['start_ticks'] < 0
            or not sha(child.get('sha256'))
            or supervision['workload_cleanup'].get('pid') != child['pid']):
        raise ValueError('verified child process identity required')
    if not sha(identity_sha256):
        raise ValueError('run identity digest required')
    prepared_directory, directory = Path(prepared_directory), Path(directory)
    prepared = json.loads(private_read(prepared_directory / 'prepared.json', 16384), object_pairs_hook=unique_object)
    manifest = json.loads(private_read(prepared_directory / 'inputs.json', 256 * 1024), object_pairs_hook=unique_object)
    native_policy = prepared.get('native_policy')
    expected_policy_fields = {'backend', 'threads', 'batch', 'microbatch', 'flash_attention',
                              'kv', 'mtp', 'extra_buffers', 'logit_stage', 'max_shard_rows'}
    expected_policy = {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                       'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                       'mtp': False, 'extra_buffers': native_policy.get('extra_buffers')
                       if isinstance(native_policy, dict) else None,
                       'logit_stage': 'post-softcap', 'max_shard_rows': 32}
    if (not isinstance(native_policy, dict) or set(native_policy) != expected_policy_fields
            or type(native_policy['extra_buffers']) is not bool
            or type(native_policy['threads']) is not int
            or type(native_policy['batch']) is not int
            or type(native_policy['microbatch']) is not int
            or type(native_policy['flash_attention']) is not bool
            or type(native_policy['mtp']) is not bool
            or type(native_policy['max_shard_rows']) is not int
            or native_policy != expected_policy
            or prepared.get('native_policy_sha256') != digest(native_policy)):
        raise ValueError('prepared native policy is unsupported or its digest is invalid')
    expected_binary = encode_inputs(manifest, extra_buffers=native_policy['extra_buffers'])
    if private_read(prepared_directory / 'inputs.bin', 256 * 1024) != expected_binary:
        raise ValueError('native input bytes differ from frozen manifest and policy')
    if prepared.get('inputs_sha256') != digest(manifest) or prepared.get('binary_input_sha256') != hashlib.sha256(expected_binary).hexdigest():
        raise ValueError('prepared input digest mismatch')
    run_identity_fields = capture_run_identity_fields(
        manifest, prepared['position_plan_sha256'], native_policy)
    if prepared.get('supervisor_run_identity_fields') != run_identity_fields:
        raise ValueError('prepared supervisor run identity fields mismatch')
    labels = [(s['id'], j) for s in manifest['sequences']
              for j in (s.get('capture_input_positions', range(len(s['token_ids']) - 1)))]
    if private_read(directory / 'started', 64) != STARTED or private_read(directory / 'complete', 64) != f'{len(labels)}\n'.encode():
        raise ValueError('incomplete native row receipt')
    shards = []
    for offset in range(0, len(labels), 32):
        name = f'logits-{offset // 32:06d}.f32'
        raw = private_read(directory / name, 32 * manifest['vocabulary'] * 4)
        if len(raw) != len(labels[offset:offset + 32]) * manifest['vocabulary'] * 4:
            raise ValueError('native shard byte count mismatch')
        shards.append({'file': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                       'positions': [list(label) for label in labels[offset:offset + 32]]})
    if {p.name for p in directory.iterdir()} != {'started', 'complete', *(s['file'] for s in shards)}:
        raise ValueError('unexpected or preexisting native capture files')
    record = {'schema': 'rosevine-logits-v4', 'identity_sha256': identity_sha256,
              'inputs_sha256': digest(manifest), 'logit_stage': 'post-softcap',
              'native_policy': native_policy,
              'native_policy_sha256': digest(native_policy),
              'position_plan_sha256': prepared['position_plan_sha256'],
              'expected_positions_sha256': digest([list(label) for label in labels]),
              'expected_position_count': len(labels),
              'vocabulary': manifest['vocabulary'], 'shards': shards}
    path = directory / 'index.json'
    # Re-read and exhaust every row: checksums, shape and finiteness must hold
    # before the create-only index declares this a usable capture.
    for _, values in rows(path, record):
        if any(not math.isfinite(value) for value in values):
            raise ValueError('nonfinite native logit row')
    write_private_json(path, record)
    binding = {'schema': 'rosevine-logit-capture-run-v1',
               'index_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'supervisor_result_sha256': hashlib.sha256(canonical(supervision) + b'\n').hexdigest(),
               'process_instance_sha256': hashlib.sha256(
                   f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest(),
               'executable_sha256': child['sha256']}
    write_private_json(directory / 'capture-run-binding.json', binding)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--plan', type=Path, required=True)
    prep.add_argument('--model-sha256', required=True)
    prep.add_argument('--tokenizer-sha256', required=True)
    prep.add_argument('--directory', type=Path, required=True)
    policy = prep.add_mutually_exclusive_group(required=True)
    policy.add_argument('--extra-buffers', dest='extra_buffers', action='store_true')
    policy.add_argument('--no-extra-buffers', dest='extra_buffers', action='store_false')
    finish = sub.add_parser('finalize')
    finish.add_argument('--directory', type=Path, required=True)
    finish.add_argument('--prepared-directory', type=Path, required=True)
    finish.add_argument('--identity-sha256', required=True)
    finish.add_argument('--supervisor-result', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(json.loads(private_read(args.plan, 256 * 1024), object_pairs_hook=unique_object),
                args.model_sha256, args.tokenizer_sha256, args.directory,
                extra_buffers=args.extra_buffers)
    else:
        finalize(args.directory, args.prepared_directory, args.identity_sha256,
                 json.loads(private_read(args.supervisor_result, 65536), object_pairs_hook=unique_object))
    print('Offline logit capture preparation/validation complete; no model launched.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Logit capture preparation/validation failed; retain partial evidence.', file=sys.stderr)
        sys.exit(1)
