#!/usr/bin/env python3
"""Prepare/finalize bounded private native per-token NLL captures.

This adapter encodes frozen corpus IDs and validates native scalar outputs. It
never launches the model. Full-vocabulary normalization occurs in the isolated
native helper; incomplete/failed supervision cannot become an NLL capture.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

from benchmark_core import canonical, digest, write_private_json
from nll_compare import validate_corpus

MAGIC = b'RVNLL001'
STARTED = b'rosevine-native-nll-v1\n'
MAX_DOCUMENTS = 1024
MAX_TARGETS = 200000
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_CORPUS_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = MAX_TARGETS * 8


def private_read(path, limit):
    path = Path(path)
    parent = path.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700
            or parent.st_uid != os.getuid()):
        raise ValueError('owned private parent directory required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > limit):
            raise ValueError('bounded owned private file required')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('private byte limit exceeded')
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def encode_inputs(manifest):
    keys = {'schema', 'corpus_sha256', 'model_sha256', 'tokenizer_sha256', 'split',
            'vocabulary', 'capacity_tokens', 'position_policy', 'documents'}
    if not isinstance(manifest, dict) or set(manifest) != keys or manifest['schema'] != 'rosevine-nll-inputs-v1':
        raise ValueError('invalid NLL input manifest')
    for field in ('corpus_sha256', 'model_sha256', 'tokenizer_sha256'):
        if not isinstance(manifest[field], str) or re.fullmatch('[0-9a-f]{64}', manifest[field]) is None:
            raise ValueError('invalid NLL input identity hash')
    vocabulary, capacity = manifest['vocabulary'], manifest['capacity_tokens']
    if (type(vocabulary) is not int or not 0 < vocabulary <= 262144
            or type(capacity) is not int or not 0 < capacity <= 4096 or capacity % 256):
        raise ValueError('NLL dimensions outside native envelope')
    if manifest['split'] not in ('calibration', 'holdout') or manifest['position_policy'] != 'zero-based causal; row j predicts token j+1':
        raise ValueError('explicit NLL split/position policy required')
    documents = manifest['documents']
    if not isinstance(documents, list) or not 1 <= len(documents) <= MAX_DOCUMENTS:
        raise ValueError('NLL document count outside capture envelope')
    raw = bytearray(struct.pack('<8sIII', MAGIC, vocabulary, capacity, len(documents)))
    ids, target_count = set(), 0
    for document in documents:
        if not isinstance(document, dict) or set(document) != {'id', 'token_ids', 'tokens_sha256', 'scored_tokens'}:
            raise ValueError('invalid NLL document input')
        identifier, tokens = document['id'], document['token_ids']
        if (not isinstance(identifier, str) or not identifier or identifier in ids
                or not isinstance(tokens, list) or not 2 <= len(tokens) <= capacity
                or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
                or digest(tokens) != document['tokens_sha256']
                or document['scored_tokens'] != len(tokens) - 1):
            raise ValueError('NLL document token/hash/count mismatch')
        ids.add(identifier)
        target_count += len(tokens) - 1
        raw.extend(struct.pack('<I', len(tokens)))
        raw.extend(struct.pack('<' + 'I' * len(tokens), *tokens))
        if target_count > MAX_TARGETS or len(raw) > MAX_INPUT_BYTES:
            raise ValueError('NLL input exceeds bounded capture envelope')
    return bytes(raw)


def prepare(corpus_path, split, model_sha256, tokenizer_sha256, directory,
            *, vocabulary=262144, capacity_tokens=4096):
    corpus = json.loads(private_read(corpus_path, MAX_CORPUS_BYTES), object_pairs_hook=_unique_object)
    documents = validate_corpus(corpus, split)
    manifest = {'schema': 'rosevine-nll-inputs-v1', 'corpus_sha256': corpus['corpus_sha256'],
                'model_sha256': model_sha256, 'tokenizer_sha256': tokenizer_sha256,
                'split': split, 'vocabulary': vocabulary, 'capacity_tokens': capacity_tokens,
                'position_policy': 'zero-based causal; row j predicts token j+1',
                'documents': [{'id': row['id'], 'token_ids': row['token_ids'],
                               'tokens_sha256': row['tokens_sha256'],
                               'scored_tokens': row['scored_tokens']}
                              for row in documents.values()]}
    raw = encode_inputs(manifest)
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(directory, 0o700)
    with (directory / 'inputs.bin').open('xb') as stream:
        os.chmod(directory / 'inputs.bin', 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    scored = sum(row['scored_tokens'] for row in manifest['documents'])
    plan = {'schema': 'rosevine-nll-capture-plan-v1', 'status': 'prepared; no model loaded',
            'inputs': manifest, 'inputs_sha256': digest(manifest),
            'binary_input_sha256': hashlib.sha256(raw).hexdigest(),
            'input_bytes': len(raw), 'document_count': len(manifest['documents']),
            'target_count': scored, 'full_vocabulary_reduction': 'stable long-double log-sum-exp; target is input j+1',
            'output_bytes': scored * 8, 'output_format': 'little-endian binary64 nats, document order then target order',
            'logit_stage': 'post-softcap',
            'limitations': 'direct corpus token IDs, reset KV per document; no generation, chat rendering, quality acceptance or model run authorization; CPU full-vocabulary reduction runtime is unmeasured and output-byte bounds do not bound compute time'}
    plan['plan_sha256'] = digest(plan)
    write_private_json(directory / 'plan.json', plan)
    return plan


def finalize(native_directory, prepared_directory, identity_sha256, supervision):
    if (not isinstance(supervision, dict) or supervision.get('status') != 'successful'
            or supervision.get('returncode') != 0 or supervision.get('launched') is not True
            or not isinstance(supervision.get('workload_cleanup'), dict)
            or supervision['workload_cleanup'].get('pending') is not False
            or supervision['workload_cleanup'].get('returncode') != 0):
        raise ValueError('successful supervised and reaped native NLL run required')
    child = supervision.get('child_process_identity')
    cleanup = supervision.get('workload_cleanup')
    if (not isinstance(child, dict) or child.get('reason') is not None
            or type(child.get('pid')) is not int or child['pid'] <= 0
            or type(child.get('start_ticks')) is not int or child['start_ticks'] < 0
            or not isinstance(child.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', child['sha256']) is None
            or cleanup.get('pid') != child['pid']):
        raise ValueError('successful reaped child process identity required')
    if not isinstance(identity_sha256, str) or re.fullmatch('[0-9a-f]{64}', identity_sha256) is None:
        raise ValueError('verified run identity digest required')
    prepared_directory, native_directory = Path(prepared_directory), Path(native_directory)
    native_info = native_directory.lstat()
    if (not stat.S_ISDIR(native_info.st_mode) or stat.S_IMODE(native_info.st_mode) != 0o700
            or native_info.st_uid != os.getuid()):
        raise ValueError('owned private native output directory required')
    if {entry.name for entry in native_directory.iterdir()} != {'started', 'complete', 'nll-000000.f64'}:
        raise ValueError('unexpected or missing native NLL output file')
    plan = json.loads(private_read(prepared_directory / 'plan.json', MAX_CORPUS_BYTES), object_pairs_hook=_unique_object)
    if (plan.get('schema') != 'rosevine-nll-capture-plan-v1'
            or digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan.get('plan_sha256')):
        raise ValueError('NLL capture plan checksum/schema mismatch')
    manifest = plan['inputs']
    raw_inputs = encode_inputs(manifest)
    if (hashlib.sha256(raw_inputs).hexdigest() != plan['binary_input_sha256']
            or private_read(prepared_directory / 'inputs.bin', MAX_INPUT_BYTES) != raw_inputs
            or digest(manifest) != plan['inputs_sha256']):
        raise ValueError('prepared NLL input identity mismatch')
    if private_read(native_directory / 'started', 64) != STARTED:
        raise ValueError('native NLL start marker missing')
    expected = plan['target_count']
    if private_read(native_directory / 'complete', 64) != ('%d\n' % expected).encode():
        raise ValueError('native NLL completion count mismatch')
    raw_values = private_read(native_directory / 'nll-000000.f64', MAX_OUTPUT_BYTES)
    if len(raw_values) != expected * 8:
        raise ValueError('native NLL output size mismatch')
    values = struct.unpack('<%dd' % expected, raw_values)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('native NLL output contains invalid values')
    documents, offset = [], 0
    for row in manifest['documents']:
        end = offset + row['scored_tokens']
        documents.append({'id': row['id'], 'tokens_sha256': row['tokens_sha256'], 'nll': list(values[offset:end])})
        offset = end
    if offset != len(values):
        raise ValueError('NLL target alignment did not reach exact EOF')
    capture = {'schema': 'rosevine-nll-v1', 'status': 'successful',
               'corpus_sha256': manifest['corpus_sha256'], 'identity_sha256': identity_sha256,
               'split': manifest['split'], 'logit_stage': plan['logit_stage'], 'documents': documents}
    write_private_json(native_directory / 'capture.json', capture)
    capture_bytes = canonical(capture) + b'\n'
    result_hash = hashlib.sha256(canonical(supervision) + b'\n').hexdigest()
    process_hash = hashlib.sha256(f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest()
    write_private_json(native_directory / 'capture-run-binding.json', {
        'schema': 'rosevine-nll-capture-run-v1',
        'capture_sha256': hashlib.sha256(capture_bytes).hexdigest(),
        'supervisor_result_sha256': result_hash,
        'process_instance_sha256': process_hash,
        'executable_sha256': child['sha256']})
    write_private_json(native_directory / 'provenance.json', {
        'schema': 'rosevine-nll-capture-provenance-v1', 'identity_sha256': identity_sha256,
        'plan_sha256': plan['plan_sha256'], 'inputs_sha256': plan['inputs_sha256'],
        'corpus_sha256': manifest['corpus_sha256'], 'split': manifest['split'],
        'target_count': expected, 'logit_stage': plan['logit_stage'],
        'native_nll_sha256': hashlib.sha256(raw_values).hexdigest(),
        'supervisor_result_sha256': result_hash,
        'supervisor_status': supervision['status'], 'supervisor_returncode': supervision['returncode'],
        'scope': 'capture integrity and corpus alignment; run identity truth remains an independent audit'})
    return capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--corpus', type=Path, required=True)
    prep.add_argument('--split', choices=('calibration', 'holdout'), required=True)
    prep.add_argument('--model-sha256', required=True)
    prep.add_argument('--tokenizer-sha256', required=True)
    prep.add_argument('--output-directory', type=Path, required=True)
    prep.add_argument('--vocabulary', type=int, default=262144)
    prep.add_argument('--capacity', type=int, default=4096)
    finish = sub.add_parser('finalize')
    finish.add_argument('--native-directory', type=Path, required=True)
    finish.add_argument('--prepared-directory', type=Path, required=True)
    finish.add_argument('--identity-sha256', required=True)
    finish.add_argument('--supervisor-result', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        result = prepare(args.corpus, args.split, args.model_sha256, args.tokenizer_sha256,
                         args.output_directory, vocabulary=args.vocabulary, capacity_tokens=args.capacity)
        print('Prepared %d private NLL targets; no model run.' % result['target_count'])
    else:
        supervision = json.loads(private_read(args.supervisor_result, 65536), object_pairs_hook=_unique_object)
        capture = finalize(args.native_directory, args.prepared_directory,
                           args.identity_sha256, supervision)
        print('Finalized %d private NLL documents; no model launched by adapter.' % len(capture['documents']))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, UnicodeError):
        print('NLL capture preparation/finalization failed; preserve partial evidence.', file=sys.stderr)
        sys.exit(1)
