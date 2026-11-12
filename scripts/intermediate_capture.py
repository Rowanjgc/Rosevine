#!/usr/bin/env python3
"""Private preparation, validation and comparison for bounded MoE captures.

This module never launches model inference. Native execution is a separate,
externally supervised operation that needs its own approved run conditions.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

from benchmark_core import canonical, digest, open_private, write_private_json
from logit_capture import encode_inputs, private_read, unique_object
from verify_capture_bundle import secure_json, verify_bundle

MAGIC = b'RVMOE001'
STARTED = b'rosevine-moe-intermediates-v1\n'
PREFIXES = ('ffn_moe_logits-', 'ffn_moe_probs-', 'ffn_moe_topk-', 'ffn_moe_weights-',
            'ffn_moe_gate_up-', 'ffn_moe_gelu-', 'ffn_moe_down-', 'ffn_moe_weighted-',
            'ffn_moe_out-', 'ffn_moe-', 'ffn_moe_combined-')
CAPTURE_LIMIT = 100 * 1024 * 1024
TENSOR_LIMIT = 16 * 1024 * 1024


def private_dir(path):
    path = Path(path)
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o700 or st.st_uid != os.getuid():
        raise ValueError('owned private directory required')
    return path


def prepare(inputs_directory, mode, output_directory, *, expected_moe_layers=None):
    inputs_directory = private_dir(inputs_directory)
    manifest = json.loads(private_read(inputs_directory / 'inputs.json', 256 * 1024), object_pairs_hook=unique_object)
    raw = private_read(inputs_directory / 'inputs.bin', 256 * 1024)
    if raw not in (encode_inputs(manifest, extra_buffers=False),
                   encode_inputs(manifest, extra_buffers=True)):
        raise ValueError('input bytes do not match the frozen manifest')
    if len(manifest['sequences']) != 1 or len(manifest['sequences'][0]['token_ids']) < 2:
        raise ValueError('one nonempty sequence required for position-zero capture')
    if mode not in ('ordinary', 'repack'):
        raise ValueError('explicit ordinary/repack control required')
    raw = encode_inputs(manifest, extra_buffers=(mode == 'repack'))
    if (expected_moe_layers is not None
            and (not isinstance(expected_moe_layers, list) or not expected_moe_layers
                 or any(type(layer) is not int or layer < 0 for layer in expected_moe_layers)
                 or expected_moe_layers != sorted(set(expected_moe_layers)))):
        raise ValueError('expected MoE layers must be an ordered unique nonnegative ID list')
    output_directory = Path(output_directory)
    output_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_private_json(output_directory / 'inputs.json', manifest)
    with open_private(output_directory / 'inputs.bin') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    write_private_json(output_directory / 'prepared.json', {
        'schema': 'rosevine-moe-intermediates-prepared-v1',
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'manifest_sha256': digest(manifest),
        'model_sha256': manifest['model_sha256'],
        'tokenizer_sha256': manifest['tokenizer_sha256'],
        'sequence_id': manifest['sequences'][0]['id'],
        'input_tokens': len(manifest['sequences'][0]['token_ids']),
        'mode': mode,
        'expected_moe_layers': expected_moe_layers,
        'capture_position': 0,
        'nodes': list(PREFIXES),
        'limits': {'total_bytes': CAPTURE_LIMIT, 'tensor_bytes': TENSOR_LIMIT},
        'native_policy': {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                          'microbatch': 1, 'flash_attention': False, 'kv': 'F16/F16',
                          'MTP': False, 'extra_buffers': mode == 'repack'},
        'instrumentation': 'ggml scheduler eval callback synchronizes and materializes selected named nodes at position zero; not a performance run',
        'authorization': 'preparation grants no model-run authority'})
    return output_directory


def parse_records(raw):
    """Parse one already-secured bounded capture byte string."""
    if not isinstance(raw, bytes) or len(raw) > CAPTURE_LIMIT:
        raise ValueError('bounded capture bytes required')
    with io.BytesIO(raw) as f:
        if f.read(8) != MAGIC:
            raise ValueError('intermediate capture magic mismatch')
        index = []
        total = 8
        while True:
            start = f.tell()
            raw = f.read(4)
            if not raw:
                break
            if len(raw) != 4:
                raise ValueError('truncated record header')
            position, = struct.unpack('<I', raw)
            name_len_raw = f.read(2)
            if len(name_len_raw) != 2:
                raise ValueError('truncated node name length')
            name_len, = struct.unpack('<H', name_len_raw)
            if not 1 <= name_len <= 127:
                raise ValueError('node name length outside envelope')
            name = f.read(name_len).decode('ascii')
            type_len_raw = f.read(2)
            if len(type_len_raw) != 2:
                raise ValueError('truncated dtype length')
            type_len, = struct.unpack('<H', type_len_raw)
            if not 1 <= type_len <= 15:
                raise ValueError('dtype name length outside envelope')
            dtype = f.read(type_len).decode('ascii')
            rank_raw = f.read(4)
            if len(rank_raw) != 4:
                raise ValueError('truncated rank')
            rank, = struct.unpack('<I', rank_raw)
            dims_raw = f.read(32)
            size_raw = f.read(8)
            if len(dims_raw) != 32 or len(size_raw) != 8:
                raise ValueError('truncated tensor descriptor')
            shape4 = struct.unpack('<4Q', dims_raw)
            nbytes, = struct.unpack('<Q', size_raw)
            shape = shape4[:rank] if 1 <= rank <= 4 else ()
            if (not shape or any(n < 1 for n in shape) or any(n != 1 for n in shape4[rank:])
                    or dtype not in ('f32', 'i32') or nbytes > TENSOR_LIMIT
                    or nbytes != math.prod(shape) * 4):
                raise ValueError('tensor dtype/shape/size contract failed')
            if not any(name.startswith(p) for p in PREFIXES) or not re.fullmatch(r'[a-z0-9_]+-[0-9]+', name):
                raise ValueError('unexpected named node')
            expected_dtype = 'i32' if name.startswith('ffn_moe_topk-') else 'f32'
            if dtype != expected_dtype:
                raise ValueError('named MoE node has unexpected dtype')
            data_offset = f.tell()
            data = f.read(nbytes)
            if len(data) != nbytes:
                raise ValueError('truncated tensor data')
            if dtype == 'f32' and any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)):
                raise ValueError('nonfinite intermediate value')
            if dtype == 'i32' and any(x[0] < 0 for x in struct.iter_unpack('<i', data)):
                raise ValueError('negative routed expert index')
            key = (position, name)
            if position != 0 or any((r['position'], r['name']) == key for r in index):
                raise ValueError('wrong or duplicate captured position/node')
            record_bytes = f.tell() - start
            total += record_bytes
            if total > CAPTURE_LIMIT:
                raise ValueError('capture total exceeds approved evidence limit')
            index.append({'position': position, 'name': name, 'dtype': dtype,
                          'rank': rank, 'shape': list(shape), 'nbytes': nbytes,
                          'record_offset': start, 'data_offset': data_offset,
                          'record_bytes': record_bytes, 'sha256': hashlib.sha256(data).hexdigest()})
        return index, total


def read_records(path):
    """Read a capture without following links or trusting its path metadata."""
    return parse_records(private_read(path, CAPTURE_LIMIT))


def finalize(capture_directory, prepared_directory, identity_sha256, supervision):
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
            or not isinstance(child.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', child['sha256'])
            or supervision['workload_cleanup'].get('pid') != child['pid']):
        raise ValueError('verified child process identity required')
    if not re.fullmatch(r'[0-9a-f]{64}', identity_sha256):
        raise ValueError('run identity digest required')
    capture_directory = private_dir(capture_directory)
    prepared_directory = private_dir(prepared_directory)
    prepared = json.loads(private_read(prepared_directory / 'prepared.json', 16384), object_pairs_hook=unique_object)
    if prepared.get('schema') != 'rosevine-moe-intermediates-prepared-v1':
        raise ValueError('prepared capture schema mismatch')
    expected_layers = prepared.get('expected_moe_layers')
    if (expected_layers is not None
            and (not isinstance(expected_layers, list) or not expected_layers
                 or any(type(layer) is not int or layer < 0 for layer in expected_layers)
                 or expected_layers != sorted(set(expected_layers)))):
        raise ValueError('prepared expected MoE layer inventory is invalid')
    native_policy = prepared.get('native_policy')
    required_policy = {'backend', 'threads', 'batch', 'microbatch', 'flash_attention',
                       'kv', 'MTP', 'extra_buffers'}
    if (not isinstance(native_policy, dict) or set(native_policy) != required_policy
            or type(native_policy.get('threads')) is not int
            or type(native_policy.get('batch')) is not int
            or type(native_policy.get('microbatch')) is not int
            or type(native_policy.get('flash_attention')) is not bool
            or type(native_policy.get('MTP')) is not bool
            or type(native_policy.get('extra_buffers')) is not bool
            or native_policy != {'backend': 'CPU only', 'threads': 2, 'batch': 1,
                                 'microbatch': 1, 'flash_attention': False,
                                 'kv': 'F16/F16', 'MTP': False,
                                 'extra_buffers': prepared.get('mode') == 'repack'}):
        raise ValueError('prepared native policy/mode mismatch')
    inputs = json.loads(private_read(prepared_directory / 'inputs.json', 256 * 1024), object_pairs_hook=unique_object)
    inputs_raw = private_read(prepared_directory / 'inputs.bin', 256 * 1024)
    if (encode_inputs(inputs, extra_buffers=native_policy['extra_buffers']) != inputs_raw
            or hashlib.sha256(inputs_raw).hexdigest() != prepared.get('input_sha256')
            or digest(inputs) != prepared.get('manifest_sha256') or inputs.get('model_sha256') != prepared.get('model_sha256')
            or inputs.get('tokenizer_sha256') != prepared.get('tokenizer_sha256')):
        raise ValueError('prepared frozen inputs failed identity validation')
    if private_read(capture_directory / 'started', 64) != STARTED:
        raise ValueError('capture start marker mismatch')
    receipt = private_read(capture_directory / 'complete', 16)
    if len(receipt) != 8:
        raise ValueError('completion receipt shape mismatch')
    count, position = struct.unpack('<2I', receipt)
    if position != 0:
        raise ValueError('unexpected capture position')
    nodes = capture_directory / 'nodes.bin'
    raw = private_read(nodes, CAPTURE_LIMIT)
    records, total = parse_records(raw)
    if len(records) != count or not records:
        raise ValueError('record count or empty capture mismatch')
    layers = {}
    for record in records:
        stage, layer = record['name'].rsplit('-', 1)
        layers.setdefault(stage, set()).add(int(layer))
    moe_layers = layers.get('ffn_moe_logits', set())
    required = ('ffn_moe_probs', 'ffn_moe_topk', 'ffn_moe_weights', 'ffn_moe_gate_up',
                'ffn_moe_gelu', 'ffn_moe_down', 'ffn_moe_out', 'ffn_moe', 'ffn_moe_combined')
    if not moe_layers or any(layers.get(stage, set()) != moe_layers for stage in required):
        raise ValueError('incomplete matched routing and routed-FFN node coverage')
    if expected_layers is not None and moe_layers != set(expected_layers):
        raise ValueError('captured MoE layers do not match the explicit expected layer inventory')
    if {p.name for p in capture_directory.iterdir()} != {'started', 'complete', 'nodes.bin'}:
        raise ValueError('unexpected capture artifacts')
    result = {'schema': 'rosevine-moe-intermediates-v1',
              'identity_sha256': identity_sha256,
              'prepared_sha256': hashlib.sha256(private_read(prepared_directory / 'prepared.json', 16384)).hexdigest(),
              'input_sha256': prepared['input_sha256'],
              'inputs_manifest_sha256': prepared['manifest_sha256'],
              'model_sha256': prepared['model_sha256'],
              'mode': prepared['mode'], 'native_policy': native_policy,
              'expected_moe_layers': expected_layers,
              'covered_moe_layers': sorted(moe_layers),
              'all_expected_moe_layers_covered': expected_layers is not None,
              'capture_position': 0,
              'nodes_sha256': hashlib.sha256(raw).hexdigest(), 'total_bytes': total,
              'records': records}
    write_private_json(capture_directory / 'index.json', result)
    binding = {'schema': 'rosevine-moe-capture-run-v1',
               'index_sha256': hashlib.sha256((capture_directory / 'index.json').read_bytes()).hexdigest(),
               'supervisor_result_sha256': hashlib.sha256(canonical(supervision) + b'\n').hexdigest(),
               'process_instance_sha256': hashlib.sha256(
                   f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest(),
               'executable_sha256': child['sha256']}
    write_private_json(capture_directory / 'capture-run-binding.json', binding)
    return result


def _verify_run_bundles(indexes, bindings, run_bundle_paths):
    if not isinstance(run_bundle_paths, (list, tuple)) or len(run_bundle_paths) != 2:
        raise ValueError('ordinary and repack supervisor bundles are both required')
    if Path(run_bundle_paths[0]).absolute() == Path(run_bundle_paths[1]).absolute():
        raise ValueError('ordinary and repack supervisor bundles must be distinct')
    instances, executables = [], []
    for index, binding, run_path in zip(indexes, bindings, run_bundle_paths):
        run_path = Path(run_path)
        summary = verify_bundle(run_path)
        if summary['run_status'] != 'successful':
            raise ValueError('capture supervisor run did not succeed')
        manifest = secure_json(run_path / 'manifest.json')
        run_identity = manifest.get('run_identity')
        if (not isinstance(run_identity, dict)
                or run_identity.get('identity_sha256') != index['identity_sha256']):
            raise ValueError('capture identity differs from supervisor manifest')
        if (index.get('all_expected_moe_layers_covered') is True
                and (run_identity.get('expected_moe_layers') != index['expected_moe_layers']
                     or run_identity.get('prepared_sha256') != index['prepared_sha256'])):
            raise ValueError('full-layer capture inventory differs from supervisor manifest')
        result_path = run_path / 'result.json'
        result_hash = private_sha256(result_path)
        if result_hash != binding['supervisor_result_sha256']:
            raise ValueError('capture binding differs from supervisor result')
        result = secure_json(result_path)
        child = result.get('child_process_identity')
        cleanup = result.get('workload_cleanup')
        if (result.get('status') != 'successful' or result.get('launched') is not True
                or result.get('returncode') != 0 or not isinstance(child, dict)
                or child.get('reason') is not None or type(child.get('pid')) is not int
                or type(child.get('start_ticks')) is not int or not isinstance(child.get('sha256'), str)
                or not isinstance(cleanup, dict) or cleanup.get('pending') is not False
                or cleanup.get('pid') != child['pid'] or cleanup.get('returncode') != 0):
            raise ValueError('supervisor bundle lacks successful reaped child identity')
        instance = hashlib.sha256(f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest()
        if (binding['process_instance_sha256'] != instance
                or binding['executable_sha256'] != child['sha256']):
            raise ValueError('capture binding disagrees with supervised process identity')
        instances.append(instance)
        executables.append(child['sha256'])
    if instances[0] == instances[1]:
        raise ValueError('ordinary/repack captures reuse one child process')
    if executables[0] != executables[1]:
        raise ValueError('ordinary/repack captures use different executables')
    return {'distinct_successful_processes_verified': True,
            'common_executable_sha256': executables[0],
            'process_instance_sha256': instances}


def private_sha256(path, limit=16 * 1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > limit):
            raise ValueError('bounded private regular file required')
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
        return hasher.hexdigest()


def compare(ordinary_directory, repack_directory, *, run_bundles=None):
    indexes, bindings = [], []
    for directory in (ordinary_directory, repack_directory):
        directory = private_dir(directory)
        index = json.loads(private_read(directory / 'index.json', CAPTURE_LIMIT), object_pairs_hook=unique_object)
        if index.get('schema') != 'rosevine-moe-intermediates-v1':
            raise ValueError('intermediate index schema mismatch')
        raw = private_read(directory / 'nodes.bin', CAPTURE_LIMIT)
        if hashlib.sha256(raw).hexdigest() != index.get('nodes_sha256'):
            raise ValueError('capture checksum mismatch')
        records, _ = parse_records(raw)
        binding = secure_json(directory / 'capture-run-binding.json', 16384)
        if (not isinstance(binding, dict) or set(binding) !=
                {'schema', 'index_sha256', 'supervisor_result_sha256',
                 'process_instance_sha256', 'executable_sha256'}
                or binding.get('schema') != 'rosevine-moe-capture-run-v1'
                or binding.get('index_sha256') != private_sha256(directory / 'index.json')):
            raise ValueError('capture run binding is missing or does not match its index')
        indexed = {r['name']: r for r in index['records']}
        if indexed != {r['name']: r for r in records}:
            raise ValueError('index does not match capture records')
        indexes.append((index, raw, indexed))
        bindings.append(binding)
    (a, araw, amap), (b, braw, bmap) = indexes
    expected_policy_fields = {'backend', 'threads', 'batch', 'microbatch',
                              'flash_attention', 'kv', 'MTP', 'extra_buffers'}
    for index in (a, b):
        policy = index.get('native_policy')
        if (not isinstance(policy, dict) or set(policy) != expected_policy_fields
                or type(policy.get('threads')) is not int or type(policy.get('batch')) is not int
                or type(policy.get('microbatch')) is not int
                or type(policy.get('flash_attention')) is not bool
                or type(policy.get('MTP')) is not bool
                or policy.get('backend') != 'CPU only' or policy.get('threads') != 2
                or policy.get('batch') != 1 or policy.get('microbatch') != 1
                or policy.get('flash_attention') is not False or policy.get('kv') != 'F16/F16'
                or policy.get('MTP') is not False
                or type(policy.get('extra_buffers')) is not bool
                or policy['extra_buffers'] != (index['mode'] == 'repack')):
            raise ValueError('capture native policy is missing or inconsistent with its mode')
    ordinary_policy = {key: value for key, value in a['native_policy'].items()
                       if key != 'extra_buffers'}
    repack_policy = {key: value for key, value in b['native_policy'].items()
                     if key != 'extra_buffers'}
    if ordinary_policy != repack_policy:
        raise ValueError('ordinary/repack comparison changes more than extra-buffer placement')
    run_evidence = (_verify_run_bundles((a, b), bindings, run_bundles)
                    if run_bundles is not None else None)
    if ({a['mode'], b['mode']} != {'ordinary', 'repack'}
            or a.get('inputs_manifest_sha256') != b.get('inputs_manifest_sha256')
            or a['model_sha256'] != b['model_sha256'] or set(amap) != set(bmap)):
        raise ValueError('matched ordinary/repack inputs and node sets required')
    if (a['expected_moe_layers'] != b['expected_moe_layers']
            or a['covered_moe_layers'] != b['covered_moe_layers']
            or a['all_expected_moe_layers_covered'] != b['all_expected_moe_layers_covered']):
        raise ValueError('ordinary/repack expected and covered MoE layer inventories differ')
    outputs = []
    for name in sorted(amap):
        ar, br = amap[name], bmap[name]
        if ar['dtype'] != br['dtype'] or ar['shape'] != br['shape']:
            raise ValueError('node dtype or shape mismatch')
        adata = araw[ar['data_offset']:ar['data_offset'] + ar['nbytes']]
        bdata = braw[br['data_offset']:br['data_offset'] + br['nbytes']]
        if ar['dtype'] == 'i32':
            neq = sum(x[0] != y[0] for x, y in zip(struct.iter_unpack('<i', adata), struct.iter_unpack('<i', bdata)))
            outputs.append({'name': name, 'dtype': 'i32', 'elements': ar['nbytes'] // 4,
                            'exact_match': neq == 0, 'mismatches': neq})
            continue
        n = ar['nbytes'] // 4
        sum_ref2 = sum_cand2 = dot = sum_err2 = max_abs = 0.0
        max_rel = 0.0
        for x, y in zip(struct.iter_unpack('<f', adata), struct.iter_unpack('<f', bdata)):
            ref, cand = x[0], y[0]
            err = cand - ref
            ae = abs(err)
            max_abs = max(max_abs, ae)
            max_rel = max(max_rel, ae / max(abs(ref), 1e-30))
            sum_ref2 += ref * ref
            sum_cand2 += cand * cand
            dot += ref * cand
            sum_err2 += err * err
        ref_rms = math.sqrt(sum_ref2 / n) if n else 0.0
        nrms = math.sqrt(sum_err2 / n) / ref_rms if ref_rms else (0.0 if max_abs == 0 else None)
        cosine = dot / math.sqrt(sum_ref2 * sum_cand2) if sum_ref2 and sum_cand2 else (1.0 if max_abs == 0 else None)
        outputs.append({'name': name, 'dtype': 'f32', 'elements': n,
                        'max_abs': max_abs, 'max_rel': max_rel,
                        'normalized_rmse': nrms, 'cosine': cosine,
                        'proposed_M_pass': (max_abs <= 1e-5 if ref_rms == 0 else
                                            nrms is not None and nrms <= 1e-3 and cosine is not None and cosine >= 0.9999)})
    def read_values(raw, record):
        data = raw[record['data_offset']:record['data_offset'] + record['nbytes']]
        code = 'f' if record['dtype'] == 'f32' else 'i'
        return tuple(value[0] for value in struct.iter_unpack('<' + code, data))

    routing = []
    for name in sorted(key for key in amap if key.startswith('ffn_moe_logits-')):
        layer = name.rsplit('-', 1)[1]
        ids_name = f'ffn_moe_topk-{layer}'
        if ids_name not in amap or ids_name not in bmap:
            raise ValueError('router logits lack matching selected expert IDs')
        ordinary_logits = read_values(araw, amap[name])
        repack_logits = read_values(braw, bmap[name])
        ordinary_ids = read_values(araw, amap[ids_name])
        repack_ids = read_values(braw, bmap[ids_name])
        k = len(ordinary_ids)
        if len(repack_ids) != k or not 0 < k < len(ordinary_logits) or len(repack_logits) != len(ordinary_logits):
            raise ValueError('router top-k/logit dimensions cannot establish the boundary margin')
        ordinary_sorted = sorted(ordinary_logits, reverse=True)
        repack_sorted = sorted(repack_logits, reverse=True)
        routing.append({'layer': int(layer), 'k': k,
                        'selected_ids_exact': ordinary_ids == repack_ids,
                        'ordinary_topk_ids': list(ordinary_ids), 'repack_topk_ids': list(repack_ids),
                        'ordinary_boundary_margin': ordinary_sorted[k - 1] - ordinary_sorted[k],
                        'repack_boundary_margin': repack_sorted[k - 1] - repack_sorted[k]})
    return {'schema': 'rosevine-moe-intermediates-comparison-v1',
            'ordinary_identity_sha256': a['identity_sha256'], 'repack_identity_sha256': b['identity_sha256'],
            'inputs_manifest_sha256': a['inputs_manifest_sha256'],
            'ordinary_input_binary_sha256': a['input_sha256'],
            'repack_input_binary_sha256': b['input_sha256'], 'position': 0,
            'run_provenance_verified': run_evidence is not None,
            'run_provenance': run_evidence,
            'expected_moe_layers': a['expected_moe_layers'],
            'covered_moe_layers': a['covered_moe_layers'],
            'all_expected_moe_layers_covered': (a['all_expected_moe_layers_covered']
                                                and run_evidence is not None),
            'proposed_M_is_calibrated': False,
            'interpretation': 'diagnostic only; an M pass does not validate model correctness and a failure does not identify the correct path',
            'router_boundary_margins': routing,
            'nodes': outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--inputs-directory', type=Path, required=True)
    prep.add_argument('--mode', choices=('ordinary', 'repack'), required=True)
    prep.add_argument('--expected-moe-layers', type=int, nargs='+')
    prep.add_argument('--output-directory', type=Path, required=True)
    finish = sub.add_parser('finalize')
    finish.add_argument('--capture-directory', type=Path, required=True)
    finish.add_argument('--prepared-directory', type=Path, required=True)
    finish.add_argument('--identity-sha256', required=True)
    finish.add_argument('--supervisor-result', type=Path, required=True)
    diff = sub.add_parser('compare')
    diff.add_argument('--ordinary-directory', type=Path, required=True)
    diff.add_argument('--repack-directory', type=Path, required=True)
    diff.add_argument('--ordinary-run-bundle', type=Path)
    diff.add_argument('--repack-run-bundle', type=Path)
    diff.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.inputs_directory, args.mode, args.output_directory,
                expected_moe_layers=args.expected_moe_layers)
    elif args.operation == 'finalize':
        supervision = json.loads(private_read(args.supervisor_result, 65536), object_pairs_hook=unique_object)
        finalize(args.capture_directory, args.prepared_directory, args.identity_sha256, supervision)
    else:
        if (args.ordinary_run_bundle is None) != (args.repack_run_bundle is None):
            raise ValueError('supply both ordinary and repack supervisor run bundles')
        bundles = ([args.ordinary_run_bundle, args.repack_run_bundle]
                   if args.ordinary_run_bundle is not None else None)
        write_private_json(args.output, compare(args.ordinary_directory, args.repack_directory,
                                                run_bundles=bundles))
    print('Offline intermediate capture preparation/validation complete; no model launched.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError, UnicodeError):
        print('Intermediate capture operation failed; retain partial evidence.', file=sys.stderr)
        sys.exit(1)
