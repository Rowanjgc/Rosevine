#!/usr/bin/env python3
"""Cross-check deterministic real-target Q4_0 blocks without model inference.

Only selected 18-byte tensor blocks are read. The file is never loaded as a
model; no GPU/backend is initialized. Results contain counts and identities, not
weight bytes or decoded tensor values, and must be written to a private path.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

from benchmark_core import write_private_json
from correctness_oracle import q4_0_decode
from logit_capture import private_read, unique_object

BLOCK_ELEMENTS = 32
BLOCK_BYTES = 18
TENSOR_RE = re.compile(r'^blk\.([0-9]+)\.ffn_(down|gate_up)_exps\.weight$')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def sample_positions(tensor_name, experts, blocks_per_expert):
    """Pick two boundary and two deterministic interior blocks per expert."""
    if type(experts) is not int or experts <= 0 or type(blocks_per_expert) is not int or blocks_per_expert < 2:
        raise ValueError('positive experts and at least two blocks per expert required')
    positions = []
    for expert in range(experts):
        base = expert * blocks_per_expert
        for local in sample_tensor_positions(f'{tensor_name}\0{expert}', blocks_per_expert):
            positions.append(base + local)
    return list(dict.fromkeys(positions))


def sample_tensor_positions(tensor_name, block_count):
    """Select at most four stable blocks, always including both boundaries."""
    if type(block_count) is not int or block_count <= 0:
        raise ValueError('positive tensor block count required')
    if block_count <= 3:
        return list(range(block_count))
    seed = hashlib.sha256(tensor_name.encode()).digest()
    a = 1 + int.from_bytes(seed[:8], 'little') % (block_count - 2)
    b = 1 + int.from_bytes(seed[8:16], 'little') % (block_count - 2)
    if a == b:
        b = 1 + (b % (block_count - 2))
    return sorted({0, block_count - 1, a, b})


def verify(model_path, descriptor_path, library_path, expected_model_sha256, output_path):
    model_path, descriptor_path, library_path = map(Path, (model_path, descriptor_path, library_path))
    fd = os.open(model_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_size <= 0:
            raise ValueError('owned nonempty regular model file required')
        descriptor = json.loads(private_read(descriptor_path, 32 * 1024 * 1024), object_pairs_hook=unique_object)
        if (descriptor.get('schema') != 1 or descriptor.get('size') != before.st_size
                or descriptor.get('sha256') != expected_model_sha256
                or not re.fullmatch(r'[0-9a-f]{64}', expected_model_sha256)):
            raise ValueError('descriptor/model identity mismatch')
        actual_hash = file_sha256(model_path)
        if actual_hash != expected_model_sha256:
            raise ValueError('target model hash mismatch')

        native = ctypes.CDLL(str(library_path.resolve(strict=True)))
        decode = native.dequantize_row_q4_0
        decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        decode.restype = None
        data_offset = descriptor.get('data_offset')
        tensors = descriptor.get('tensors')
        if type(data_offset) is not int or not isinstance(tensors, list):
            raise ValueError('invalid independent descriptor')
        selected = []
        q4_count = 0
        for tensor in tensors:
            match = TENSOR_RE.fullmatch(tensor.get('name', ''))
            if tensor.get('type') != 'Q4_0':
                continue
            shape = tensor.get('shape')
            if not isinstance(shape, list) or not 1 <= len(shape) <= 4 or not all(
                    type(x) is int and x > 0 for x in shape):
                raise ValueError('invalid Q4_0 tensor dimensions')
            if shape[0] % BLOCK_ELEMENTS:
                raise ValueError('invalid Q4_0 row width')
            block_count = math.prod(shape) // BLOCK_ELEMENTS
            expected_bytes = block_count * BLOCK_BYTES
            if tensor.get('bytes') != expected_bytes:
                raise ValueError('tensor byte count mismatch')
            relative = tensor.get('offset')
            if type(relative) is not int or relative < 0 or data_offset + relative + expected_bytes > before.st_size:
                raise ValueError('Q4_0 tensor range outside model file')
            if match:
                if len(shape) != 3:
                    raise ValueError('routed expert tensor rank mismatch')
                ne0, ne1, experts = shape
                blocks_per_expert = (ne0 // BLOCK_ELEMENTS) * ne1
                if block_count != blocks_per_expert * experts:
                    raise ValueError('routed expert block count mismatch')
                positions = sample_positions(tensor['name'], experts, blocks_per_expert)
            else:
                experts, blocks_per_expert = 1, block_count
                positions = sample_tensor_positions(tensor['name'], block_count)
            selected.append((tensor, match, experts, blocks_per_expert, positions))
            q4_count += 1
        routed_families = {match.group(2) for tensor, match, *_ in selected if match}
        if not selected or routed_families != {'down', 'gate_up'}:
            raise ValueError('Q4_0 inventory or both routed expert tensor families missing')

        tensor_results = []
        total_blocks = 0
        with os.fdopen(os.dup(fd), 'rb', closefd=True) as stream:
            for tensor, match, experts, blocks_per_expert, block_indices in selected:
                for block_index in block_indices:
                    position = data_offset + tensor['offset'] + block_index * BLOCK_BYTES
                    stream.seek(position)
                    block = stream.read(BLOCK_BYTES)
                    if len(block) != BLOCK_BYTES:
                        raise ValueError('truncated Q4_0 tensor block')
                    reference = q4_0_decode(block)
                    packed = ctypes.create_string_buffer(block)
                    native_output = (ctypes.c_float * BLOCK_ELEMENTS)()
                    decode(packed, native_output, BLOCK_ELEMENTS)
                    if bytes(native_output) != struct.pack('<32f', *reference):
                        raise ValueError('native/scalar FP32 block mismatch')
                tensor_results.append({'name': tensor['name'],
                                       'layer': int(match.group(1)) if match else None,
                                       'family': match.group(2) if match else 'other_q4_0',
                                       'expert_count': experts if match else None,
                                       'blocks_per_expert': blocks_per_expert,
                                       'sampled_blocks': len(block_indices),
                                       'sampled_elements': len(block_indices) * BLOCK_ELEMENTS})
                total_blocks += len(block_indices)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('model file changed during block verification')
        result = {'schema': 'rosevine-native-q4-model-blocks-v1', 'status': 'successful',
                  'model_sha256': actual_hash, 'model_size': before.st_size,
                  'descriptor_sha256': file_sha256(descriptor_path),
                  'native_library_sha256': file_sha256(library_path),
                  'tensor_count': len(tensor_results), 'q4_0_tensor_count': q4_count,
                  'routed_expert_tensor_count': sum(1 for t, *_ in selected if TENSOR_RE.fullmatch(t['name'])),
                  'sample_policy': 'all Q4_0 tensors: first/last plus up to two deterministic interior blocks; routed experts sampled per expert',
                  'sampled_blocks': total_blocks, 'sampled_elements': total_blocks * BLOCK_ELEMENTS,
                  'comparison': 'bitwise FP32 equality for all decoded values',
                  'inference_run': False, 'gpu_initialized': False, 'tensors': tensor_results}
        write_private_json(output_path, result)
        return result
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.model, args.descriptor, args.library, args.expected_model_sha256, args.output)
    print('Actual target Q4_0 block cross-check passed: %d blocks across %d tensors (%d routed-expert tensors); no model inference.' %
          (result['sampled_blocks'], result['tensor_count'], result['routed_expert_tensor_count']))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error):
        print('Native target Q4_0 check failed; private diagnostics retained.', file=sys.stderr)
        sys.exit(1)
