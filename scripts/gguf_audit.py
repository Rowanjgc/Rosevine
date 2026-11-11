#!/usr/bin/env python3
"""Independent bounded GGUF v3 descriptor audit; output is PRIVATE evidence.

No model loading, inference, imports from gguf-py, or artifact modifications.
Only F32 and Q4_0 layouts at the reference GGML revision are supported.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


class InvalidGGUF(ValueError):
    """Malformed or unsupported input; messages contain no artifact content."""


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.size = os.fstat(stream.fileno()).st_size
        self.budget = 128 * 1024 * 1024

    def read(self, count):
        if count < 0 or count > self.budget:
            raise InvalidGGUF('metadata budget exceeded')
        self.budget -= count
        data = self.stream.read(count)
        if len(data) != count:
            raise InvalidGGUF('truncated header')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.read(self.number('Q')).decode('utf-8', errors='strict')

    def value(self, kind, depth=0):
        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                   6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in formats:
            return self.number(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9 and depth == 0:
            child, count = self.number('I'), self.number('Q')
            if count > 2_000_000:
                raise InvalidGGUF('array limit exceeded')
            return [self.value(child, depth + 1) for _ in range(count)]
        raise InvalidGGUF('unsupported metadata type')


def audit(path, full_hash=False):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        reader = Reader(stream)
        if reader.read(4) != b'GGUF' or reader.number('I') != 3:
            raise InvalidGGUF('expected little-endian GGUF v3')
        tensor_count, kv_count = reader.number('Q'), reader.number('Q')
        if not 0 < tensor_count <= 100_000 or kv_count > 100_000:
            raise InvalidGGUF('descriptor count limit exceeded')
        metadata = {}
        for _ in range(kv_count):
            key = reader.string()
            if key in metadata:
                raise InvalidGGUF('duplicate metadata key')
            metadata[key] = reader.value(reader.number('I'))
        alignment = metadata.get('general.alignment', 32)
        if type(alignment) is not int or not 0 < alignment <= 1024 * 1024 or alignment & (alignment - 1):
            raise InvalidGGUF('invalid alignment')
        tensors, names = [], set()
        layouts = {0: ('F32', 1, 4), 2: ('Q4_0', 32, 18)}
        for _ in range(tensor_count):
            name, dims = reader.string(), reader.number('I')
            if name in names or not name or not 1 <= dims <= 4:
                raise InvalidGGUF('invalid tensor name or rank')
            names.add(name)
            shape = [reader.number('Q') for _ in range(dims)]
            kind, offset = reader.number('I'), reader.number('Q')
            if kind not in layouts:
                raise InvalidGGUF('unsupported tensor layout')
            label, block, size = layouts[kind]
            if not all(shape) or shape[0] % block or offset % alignment:
                raise InvalidGGUF('invalid shape or tensor alignment')
            length = math.prod(shape) // block * size
            tensors.append(dict(name=name, shape=shape, type=label, offset=offset, bytes=length))
        start = (stream.tell() + alignment - 1) // alignment * alignment
        end = 0
        for tensor in sorted(tensors, key=lambda item: item['offset']):
            offset = tensor['offset']
            if offset < end or start + offset + tensor['bytes'] > reader.size:
                raise InvalidGGUF('overlapping or out-of-file tensor range')
            end = offset + tensor['bytes']
        digest = None
        if full_hash:
            stream.seek(0)
            hasher = hashlib.sha256()
            while chunk := stream.read(8 * 1024 * 1024):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise InvalidGGUF('artifact changed during inspection')
        return dict(schema=1, size=reader.size, sha256=digest, data_offset=start,
                    tensor_count=tensor_count, tensors=tensors, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--full-hash', action='store_true', help='read all bytes; schedule outside timings')
    args = parser.parse_args()
    try:
        result = audit(args.artifact, args.full_hash)
    except (OSError, ValueError, struct.error, OverflowError):
        print('GGUF audit failed: unreadable, malformed, unsupported or changing artifact.', file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, ensure_ascii=True, allow_nan=False)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
