#!/usr/bin/env python3
"""CPU-only exhaustive finite-scale Q4_0 cross-check against an explicit library.

No model load or GPU initialization. Uses dequantize_row_q4_0's C ABI: packed
18-byte block pointer, FP32 output pointer, int64 logical element count. A native
library is executable code: select only the locally verified reference artifact.
"""
import argparse
import ctypes
import hashlib
from pathlib import Path
import struct
import sys

from benchmark_core import write_private_json
from correctness_oracle import q4_0_decode


def verify(library):
    library = Path(library).resolve(strict=True)
    native = ctypes.CDLL(str(library))
    decode = native.dequantize_row_q4_0
    decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    decode.restype = None
    payloads = [bytes((15 - i) * 16 + i for i in range(16)), bytes(16), bytes([255] * 16)]
    checked = 0
    for half in range(65536):
        if half & 0x7c00 == 0x7c00:
            continue
        for payload in payloads:
            block = struct.pack('<H', half) + payload
            packed = ctypes.create_string_buffer(block)
            actual = (ctypes.c_float * 32)()
            decode(packed, actual, 32)
            expected = struct.pack('<32f', *q4_0_decode(block))
            if bytes(actual) != expected:
                raise ValueError('native/scalar FP32 bit mismatch')
            checked += 1
    return {'status': 'successful', 'blocks': checked, 'finite_fp16_scales': 63488,
            'payloads_per_scale': len(payloads), 'comparison': 'exact FP32 bits including signed zero',
            'coverage': 'all nibble values in both halves; all-zero and all-15 payloads; every finite FP16 scale',
            'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
            'limitations': 'synthetic decoder only; no tensor capture, quantizer or graph equivalence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new file in private 0700 parent')
    args = parser.parse_args()
    result = verify(args.library)
    write_private_json(args.output, result)
    print('Native/scalar synthetic Q4_0 FP32 bit cross-check passed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, AttributeError):
        print('Native Q4_0 check failed; no correctness pass recorded.', file=sys.stderr)
        sys.exit(1)
