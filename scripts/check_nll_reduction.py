#!/usr/bin/env python3
"""Cross-check Python target NLL against an independent long-double reducer.

Consumes existing private full-FP32 logit indexes; it does not run a model.
Builds a tiny standard-library-only executable and retains only private scalar
reduction results and artifact hashes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys

from benchmark_core import digest, open_private, write_private_json
from logit_capture import encode_inputs, private_read
from logit_compare import load_index, rows
from nll_compare import target_nll

ROOT = Path(__file__).resolve().parents[1]
LIMIT_NATS = 2e-12


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def compile_reducer(source, binary, report_directory):
    source, binary = Path(source).resolve(strict=True), Path(binary).absolute()
    expected_source = ROOT / 'research/nll_row_reduction.cpp'
    if source != expected_source.resolve():
        raise ValueError('independent reducer source must be the reviewed repository implementation')
    parent = binary.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700
            or parent.st_uid != os.getuid() or binary.exists() or binary.is_symlink()):
        raise ValueError('new binary under owned private .tools directory required')
    compiler = shutil.which('g++')
    if not compiler:
        raise ValueError('existing g++ is required; no compiler installation attempted')
    command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               '-fno-fast-math', '-ffp-contract=off', str(source), '-o', str(binary)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={'PATH': '/usr/bin:/bin', 'LANG': 'C'}, timeout=60, check=False)
    with open_private(Path(report_directory) / 'build.stdout') as stream:
        stream.write(result.stdout)
    with open_private(Path(report_directory) / 'build.stderr') as stream:
        stream.write(result.stderr)
    if result.returncode:
        raise ValueError('independent reducer build failed; private diagnostics retained')
    os.chmod(binary, 0o700)
    return command


def _inputs_manifest(path):
    manifest = json.loads(private_read(path, 256 * 1024))
    encode_inputs(manifest, extra_buffers=False)  # Validate the exact frozen input schema.
    return manifest


def _expected_rows(manifest):
    return {(sequence['id'], position): sequence['token_ids'][position + 1]
            for sequence in manifest['sequences']
            for position in sequence.get('capture_input_positions', range(len(sequence['token_ids']) - 1))}


def verify(reference_index, repeat_index, inputs_path, source, binary, output):
    output = Path(output).absolute()
    report_directory = output.parent
    parent = report_directory.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700
            or parent.st_uid != os.getuid() or output.exists() or output.is_symlink()):
        raise ValueError('new private report file in owned 0700 directory required')
    manifest = _inputs_manifest(inputs_path)
    inputs_sha = digest(manifest)
    expected = _expected_rows(manifest)
    if not expected:
        raise ValueError('nonempty frozen full-logit positions required')
    reference = load_index(reference_index)
    repeat = load_index(repeat_index)
    for index in (reference, repeat):
        if (index['inputs_sha256'] != inputs_sha or index['vocabulary'] != manifest['vocabulary']
                or index['logit_stage'] not in ('pre-softcap', 'post-softcap')):
            raise ValueError('full-logit index differs from frozen input identity')
        labels = [(sequence, position) for shard in index['shards'] for sequence, position in shard['positions']]
        if set(labels) != set(expected) or len(labels) != len(expected):
            raise ValueError('full-logit positions do not exactly cover target IDs')
    if (reference['inputs_sha256'] != repeat['inputs_sha256']
            or reference['logit_stage'] != repeat['logit_stage']
            or reference['vocabulary'] != repeat['vocabulary']):
        raise ValueError('reference/repeat full-logit contracts differ')
    command = compile_reducer(source, binary, report_directory)
    comparisons, max_difference = [], 0.0
    for capture_name, capture_path, index in (
            ('reference', reference_index, reference), ('repeat', repeat_index, repeat)):
        count = 0
        for label, values in rows(capture_path, index):
            if label not in expected or any(not math.isfinite(value) for value in values):
                raise ValueError('unexpected or nonfinite full-logit row')
            target = expected[label]
            python_value = target_nll(values, target)
            raw = struct.pack('<%df' % len(values), *values)
            payload = struct.pack('<II', target, len(values)) + raw
            reduced = subprocess.run([str(binary)], input=payload, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=30, check=False)
            if reduced.returncode != 0:
                raise ValueError('independent full-logit reducer rejected a retained row')
            try:
                independent_value = float(reduced.stdout.decode('ascii').strip())
            except (UnicodeDecodeError, ValueError):
                raise ValueError('independent reducer returned malformed output') from None
            if not math.isfinite(independent_value) or independent_value < 0:
                raise ValueError('independent reducer returned invalid NLL')
            difference = abs(python_value - independent_value)
            max_difference = max(max_difference, difference)
            comparisons.append({'capture': capture_name, 'sequence_id': label[0],
                                'input_position': label[1], 'python_target_nll_nats': python_value,
                                'independent_long_double_nll_nats': independent_value,
                                'absolute_difference_nats': difference})
            count += 1
        if count != len(expected):
            raise ValueError('capture row iterator did not reach every expected position')
    if max_difference > LIMIT_NATS:
        raise ValueError('Python and independent full-logit reductions disagree')
    result = {'schema': 'rosevine-nll-full-logit-reduction-v1', 'status': 'successful',
              'input_manifest_sha256': inputs_sha,
              'reference_index_sha256': sha256(reference_index),
              'repeat_index_sha256': sha256(repeat_index),
              'reducer_source_sha256': sha256(source), 'reducer_binary_sha256': sha256(binary),
              'compile_command': command, 'logit_stage': reference['logit_stage'],
              'vocabulary': reference['vocabulary'], 'capture_count': 2,
              'rows_per_capture': len(expected), 'total_rows': len(comparisons),
              'max_absolute_reduction_difference_nats': max_difference,
              'numerical_cross_check_limit_nats': LIMIT_NATS,
              'comparison': 'Python binary64 fsum target NLL versus independent C++ long-double full-vocabulary log-sum-exp',
              'rows': comparisons,
              'scope': 'reduction agreement on existing full-logit rows only; not actual corpus NLL, model quality or noninferiority'}
    write_private_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-index', type=Path, required=True)
    parser.add_argument('--repeat-index', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'research/nll_row_reduction.cpp')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference_index, args.repeat_index, args.inputs,
                    args.source, args.binary, args.output)
    print('Independent NLL reduction matched %d existing full-logit rows; no model run.' % result['total_rows'])


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, subprocess.SubprocessError):
        print('Full-logit NLL reduction check failed; private diagnostics retained.', file=sys.stderr)
        sys.exit(1)
