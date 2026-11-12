#!/usr/bin/env python3
"""Optional pinned-header native capture control-flow check with an API double.

Requires an existing compiler and an explicitly selected clean pinned source
checkout. Links NO llama/ggml library, opens no weights and initializes no GPU.
The binary path is a new ignored local tool; all reports go to a new private
directory. This does not validate the deployed ABI, kernels or real cancellation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmark_core import digest, supervise, write_private_json
from logit_capture import finalize, prepare
from logit_compare import load_index, rows
from nll_capture import finalize as finalize_nll, prepare as prepare_nll
from nll_compare import target_nll

REVISION = 'cd959bae99ff4bed22bef2604f45241c09bbb8c8'
ROOT = Path(__file__).resolve().parents[1]


def synthetic_plan():
    sequences = []
    for name, first, length in [('first', 1, 18), ('second', 20, 17)]:
        ids = list(range(first, first + length))
        sequences.append({'id': name, 'token_ids': ids, 'tokens_sha256': digest(ids),
                          'reset_kv_before_sequence': True,
                          'scored_input_positions': [0, length - 2],
                          'scored_target_indices': [1, length - 1]})
    record = {'schema': 'rosevine-reference-positions-v1', 'sequences': sequences,
              'source_split': 'calibration', 'scored_positions': 33,
              'vocabulary': 64, 'capacity_tokens': 256}
    record['plan_sha256'] = digest(record)
    return record


def synthetic_context_plan():
    positions, streams = [2, 512, 598], []
    for name, first in [('context-first', 1), ('context-second', 20), ('context-third', 40)]:
        ids = [(first + i) % 64 for i in range(600)]
        streams.append({'id': name, 'token_ids': ids, 'tokens_sha256': digest(ids),
                        'reset_kv_before_sequence': True, 'capture_input_positions': positions,
                        'scored_target_indices': [p + 1 for p in positions],
                        'target_token_ids': [ids[p + 1] for p in positions]})
    record = {'schema': 'rosevine-context-positions-v1', 'streams': streams,
              'captured_input_positions': positions, 'source_split': 'calibration',
              'capture_rows': len(streams) * len(positions), 'vocabulary': 64,
              'capacity_tokens': 1024}
    record['plan_sha256'] = digest(record)
    return record


def synthetic_nll_corpus():
    documents = []
    for identifier, tokens in [('nll-first', [1, 3, 5, 7]), ('nll-second', [2, 4, 6])]:
        documents.append({'id': identifier, 'stratum': 'problem-prose', 'split': 'calibration',
                          'text': 'synthetic API-double input', 'text_sha256': 'a' * 64,
                          'dedup_sha256': 'b' * 64, 'token_ids': tokens,
                          'tokens_sha256': digest(tokens), 'scored_tokens': len(tokens) - 1,
                          'first_scored_target_index': 1})
    record = {'version': 'rosevine-fidelity-corpus-v1', 'documents': documents}
    record['corpus_sha256'] = digest(record)
    return record


def check(reference_source, binary, directory):
    reference_source, binary, directory = map(Path, (reference_source, binary, directory))
    reference_source, binary = reference_source.resolve(), binary.absolute()
    if binary.exists() or binary.is_symlink():
        raise ValueError('new test-double binary path required')
    revision = subprocess.check_output(['git', '-C', str(reference_source), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(reference_source), 'diff', 'HEAD', '--'], timeout=10)
    if revision != REVISION or dirty:
        raise ValueError('clean pinned reference headers required')
    compiler = shutil.which('g++')
    if not compiler:
        raise ValueError('existing C++ compiler required; no install attempted')
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    directory = directory.resolve()
    source = ROOT / 'research/capture_logits.cpp'
    protocol = ROOT / 'research/capture_input_protocol.h'
    double = ROOT / 'tests/research/native_capture_double.cpp'
    command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               '-fno-fast-math', '-ffp-contract=off', '-I' + str(reference_source / 'include'),
               '-I' + str(reference_source / 'ggml/include'), str(source), str(double), '-o', str(binary)]
    build = supervise(command, directory / 'build', timeout_s=60, env={'PATH': '/usr/bin:/bin', 'LANG': 'C'})
    if build['status'] != 'successful':
        raise ValueError('test-double compilation failed; private diagnostics retained')
    binary.chmod(0o700)
    write_private_json(directory / 'build-identity.json', {
        'scope': 'API double only; no native inference libraries linked', 'reference_revision': revision,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (source, protocol, double)},
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'command': command})
    plan = synthetic_plan()
    prepare(plan, 'a' * 64, 'b' * 64, directory / 'prepared', extra_buffers=False)
    results = {}
    scenarios = ['success', 'load-failure', 'wrong-vocabulary', 'context-failure',
                 'changed-capacity', 'decode-warning', 'decode-abort', 'decode-error',
                 'missing-logits', 'nonfinite', 'hang-load', 'hang-decode', 'noncpu', 'no-device']
    for scenario in scenarios:
        case = directory / scenario
        case.mkdir(mode=0o700)
        native = case / 'shards'
        native.mkdir(mode=0o700)
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C'}
        if scenario == 'noncpu':
            env['ROSEVINE_DOUBLE_GPU'] = '1'
        if scenario == 'no-device':
            env['ROSEVINE_DOUBLE_NO_DEVICE'] = '1'
        result = supervise([str(binary), '--run', scenario, str(directory / 'prepared/inputs.bin'), str(native)],
                           case / 'supervision', timeout_s=.4 if scenario.startswith('hang-') else 5,
                           grace_s=.05, env=env)
        if scenario == 'success':
            if result['status'] != 'successful':
                raise ValueError('test-double successful scenario failed')
            finalize(native, directory / 'prepared', 'c' * 64, result)
            observed = list(rows(native / 'index.json', load_index(native / 'index.json')))
            expected = [(s['id'], seq, j, token) for seq, s in enumerate(plan['sequences'])
                        for j, token in enumerate(s['token_ids'][:-1])]
            if len(observed) != len(expected):
                raise ValueError('native emitted wrong row count')
            for (label, values), (name, seq, j, token) in zip(observed, expected):
                if label != (name, j) or list(values) != [seq * 10000 + j * 100 + token + v for v in range(64)]:
                    raise ValueError('native row order, input ID or position mismatch')
            trace = (case / 'supervision/stdout.txt').read_text()
            expected_trace = [f'DOUBLE decode {seq} {j} {token}' for _, seq, j, token in expected]
            if [line for line in trace.splitlines() if line.startswith('DOUBLE decode ')] != expected_trace:
                raise ValueError('native decoded targets or changed explicit sequence positions')
            if trace.count('DOUBLE clear ') != 2 or trace.index('DOUBLE context-free') > trace.index('DOUBLE model-free'):
                raise ValueError('KV reset or cleanup order mismatch')
        else:
            if result['status'] == 'successful' or (native / 'complete').exists():
                raise ValueError('failed native producer reported completion')
            try:
                finalize(native, directory / 'prepared', 'c' * 64, result)
            except ValueError:
                pass
            else:
                raise ValueError('failed native producer accepted by finalizer')
            if scenario in ('decode-warning', 'decode-abort', 'decode-error', 'missing-logits', 'nonfinite', 'hang-decode'):
                if (native / 'logits-000000.f32').stat().st_size != 64 * 4:
                    raise ValueError('failed native producer lost its valid first row')
            if scenario.startswith('hang-'):
                cleanup = result['workload_cleanup']
                if cleanup['pending'] or cleanup['returncode'] != -9:
                    raise ValueError('TERM-ignoring test double was not killed/reaped')
        results[scenario] = {'passed': True, 'supervisor_status': result['status'],
                             'rows_retained': sum(p.stat().st_size // 256 for p in native.glob('*.f32'))}
    extra_prepared = directory / 'prepared-extra-buffers'
    prepare(plan, 'a' * 64, 'b' * 64, extra_prepared, extra_buffers=True)
    extra_case = directory / 'extra-buffers'
    extra_case.mkdir(mode=0o700)
    extra_native = extra_case / 'shards'
    extra_native.mkdir(mode=0o700)
    extra_result = supervise([str(binary), '--run', 'extra-buffers',
                              str(extra_prepared / 'inputs.bin'), str(extra_native)],
                             extra_case / 'supervision', timeout_s=5,
                             grace_s=.05, env={'PATH': '/usr/bin:/bin', 'LANG': 'C'})
    if extra_result['status'] != 'successful':
        raise ValueError('native extra-buffer policy test-double scenario failed')
    extra_index = finalize(extra_native, extra_prepared, 'c' * 64, extra_result)
    if extra_index['native_policy']['extra_buffers'] is not True:
        raise ValueError('native extra-buffer policy was not preserved in the index')
    results['extra-buffers'] = {'passed': True, 'supervisor_status': extra_result['status'],
                                'policy': 'extra_buffers=true'}
    selected_plan = synthetic_context_plan()
    selected_prepared = directory / 'prepared-selected-context'
    prepare(selected_plan, 'a' * 64, 'b' * 64, selected_prepared, extra_buffers=False)
    selected_case = directory / 'selected-context'
    selected_case.mkdir(mode=0o700)
    selected_native = selected_case / 'shards'
    selected_native.mkdir(mode=0o700)
    selected_result = supervise([str(binary), '--run', 'selected-context',
                                 str(selected_prepared / 'inputs.bin'), str(selected_native)],
                                selected_case / 'supervision', timeout_s=5,
                                grace_s=.05, env={'PATH': '/usr/bin:/bin', 'LANG': 'C'})
    if selected_result['status'] != 'successful':
        raise ValueError('selected context-row test-double scenario failed')
    selected_index = finalize(selected_native, selected_prepared, 'c' * 64, selected_result)
    selected_rows = list(rows(selected_native / 'index.json', load_index(selected_native / 'index.json')))
    expected_selected = [(s, p, selected_plan['streams'][s]['token_ids'][p])
                         for s in range(len(selected_plan['streams'])) for p in selected_plan['captured_input_positions']]
    if len(selected_rows) != len(expected_selected) or selected_index['shards'][0]['positions'] != [
            [selected_plan['streams'][s]['id'], p] for s, p, _ in expected_selected]:
        raise ValueError('selected context rows were not indexed in exact input-position order')
    for (label, values), (sequence, position, token) in zip(selected_rows, expected_selected):
        if label != (selected_plan['streams'][sequence]['id'], position) or list(values) != [
                sequence * 10000 + position * 100 + token + v for v in range(64)]:
            raise ValueError('selected context row content or identity mismatch')
    selected_trace = (selected_case / 'supervision/stdout.txt').read_text()
    expected_selected_trace = [f'DOUBLE selected {s} {p}' for s, p, _ in expected_selected]
    if [line for line in selected_trace.splitlines() if line.startswith('DOUBLE selected ')] != expected_selected_trace:
        raise ValueError('native selected-row mask differed from the frozen context positions')
    results['selected-context'] = {'passed': True, 'supervisor_status': selected_result['status'],
                                   'rows_retained': len(selected_rows)}

    nll_source = directory / 'nll-corpus'
    nll_source.mkdir(mode=0o700)
    write_private_json(nll_source / 'corpus.json', synthetic_nll_corpus())
    nll_prepared = directory / 'prepared-nll'
    nll_plan = prepare_nll(nll_source / 'corpus.json', 'calibration', 'a' * 64, 'b' * 64,
                           nll_prepared, vocabulary=64, capacity_tokens=256)
    nll_case = directory / 'nll-mode'
    nll_case.mkdir(mode=0o700)
    nll_native = nll_case / 'native'
    nll_native.mkdir(mode=0o700)
    nll_result = supervise([str(binary), '--run', 'nll', str(nll_prepared / 'inputs.bin'), str(nll_native)],
                           nll_case / 'supervision', timeout_s=5, grace_s=.05,
                           env={'PATH': '/usr/bin:/bin', 'LANG': 'C'})
    if nll_result['status'] != 'successful':
        raise ValueError('native NLL test-double scenario failed')
    nll_capture = finalize_nll(nll_native, nll_prepared, 'd' * 64, nll_result)
    observed_nll = [value for document in nll_capture['documents'] for value in document['nll']]
    expected_nll = []
    for sequence_index, document in enumerate(nll_plan['inputs']['documents']):
        for position, target in enumerate(document['token_ids'][:-1]):
            fake_logits = [float(sequence_index * 10000 + position * 100 + target + v)
                           for v in range(64)]
            expected_nll.append(target_nll(fake_logits, document['token_ids'][position + 1]))
    if len(observed_nll) != len(expected_nll) or any(
            abs(actual - expected) > 1e-12 for actual, expected in zip(observed_nll, expected_nll)):
        raise ValueError('native NLL reduction, target alignment or output order mismatch')
    trace = (nll_case / 'supervision/stdout.txt').read_text()
    if trace.count('DOUBLE decode ') != nll_plan['target_count'] or trace.count('DOUBLE clear ') != len(nll_plan['inputs']['documents']):
        raise ValueError('native NLL decode or per-document reset sequence mismatch')
    results['nll-mode'] = {'passed': True, 'supervisor_status': nll_result['status'],
                           'target_count': len(observed_nll)}
    write_private_json(directory / 'results.json', {
        'status': 'passed', 'scenarios': results, 'successful_rows': 33, 'selected_context_rows': len(expected_selected),
        'nll_targets': len(observed_nll),
        'scope': 'native adapter control flow with API double; NLL reduction checked against Python on synthetic logits; no weights, real backend, model numerical equivalence or real native cancellation validated'})
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    count = check(args.reference_source, args.binary, args.directory)
    print(f'Native capture API-double checks passed ({count} scenarios); no model inference.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        print('Native capture API-double check failed; inspect retained private evidence.', file=sys.stderr)
        sys.exit(1)
