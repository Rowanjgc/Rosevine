#!/usr/bin/env python3
"""Compile the pinned MoE capture adapter and exercise its scheduler callback.

This optional check loads no model and performs no inference. It requires an
existing compiler, pinned clean source checkout and selected installed libraries.
All binaries and reports must be new private artifacts.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from benchmark_core import supervise, write_private_json
from intermediate_capture import CAPTURE_LIMIT, read_records, private_dir
from logit_capture import private_read

REVISION = 'cd959bae99ff4bed22bef2604f45241c09bbb8c8'
ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ('libllama.so.0.9.0', 'libggml-cpu.so.0.29.0',
             'libggml-base.so.0.29.0', 'libggml.so.0.29.0')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exact_values(path, records):
    raw = private_read(path, CAPTURE_LIMIT)
    result = {}
    for record in records:
        start = record['data_offset']
        end = start + record['nbytes']
        data = raw[start:end]
        code = 'f' if record['dtype'] == 'f32' else 'i'
        result[record['name']] = tuple(x[0] for x in struct.iter_unpack('<' + code, data))
    return result


def check(reference_source, library_dir, adapter_binary, smoke_binary, directory):
    reference_source = Path(reference_source).resolve()
    library_dir = Path(library_dir).resolve()
    adapter_binary, smoke_binary = Path(adapter_binary).absolute(), Path(smoke_binary).absolute()
    directory = Path(directory)
    for path in (adapter_binary, smoke_binary):
        if path.exists() or path.is_symlink():
            raise ValueError('new ignored binary path required')
    tools_root = (ROOT / '.tools').resolve()
    for path in (adapter_binary.parent, smoke_binary.parent):
        resolved = path.resolve()
        if not path.is_dir() or (resolved != tools_root and tools_root not in resolved.parents):
            raise ValueError('binaries must be placed under the ignored project .tools directory')
    revision = subprocess.check_output(['git', '-C', str(reference_source), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
    status = subprocess.check_output(['git', '-C', str(reference_source), 'status', '--porcelain', '--untracked-files=all'], timeout=10, text=True)
    if revision != REVISION or status:
        raise ValueError('clean pinned reference checkout required')
    if any(not (library_dir / name).is_file() for name in LIBRARIES):
        raise ValueError('all selected llama/ggml libraries must exist')
    compiler = shutil.which('g++')
    if not compiler:
        raise ValueError('existing C++ compiler required; no install attempted')
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    directory = private_dir(directory.resolve())
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'}
    common = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
              '-fno-fast-math', '-ffp-contract=off',
              '-I' + str(reference_source / 'include'),
              '-I' + str(reference_source / 'ggml/include')]

    def build(source, binary, name):
        command = common + [str(ROOT / source), '-L' + str(library_dir),
                            '-Wl,-rpath,' + str(library_dir), '-lllama',
                            '-lggml-cpu', '-lggml-base', '-lggml', '-o', str(binary)]
        result = supervise(command, directory / name, timeout_s=60, env=env)
        if result['status'] != 'successful' or result['returncode'] != 0:
            raise ValueError(f'{name} failed; private compiler diagnostics retained')
        binary.chmod(0o700)
        return command

    adapter_command = build('research/capture_moe_intermediates.cpp', adapter_binary, 'build-adapter')
    smoke_command = build('research/check_moe_capture_callback.cpp', smoke_binary, 'build-smoke')
    output_dir = directory / 'smoke-output'
    output_dir.mkdir(mode=0o700)
    native = output_dir / 'nodes.bin'
    result = supervise([str(smoke_binary), str(native)], directory / 'smoke-run',
                       timeout_s=10, env={**env, 'GGML_DISABLE_VULKAN': '1'})
    if result['status'] != 'successful' or result['returncode'] != 0 or result['workload_cleanup']['pending']:
        raise ValueError('scheduler callback smoke execution failed; private evidence retained')
    records, total = read_records(native)
    values = exact_values(native, records)
    expected = {'ffn_moe_logits-0': (-2.0, 0.0, 1.0, 4.0),
                'ffn_moe_topk-0': (1, 3),
                'ffn_moe_weights-0': (0.25, 0.75)}
    if values != expected or total != 252 or len(records) != 3:
        raise ValueError('scheduler callback output did not match the synthetic oracle')
    write_private_json(directory / 'result.json', {
        'schema': 'rosevine-moe-capture-adapter-check-v1',
        'status': 'passed', 'scope': 'pinned scheduler callback with synthetic CPU graph; no model or inference',
        'reference_revision': revision,
        'source_sha256': {name: sha(ROOT / name) for name in
                          ('research/capture_moe_intermediates.cpp', 'research/check_moe_capture_callback.cpp')},
        'binary_sha256': {str(adapter_binary): sha(adapter_binary), str(smoke_binary): sha(smoke_binary)},
        'library_sha256': {str(library_dir / name): sha(library_dir / name) for name in LIBRARIES},
        'commands': {'adapter': adapter_command, 'smoke': smoke_command},
        'records': records, 'values': values, 'bytes': total,
        'covered': ['scheduler ask/observe callback lifecycle', 'CPU host-backed node read',
                    'exact emitted values', 'position/name filtering', 'nonfinite rejection']})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--adapter-binary', type=Path, required=True)
    parser.add_argument('--smoke-binary', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    records = check(args.reference_source, args.library_dir, args.adapter_binary,
                    args.smoke_binary, args.directory)
    print(f'Pinned MoE capture callback check passed ({len(records)} synthetic records); no model loaded.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        print('MoE capture callback check failed; inspect retained private evidence.', file=sys.stderr)
        sys.exit(1)
