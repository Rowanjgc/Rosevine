#!/usr/bin/env python3
"""Create private capacity-only controls from frozen rendered workloads.

This reuses exact prompt/render/token bytes and changes only execution capacity.
It neither invokes native tools nor starts inference. Output manifests prove input
identity across capacities, not runtime allocation or performance behavior.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from benchmark_core import VERSION, rebind_workload_capacity, write_private_json


def read_private_json(path, limit=8 * 1024 * 1024):
    path = Path(path)
    parent = path.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700
            or parent.st_uid != os.getuid()):
        raise ValueError('owned private source directory required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > limit):
            raise ValueError('bounded owned private source file required')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('private manifest byte limit exceeded')
    return json.loads(raw)


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def build(source_directory, output_directory, *, capacities=None):
    source_directory, output_directory = Path(source_directory), Path(output_directory)
    capacities = capacities or {1900: 4096, 29000: 32768}
    if set(capacities) != {1900, 29000} or any(type(v) is not int or v <= 0 for v in capacities.values()):
        raise ValueError('explicit capacities required for both frozen occupancies')
    source_info = source_directory.lstat()
    if (not stat.S_ISDIR(source_info.st_mode) or stat.S_IMODE(source_info.st_mode) != 0o700
            or source_info.st_uid != os.getuid()):
        raise ValueError('owned private source directory required')
    output_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(output_directory, 0o700)
    write_private_json(output_directory / 'started.json', {'status': 'incomplete'})
    entries, seen = [], set()
    try:
        for source in sorted(source_directory.glob('*.json')):
            record = read_private_json(source)
            if not isinstance(record, dict) or record.get('version') != VERSION:
                continue
            occupied = record.get('occupied_input_tokens')
            if occupied not in capacities or record.get('capacity_tokens') != 65536:
                raise ValueError('unexpected source occupancy or baseline capacity')
            rebound = rebind_workload_capacity(record, capacities[occupied])
            key = (record['fixture_id'], occupied)
            if key in seen:
                raise ValueError('duplicate fixture/occupancy source manifest')
            seen.add(key)
            name = '%s-%d-cap%d.json' % (record['fixture_id'], occupied, capacities[occupied])
            write_private_json(output_directory / name, rebound)
            entries.append({'source_sha256': file_sha256(source),
                            'control_sha256': file_sha256(output_directory / name),
                            'fixture_id': record['fixture_id'], 'occupied_input_tokens': occupied,
                            'baseline_capacity_tokens': record['capacity_tokens'],
                            'control_capacity_tokens': rebound['capacity_tokens'],
                            'rendered_sha256': record['rendered_sha256'],
                            'tokens_sha256': record['tokens_sha256'],
                            'changed_fields': ['capacity_tokens', 'manifest_sha256']})
        if len(entries) != 16:
            raise ValueError('expected eight fixtures at each frozen occupancy')
        result = {'schema': 'rosevine-capacity-controls-v1', 'status': 'successful',
                  'source_count': len(entries), 'capacity_controls': entries,
                  'method': 'verified source hash, prompt/render/token identity, and reserve; changed capacity and manifest hash only',
                  'scope': 'manifest identity controls only; no tokenizer replay, context allocation, inference, or performance measurement'}
        write_private_json(output_directory / 'result.json', result)
        return result
    except Exception as error:
        write_private_json(output_directory / 'result.json', {
            'schema': 'rosevine-capacity-controls-v1', 'status': 'failed',
            'failure_category': type(error).__name__, 'completed_controls': len(entries)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_directory, args.output_directory)
    print('Created %d private exact-input capacity controls; no inference.' % len(result['capacity_controls']))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        print('Capacity-control construction failed; private partial evidence retained.', file=sys.stderr)
        sys.exit(1)
