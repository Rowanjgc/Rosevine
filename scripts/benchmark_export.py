#!/usr/bin/env python3
"""Validate an allowlisted measurement record; never redact arbitrary raw logs."""
import argparse
import json
import math
import re
import sys
from guardrails import inspect

TEXT = {'hardware_class', 'os_version', 'software_revision', 'compiler_flags',
        'model', 'quantization', 'methodology', 'status', 'timing_unit'}
NUMBERS = {'vram_gib', 'warmup_runs', 'measured_runs', 'median', 'p95'}
FIELDS = TEXT | NUMBERS


def validate(record):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError('Record must contain exactly the documented benchmark fields.')
    if record['status'] not in {'successful', 'failed', 'incomplete'} or record['timing_unit'] not in {'ms', 's', 'tokens/s'}:
        raise ValueError('Invalid measurement status/unit.')
    for key in TEXT:
        if not isinstance(record[key], str) or not record[key].strip() or len(record[key]) > 500:
            raise ValueError('Invalid provenance text.')
    if not re.fullmatch(r'[0-9a-f]{40}', record['software_revision']):
        raise ValueError('Use the complete software commit ID.')
    for key in NUMBERS:
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid numeric measurement.')
    for key in ['warmup_runs', 'measured_runs']:
        if type(record[key]) is not int:
            raise ValueError('Run counts must be integers.')
    if record['measured_runs'] < 1 or record['vram_gib'] <= 0:
        raise ValueError('Measurements and hardware capacity must be positive.')
    inspect('benchmark-summary', json.dumps(record).encode())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', help='Private structured candidate file; never an arbitrary terminal dump')
    parser.add_argument('--reviewed', action='store_true', help='Human confirmed every field is public and accurate')
    args = parser.parse_args()
    if not args.reviewed:
        parser.error('Human review required: inspect every field, then use --reviewed.')
    with open(args.record) as stream:
        record = validate(json.load(stream))
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError):
        print('Benchmark validation failed; inspect the private candidate locally. No raw data emitted.', file=sys.stderr)
        sys.exit(1)
