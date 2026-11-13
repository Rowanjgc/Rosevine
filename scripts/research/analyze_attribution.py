#!/usr/bin/env python3
"""Reduce R3a.2 private captures; never exports or approves benchmark publication.

Usage: python3 scripts/research/analyze_attribution.py PRIVATE_CAMPAIGN
Redirect stdout to a private file. No machine probing or model execution occurs.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_core import percentile  # noqa: E402


def distribution(values):
    if not values:
        raise ValueError('empty sample set')
    return {'n': len(values), 'median': statistics.median(values),
            'p95': percentile(values, .95), 'mean': statistics.mean(values)}


def reduce_cpu(path):
    groups = defaultdict(list)
    for row in csv.DictReader(path.open()):
        key = '/'.join(row[k] for k in ('kind', 'threads', 'condition'))
        groups[key].append(row)
    return {key: {field: distribution([float(row[field]) for row in rows])
                  for field in ('gate_ms', 'activation_ms', 'down_ms', 'combine_ms', 'total_ms')}
            for key, rows in groups.items()}


def reduce_upload(path):
    groups = defaultdict(list)
    for row in csv.DictReader(path.open()):
        key = '/'.join(row[k] for k in ('staging', 'api', 'idle_us', 'pre_marker'))
        groups[key].append(row)
    return {key: {field: distribution([float(row[field]) for row in rows])
                  if rows[0][field] != 'null' else None
                  for field in ('marker_host_us', 'enqueue_us', 'wait_us', 'total_us', 'event_us')}
            for key, rows in groups.items()}


def reduce_hip(path):
    result, current, spans = [], None, defaultdict(list)
    first_n = None
    for line in path.read_text().splitlines():
        if line.startswith('configuration N='):
            first_n = int(line.split()[1].split('=')[1])
        elif line.startswith('round='):
            match = re.fullmatch(r'round=(\d+) mode=(.*?) wall_s=([\d.]+)', line)
            if match is None:
                raise ValueError('invalid round record')
            current = {'round': int(match[1]), 'mode': match[2], 'wall_s': float(match[3]),
                       'token_ms': [], 'ids': [], 'spans': {}}
            result.append(current)
        elif line.startswith('token,'):
            _, step, ns, token = line.split(',')
            if int(step) != len(current['ids']):
                raise ValueError('non-contiguous token samples')
            current['token_ms'].append(int(ns) / 1e6)
            current['ids'].append(int(token))
        elif line.startswith('span,'):
            _, step, layer, name, count, host, dev, launches, h2d, h2db, d2h, d2hb, d2d = line.split(',')
            key = (current['round'], int(layer), name)
            values = [int(v) for v in (count, host, dev, launches, h2d, h2db, d2h, d2hb, d2d)]
            old = spans[key][-1][1] if spans[key] else [0] * len(values)
            delta = [v-o for v, o in zip(values, old)]
            if any(v < 0 for v in delta):
                raise ValueError('counter discontinuity')
            spans[key].append((int(step), values, delta))
        elif line.startswith('loop_closure='):
            current['loop_closure'] = float(line.split('=')[1])
    if not result:
        raise ValueError('missing HIP samples')
    reference_ids = result[0]['ids']
    for row in result:
        if len(row['ids']) != 32 or row['ids'] != reference_ids:
            raise ValueError('incomplete run or greedy mismatch')
        row['token_samples_ms'] = row['token_ms'][:]
        row['token_ms'] = distribution(row['token_ms'])
        per_token = defaultdict(lambda: defaultdict(float))
        counters = [0] * 6
        for (round_id, layer, name), records in spans.items():
            if round_id != row['round']:
                continue
            for step, _, delta in records:
                per_token[name][step] += delta[1] / 1e6
                for i, value in enumerate(delta[3:]):
                    counters[i] += value
            row['spans'][f'{layer}/{name}'] = {
                'host_ms': distribution([r[2][1]/1e6 for r in records]),
                'event_ms': distribution([r[2][2]/1e6 for r in records])}
        row['stage_ms'] = {name: distribution(list(samples.values())) for name, samples in per_token.items()}
        row['calls_per_token'] = dict(zip(('launches', 'h2d', 'h2d_bytes', 'd2h', 'd2h_bytes', 'd2d'),
                                          [n / 32 for n in counters])) if spans and row['mode'] != 'None' else None
        if first_n is not None and row['calls_per_token'] is not None:
            expected = dict(launches=1319-19*first_n, h2d=121-3*first_n,
                            h2d_bytes=4+first_n*11264+(30-first_n)*128,
                            d2h=31+first_n, d2h_bytes=1048576+30*512+first_n*11264, d2d=5)
            if row['calls_per_token'] != expected:
                raise ValueError('calls disagree with source hand count')
        span_total = sum(d['mean'] for d in row['stage_ms'].values())
        row['stage_fraction'] = {name: d['mean'] / span_total for name, d in row['stage_ms'].items()}
        del row['ids']
    return result


def reduce_quality(directory):
    session = json.loads((directory / 'session.json').read_text())
    snapshots = [json.loads(line)['snapshot'] for line in
                 (directory / 'supervision/telemetry.jsonl').read_text().splitlines()]
    if not snapshots:
        raise ValueError('missing telemetry')
    temperatures = {sensor['label']: max(s['temperatures_c'][sensor['label']] for s in snapshots)
                    for sensor in session['sensors']}
    flagged = {sensor['label']: sum(s['temperatures_c'][sensor['label']] >= sensor['flag_c']
                                   for s in snapshots)
               for sensor in session['sensors'] if 'flag_c' in sensor}
    deltas = [s['paging']['delta_bytes'] for s in snapshots]
    contamination = directory / 'contamination.json'
    excluded = contamination.exists() and json.loads(contamination.read_text()).get('excluded', False)
    return {'measurement_excluded': excluded,
            'max_temperatures_c': temperatures, 'flagged_sample_counts': flagged,
            'thermal_flagged': any(flagged.values()),
            'min_mem_available_gib': min(s['memory']['MemAvailable'] for s in snapshots) / 2**30,
            'peak_vram_gib': max(s['vram_used_bytes'] for s in snapshots) / 2**30,
            'observed_paging_mib': sum(v for v in deltas if v is not None) / 2**20,
            'paging_baseline_samples': sum(v is None for v in deltas),
            'paging_warning_samples': sum(bool(s['paging']['warning']) for s in snapshots),
            'foreign_growth_observed': any(s['foreign_server_growth_bytes'] for s in snapshots)}


def main(root):
    result = {'quality': {}}
    for directory in sorted(Path(root).glob('s*')):
        summary = directory / 'run-summary.json'
        if not summary.is_file():
            continue
        status = json.loads(summary.read_text())
        result['quality'][directory.name] = reduce_quality(directory)
        if status['status'] != 'successful' or status['cleanup_pending']:
            result[directory.name] = {'status': status}
            continue
        path = directory / 'command.log'
        if '-hip-' in directory.name:
            reduced = reduce_hip(path)
        elif '-cpu' in directory.name:
            reduced = reduce_cpu(path)
        elif '-upload' in directory.name:
            reduced = reduce_upload(path)
        else:
            continue
        result[directory.name] = reduced
    pooled = defaultdict(list)
    for name, rows in result.items():
        match = re.search(r'-hip-(\d+)-(\d+)-(\d+)$', name)
        if match is None or not isinstance(rows, list) or result['quality'][name]['measurement_excluded']:
            continue
        for row in rows:
            if row['round'] > 0:
                pooled['/'.join((*match.groups(), row['mode']))].extend(row['token_samples_ms'])
    result['pooled_token_ms'] = {key: distribution(values) for key, values in pooled.items()}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main(sys.argv[1])
