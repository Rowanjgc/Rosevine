#!/usr/bin/env python3
"""Reduce the S1 (rv-e07-ktl.1) pinned-MTP reference trials; never exports or approves publication.

Usage: python3 scripts/research/analyze_s1.py PRIVATE_CAMPAIGN > PRIVATE_OUTPUT.json

Cells are single-shot `gpu_campaign` server sessions of the pinned llama.cpp server with trace logging
(`LLAMA_TRACE=1`, verbosity 4): `nat-mtp` (natural greedy MTP), `nat-off` (target-only control) and
`synth-<k>` (synthetic acceptance forcing k accepted drafts per verification). Per request it reports
verification count, the accepted-count histogram, per-position acceptance, draft lengths, decode time
per verification and, for `nat-mtp` against `nat-off`, the first differing output token. Synthetic
acceptance is cost evidence only: its output tokens are not generation evidence.
No machine probing or model execution occurs.
"""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ACCEPTED = re.compile(r'\| task (\d+) \| accepted\s+(\d+)/\s*(\d+) draft tokens')
STATS = re.compile(r'statistics\s+draft-mtp: #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+).*?'
                   r'#gen tokens =\s*(\d+), #acc tokens =\s*(\d+).*?dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms')
TASK = re.compile(r'\| task (\d+) \|')


def parse_log(text):
    """({task: [(accepted, drafted)]}, [cumulative draft-mtp statistics rows], task order)."""
    steps, stats, order = defaultdict(list), [], []
    for line in text.splitlines():
        m = TASK.search(line)
        if m and int(m.group(1)) not in order and 'launch_slot' in line:
            order.append(int(m.group(1)))
        m = ACCEPTED.search(line)
        if m:
            steps[int(m.group(1))].append((int(m.group(2)), int(m.group(3))))
            continue
        m = STATS.search(line)
        if m:
            b, g, a, gen, acc = (int(m.group(i)) for i in range(1, 6))
            stats.append({'calls_begin': b, 'calls_draft': g, 'calls_accept': a, 'gen_tokens': gen, 'acc_tokens': acc,
                          'dur_begin_ms': float(m.group(6)), 'dur_draft_ms': float(m.group(7)),
                          'dur_accept_ms': float(m.group(8))})
    return steps, stats, order


def load_requests(case):
    rows = [json.loads(x) for x in (case / 'requests.jsonl').read_text().splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r['client_sent_ns'])


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(rows, steps):
    hist = Counter(acc for acc, _ in steps)
    drafted = Counter(n for _, n in steps)
    n = len(steps)
    max_draft = max(drafted) if drafted else 0
    per_pos = [sum(1 for acc, _ in steps if acc > i) / n for i in range(max_draft)] if n else []
    t = rows['timings']
    return {'verifications': n, 'accepted_hist': {str(k): hist[k] for k in sorted(hist)},
            'draft_len_hist': {str(k): drafted[k] for k in sorted(drafted)}, 'accept_rate_per_pos': per_pos,
            'committed_per_verification': (sum(acc for acc, _ in steps) + n) / n if n else None,
            'predicted_n': t['predicted_n'], 'predicted_ms': t['predicted_ms'], 'prompt_n': t['prompt_n'],
            'decode_tok_s': t['predicted_per_second'],
            'ms_per_verification': t['predicted_ms'] / n if n else None,
            'server_draft_n': t.get('draft_n'), 'server_draft_accepted': t.get('draft_n_accepted')}


def reduce(campaign):
    report, outputs = {'cells': {}, 'failed': {}}, {}
    for case in sorted(Path(campaign).iterdir()):
        if not case.is_dir():
            continue
        summary = case / 'run-summary.json'
        if not summary.exists() or json.loads(summary.read_text())['status'] != 'successful':
            report['failed'][case.name] = 'not successful or no run summary'
            continue
        rows = load_requests(case)
        steps, stats, order = parse_log((case / 'server.log').read_text(errors='replace'))
        if len(order) != len(rows):
            raise ValueError(f'{case.name}: {len(order)} tasks in the log, {len(rows)} requests')
        cell = {'requests': {}, 'draft_stats_final': stats[-1] if stats else None}
        for task, row in zip(order, rows):
            if not row.get('ok'):
                raise ValueError(f'{case.name}: request {row["id"]} failed')
            cell['requests'][row['id']] = summarize(row, steps.get(task, []))
            outputs[(case.name, row['id'])] = row['output_token_ids']
        measured = [v for k, v in cell['requests'].items() if not k.startswith('warm')]
        if measured and measured[0]['verifications']:
            for size in ('1900', '29000'):
                group = [v for k, v in cell['requests'].items() if k.endswith(f'-{size}-r0') and v['verifications']]
                if group:
                    cell[f'pooled_{size}'] = {
                        'verifications': sum(v['verifications'] for v in group),
                        'committed_per_verification': sum(v['committed_per_verification'] * v['verifications'] for v in group)
                        / sum(v['verifications'] for v in group),
                        'ms_per_verification_median': statistics.median(v['ms_per_verification'] for v in group),
                        'decode_tok_s_median': statistics.median(v['decode_tok_s'] for v in group)}
        report['cells'][case.name] = cell
    if 'nat-mtp' in report['cells'] and 'nat-off' in report['cells']:
        report['mtp_vs_off_first_difference'] = {
            rid: first_difference(outputs[('nat-mtp', rid)], outputs[('nat-off', rid)])
            for rid in report['cells']['nat-mtp']['requests'] if ('nat-off', rid) in outputs}
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    json.dump(reduce(sys.argv[1]), sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
