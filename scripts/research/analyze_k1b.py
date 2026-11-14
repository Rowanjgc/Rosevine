#!/usr/bin/env python3
"""Reduce the K1b (rv-e05-7n3.7) verification-width campaign; never exports or approves publication.

Usage: python3 scripts/research/analyze_k1b.py PRIVATE_CAMPAIGN > PRIVATE_OUTPUT.json

Cells are `rosevine-inspect verify-sweep` command sessions labelled `s<session>-d<depth>-<off|sync>`.
Each block evaluates verification steps of `width` rows keeping `accept + 1`. For every
(depth, width, accept) shape it reports the median uninstrumented step and the cost per committed
token, sync-mode spans per verification (with the global/local `attn_core` split), span closure,
instrumentation overhead against the uninstrumented cell, and top-1 identity between modes.
No machine probing or model execution occurs.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_k1  # noqa: E402

LABEL = re.compile(r'^s(\d+)-(d\d+)-(off|sync)$')


def blocks(text):
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    if not rows or rows[0].get('schema') != 'rosevine-verify-sweep-v1':
        raise ValueError('not a verify-sweep output')
    return [r for r in rows if r.get('kind') == 'block']


def reduce(campaign):
    report = {'failed': {}, 'shapes': {}, 'top1_mismatches': 0, 'top1_compared': 0}
    steps = defaultdict(list)          # (depth, mode, w, a) -> per-block median step ms
    spans = defaultdict(list)          # (depth, w, a) -> per-token span dicts (sync)
    closure = defaultdict(list)
    top1 = {}
    for case in sorted(Path(campaign).iterdir()):
        m = LABEL.match(case.name)
        if not case.is_dir() or not m:
            continue
        summary = case / 'run-summary.json'
        if not summary.exists() or json.loads(summary.read_text())['status'] != 'successful':
            report['failed'][case.name] = 'not successful or no run summary'
            continue
        session, depth, mode = int(m.group(1)), m.group(2), m.group(3)
        bl = blocks((case / 'out.txt').read_text())
        for b in bl:
            key = (depth, b['width'], b['accept'])
            steps[(depth, mode, b['width'], b['accept'])].append(statistics.median(b['step_ns']) / 1e6)
            top1[(session, mode, key, b['rep'])] = b['top1']
        if mode == 'sync':
            phases = [p for p in analyze_k1.parse_err((case / 'err.txt').read_text()) if p['phase'] != 'decode']
            if len(phases) != len(bl):
                raise ValueError(f'{case.name}: {len(phases)} phases for {len(bl)} blocks')
            for p, b in zip(phases, bl):
                if p['phase'] != f'w{b["width"]}a{b["accept"]}':
                    raise ValueError(f'{case.name}: phase {p["phase"]} out of order')
                key = (depth, b['width'], b['accept'])
                spans[key].append({k: v['host_ms'] for k, v in analyze_k1.per_token(p).items()})
                closure[key].append(p['closure'])
    for (s, mode, key, rep), t in top1.items():
        if mode == 'sync' and (s, 'off', key, rep) in top1:
            report['top1_compared'] += len(t)
            report['top1_mismatches'] += sum(a != b for a, b in zip(t, top1[(s, 'off', key, rep)]))
    for (depth, mode, w, a), meds in sorted(steps.items()):
        if mode != 'off':
            continue
        key = (depth, w, a)
        off = statistics.median(meds)
        entry = {'blocks': len(meds), 'step_ms': off, 'ms_per_committed_token': off / (a + 1)}
        sync = steps.get((depth, 'sync', w, a))
        if sync:
            entry['sync_step_ms'] = statistics.median(sync)
            entry['sync_overhead'] = statistics.median(sync) / off - 1
        if spans.get(key):
            names = sorted({n for d in spans[key] for n in d})
            entry['spans_ms_per_verification'] = {n: statistics.median(d.get(n, 0.0) for d in spans[key]) for n in names}
            entry['closure_min'] = min(closure[key])
        report['shapes'][f'{depth}-w{w}a{a}'] = entry
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    json.dump(reduce(sys.argv[1]), sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
