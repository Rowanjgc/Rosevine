#!/usr/bin/env python3
"""Reduce the K3 (rv-e05-7n3.3) split-K attention campaign; never exports or approves publication.

Usage: python3 scripts/research/analyze_k3.py PRIVATE_CAMPAIGN > PRIVATE_OUTPUT.json

Cells (single-shot `gpu_campaign` command sessions labelled `s<session>-<cell>`):
- `ab-p<P>`: `rosevine-inspect ab-attention` stdout. Per pair, the paired difference is the
  reference block's median step minus the split block's (positive = split faster). Pairs are pooled
  over sessions; the 95% interval is a percentile bootstrap of the median difference.
- `split-p<P>-sync`: a sync-instrumented split replay; reports `attn_core` per decode token.
- `prefill-p<P>-<ref|split>-<a|b>`: uninstrumented replays; split/ref prefill-rate ratio per session.
- `kl-<base>-<ref|split>`: `rosevine-inspect kld` summaries.
No machine probing or model execution occurs.
"""
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_k1  # noqa: E402

LABEL = re.compile(r'^s(\d+)-(.+)$')
KLD = re.compile(r'^kld rows=(\d+) mean=([\d.]+) .*same_top=([\d.]+) ppl=([\d.]+) ppl_base=([\d.]+)')


def bootstrap_median(values, *, seed=20261113, resamples=10_000):
    """Median and 95% percentile-bootstrap interval of the median."""
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples))
    return {'n': len(values), 'median': statistics.median(values),
            'ci95': [meds[int(0.025 * resamples)], meds[int(0.975 * resamples) - 1]]}


def ab_pairs(text):
    """[(pair, ref_ms, split_ms, ref_top1, split_top1)] from ab-attention stdout."""
    blocks = defaultdict(dict)
    for line in text.splitlines():
        row = json.loads(line)
        if row.get('kind') == 'block':
            blocks[row['pair']][row['attention']] = row
    out = []
    for pair, kernels in sorted(blocks.items()):
        if set(kernels) != {'ref', 'split'}:
            raise ValueError(f'pair {pair} incomplete')
        ref, split = kernels['ref'], kernels['split']
        out.append((pair, statistics.median(ref['decode_ns']) / 1e6, statistics.median(split['decode_ns']) / 1e6,
                    ref['top1'], split['top1']))
    if not out:
        raise ValueError('no complete pair')
    return out


def successful(case):
    path = case / 'run-summary.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'successful'


def reduce(campaign):
    failed, ab, attn, prefill, kl = {}, defaultdict(list), {}, defaultdict(dict), {}
    for case in sorted(Path(campaign).iterdir()):
        m = LABEL.match(case.name)
        if not case.is_dir() or not m:
            continue
        session, cell = int(m.group(1)), m.group(2)
        if not successful(case):
            failed[case.name] = 'not successful or no run summary'
            continue
        out = (case / 'out.txt').read_text()
        if cell.startswith('ab-'):
            ab[cell] += [(session, *p) for p in ab_pairs(out)]
        elif cell.startswith('split-') and cell.endswith('-sync'):
            phases = analyze_k1.parse_err((case / 'err.txt').read_text())
            decode = [p for p in phases if p['phase'] == 'decode'][-1]
            pt = analyze_k1.per_token(decode)
            attn[f's{session}-{cell}'] = {
                'attn_core_ms': sum(v['host_ms'] for k, v in pt.items() if k in ('attn_core/host', 'attn_core/gpu')),
                'attn_core_global_ms': pt['attn_core/global']['host_ms'],
                'closure': decode['closure']}
        elif cell.startswith('prefill-'):
            _, size, kernel, rep = cell.split('-')
            config, reps = analyze_k1.parse_out(out)
            prefill[(session, size)].setdefault(kernel, []).append(config['n_prompt'] / (reps[-1]['prefill_ns'] / 1e9))
        elif cell.startswith('kl-'):
            line = next((x for x in out.splitlines() if x.startswith('kld rows=')), None)
            if line is None:
                raise ValueError(f'{case.name}: no kld summary')
            k = KLD.match(line)
            kl[cell] = {'rows': int(k.group(1)), 'mean_kl': float(k.group(2)), 'same_top': float(k.group(3)),
                        'ppl': float(k.group(4)), 'ppl_base': float(k.group(5))}
    report = {'failed': failed, 'ab': {}, 'attn_core_sync': attn, 'prefill': {}, 'kl': kl}
    for cell, rows in ab.items():
        diffs = [r - s for _, _, r, s, _, _ in rows]
        same = sum(a == b for *_, rt, st in rows for a, b in zip(rt, st))
        steps = sum(len(rt) for *_, rt, _ in rows)
        report['ab'][cell] = {
            'pairs': len(rows), 'sessions': sorted({r[0] for r in rows}),
            'ref_ms': statistics.median(r for _, _, r, _, _, _ in rows),
            'split_ms': statistics.median(s for _, _, _, s, _, _ in rows),
            'diff_ms': bootstrap_median(diffs),
            'ratio': bootstrap_median([s / r for _, _, r, s, _, _ in rows]),
            'top1_agreement': same / steps}
    for (session, size), kernels in sorted(prefill.items()):
        if set(kernels) == {'ref', 'split'}:
            report['prefill'][f's{session}-{size}'] = {
                'ref_tok_s': statistics.mean(kernels['ref']), 'split_tok_s': statistics.mean(kernels['split']),
                'split_over_ref': statistics.mean(kernels['split']) / statistics.mean(kernels['ref'])}
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    json.dump(reduce(sys.argv[1]), sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
