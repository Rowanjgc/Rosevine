#!/usr/bin/env python3
"""Reduce the K2 (rv-e05-7n3.2) Q4_0 GEMV campaign; never exports or approves publication.

Usage: python3 scripts/research/analyze_k2.py PRIVATE_CAMPAIGN > PRIVATE_OUTPUT.json

Cells (single-shot `gpu_campaign` command sessions labelled `s<session>-<cell>`):
- `ab-p<P>`: `rosevine-inspect ab-attention --compare q4` stdout. Per pair, the step ratio is the vec
  block's median step over the ref block's (below 1 = vec faster). Pairs are pooled over sessions;
  95% intervals are percentile bootstraps of the median.
- `<ref|vec>-p<P>-events`: events-instrumented replays; device ms per decode token per span.
- `prefill-p<P>-<ref|vec>-<a|b>`: uninstrumented replays; vec/ref prefill-rate ratio per session.
- `kl-<base>-<ref|vec>`: `rosevine-inspect kld` summaries.
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
import analyze_k3  # noqa: E402

LABEL = re.compile(r'^s(\d+)-(.+)$')
# Spans whose Q4_0 matmuls the kernel choice changes (criterion 5).
GEMV_SPANS = ('attn', 'attn_out', 'mlp', 'gpu_experts', 'head', 'router')


def ab_pairs(text):
    """[(pair, ref_ms, vec_ms, ref_top1, vec_top1)] from `ab-attention --compare q4` stdout."""
    blocks = defaultdict(dict)
    for line in text.splitlines():
        row = json.loads(line)
        if row.get('kind') == 'config' and row.get('compare') != 'q4':
            raise ValueError('not a q4 comparison')
        if row.get('kind') == 'block':
            blocks[row['pair']][row['q4']] = row
    out = []
    for pair, arms in sorted(blocks.items()):
        if set(arms) != {'ref', 'vec'}:
            raise ValueError(f'pair {pair} incomplete')
        ref, vec = arms['ref'], arms['vec']
        out.append((pair, statistics.median(ref['decode_ns']) / 1e6, statistics.median(vec['decode_ns']) / 1e6,
                    ref['top1'], vec['top1']))
    if not out:
        raise ValueError('no complete pair')
    return out


def device_ms(err_text):
    """Device ms per decode token for each span (summed over groups) from an events replay."""
    decode = [p for p in analyze_k1.parse_err(err_text) if p['phase'] == 'decode'][-1]
    spans = defaultdict(float)
    for key, v in analyze_k1.per_token(decode).items():
        span, group = key.split('/')
        if group not in ('local', 'global'):
            spans[span] += v['dev_ms']
    return dict(spans)


def reduce(campaign):
    failed, ab, events, prefill, kl = {}, defaultdict(list), defaultdict(dict), defaultdict(dict), {}
    for case in sorted(Path(campaign).iterdir()):
        m = LABEL.match(case.name)
        if not case.is_dir() or not m:
            continue
        session, cell = int(m.group(1)), m.group(2)
        if not analyze_k3.successful(case):
            failed[case.name] = 'not successful or no run summary'
            continue
        out = (case / 'out.txt').read_text()
        if cell.startswith('ab-'):
            ab[cell] += [(session, *p) for p in ab_pairs(out)]
        elif cell.endswith('-events'):
            kernel, size, _ = cell.split('-')
            events[f's{session}-{size}'][kernel] = device_ms((case / 'err.txt').read_text())
        elif cell.startswith('prefill-'):
            _, size, kernel, _ = cell.split('-')
            config, reps = analyze_k1.parse_out(out)
            prefill[(session, size)].setdefault(kernel, []).append(config['n_prompt'] / (reps[-1]['prefill_ns'] / 1e9))
        elif cell.startswith('kl-'):
            line = next((x for x in out.splitlines() if x.startswith('kld rows=')), None)
            if line is None:
                raise ValueError(f'{case.name}: no kld summary')
            k = analyze_k3.KLD.match(line)
            kl[cell] = {'rows': int(k.group(1)), 'mean_kl': float(k.group(2)), 'same_top': float(k.group(3)),
                        'ppl': float(k.group(4)), 'ppl_base': float(k.group(5))}
    report = {'failed': failed, 'ab': {}, 'events_dev_ms': {}, 'prefill': {}, 'kl': kl}
    for cell, rows in ab.items():
        same = sum(a == b for *_, rt, vt in rows for a, b in zip(rt, vt))
        steps = sum(len(rt) for *_, rt, _ in rows)
        report['ab'][cell] = {
            'pairs': len(rows), 'sessions': sorted({r[0] for r in rows}),
            'ref_ms': statistics.median(r for _, _, r, _, _, _ in rows),
            'vec_ms': statistics.median(v for _, _, _, v, _, _ in rows),
            'diff_ms': analyze_k3.bootstrap_median([r - v for _, _, r, v, _, _ in rows]),
            'ratio': analyze_k3.bootstrap_median([v / r for _, _, r, v, _, _ in rows]),
            'top1_agreement': same / steps}
    for key, kernels in sorted(events.items()):
        if set(kernels) == {'ref', 'vec'}:
            report['events_dev_ms'][key] = {span: {'ref': kernels['ref'].get(span, 0.0), 'vec': kernels['vec'].get(span, 0.0)}
                                            for span in GEMV_SPANS}
    for (session, size), kernels in sorted(prefill.items()):
        if set(kernels) == {'ref', 'vec'}:
            report['prefill'][f's{session}-{size}'] = {
                'ref_tok_s': statistics.mean(kernels['ref']), 'vec_tok_s': statistics.mean(kernels['vec']),
                'vec_over_ref': statistics.mean(kernels['vec']) / statistics.mean(kernels['ref'])}
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    json.dump(reduce(sys.argv[1]), sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
