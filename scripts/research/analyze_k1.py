#!/usr/bin/env python3
"""Reduce the K1 (rv-e05-7n3.1) attribution campaign; never exports or approves publication.

Usage: python3 scripts/research/analyze_k1.py PRIVATE_CAMPAIGN > PRIVATE_OUTPUT.json

Reads single-shot `gpu_campaign` command sessions labelled `s<session>-<cell>`. Each holds the
`rosevine-inspect replay` stdout (`out.txt`) and the instrumentation summaries on stderr
(`err.txt`). The last repetition is the measured one whenever a cell ran more than one.

Per cell and session it reports, for each phase (prefill, decode):
- wall-time closure of the spans and of the harness's own loop timer;
- per-token host milliseconds for every (span, group), and device milliseconds in events mode;
- the depth-dependent attention kernel (`attn_core`) split into local and global layers;
- uninstrumented rates and the overhead of each mode against them;
- whether instrumentation changed any top-1 token (it must not).
No machine probing or model execution occurs.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LABEL = re.compile(r'^s(\d+)-(.+)-(off|sync|events)$')
PHASE = re.compile(r'^(?:phase=(\w+) tokens=(\d+) loop_ms=([\d.]+)|decode_loop_ms=([\d.]+)) loop_closure=([\d.]+)$')
HEAD = re.compile(r'^mode=(\w+) evals=(\d+) aborted=(\d+) wall_ms=([\d.]+) span_ms=([\d.]+) closure=([\d.]+)')
ROW = re.compile(r'^(\w+)\s+(eval|host|gpu)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
PER_EVAL = re.compile(r'^per_eval (.*)$')


def is_global(layer):
    """Gemma 4 global-attention layers (the contract's `is_local` is `layer % 6 != 5`)."""
    return layer % 6 == 5


def parse_err(text):
    """Instrumented phases in order: [{phase, tokens, loop_ms, closure, evals, spans, layers}]."""
    phases, cur, layer_cols = [], None, None
    for line in text.splitlines():
        m = PHASE.match(line)
        if m:
            name, tokens, loop_ms, dec_ms, loop_closure = m.groups()
            cur = {'phase': name or 'decode', 'tokens': int(tokens) if tokens else None,
                   'loop_ms': float(loop_ms or dec_ms), 'loop_closure': float(loop_closure),
                   'spans': {}, 'layers': {}, 'counters': {}}
            phases.append(cur)
            layer_cols = None
            continue
        if cur is None:
            continue
        m = HEAD.match(line)
        if m:
            cur.update(mode=m.group(1), evals=int(m.group(2)), aborted=int(m.group(3)),
                       wall_ms=float(m.group(4)), closure=float(m.group(6)))
            if cur['tokens'] is None:
                cur['tokens'] = cur['evals']
            continue
        m = PER_EVAL.match(line)
        if m:
            cur['counters'] = {k: float(v) for k, v in (kv.split('=') for kv in m.group(1).split())}
            continue
        m = ROW.match(line)
        if m:
            span, group = m.group(1), m.group(2)
            cur['spans'][f'{span}/{group}'] = {'ms_per_eval': float(m.group(4)), 'dev_ms_per_eval': float(m.group(6)),
                                               'launch_per_eval': float(m.group(8))}
            continue
        if line.startswith('layer '):
            layer_cols = line.split()[1:]
            continue
        if layer_cols and line[:1].isdigit():
            fields = line.split()
            cur['layers'][int(fields[0])] = {c: tuple(float(x) for x in v.split('/'))
                                             for c, v in zip(layer_cols, fields[1:])}
    return phases


def per_token(phase):
    """Host (and device) ms per token for each span/group, plus the attn_core local/global split."""
    k = phase['evals'] / phase['tokens']
    out = {key: {'host_ms': v['ms_per_eval'] * k, 'dev_ms': v['dev_ms_per_eval'] * k}
           for key, v in phase['spans'].items()}
    split = defaultdict(lambda: [0.0, 0.0])
    for layer, cols in phase['layers'].items():
        if 'attn_core' in cols:
            kind = 'global' if is_global(layer) else 'local'
            split[kind][0] += cols['attn_core'][0] * k
            split[kind][1] += cols['attn_core'][1] * k
    for kind, (host, dev) in split.items():
        out[f'attn_core/{kind}'] = {'host_ms': host, 'dev_ms': dev}
    return out


def parse_out(text):
    reps = [json.loads(line) for line in text.splitlines() if '"kind":"rep"' in line]
    config = next(json.loads(line) for line in text.splitlines() if '"kind":"config"' in line)
    if not reps:
        raise ValueError('no repetition recorded')
    return config, reps


def load(case):
    summary = json.loads((case / 'run-summary.json').read_text())
    if summary['status'] != 'successful':
        return {'failed': summary.get('reason') or summary['status']}
    config, reps = parse_out((case / 'out.txt').read_text())
    last = reps[-1]
    phases = parse_err((case / 'err.txt').read_text())
    # Each repetition prints prefill then decode; keep the measured (last) repetition's.
    per_rep = 2 if config['n_decode'] else 1
    phases = phases[-per_rep:] if phases else []
    if any(p.get('aborted') for p in phases):
        raise ValueError(f'{case.name}: aborted evaluations')
    decode_ns = last['decode_ns']
    return {
        'config': {k: config[k] for k in ('n_prompt', 'n_decode', 'reps', 'n_batch', 'cpu_moe_layers',
                                          'cpu_expert_activation')},
        'prefill_tok_s': config['n_prompt'] / (last['prefill_ns'] / 1e9),
        'decode_ms': statistics.median(decode_ns) / 1e6 if decode_ns else None,
        'prefill_top1': last['prefill_top1'], 'top1': last['top1'],
        'phases': {p['phase']: {'closure': p['closure'], 'loop_closure': p['loop_closure'], 'evals': p['evals'],
                                'tokens': p['tokens'], 'loop_ms': p['loop_ms'], 'counters': p['counters'],
                                'per_token': per_token(p)} for p in phases},
    }


def reduce(campaign):
    cells = defaultdict(dict)
    failed = {}
    for case in sorted(Path(campaign).iterdir()):
        m = LABEL.match(case.name)
        if not case.is_dir() or not m:
            continue
        session, cell, mode = int(m.group(1)), m.group(2), m.group(3)
        row = load(case)
        if 'failed' in row:
            failed[case.name] = row['failed']
            continue
        cells[cell].setdefault(session, {})[mode] = row
    report = {'failed': failed, 'cells': {}}
    for cell, sessions in cells.items():
        out = {'sessions': {}}
        for s, modes in sessions.items():
            off = modes.get('off')
            entry = {}
            for mode, row in modes.items():
                e = {'prefill_tok_s': row['prefill_tok_s'], 'decode_ms': row['decode_ms'], 'phases': row['phases']}
                if off and mode != 'off':
                    e['prefill_overhead'] = off['prefill_tok_s'] / row['prefill_tok_s'] - 1
                    if off['decode_ms'] and row['decode_ms']:
                        e['decode_overhead'] = row['decode_ms'] / off['decode_ms'] - 1
                    e['top1_identical_to_off'] = (row['prefill_top1'] == off['prefill_top1']
                                                  and row['top1'] == off['top1'])
                entry[mode] = e
            out['sessions'][s] = entry
        report['cells'][cell] = out
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    json.dump(reduce(sys.argv[1]), sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == '__main__':
    main()
