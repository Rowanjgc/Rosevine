#!/usr/bin/env python3
"""Reduce the private R3a.3 campaign; never approves or publishes its output."""
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_attribution import reduce_quality  # noqa: E402
from benchmark_core import paired_improvement, percentile  # noqa: E402


def read(path):
    with Path(path).open() as stream:
        text = stream.read(20 * 1024 * 1024 + 1)
    if len(text) > 20 * 1024 * 1024:
        raise ValueError('capture limit exceeded')
    return text


def generation(text):
    matches = re.findall(r'^prompt=(\d+) prefill_s=([\d.]+) steps=(\d+) decode_s=([\d.]+) decode_tok_s=([\d.]+)$', text, re.M)
    ids = re.findall(r'^generated (.*)$', text, re.M)
    if len(matches) != 1 or len(ids) != 1:
        raise ValueError('missing or duplicate generation result')
    prompt, prefill, steps, elapsed, rate = matches[0]
    tokens = [int(t) for t in ids[0].split()]
    values = [float(v) for v in (prefill, elapsed, rate)]
    if any(not math.isfinite(v) or v <= 0 for v in values) or len(tokens) != int(steps):
        raise ValueError('invalid generation samples')
    return dict(prompt=int(prompt), steps=int(steps), prefill_s=values[0], decode_s=values[1],
                rate=values[2], tokens=tokens)


def kld(text):
    rows = re.findall(r'^kld rows=(\d+) mean=([\d.]+).*?same_top=([\d.]+).*?logits_sha256=([0-9a-f]{64})$', text, re.M)
    chunks = re.findall(r'^chunk (\d+).*?logits_sha256=([0-9a-f]{64})$', text, re.M)
    if len(rows) != 1 or len(chunks) != 7 or [int(c[0]) for c in chunks] != list(range(7)):
        raise ValueError('incomplete KL coverage')
    n, mean, top, digest = rows[0]
    if int(n) != 1785 or not 0 <= float(mean) <= .08 or not .92 <= float(top) <= 1:
        raise ValueError('frozen KL band failed')
    return dict(rows=int(n), mean=float(mean), top1=float(top), digest=digest,
                chunk_digests=[c[1] for c in chunks])


def successful(directory):
    result = json.loads(read(directory / 'run-summary.json'))
    if result['status'] != 'successful' or result['cleanup_pending'] or result['returncode'] != 0:
        raise ValueError('supervised run was not successful and reaped')


def paired(root, prefix, steps):
    baseline, candidate, clean_b, clean_c, records = [], [], [], [], []
    for index in range(1, 8):
        session = 1 if index <= 4 else 2
        pair = {}
        argv = []
        for kind in ('baseline', 'candidate'):
            directory = root / f'{prefix}s{session}-p{index}-{kind}'
            successful(directory)
            work = json.loads(read(directory / 'work.json'))['argv']
            argv.append(work[work.index('generate'):])
            sample = generation(read(directory / 'generation.log'))
            if sample['prompt'] != 40 or sample['steps'] != steps:
                raise ValueError('workload shape mismatch')
            sample['quality'] = reduce_quality(directory)
            observer = json.loads(read(directory / 'observer.json'))['lateness_ns']
            if not observer or any(not isinstance(v, int) or v < 0 for v in observer):
                raise ValueError('invalid observer samples')
            sample['observer_p95_ms'] = percentile(observer, .95) / 1e6
            pair[kind] = sample
        if argv[0] != argv[1]:
            raise ValueError('unmatched generation options')
        equal = pair['baseline']['tokens'] == pair['candidate']['tokens']
        if prefix and not equal:
            raise ValueError('matched-prefix continuations differ')
        pair['identical_tokens'] = equal
        b, c = pair['baseline']['rate'], pair['candidate']['rate']
        baseline.append(b)
        candidate.append(c)
        if not any(pair[k]['quality']['thermal_flagged'] for k in ('baseline', 'candidate')):
            clean_b.append(b)
            clean_c.append(c)
        for kind in ('baseline', 'candidate'):
            del pair[kind]['tokens']
        records.append(pair)
    return dict(baseline_median=statistics.median(baseline), candidate_median=statistics.median(candidate),
                all_pairs=paired_improvement(baseline, candidate, higher_is_better=True, seed=20261113),
                unflagged_pairs=len(clean_b),
                unflagged_interval=paired_improvement(clean_b, clean_c, higher_is_better=True, seed=20261113)
                if len(clean_b) >= 7 else None,
                unflagged_interval_gap=None if len(clean_b) >= 7 else 'fewer than seven unflagged matched pairs',
                records=records)


def main(root):
    root = Path(root)
    result = dict(primary=paired(root, '', 200), matched100=paired(root, 'matched100-', 100))
    references = {}
    for name in ('kl-cpu', 'kl-vk-fast', 'kl-step-cpu'):
        directory = root / name
        successful(directory)
        references[name] = kld(read(directory / 'command.log'))
    if any(references[name]['chunk_digests'] != references['kl-cpu']['chunk_digests']
           for name in ('kl-vk-fast', 'kl-step-cpu')):
        raise ValueError('batch/step or reference-input digest mismatch')
    result['references'] = references
    a = generation(read(root / 'fixture-baseline/generation.log'))['tokens']
    b = generation(read(root / 'fixture-candidate/generation.log'))['tokens']
    historical = read(root.parent / 'r3a1/fixture-off.txt').splitlines()[1].split()[1:]
    if list(map(str, a)) != historical or len(a) != 520 or len(b) != 520:
        raise ValueError('historical fixture baseline mismatch')
    result['fixture'] = dict(first_divergence_one_based=next((i+1 for i, (x, y) in enumerate(zip(a, b)) if x != y), None),
                             same_positions=sum(x == y for x, y in zip(a, b)))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main(sys.argv[1])
