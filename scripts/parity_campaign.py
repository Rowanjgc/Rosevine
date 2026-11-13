#!/usr/bin/env python3
"""Parity track P3: reduce a private replay campaign into the P1 parity report.

Offline only. Reads single-shot `gpu_campaign` command sessions whose labels are
`s<session>-<arm>-p<P>` (arm: rv = Rosevine HIP, hip/vk = pinned llama.cpp), each
holding the replay adapter's stdout. Produces per-cell paired ratios, correctness
(top-1 agreement), resource ratios and the frozen-denominator report. Output stays
private; publication needs human review. See docs/parity-contract.md.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

import parity_contract as P

ARMS = {'rv': ('rosevine', 'hip'), 'hip': ('llama.cpp', 'hip'), 'vk': ('llama.cpp', 'vulkan')}
LABEL = re.compile(r'^s(\d+)-(rv|hip|vk)-p(\d+)$')


def resources(case):
    """`/usr/bin/time -v` and supervised telemetry; missing values stay None."""
    out = {'high_water_rss_bytes': None, 'cpu_core_seconds': None, 'vram_peak_bytes': None,
           'thermal_flag_samples': 0, 'paging_warnings': 0, 'cpu_core_equivalent': None}
    text = (case / 'resources.txt').read_text() if (case / 'resources.txt').exists() else ''
    rss = re.search(r'Maximum resident set size \(kbytes\): (\d+)', text)
    share = re.search(r'Percent of CPU this job got: (\d+)%', text)
    out['cpu_core_equivalent'] = int(share.group(1)) / 100 if share else None
    user = re.search(r'User time \(seconds\): ([\d.]+)', text)
    system = re.search(r'System time \(seconds\): ([\d.]+)', text)
    if rss:
        out['high_water_rss_bytes'] = int(rss.group(1)) * 1024
    if user and system:
        out['cpu_core_seconds'] = float(user.group(1)) + float(system.group(1))
    telemetry = case / 'supervision' / 'telemetry.jsonl'
    session = json.loads((case / 'session.json').read_text())
    flags = {s['label']: s['flag_c'] for s in session['sensors'] if 'flag_c' in s}
    idle, peak = None, None
    for line in telemetry.read_text().splitlines() if telemetry.exists() else []:
        row = json.loads(line)
        snap = row.get('snapshot') or {}
        if 'vram_used_bytes' not in snap:
            continue
        if row.get('stage') == 'preflight':
            idle = snap['vram_used_bytes'] if idle is None else min(idle, snap['vram_used_bytes'])
        elif row.get('stage') == 'running':
            peak = max(peak or 0, snap['vram_used_bytes'])
            temps = snap.get('temperatures_c') or {}
            if any(temps.get(k, -1) >= v for k, v in flags.items()):
                out['thermal_flag_samples'] += 1
            if (snap.get('paging') or {}).get('warning'):
                out['paging_warnings'] += 1
    if peak is not None and idle is not None:
        out['vram_peak_bytes'] = peak - idle  # process-attributable estimate over the idle desktop
    return out


def load_case(case, label_match):
    session, arm, prompt = int(label_match.group(1)), label_match.group(2), int(label_match.group(3))
    summary = json.loads((case / 'run-summary.json').read_text())
    if summary['status'] != 'successful':
        return {'session': session, 'arm': arm, 'p': prompt, 'failed': summary['reason'] or summary['status']}
    replay = P.parse_replay((case / 'replay.jsonl').read_text(), engine=ARMS[arm][0])
    reps = replay['repetitions']
    measured = reps[1:] if len(reps) > 1 else reps  # rep 0 is a warmup whenever reps > 1
    resident = None
    err = (case / 'replay.err').read_text() if (case / 'replay.err').exists() else ''
    m = re.search(r'device_bytes=(\d+)', err)
    if m:
        resident = int(m.group(1))
    elif replay['config'].get('buffers') is not None:
        resident = sum(int(float(x) * 2**20) for line in replay['config']['buffers']
                       for x in re.findall(r'=\s*([\d.]+) MiB', line) if 'CPU' not in line.split('=')[0])
    return {'session': session, 'arm': arm, 'p': prompt, 'replay': replay, 'warm': len(reps) > 1,
            'prefill_tokens_per_s': statistics.median(r['prefill_tokens_per_s'] for r in measured),
            'decode_tokens_per_s': statistics.median(r['decode_tokens_per_s'] for r in measured)
            if replay['config']['n_decode'] else None,
            'top1': measured[-1]['top1'], 'prefill_top1': measured[-1]['prefill_top1'],
            'resident_bytes': resident, 'resources': resources(case)}


def declared(case, identity):
    engine, backend = ARMS[case['arm']]
    binary = identity['binaries'][case['arm']]
    return dict(backend=backend, binary_sha256=binary,
                source_revision=identity['rosevine_revision'] if engine == 'rosevine' else P.FROZEN_PIN,
                model_sha256=identity['model_sha256'], fixture_sha256=identity['fixture_sha256'],
                timing_family='engine', boundary=dict(P.ENGINE_BOUNDARY),
                warm_state='warm' if case['warm'] else 'cold', hardware_class=identity['hardware_class'],
                power_policy=identity['power_policy'], affinity=identity['affinity'],
                resident_bytes=case['resident_bytes'] or 0, untuned=engine == 'rosevine', series='frozen')


def reduce(campaign, identity, *, seed, agreement_floor, matrix=None):
    matrix = matrix or P.matrix_v1()
    cases = []
    for path in sorted(Path(campaign).iterdir()):
        m = LABEL.match(path.name)
        if m and (path / 'run-summary.json').exists():
            cases.append(load_case(path, m))
    failures = [c for c in cases if 'failed' in c]
    good = {(c['session'], c['arm'], c['p']): c for c in cases if 'failed' not in c}
    comparisons, details = {}, {}
    for cell in matrix['cells']:
        if cell['family'] != 'engine':
            continue
        metric = 'decode_tokens_per_s' if cell['phase'] == 'decode' else 'prefill_tokens_per_s'
        # Prefill cells reuse the decode process at the same P when no prefill-only run exists.
        for baseline, ref_arm in (('vulkan', 'vk'), ('hip', 'hip')):
            pairs, agreement, arms, res, prefill_matches, cores, flags = [], [], None, [], [], [], []
            for (session, arm, prompt), cand in sorted(good.items()):
                if arm != 'rv' or prompt != cell['n_prompt']:
                    continue
                ref = good.get((session, ref_arm, prompt))
                if ref is None or cand[metric] is None or ref[metric] is None:
                    continue
                if cell['phase'] == 'decode' and cand['replay']['config']['n_decode'] != cell['n_decode']:
                    continue
                pairs.append((ref[metric], cand[metric]))
                if cand['top1'] and ref['top1']:
                    agreement.append(P.top1_agreement(cand['top1'], ref['top1']))
                elif cand['prefill_top1'] is not None and ref['prefill_top1'] is not None:
                    # Prefill-only run: one argmax per session; agreement is the session fraction.
                    prefill_matches.append(cand['prefill_top1'] == ref['prefill_top1'])
                res.append(P.resource_ratios(ref['resources'], cand['resources'], {}))
                cores.append((ref['resources']['cpu_core_equivalent'], cand['resources']['cpu_core_equivalent']))
                flags.append(ref['resources']['thermal_flag_samples'] + cand['resources']['thermal_flag_samples'])
                arms = (ref, cand)
            if arms is None:
                continue
            if prefill_matches:
                agreement.append(sum(prefill_matches) / len(prefill_matches))
            ref_arm_id = P.arm_from_replay(arms[0]['replay'], **declared(arms[0], identity))
            cand_arm_id = P.arm_from_replay(arms[1]['replay'], **declared(arms[1], identity))
            pairing = P.classify_pair(ref_arm_id, cand_arm_id, cell, baseline)
            detail = {'pairs': len(pairs), 'top1_agreement': agreement,
                      'agreement_method': 'per-session prefill argmax match' if prefill_matches
                      else 'per-step decode argmax agreement',
                      'process_cpu_core_equivalent': [list(c) for c in cores],
                      'pairs_with_thermal_flag_samples': sum(1 for f in flags if f),
                      'reference_values': [p[0] for p in pairs], 'candidate_values': [p[1] for p in pairs]}
            details[(cell['id'], baseline)] = detail
            if len(pairs) < P.MIN_PAIRS:
                detail['note'] = 'fewer than %d pairs; no interval' % P.MIN_PAIRS
                continue
            ratio = P.paired_ratio([p[0] for p in pairs], [p[1] for p in pairs],
                                   higher_is_better=True, seed=seed)
            comparisons[(cell['id'], baseline)] = {
                'pairing': pairing, 'ratio': ratio,
                'correctness_pass': bool(agreement) and min(agreement) >= agreement_floor,
                'resources': {k: statistics.median(r[k]['ratio'] for r in res if r[k]['ratio'] is not None)
                              if any(r[k]['ratio'] is not None for r in res) else None
                              for k in ('vram_peak_bytes', 'high_water_rss_bytes', 'cpu_core_seconds')}}
    report = P.build_report(matrix, comparisons)
    report['details'] = {'%s|%s' % k: v for k, v in details.items()}
    report['failures'] = [{k: c[k] for k in ('session', 'arm', 'p', 'failed')} for c in failures]
    report['agreement_floor'] = agreement_floor
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign')
    parser.add_argument('identity', help='private JSON: binaries{rv,hip,vk}, rosevine_revision, model_sha256, '
                        'fixture_sha256, hardware_class, power_policy, affinity')
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--agreement-floor', type=float, default=0.85)
    parser.add_argument('--transitions', help='private JSON list of {cell, to, evidence} feasibility outcomes')
    args = parser.parse_args()
    identity = json.loads(Path(args.identity).read_text())
    matrix = P.matrix_v1()
    for t in json.loads(Path(args.transitions).read_text()) if args.transitions else []:
        matrix = P.advance_cell(matrix, t['cell'], t['to'], t['evidence'])
    print(json.dumps(reduce(args.campaign, identity, seed=args.seed, agreement_floor=args.agreement_floor,
                            matrix=matrix), indent=1))


if __name__ == '__main__':
    main()
