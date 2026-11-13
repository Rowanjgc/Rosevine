#!/usr/bin/env python3
"""Parity track P1: frozen Vulkan/HIP parity matrix, arm identity and comparison rules.

Offline only: no endpoint, process launch or hardware probe. This module turns
private adapter output into per-cell paired ratios and milestone flags. It does
not authorize a run, does not average across workloads or baselines and never
turns an unmatched comparison into an execution-machinery parity claim.
See docs/parity-contract.md.
"""
import json
import math
import re
import statistics

from benchmark_core import canonical, digest, paired_schedule, percentile, summarize

MATRIX_SCHEMA = 'rosevine-parity-matrix-v1'
REPLAY_SCHEMA = 'rosevine-parity-replay-v1'
ARM_SCHEMA = 'rosevine-parity-arm-v1'
FROZEN_PIN = 'cd959bae99ff4bed22bef2604f45241c09bbb8c8'
NEWER_CONTROL_PIN = '9daa0aa22b8aa0143344d7549dee1f0e32f048bb'
BASELINES = ('vulkan', 'hip')
FAMILIES = ('engine', 'user_visible')
# Rosevine positions beyond ~1,030 are not yet exercised on the real artifact
# (runtime-r2-execution.md Limits); such cells need a feasibility check first.
VERIFIED_POSITIONS = 1030
DECODE_DEPTHS = (512, 2048, 8192, 16384)
PREFILL_SIZES = (128, 512, 2048, 8192)
DECODE_TOKENS = 128
MIN_PAIRS = 7
P90_LOWER = 0.90
PARITY_MARGIN_LOWER = 0.98
ADVANTAGE_PRACTICAL = 1.05

# Must be identical in both arms, otherwise the pair is rejected outright.
HARD_FIELDS = ('model_sha256', 'fixture_sha256', 'tokens_fnv1a64', 'n_prompt', 'n_decode',
               'timing_family', 'boundary', 'warm_state', 'hardware_class', 'kv_type', 'power_policy')
# Numerics/placement-visible state. A difference downgrades the pair to a labelled
# practical-configuration comparison; it is never execution-machinery parity.
STATE_FIELDS = ('cpu_moe_layers', 'tensor_homes', 'cpu_expert_activation', 'capacity_tokens')
# Implementation choices under test. They must be declared and are recorded, not matched.
MACHINERY_FIELDS = ('microbatch', 'threads', 'affinity', 'attention_policy', 'cpu_repack',
                    'expert_kernel', 'expert_wait', 'resident_bytes')
IDENTITY_FIELDS = ('engine', 'backend', 'source_revision', 'binary_sha256', 'untuned', 'series')
ARM_FIELDS = set(HARD_FIELDS + STATE_FIELDS + MACHINERY_FIELDS + IDENTITY_FIELDS)
ENGINE_BOUNDARY = {'synchronized': True, 'logits': 'last_position_host', 'sampling': False,
                   'tokenization': False, 'delivery': False}
USER_BOUNDARY = {'synchronized': True, 'logits': 'last_position_host', 'sampling': True,
                 'tokenization': True, 'delivery': True}


def is_hash(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _cell(cell_id, phase, family, n_prompt, n_decode, metric, unit, higher_is_better):
    status, reason = 'supported', None
    if family == 'user_visible':
        status = 'unsupported'
        reason = 'Rosevine has no request-admission/delivery path yet; no matched generation harness'
    elif n_prompt + n_decode > VERIFIED_POSITIONS:
        status = 'feasibility_check_required'
        reason = ('Rosevine positions beyond %d are unexercised on the real artifact; '
                  'budget and attention feasibility must pass before timing' % VERIFIED_POSITIONS)
    return {'id': cell_id, 'phase': phase, 'family': family, 'n_prompt': n_prompt,
            'n_decode': n_decode, 'metric': metric, 'unit': unit,
            'higher_is_better': higher_is_better, 'status': status, 'reason': reason}


def matrix_v1():
    """The predeclared denominator. Statuses may only move forward after evidence,
    never be dropped after seeing results."""
    cells = []
    for depth in DECODE_DEPTHS:
        cells.append(_cell('decode-d%d' % depth, 'decode', 'engine', depth, DECODE_TOKENS,
                           'decode_tokens_per_s', 'tokens/s', True))
    for size in PREFILL_SIZES:
        cells.append(_cell('prefill-p%d' % size, 'prefill', 'engine', size, 0,
                           'prefill_tokens_per_s', 'tokens/s', True))
    for size in PREFILL_SIZES:
        cells.append(_cell('ttft-warm-p%d' % size, 'ttft', 'user_visible', size, 1,
                           'ttft_s', 's', False))
    matrix = {'schema': MATRIX_SCHEMA, 'reference_pin': FROZEN_PIN, 'baselines': list(BASELINES),
              'cells': cells,
              'milestones': {'p90_lower': P90_LOWER, 'parity_point': 1.0,
                             'parity_lower': PARITY_MARGIN_LOWER, 'advantage_point': ADVANTAGE_PRACTICAL,
                             'advantage_lower_exclusive': 1.0, 'confidence': 0.95,
                             'min_pairs': MIN_PAIRS}}
    matrix['matrix_sha256'] = digest(matrix)
    return matrix


def validate_matrix(matrix):
    body = {k: v for k, v in matrix.items() if k != 'matrix_sha256'}
    if matrix.get('schema') != MATRIX_SCHEMA or digest(body) != matrix.get('matrix_sha256'):
        raise ValueError('matrix schema/checksum mismatch: the denominator is frozen before runs')
    ids = [c['id'] for c in matrix['cells']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate matrix cell')
    for c in matrix['cells']:
        if c['status'] not in ('supported', 'feasibility_check_required', 'unsupported'):
            raise ValueError('unknown cell status')
        if c['status'] != 'supported' and not (isinstance(c['reason'], str) and c['reason'].strip()):
            raise ValueError('non-supported cell needs an explicit reason')
    return matrix


# ---------------------------------------------------------------- adapters

def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid %s' % name)
    return value


def tokens_fnv1a64(tokens):
    """Binding hash both replay adapters print (FNV-1a 64 over little-endian u32 ids)."""
    h = 0xcbf29ce484222325
    for t in tokens:
        if type(t) is not int or not 0 <= t < 1 << 31:
            raise ValueError('token id out of range')
        for byte in t.to_bytes(4, 'little'):
            h = ((h ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return '%016x' % h


def _reject_constant(name):
    raise ValueError('nonfinite JSON constant %s' % name)


def parse_replay(text, *, engine):
    """Parse one replay process's stdout (Rosevine `replay` or research/parity_replay.cpp)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError('empty replay output')
    rows = []
    for line in lines:
        row = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(row, dict) or row.get('schema') != REPLAY_SCHEMA:
            raise ValueError('foreign replay row')
        rows.append(row)
    config, reps = rows[0], rows[1:]
    if config.get('kind') != 'config' or config.get('engine') != engine:
        raise ValueError('replay config row for the declared engine required first')
    n_prompt = _int(config.get('n_prompt'), 'n_prompt', 1)
    n_decode = _int(config.get('n_decode'), 'n_decode')
    if not is_hash(config.get('tokens_fnv1a64'), 16):
        raise ValueError('token binding hash missing')
    if len(reps) != _int(config.get('reps'), 'reps', 1):
        raise ValueError('incomplete replay: repetition count differs from config')
    out = []
    for index, rep in enumerate(reps):
        if rep.get('kind') != 'rep' or rep.get('rep') != index:
            raise ValueError('replay repetitions out of order')
        prefill = _int(rep.get('prefill_ns'), 'prefill_ns', 1)
        steps = rep.get('decode_ns')
        if not isinstance(steps, list) or len(steps) != n_decode:
            raise ValueError('decode step count differs from config')
        for s in steps:
            _int(s, 'decode_ns', 1)
        top1 = rep.get('top1')
        if top1 is not None and (not isinstance(top1, list) or len(top1) != n_decode
                                 or any(type(t) is not int or t < 0 for t in top1)):
            raise ValueError('top-1 record differs from decode count')
        out.append({'top1': top1, 'prefill_top1': rep.get('prefill_top1'),
                    'prefill_tokens_per_s': n_prompt * 1e9 / prefill,
                    'decode_tokens_per_s': n_decode * 1e9 / sum(steps) if n_decode else None,
                    'decode_step_ms_median': statistics.median(steps) / 1e6 if n_decode else None})
    return {'config': config, 'repetitions': out}


def parse_llama_bench_json(text):
    """llama-bench `-o json`. Diagnostic only: it feeds random token IDs, so MoE routing
    differs from any frozen fixture; records carry fixture 'random-token' and can only
    pair with other llama-bench records."""
    rows = json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError('llama-bench JSON list required')
    out = []
    for row in rows:
        n_prompt, n_gen = _int(row.get('n_prompt'), 'n_prompt'), _int(row.get('n_gen'), 'n_gen')
        samples = row.get('samples_ts')
        if not isinstance(samples, list) or not samples or (n_prompt and n_gen):
            raise ValueError('one-phase llama-bench rows with samples required')
        for v in samples:
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValueError('invalid llama-bench sample')
        out.append({'phase': 'prefill' if n_prompt else 'decode', 'n_prompt': n_prompt,
                    'n_gen': n_gen, 'depth': _int(row.get('n_depth', 0), 'n_depth'),
                    'build_commit': row.get('build_commit'), 'backend': row.get('backends'),
                    'samples_tokens_per_s': samples, 'fixture': 'random-token'})
    return out


# ---------------------------------------------------------------- identities

def validate_arm(arm):
    if not isinstance(arm, dict) or arm.get('schema') != ARM_SCHEMA or set(arm) - {'schema'} != ARM_FIELDS:
        raise ValueError('complete parity arm identity required; unknown is not matched')
    for name in ARM_FIELDS:
        if arm[name] is None:
            raise ValueError('null arm field %s: supply a value' % name)
    if arm['engine'] not in ('rosevine', 'llama.cpp'):
        raise ValueError('unknown engine')
    if arm['backend'] not in BASELINES:
        raise ValueError('unknown backend')
    if not is_hash(arm['source_revision'], 40):
        raise ValueError('full source revision required')
    for name in ('model_sha256', 'fixture_sha256', 'binary_sha256'):
        if not is_hash(arm[name]) and not (name == 'fixture_sha256' and arm[name] == 'random-token'):
            raise ValueError('invalid %s' % name)
    if arm['timing_family'] not in FAMILIES:
        raise ValueError('unknown timing family')
    expected = ENGINE_BOUNDARY if arm['timing_family'] == 'engine' else USER_BOUNDARY
    if arm['boundary'] != expected:
        raise ValueError('timing boundary does not match its family')
    if arm['warm_state'] not in ('warm', 'cold'):
        raise ValueError('warm or cold state required')
    if arm['series'] not in ('frozen', 'newer-control'):
        raise ValueError('series must be frozen or newer-control')
    if arm['engine'] == 'llama.cpp':
        pin = FROZEN_PIN if arm['series'] == 'frozen' else NEWER_CONTROL_PIN
        if arm['source_revision'] != pin:
            raise ValueError('reference revision does not match its declared series')
    if type(arm['untuned']) is not bool:
        raise ValueError('explicit untuned flag required')
    canonical(arm)
    return arm


def arm_from_replay(replay, **declared):
    """Bind a parsed replay config to a declared identity; adapter facts win over claims."""
    config = replay['config']
    facts = {'n_prompt': config['n_prompt'], 'n_decode': config['n_decode'],
             'tokens_fnv1a64': config['tokens_fnv1a64'], 'kv_type': config['kv_type'],
             'cpu_moe_layers': config['cpu_moe_layers'],
             'tensor_homes': 'routed experts of layers [0,%d) on host; all else device' % config['cpu_moe_layers'],
             'cpu_expert_activation': config['cpu_expert_activation'],
             'capacity_tokens': config['n_ctx'], 'microbatch': config['n_ubatch'],
             'threads': config['threads']}
    for name, value in facts.items():
        if name in declared and canonical(declared[name]) != canonical(value):
            raise ValueError('declared %s contradicts adapter output' % name)
    engine = 'llama.cpp' if config['engine'] == 'llama.cpp' else 'rosevine'
    arm = dict(declared, **facts, schema=ARM_SCHEMA, engine=engine)
    if engine == 'llama.cpp':
        arm.setdefault('cpu_repack', config['cpu_repack'])
        arm.setdefault('attention_policy', 'flash_attn=' + config['flash_attn'])
        arm.setdefault('expert_kernel', 'ggml-cpu')
        arm.setdefault('expert_wait', 'ggml-threadpool')
    else:
        arm.setdefault('cpu_repack', 'off')
        arm.setdefault('attention_policy', 'rosevine-native')
        arm.setdefault('expert_kernel', config['expert_kernel'])
        arm.setdefault('expert_wait', config['expert_wait'])
    return validate_arm(arm)


def top1_agreement(a, b):
    """Fraction of decode steps whose argmax ids agree; a correctness signal, not a speed one."""
    if not a or a is None or b is None or len(a) != len(b):
        raise ValueError('matched nonempty top-1 records required')
    return sum(x == y for x, y in zip(a, b)) / len(a)


def classify_pair(reference, candidate, cell, baseline):
    """Reject incompatible pairs; otherwise label machinery parity or practical comparison."""
    validate_arm(reference)
    validate_arm(candidate)
    if reference['engine'] != 'llama.cpp' or candidate['engine'] != 'rosevine':
        raise ValueError('reference must be llama.cpp and candidate Rosevine')
    if reference['backend'] != baseline or baseline not in BASELINES:
        raise ValueError('reference backend differs from the predeclared baseline')
    if candidate['backend'] != 'hip':
        raise ValueError('Rosevine candidate runs HIP under ADR 0001')
    if cell['family'] != reference['timing_family']:
        raise ValueError('cell timing family differs from arms')
    # Prefill timing closes before any decode step, so a prefill cell needs only P to match;
    # decode and TTFT cells need both P and D.
    wanted = (cell['n_prompt'],) if cell['phase'] == 'prefill' else (cell['n_prompt'], cell['n_decode'])
    if wanted != (reference['n_prompt'], reference['n_decode'])[:len(wanted)]:
        raise ValueError('arms do not run this cell workload')
    if reference['fixture_sha256'] == 'random-token':
        raise ValueError('random-token llama-bench data cannot pair with a frozen fixture')
    hard = [n for n in HARD_FIELDS if canonical(reference[n]) != canonical(candidate[n])]
    if hard:
        raise ValueError('incompatible comparison; differing fields: ' + ', '.join(hard))
    state = [n for n in STATE_FIELDS if canonical(reference[n]) != canonical(candidate[n])]
    return {'class': 'practical_configuration' if state else 'machinery_parity',
            'state_differences': {n: [reference[n], candidate[n]] for n in state},
            'machinery': {n: [reference[n], candidate[n]] for n in MACHINERY_FIELDS},
            'reference_series': reference['series'], 'candidate_untuned': candidate['untuned']}


# ---------------------------------------------------------------- statistics

def paired_ratio(reference, candidate, *, higher_is_better, seed, resamples=10000):
    """Per-pair speed ratio (larger is better), median and percentile-bootstrap 95% CI.

    Throughput: Rosevine / reference. Latency: reference / Rosevine. Each pair is one
    matched session/repetition, never individual tokens.
    """
    import random
    if len(reference) != len(candidate) or len(reference) < MIN_PAIRS:
        raise ValueError('at least %d matched pairs required' % MIN_PAIRS)
    for v in list(reference) + list(candidate):
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('positive finite samples required')
    if type(resamples) is not int or resamples < 1000:
        raise ValueError('at least 1000 bootstrap resamples required')
    ratios = [c / r if higher_is_better else r / c for r, c in zip(reference, candidate)]
    rng = random.Random(seed)
    boot = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(resamples)]
    return {'pairs': len(ratios), 'ratio_median': statistics.median(ratios),
            'ci95': [percentile(boot, .025), percentile(boot, .975)],
            'reference': summarize(list(reference)), 'candidate': summarize(list(candidate)),
            'method': 'paired ratios; median; percentile bootstrap', 'seed': seed,
            'resamples': resamples}


def milestones(ratio):
    point, (lower, _) = ratio['ratio_median'], ratio['ci95']
    return {'p90': lower >= P90_LOWER,
            'operational_parity': point >= 1.0 and lower >= PARITY_MARGIN_LOWER,
            'strict_parity': lower >= 1.0,
            'specialization_advantage': lower > 1.0 and point >= ADVANTAGE_PRACTICAL}


def resource_ratios(reference, candidate, budgets):
    """Per-cell resource ratios in natural units (candidate / reference, lower is better).

    A missing value stays missing with its reason; a budget breach is reported even
    when speed improves."""
    out = {}
    for name in ('vram_peak_bytes', 'high_water_rss_bytes', 'cpu_core_seconds'):
        r, c = reference.get(name), candidate.get(name)
        if not isinstance(r, (int, float)) or not isinstance(c, (int, float)) or r <= 0:
            out[name] = {'ratio': None, 'reason': 'missing or nonpositive measurement in an arm'}
            continue
        limit = budgets.get(name)
        out[name] = {'reference': r, 'candidate': c, 'ratio': c / r,
                     'budget': limit, 'breach': limit is not None and c > limit}
    return out


def pair_schedule(cell_ids, seed, warmups=2, trials=MIN_PAIRS):
    """Alternating reference/candidate schedule reusing the experiment-protocol helper."""
    return paired_schedule(cell_ids, ['reference', 'candidate'], seed, warmups=warmups, trials=trials)


# ---------------------------------------------------------------- report

def build_report(matrix, comparisons):
    """One row per (cell, baseline). No aggregate score; unsupported/not-run stay in the
    denominator. `comparisons[(cell_id, baseline)]` = {'pairing': classify_pair(...),
    'ratio': paired_ratio(...), 'correctness_pass': bool, 'resources': resource_ratios(...)}."""
    validate_matrix(matrix)
    known = {(c['id'], b) for c in matrix['cells'] for b in BASELINES}
    if set(comparisons) - known:
        raise ValueError('comparison outside the frozen matrix')
    rows, counts = [], {b: {'cells': 0, 'p90': 0, 'operational_parity': 0,
                            'specialization_advantage': 0} for b in BASELINES}
    for cell in matrix['cells']:
        for baseline in BASELINES:
            counts[baseline]['cells'] += 1
            entry = comparisons.get((cell['id'], baseline))
            row = {'cell': cell['id'], 'baseline': baseline, 'family': cell['family'],
                   'metric': cell['metric'], 'unit': cell['unit']}
            if cell['status'] == 'unsupported':
                row.update(outcome='unsupported', reason=cell['reason'])
            elif entry is None:
                row.update(outcome='not_run', reason=cell['reason'] or 'no paired result recorded')
            elif entry['correctness_pass'] is not True:
                row.update(outcome='correctness_failed', ratio=entry['ratio'])
            else:
                flags = milestones(entry['ratio'])
                cls = entry['pairing']['class']
                row.update(outcome='measured', comparison_class=cls, ratio=entry['ratio'],
                           milestones=flags, resources=entry.get('resources'),
                           untuned=entry['pairing']['candidate_untuned'],
                           state_differences=entry['pairing']['state_differences'])
                if cls == 'machinery_parity':
                    for name, hit in flags.items():
                        if hit and name in counts[baseline]:
                            counts[baseline][name] += 1
            rows.append(row)
    return {'matrix_sha256': matrix['matrix_sha256'], 'rows': rows,
            'milestone_counts_by_baseline': counts,
            'notes': ['counts use the full predeclared denominator; practical-configuration '
                      'rows never count toward parity milestones',
                      'HIP results diagnose same-backend machinery; Vulkan is the Gate 4 '
                      'comparator and a Vulkan deficit is not offset by HIP',
                      'early comparisons with untuned=true are diagnostics, not Gate 4 claims']}
