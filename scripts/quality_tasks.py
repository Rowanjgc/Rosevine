#!/usr/bin/env python3
"""Original deterministic scored tasks and paired task-accuracy diagnostics.

No model execution or external corpus. Scoring accepts the visible response only,
not reasoning channels. Shared problem templates limit the scope of generality;
these instruction/reasoning checks do not replace prose/code NLL evaluation.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

from benchmark_core import canonical, digest, write_private_json

VERSION = 'rosevine-quality-tasks-v1'
CATEGORIES = ('arithmetic', 'structured-transform', 'algorithm-trace', 'text-instructions', 'retrieval')
WORDS = ('amber', 'birch', 'cedar', 'delta', 'ember', 'fern', 'granite', 'harbor', 'indigo', 'juniper')
PAIRED_BOUND_ALPHA = .025


def build_tasks(split, seed, per_category):
    if split not in ('calibration', 'holdout') or type(seed) is not int or type(per_category) is not int or per_category < 1:
        raise ValueError('explicit split, seed and positive category count required')
    rng, result = random.Random(seed), []
    for category in CATEGORIES:
        for index in range(per_category):
            if category == 'arithmetic':
                a, b, c, d = [rng.randrange(11, 99) for _ in range(4)]
                prompt = f'Compute ({a} * {b}) - ({c} + {d}). Return the integer under key result.'
                expected = {'result': a * b - c - d}
            elif category == 'structured-transform':
                values = rng.sample(range(-90, 100), 9)
                prompt = f'From {json.dumps(values)}, select positive even integers, sort ascending, and return the list under key values.'
                expected = {'values': sorted(v for v in values if v > 0 and v % 2 == 0)}
            elif category == 'algorithm-trace':
                values = [rng.randrange(-20, 40) for _ in range(7)]
                multiplier, offset = rng.randrange(2, 6), rng.randrange(1, 10)
                prompt = (f'Trace this pseudocode on xs={json.dumps(values)}: total=0; for each x in xs, '
                          f'if x>0 then total=total+{multiplier}*x else total=total-{offset}. '
                          'Return total under key total. Do not execute code.')
                expected = {'total': sum(multiplier * x if x > 0 else -offset for x in values)}
            elif category == 'text-instructions':
                words = rng.sample(WORDS, 6)
                prompt = (f'Given the ordered words {json.dumps(words)}, keep the words at one-based '
                          'positions 2, 4, and 6, then reverse that selected list and uppercase each word. '
                          'Return the list under key words.')
                expected = {'words': [words[i].upper() for i in [5, 3, 1]]}
            else:
                keys = rng.sample(WORDS, 8)
                values = rng.sample(range(100, 1000), 9)
                target = keys[rng.randrange(len(keys))]
                records = [f'{k}={v}' for k, v in zip(keys, values)]
                insertion = rng.randrange(1, len(records) + 1)
                records.insert(insertion, f'{target}={values[-1]}')
                last = [int(record.split('=')[1]) for record in records if record.startswith(target + '=')][-1]
                prompt = (f'Records in order: {"; ".join(records)}. Later records replace earlier values '
                          f'for the same key. Return the final value of {target} under key value.')
                expected = {'value': last}
            prompt += ' Return only a JSON object, without markdown or additional keys.'
            result.append({'id': f'{split}-{category}-{index:03d}', 'category': category,
                           'split': split, 'prompt': prompt, 'expected': expected})
    return result


def suite():
    calibration = build_tasks('calibration', 20261112, 4)
    holdout = build_tasks('holdout', 20261113, 20)
    prompts = [t['prompt'] for t in calibration + holdout]
    if len(set(prompts)) != len(prompts):
        raise ValueError('duplicate quality prompt across or within splits')
    return {'version': VERSION, 'license': 'GPL-3.0-only',
            'provenance': 'Original synthetic Rosevine task definitions; no copied external tasks.',
            'calibration': calibration, 'holdout': holdout,
            'calibration_sha256': digest(calibration), 'holdout_sha256': digest(holdout),
            'limitations': 'Shared templates across disjoint prompts; no claim of unseen task-family or pretraining exclusion; not an NLL corpus.'}


def score(task, visible_response):
    if not isinstance(task, dict) or not isinstance(task.get('expected'), dict) or not task['expected']:
        raise ValueError('valid expected JSON object required')
    expected_json = canonical(task['expected'])
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    try:
        if not isinstance(visible_response, str):
            return False
        candidate = json.loads(visible_response, object_pairs_hook=unique_object)
        # JSON encoding distinguishes booleans from numbers and rejects NaN/Inf.
        return canonical(candidate) == expected_json
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False


def binomial_cdf(k, n, probability):
    if k < 0:
        return 0.0
    if k >= n or probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    return min(1.0, math.fsum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                     + i * math.log(probability) + (n - i) * math.log1p(-probability))
                              for i in range(k + 1)))


def cdf_inverse_probability(k, n, target):
    low, high = 0.0, 1.0
    for _ in range(64):
        mid = (low + high) / 2
        if binomial_cdf(k, n, mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def minimum_zero_discordance_tasks(margin, alpha=PAIRED_BOUND_ALPHA):
    """Best-case iid task count for a one-sided CP bound within ``margin``.

    Assumes zero paired wins and losses. ``alpha`` is the one-sided loss-bound
    tail probability used by the Bonferroni paired-accuracy construction; it is
    not a correction for task-family clustering.
    """
    if (type(margin) not in (int, float) or not math.isfinite(margin) or not 0 < margin < 1
            or type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('finite margin and alpha must each lie strictly between zero and one')
    n = max(1, math.ceil(math.log(alpha) / math.log1p(-margin)))

    def upper_bound(count):
        return 1.0 - alpha ** (1 / count)

    while upper_bound(n) > margin:
        n += 1
    while n > 1 and upper_bound(n - 1) <= margin:
        n -= 1
    return {'minimum_iid_tasks': n, 'margin': float(margin), 'one_sided_alpha': float(alpha),
            'upper_bound_at_minimum': upper_bound(n),
            'upper_bound_at_previous_n': upper_bound(n - 1) if n > 1 else None,
            'assumption': 'zero wins and losses; independent task pairs; best-case planning bound only'}


def paired_accuracy(reference, candidate):
    """Conservative 95% upper bound for paired accuracy loss.

    For task-wise D=reference-candidate, E[D]=P(loss)-P(win). Bonferroni
    combines a 97.5% one-sided Clopper-Pearson loss upper bound and win lower
    bound, so their difference is a conservative >=95% upper bound. This avoids
    a spuriously zero bootstrap interval when all observed pairs agree.
    Assumes independent task sampling; shared templates may violate that model.
    """
    if not isinstance(reference, dict) or not isinstance(candidate, dict) or not reference or reference.keys() != candidate.keys() or any(
            type(value) is not bool for value in list(reference.values()) + list(candidate.values())):
        raise ValueError('matched task IDs with boolean scores required')
    n = len(reference)
    losses = sum(reference[k] and not candidate[k] for k in reference)
    wins = sum(candidate[k] and not reference[k] for k in reference)
    loss_upper = 1.0 if losses == n else cdf_inverse_probability(losses, n, PAIRED_BOUND_ALPHA)
    win_lower = 0.0 if wins == 0 else cdf_inverse_probability(wins - 1, n, 1 - PAIRED_BOUND_ALPHA)
    return {'tasks': n, 'reference_accuracy': sum(reference.values()) / n,
            'candidate_accuracy': sum(candidate.values()) / n,
            'losses': losses, 'wins': wins, 'accuracy_drop': (losses - wins) / n,
            'conservative_95_upper_drop': loss_upper - win_lower,
            'method': 'paired discordance; Bonferroni one-sided 97.5% Clopper-Pearson loss/win bounds',
            'independence_caveat': 'shared-template task dependence requires cluster-aware review',
            'acceptance': 'not decided; frozen quality margins and task sampling review required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new file in private 0700 directory')
    args = parser.parse_args()
    write_private_json(args.output, suite())
    print('Private synthetic task manifest created; no model evaluation performed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError):
        print('Task manifest construction failed.', file=sys.stderr)
        sys.exit(1)
