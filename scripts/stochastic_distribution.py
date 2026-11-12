#!/usr/bin/env python3
"""Compare sampled token distributions from paired, frozen generation captures.

The analyzer reports empirical per-position marginals only. It does not claim
sequence-level equivalence, runtime provenance, statistical acceptance, or a
quality result; callers must separately bind captures to supervised runs.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import stat
import sys

from benchmark_core import write_private_json
from stochastic_sampler_events import parse as parse_sampler_events
from verify_capture_bundle import secure_bytes, secure_json, verify_bundle


def is_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def load_capture(path):
    path = Path(path)
    raw = secure_bytes(path, 32 * 1024 * 1024)
    capture = secure_json(path, 32 * 1024 * 1024)
    return parse_capture(capture, hashlib.sha256(raw).hexdigest())


def parse_capture(capture, capture_sha256):
    expected = {'schema', 'model_sha256', 'sampler_policy_sha256', 'sampler_observer_sha256', 'vocabulary_size',
                'eos_token_id', 'max_output_tokens', 'seed_schedule', 'prompts'}
    if not isinstance(capture, dict) or set(capture) != expected or capture['schema'] != 'rosevine-stochastic-output-v3':
        raise ValueError('stochastic capture schema mismatch')
    if any(not is_hash(capture[key]) for key in
           ('model_sha256', 'sampler_policy_sha256', 'sampler_observer_sha256')):
        raise ValueError('stochastic model, sampler-policy and observer hashes are required')
    vocab, eos, maximum = (capture['vocabulary_size'], capture['eos_token_id'],
                           capture['max_output_tokens'])
    if (type(vocab) is not int or vocab <= 1 or type(eos) is not int or not 0 <= eos < vocab
            or type(maximum) is not int or maximum <= 0):
        raise ValueError('invalid vocabulary, EOS or output bound')
    seeds = capture['seed_schedule']
    if (not isinstance(seeds, list) or len(seeds) < 20
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('at least 20 distinct nonnegative seeds are required')
    prompts = capture['prompts']
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('nonempty prompt capture list required')
    by_prompt, prompt_hashes = {}, set()
    for prompt in prompts:
        if (not isinstance(prompt, dict) or set(prompt) != {'prompt_id', 'prompt_sha256', 'samples'}
                or not isinstance(prompt['prompt_id'], str) or not prompt['prompt_id']
                or not is_hash(prompt['prompt_sha256']) or prompt['prompt_id'] in by_prompt
                or prompt['prompt_sha256'] in prompt_hashes):
            raise ValueError('invalid or duplicate stochastic prompt identity')
        prompt_hashes.add(prompt['prompt_sha256'])
        samples = prompt['samples']
        if not isinstance(samples, list) or len(samples) != len(seeds):
            raise ValueError('every prompt requires one sample for every planned seed')
        by_seed = {}
        for sample in samples:
            if (not isinstance(sample, dict)
                    or set(sample) != {'seed', 'status', 'finish_reason', 'token_ids', 'run_bundle'}
                    or type(sample['seed']) is not int or sample['seed'] in by_seed
                    or not isinstance(sample['run_bundle'], str) or not sample['run_bundle']):
                raise ValueError('invalid or duplicate stochastic sample')
            if sample['status'] != 'successful' or sample['finish_reason'] not in ('eos', 'limit'):
                raise ValueError('failed or unplanned sample outcome cannot be dropped')
            tokens = sample['token_ids']
            if (not isinstance(tokens, list) or not 1 <= len(tokens) <= maximum
                    or any(type(token) is not int or not 0 <= token < vocab for token in tokens)):
                raise ValueError('sample token IDs violate the frozen output bounds')
            if sample['finish_reason'] == 'eos':
                if tokens[-1] != eos or eos in tokens[:-1]:
                    raise ValueError('EOS completion must end with one EOS token')
            elif len(tokens) != maximum or eos in tokens:
                raise ValueError('length completion must reach the output bound without EOS')
            by_seed[sample['seed']] = {'token_ids': tokens,
                                       'finish_reason': sample['finish_reason'],
                                       'run_bundle': sample['run_bundle']}
        if set(by_seed) != set(seeds):
            raise ValueError('sample seeds differ from the frozen schedule')
        by_prompt[prompt['prompt_id']] = {'prompt_sha256': prompt['prompt_sha256'],
                                          'by_seed': by_seed}
    return {'capture_sha256': capture_sha256,
            'model_sha256': capture['model_sha256'],
            'sampler_policy_sha256': capture['sampler_policy_sha256'],
            'sampler_observer_sha256': capture['sampler_observer_sha256'],
            'vocabulary_size': vocab, 'eos_token_id': eos,
            'max_output_tokens': maximum, 'seed_schedule': seeds,
            'prompts': by_prompt}


def categorical_metrics(left, right):
    total = len(left)
    left_counts, right_counts = {}, {}
    for token in left:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right:
        right_counts[token] = right_counts.get(token, 0) + 1
    support = set(left_counts) | set(right_counts)
    p = {token: left_counts.get(token, 0) / total for token in support}
    q = {token: right_counts.get(token, 0) / total for token in support}
    tv = math.fsum(abs(p[token] - q[token]) for token in support) / 2
    js = 0.0
    for token in support:
        midpoint = (p[token] + q[token]) / 2
        if p[token]:
            js += .5 * p[token] * math.log2(p[token] / midpoint)
        if q[token]:
            js += .5 * q[token] * math.log2(q[token] / midpoint)
    return {'total_variation': tv, 'jensen_shannon_bits': js,
            'left_support_size': len(left_counts), 'right_support_size': len(right_counts)}


def resolve_bundle_path(bundle_root, raw_path):
    root = Path(bundle_root).absolute()
    if root.resolve(strict=True) != root:
        raise ValueError('stochastic capture directory cannot traverse a symlink')
    root_info = root.lstat()
    if (stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != 0o700 or root_info.st_uid != os.getuid()):
        raise ValueError('stochastic capture parent must be an owned private directory')
    supplied = Path(raw_path)
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(root)
        except ValueError as error:
            raise ValueError('supervisor bundle path escapes the stochastic capture directory') from error
    else:
        relative = supplied
    if not relative.parts or any(part in ('', '.', '..') for part in relative.parts):
        raise ValueError('invalid supervisor bundle path')
    resolved = root
    for part in relative.parts:
        resolved = resolved / part
        if resolved.is_symlink():
            raise ValueError('supervisor bundle path contains a symlink')
    return resolved


def verify_run_bindings(capture, bundle_root):
    bundle_paths, child_instances, executables, bindings = set(), set(), set(), []
    verified = 0
    for prompt_id, prompt in capture['prompts'].items():
        for seed, sample in prompt['by_seed'].items():
            path = resolve_bundle_path(bundle_root, sample['run_bundle'])
            if path in bundle_paths:
                raise ValueError('each stochastic seed requires a distinct supervisor bundle')
            bundle_paths.add(path)
            summary = verify_bundle(path)
            if summary['run_status'] != 'successful':
                raise ValueError('stochastic supervisor bundle did not complete successfully')
            manifest = secure_json(path / 'manifest.json')
            result = secure_json(path / 'result.json')
            expected_identity = {
                'schema': 'rosevine-stochastic-sample-v2',
                'model_sha256': capture['model_sha256'],
                'sampler_policy_sha256': capture['sampler_policy_sha256'],
                'sampler_observer_sha256': capture['sampler_observer_sha256'],
                'prompt_id': prompt_id,
                'prompt_sha256': prompt['prompt_sha256'],
                'seed': seed,
                'vocabulary_size': capture['vocabulary_size'],
                'eos_token_id': capture['eos_token_id'],
                'max_output_tokens': capture['max_output_tokens']}
            if manifest.get('run_identity') != {'stochastic_sample': expected_identity}:
                raise ValueError('stochastic sample identity differs from supervisor manifest')
            artifact_checks = result.get('artifact_checks')
            expected_artifacts = {'target_model': capture['model_sha256'],
                                  'sample_observer': capture['sampler_observer_sha256']}
            if not isinstance(artifact_checks, list) or any(not any(
                    isinstance(item, dict) and item.get('role') == role
                    and item.get('status') == 'matched'
                    and item.get('expected_sha256') == expected_hash
                    and item.get('actual_sha256') == expected_hash
                    for item in artifact_checks) for role, expected_hash in expected_artifacts.items()):
                raise ValueError('stochastic run lacks matched model and observer prelaunch checks')
            child, cleanup = result.get('child_process_identity'), result.get('workload_cleanup')
            if (not isinstance(child, dict) or child.get('reason') is not None
                    or not isinstance(child.get('sha256'), str)
                    or not isinstance(cleanup, dict) or cleanup.get('pending') is not False
                    or cleanup.get('returncode') != 0 or cleanup.get('pid') != child.get('pid')
                    or result.get('returncode') != 0 or result.get('launched') is not True):
                raise ValueError('stochastic sample lacks successful child identity and cleanup')
            instance = (child.get('pid'), child.get('start_ticks'))
            if instance in child_instances:
                raise ValueError('stochastic samples reused a child process instance')
            child_instances.add(instance)
            executables.add(child['sha256'])
            stderr = secure_bytes(path / 'stderr.txt', 8 * 1024 * 1024)
            receipt = parse_sampler_events(stderr, vocabulary_size=capture['vocabulary_size'],
                                           eos_token_id=capture['eos_token_id'],
                                           max_output_tokens=capture['max_output_tokens'])
            if receipt != {'token_ids': sample['token_ids'], 'finish_reason': sample['finish_reason']}:
                raise ValueError('stochastic token IDs do not match checksummed sampler events')
            bindings.append({'prompt_id': prompt_id, 'seed': seed,
                             'checksums_index_sha256': summary['checksum_index_sha256'],
                             'process_instance_sha256': hashlib.sha256(
                                 f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest(),
                             'executable_sha256': child['sha256']})
            verified += 1
    if len(executables) != 1:
        raise ValueError('stochastic samples used inconsistent executables')
    return {'verified_supervisor_bundles': verified,
            'distinct_child_processes': len(child_instances),
            'common_executable_sha256': next(iter(executables)),
            'bundle_paths': bundle_paths, 'bindings': bindings}


def compare(reference, candidate):
    reference_path, candidate_path = Path(reference), Path(candidate)
    left, right = load_capture(reference_path), load_capture(candidate_path)
    for key in ('sampler_policy_sha256', 'sampler_observer_sha256', 'vocabulary_size', 'eos_token_id',
                'max_output_tokens', 'seed_schedule'):
        if left[key] != right[key]:
            raise ValueError(f'paired stochastic capture {key} mismatch')
    if left['prompts'].keys() != right['prompts'].keys():
        raise ValueError('paired stochastic prompt inventory mismatch')
    for prompt_id in left['prompts']:
        if left['prompts'][prompt_id]['prompt_sha256'] != right['prompts'][prompt_id]['prompt_sha256']:
            raise ValueError('paired stochastic prompt bytes differ')
    left_bindings = verify_run_bindings(left, reference_path.absolute().parent)
    right_bindings = verify_run_bindings(right, candidate_path.absolute().parent)
    if left_bindings['bundle_paths'] & right_bindings['bundle_paths']:
        raise ValueError('reference and candidate stochastic runs must use distinct bundles')
    prompt_results = []
    for prompt_id in sorted(left['prompts']):
        ref_prompt, cand_prompt = left['prompts'][prompt_id], right['prompts'][prompt_id]
        positions = []
        for position in range(left['max_output_tokens']):
            ref_tokens, cand_tokens = [], []
            for seed in left['seed_schedule']:
                ref_sample = ref_prompt['by_seed'][seed]
                cand_sample = cand_prompt['by_seed'][seed]
                ref_sequence, cand_sequence = ref_sample['token_ids'], cand_sample['token_ids']
                ref_tokens.append(ref_sequence[position] if position < len(ref_sequence)
                                  else left['eos_token_id'])
                cand_tokens.append(cand_sequence[position] if position < len(cand_sequence)
                                   else left['eos_token_id'])
            metric = categorical_metrics(ref_tokens, cand_tokens)
            metric.update({'position': position,
                           'reference_eos_rate': ref_tokens.count(left['eos_token_id']) / len(ref_tokens),
                           'candidate_eos_rate': cand_tokens.count(left['eos_token_id']) / len(cand_tokens),
                           'seed_count': len(left['seed_schedule'])})
            positions.append(metric)
        prompt_results.append({'prompt_id': prompt_id,
                               'prompt_sha256': ref_prompt['prompt_sha256'],
                               'positions': positions,
                               'mean_position_tv': math.fsum(x['total_variation'] for x in positions) / len(positions),
                               'mean_position_js_bits': math.fsum(x['jensen_shannon_bits'] for x in positions) / len(positions)})
    return {'schema': 'rosevine-stochastic-distribution-v3',
            'status': 'descriptive; no acceptance decision',
            'reference_capture_sha256': left['capture_sha256'],
            'candidate_capture_sha256': right['capture_sha256'],
            'reference_model_sha256': left['model_sha256'],
            'candidate_model_sha256': right['model_sha256'],
            'sampler_policy_sha256': left['sampler_policy_sha256'],
            'seed_count_per_prompt': len(left['seed_schedule']),
            'prompt_count': len(prompt_results),
            'verified_supervisor_bundles': left_bindings['verified_supervisor_bundles'] + right_bindings['verified_supervisor_bundles'],
            'distinct_child_processes': left_bindings['distinct_child_processes'] + right_bindings['distinct_child_processes'],
            'reference_run_bindings': left_bindings['bindings'],
            'candidate_run_bindings': right_bindings['bindings'],
            'aggregate_equal_prompt_mean_tv': math.fsum(x['mean_position_tv'] for x in prompt_results) / len(prompt_results),
            'aggregate_equal_prompt_mean_js_bits': math.fsum(x['mean_position_js_bits'] for x in prompt_results) / len(prompt_results),
            'prompts': prompt_results,
            'sampler_observer_sha256': left['sampler_observer_sha256'],
            'provenance': 'sampled token events match checksummed supervisor stderr and run identity; target model and observer hashes matched prelaunch expected-artifact checks; model hash does not prove runtime-loaded model bytes',
            'limitations': 'position-wise empirical marginals do not establish sequence-level distribution equivalence; no tolerance or acceptance threshold is applied'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True,
                        help='private rosevine-stochastic-output-v1 capture')
    parser.add_argument('--candidate', type=Path, required=True,
                        help='paired private capture with the same prompt/seed/policy plan')
    parser.add_argument('--output', type=Path, required=True,
                        help='new file in a private 0700 directory')
    args = parser.parse_args()
    write_private_json(args.output, compare(args.reference, args.candidate))
    print('Private sampled-token distribution summary written; no acceptance asserted.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Stochastic distribution comparison failed; no acceptance inferred.', file=sys.stderr)
        sys.exit(1)
