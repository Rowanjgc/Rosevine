#!/usr/bin/env python3
"""Document-paired NLL analysis over a frozen private corpus and complete captures.

Capture schema rosevine-nll-v1: status, corpus_sha256, identity_sha256, split,
logit_stage, and documents [{id, tokens_sha256, nll}]. nll is ordered per-target
negative log probability in nats for every token after context token zero.
No model execution, candidate selection, unpaired token bootstrap or automatic GO.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

from benchmark_core import digest, percentile, write_private_json
from correctness_oracle import finite_vector
from fidelity_corpus import VERSION
from verify_capture_bundle import secure_bytes, secure_json, verify_bundle


def verify_run_binding(capture_path, run_bundle):
    """Bind one finalized NLL capture to its successful retained supervisor run."""
    capture_path, run_bundle = Path(capture_path), Path(run_bundle)
    capture_raw = secure_bytes(capture_path)
    capture = secure_json(capture_path)
    binding = secure_json(capture_path.parent / 'capture-run-binding.json', 16384)
    provenance = secure_json(capture_path.parent / 'provenance.json', 16384)
    if (not isinstance(binding, dict) or set(binding) != {
            'schema', 'capture_sha256', 'supervisor_result_sha256',
            'process_instance_sha256', 'executable_sha256'}
            or binding['schema'] != 'rosevine-nll-capture-run-v1'
            or binding['capture_sha256'] != hashlib.sha256(capture_raw).hexdigest()):
        raise ValueError('NLL capture run binding is missing or mismatched')
    if (not isinstance(provenance, dict) or provenance.get('schema') != 'rosevine-nll-capture-provenance-v1'
            or provenance.get('identity_sha256') != capture.get('identity_sha256')):
        raise ValueError('NLL capture provenance is missing or mismatched')
    summary = verify_bundle(run_bundle)
    if summary['run_status'] != 'successful':
        raise ValueError('NLL supervisor bundle did not complete successfully')
    manifest = secure_json(run_bundle / 'manifest.json')
    identity = manifest.get('run_identity')
    expected = {'identity_sha256': provenance['identity_sha256'],
                'nll_plan_sha256': provenance['plan_sha256'],
                'inputs_sha256': provenance['inputs_sha256'],
                'corpus_sha256': provenance['corpus_sha256'],
                'split': provenance['split'], 'target_count': provenance['target_count'],
                'logit_stage': provenance['logit_stage']}
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError('NLL capture identity differs from supervisor run manifest')
    result_raw = secure_bytes(run_bundle / 'result.json')
    result = secure_json(run_bundle / 'result.json')
    if hashlib.sha256(result_raw).hexdigest() != binding['supervisor_result_sha256']:
        raise ValueError('NLL capture binding differs from supervisor result')
    child, cleanup = result.get('child_process_identity'), result.get('workload_cleanup')
    if (result.get('status') != 'successful' or result.get('launched') is not True
            or result.get('returncode') != 0 or not isinstance(child, dict)
            or child.get('reason') is not None or type(child.get('pid')) is not int
            or type(child.get('start_ticks')) is not int or child.get('sha256') != binding['executable_sha256']
            or not isinstance(cleanup, dict) or cleanup.get('pending') is not False
            or cleanup.get('pid') != child['pid'] or cleanup.get('returncode') != 0):
        raise ValueError('NLL supervisor bundle lacks successful reaped child identity')
    process = hashlib.sha256(f"{child['pid']}:{child['start_ticks']}".encode('ascii')).hexdigest()
    if process != binding['process_instance_sha256']:
        raise ValueError('NLL capture binding disagrees with supervised process identity')
    return {'verified': True, 'supervisor_result_sha256': binding['supervisor_result_sha256'],
            'process_instance_sha256': process, 'executable_sha256': binding['executable_sha256'],
            'mapped_artifact_identity': summary['mapped_artifact_identity']}


def target_nll(logits, target):
    """Stable full-vocabulary normalization; binary64 analysis reference."""
    finite_vector(logits)
    if type(target) is not int or not 0 <= target < len(logits):
        raise ValueError('target token outside vocabulary')
    peak = max(logits)
    value = (peak - logits[target]) + math.log(math.fsum(math.exp(x - peak) for x in logits))
    if not math.isfinite(value) or value < 0:
        raise ValueError('nonfinite or negative target NLL')
    return value


def validate_corpus(corpus, split):
    if not isinstance(corpus, dict) or corpus.get('version') != VERSION or split not in ('calibration', 'holdout'):
        raise ValueError('versioned corpus and explicit split required')
    if digest({k: v for k, v in corpus.items() if k != 'corpus_sha256'}) != corpus.get('corpus_sha256'):
        raise ValueError('corpus checksum mismatch')
    expected, ids = {}, set()
    for document in corpus['documents']:
        identifier = document['id']
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('duplicate/invalid corpus document ID')
        ids.add(identifier)
        tokens = document['token_ids']
        if not isinstance(tokens, list) or len(tokens) < 2 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('invalid corpus token IDs')
        if digest(tokens) != document['tokens_sha256'] or document['scored_tokens'] != len(tokens) - 1 or document['first_scored_target_index'] != 1:
            raise ValueError('corpus token hash/count/target alignment mismatch')
        if document['split'] not in ('calibration', 'holdout') or not isinstance(document['stratum'], str) or not document['stratum']:
            raise ValueError('invalid corpus split/stratum')
        if document['split'] == split:
            expected[identifier] = document
    if not expected:
        raise ValueError('empty selected corpus split')
    return expected


def validate_capture(capture, corpus_sha, expected, split):
    keys = {'schema', 'status', 'corpus_sha256', 'identity_sha256', 'split', 'logit_stage', 'documents'}
    if not isinstance(capture, dict) or set(capture) != keys or capture['schema'] != 'rosevine-nll-v1':
        raise ValueError('invalid NLL capture schema')
    if capture['status'] != 'successful' or capture['corpus_sha256'] != corpus_sha or capture['split'] != split:
        raise ValueError('incomplete, mismatched or wrong-split NLL capture')
    sha = capture['identity_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('run identity hash required')
    if capture['logit_stage'] not in ('pre-softcap', 'post-softcap') or not isinstance(capture['documents'], list):
        raise ValueError('explicit logit stage and document list required')
    documents = {}
    for row in capture['documents']:
        if not isinstance(row, dict) or set(row) != {'id', 'tokens_sha256', 'nll'} or row['id'] not in expected or row['id'] in documents:
            raise ValueError('unknown/duplicate NLL document')
        if row['tokens_sha256'] != expected[row['id']]['tokens_sha256']:
            raise ValueError('NLL document token identity mismatch')
        nll = row['nll']
        if not isinstance(nll, list) or len(nll) != expected[row['id']]['scored_tokens'] or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in nll):
            raise ValueError('NLL target count or finite nonnegative nats mismatch')
        documents[row['id']] = nll
    if documents.keys() != expected.keys():
        raise ValueError('missing NLL corpus documents; do not compare a selected subset')
    return documents


def perplexity(mean):
    try:
        return {'value': math.exp(mean), 'reason': None}
    except OverflowError:
        return {'value': None, 'reason': 'perplexity exceeds binary64 range; NLL remains available'}


def compare(corpus, reference, candidate, *, split='holdout', seed=20261111, resamples=10000):
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 1000:
        raise ValueError('nonnegative bootstrap seed and >=1000 resamples required')
    expected = validate_corpus(corpus, split)
    ref = validate_capture(reference, corpus['corpus_sha256'], expected, split)
    cand = validate_capture(candidate, corpus['corpus_sha256'], expected, split)
    if reference['logit_stage'] != candidate['logit_stage']:
        raise ValueError('pre/post-softcap comparison mismatch')
    strata, rows = {}, []
    for identifier, document in expected.items():
        n = document['scored_tokens']
        row = {'id': identifier, 'stratum': document['stratum'], 'tokens': n,
               'reference_nll_sum': math.fsum(ref[identifier]),
               'candidate_nll_sum': math.fsum(cand[identifier]),
               'delta_nll_sum': math.fsum(c - r for r, c in zip(ref[identifier], cand[identifier]))}
        rows.append(row)
        strata.setdefault(row['stratum'], []).append(row)
    tokens = sum(row['tokens'] for row in rows)
    reference_mean = math.fsum(row['reference_nll_sum'] for row in rows) / tokens
    candidate_mean = math.fsum(row['candidate_nll_sum'] for row in rows) / tokens
    delta = math.fsum(row['delta_nll_sum'] for row in rows) / tokens
    weights = {name: sum(row['tokens'] for row in group) / tokens for name, group in strata.items()}
    rng, bootstrap = random.Random(seed), []
    # Resample whole paired documents within each stratum, keeping the frozen
    # corpus's token-weighted stratum mixture. Never treat tokens as IID samples.
    for _ in range(resamples):
        changes = []
        for name in sorted(strata):
            group = rng.choices(strata[name], k=len(strata[name]))
            changes.append(weights[name] * math.fsum(row['delta_nll_sum'] for row in group) /
                           sum(row['tokens'] for row in group))
        bootstrap.append(math.fsum(changes))
    return {'schema': 'rosevine-nll-analysis-v1', 'status': 'complete',
            'corpus_sha256': corpus['corpus_sha256'], 'split': split,
            'reference_identity_sha256': reference['identity_sha256'],
            'candidate_identity_sha256': candidate['identity_sha256'],
            'logit_stage': reference['logit_stage'], 'documents': len(rows), 'scored_tokens': tokens,
            'minimum_100k_holdout_met': split == 'holdout' and tokens >= 100000,
            'reference_mean_nll_nats': reference_mean, 'candidate_mean_nll_nats': candidate_mean,
            'reference_perplexity': perplexity(reference_mean), 'candidate_perplexity': perplexity(candidate_mean),
            'mean_nll_increase_nats': delta,
            'paired_ci95_nats': [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            'bootstrap': {'seed': seed, 'resamples': resamples,
                          'unit': 'paired document, stratified; frozen token-weighted stratum mixture',
                          'stratum_weights': weights}, 'per_document': rows,
            'run_bundle_provenance_verified': False, 'run_bundle_provenance': None,
            'acceptance': 'not decided; reviewed corpus/margins and reference variability required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-run-bundle', type=Path)
    parser.add_argument('--candidate-run-bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['calibration', 'holdout'], default='holdout')
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if (args.reference_run_bundle is None) != (args.candidate_run_bundle is None):
        raise ValueError('supply both reference and candidate supervisor bundles')
    run_evidence = None
    if args.reference_run_bundle is not None:
        if args.reference_run_bundle.absolute() == args.candidate_run_bundle.absolute():
            raise ValueError('reference and candidate require distinct supervisor bundles')
        run_evidence = [verify_run_binding(args.reference, args.reference_run_bundle),
                        verify_run_binding(args.candidate, args.candidate_run_bundle)]
        if run_evidence[0]['process_instance_sha256'] == run_evidence[1]['process_instance_sha256']:
            raise ValueError('reference and candidate must be captured in distinct child processes')
    result = compare(json.loads(args.corpus.read_text()), json.loads(args.reference.read_text()),
                     json.loads(args.candidate.read_text()), split=args.split,
                     seed=args.seed, resamples=args.resamples)
    result['run_bundle_provenance_verified'] = run_evidence is not None
    result['run_bundle_provenance'] = run_evidence
    write_private_json(args.output, result)
    print('Private paired NLL analysis complete; no model quality acceptance asserted.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('NLL comparison failed; no quality result recorded.', file=sys.stderr)
        sys.exit(1)
