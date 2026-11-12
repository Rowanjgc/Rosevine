"""Stable full-vocabulary reduction, target alignment and document-paired NLL."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('nll_compare', ROOT / 'scripts/nll_compare.py')
N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(N)


def fixture():
    docs = []
    for identifier, count, stratum, split in [('a', 2, 'prose', 'holdout'), ('b', 8, 'code', 'holdout'),
                                              ('c', 2, 'prose', 'calibration')]:
        tokens = [1] * (count + 1)
        docs.append({'id': identifier, 'stratum': stratum, 'split': split, 'token_ids': tokens,
                     'tokens_sha256': N.digest(tokens), 'scored_tokens': count, 'first_scored_target_index': 1})
    corpus = {'version': N.VERSION, 'documents': docs}
    corpus['corpus_sha256'] = N.digest(corpus)
    capture = {'schema': 'rosevine-nll-v1', 'status': 'successful', 'corpus_sha256': corpus['corpus_sha256'],
               'identity_sha256': 'a' * 64, 'split': 'holdout', 'logit_stage': 'post-softcap',
               'documents': [{'id': d['id'], 'tokens_sha256': d['tokens_sha256'],
                              'nll': [1.0] * d['scored_tokens']} for d in docs if d['split'] == 'holdout']}
    return corpus, capture


class NLLTests(unittest.TestCase):
    def test_stable_full_vocabulary_target_nll(self):
        self.assertAlmostEqual(N.target_nll([0, math.log(3)], 0), math.log(4))
        self.assertAlmostEqual(N.target_nll([10000, 10000 + math.log(3)], 1), math.log(4 / 3), places=11)
        for logits, target in [([0, 1], 2), ([0, 1], True), ([float('nan')], 0)]:
            with self.assertRaises(ValueError):
                N.target_nll(logits, target)

    def test_token_denominators_and_fixed_stratum_bootstrap(self):
        corpus, reference = fixture()
        candidate = copy.deepcopy(reference)
        candidate['identity_sha256'] = 'b' * 64
        candidate['documents'][0]['nll'] = [1.1] * 2
        candidate['documents'][1]['nll'] = [1.3] * 8
        result = N.compare(corpus, reference, candidate, seed=7, resamples=1000)
        self.assertEqual(result['scored_tokens'], 10)
        self.assertAlmostEqual(result['mean_nll_increase_nats'], .26)
        for bound in result['paired_ci95_nats']:
            self.assertAlmostEqual(bound, .26)
        self.assertEqual(result['bootstrap']['stratum_weights'], {'prose': .2, 'code': .8})
        self.assertFalse(result['minimum_100k_holdout_met'])
        self.assertFalse(result['run_bundle_provenance_verified'])
        self.assertIsNone(result['run_bundle_provenance'])
        self.assertEqual(result, N.compare(corpus, reference, candidate, seed=7, resamples=1000))

    def test_missing_wrong_split_stage_identity_and_target_counts_fail(self):
        corpus, reference = fixture()
        cases = []
        candidate = copy.deepcopy(reference);candidate['documents'].pop();cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['split'] = 'calibration';cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['logit_stage'] = 'pre-softcap';cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['documents'][0]['tokens_sha256'] = 'c' * 64;cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['documents'][0]['nll'].pop();cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['documents'][0]['nll'][0] = float('inf');cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['status'] = 'incomplete';cases.append(candidate)
        candidate = copy.deepcopy(reference);candidate['documents'].append(candidate['documents'][0]);cases.append(candidate)
        for candidate in cases:
            with self.assertRaises(ValueError):
                N.compare(corpus, reference, candidate, resamples=1000)

    def test_corpus_tampering_and_perplexity_overflow(self):
        corpus, reference = fixture()
        corpus['documents'][0]['token_ids'][0] = 2
        with self.assertRaises(ValueError):
            N.compare(corpus, reference, reference, resamples=1000)
        value = N.perplexity(1000)
        self.assertIsNone(value['value'])
        self.assertTrue(value['reason'])
