"""Document split, exact-code dedup, token quotas, replay and shortage behavior."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('fidelity_corpus', ROOT / 'scripts/fidelity_corpus.py')
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def sources():
    gsm = [{'question': 'Synthetic problem ' + str(i), 'answer': 'Synthetic reasoning ' + str(i)} for i in range(40)]
    mbpp = [{'task_id': i, 'text': 'Synthetic code task ' + str(i), 'code': 'def fn_' + str(i) + '(): return ' + str(i),
             'test_list': ['assert True'], 'test_setup_code': ''} for i in [11, 12, 13, 511, 512, 513]]
    return gsm, mbpp


class FidelityCorpusTests(unittest.TestCase):
    def test_whole_document_quotas_replay_and_split(self):
        gsm, mbpp = sources()
        corpus = F.build(gsm, mbpp, lambda text: [1] + list(text.encode()),
                         holdout_per_stratum=40, calibration_per_stratum=40)
        for split in corpus['counts'].values():
            self.assertTrue(all(count >= 40 for count in split.values()))
        for doc in corpus['documents']:
            self.assertEqual(doc['scored_tokens'], len(doc['token_ids']) - 1)
            if doc['id'].startswith('mbpp-'):
                self.assertEqual(doc['split'], 'calibration' if int(doc['id'][5:]) >= 511 else 'holdout')
        self.assertEqual(corpus, F.build(gsm, mbpp, lambda text: [1] + list(text.encode()),
                                         holdout_per_stratum=40, calibration_per_stratum=40))

    def test_duplicate_code_body_is_not_shared_across_calibration_holdout(self):
        gsm, mbpp = sources()
        mbpp[0]['code'] = mbpp[3]['code']
        docs = F.documents(gsm, mbpp)
        self.assertNotIn('mbpp-11', {doc['id'] for doc in docs})
        self.assertIn('mbpp-511', {doc['id'] for doc in docs})

    def test_shortage_replay_mismatch_and_overlong_docs_fail(self):
        gsm, mbpp = sources()
        with self.assertRaises(ValueError):
            F.build(gsm, mbpp, lambda text: [1, 2], holdout_per_stratum=50000)
        counter = 0

        def changing(text):
            nonlocal counter
            counter += 1
            return [1, counter]

        with self.assertRaises(ValueError):
            F.build(gsm, mbpp, changing)
        with self.assertRaises(ValueError):
            F.build(gsm, mbpp, lambda text: [1] * 100, max_document_tokens=32)
