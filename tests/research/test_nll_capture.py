"""Private per-token NLL input framing and supervised-output finalization."""
import json
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_core as B
import fidelity_corpus as F
import nll_capture as N
import nll_compare as A


def corpus():
    documents = []
    for identifier, tokens in [('doc-a', [1, 2, 3]), ('doc-b', [1, 4, 5, 6])]:
        documents.append({'id': identifier, 'stratum': 'problem-prose', 'split': 'calibration',
                          'text': 'private synthetic fixture', 'text_sha256': 'a' * 64,
                          'dedup_sha256': 'b' * 64, 'token_ids': tokens,
                          'tokens_sha256': B.digest(tokens), 'scored_tokens': len(tokens) - 1,
                          'first_scored_target_index': 1})
    record = {'version': F.VERSION, 'documents': documents,
              'corpus_sha256': None}
    record['corpus_sha256'] = B.digest({k: v for k, v in record.items() if k != 'corpus_sha256'})
    return record


def save(path, raw):
    path.write_bytes(raw)
    os.chmod(path, 0o600)


SUCCESS = {'status': 'successful', 'returncode': 0, 'launched': True,
           'child_process_identity': {'reason': None, 'pid': 321, 'start_ticks': 654,
                                      'sha256': 'd' * 64},
           'workload_cleanup': {'pending': False, 'pid': 321, 'returncode': 0}}


class NLLCaptureTests(unittest.TestCase):
    def prepared(self, root):
        source = root / 'corpus-source'
        source.mkdir(mode=0o700)
        B.write_private_json(source / 'corpus.json', corpus())
        prepared = root / 'prepared'
        plan = N.prepare(source / 'corpus.json', 'calibration', 'a' * 64, 'b' * 64,
                         prepared, vocabulary=16, capacity_tokens=256)
        self.assertEqual(plan['target_count'], 5)
        self.assertEqual(plan['output_bytes'], 40)
        binary = (prepared / 'inputs.bin').read_bytes()
        self.assertEqual(binary[:20], b'RVNLL001' + struct.pack('<III', 16, 256, 2))
        return prepared

    def native(self, root, values):
        directory = root / 'native'
        directory.mkdir(mode=0o700)
        save(directory / 'started', N.STARTED)
        save(directory / 'complete', ('%d\n' % len(values)).encode())
        save(directory / 'nll-000000.f64', struct.pack('<%dd' % len(values), *values))
        return directory

    def test_prepares_exact_private_tokens_and_finalizes_document_aligned_nats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = self.prepared(root)
            native = self.native(root, [0.1, 0.2, 0.3, 0.4, 0.5])
            result = N.finalize(native, prepared, 'c' * 64, SUCCESS)
            self.assertEqual(result['schema'], 'rosevine-nll-v1')
            self.assertEqual(result['split'], 'calibration')
            self.assertEqual([len(row['nll']) for row in result['documents']], [2, 3])
            self.assertEqual(result['documents'][0]['nll'], [0.1, 0.2])
            self.assertEqual(result['documents'][1]['nll'], [0.3, 0.4, 0.5])
            expected = A.validate_corpus(corpus(), 'calibration')
            aligned = A.validate_capture(result, corpus()['corpus_sha256'], expected, 'calibration')
            self.assertEqual(aligned['doc-a'], [0.1, 0.2])
            self.assertTrue((native / 'provenance.json').is_file())
            binding = json.loads((native / 'capture-run-binding.json').read_text())
            self.assertEqual(binding['schema'], 'rosevine-nll-capture-run-v1')
            self.assertEqual(binding['capture_sha256'], hashlib.sha256(
                (native / 'capture.json').read_bytes()).hexdigest())
            self.assertEqual((native / 'capture.json').stat().st_mode & 0o777, 0o600)

    def test_incomplete_supervision_bad_receipt_nonfinite_and_extra_files_reject(self):
        failures = [
            lambda n: (SUCCESS | {'status': 'incomplete'}, [0.1] * 5),
            lambda n: (SUCCESS, [0.1] * 4),
            lambda n: (SUCCESS, [0.1, 0.2, float('nan'), 0.4, 0.5]),
            lambda n: (SUCCESS, [0.1] * 5, True),
        ]
        for create in failures:
            with self.subTest(case=failures.index(create)), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prepared = self.prepared(root)
                args = create(root)
                supervision, values = args[:2]
                native = self.native(root, values)
                if len(args) == 3:
                    save(native / 'unexpected', b'x')
                elif values == [0.1] * 4:
                    save(native / 'complete', b'5\n')
                with self.assertRaises(ValueError):
                    N.finalize(native, prepared, 'c' * 64, supervision)
                self.assertFalse((native / 'capture.json').exists())

    def test_capture_binding_verifies_against_checksums_and_exact_run_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = self.prepared(root)
            plan = json.loads((prepared / 'plan.json').read_text())
            native = self.native(root, [0.1, 0.2, 0.3, 0.4, 0.5])
            run_identity = {'identity_sha256': 'c' * 64, 'nll_plan_sha256': plan['plan_sha256'],
                            'inputs_sha256': plan['inputs_sha256'],
                            'corpus_sha256': plan['inputs']['corpus_sha256'],
                            'split': 'calibration', 'target_count': 5,
                            'logit_stage': plan['logit_stage']}
            run = root / 'run'
            supervision = B.supervise([sys.executable, '-c', 'pass'], run, timeout_s=2,
                                      run_identity=run_identity)
            N.finalize(native, prepared, 'c' * 64, supervision)
            result = A.verify_run_binding(native / 'capture.json', run)
            self.assertTrue(result['verified'])
            binding = json.loads((native / 'capture-run-binding.json').read_text())
            binding['capture_sha256'] = '0' * 64
            (native / 'capture-run-binding.json').write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError, 'binding is missing or mismatched'):
                A.verify_run_binding(native / 'capture.json', run)

    def test_invalid_vocab_target_count_and_unreaped_status_reject(self):
        base = {'schema': 'rosevine-nll-inputs-v1', 'corpus_sha256': 'a' * 64,
                'model_sha256': 'b' * 64, 'tokenizer_sha256': 'c' * 64,
                'split': 'calibration', 'vocabulary': 16, 'capacity_tokens': 256,
                'position_policy': 'zero-based causal; row j predicts token j+1',
                'documents': [{'id': 'doc', 'token_ids': [1, 2],
                               'tokens_sha256': B.digest([1, 2]), 'scored_tokens': 1}]}
        for field, value in [('vocabulary', 0), ('capacity_tokens', 257),
                             ('position_policy', 'unknown')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                N.encode_inputs({**base, field: value})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = self.prepared(root)
            native = self.native(root, [0.1] * 5)
            bad = {**SUCCESS, 'workload_cleanup': {'pending': True, 'returncode': None}}
            with self.assertRaises(ValueError):
                N.finalize(native, prepared, 'c' * 64, bad)


if __name__ == '__main__':
    unittest.main()
