"""Native input framing, row alignment, failure retention and index finalization."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import logit_capture as C
import logit_compare as L


def plan():
    sequences = []
    for label, length in [('first', 18), ('second', 17)]:
        ids = [i % 2 for i in range(length)]
        sequences.append({'id': label, 'token_ids': ids, 'tokens_sha256': C.digest(ids),
                          'reset_kv_before_sequence': True,
                          'scored_input_positions': [0, length - 2],
                          'scored_target_indices': [1, length - 1]})
    result = {'schema': 'rosevine-reference-positions-v1', 'sequences': sequences,
              'scored_positions': 33, 'source_split': 'calibration',
              'vocabulary': 2, 'capacity_tokens': 256}
    result['plan_sha256'] = C.digest(result)
    return result


def context_plan():
    positions = [2, 512, 598]
    streams = []
    for i in range(3):
        ids = [(i * 10 + j + 1) % 32 for j in range(600)]
        streams.append({'id': f'context-{i}', 'token_ids': ids, 'tokens_sha256': C.digest(ids),
                        'reset_kv_before_sequence': True, 'capture_input_positions': positions,
                        'scored_target_indices': [p + 1 for p in positions],
                        'target_token_ids': [ids[p + 1] for p in positions]})
    result = {'schema': 'rosevine-context-positions-v1', 'streams': streams,
              'captured_input_positions': positions, 'source_split': 'calibration',
              'capture_rows': 9, 'vocabulary': 32, 'capacity_tokens': 1024}
    result['plan_sha256'] = C.digest(result)
    return result


def save(path, raw):
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def native_output(directory):
    directory.mkdir(mode=0o700)
    save(directory / 'started', C.STARTED)
    save(directory / 'complete', b'33\n')
    save(directory / 'logits-000000.f32', b''.join(struct.pack('<ff', i, -i) for i in range(32)))
    save(directory / 'logits-000001.f32', struct.pack('<ff', 32, -32))


def native_selected_output(directory):
    directory.mkdir(mode=0o700)
    save(directory / 'started', C.STARTED)
    save(directory / 'complete', b'9\n')
    rows = [struct.pack('<32f', *[float(row * 100 + token) for token in range(32)]) for row in range(9)]
    save(directory / 'logits-000000.f32', b''.join(rows))


SUCCESS = {'status': 'successful', 'returncode': 0, 'launched': True,
           'child_process_identity': {'pid': 101, 'start_ticks': 77, 'sha256': 'd' * 64,
                                      'reason': None},
           'workload_cleanup': {'pending': False, 'returncode': 0, 'pid': 101}}


class CaptureTests(unittest.TestCase):
    def test_native_input_bytes_and_cross_sequence_shard_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = C.prepare(plan(), 'a' * 64, 'b' * 64, root / 'prepared', extra_buffers=False)
            raw = (root / 'prepared/inputs.bin').read_bytes()
            self.assertEqual(raw[:24], b'RVLOG003' + struct.pack('<IIII', 2, 256, 2, 0))
            self.assertEqual(raw[24:28], struct.pack('<I', 18))
            self.assertEqual(len(raw), 24 + 8 + 35 * 4)
            native_output(root / 'rows')
            index = C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)
            self.assertEqual(index['inputs_sha256'], C.digest(manifest))
            self.assertEqual(index['schema'], 'rosevine-logits-v4')
            self.assertFalse(index['native_policy']['extra_buffers'])
            self.assertEqual(index['native_policy_sha256'], C.digest(index['native_policy']))
            prepared = json.loads((root / 'prepared/prepared.json').read_text())
            self.assertEqual(prepared['supervisor_run_identity_fields'], {
                'inputs_sha256': index['inputs_sha256'],
                'position_plan_sha256': plan()['plan_sha256'],
                'expected_positions_sha256': index['expected_positions_sha256'],
                'expected_position_count': index['expected_position_count'],
                'native_policy_sha256': index['native_policy_sha256']})
            binding = json.loads((root / 'rows/capture-run-binding.json').read_text())
            self.assertEqual(binding['schema'], 'rosevine-logit-capture-run-v1')
            self.assertEqual(binding['index_sha256'], __import__('hashlib').sha256(
                (root / 'rows/index.json').read_bytes()).hexdigest())
            self.assertNotIn('pid', binding)
            self.assertEqual(index['shards'][0]['positions'][17], ['second', 0])
            self.assertEqual(index['shards'][1]['positions'], [['second', 15]])
            values = list(L.rows(root / 'rows/index.json', L.load_index(root / 'rows/index.json')))
            self.assertEqual(values[-1][0], ('second', 15))
            self.assertEqual(list(values[-1][1]), [32, -32])
            self.assertEqual(L.compare(root / 'rows/index.json', root / 'rows/index.json')['mean_kl_nats'], 0)
            with self.assertRaises(ValueError):
                C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)

    def test_failed_or_unreaped_supervisor_cannot_finalize(self):
        for result in [{}, {**SUCCESS, 'status': 'incomplete'},
                       {**SUCCESS, 'returncode': 1}, {**SUCCESS, 'launched': False},
                       {**SUCCESS, 'workload_cleanup': {'pending': True, 'returncode': None}}]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                C.finalize('unused', 'unused', 'c' * 64, result)

    def test_context_plan_encodes_only_selected_rows_after_full_context_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = C.prepare(context_plan(), 'a' * 64, 'b' * 64, root / 'prepared', extra_buffers=False)
            raw = (root / 'prepared/inputs.bin').read_bytes()
            self.assertEqual(raw[:24], b'RVLOG003' + struct.pack('<IIII', 32, 1024, 3, 1))
            self.assertEqual(manifest['sequences'][0]['capture_input_positions'], [2, 512, 598])
            native_selected_output(root / 'rows')
            index = C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)
            expected = [[f'context-{sequence}', position]
                        for sequence in range(3) for position in (2, 512, 598)]
            self.assertEqual([row for shard in index['shards'] for row in shard['positions']], expected)
            self.assertEqual(len(list(L.rows(root / 'rows/index.json', L.load_index(root / 'rows/index.json')))), 9)

    def test_context_plan_rejects_invalid_selected_rows_before_preparation(self):
        for fault in ('duplicate', 'unordered', 'past-target', 'wrong-target'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                record = context_plan()
                stream = record['streams'][0]
                if fault == 'duplicate':
                    stream['capture_input_positions'] = [2, 2, 598]
                elif fault == 'unordered':
                    stream['capture_input_positions'] = [512, 2, 598]
                elif fault == 'past-target':
                    stream['capture_input_positions'] = [2, 512, 600]
                elif fault == 'wrong-target':
                    stream['target_token_ids'] = [0, 0, 0]
                record['plan_sha256'] = C.digest({k: v for k, v in record.items() if k != 'plan_sha256'})
                with self.assertRaises(ValueError):
                    C.prepare(record, 'a' * 64, 'b' * 64, Path(temp) / 'prepared', extra_buffers=False)

    def test_partial_corrupt_nonfinite_or_unexpected_outputs_retained_without_index(self):
        for fault in ['missing-receipt', 'truncated', 'nonfinite', 'extra-file', 'public-mode', 'symlink']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                C.prepare(plan(), 'a' * 64, 'b' * 64, root / 'prepared', extra_buffers=False)
                native_output(root / 'rows')
                row = root / 'rows/logits-000001.f32'
                if fault == 'missing-receipt':
                    (root / 'rows/complete').unlink()
                elif fault == 'truncated':
                    row.write_bytes(b'')
                elif fault == 'nonfinite':
                    row.write_bytes(struct.pack('<ff', float('nan'), 0))
                elif fault == 'extra-file':
                    save(root / 'rows/unexpected', b'')
                elif fault == 'public-mode':
                    os.chmod(row, 0o644)
                elif fault == 'symlink':
                    row.unlink()
                    row.symlink_to('logits-000000.f32')
                with self.assertRaises((ValueError, OSError)):
                    C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)
                self.assertFalse((root / 'rows/index.json').exists())
                self.assertTrue((root / 'rows/started').exists())

    def test_changed_binary_input_and_prepared_digest_fail(self):
        for fault in ['binary', 'digest']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                C.prepare(plan(), 'a' * 64, 'b' * 64, root / 'prepared', extra_buffers=False)
                native_output(root / 'rows')
                if fault == 'binary':
                    path = root / 'prepared/inputs.bin'
                    path.write_bytes(path.read_bytes()[:-1] + b'\xff')
                else:
                    path = root / 'prepared/prepared.json'
                    record = json.loads(path.read_text())
                    record['inputs_sha256'] = 'd' * 64
                    path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)

    def test_extra_buffer_setting_is_explicit_and_bound_to_prepared_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            C.prepare(plan(), 'a' * 64, 'b' * 64, root / 'prepared', extra_buffers=True)
            raw = (root / 'prepared/inputs.bin').read_bytes()
            self.assertEqual(raw[:24], b'RVLOG003' + struct.pack('<IIII', 2, 256, 2, 2))
            record = json.loads((root / 'prepared/prepared.json').read_text())
            self.assertTrue(record['native_policy']['extra_buffers'])
            self.assertEqual(record['native_policy_sha256'], C.digest(record['native_policy']))
            native_output(root / 'rows')
            record['native_policy']['extra_buffers'] = False
            (root / 'prepared/prepared.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)

    def test_finalizer_rejects_digest_consistent_but_unsupported_native_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            C.prepare(plan(), 'a' * 64, 'b' * 64, root / 'prepared', extra_buffers=False)
            native_output(root / 'rows')
            path = root / 'prepared/prepared.json'
            record = json.loads(path.read_text())
            record['native_policy']['threads'] = 4
            record['native_policy_sha256'] = C.digest(record['native_policy'])
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'native policy'):
                C.finalize(root / 'rows', root / 'prepared', 'c' * 64, SUCCESS)

    def test_invalid_plan_and_outside_native_envelope_rejected_before_creation(self):
        for fault in ['checksum', 'positions', 'token', 'duplicate', 'capacity', 'reset']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                record = plan()
                if fault == 'positions':
                    record['sequences'][0]['scored_target_indices'] = [0, 16]
                elif fault == 'token':
                    record['sequences'][0]['token_ids'][0] = True
                    record['sequences'][0]['tokens_sha256'] = C.digest(record['sequences'][0]['token_ids'])
                elif fault == 'duplicate':
                    record['sequences'][1]['id'] = 'first'
                elif fault == 'capacity':
                    record['capacity_tokens'] = 5000
                elif fault == 'reset':
                    record['sequences'][0]['reset_kv_before_sequence'] = False
                record['plan_sha256'] = C.digest({k: v for k, v in record.items() if k != 'plan_sha256'})
                if fault == 'checksum':
                    record['plan_sha256'] = 'e' * 64
                directory = Path(temp) / 'prepared'
                with self.assertRaises(ValueError):
                    C.prepare(record, 'a' * 64, 'b' * 64, directory, extra_buffers=False)
                self.assertFalse(directory.exists())
