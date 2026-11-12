"""Private framed intermediate captures, integrity checks and paired metrics."""
import json
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import benchmark_core as B
import intermediate_capture as C


SUCCESS = {'status': 'successful', 'returncode': 0, 'launched': True,
           'child_process_identity': {'pid': 101, 'start_ticks': 11,
                                      'sha256': 'e' * 64, 'reason': None},
           'workload_cleanup': {'pending': False, 'returncode': 0, 'pid': 101}}


def prepared_inputs(root):
    inp = root / 'inputs'
    if inp.exists():
        return inp
    inp.mkdir(mode=0o700)
    manifest = {'schema': 'rosevine-logit-inputs-v1', 'model_sha256': 'a' * 64,
                'tokenizer_sha256': 'b' * 64, 'vocabulary': 64, 'capacity_tokens': 256,
                'reset_kv_before_sequence': True,
                'position_policy': 'zero-based causal; row j predicts input ID j+1',
                'sequences': [{'id': 'synthetic', 'token_ids': [2, 3, 4]}]}
    B.write_private_json(inp / 'inputs.json', manifest)
    with B.open_private(inp / 'inputs.bin') as f:
        f.write(C.encode_inputs(manifest, extra_buffers=False))
    return inp


def record(position, name, dtype, shape, values):
    encoded_name, encoded_type = name.encode(), dtype.encode()
    if dtype == 'f32':
        data = struct.pack('<' + 'f' * len(values), *values)
    else:
        data = struct.pack('<' + 'i' * len(values), *values)
    return (struct.pack('<I', position) + struct.pack('<H', len(encoded_name)) + encoded_name
            + struct.pack('<H', len(encoded_type)) + encoded_type + struct.pack('<I', len(shape))
            + struct.pack('<4Q', *(list(shape) + [1] * (4 - len(shape))))
            + struct.pack('<Q', len(data)) + data)


def capture(root, mode, logits=(1.0, 2.0, 3.0, 4.0), ids=(1, 3), weights=(0.25, 0.75),
            fault=None, expected_moe_layers=None):
    prep = C.prepare(prepared_inputs(root), mode, root / f'prepared-{mode}',
                     expected_moe_layers=expected_moe_layers)
    directory = root / f'capture-{mode}'
    directory.mkdir(mode=0o700)
    (directory / 'started').write_bytes(C.STARTED)
    (directory / 'started').chmod(0o600)
    values = [record(0, 'ffn_moe_logits-0', 'f32', [4], logits),
              record(0, 'ffn_moe_probs-0', 'f32', [2], (0.1, 0.9)),
              record(0, 'ffn_moe_topk-0', 'i32', [2], ids),
              record(0, 'ffn_moe_weights-0', 'f32', [2], weights),
              record(0, 'ffn_moe_gate_up-0', 'f32', [2], (0.5, 1.5)),
              record(0, 'ffn_moe_gelu-0', 'f32', [2], (0.2, 1.4)),
              record(0, 'ffn_moe_down-0', 'f32', [2], (0.3, 1.3)),
              record(0, 'ffn_moe_out-0', 'f32', [2], (0.4, 1.2)),
              record(0, 'ffn_moe-0', 'f32', [2], (0.6, 1.1)),
              record(0, 'ffn_moe_combined-0', 'f32', [2], (0.7, 1.0))]
    if fault == 'nonfinite':
        values[0] = record(0, 'ffn_moe_logits-0', 'f32', [4], (float('nan'), 2.0, 3.0, 4.0))
    if fault == 'duplicate':
        values.append(values[-1])
    raw = C.MAGIC + b''.join(values)
    with B.open_private(directory / 'nodes.bin') as f:
        f.write(raw)
    (directory / 'complete').write_bytes(struct.pack('<2I', len(values), 0))
    (directory / 'complete').chmod(0o600)
    return prep, directory


class IntermediateCaptureTests(unittest.TestCase):
    def test_prepare_binds_exact_input_and_mode_without_authorizing_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep = C.prepare(prepared_inputs(root), 'repack', root / 'prepared')
            manifest = json.loads((prep / 'prepared.json').read_text())
            self.assertEqual(manifest['mode'], 'repack')
            self.assertTrue(manifest['native_policy']['extra_buffers'])
            self.assertEqual(manifest['capture_position'], 0)
            self.assertIn('no model-run authority', manifest['authorization'])
            self.assertEqual((prep / 'inputs.bin').read_bytes(), C.encode_inputs(
                json.loads((prep / 'inputs.json').read_text()), extra_buffers=True))

    def test_finalize_and_compare_matched_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oprep, odir = capture(root, 'ordinary')
            rprep, rdir = capture(root, 'repack', logits=(1.0, 2.0005, 3.0, 4.0), weights=(0.2501, 0.7499))
            oi = C.finalize(odir, oprep, 'c' * 64, SUCCESS)
            ri = C.finalize(rdir, rprep, 'd' * 64, SUCCESS)
            self.assertNotEqual(oi['input_sha256'], ri['input_sha256'])
            self.assertEqual(oi['inputs_manifest_sha256'], ri['inputs_manifest_sha256'])
            self.assertEqual(len(oi['records']), 10)
            self.assertFalse(oi['native_policy']['extra_buffers'])
            self.assertTrue(ri['native_policy']['extra_buffers'])
            comparison = C.compare(odir, rdir)
            self.assertEqual(comparison['ordinary_input_binary_sha256'], oi['input_sha256'])
            self.assertEqual(comparison['repack_input_binary_sha256'], ri['input_sha256'])
            nodes = {item['name']: item for item in comparison['nodes']}
            self.assertTrue(nodes['ffn_moe_topk-0']['exact_match'])
            self.assertTrue(nodes['ffn_moe_logits-0']['proposed_M_pass'])
            self.assertTrue(comparison['router_boundary_margins'][0]['selected_ids_exact'])
            self.assertEqual(comparison['router_boundary_margins'][0]['ordinary_boundary_margin'], 1.0)
            self.assertFalse(comparison['proposed_M_is_calibrated'])
            self.assertFalse(comparison['run_provenance_verified'])
            self.assertEqual(ri['mode'], 'repack')
            self.assertFalse(oi['all_expected_moe_layers_covered'])

    def test_explicit_layer_inventory_must_be_covered_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, directory = capture(root, 'ordinary', expected_moe_layers=[0, 1])
            with self.assertRaisesRegex(ValueError, 'expected layer inventory'):
                C.finalize(directory, prep, 'c' * 64, SUCCESS)
            self.assertFalse((directory / 'index.json').exists())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, directory = capture(root, 'ordinary', expected_moe_layers=[0])
            index = C.finalize(directory, prep, 'c' * 64, SUCCESS)
            self.assertTrue(index['all_expected_moe_layers_covered'])
            self.assertEqual(index['covered_moe_layers'], [0])

    def test_full_layer_claim_requires_supervisor_bound_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indexes, bundles = [], []
            for mode in ('ordinary', 'repack'):
                prep, directory = capture(root, mode, expected_moe_layers=[0])
                prepared_sha = hashlib.sha256((prep / 'prepared.json').read_bytes()).hexdigest()
                identity_sha = 'c' * 64 if mode == 'ordinary' else 'd' * 64
                run = root / (mode + '-run')
                result = B.supervise([sys.executable, '-c', 'pass'], run, timeout_s=2,
                                     run_identity={'identity_sha256': identity_sha,
                                                   'expected_moe_layers': [0],
                                                   'prepared_sha256': prepared_sha})
                C.finalize(directory, prep, identity_sha, result)
                indexes.append(directory)
                bundles.append(run)
            result = C.compare(indexes[0], indexes[1], run_bundles=bundles)
            self.assertTrue(result['all_expected_moe_layers_covered'])
            self.assertEqual(result['covered_moe_layers'], [0])

    def test_compare_can_verify_distinct_supervisor_bundles_and_identity_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ordinary_run = B.supervise([sys.executable, '-c', 'pass'], root / 'ordinary-run',
                                       timeout_s=2, run_identity={'identity_sha256': 'c' * 64})
            repack_run = B.supervise([sys.executable, '-c', 'pass'], root / 'repack-run',
                                     timeout_s=2, run_identity={'identity_sha256': 'd' * 64})
            oprep, ordinary = capture(root, 'ordinary')
            rprep, repack = capture(root, 'repack')
            C.finalize(ordinary, oprep, 'c' * 64, ordinary_run)
            C.finalize(repack, rprep, 'd' * 64, repack_run)
            result = C.compare(ordinary, repack, run_bundles=[root / 'ordinary-run', root / 'repack-run'])
            self.assertTrue(result['run_provenance_verified'])
            self.assertTrue(result['run_provenance']['distinct_successful_processes_verified'])

    def test_finalize_rejects_failure_nonfinite_duplicate_and_unexpected_file(self):
        cases = ('nonfinite', 'duplicate')
        for fault in cases:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prep, directory = capture(root, 'ordinary', fault=fault)
                with self.assertRaises(ValueError):
                    C.finalize(directory, prep, 'c' * 64, SUCCESS)
                self.assertFalse((directory / 'index.json').exists())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, directory = capture(root, 'ordinary')
            with self.assertRaises(ValueError):
                C.finalize(directory, prep, 'c' * 64, {**SUCCESS, 'status': 'incomplete'})
            (directory / 'unexpected').write_text('x')
            (directory / 'unexpected').chmod(0o600)
            with self.assertRaises(ValueError):
                C.finalize(directory, prep, 'c' * 64, SUCCESS)

    def test_finalize_rejects_tampered_prepared_input_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, directory = capture(root, 'ordinary')
            path = prep / 'inputs.bin'
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                C.finalize(directory, prep, 'c' * 64, SUCCESS)

    def test_finalize_and_compare_never_follow_nodes_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, directory = capture(root, 'ordinary')
            external = root / 'external.bin'
            external.write_bytes((directory / 'nodes.bin').read_bytes())
            external.chmod(0o600)
            (directory / 'nodes.bin').unlink()
            (directory / 'nodes.bin').symlink_to(external)
            with self.assertRaises(OSError):
                C.finalize(directory, prep, 'c' * 64, SUCCESS)
            self.assertFalse((directory / 'index.json').exists())

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oprep, ordinary = capture(root, 'ordinary')
            rprep, repack = capture(root, 'repack')
            C.finalize(ordinary, oprep, 'c' * 64, SUCCESS)
            C.finalize(repack, rprep, 'd' * 64, SUCCESS)
            external = root / 'external.bin'
            external.write_bytes((ordinary / 'nodes.bin').read_bytes())
            external.chmod(0o600)
            (ordinary / 'nodes.bin').unlink()
            (ordinary / 'nodes.bin').symlink_to(external)
            with self.assertRaises(OSError):
                C.compare(ordinary, repack)

    def test_compare_rejects_mismatched_selected_experts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oprep, odir = capture(root, 'ordinary')
            rprep, rdir = capture(root, 'repack', ids=(2, 3))
            C.finalize(odir, oprep, 'c' * 64, SUCCESS)
            C.finalize(rdir, rprep, 'd' * 64, SUCCESS)
            result = C.compare(odir, rdir)
            topk = next(x for x in result['nodes'] if x['name'] == 'ffn_moe_topk-0')
            self.assertFalse(topk['exact_match'])
            self.assertEqual(topk['mismatches'], 1)

    def test_compare_rejects_unintended_native_policy_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            oprep, ordinary = capture(root, 'ordinary')
            rprep, repack = capture(root, 'repack')
            C.finalize(ordinary, oprep, 'c' * 64, SUCCESS)
            C.finalize(repack, rprep, 'd' * 64, SUCCESS)
            index_path = repack / 'index.json'
            index = json.loads(index_path.read_text())
            index['native_policy']['threads'] = 4
            index_path.write_text(json.dumps(index, sort_keys=True, separators=(',', ':')) + '\n')
            binding_path = repack / 'capture-run-binding.json'
            binding = json.loads(binding_path.read_text())
            binding['index_sha256'] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            binding_path.write_text(json.dumps(binding, sort_keys=True, separators=(',', ':')) + '\n')
            with self.assertRaisesRegex(ValueError, 'native policy|changes more than'):
                C.compare(ordinary, repack)


if __name__ == '__main__':
    unittest.main()
