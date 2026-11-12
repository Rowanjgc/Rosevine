"""Private capacity-only manifests preserve exact frozen workload inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('capacity_controls', ROOT / 'scripts/capacity_controls.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
sys_spec = importlib.util.spec_from_file_location('benchmark_core', ROOT / 'scripts/benchmark_core.py')
B = importlib.util.module_from_spec(sys_spec)
sys_spec.loader.exec_module(B)


def manifest(fixture_id, occupied):
    prompt = 'p' * occupied
    rendered = '<' + prompt + '>'
    tokens = [index % 256 for index in range(occupied)]
    record = {'version': B.VERSION, 'fixture_id': fixture_id,
              'fixture_sha256': 'a' * 64,
              'identity': {key: 'b' * 64 for key in ('model_sha256', 'template_sha256', 'tokenizer_sha256')},
              'policy': {'thinking': True, 'seed': 7, 'sampler_order': ['temperature'],
                         'sampler_parameters': {'temperature': 0}, 'stops': []},
              'occupied_input_tokens': occupied, 'capacity_tokens': 65536,
              'max_output_tokens': 512, 'eos_policy': 'retain separately if shorter',
              'prompt': prompt, 'rendered': rendered, 'token_ids': tokens,
              'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
              'tokens_sha256': B.digest(tokens)}
    record['manifest_sha256'] = B.digest(record)
    return record


class CapacityControlTests(unittest.TestCase):
    def test_builds_exact_capacity_pairs_without_rewriting_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir(mode=0o700)
            os.chmod(source, 0o700)
            originals = {}
            for number in range(8):
                for occupied in (1900, 29000):
                    record = manifest('fixture-%d' % number, occupied)
                    name = '%s-%d.json' % (record['fixture_id'], occupied)
                    B.write_private_json(source / name, record)
                    originals[(record['fixture_id'], occupied)] = record
            output = root / 'controls'
            result = C.build(source, output)
            self.assertEqual(result['status'], 'successful')
            self.assertEqual(result['source_count'], 16)
            for item in result['capacity_controls']:
                key = (item['fixture_id'], item['occupied_input_tokens'])
                source_record = originals[key]
                capacity = 4096 if key[1] == 1900 else 32768
                target = output / ('%s-%d-cap%d.json' % (key[0], key[1], capacity))
                control = json.loads(target.read_text())
                self.assertEqual(control['capacity_tokens'], capacity)
                for field in ('prompt', 'rendered', 'token_ids', 'rendered_sha256', 'tokens_sha256'):
                    self.assertEqual(control[field], source_record[field])
                self.assertEqual(item['changed_fields'], ['capacity_tokens', 'manifest_sha256'])

    def test_rejects_duplicate_or_infeasible_capacity_controls(self):
        record = manifest('fixture', 1900)
        with self.assertRaises(ValueError):
            B.rebind_workload_capacity(record, 2911)
        with self.assertRaises(ValueError):
            B.rebind_workload_capacity(dict(record, manifest_sha256='0' * 64), 4096)


if __name__ == '__main__':
    unittest.main()
