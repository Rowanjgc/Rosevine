"""Sampled-token comparisons are paired, bounded and explicitly descriptive."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_core import canonical, supervise, write_private_json
import stochastic_distribution as S
import stochastic_capture as C


def capture(model_hash, token):
    seeds = list(range(20))
    return {'schema': 'rosevine-stochastic-output-v3',
            'model_sha256': model_hash, 'sampler_policy_sha256': 'b' * 64,
            'sampler_observer_sha256': 'd' * 64,
            'vocabulary_size': 32, 'eos_token_id': 0, 'max_output_tokens': 3,
            'seed_schedule': seeds,
            'prompts': [{'prompt_id': 'task-1', 'prompt_sha256': 'c' * 64,
                         'samples': [{'seed': seed, 'status': 'successful',
                                      'finish_reason': 'limit',
                                      'token_ids': [token, token, token],
                                      'run_bundle': f'/unverified/bundle-{seed}'}
                                     for seed in seeds]}]}


def bound_capture(root, model_bytes, output_tokens):
    model_path = root / ('model-' + hashlib.sha256(model_bytes).hexdigest()[:8])
    model_path.write_bytes(model_bytes)
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    observer_path = root / 'sampler-observer.so'
    observer_path.write_bytes(b'synthetic sampler observer')
    observer_hash = hashlib.sha256(observer_path.read_bytes()).hexdigest()
    policy_hash, prompt_hash = 'b' * 64, 'c' * 64
    seeds = list(range(20))
    samples = []
    for seed in seeds:
        tokens, finish = output_tokens(seed)
        identity = {'schema': 'rosevine-stochastic-sample-v2',
                    'model_sha256': model_hash,
                    'sampler_policy_sha256': policy_hash,
                    'sampler_observer_sha256': observer_hash,
                    'prompt_id': 'task-1', 'prompt_sha256': prompt_hash,
                    'seed': seed, 'vocabulary_size': 32, 'eos_token_id': 0,
                    'max_output_tokens': 3}
        events = ['ROSEVINE_SAMPLER_HOOK_V1 READY']
        events.extend(f'ROSEVINE_SAMPLED_TOKEN_V1 {index} {token}'
                      for index, token in enumerate(tokens))
        events.append(f'ROSEVINE_SAMPLER_HOOK_V1 DONE {len(tokens)} 0')
        bundle = root / f"run-{len(list(root.glob('run-*')))}"
        command = [sys.executable, '-c',
                   'import sys; sys.stdout.write("synthetic output\\n"); sys.stderr.write('
                   + repr('\n'.join(events) + '\n') + ')']
        artifact = {'role': 'target_model', 'path': str(model_path),
                    'sha256': model_hash, 'size_bytes': len(model_bytes)}
        observer = {'role': 'sample_observer', 'path': str(observer_path),
                    'sha256': observer_hash, 'size_bytes': observer_path.stat().st_size}
        supervise(command, bundle, timeout_s=2,
                  run_identity={'stochastic_sample': identity},
                  expected_artifacts=[artifact, observer])
        samples.append({'seed': seed, 'status': 'successful',
                        'finish_reason': finish, 'token_ids': tokens,
                        'run_bundle': str(bundle)})
    return {'schema': 'rosevine-stochastic-output-v3',
            'model_sha256': model_hash, 'sampler_policy_sha256': policy_hash,
            'sampler_observer_sha256': observer_hash,
            'vocabulary_size': 32, 'eos_token_id': 0, 'max_output_tokens': 3,
            'seed_schedule': seeds,
            'prompts': [{'prompt_id': 'task-1', 'prompt_sha256': prompt_hash,
                         'samples': samples}]}


class StochasticDistributionTests(unittest.TestCase):
    def test_categorical_metrics_report_identical_and_disjoint_support(self):
        self.assertEqual(S.categorical_metrics([1, 2, 1], [1, 2, 1])['jensen_shannon_bits'], 0)
        disjoint = S.categorical_metrics([1] * 20, [2] * 20)
        self.assertEqual(disjoint['total_variation'], 1)
        self.assertEqual(disjoint['jensen_shannon_bits'], 1)

    def test_verified_bundles_bind_all_seed_receipts_and_preserve_eos_behavior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / 'reference.json'
            candidate = root / 'candidate.json'
            reference_data = bound_capture(root, b'reference model', lambda _seed: ([3, 3, 3], 'limit'))
            candidate_data = bound_capture(root, b'candidate model',
                lambda seed: (([4, 0], 'eos') if seed < 2 else ([4, 4, 4], 'limit')))
            plan = C.prepare_plan({
                'schema': 'rosevine-stochastic-plan-v2',
                'model_sha256': reference_data['model_sha256'],
                'sampler_policy_sha256': reference_data['sampler_policy_sha256'],
                'sampler_observer_sha256': reference_data['sampler_observer_sha256'],
                'vocabulary_size': reference_data['vocabulary_size'],
                'eos_token_id': reference_data['eos_token_id'],
                'max_output_tokens': reference_data['max_output_tokens'],
                'seed_schedule': reference_data['seed_schedule'],
                'prompts': [{'prompt_id': row['prompt_id'], 'prompt_sha256': row['prompt_sha256']}
                            for row in reference_data['prompts']]})
            bundle_map = {'schema': 'rosevine-stochastic-bundle-map-v1',
                          'plan_sha256': plan['plan_sha256'],
                          'runs': [{'prompt_id': row['prompt_id'], 'seed': sample['seed'],
                                    'run_bundle': sample['run_bundle']}
                                   for row in reference_data['prompts']
                                   for sample in row['samples']]}
            record, finalized = C.finalize_capture(plan, bundle_map, root)
            self.assertEqual(record['status'], 'ready')
            self.assertEqual(record['successful_sample_count'], 20)
            self.assertEqual(finalized, reference_data)
            reference_data = finalized
            write_private_json(reference, reference_data)
            write_private_json(candidate, candidate_data)
            result = S.compare(reference, candidate)
            self.assertEqual(result['seed_count_per_prompt'], 20)
            self.assertEqual(result['verified_supervisor_bundles'], 40)
            self.assertEqual(result['distinct_child_processes'], 40)
            self.assertEqual(result['prompts'][0]['positions'][0]['total_variation'], 1)
            self.assertEqual(result['prompts'][0]['positions'][0]['jensen_shannon_bits'], 1)
            eos_position = result['prompts'][0]['positions'][2]
            self.assertEqual(eos_position['reference_eos_rate'], 0)
            self.assertEqual(eos_position['candidate_eos_rate'], .1)
            self.assertIn('no acceptance decision', result['status'])
            # Captured IDs cannot diverge from the checksummed sampler events.
            candidate_data['prompts'][0]['samples'][0]['token_ids'] = [5, 0]
            write_private_json(root / 'tampered.json', candidate_data)
            with self.assertRaisesRegex(ValueError, 'do not match checksummed sampler events'):
                S.compare(reference, root / 'tampered.json')

    def test_requires_twenty_seeds_and_retains_no_failed_sample_silently(self):
        for mutate, message in [
                (lambda data: data.update(seed_schedule=list(range(19))), 'at least 20'),
                (lambda data: data['prompts'][0]['samples'][0].update(status='failed'), 'failed or unplanned')]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = capture('a' * 64, 3)
                mutate(data)
                write_private_json(root / 'capture.json', data)
                with self.assertRaisesRegex(ValueError, message):
                    S.load_capture(root / 'capture.json')

    def test_plan_does_not_count_one_prompt_twice_under_different_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {'schema': 'rosevine-stochastic-plan-v2',
                    'model_sha256': 'a' * 64, 'sampler_policy_sha256': 'b' * 64,
                    'sampler_observer_sha256': 'd' * 64,
                    'vocabulary_size': 32, 'eos_token_id': 0, 'max_output_tokens': 3,
                    'seed_schedule': list(range(20)),
                    'prompts': [{'prompt_id': 'one', 'prompt_sha256': 'c' * 64},
                                {'prompt_id': 'two', 'prompt_sha256': 'c' * 64}]}
            with self.assertRaisesRegex(ValueError, 'duplicate stochastic prompt identity'):
                C.prepare_plan(plan)

    def test_pairing_requires_same_prompt_seed_and_sampler_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = capture('a' * 64, 3)
            candidate = capture('d' * 64, 3)
            candidate['sampler_policy_sha256'] = 'e' * 64
            write_private_json(root / 'reference.json', reference)
            write_private_json(root / 'candidate.json', candidate)
            with self.assertRaisesRegex(ValueError, 'sampler_policy_sha256 mismatch'):
                S.compare(root / 'reference.json', root / 'candidate.json')
            candidate['sampler_policy_sha256'] = reference['sampler_policy_sha256']
            candidate['prompts'][0]['prompt_sha256'] = 'f' * 64
            write_private_json(root / 'candidate-2.json', candidate)
            with self.assertRaisesRegex(ValueError, 'prompt bytes differ'):
                S.compare(root / 'reference.json', root / 'candidate-2.json')

    def test_supervisor_bundle_paths_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root.parent / (root.name + '-outside')
            outside.mkdir(mode=0o700)
            try:
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    S.resolve_bundle_path(root, str(outside / 'bundle'))
                (root / 'linked').symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'symlink'):
                    S.resolve_bundle_path(root, 'linked/bundle')
            finally:
                (root / 'linked').unlink(missing_ok=True)
                outside.rmdir()

    def test_finalization_retains_failed_and_missing_seed_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = C.prepare_plan({'schema': 'rosevine-stochastic-plan-v2',
                'model_sha256': 'a' * 64, 'sampler_policy_sha256': 'b' * 64,
                'sampler_observer_sha256': 'd' * 64,
                'vocabulary_size': 32, 'eos_token_id': 0, 'max_output_tokens': 3,
                'seed_schedule': list(range(20)),
                'prompts': [{'prompt_id': 'task-1', 'prompt_sha256': 'c' * 64}]})
            failed_bundle = root / 'failed-run'
            supervise([sys.executable, '-c', 'raise SystemExit(1)'], failed_bundle,
                      timeout_s=2)
            mapping = {'schema': 'rosevine-stochastic-bundle-map-v1',
                       'plan_sha256': plan['plan_sha256'],
                       'runs': [{'prompt_id': 'task-1', 'seed': 0,
                                 'run_bundle': str(failed_bundle)}]}
            record, capture_data = C.finalize_capture(plan, mapping, root)
        self.assertEqual(record['status'], 'incomplete')
        self.assertIsNone(capture_data)
        self.assertEqual(record['planned_sample_count'], 20)
        self.assertEqual(record['successful_sample_count'], 0)
        self.assertEqual(record['failed_or_missing_sample_count'], 20)
        statuses = [row['status'] for row in record['run_records']]
        self.assertEqual(statuses.count('failed'), 1)
        self.assertEqual(statuses.count('missing'), 19)

if __name__ == '__main__':
    unittest.main()
