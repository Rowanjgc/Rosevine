"""Strict parsing and native forwarding checks for sampled-token observation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from stochastic_sampler_events import parse


def event_stream(tokens, *, done_count=None, failed=False):
    rows = ['ROSEVINE_SAMPLER_HOOK_V1 READY']
    rows.extend(f'ROSEVINE_SAMPLED_TOKEN_V1 {index} {token}'
                for index, token in enumerate(tokens))
    rows.append(f'ROSEVINE_SAMPLER_HOOK_V1 DONE '
                f'{len(tokens) if done_count is None else done_count} {int(failed)}')
    return ('native diagnostic line\n' + '\n'.join(rows) + '\n').encode('ascii')


class SampledTokenEventTests(unittest.TestCase):
    def test_exact_eos_and_limit_receipts(self):
        self.assertEqual(parse(event_stream([7, 0]), vocabulary_size=16,
                               eos_token_id=0, max_output_tokens=3),
                         {'token_ids': [7, 0], 'finish_reason': 'eos'})
        self.assertEqual(parse(event_stream([7, 8, 9]), vocabulary_size=16,
                               eos_token_id=0, max_output_tokens=3),
                         {'token_ids': [7, 8, 9], 'finish_reason': 'limit'})

    def test_missing_reordered_dropped_or_early_stop_receipts_reject(self):
        valid = event_stream([7, 8, 9])
        faults = [b'ROSEVINE_SAMPLED_TOKEN_V1 0 7\n',
                  valid.replace(b'ROSEVINE_SAMPLED_TOKEN_V1 1 8\n', b''),
                  valid.replace(b'ROSEVINE_SAMPLED_TOKEN_V1 1 8\n',
                                b'ROSEVINE_SAMPLED_TOKEN_V1 2 8\n'),
                  valid.replace(b'ROSEVINE_SAMPLED_TOKEN_V1 1 8\n',
                                b'ROSEVINE_SAMPLER_HOOK_V1 DONE 3 0\n'
                                b'ROSEVINE_SAMPLED_TOKEN_V1 1 8\n'),
                  valid.replace(b'ROSEVINE_SAMPLER_HOOK_V1 READY\n',
                                b'ROSEVINE_SAMPLER_HOOK_V1 READY extra\n'),
                  event_stream([7], done_count=0),
                  event_stream([7, 8], failed=True),
                  event_stream([7, 0, 8])]
        for raw in faults:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw, vocabulary_size=16, eos_token_id=0, max_output_tokens=3)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler unavailable')
    def test_native_interposer_forwards_return_and_records_exact_token(self):
        source = ROOT / 'research/capture_sampled_ids.cpp'
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            target_source = temp / 'target.cpp'
            target_source.write_text('''
struct common_sampler; struct llama_context;
__attribute__((visibility("default"))) int common_sampler_sample(
    common_sampler *, llama_context *, int id, bool grammar_first) {
    return id + (grammar_first ? 70 : 0);
}
''')
            fixture = temp / 'caller.cpp'
            fixture.write_text('''
struct common_sampler; struct llama_context;
int common_sampler_sample(common_sampler *, llama_context *, int, bool);
int main() { return common_sampler_sample(nullptr, nullptr, 7, true) == 77 ? 0 : 9; }
''')
            observer = temp / 'observer.so'
            target = temp / 'libsampler_target.so'
            caller = temp / 'caller'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-shared', '-fPIC', '-O2', '-fno-fast-math', '-ffp-contract=off',
                            str(source), '-ldl', '-pthread', '-o', str(observer)],
                           check=True, capture_output=True)
            subprocess.run(['c++', '-std=c++17', '-shared', '-fPIC', '-O2',
                            str(target_source), '-o', str(target)],
                           check=True, capture_output=True)
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wl,--export-dynamic',
                            str(fixture), '-L', str(temp), '-lsampler_target',
                            '-Wl,-rpath,' + str(temp), '-o', str(caller)],
                           check=True, capture_output=True)
            environment = {'PATH': os.defpath, 'LC_ALL': 'C',
                           'LD_PRELOAD': str(observer)}
            result = subprocess.run([str(caller)], env=environment,
                                    check=False, capture_output=True)
        self.assertEqual(result.returncode, 0)
        try:
            receipt = parse(result.stderr, vocabulary_size=100,
                            eos_token_id=0, max_output_tokens=1)
        except ValueError as error:
            self.fail(f'{error}: {result.stderr!r}')
        self.assertEqual(receipt,
                         {'token_ids': [77], 'finish_reason': 'limit'})


if __name__ == '__main__':
    unittest.main()
