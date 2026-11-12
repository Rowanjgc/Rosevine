#!/usr/bin/env python3
"""Build private exact-token fixtures with explicitly selected native tools.

CPU-only metadata/template/tokenization, not inference. The direct Jinja path
matches Epic 1's limited text contract; it does not establish server preprocessing
parity. No PATH fallback, endpoint, router configuration or model download.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

from benchmark_core import digest, make_workload, open_private, write_private_json
from gguf_audit import audit


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class NativeTextAdapter:
    def __init__(self, renderer, tokenizer, model, template, metadata, private_parent):
        self.renderer, self.tokenizer = str(renderer), str(tokenizer)
        self.model, self.template = str(model), str(template)
        self.metadata, self.private_parent = metadata, private_parent
        self.calls = 0

    def __call__(self, prompt, policy):
        self.calls += 1
        tokens = self.metadata['tokenizer.ggml.tokens']
        values = {'messages': [{'role': 'user', 'content': prompt}],
                  'enable_thinking': policy['thinking'], 'add_generation_prompt': True,
                  'tools': [],
                  'bos_token': tokens[self.metadata['tokenizer.ggml.bos_token_id']],
                  'eos_token': tokens[self.metadata['tokenizer.ggml.eos_token_id']]}
        temp = Path(tempfile.mkdtemp(dir=self.private_parent))
        completed = False
        try:
            write_private_json(temp / 'input.json', values)
            rendered = temp / 'rendered.txt'
            # Native tools create their own files; process umask is set by main.
            with open_private(temp / 'native-stderr.txt') as errors:
                rendered_run = subprocess.run(
                    [self.renderer, '--no-common', '--json', str(temp / 'input.json'),
                     '--output', str(rendered), self.template],
                    stdout=subprocess.DEVNULL, stderr=errors, timeout=30, check=False, umask=0o077)
                if rendered_run.returncode or not rendered.is_file():
                    raise ValueError('native render failed')
                tokenized = subprocess.run(
                    [self.tokenizer, '-m', self.model, '-f', str(rendered),
                     '--ids', '--no-bos', '--offline'],
                    stdout=subprocess.PIPE, stderr=errors, timeout=30, check=False, umask=0o077)
                if tokenized.returncode:
                    raise ValueError('native tokenize failed')
            result = json.loads(tokenized.stdout)
            text = rendered.read_text()
            completed = True
            return text, result
        finally:
            if completed:
                shutil.rmtree(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private directory')
    args = parser.parse_args()
    import os
    os.umask(0o077)
    # Explicit resolved paths, no PATH-selected tool substitution.
    renderer, tokenizer, model = [p.resolve(strict=True) for p in
                                  (args.renderer, args.tokenizer, args.model)]
    args.output.mkdir(mode=0o700, exist_ok=False)
    manifest = {'status': 'incomplete', 'mode': 'direct-Jinja text only'}
    write_private_json(args.output / 'started.json', manifest)
    try:
        metadata_audit = audit(model, full_hash=True)
        metadata = metadata_audit['metadata']
        template = args.output / 'embedded.jinja'
        with open(template, 'x') as stream:
            stream.write(metadata['tokenizer.chat_template'])
        suite_path = Path(__file__).resolve().parents[1] / 'research/fixtures/short-text-v1.json'
        suite = json.loads(suite_path.read_text())
        identity = {'model_sha256': metadata_audit['sha256'],
                    'template_sha256': file_hash(template),
                    'tokenizer_sha256': digest({k: v for k, v in metadata.items()
                                                if k.startswith('tokenizer.') and k != 'tokenizer.chat_template'})}
        policy = {'thinking': True, 'seed': 20261111,
                  'sampler_order': ['temperature'], 'sampler_parameters': {'temperature': 0},
                  'stops': []}
        adapter = NativeTextAdapter(renderer, tokenizer, model, template, metadata, args.output)
        manifest.update({'identity': identity, 'model_size': metadata_audit['size'],
                         'renderer_sha256': file_hash(renderer), 'tokenizer_binary_sha256': file_hash(tokenizer),
                         'fixture_sha256': file_hash(suite_path), 'policy': policy,
                         'policy_status': 'proposed greedy control; not historical sampling',
                         'padding': ' x', 'workloads': []})
        write_private_json(args.output / 'configuration.json', manifest)
        for fixture in suite['fixtures']:
            for occupied in (1900, 29000):
                record = make_workload(fixture, ' x', occupied, 65536, adapter,
                                       identity, policy, max_attempts=128)
                name = fixture['id'] + '-' + str(occupied) + '.json'
                write_private_json(args.output / name, record)
                manifest['workloads'].append({'file': name, 'sha256': file_hash(args.output / name),
                                              'token_count': len(record['token_ids'])})
        manifest.update({'status': 'successful', 'native_calls': adapter.calls})
    except Exception as error:
        manifest.update({'status': 'failed', 'failure_category': type(error).__name__})
        raise
    finally:
        write_private_json(args.output / 'result.json', manifest)
    print('Private workload construction complete; no inference or benchmark executed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        print('Workload construction failed; inspect the private bundle locally.', file=sys.stderr)
        sys.exit(1)
