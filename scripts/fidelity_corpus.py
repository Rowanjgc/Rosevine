#!/usr/bin/env python3
"""Private document-level fidelity corpus construction from explicit local data.

No download, execution of source code, inference, prompt padding or truncation.
The prose stratum is human math-problem prose/reasoning; the code stratum is Python
solutions/tests. This distribution does not establish broad natural-language or
multilingual quality. Source licensing/provenance is a separate frozen manifest.
"""
import hashlib

from benchmark_core import digest

VERSION = 'rosevine-fidelity-corpus-v1'


def documents(gsm, mbpp):
    """Respect MBPP test/validation IDs; split GSM test documents by fixed hash.

    Duplicate full text or code bodies are assigned only once across all splits.
    Local holdout separation makes no assertion about model pretraining exposure.
    """
    output, ids, fingerprints = [], set(), set()
    for index, row in enumerate(gsm):
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k] for k in ('question', 'answer')):
            raise ValueError('invalid GSM source record')
        text = 'Question: ' + row['question'] + '\nAnswer: ' + row['answer'] + '\n'
        sha = hashlib.sha256(text.encode()).hexdigest()
        split = 'calibration' if int(sha[:8], 16) % 10 < 2 else 'holdout'
        output.append({'id': 'gsm-test-' + str(index), 'stratum': 'problem-prose',
                       'split': split, 'text': text, 'text_sha256': sha, 'dedup_sha256': sha})
    for row in mbpp:
        task_id = row.get('task_id') if isinstance(row, dict) else None
        if type(task_id) is not int or task_id < 1 or task_id in ids:
            raise ValueError('invalid or duplicate MBPP task ID')
        ids.add(task_id)
        if not 11 <= task_id <= 600:
            continue
        if any(not isinstance(row.get(k), str) or not row[k] for k in ('text', 'code')) or not isinstance(row.get('test_list'), list) or not row['test_list'] or any(not isinstance(t, str) for t in row['test_list']):
            raise ValueError('invalid MBPP source record')
        setup = row.get('test_setup_code', '')
        if not isinstance(setup, str):
            raise ValueError('invalid MBPP setup text')
        text = '\n'.join('# ' + line for line in row['text'].splitlines()) + '\n'
        text += setup + '\n' + row['code'] + '\n' + '\n'.join(row['test_list']) + '\n'
        output.append({'id': 'mbpp-' + str(task_id), 'stratum': 'python-code',
                       'split': 'holdout' if task_id <= 510 else 'calibration',
                       'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                       'dedup_sha256': hashlib.sha256(row['code'].strip().encode()).hexdigest()})
    # Calibration first: its exact text/code cannot later appear in the holdout.
    output.sort(key=lambda d: (d['split'] != 'calibration', digest(['selection-v1', d['id']])))
    unique = []
    for document in output:
        if document['text_sha256'] in fingerprints or document['dedup_sha256'] in fingerprints:
            continue
        fingerprints.update((document['text_sha256'], document['dedup_sha256']))
        unique.append(document)
    return unique


def build(gsm, mbpp, tokenize, *, holdout_per_stratum=50000, calibration_per_stratum=2000,
          max_document_tokens=4096):
    """Retain whole documents until each quota is met; score tokens after token 0.

    tokenize returns exact integer IDs including the reference's declared BOS
    policy. Every selected document is replayed. A source/quota shortage fails;
    repeated documents or inferred character-based token counts never fill it.
    """
    for count in (holdout_per_stratum, calibration_per_stratum, max_document_tokens):
        if type(count) is not int or count <= 1:
            raise ValueError('positive token quotas/context limit required')
    quotas = {(split, stratum): holdout_per_stratum if split == 'holdout' else calibration_per_stratum
              for split in ('calibration', 'holdout') for stratum in ('problem-prose', 'python-code')}
    counts = {key: 0 for key in quotas}
    selected, rejected = [], []
    for document in documents(gsm, mbpp):
        key = document['split'], document['stratum']
        if counts[key] >= quotas[key]:
            continue
        ids = tokenize(document['text'])
        if not isinstance(ids, list) or len(ids) < 2 or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('invalid corpus token stream')
        if len(ids) > max_document_tokens:
            rejected.append({'id': document['id'], 'reason': 'document exceeds frozen context limit',
                             'tokens': len(ids)})
            continue
        if ids != tokenize(document['text']):
            raise ValueError('corpus token replay mismatch')
        selected.append(dict(document, token_ids=ids, tokens_sha256=digest(ids),
                             scored_tokens=len(ids) - 1, first_scored_target_index=1))
        counts[key] += len(ids) - 1
    missing = [split + '/' + stratum for (split, stratum), count in counts.items() if count < quotas[(split, stratum)]]
    if missing:
        raise ValueError('insufficient source tokens for frozen corpus quotas')
    record = {'version': VERSION, 'documents': selected, 'rejected_documents': rejected,
              'counts': {split: {stratum: counts[(split, stratum)] for stratum in ('problem-prose', 'python-code')}
                         for split in ('calibration', 'holdout')},
              'max_document_tokens': max_document_tokens,
              'score_rule': 'logits at input j predict token j+1; reset KV per document; token index 0 is context only',
              'selection': 'fixed SHA256 ID ranking; calibration-first exact text/code dedup; whole documents',
              'limitations': 'math-problem prose and Python only; shared task styles; no claim of pretraining exclusion, semantic dedup or broad-language coverage'}
    record['corpus_sha256'] = digest(record)
    return record
