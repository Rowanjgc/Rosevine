#!/usr/bin/env python3
"""Plan calibration-only token streams for local-window boundary captures.

This prepares exact input IDs and selected output rows; it does not load a model,
capture logits, establish semantic retrieval, or approve long-context execution.
"""
import argparse
import json
from pathlib import Path
import sys

from benchmark_core import digest, write_private_json
from nll_compare import validate_corpus


def make_plan(corpus, bos_id, *, local_window=1024, capacity=4096, vocabulary=262144, streams=3):
    if any(type(x) is not int or x <= 0 for x in (local_window, capacity, vocabulary, streams)):
        raise ValueError('positive integer context dimensions required')
    if local_window < 2 or capacity % 256 or local_window + 3 > capacity:
        raise ValueError('context boundary does not fit the aligned capacity')
    if type(bos_id) is not int or not 0 <= bos_id < vocabulary:
        raise ValueError('valid BOS ID required')
    documents = validate_corpus(corpus, 'calibration')
    pool, spans = [], []
    for identifier in sorted(documents):
        document = documents[identifier]
        tokens = document['token_ids']
        if tokens[0] != bos_id or any(token >= vocabulary for token in tokens):
            raise ValueError('calibration corpus BOS/vocabulary mismatch')
        start = len(pool)
        pool.extend(tokens[1:])
        spans.append({'document_id': identifier, 'start': start, 'end': len(pool),
                      'stratum': document['stratum'], 'tokens_sha256': document['tokens_sha256']})
    sequence_length = local_window + 3
    pool_width = sequence_length - 1
    if len(pool) < pool_width or streams > len(pool):
        raise ValueError('insufficient distinct calibration positions for boundary streams')
    positions = [local_window - 1, local_window, local_window + 1]
    sequences, seen = [], set()
    for index in range(streams):
        offset = index * len(pool) // streams
        tokens = [bos_id] + [pool[(offset + j) % len(pool)] for j in range(pool_width)]
        if any(token >= vocabulary for token in tokens):
            raise ValueError('input token outside vocabulary')
        token_hash = digest(tokens)
        if token_hash in seen:
            raise ValueError('duplicate context boundary streams')
        seen.add(token_hash)
        sequences.append({'id': f'context-boundary-{index:03d}', 'pool_offset': offset,
                          'token_ids': tokens, 'tokens_sha256': token_hash,
                          'capture_input_positions': positions,
                          'scored_target_indices': [position + 1 for position in positions],
                          'target_token_ids': [tokens[position + 1] for position in positions],
                          'reset_kv_before_sequence': True})
    rows = streams * len(positions)
    result = {'schema': 'rosevine-context-positions-v1',
              'status': 'planned; no logits captured',
              'source_split': 'calibration', 'corpus_sha256': corpus['corpus_sha256'],
              'input_kind': 'direct token IDs; no chat rendering',
              'rendered_prompt_bytes': {'value': None, 'reason': 'direct token boundary probe'},
              'bos_id': bos_id, 'vocabulary': vocabulary, 'capacity_tokens': capacity,
              'proposed_local_window_tokens': local_window,
              'captured_input_positions': positions,
              'sequence_tokens': sequence_length, 'streams': sequences, 'pool_spans': spans,
              'capture_rows': rows, 'fp32_bytes_per_repetition': rows * vocabulary * 4,
              'two_repetitions_bytes': rows * vocabulary * 8,
              'policy_review': 'confirm target local-window semantics; direct-token input and long-context run window require review',
              'limitations': 'synthetic circular calibration-token windows; tests boundary logits only; not semantic retrieval, global-position, continuation, chunked-prefill, resume, natural-document or captured-correctness evidence'}
    result['plan_sha256'] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--bos-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-window', type=int, default=1024)
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--vocabulary', type=int, default=262144)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    write_private_json(args.output, make_plan(corpus, args.bos_id,
                       local_window=args.local_window, capacity=args.capacity,
                       vocabulary=args.vocabulary))
    print('Private context-boundary input plan created; no model or logits captured.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Context-boundary planning failed; no execution authorized.', file=sys.stderr)
        sys.exit(1)
