#!/usr/bin/env python3
"""Plan direct-token teacher-forced reference rows from calibration inputs only.

No rendering, retokenization, inference or quality scoring. Synthetic sequence
boundaries and circular windows deliberately differ from natural documents; this
is graph/logit coverage input, not the held-out NLL corpus or a chat request.
"""
import argparse
import json
from pathlib import Path
import sys

from benchmark_core import digest, write_private_json
from nll_compare import validate_corpus


def make_plan(corpus, bos_id, *, streams=8, positions=512, capacity=4096, vocabulary=262144):
    for count in (streams, positions, capacity, vocabulary):
        if type(count) is not int or count < 1:
            raise ValueError('positive integer position-plan dimensions required')
    if type(bos_id) is not int or not 0 <= bos_id < vocabulary or positions + 1 > capacity:
        raise ValueError('valid BOS ID and capacity for scored positions required')
    documents = validate_corpus(corpus, 'calibration')
    pool, spans = [], []
    for identifier in sorted(documents):
        document = documents[identifier]
        tokens = document['token_ids']
        if tokens[0] != bos_id or any(t >= vocabulary for t in tokens):
            raise ValueError('corpus does not match explicit BOS/vocabulary policy')
        start = len(pool)
        pool.extend(tokens[1:])
        spans.append({'document_id': identifier, 'start': start, 'end': len(pool),
                      'stratum': document['stratum'], 'tokens_sha256': document['tokens_sha256']})
    if len(pool) < positions:
        raise ValueError('insufficient calibration tokens; do not fill with holdout inputs')
    sequences = []
    for i in range(streams):
        offset = i * len(pool) // streams
        ids = [bos_id] + [pool[(offset + j) % len(pool)] for j in range(positions)]
        sequences.append({'id': f'reference-{i:03d}', 'pool_offset': offset,
                          'token_ids': ids, 'tokens_sha256': digest(ids),
                          'scored_input_positions': [0, positions - 1],
                          'scored_target_indices': [1, positions],
                          'reset_kv_before_sequence': True})
    if len({s['tokens_sha256'] for s in sequences}) != streams:
        raise ValueError('duplicate reference streams; revise the position-plan version')
    count = streams * positions
    record = {'schema': 'rosevine-reference-positions-v1', 'status': 'planned; no logits captured',
              'corpus_sha256': corpus['corpus_sha256'], 'source_split': 'calibration',
              'input_kind': 'direct token IDs; no chat rendering',
              'rendered_prompt_bytes': {'value': None, 'reason': 'direct-token synthetic graph input; source full documents remain in corpus'},
              'protocol_scope': {
                  'decision': 'direct-token teacher-forced input is applicable to final-logit numerical comparison',
                  'basis': 'the frozen token IDs isolate model graph outputs from rendering and tokenization variation',
                  'limitations': [
                      'does not validate rendering, tokenizer behavior, chat templates, or user-visible workloads',
                      'overlapping/circular calibration windows are correlated and are not quality samples',
                      'does not substitute for the held-out corpus NLL or task-quality protocols',
                  ],
                  'execution_status': 'planned only; separate run-window and resource review still required',
              },
              'bos_id': bos_id, 'vocabulary': vocabulary, 'capacity_tokens': capacity,
              'pool_tokens': len(pool), 'pool_spans': spans, 'sequences': sequences,
              'scored_positions': count, 'minimum_4096_positions_planned': count >= 4096,
              'fp32_full_vocabulary_bytes_per_repetition': count * vocabulary * 4,
              'two_repetitions_bytes': count * vocabulary * 8,
              'limitations': 'Overlapping calibration-token windows; not independent quality samples, natural-document NLL, long-context coverage or captured correctness evidence.'}
    record['plan_sha256'] = digest(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--bos-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_private_json(args.output, make_plan(json.loads(args.corpus.read_text()), args.bos_id))
    print('Private direct-token reference position plan created; no logits captured.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError):
        print('Reference position planning failed; no capture authorized.', file=sys.stderr)
        sys.exit(1)
