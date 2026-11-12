#!/usr/bin/env python3
"""Read pinned native completion diagnostics without launching any process.

Prompt-ID replay is separate from phase counters: the native timer classifies
single-row evaluations as decode, even when a row belongs to the prompt. Its
decode rate is therefore not a committed-output or delivery-rate measurement.
Raw log text and all returned observations remain private research evidence.
"""
import math
import re

from benchmark_core import missing_metrics
from benchmark_core import METRIC_UNITS


def prompt_replay(text, expected_ids):
    if not isinstance(text, str) or not isinstance(expected_ids, list) or not expected_ids or any(type(t) is not int or t < 0 for t in expected_ids):
        raise ValueError('native log text and explicit expected token IDs required')
    counts = re.findall(r'number of tokens in prompt\s*=\s*(\d+)', text)
    ids = [int(t) for t in re.findall(r'(?m)^[0-9.]+\s+I\s+(\d+)\s+->\s*\x27', text)]
    if counts != [str(len(expected_ids))] or ids != expected_ids:
        raise ValueError('native evaluated prompt count or IDs differ from frozen input')
    return {'status': 'exact', 'occupied_input_tokens': len(ids), 'token_ids': ids}


def _timing(text, phase, unit):
    pattern = (r'(?m)^[0-9.]+\s+I\s+common_perf_print:\s*' + re.escape(phase)
               + r' time\s*=\s*([\w.+-]+)\s*ms\s*/\s*(\d+)\s*' + unit + r'\b')
    matches = re.findall(pattern, text)
    if not matches:
        return {'value': None, 'reason': 'native phase timing not reported'}
    if len(matches) != 1:
        raise ValueError('ambiguous native phase timing')
    duration, count = float(matches[0][0]), int(matches[0][1])
    if not math.isfinite(duration) or duration < 0 or count < 1:
        raise ValueError('invalid native phase timing')
    return {'value': {'duration_s': duration / 1000, 'count': count}, 'reason': None}


def observations(text, expected_ids):
    replay = prompt_replay(text, expected_ids)
    prefill = _timing(text, 'prompt eval', 'tokens')
    decode = _timing(text, 'eval', 'runs')
    metrics = missing_metrics('no native adapter for this metric; not inferred from console throughput')
    source = prefill['value']
    if source and source['duration_s'] > 0 and source['count'] == len(expected_ids) and len(expected_ids) > 1:
        metrics['prefill_tokens_per_s'] = {
            'value': source['count'] / source['duration_s'], 'reason': None,
            'unit': METRIC_UNITS['prefill_tokens_per_s']}
    else:
        metrics['prefill_tokens_per_s']['reason'] = 'native phase does not separately time the complete frozen prompt'
    metrics['target_decode_tokens_per_s']['reason'] = 'native evaluation count is not a committed-output count; no matching decode-loop boundary'
    metrics['ttft_s']['reason'] = 'no committed-token delivery timestamp'
    metrics['steady_output_tokens_per_s']['reason'] = 'no committed-token delivery timestamps'
    metrics['end_to_end_s']['reason'] = 'native phase timing excludes admission/delivery and supervisor includes cleanup'
    return {'prompt_replay': replay, 'metrics': metrics,
            'native_prompt_phase': prefill, 'native_decode_evaluation_phase': decode,
            'timing_scope': 'native host phase diagnostics; cache state/overhead and controlled trials still required',
            'acceptance': 'not a performance or numerical-correctness acceptance'}
