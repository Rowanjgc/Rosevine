#!/usr/bin/env python3
"""Reduce explicit native MTP verification events to run-level metrics.

This module reads no files, launches no model and infers no event from console
text. The producer must supply prefix-acceptance counts, target evaluation rows,
disjoint monotonic host phase intervals and rollback operation counts.
"""
import math

from benchmark_core import METRIC_UNITS, missing_metrics


def reduce_events(record):
    required = {'schema', 'mtp_enabled', 'max_draft_tokens', 'verifications'}
    if not isinstance(record, dict) or set(record) != required or record['schema'] != 'rosevine-native-mtp-events-v1':
        raise ValueError('versioned native MTP event record required')
    enabled, depth, events = record['mtp_enabled'], record['max_draft_tokens'], record['verifications']
    if type(enabled) is not bool or type(depth) is not int or not 1 <= depth <= 32:
        raise ValueError('invalid MTP event policy')
    if not isinstance(events, list) or len(events) > 1_000_000:
        raise ValueError('bounded verification event list required')
    output = missing_metrics('native MTP event source does not provide this metric')

    def put(name, value, reason=None):
        output[name] = {'value': value, 'reason': reason,
                        'unit': METRIC_UNITS[name]}

    if not enabled:
        if events:
            raise ValueError('MTP-disabled record cannot contain verification events')
        for name in ('mtp_proposed', 'mtp_accepted', 'draft_s', 'correction_s', 'rollback_work'):
            put(name, 0)
        put('mtp_acceptance_rate', None, 'MTP was disabled; no proposals were made')
        put('mtp_acceptance_by_position', [], None)
        put('mtp_committed_per_verification', None, 'no MTP verification occurred')
        put('target_evaluated_rows', None, 'target evaluation rows require a native phase source')
        put('verify_s', None, 'MTP verification did not run')
        return output
    if not events:
        raise ValueError('enabled MTP run requires at least one verification event')

    fields = {'verification_id', 'proposed_tokens', 'accepted_tokens',
              'committed_tokens', 'target_evaluated_rows', 'phase_intervals_ns',
              'rollback_operations'}
    ids, intervals = set(), []
    proposed_total = accepted_total = committed_total = target_rows = rollback_total = 0
    phase_totals = {'draft': 0, 'verify': 0, 'correction': 0}
    denominators = [0] * depth
    numerators = [0] * depth
    for event in events:
        if not isinstance(event, dict) or set(event) != fields:
            raise ValueError('native MTP verification event schema mismatch')
        identifier = event['verification_id']
        proposed, accepted = event['proposed_tokens'], event['accepted_tokens']
        committed, rows, rollback = (event['committed_tokens'], event['target_evaluated_rows'],
                                     event['rollback_operations'])
        if (not isinstance(identifier, str) or not identifier or identifier in ids
                or any(type(value) is not int or value < 0 for value in
                       (proposed, accepted, committed, rows, rollback))
                or proposed > depth or accepted > proposed or committed > accepted + 1
                or rows <= 0):
            raise ValueError('invalid or inconsistent MTP event counts')
        ids.add(identifier)
        proposed_total += proposed
        accepted_total += accepted
        committed_total += committed
        target_rows += rows
        rollback_total += rollback
        for position in range(proposed):
            denominators[position] += 1
            if accepted > position:
                numerators[position] += 1
        phases = event['phase_intervals_ns']
        if not isinstance(phases, dict) or set(phases) != {'draft', 'verify', 'correction'}:
            raise ValueError('all MTP phase intervals must be explicitly represented')
        for phase, interval in phases.items():
            if interval is None and phase == 'correction':
                continue
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(value) is not int or value < 0 for value in interval)
                    or interval[1] <= interval[0]):
                raise ValueError('invalid monotonic MTP phase interval')
            phase_totals[phase] += interval[1] - interval[0]
            intervals.append((interval[0], interval[1], phase))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError('overlapping phase intervals cannot be summed as elapsed time')
    put('mtp_proposed', proposed_total)
    put('mtp_accepted', accepted_total)
    put('mtp_acceptance_rate', accepted_total / proposed_total if proposed_total else None,
        None if proposed_total else 'no proposed tokens were recorded')
    put('mtp_acceptance_by_position', [
        {'position': position + 1, 'accepted': numerators[position],
         'proposed': denominators[position],
         'fraction': numerators[position] / denominators[position]
         if denominators[position] else None}
        for position in range(depth)])
    put('mtp_committed_per_verification', committed_total / len(events))
    put('target_evaluated_rows', target_rows)
    put('draft_s', phase_totals['draft'] / 1e9)
    put('verify_s', phase_totals['verify'] / 1e9)
    put('correction_s', phase_totals['correction'] / 1e9)
    put('rollback_work', rollback_total)
    if any(not math.isfinite(entry['value']) for entry in output.values()
           if entry['value'] is not None and isinstance(entry['value'], float)):
        raise ValueError('nonfinite native MTP aggregate')
    return output
