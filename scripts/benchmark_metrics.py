#!/usr/bin/env python3
"""Reduce explicitly supplied telemetry samples to private run-level metrics.

This module reads no files and probes no hardware. It preserves the unit and
missing-reason contract from benchmark_core and refuses to infer measurements
from incomplete series, counter resets, process reuse or unspecified steady
windows.
"""
import math

from benchmark_core import METRIC_UNITS, canonical, delivered_metrics, missing_metrics, percentile
from benchmark_resources import sampled_energy


SOURCE_UNITS = {
    'vram_used_bytes': 'bytes', 'rss_bytes': 'bytes', 'pss_bytes': 'bytes',
    'anonymous_bytes': 'bytes', 'file_backed_bytes': 'bytes',
    'high_water_rss_bytes': 'bytes', 'mem_available_bytes': 'bytes',
    'page_faults': 'count', 'swap_in_bytes': 'bytes', 'swap_out_bytes': 'bytes',
    'gpu_busy_fraction': 'fraction', 'cpu_core_seconds': 'core*s',
    'temperature_c': 'C', 'clock_hz': 'Hz', 'board_power_w': 'W',
    'process_start_ticks': 'ticks',
}


def merge_metric_sources(sources):
    """Join complete metric inventories without hiding conflicting producers.

    `sources` is an ordered mapping of source name to a complete metric mapping.
    Values must carry the shared unit and missing-reason contract. A later
    source may fill an unavailable value; two different measured values reject
    rather than silently selecting a preferred producer.
    """
    if not isinstance(sources, dict) or not sources:
        raise ValueError('named metric sources required')
    merged = missing_metrics('no metric producer supplied this field')
    provenance = {key: [] for key in METRIC_UNITS}
    for source_name, inventory in sources.items():
        if not isinstance(source_name, str) or not source_name.strip() or not isinstance(inventory, dict) or set(inventory) != set(METRIC_UNITS):
            raise ValueError('each named source must provide the complete shared metric inventory')
        for key, entry in inventory.items():
            if not isinstance(entry, dict) or set(entry) != {'value', 'reason', 'unit'} or entry['unit'] != METRIC_UNITS[key]:
                raise ValueError(f'{source_name}.{key} has an invalid metric schema or unit')
            value, reason = entry['value'], entry['reason']
            if value is None:
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f'{source_name}.{key} is missing without a reason')
                if merged[key]['value'] is None:
                    merged[key] = dict(entry)
                continue
            if reason is not None or isinstance(value, bool):
                raise ValueError(f'{source_name}.{key} has invalid value/reason pairing')
            if type(value) in (int, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{source_name}.{key} has invalid numeric value')
            try:
                canonical(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f'{source_name}.{key} is not finite JSON data') from None
            prior = merged[key]['value']
            if prior is not None and prior != value:
                raise ValueError(f'conflicting measured values for {key}')
            merged[key] = dict(entry)
            provenance[key].append(source_name)
    return {'metrics': merged, 'metric_sources': provenance}


def delivery_metric_inventory(admitted_ns, completed_ns, token_times_ns):
    """Adapt explicit monotonic committed-token delivery timestamps.

    This reports user-visible delivery latency/rate only; it does not claim a
    synchronized device phase or target-only decode-loop rate.
    """
    delivered = delivered_metrics(admitted_ns, completed_ns, token_times_ns)
    metrics = missing_metrics('no committed-token delivery metric source')
    for key, value in delivered.items():
        metrics[key] = {'value': value,
                        'reason': None if value is not None else
                        ('no committed output was delivered' if key == 'ttft_s' else
                         'at least two committed output timestamps are required'
                         if key == 'steady_output_tokens_per_s' else
                         'delivery completion boundary unavailable'),
                        'unit': METRIC_UNITS[key]}
    return metrics


def summarize_resource_metrics(samples, *, steady_start_ns=None, idle_samples=None,
                               committed_output_tokens=None, max_weighted_gap_s=None):
    """Summarize normalized per-sample resource records.

    `samples` is an ordered list of `{timestamp_ns, metrics}` where metrics is
    the mapping returned by `benchmark_resources.linux_metrics`. Optional
    `idle_samples` must be acquired separately before the workload. The caller
    declares `steady_start_ns` after warmup; this function never guesses it.
    Time-weighted utilization/clock means are produced only when the caller also
    supplies a maximum acceptable sample gap. Board energy reports actual sample
    coverage and its largest gap, without extrapolating beyond endpoints.
    """
    if not isinstance(samples, list) or not samples:
        raise ValueError('nonempty ordered telemetry samples required')
    if steady_start_ns is not None and (type(steady_start_ns) is not int or steady_start_ns < 0):
        raise ValueError('steady-state boundary must be monotonic nanoseconds')
    if committed_output_tokens is not None and (
            type(committed_output_tokens) is not int or committed_output_tokens < 0):
        raise ValueError('committed output count must be a nonnegative integer')
    if max_weighted_gap_s is not None and (
            type(max_weighted_gap_s) not in (int, float) or not math.isfinite(max_weighted_gap_s)
            or max_weighted_gap_s <= 0):
        raise ValueError('maximum weighted-sample gap must be positive and finite')

    timestamps, records = [], []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {'timestamp_ns', 'metrics'}:
            raise ValueError('each sample requires timestamp_ns and metrics')
        timestamp, metrics = sample['timestamp_ns'], sample['metrics']
        if type(timestamp) is not int or timestamp < 0 or not isinstance(metrics, dict):
            raise ValueError('invalid telemetry timestamp or metric mapping')
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError('telemetry timestamps must strictly increase')
        timestamps.append(timestamp)
        records.append(metrics)

    if idle_samples is not None:
        if not isinstance(idle_samples, list) or not idle_samples:
            idle_records = None
            idle_validation_reason = 'separate idle device sample list is empty or invalid'
        else:
            idle_records, idle_times = [], []
            idle_validation_reason = None
            for sample in idle_samples:
                if not isinstance(sample, dict) or set(sample) != {'timestamp_ns', 'metrics'}:
                    idle_validation_reason = 'idle sample requires timestamp_ns and metrics'
                    break
                timestamp = sample['timestamp_ns']
                if type(timestamp) is not int or timestamp < 0 or (idle_times and timestamp <= idle_times[-1]):
                    idle_validation_reason = 'idle timestamps must be nonnegative and strictly increasing'
                    break
                if not isinstance(sample['metrics'], dict):
                    idle_validation_reason = 'idle metrics mapping is invalid'
                    break
                idle_times.append(timestamp)
                idle_records.append(sample['metrics'])
    else:
        idle_records, idle_validation_reason = None, 'separate idle device samples not supplied'

    output = missing_metrics('no validated run-level adapter for this metric')

    def read_series(records_to_read, name):
        expected_unit = SOURCE_UNITS[name]
        values = []
        for index, record in enumerate(records_to_read):
            entry = record.get(name)
            if not isinstance(entry, dict) or set(entry) != {'value', 'reason', 'unit'}:
                return None, f'{name} missing or malformed at sample {index}'
            if entry['unit'] != expected_unit:
                return None, f'{name} unit mismatch at sample {index}'
            value, reason = entry['value'], entry['reason']
            if value is None:
                if not isinstance(reason, str) or not reason.strip():
                    return None, f'{name} missing without reason at sample {index}'
                return None, f'{name} unavailable at sample {index}: {reason}'
            if reason is not None or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return None, f'{name} invalid at sample {index}'
            values.append(value)
        return values, None

    def put(key, value, reason=None):
        if value is None:
            output[key] = {'value': None, 'reason': reason or 'source unavailable',
                           'unit': METRIC_UNITS[key]}
        else:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                output[key] = {'value': None, 'reason': 'aggregate is invalid or nonfinite',
                               'unit': METRIC_UNITS[key]}
            else:
                output[key] = {'value': value, 'reason': None, 'unit': METRIC_UNITS[key]}

    def aggregate_max(source, target):
        values, reason = read_series(records, source)
        put(target, max(values) if values is not None else None, reason)

    def aggregate_min(source, target):
        values, reason = read_series(records, source)
        put(target, min(values) if values is not None else None, reason)

    def delta(source, target, *, process_identity=False):
        values, reason = read_series(records, source)
        if values is None:
            put(target, None, reason)
            return
        if process_identity:
            identities, identity_reason = read_series(records, 'process_start_ticks')
            if identities is None or len(set(identities)) != 1:
                put(target, None, identity_reason or 'process identity changed during capture')
                return
        if any(current < previous for previous, current in zip(values, values[1:])):
            put(target, None, f'{source} counter reset during capture')
            return
        put(target, values[-1] - values[0])

    for source, target in (
        ('rss_bytes', 'rss_bytes'), ('pss_bytes', 'pss_bytes'),
        ('anonymous_bytes', 'anonymous_bytes'), ('file_backed_bytes', 'file_backed_bytes'),
        ('high_water_rss_bytes', 'high_water_rss_bytes'), ('vram_used_bytes', 'vram_peak_bytes'),
    ):
        aggregate_max(source, target)
    aggregate_min('mem_available_bytes', 'mem_available_bytes')
    delta('page_faults', 'page_faults', process_identity=True)
    delta('cpu_core_seconds', 'cpu_core_seconds', process_identity=True)
    delta('swap_in_bytes', 'swap_in_bytes')
    delta('swap_out_bytes', 'swap_out_bytes')

    temperatures, temp_reason = read_series(records, 'temperature_c')
    put('temperature_start_c', temperatures[0] if temperatures is not None else None, temp_reason)
    put('temperature_end_c', temperatures[-1] if temperatures is not None else None, temp_reason)

    if idle_records is None:
        put('vram_idle_bytes', None, idle_validation_reason)
    else:
        idle_values, idle_reason = read_series(idle_records, 'vram_used_bytes')
        put('vram_idle_bytes', percentile(idle_values, .5) if idle_values is not None else None,
            idle_reason)

    if steady_start_ns is None:
        steady_records, steady_times = [], []
        steady_reason = 'steady-state boundary not supplied after warmup'
    else:
        selected = [(t, m) for t, m in zip(timestamps, records) if t >= steady_start_ns]
        steady_times = [t for t, _ in selected]
        steady_records = [m for _, m in selected]
        steady_reason = 'no samples at or after declared steady-state boundary'
    if steady_records:
        values, reason = read_series(steady_records, 'vram_used_bytes')
        put('vram_steady_p50_bytes', percentile(values, .5) if values is not None else None,
            reason)
        put('vram_steady_p95_bytes', percentile(values, .95) if values is not None else None,
            reason)
    else:
        put('vram_steady_p50_bytes', None, steady_reason)
        put('vram_steady_p95_bytes', None, steady_reason)

    def time_weighted_mean(source, target):
        if max_weighted_gap_s is None:
            put(target, None, 'maximum acceptable sample gap was not specified')
            return
        values, reason = read_series(records, source)
        if values is None:
            put(target, None, reason)
            return
        gaps = [(b - a) / 1e9 for a, b in zip(timestamps, timestamps[1:])]
        if not gaps:
            put(target, None, 'at least two samples are required')
            return
        if max(gaps) > max_weighted_gap_s:
            put(target, None, f'sample gap {max(gaps):.9g}s exceeds declared {max_weighted_gap_s:.9g}s')
            return
        area = math.fsum((a + b) * .5 * gap
                         for a, b, gap in zip(values, values[1:], gaps))
        put(target, area / math.fsum(gaps))

    time_weighted_mean('gpu_busy_fraction', 'gpu_busy_fraction')
    time_weighted_mean('clock_hz', 'clock_hz')

    rss_values, rss_reason = read_series(records, 'rss_bytes')
    identities, identity_reason = read_series(records, 'process_start_ticks')
    if rss_values is None:
        put('memory_growth_bytes', None, rss_reason)
    elif identities is None or len(set(identities)) != 1:
        put('memory_growth_bytes', None, identity_reason or 'process identity changed during capture')
    else:
        put('memory_growth_bytes', max(rss_values) - rss_values[0])

    power, power_reason = read_series(records, 'board_power_w')
    energy_coverage_s, energy_max_gap_s = None, None
    if power is None:
        put('board_energy_j', None, power_reason)
    elif len(power) < 2:
        put('board_energy_j', None, 'at least two board-power samples are required')
    else:
        energy = sampled_energy(list(zip(timestamps, power)))
        put('board_energy_j', energy['energy_j'])
        energy_coverage_s, energy_max_gap_s = energy['coverage_s'], energy['max_gap_s']
    energy_value = output['board_energy_j']['value']
    if energy_value is None:
        put('energy_j_per_committed_token', None, output['board_energy_j']['reason'])
    elif committed_output_tokens is None:
        put('energy_j_per_committed_token', None, 'committed output token count not supplied')
    elif committed_output_tokens == 0:
        put('energy_j_per_committed_token', None, 'no committed output tokens')
    else:
        put('energy_j_per_committed_token', energy_value / committed_output_tokens)

    # These values require event, allocation or phase-specific adapters that
    # current snapshots do not provide. Keep their individual missing reasons.
    for key, reason in (
        ('process_device_bytes', 'no validated per-process device-allocation source'),
        ('pcie_h2d_bytes', 'no validated PCIe transfer counter source'),
        ('pcie_d2h_bytes', 'no validated PCIe transfer counter source'),
        ('errors', 'run event/error counter source not supplied'),
        ('resets', 'run reset event source not supplied'),
    ):
        put(key, None, reason)

    coverage = {
        'sample_count': len(samples),
        'maximum_sample_gap_s': max((b - a) / 1e9 for a, b in zip(timestamps, timestamps[1:]))
        if len(timestamps) > 1 else None,
        'steady_sample_count': len(steady_records),
        'steady_start_ns': steady_start_ns,
        'weighted_mean_max_gap_s': max_weighted_gap_s,
        'paging_scope': 'system-wide cumulative counters; process attribution unknown',
        'energy_scope': 'sampled board power integrated only between supplied endpoints',
        'energy_coverage_s': energy_coverage_s,
        'energy_max_gap_s': energy_max_gap_s,
    }
    return {'metrics': output, 'coverage': coverage}


def summarize_resource_guard_telemetry(rows, *, steady_start_ns=None, idle_rows=None,
                                       committed_output_tokens=None,
                                       max_weighted_gap_s=None):
    """Adapt captured ResourceGuard rows, then summarize running samples only.

    Preflight rows are excluded from run metrics. Temperature start/end use the
    hottest selected sensor in each sample; those endpoints can refer to
    different sensors. Raw sensor names and snapshots are not copied to output.
    """
    if not isinstance(rows, list):
        raise ValueError('telemetry rows must be a list')

    def adapt(row):
        if not isinstance(row, dict) or not isinstance(row.get('snapshot'), dict):
            raise ValueError('captured telemetry row shape invalid')
        snapshot = row['snapshot']
        values = {}

        def direct(name, source, unit):
            # ResourceGuard process fields use plain scalar values; PSS uses
            # the linux_metrics {value, reason} envelope.
            if isinstance(source, dict) and set(source) == {'value', 'reason'}:
                if source['reason'] is None and type(source['value']) in (int, float) and math.isfinite(source['value']) and source['value'] >= 0:
                    values[name] = {'value': source['value'], 'reason': None, 'unit': unit}
                    return
                reason = source.get('reason')
                values[name] = {'value': None, 'reason': reason if isinstance(reason, str) and reason else 'source missing or invalid in captured ResourceGuard sample', 'unit': unit}
            elif type(source) in (int, float) and math.isfinite(source) and source >= 0:
                values[name] = {'value': source, 'reason': None, 'unit': unit}
            else:
                values[name] = {'value': None, 'reason': 'source missing or invalid in captured ResourceGuard sample',
                                'unit': unit}

        memory = snapshot.get('memory')
        direct('mem_available_bytes', memory.get('MemAvailable') if isinstance(memory, dict) else None, 'bytes')
        swap = snapshot.get('swap')
        direct('swap_in_bytes', swap.get('pswpin') if isinstance(swap, dict) else None, 'bytes')
        direct('swap_out_bytes', swap.get('pswpout') if isinstance(swap, dict) else None, 'bytes')
        process = snapshot.get('process')
        process_memory = process.get('memory') if isinstance(process, dict) else None
        process_counters = process.get('counters') if isinstance(process, dict) else None
        for name in ('rss_bytes', 'pss_bytes', 'anonymous_bytes', 'file_backed_bytes', 'high_water_rss_bytes'):
            if name == 'pss_bytes':
                direct(name, process.get('pss_bytes') if isinstance(process, dict) else None, 'bytes')
            else:
                direct(name, process_memory.get(name) if isinstance(process_memory, dict) else None, 'bytes')
        for name, unit in (('page_faults', 'count'), ('cpu_core_seconds', 'core*s'),
                           ('process_start_ticks', 'ticks')):
            direct(name, process_counters.get(name) if isinstance(process_counters, dict) else None, unit)
        direct('vram_used_bytes', snapshot.get('vram_used_bytes'), 'bytes')
        temperatures = snapshot.get('temperatures_c')
        if isinstance(temperatures, dict) and temperatures and all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in temperatures.values()):
            direct('temperature_c', max(temperatures.values()), 'C')
        else:
            direct('temperature_c', None, 'C')
        busy = snapshot.get('gpu_busy_percent')
        if type(busy) in (int, float) and math.isfinite(busy) and 0 <= busy <= 100:
            values['gpu_busy_fraction'] = {'value': busy / 100, 'reason': None, 'unit': 'fraction'}
        else:
            values['gpu_busy_fraction'] = {'value': None, 'reason': 'GPU busy percentage missing or invalid', 'unit': 'fraction'}
        # The current ResourceGuard worker does not collect these signals.
        values['clock_hz'] = {'value': None, 'reason': 'clock telemetry is not in the captured ResourceGuard schema', 'unit': 'Hz'}
        values['board_power_w'] = {'value': None, 'reason': 'board power telemetry is not in the captured ResourceGuard schema', 'unit': 'W'}
        return {'timestamp_ns': row.get('timestamp_ns'), 'metrics': values}

    running = [adapt(row) for row in rows if isinstance(row, dict) and row.get('stage') == 'running']
    if not running:
        raise ValueError('no running ResourceGuard samples supplied')
    idle = None
    if idle_rows is not None:
        if not isinstance(idle_rows, list):
            raise ValueError('idle telemetry rows must be a list')
        idle = [adapt(row) for row in idle_rows if isinstance(row, dict) and row.get('stage') == 'running']
        if not idle:
            idle = []
    result = summarize_resource_metrics(
        running, steady_start_ns=steady_start_ns, idle_samples=idle,
        committed_output_tokens=committed_output_tokens,
        max_weighted_gap_s=max_weighted_gap_s)
    result['coverage']['source'] = 'captured ResourceGuard telemetry; running rows only'
    result['coverage']['temperature_scope'] = (
        'hottest selected sensor per sample; start/end may refer to different sensors')
    return result


def resource_guard_metric_adapter(*, additional_sources=None, steady_start_ns=None,
                                  idle_rows=None, committed_output_tokens=None,
                                  max_weighted_gap_s=None):
    """Build a supervisor adapter for ResourceGuard rows and other producers.

    `additional_sources`, when supplied, is either a mapping of complete
    inventories or a callback `(result, telemetry_rows) -> mapping`. This lets a
    run-specific adapter add native phase and committed-token delivery metrics
    while keeping this reducer independent of model output formats.
    """
    if additional_sources is not None and not isinstance(additional_sources, dict) and not callable(additional_sources):
        raise ValueError('additional metric sources must be a mapping or callback')

    def adapt(result, telemetry_rows):
        resource = summarize_resource_guard_telemetry(
            telemetry_rows, steady_start_ns=steady_start_ns, idle_rows=idle_rows,
            committed_output_tokens=committed_output_tokens,
            max_weighted_gap_s=max_weighted_gap_s)
        sources = {'resource_guard': resource['metrics']}
        extra = additional_sources(result, telemetry_rows) if callable(additional_sources) else additional_sources
        if extra is not None:
            if not isinstance(extra, dict) or 'resource_guard' in extra:
                raise ValueError('additional metric source names must be unique')
            sources.update(extra)
        return {'sources': sources,
                'metadata': {'resource_guard_coverage': resource['coverage']}}

    return adapt
