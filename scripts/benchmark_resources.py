#!/usr/bin/env python3
"""Explicit-source Linux telemetry parsers and conservative budget decisions.

Callers supply snapshots; importing this module never probes the workstation.
Counter interpretation and cadence require separate calibration. This is not a
GPU hang detector or proof that a requested model allocation will fit.
"""
import math


def meminfo(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        key = fields[0].rstrip(':') if fields else ''
        if key not in ('MemAvailable', 'SwapFree', 'SwapTotal'):
            continue
        if len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit() or key in result:
            raise ValueError('invalid memory counter')
        result[key] = int(fields[1]) * 1024
    if set(result) != {'MemAvailable', 'SwapFree', 'SwapTotal'}:
        raise ValueError('missing memory counter')
    if result['SwapFree'] > result['SwapTotal']:
        raise ValueError('inconsistent swap counters')
    return result


def vmstat(text, page_size):
    if type(page_size) is not int or page_size <= 0:
        raise ValueError('positive page size required')
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ('pswpin', 'pswpout'):
            if len(fields) != 2 or not fields[1].isdigit() or fields[0] in result:
                raise ValueError('invalid swap counter')
            result[fields[0]] = int(fields[1]) * page_size
    if set(result) != {'pswpin', 'pswpout'}:
        raise ValueError('missing swap counter')
    return result


def budget_rejection(snapshot, limits, initial_swap, *, require_device_health=True):
    """Missing safety readings fail closed. Limits are externally reviewed inputs.

    snapshots use bytes, Celsius, monotonic cumulative swap byte counters and
    elapsed seconds. VRAM reserve includes display/allocation headroom. No default
    thermal limit is invented. Paging trends are evaluated by PagingMonitor,
    not by treating one global counter increment as evidence of memory pressure.
    Unknown device health is allowed only for an explicitly scoped CPU diagnostic.
    A known device failure always stops. This function never evicts a workload.
    """
    required_limits = {'ram_reserve_bytes', 'vram_reserve_bytes', 'disk_reserve_bytes',
                       'max_temperature_c', 'max_elapsed_s', 'max_capture_bytes'}
    if set(limits) != required_limits or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in limits.values()):
        raise ValueError('complete positive finite reviewed limits required')
    required_snapshot = {'mem_available_bytes', 'vram_free_bytes', 'disk_free_bytes',
                         'temperature_c', 'elapsed_s', 'capture_bytes',
                         'swap_in_bytes', 'swap_out_bytes', 'device_healthy'}
    if type(require_device_health) is not bool:
        raise ValueError('explicit device-health requirement must be boolean')
    if set(snapshot) != required_snapshot or (type(snapshot['device_healthy']) is not bool
            and not (snapshot['device_healthy'] is None and not require_device_health)):
        return 'missing or invalid safety reading'
    if any(type(snapshot[k]) not in (int, float) or not math.isfinite(snapshot[k]) or snapshot[k] < 0
           for k in required_snapshot - {'device_healthy'}):
        return 'missing or invalid safety reading'
    if set(initial_swap) != {'swap_in_bytes', 'swap_out_bytes'} or any(
            type(v) is not int or v < 0 for v in initial_swap.values()):
        raise ValueError('initial swap counters required')
    if snapshot['device_healthy'] is False:
        return 'device health failure'
    for reading, limit in [('mem_available_bytes', 'ram_reserve_bytes'),
                           ('vram_free_bytes', 'vram_reserve_bytes'),
                           ('disk_free_bytes', 'disk_reserve_bytes')]:
        if snapshot[reading] < limits[limit]:
            return reading + ' reserve breach'
    for reading, limit in [('temperature_c', 'max_temperature_c'),
                           ('elapsed_s', 'max_elapsed_s'), ('capture_bytes', 'max_capture_bytes')]:
        if snapshot[reading] >= limits[limit]:
            return reading + ' limit reached'
    return None


class PagingMonitor:
    """Record global paging, stop only on sustained paging with low headroom.

    Policy v2 defaults: >=16 MiB/s combined swap-in/out in consecutive >=1 s
    observation windows for >=3 s, with MemAvailable <= twice the hard RAM reserve
    throughout each window. These are diagnostic policy choices, not kernel
    guarantees or process attribution. The separate hard RAM reserve still acts
    immediately. A single burst, counter reset or gap records a warning and cannot
    manufacture a sustained trend. State is constant-size regardless of run length.
    """
    def __init__(self, ram_reserve_bytes, *, rate_bytes_per_s=16 * 1024**2,
                 sustained_s=3, window_s=1, max_gap_s=1):
        settings = (ram_reserve_bytes, rate_bytes_per_s, sustained_s, window_s, max_gap_s)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in settings):
            raise ValueError('positive finite paging policy required')
        if sustained_s < window_s or max_gap_s > window_s:
            raise ValueError('paging windows must cover consecutive observations')
        self.headroom = 2 * ram_reserve_bytes
        self.rate_limit, self.sustained_s = rate_bytes_per_s, sustained_s
        self.window_s, self.max_gap_s = window_s, max_gap_s
        self.previous = None
        self.window = None
        self.high_for_s = 0

    def update(self, timestamp_s, swap_in_bytes, swap_out_bytes, mem_available_bytes):
        if type(timestamp_s) not in (int, float) or not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError('valid paging timestamp required')
        if any(type(v) is not int or v < 0 for v in (swap_in_bytes, swap_out_bytes, mem_available_bytes)):
            raise ValueError('nonnegative integer memory/paging readings required')
        current = (timestamp_s, swap_in_bytes, swap_out_bytes)
        result = {'rejection': None, 'warning': None, 'delta_bytes': 0,
                  'window_rate_bytes_per_s': None, 'pressure_duration_s': 0,
                  'scope': 'system-wide counters; responsible process unknown'}
        if self.previous is not None:
            if timestamp_s <= self.previous[0]:
                raise ValueError('paging timestamps must increase')
            reset = swap_in_bytes < self.previous[1] or swap_out_bytes < self.previous[2]
            gap = timestamp_s - self.previous[0] > self.max_gap_s
            if reset or gap:
                result['warning'] = 'swap counter reset; trend rebased' if reset else 'paging observation gap; trend rebased'
                self.window, self.high_for_s = None, 0
            else:
                result['delta_bytes'] = swap_in_bytes - self.previous[1] + swap_out_bytes - self.previous[2]
                if result['delta_bytes']:
                    result['warning'] = 'paging observed; attribution unknown'
        if self.window is None:
            self.window = (*current, mem_available_bytes)
        else:
            start, first_in, first_out, maximum_available = self.window
            maximum_available = max(maximum_available, mem_available_bytes)
            duration = timestamp_s - start
            if duration >= self.window_s:
                rate = (swap_in_bytes - first_in + swap_out_bytes - first_out) / duration
                self.high_for_s = self.high_for_s + duration if rate >= self.rate_limit and maximum_available <= self.headroom else 0
                result['window_rate_bytes_per_s'] = rate
                self.window = (*current, mem_available_bytes)
                if self.high_for_s >= self.sustained_s:
                    result['rejection'] = 'sustained paging under memory pressure'
            else:
                self.window = (start, first_in, first_out, maximum_available)
        self.previous = current
        result['pressure_duration_s'] = self.high_for_s
        return result


def process_status(text):
    """Parse selected process memory fields only; discard identity information."""
    names = {'VmRSS': 'rss_bytes', 'VmHWM': 'high_water_rss_bytes',
             'RssAnon': 'anonymous_bytes', 'RssFile': 'file_backed_bytes'}
    result = {}
    for line in text.splitlines():
        fields = line.split()
        name = fields[0].rstrip(':') if fields else ''
        if name in names:
            if len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit() or names[name] in result:
                raise ValueError('invalid process memory counter')
            result[names[name]] = int(fields[1]) * 1024
    if not result:
        raise ValueError('missing process memory counters')
    return result


def process_pss(text):
    """PSS from smaps_rollup, without retaining address ranges or mappings."""
    found = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Pss:':
            if len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit():
                raise ValueError('invalid PSS counter')
            found.append(int(fields[1]) * 1024)
    if len(found) != 1:
        raise ValueError('missing or duplicate PSS counter')
    return found[0]


def process_counters(text, ticks_per_second):
    """proc stat: cumulative minor+major faults and user+system core seconds.

    The command name is discarded, including embedded spaces/parentheses. The
    start tick identifies PID reuse within a private sample stream. No child
    CPU/fault counters are included or silently attributed to the parent.
    """
    if type(ticks_per_second) is not int or ticks_per_second <= 0:
        raise ValueError('positive clock tick rate required')
    end = text.rfind(')')
    if '(' not in text or end < text.find('('):
        raise ValueError('invalid process stat')
    fields = text[end + 1:].split()
    indices = (7, 9, 11, 12, 19)
    if len(fields) <= max(indices) or any(not fields[i].isdigit() for i in indices):
        raise ValueError('invalid process stat counters')
    return {'page_faults': int(fields[7]) + int(fields[9]),
            'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
            'cpu_core_seconds': (int(fields[11]) + int(fields[12])) / ticks_per_second,
            'process_start_ticks': int(fields[19])}


def linux_metrics(*, memory=None, swap=None, status=None, smaps=None,
                  process_stat=None, page_size=None, ticks_per_second=None, gpu=None):
    """Adapt explicitly supplied snapshots; no file access or device discovery.

    Missing/invalid sources return null with reason. The caller records actual
    page size/tick rate, timestamp/cadence, selected sensor identity and provenance.
    GPU values are samples, not calibrated counters or process allocations. Power
    input is microwatts, not the device power cap. Energy requires integration.
    """
    units = {'rss_bytes': 'bytes', 'high_water_rss_bytes': 'bytes',
             'anonymous_bytes': 'bytes', 'file_backed_bytes': 'bytes', 'pss_bytes': 'bytes',
             'mem_available_bytes': 'bytes', 'swap_in_bytes': 'bytes', 'swap_out_bytes': 'bytes',
             'page_faults': 'count', 'minor_faults': 'count', 'major_faults': 'count', 'cpu_core_seconds': 'core*s',
             'vram_used_bytes': 'bytes', 'vram_total_bytes': 'bytes',
             'gpu_busy_fraction': 'fraction', 'temperature_c': 'C',
             'clock_hz': 'Hz', 'board_power_w': 'W', 'process_start_ticks': 'ticks'}
    metrics = {key: {'value': None, 'reason': 'source not supplied', 'unit': unit}
               for key, unit in units.items()}

    def update(keys, raw, parse):
        if raw is None:
            return
        try:
            values = parse(raw)
            for key in keys:
                if key in values:
                    metrics[key].update(value=values[key], reason=None)
                else:
                    metrics[key]['reason'] = 'counter absent from source'
        except (ValueError, TypeError, OverflowError):
            for key in keys:
                metrics[key]['reason'] = 'invalid source counters'

    update(['mem_available_bytes'], memory,
           lambda text: {'mem_available_bytes': meminfo(text)['MemAvailable']})
    def parse_swap(text):
        counters = vmstat(text, page_size)
        return {'swap_in_bytes': counters['pswpin'], 'swap_out_bytes': counters['pswpout']}

    update(['swap_in_bytes', 'swap_out_bytes'], swap, parse_swap)
    update(['rss_bytes', 'high_water_rss_bytes', 'anonymous_bytes', 'file_backed_bytes'], status, process_status)
    update(['pss_bytes'], smaps, lambda text: {'pss_bytes': process_pss(text)})
    update(['page_faults', 'minor_faults', 'major_faults', 'cpu_core_seconds', 'process_start_ticks'], process_stat,
           lambda text: process_counters(text, ticks_per_second))
    gpu = gpu or {}
    conversions = {'vram_used_bytes': ('vram_used_bytes', 1),
                   'vram_total_bytes': ('vram_total_bytes', 1),
                   'gpu_busy_percent': ('gpu_busy_fraction', .01),
                   'temperature_millicelsius': ('temperature_c', .001),
                   'clock_hz': ('clock_hz', 1), 'power_microwatts': ('board_power_w', 1e-6)}
    if set(gpu) - set(conversions):
        raise ValueError('unexpected GPU input field')
    for source, value in gpu.items():
        target, factor = conversions[source]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (
                source == 'gpu_busy_percent' and value > 100):
            metrics[target]['reason'] = 'invalid source counter'
        else:
            metrics[target].update(value=value * factor, reason=None)
    used, total = (metrics[key]['value'] for key in ['vram_used_bytes', 'vram_total_bytes'])
    if used is not None and total is not None and used > total:
        for key in ['vram_used_bytes', 'vram_total_bytes']:
            metrics[key].update(value=None, reason='inconsistent VRAM counters')
    return metrics


def sampled_energy(samples):
    """Trapezoidal board energy over timestamped power samples, without extrapolation.

    samples: ordered (monotonic_ns, watts). Coverage begins/ends at actual samples;
    the caller must report gaps/cadence and keep response/prefill boundaries apart.
    """
    if len(samples) < 2:
        raise ValueError('at least two power samples required')
    for timestamp, watts in samples:
        if type(timestamp) is not int or timestamp < 0 or type(watts) not in (int, float) or not math.isfinite(watts) or watts < 0:
            raise ValueError('invalid power sample')
    spans = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    if any(span <= 0 for span in spans):
        raise ValueError('strictly increasing power timestamps required')
    return {'energy_j': math.fsum((a[1] + b[1]) * .5 * span / 1e9
                                  for a, b, span in zip(samples, samples[1:], spans)),
            'coverage_s': (samples[-1][0] - samples[0][0]) / 1e9,
            'max_gap_s': max(spans) / 1e9, 'scope': 'sampled board power only'}


class ResourceGuard:
    """Connect explicit snapshot acquisition to the supervisor telemetry callback.

    provider(pid_or_none) supplies budget_rejection's safety readings except
    elapsed_s/capture_bytes. Missing applicable health/thermal readings reject
    preflight; a CPU diagnostic can explicitly make unknown GPU health optional.
    Successful file reads must not be substituted for a validated health signal.
    The same provider must represent the same selected device throughout a run.
    """
    def __init__(self, provider, limits, *, require_device_health=True):
        self.provider, self.limits = provider, dict(limits)
        self.require_device_health = require_device_health
        self.previous_swap = None
        self.paging = PagingMonitor(limits['ram_reserve_bytes'])

    def __call__(self, pid, elapsed_s, capture_bytes):
        snapshot = dict(self.provider(pid))
        snapshot.update(elapsed_s=elapsed_s, capture_bytes=capture_bytes)
        counters = {key: snapshot.get(key) for key in ('swap_in_bytes', 'swap_out_bytes')}
        if any(type(value) is not int or value < 0 for value in counters.values()):
            return {'readings': snapshot, 'rejection': 'missing or invalid swap reading'}
        initial = self.previous_swap if self.previous_swap is not None else counters
        rejection = budget_rejection(snapshot, self.limits, initial,
                                     require_device_health=self.require_device_health)
        paging = None
        if rejection is None:
            paging = self.paging.update(elapsed_s, counters['swap_in_bytes'],
                                        counters['swap_out_bytes'], snapshot['mem_available_bytes'])
            rejection = paging['rejection']
        self.previous_swap = counters
        return {'readings': snapshot, 'rejection': rejection, 'paging': paging}
