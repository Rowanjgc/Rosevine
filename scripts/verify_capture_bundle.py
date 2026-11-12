#!/usr/bin/env python3
"""Recheck internal integrity and metric contracts of one private run bundle.

This is an offline consistency verifier, not an external signature, scientific
acceptance decision, or proof that a manifest describes the executed process.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

KNOWN = {'manifest.json', 'stdout.txt', 'stderr.txt', 'telemetry.jsonl',
         'result.json', 'collector-stderr.txt', 'collector-result.json',
         'collector-invalid-response.txt', 'measurement-telemetry.jsonl',
         'measurement-collector-stderr.txt', 'measurement-collector-result.json',
         'measurement-collector-invalid-response.txt'}
REQUIRED = {'manifest.json', 'stdout.txt', 'stderr.txt', 'result.json'}
MAX_JSON = 32 * 1024 * 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def reject_constant(_value):
    raise ValueError('nonfinite JSON constant')


def secure_bytes(path, max_bytes=MAX_JSON):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > max_bytes):
            raise ValueError('bundle file must be bounded, owned, regular and private')
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError('bundle file size limit exceeded')
    return raw


def secure_json(path, max_bytes=MAX_JSON):
    return json.loads(secure_bytes(path, max_bytes), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def percentile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    low, high = math.floor(point), math.ceil(point)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def resource_guard_aggregates(rows, coverage):
    """Independently reconstruct the ResourceGuard scalar aggregates used here."""
    samples = [row for row in rows if row['stage'] == 'running']
    if not samples:
        return {}, {'sample_count': 0, 'maximum_sample_gap_s': None}
    timestamps = [row['timestamp_ns'] for row in samples]
    snapshots = [row['snapshot'] for row in samples]

    def scalar_series(getter):
        values = [getter(snapshot) for snapshot in snapshots]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in values):
            return None
        return values

    def process_value(snapshot, group, name):
        process = snapshot.get('process')
        if not isinstance(process, dict):
            return None
        if name == 'pss_bytes':
            entry = process.get('pss_bytes')
            value = entry.get('value') if isinstance(entry, dict) and entry.get('reason') is None else None
            return value
        record = process.get(group)
        return record.get(name) if isinstance(record, dict) else None

    aggregates = {}
    for metric, group, raw_name in (
            ('rss_bytes', 'memory', 'rss_bytes'),
            ('anonymous_bytes', 'memory', 'anonymous_bytes'),
            ('file_backed_bytes', 'memory', 'file_backed_bytes'),
            ('high_water_rss_bytes', 'memory', 'high_water_rss_bytes'),
            ('pss_bytes', 'memory', 'pss_bytes')):
        values = scalar_series(lambda snapshot: process_value(snapshot, group, raw_name))
        aggregates[metric] = max(values) if values is not None else None
    values = scalar_series(lambda snapshot: snapshot.get('vram_used_bytes'))
    aggregates['vram_peak_bytes'] = max(values) if values is not None else None
    values = scalar_series(lambda snapshot: snapshot.get('memory', {}).get('MemAvailable')
                           if isinstance(snapshot.get('memory'), dict) else None)
    aggregates['mem_available_bytes'] = min(values) if values is not None else None

    def counter_values(group, key):
        def read(snapshot):
            record = snapshot.get(group)
            if not isinstance(record, dict):
                return None
            if group == 'process':
                counters = record.get('counters')
                return counters.get(key) if isinstance(counters, dict) else None
            return record.get(key)
        return scalar_series(read)

    starts = counter_values('process', 'process_start_ticks')
    same_process = starts is not None and len(set(starts)) == 1
    for metric, group, key, require_process in (
            ('page_faults', 'process', 'page_faults', True),
            ('cpu_core_seconds', 'process', 'cpu_core_seconds', True),
            ('swap_in_bytes', 'swap', 'pswpin', False),
            ('swap_out_bytes', 'swap', 'pswpout', False)):
        values = counter_values(group, key)
        if values is not None and (not require_process or same_process) and all(
                current >= previous for previous, current in zip(values, values[1:])):
            aggregates[metric] = values[-1] - values[0]
        else:
            aggregates[metric] = None

    temperatures = []
    for snapshot in snapshots:
        sensors = snapshot.get('temperatures_c')
        if (not isinstance(sensors, dict) or not sensors or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in sensors.values())):
            temperatures = []
            break
        temperatures.append(max(sensors.values()))
    aggregates['temperature_start_c'] = temperatures[0] if temperatures else None
    aggregates['temperature_end_c'] = temperatures[-1] if temperatures else None

    rss = scalar_series(lambda snapshot: process_value(snapshot, 'memory', 'rss_bytes'))
    aggregates['memory_growth_bytes'] = max(rss) - rss[0] if rss is not None and same_process else None

    gaps = [(right - left) / 1e9 for left, right in zip(timestamps, timestamps[1:])]
    coverage_result = {'sample_count': len(samples),
                       'maximum_sample_gap_s': max(gaps) if gaps else None}
    max_gap = coverage.get('weighted_mean_max_gap_s') if isinstance(coverage, dict) else None
    busy = scalar_series(lambda snapshot: snapshot.get('gpu_busy_percent') / 100
                         if type(snapshot.get('gpu_busy_percent')) in (int, float)
                         and 0 <= snapshot['gpu_busy_percent'] <= 100 else None)
    if (type(max_gap) in (int, float) and math.isfinite(max_gap) and max_gap > 0
            and busy is not None and len(busy) > 1 and gaps and max(gaps) <= max_gap):
        area = math.fsum((left + right) * .5 * gap
                         for left, right, gap in zip(busy, busy[1:], gaps))
        aggregates['gpu_busy_fraction'] = area / math.fsum(gaps)
    else:
        aggregates['gpu_busy_fraction'] = None
    return aggregates, coverage_result


def finite_json(value):
    try:
        json.dumps(value, allow_nan=False, separators=(',', ':'))
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def verify_bundle(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError('private run directory with mode 0700 required')
    if directory.stat().st_uid != os.getuid():
        raise ValueError('run directory must be owned by the current user')

    index = secure_json(directory / 'checksums.json')
    if not isinstance(index, dict) or not REQUIRED <= set(index) or set(index) - KNOWN:
        raise ValueError('checksum index has missing or unknown entries')
    if any(not isinstance(name, str) or not isinstance(digest, str) or
           len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest)
           for name, digest in index.items()):
        raise ValueError('invalid checksum index entry')
    for name, expected in index.items():
        actual = hashlib.sha256(secure_bytes(directory / name)).hexdigest()
        if actual != expected:
            raise ValueError('capture checksum mismatch')

    manifest = secure_json(directory / 'manifest.json')
    result = secure_json(directory / 'result.json')
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise ValueError('manifest and terminal result must be JSON objects')
    trial_identity = manifest.get('trial_identity')
    if trial_identity is not None and (not isinstance(trial_identity, dict) or not finite_json(trial_identity)):
        raise ValueError('manifest trial identity must be finite JSON object data')
    expected_artifacts = manifest.get('expected_artifacts')
    artifact_checks = result.get('artifact_checks')
    if expected_artifacts is None:
        if artifact_checks is not None:
            raise ValueError('unrequested artifact checks were stored')
    else:
        if (not isinstance(expected_artifacts, list) or len(expected_artifacts) > 64
                or not isinstance(artifact_checks, list)
                or len(artifact_checks) != len(expected_artifacts)):
            raise ValueError('artifact identity inventory is missing or invalid')
        roles = set()
        artifact_failure = False
        for expected, checked in zip(expected_artifacts, artifact_checks):
            legacy_check_schema = isinstance(checked, dict) and set(checked) == {
                'role', 'path', 'expected_sha256', 'expected_size_bytes', 'status',
                'actual_sha256', 'actual_size_bytes', 'reason'}
            if (not isinstance(expected, dict) or set(expected) != {'role', 'path', 'sha256', 'size_bytes'}
                    or not isinstance(checked, dict) or (not legacy_check_schema and set(checked) !=
                    {'role', 'path', 'expected_sha256', 'expected_size_bytes', 'status',
                     'actual_sha256', 'actual_size_bytes', 'actual_device', 'actual_inode', 'reason'})):
                raise ValueError('artifact identity entry schema is invalid')
            role = expected['role']
            digest_value = expected['sha256']
            size_value = expected['size_bytes']
            if (not isinstance(role, str) or not role or role in roles
                    or not isinstance(expected['path'], str) or not expected['path']
                    or not isinstance(digest_value, str) or len(digest_value) != 64
                    or any(ch not in '0123456789abcdef' for ch in digest_value)
                    or type(size_value) is not int or size_value <= 0
                    or checked['role'] != role or checked['path'] != expected['path']
                    or checked['expected_sha256'] != digest_value
                    or checked['expected_size_bytes'] != size_value):
                raise ValueError('artifact identity expectations do not match checks')
            roles.add(role)
            if not legacy_check_schema and ((checked['actual_device'] is not None and
                    (type(checked['actual_device']) is not int or checked['actual_device'] < 0))
                    or (checked['actual_inode'] is not None and
                    (type(checked['actual_inode']) is not int or checked['actual_inode'] <= 0))):
                raise ValueError('artifact file identity fields are invalid')
            if checked['status'] not in ('matched', 'mismatch', 'unavailable'):
                raise ValueError('artifact check status is invalid')
            if checked['status'] == 'matched':
                if (checked['actual_sha256'] != digest_value or checked['actual_size_bytes'] != size_value
                        or checked['reason'] is not None):
                    raise ValueError('matched artifact check disagrees with expectation')
                if not legacy_check_schema and (checked['actual_device'] is None or checked['actual_inode'] is None):
                    raise ValueError('matched artifact check lacks file identity')
            else:
                artifact_failure = True
                if (not isinstance(checked['reason'], str) or not checked['reason']
                        or (checked['actual_sha256'] is not None and
                            (not isinstance(checked['actual_sha256'], str)
                             or len(checked['actual_sha256']) != 64
                             or any(ch not in '0123456789abcdef' for ch in checked['actual_sha256'])))
                        or (checked['actual_size_bytes'] is not None and
                            (type(checked['actual_size_bytes']) is not int or checked['actual_size_bytes'] < 0))):
                    raise ValueError('failed artifact check fields are invalid')
        if artifact_failure and (result.get('launched') is True
                                 or result.get('status') != 'incomplete'
                                 or result.get('reason') != 'artifact_identity_check_failed'):
            raise ValueError('failed artifact identity did not block launch')
    output_limit = manifest.get('max_output_bytes')
    telemetry_limit = manifest.get('max_telemetry_bytes')
    if type(output_limit) is not int or output_limit <= 0 or type(telemetry_limit) is not int or telemetry_limit <= 0:
        raise ValueError('manifest capture limits are missing or invalid')
    output_bytes = sum((directory / name).stat().st_size for name in ('stdout.txt', 'stderr.txt'))
    if output_bytes != result.get('captured_output_bytes') or output_bytes > output_limit:
        raise ValueError('captured output byte total does not match terminal result')

    telemetry_rows = 0
    parsed_rows = []
    telemetry_path = directory / 'telemetry.jsonl'
    if telemetry_path.name in index:
        raw = secure_bytes(telemetry_path, telemetry_limit)
        if len(raw) != result.get('telemetry_bytes'):
            raise ValueError('telemetry byte total does not match terminal result')
        previous = None
        for line in raw.splitlines():
            row = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
            if (not isinstance(row, dict) or not {'timestamp_ns', 'stage', 'collection_ns', 'snapshot'} <= set(row)
                    or type(row['timestamp_ns']) is not int or type(row['collection_ns']) is not int
                    or row['timestamp_ns'] < 0 or row['collection_ns'] < 0
                    or row['stage'] not in ('preflight', 'running') or not isinstance(row['snapshot'], dict)
                    or not finite_json(row)):
                raise ValueError('invalid captured telemetry row')
            if previous is not None and row['timestamp_ns'] <= previous:
                raise ValueError('telemetry timestamps are not strictly increasing')
            previous = row['timestamp_ns']
            parsed_rows.append(row)
            telemetry_rows += 1
    elif result.get('telemetry_bytes') != 0:
        raise ValueError('terminal result declares telemetry without a capture file')

    legacy_exceptions = []
    measurement_path = directory / 'measurement-telemetry.jsonl'
    measurement_keys = {'measurement_telemetry_enabled', 'measurement_telemetry_bytes',
                        'measurement_collector_cycles', 'measurement_collector_cycle_durations_ns',
                        'measurement_collector_cycle_p95_ns'}
    legacy_measurement_schema = (
        'measurement_telemetry_enabled' not in manifest
        and not (measurement_keys & set(result))
        and measurement_path.name not in index)
    if legacy_measurement_schema:
        measurement_enabled = False
        legacy_exceptions.append('bundle predates separate optional measurement collector accounting')
    else:
        measurement_enabled = manifest.get('measurement_telemetry_enabled')
        if type(measurement_enabled) is not bool or result.get('measurement_telemetry_enabled') != measurement_enabled:
            raise ValueError('measurement collector state differs between manifest and result')
    measurement_bytes = result.get('measurement_telemetry_bytes')
    measurement_cycles = result.get('measurement_collector_cycle_durations_ns')
    if legacy_measurement_schema:
        measurement_bytes, measurement_cycles = 0, []
    elif (type(measurement_bytes) is not int or measurement_bytes < 0
          or not isinstance(measurement_cycles, list)
          or result.get('measurement_collector_cycles') != len(measurement_cycles)):
        raise ValueError('measurement collector totals are missing or invalid')
    if measurement_enabled:
        if measurement_path.name not in index:
            raise ValueError('enabled measurement collector has no retained telemetry')
        raw_measurement = secure_bytes(measurement_path, telemetry_limit)
        if len(raw_measurement) != measurement_bytes:
            raise ValueError('measurement telemetry byte total does not match terminal result')
        measurement_rows = []
        previous_measurement = None
        for line in raw_measurement.splitlines():
            row = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
            if (not isinstance(row, dict) or not {'timestamp_ns', 'stage', 'collection_ns', 'snapshot'} <= set(row)
                    or type(row['timestamp_ns']) is not int or type(row['collection_ns']) is not int
                    or row['timestamp_ns'] < 0 or row['collection_ns'] < 0
                    or row['stage'] not in ('preflight', 'running') or not isinstance(row['snapshot'], dict)
                    or 'rejection' in row['snapshot'] or not finite_json(row)):
                raise ValueError('invalid captured measurement telemetry row')
            if previous_measurement is not None and row['timestamp_ns'] <= previous_measurement:
                raise ValueError('measurement telemetry timestamps are not strictly increasing')
            previous_measurement = row['timestamp_ns']
            measurement_rows.append(row)
        if len(measurement_rows) != len(measurement_cycles):
            raise ValueError('measurement collector cycle and row counts differ')
    elif measurement_bytes != 0 or measurement_cycles or measurement_path.name in index:
        raise ValueError('disabled measurement collector has retained measurements')
    if any(type(value) is not int or value <= 0 for value in measurement_cycles):
        raise ValueError('measurement collector cycle durations are invalid')
    expected_measurement_p95 = percentile(measurement_cycles, .95) if measurement_cycles else None
    if (not legacy_measurement_schema
            and result.get('measurement_collector_cycle_p95_ns') != expected_measurement_p95):
        raise ValueError('measurement collector p95 does not match retained durations')

    cycles = result.get('collector_cycle_durations_ns')
    cycle_p95_recomputed = isinstance(cycles, list)
    if cycle_p95_recomputed:
        if any(type(value) is not int or value <= 0 for value in cycles):
            raise ValueError('collector cycle duration list is invalid')
        if result.get('collector_cycles') != len(cycles):
            raise ValueError('collector cycle count mismatch')
        expected_p95 = percentile(cycles, .95) if cycles else None
        if result.get('collector_cycle_p95_ns') != expected_p95:
            raise ValueError('collector p95 does not match retained durations')
    else:
        count, reported_p95 = result.get('collector_cycles'), result.get('collector_cycle_p95_ns')
        if type(count) is not int or count < 0 or (reported_p95 is not None and
                (type(reported_p95) not in (int, float) or not math.isfinite(reported_p95) or reported_p95 < 0)):
            raise ValueError('legacy collector summary is invalid')
        legacy_exceptions.append('full collector cycle durations absent; p95 cannot be recomputed')

    persisted_metrics = result.get('metrics')
    manifest_metrics = manifest.get('metrics')
    if persisted_metrics is None:
        if not isinstance(manifest_metrics, dict):
            raise ValueError('neither terminal nor manifest metric inventory is available')
        metrics = manifest_metrics
        legacy_exceptions.append('terminal result predates run-level metric persistence')
    else:
        metrics = persisted_metrics
    if not isinstance(metrics, dict) or not isinstance(manifest_metrics, dict) or set(metrics) != set(manifest_metrics):
        raise ValueError('run metric inventory differs from initial manifest inventory')
    legacy_metric_schema = persisted_metrics is None and all(
        isinstance(entry, dict) and set(entry) == {'value', 'reason'}
        for entry in metrics.values())
    if legacy_metric_schema:
        legacy_exceptions.append('initial metric inventory predates explicit unit fields')
    for key, entry in metrics.items():
        initial = manifest_metrics[key]
        expected_keys = {'value', 'reason'} if legacy_metric_schema else {'value', 'reason', 'unit'}
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError('run metric schema mismatch')
        if not legacy_metric_schema and (not isinstance(initial, dict) or entry.get('unit') != initial.get('unit')
                                         or not isinstance(entry.get('unit'), str) or not entry['unit']):
            raise ValueError('run metric schema or unit mismatch')
        if entry['value'] is None:
            if not isinstance(entry['reason'], str) or not entry['reason'].strip():
                raise ValueError('missing run metric lacks a reason')
        elif entry['reason'] is not None or isinstance(entry['value'], bool) or not finite_json(entry['value']):
            raise ValueError('measured run metric has invalid value/reason')

    sources = result.get('metric_sources', {})
    if not isinstance(sources, dict) or (sources and set(sources) != set(metrics)):
        raise ValueError('metric source provenance inventory is incomplete')
    for key, source_names in sources.items():
        if not isinstance(source_names, list) or any(not isinstance(name, str) or not name for name in source_names):
            raise ValueError('invalid metric source provenance')
        if metrics[key]['value'] is not None and not source_names:
            raise ValueError('measured metric has no source provenance')
    if 'metric_metadata' in result and not finite_json(result['metric_metadata']):
        raise ValueError('metric metadata is not finite JSON data')

    resource_coverage = result.get('metric_metadata', {}).get('resource_guard_coverage') if isinstance(result.get('metric_metadata'), dict) else None
    if resource_coverage is not None:
        if not isinstance(resource_coverage, dict):
            raise ValueError('ResourceGuard coverage metadata must be an object')
        recomputed, recomputed_coverage = resource_guard_aggregates(parsed_rows, resource_coverage)
        if resource_coverage.get('sample_count') != recomputed_coverage['sample_count'] or resource_coverage.get('maximum_sample_gap_s') != recomputed_coverage['maximum_sample_gap_s']:
            raise ValueError('ResourceGuard coverage metadata disagrees with raw telemetry')
        for key, expected in recomputed.items():
            if key not in metrics or 'resource_guard' not in sources.get(key, []):
                continue
            actual = metrics[key]['value']
            if expected is None or actual is None:
                if actual != expected:
                    raise ValueError('ResourceGuard metric availability disagrees with raw telemetry')
            elif type(expected) in (int, float) and type(actual) in (int, float):
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('ResourceGuard metric value disagrees with raw telemetry')
            elif actual != expected:
                raise ValueError('ResourceGuard metric value disagrees with raw telemetry')

    status = result.get('status')
    if status not in ('successful', 'failed', 'incomplete'):
        raise ValueError('invalid terminal run status')
    if status == 'successful' and (result.get('launched') is not True or result.get('returncode') != 0):
        raise ValueError('successful status conflicts with launch/return code')
    child_identity = result.get('child_process_identity')
    if child_identity is None:
        if result.get('launched') is True:
            legacy_exceptions.append('child executable identity was not retained')
    elif result.get('launched') is not True:
        raise ValueError('unlaunched result carries a child identity')
    else:
        required_identity = {'pid', 'start_ticks', 'executable_path', 'size_bytes',
                             'device', 'inode', 'sha256', 'reason'}
        if not isinstance(child_identity, dict) or set(child_identity) != required_identity:
            raise ValueError('child process identity schema is invalid')
        if type(child_identity['pid']) is not int or child_identity['pid'] <= 0:
            raise ValueError('child process identity PID is invalid')
        if child_identity['reason'] is None:
            if (type(child_identity['start_ticks']) is not int or child_identity['start_ticks'] < 0
                    or not isinstance(child_identity['executable_path'], str)
                    or type(child_identity['size_bytes']) is not int or child_identity['size_bytes'] <= 0
                    or type(child_identity['device']) is not int or type(child_identity['inode']) is not int
                    or child_identity['device'] < 0 or child_identity['inode'] <= 0
                    or not isinstance(child_identity['sha256'], str) or len(child_identity['sha256']) != 64
                    or any(ch not in '0123456789abcdef' for ch in child_identity['sha256'])):
                raise ValueError('child executable identity fields are invalid')
            cleanup = result.get('workload_cleanup')
            if not isinstance(cleanup, dict) or cleanup.get('pid') != child_identity['pid']:
                raise ValueError('child executable identity PID differs from cleanup record')
        elif (not isinstance(child_identity['reason'], str) or not child_identity['reason']
              or child_identity['sha256'] is not None):
            raise ValueError('failed child executable identity must carry only a reason')
    library_identity = result.get('loaded_library_identity')
    if manifest.get('audit_loaded_libraries') is True:
        required_library_fields = {'status', 'scope', 'sample_count', 'sample_durations_ns',
                                   'hash_duration_ns', 'unique_library_count', 'total_pinned_bytes',
                                   'reason_codes', 'libraries'}
        if not isinstance(library_identity, dict) or set(library_identity) != required_library_fields:
            raise ValueError('loaded-library identity schema is missing or invalid')
        if (library_identity['status'] not in ('complete', 'incomplete', 'not_run')
                or not isinstance(library_identity['scope'], str) or not library_identity['scope']
                or type(library_identity['sample_count']) is not int or library_identity['sample_count'] < 0
                or type(library_identity['hash_duration_ns']) is not int or library_identity['hash_duration_ns'] < 0
                or type(library_identity['unique_library_count']) is not int or library_identity['unique_library_count'] < 0
                or type(library_identity['total_pinned_bytes']) is not int or library_identity['total_pinned_bytes'] < 0
                or not isinstance(library_identity['reason_codes'], list)
                or any(not isinstance(code, str) or not code for code in library_identity['reason_codes'])
                or not isinstance(library_identity['libraries'], list)
                or library_identity['unique_library_count'] != len(library_identity['libraries'])):
            raise ValueError('loaded-library identity summary fields are invalid')
        if (library_identity['status'] == 'complete' and
                (not library_identity['libraries'] or library_identity['reason_codes'])):
            raise ValueError('complete library inventory conflicts with its entries/reason codes')
        if library_identity['status'] == 'not_run' and result.get('launched') is True:
            raise ValueError('launched result cannot have a not-run library audit')
        durations = library_identity['sample_durations_ns']
        if (not isinstance(durations, list) or len(durations) != library_identity['sample_count']
                or any(type(value) is not int or value <= 0 for value in durations)):
            raise ValueError('loaded-library sample durations are invalid')
        for library in library_identity['libraries']:
            if (not isinstance(library, dict) or set(library) !=
                    {'path', 'size_bytes', 'device', 'inode', 'sha256'}
                    or not isinstance(library['path'], str) or not library['path'].startswith('/')
                    or type(library['size_bytes']) is not int or library['size_bytes'] < 0
                    or type(library['device']) is not int or library['device'] < 0
                    or type(library['inode']) is not int or library['inode'] <= 0
                    or not isinstance(library['sha256'], str) or len(library['sha256']) != 64
                    or any(ch not in '0123456789abcdef' for ch in library['sha256'])):
                raise ValueError('loaded-library identity entry is invalid')
    elif library_identity is not None:
        raise ValueError('unrequested loaded-library identity was stored')
    mapped_identity = result.get('mapped_artifact_identity')
    audit_mapped = manifest.get('audit_mapped_artifacts', False)
    if type(audit_mapped) is not bool:
        raise ValueError('mapped-artifact audit switch must be boolean')
    if audit_mapped:
        if not isinstance(manifest.get('expected_artifacts'), list) or not manifest['expected_artifacts']:
            raise ValueError('mapped-artifact audit requires declared artifacts')
        required_mapped_fields = {'status', 'scope', 'sample_count', 'sample_durations_ns',
                                  'reason_codes', 'artifacts'}
        if not isinstance(mapped_identity, dict) or set(mapped_identity) != required_mapped_fields:
            raise ValueError('mapped-artifact identity schema is missing or invalid')
        if (mapped_identity['status'] not in ('complete', 'incomplete', 'not_run')
                or not isinstance(mapped_identity['scope'], str) or not mapped_identity['scope']
                or type(mapped_identity['sample_count']) is not int or mapped_identity['sample_count'] < 0
                or not isinstance(mapped_identity['reason_codes'], list)
                or any(not isinstance(code, str) or not code for code in mapped_identity['reason_codes'])
                or not isinstance(mapped_identity['artifacts'], list)
                or not isinstance(mapped_identity['sample_durations_ns'], list)
                or len(mapped_identity['sample_durations_ns']) != mapped_identity['sample_count']
                or any(type(value) is not int or value <= 0 for value in mapped_identity['sample_durations_ns'])):
            raise ValueError('mapped-artifact summary fields are invalid')
        expected_roles = {item['role'] for item in manifest.get('expected_artifacts', [])}
        checked_by_role = {item['role']: item for item in (artifact_checks or [])}
        actual_roles = set()
        for artifact in mapped_identity['artifacts']:
            if (not isinstance(artifact, dict) or set(artifact) !=
                    {'role', 'device', 'inode', 'size_bytes', 'mapping_observations',
                     'status', 'stable_file_identity'}
                    or not isinstance(artifact['role'], str) or artifact['role'] in actual_roles
                    or type(artifact['device']) is not int or artifact['device'] < 0
                    or type(artifact['inode']) is not int or artifact['inode'] <= 0
                    or type(artifact['size_bytes']) is not int or artifact['size_bytes'] <= 0
                    or type(artifact['mapping_observations']) is not int or artifact['mapping_observations'] < 0
                    or artifact['status'] not in ('observed', 'not_observed')
                    or type(artifact['stable_file_identity']) is not bool
                    or (artifact['status'] == 'observed') != (artifact['mapping_observations'] > 0)):
                raise ValueError('mapped-artifact identity entry is invalid')
            actual_roles.add(artifact['role'])
            checked = checked_by_role.get(artifact['role'])
            if (checked is None or checked.get('status') != 'matched'
                    or checked.get('actual_device') != artifact['device']
                    or checked.get('actual_inode') != artifact['inode']
                    or checked.get('actual_size_bytes') != artifact['size_bytes']):
                raise ValueError('mapped-artifact identity disagrees with prelaunch artifact check')
        if not actual_roles <= expected_roles:
            raise ValueError('mapped-artifact audit contains an undeclared role')
        if mapped_identity['status'] == 'complete' and actual_roles != expected_roles:
            raise ValueError('complete mapped-artifact audit lacks a declared role')
        if mapped_identity['status'] == 'not_run':
            required_reason = ('child_process_identity_unavailable'
                               if result.get('launched') is True else 'workload_not_launched')
            if (required_reason not in mapped_identity['reason_codes']
                    or mapped_identity['sample_count'] != 0 or mapped_identity['artifacts']):
                raise ValueError('not-run mapped-artifact audit conflicts with launch state')
    elif mapped_identity is not None:
        raise ValueError('unrequested mapped-artifact identity was stored')
    if not isinstance(result.get('elapsed_s'), (int, float)) or not math.isfinite(result['elapsed_s']) or result['elapsed_s'] < 0:
        raise ValueError('invalid supervision elapsed time')

    indexed_files = set(index) | {'checksums.json'}
    unindexed_regular_files = 0
    for path in directory.iterdir():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('symlink found in run directory')
        if stat.S_ISREG(info.st_mode) and path.name not in indexed_files:
            unindexed_regular_files += 1
        elif not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise ValueError('unsupported file type in run directory')
    available_metrics = sum(entry['value'] is not None for entry in metrics.values())
    return {'status': 'internally_consistent', 'run_status': status,
            'checksummed_files': len(index), 'telemetry_rows': telemetry_rows,
            'collector_cycles': result.get('collector_cycles'),
            'collector_p95_recomputed': cycle_p95_recomputed,
            'available_metrics': available_metrics, 'metric_count': len(metrics),
            'metric_inventory_persisted': persisted_metrics is not None,
            'resource_aggregates_recomputed': resource_coverage is not None,
            'child_executable_identity_available': isinstance(child_identity, dict) and child_identity.get('sha256') is not None,
            'loaded_library_audit_status': library_identity.get('status') if isinstance(library_identity, dict) else 'not_requested',
            'loaded_library_count': library_identity.get('unique_library_count', 0) if isinstance(library_identity, dict) else 0,
            'mapped_artifact_audit_status': mapped_identity.get('status') if isinstance(mapped_identity, dict) else 'not_requested',
            'mapped_artifact_count': len(mapped_identity.get('artifacts', [])) if isinstance(mapped_identity, dict) else 0,
            'mapped_artifact_identity': ({'status': mapped_identity['status'],
                                          'reason_codes': mapped_identity['reason_codes'],
                                          'artifacts': mapped_identity['artifacts']}
                                         if isinstance(mapped_identity, dict) else None),
            'legacy_exceptions': legacy_exceptions,
            'unindexed_regular_files': unindexed_regular_files,
            'manual_review_required': unindexed_regular_files > 0,
            'checksum_index_sha256': hashlib.sha256(secure_bytes(directory / 'checksums.json')).hexdigest(),
            'scope': 'internal consistency only; checksum index is not an external authenticity anchor'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = verify_bundle(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print('Capture bundle verification failed.', file=sys.stderr)
        sys.exit(1)
