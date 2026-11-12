#!/usr/bin/env python3
"""Offline benchmark building blocks; no endpoint, model launch or telemetry probe.

Tokenization/rendering is injected by an isolated reference adapter. All returned
manifests and run bundles are private, even for synthetic inputs. This module does
not approve a run envelope or infer counters from throughput.
"""
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import signal
import shutil
import statistics
import stat
import subprocess
import time

VERSION = 'rosevine-benchmark-v1'
METRICS = (
    'prefill_tokens_per_s', 'target_decode_tokens_per_s',
    'speculative_decode_tokens_per_s', 'steady_output_tokens_per_s',
    'ttft_s', 'end_to_end_s', 'mtp_proposed', 'mtp_accepted', 'mtp_acceptance_rate',
    'mtp_acceptance_by_position', 'mtp_committed_per_verification',
    'target_evaluated_rows', 'draft_s', 'verify_s', 'correction_s',
    'rollback_work', 'vram_idle_bytes', 'vram_peak_bytes',
    'vram_steady_p50_bytes', 'vram_steady_p95_bytes', 'process_device_bytes',
    'rss_bytes', 'pss_bytes', 'anonymous_bytes', 'file_backed_bytes',
    'high_water_rss_bytes', 'mem_available_bytes', 'page_faults',
    'swap_in_bytes', 'swap_out_bytes', 'gpu_busy_fraction', 'cpu_core_seconds',
    'pcie_h2d_bytes', 'pcie_d2h_bytes', 'board_energy_j',
    'energy_j_per_committed_token', 'temperature_start_c', 'temperature_end_c',
    'clock_hz', 'memory_growth_bytes', 'errors', 'resets',
)
METRIC_UNITS = {
    'prefill_tokens_per_s': 'tokens/s', 'target_decode_tokens_per_s': 'tokens/s',
    'speculative_decode_tokens_per_s': 'tokens/s', 'steady_output_tokens_per_s': 'tokens/s',
    'ttft_s': 's', 'end_to_end_s': 's', 'mtp_proposed': 'count', 'mtp_accepted': 'count',
    'mtp_acceptance_rate': 'fraction',
    'mtp_acceptance_by_position': 'fraction by position',
    'mtp_committed_per_verification': 'tokens/verification', 'target_evaluated_rows': 'count',
    'draft_s': 's', 'verify_s': 's', 'correction_s': 's', 'rollback_work': 'operation count',
    'vram_idle_bytes': 'bytes', 'vram_peak_bytes': 'bytes', 'vram_steady_p50_bytes': 'bytes',
    'vram_steady_p95_bytes': 'bytes', 'process_device_bytes': 'bytes', 'rss_bytes': 'bytes',
    'pss_bytes': 'bytes', 'anonymous_bytes': 'bytes', 'file_backed_bytes': 'bytes',
    'high_water_rss_bytes': 'bytes', 'mem_available_bytes': 'bytes', 'page_faults': 'count',
    'swap_in_bytes': 'bytes', 'swap_out_bytes': 'bytes', 'gpu_busy_fraction': 'fraction',
    'cpu_core_seconds': 'core*s', 'pcie_h2d_bytes': 'bytes', 'pcie_d2h_bytes': 'bytes',
    'board_energy_j': 'J', 'energy_j_per_committed_token': 'J/token',
    'temperature_start_c': 'C', 'temperature_end_c': 'C', 'clock_hz': 'Hz',
    'memory_growth_bytes': 'bytes', 'errors': 'count', 'resets': 'count',
}
if set(METRIC_UNITS) != set(METRICS):
    raise RuntimeError('every benchmark metric must declare exactly one unit')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def positive_integer(value):
    return type(value) is int and value > 0


def validate_expected_artifacts(artifacts):
    if artifacts is None:
        return None
    if not isinstance(artifacts, list) or len(artifacts) > 64:
        raise ValueError('expected artifacts must be a bounded list')
    seen = set()
    normalized = []
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {'role', 'path', 'sha256', 'size_bytes'}:
            raise ValueError('artifact identity entry schema is invalid')
        role, path, expected_hash, size = (item['role'], item['path'], item['sha256'], item['size_bytes'])
        if (not isinstance(role, str) or not role or len(role) > 64 or role in seen
                or not isinstance(path, str) or not path or '\x00' in path
                or not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(ch not in '0123456789abcdef' for ch in expected_hash)
                or type(size) is not int or size <= 0):
            raise ValueError('artifact identity entry fields are invalid')
        seen.add(role)
        normalized.append({'role': role, 'path': path, 'sha256': expected_hash,
                           'size_bytes': size})
    return normalized


def check_expected_artifact(item):
    record = {'role': item['role'], 'path': item['path'],
              'expected_sha256': item['sha256'], 'expected_size_bytes': item['size_bytes'],
              'status': 'unavailable', 'actual_sha256': None, 'actual_size_bytes': None,
              'actual_device': None, 'actual_inode': None, 'reason': None}
    fd = None
    try:
        resolved = Path(item['path']).resolve(strict=True)
        fd = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(fd)
        record['actual_device'], record['actual_inode'] = before.st_dev, before.st_ino
        if not stat.S_ISREG(before.st_mode):
            record['reason'] = 'not_regular_file'
            return record
        hasher = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            size += len(block)
            hasher.update(block)
        after = os.fstat(fd)
        stable_before = (before.st_dev, before.st_ino, before.st_size,
                         before.st_mtime_ns, before.st_ctime_ns)
        stable_after = (after.st_dev, after.st_ino, after.st_size,
                        after.st_mtime_ns, after.st_ctime_ns)
        if stable_before != stable_after or size != after.st_size:
            record['reason'] = 'file_changed_during_check'
            return record
        record['actual_sha256'], record['actual_size_bytes'] = hasher.hexdigest(), size
        record['status'] = ('matched' if record['actual_sha256'] == item['sha256']
                            and size == item['size_bytes'] else 'mismatch')
        if record['status'] == 'mismatch':
            record['reason'] = 'identity_mismatch'
        return record
    except (OSError, RuntimeError, ValueError) as error:
        record['reason'] = type(error).__name__
        return record
    finally:
        if fd is not None:
            os.close(fd)


def make_workload(fixture, padding, occupied, capacity, render_tokenize,
                  identity, policy, max_attempts=2048):
    """Find exact occupancy, then replay to detect nondeterministic adapters.

    render_tokenize returns (rendered UTF-8 text, list of integer token IDs).
    No token truncation: trimming a rendered template could remove its suffix.
    Bounded search may fail; failure is preferable to invented exact lengths.
    """
    if not positive_integer(occupied) or not positive_integer(capacity):
        raise ValueError('positive integer occupancy/capacity required')
    if occupied + 512 + 512 > capacity:
        raise ValueError('capacity must reserve output and control margin')
    required = {'template_sha256', 'tokenizer_sha256', 'model_sha256'}
    if set(identity) != required or any(
            not isinstance(v, str) or len(v) != 64 or
            any(c not in '0123456789abcdef' for c in v) for v in identity.values()):
        raise ValueError('reference hashes required')
    if not positive_integer(max_attempts):
        raise ValueError('positive bounded attempt count required')
    if not isinstance(padding, str) or not padding:
        raise ValueError('nonempty padding required')
    required_policy = {'thinking', 'seed', 'sampler_order', 'sampler_parameters', 'stops'}
    if set(policy) != required_policy:
        raise ValueError('complete effective decoding policy required')

    def evaluate(count):
        prompt = padding * count + '\n' + fixture['prompt']
        rendered, tokens = render_tokenize(prompt, policy)
        if not isinstance(rendered, str) or not isinstance(tokens, list) or not tokens:
            raise ValueError('invalid tokenizer result')
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('invalid token IDs')
        return prompt, rendered, tokens

    # Exponential bracket plus binary search, followed by a bounded local scan.
    # Token lengths need not be strictly monotonic at template boundaries.
    low, high, attempts = 0, 1, 0
    while True:
        result = evaluate(high)
        attempts += 1
        if len(result[2]) >= occupied or attempts >= max_attempts or high >= occupied * 2:
            break
        low, high = high, min(high * 2, occupied * 2)
    while low <= high and attempts < max_attempts:
        mid = (low + high) // 2
        result = evaluate(mid)
        attempts += 1
        if len(result[2]) == occupied:
            break
        if len(result[2]) < occupied:
            low = mid + 1
        else:
            high = mid - 1
    if len(result[2]) != occupied:
        for count in range(max(0, low - 32), low + 33):
            if attempts >= max_attempts:
                break
            result = evaluate(count)
            attempts += 1
            if len(result[2]) == occupied:
                break
    if len(result[2]) != occupied:
        raise ValueError('exact occupancy unavailable with this padding; revise fixture version')
    prompt, rendered, tokens = result
    if evaluate((len(prompt) - len(fixture['prompt']) - 1) // len(padding)) != result:
        raise ValueError('render/token replay mismatch')
    record = {'version': VERSION, 'fixture_id': fixture['id'],
              'fixture_sha256': digest(fixture), 'identity': identity, 'policy': policy,
              'occupied_input_tokens': occupied, 'capacity_tokens': capacity,
              'max_output_tokens': 512, 'eos_policy': 'retain separately if shorter',
              'prompt': prompt, 'rendered': rendered, 'token_ids': tokens,
              'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
              'tokens_sha256': digest(tokens)}
    record['manifest_sha256'] = digest(record)
    return record


def rebind_workload_capacity(record, capacity):
    """Make a capacity control while preserving exact frozen request bytes/IDs.

    Capacity is execution metadata; it must not trigger prompt rewriting or
    retokenization. This checks the source manifest and its reserves, then changes
    only capacity and the resulting manifest digest. It does not allocate a model
    context or establish runtime capacity behavior.
    """
    required = {'version', 'fixture_id', 'fixture_sha256', 'identity', 'policy',
                'occupied_input_tokens', 'capacity_tokens', 'max_output_tokens',
                'eos_policy', 'prompt', 'rendered', 'token_ids',
                'rendered_sha256', 'tokens_sha256', 'manifest_sha256'}
    if not isinstance(record, dict) or set(record) != required or record['version'] != VERSION:
        raise ValueError('versioned exact workload manifest required')
    if digest({key: value for key, value in record.items() if key != 'manifest_sha256'}) != record['manifest_sha256']:
        raise ValueError('source workload manifest checksum mismatch')
    occupied = record['occupied_input_tokens']
    if (type(occupied) is not int or occupied <= 0 or type(record['max_output_tokens']) is not int
            or record['max_output_tokens'] <= 0 or not isinstance(record['token_ids'], list)
            or len(record['token_ids']) != occupied or digest(record['token_ids']) != record['tokens_sha256']
            or not isinstance(record['rendered'], str)
            or hashlib.sha256(record['rendered'].encode('utf-8')).hexdigest() != record['rendered_sha256']):
        raise ValueError('source workload bytes/IDs/count mismatch')
    if (type(capacity) is not int or capacity <= 0
            or occupied + record['max_output_tokens'] + 512 > capacity):
        raise ValueError('capacity cannot hold exact workload, output and control reserve')
    rebound = dict(record)
    rebound['capacity_tokens'] = capacity
    rebound.pop('manifest_sha256')
    rebound['manifest_sha256'] = digest(rebound)
    return rebound


def paired_schedule(workload_ids, configurations, seed, warmups=2, trials=7):
    """Randomize order within each prompt/seed pair, preserving warmup labels."""
    if len(configurations) < 2 or len(set(configurations)) != len(configurations):
        raise ValueError('distinct paired configurations required')
    if not workload_ids or len(set(workload_ids)) != len(workload_ids):
        raise ValueError('distinct workloads required')
    if type(warmups) is not int or warmups < 2 or type(trials) is not int or trials < 7:
        raise ValueError('short protocol requires >=2 warmups and >=7 trials')
    rng, result = random.Random(seed), []
    for phase, count in [('warmup', warmups), ('measured', trials)]:
        for trial in range(count):
            order = list(workload_ids)
            rng.shuffle(order)
            for workload in order:
                configs = list(configurations)
                rng.shuffle(configs)
                for config in configs:
                    result.append({'phase': phase, 'pair': trial,
                                   'workload': workload, 'configuration': config,
                                   'seed': seed + trial})
    return result


def finite_samples(values):
    if not values or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError('nonempty finite samples required')


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(values):
    finite_samples(values)
    mean = statistics.mean(values)
    return {'n': len(values), 'median': statistics.median(values),
            'minimum': min(values), 'maximum': max(values),
            'iqr': percentile(values, .75) - percentile(values, .25),
            'p95': percentile(values, .95),
            'p95_caveat': 'descriptive; fewer than 30 samples' if len(values) < 30 else None,
            'cv': statistics.stdev(values) / abs(mean) if len(values) > 1 and mean else None}


def paired_improvement(reference, candidate, *, higher_is_better, seed, resamples=10000):
    """Percent change: median paired relative benefit, percentile bootstrap CI.

    Each input row must be one matched prompt/seed trial, never individual tokens.
    Caller must join identities and report exclusions before calling this helper.
    """
    finite_samples(reference)
    finite_samples(candidate)
    if len(reference) != len(candidate) or len(reference) < 7:
        raise ValueError('at least seven matched pairs required')
    if any(v <= 0 for v in reference) or any(v < 0 for v in candidate):
        raise ValueError('positive reference and nonnegative candidate required')
    if type(resamples) is not int or resamples < 1000:
        raise ValueError('at least 1000 bootstrap resamples required')
    direction = 1 if higher_is_better else -1
    changes = [direction * 100 * (c - r) / r for r, c in zip(reference, candidate)]
    rng = random.Random(seed)
    bootstrap = [statistics.median(rng.choices(changes, k=len(changes)))
                 for _ in range(resamples)]
    return {'pairs': len(changes), 'median_benefit_percent': statistics.median(changes),
            'ci95_percent': [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            'method': 'paired relative changes; median; percentile bootstrap',
            'seed': seed, 'resamples': resamples}


def missing_metrics(reason):
    if not reason:
        raise ValueError('missing counter reason required')
    return {key: {'value': None, 'reason': reason, 'unit': METRIC_UNITS[key]}
            for key in METRICS}


def delivered_metrics(admitted_ns, completed_ns, token_times_ns):
    """Delivery timings only; never reinterpret as synchronized device timings."""
    times = [admitted_ns, *token_times_ns, completed_ns]
    if any(type(t) is not int or t < 0 for t in times) or times != sorted(times):
        raise ValueError('ordered monotonic nanoseconds required')
    span = token_times_ns[-1] - token_times_ns[0] if token_times_ns else 0
    return {'ttft_s': (token_times_ns[0] - admitted_ns) / 1e9 if token_times_ns else None,
            'end_to_end_s': (completed_ns - admitted_ns) / 1e9,
            'steady_output_tokens_per_s': (len(token_times_ns) - 1) * 1e9 / span
            if len(token_times_ns) > 1 and span > 0 else None}


def write_private_json(path, value):
    """Create once; never overwrite evidence or follow an existing file symlink."""
    path = Path(path)
    if path.parent.is_symlink() or path.parent.stat().st_mode & 0o077:
        raise ValueError('private parent directory must have mode 0700')
    raw = canonical(value) + b'\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def campaign_usage(directory):
    """Logical bytes in a private campaign; never read file contents or delete data.

    Intended for a single sequential campaign writer. Periodic checks are not a
    filesystem quota and cannot prevent growth between checks or elsewhere.
    Symlinks and nonregular files reject accounting rather than escape its scope.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('private campaign directory required')
    total = 0

    def fail(error):
        raise error

    for root, directories, files in os.walk(directory, followlinks=False, onerror=fail):
        if stat.S_IMODE(Path(root).stat().st_mode) != 0o700:
            raise ValueError('campaign directories must have mode 0700')
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError('campaign symlinks unsupported')
        for name in files:
            info = (Path(root) / name).lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError('campaign files must be regular and mode 0600')
            total += info.st_size
    return total


def campaign_rejection(used_bytes, expected_bytes, limit_bytes, free_bytes, reserve_bytes):
    """Conservative projected growth check; includes previously failed captures."""
    values = (used_bytes, expected_bytes, limit_bytes, free_bytes, reserve_bytes)
    if any(type(v) is not int or v < 0 for v in values) or limit_bytes == 0:
        raise ValueError('nonnegative byte counts and positive campaign limit required')
    if used_bytes + expected_bytes > limit_bytes:
        return 'campaign byte budget exceeded'
    if free_bytes < expected_bytes + reserve_bytes:
        return 'projected disk reserve breach'
    return None


def terminate_group(process, grace_s, cleanup_timeout_s):
    """Signal an owned process group and bound waiting for its direct child.

    A SIGKILL request does not prove a kernel-blocked task has exited. Pending
    cleanup is returned with the owned PID; callers must not start another trial
    until that process/resource state has been checked. Descendants must not
    daemonize into another session. Reaping the leader does not prove GPU release.
    """
    if process is None:
        return None
    signals = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
        signals.append('SIGTERM')
        time.sleep(grace_s)
        try:
            os.killpg(process.pid, signal.SIGKILL)
            signals.append('SIGKILL')
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    try:
        returncode = process.wait(timeout=cleanup_timeout_s)
        pending = False
    except subprocess.TimeoutExpired:
        returncode, pending = None, True
    return {'pid': process.pid, 'returncode': returncode, 'pending': pending,
            'group_signals': signals, 'scope': 'direct-child reap; group signals do not prove device resource release'}


def open_child_executable_identity(process):
    """Pin the launched process executable for hashing after it exits."""
    pid = process.pid
    identity = {'pid': pid, 'start_ticks': None, 'executable_path': None,
                'size_bytes': None, 'device': None, 'inode': None,
                'sha256': None, 'reason': None}
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        if len(fields) <= 19:
            raise ValueError('invalid process start identity')
        identity['start_ticks'] = int(fields[19])
        if process.poll() is not None:
            raise ProcessLookupError('child exited before executable identity capture')
        identity['executable_path'] = os.readlink(f'/proc/{pid}/exe')
        fd = os.open(f'/proc/{pid}/exe', os.O_RDONLY | os.O_CLOEXEC)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ValueError('process executable is not a regular file')
        fields = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        if len(fields) <= 19 or int(fields[19]) != identity['start_ticks'] or process.poll() is not None:
            os.close(fd)
            raise ProcessLookupError('child identity changed during executable capture')
        identity.update(size_bytes=info.st_size, device=info.st_dev, inode=info.st_ino)
        return identity, fd
    except (OSError, ValueError, IndexError) as error:
        identity['reason'] = type(error).__name__
        return identity, None


def finish_child_executable_identity(identity, fd):
    if fd is None:
        return identity
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        hasher = hashlib.sha256()
        for block in iter(lambda: os.read(fd, 1024 * 1024), b''):
            hasher.update(block)
        identity['sha256'] = hasher.hexdigest()
    except OSError as error:
        identity['reason'] = type(error).__name__
    finally:
        os.close(fd)
    return identity


class LoadedLibraryAudit:
    """Pin and hash bounded shared-object mappings for one live child."""
    def __init__(self, pid, start_ticks, *, max_files=128, max_total_bytes=4 * 1024**3):
        self.pid, self.start_ticks = pid, start_ticks
        self.max_files, self.max_total_bytes = max_files, max_total_bytes
        self.entries, self.reason_codes, self.sample_durations_ns = {}, set(), []
        self.total_bytes, self.samples = 0, 0

    def sample(self):
        started = time.monotonic_ns()
        self.samples += 1
        try:
            fields = Path(f'/proc/{self.pid}/stat').read_text().rpartition(')')[2].split()
            if len(fields) <= 19 or int(fields[19]) != self.start_ticks:
                self.reason_codes.add('process_identity_changed')
                return
            with open(f'/proc/{self.pid}/maps', 'rb') as stream:
                raw = stream.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                self.reason_codes.add('maps_limit_exceeded')
                return
            for line in raw.decode('utf-8', errors='replace').splitlines():
                parts = line.split(None, 5)
                if len(parts) != 6:
                    continue
                path = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), parts[5])
                basename = path.rsplit('/', 1)[-1].split(' (deleted)', 1)[0]
                if not path.startswith('/') or not (basename.endswith('.so') or '.so.' in basename):
                    continue
                if path.endswith(' (deleted)'):
                    self.reason_codes.add('deleted_library_mapping')
                    continue
                try:
                    major, minor = (int(value, 16) for value in parts[3].split(':', 1))
                    inode = int(parts[4])
                except (ValueError, IndexError):
                    self.reason_codes.add('invalid_mapping_identity')
                    continue
                key = (major, minor, inode)
                if key in self.entries or inode == 0:
                    continue
                if len(self.entries) >= self.max_files:
                    self.reason_codes.add('library_count_limit_exceeded')
                    continue
                if len(path) > 1024:
                    self.reason_codes.add('library_path_limit_exceeded')
                    continue
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_ino != inode:
                        os.close(fd)
                        self.reason_codes.add('mapped_file_identity_mismatch')
                        continue
                    if info.st_size < 0 or self.total_bytes + info.st_size > self.max_total_bytes:
                        os.close(fd)
                        self.reason_codes.add('library_byte_limit_exceeded')
                        continue
                    self.entries[key] = {'path': path, 'fd': fd, 'size_bytes': info.st_size,
                                         'device': info.st_dev, 'inode': info.st_ino}
                    self.total_bytes += info.st_size
                except OSError:
                    self.reason_codes.add('mapped_file_unavailable')
        except (OSError, ValueError):
            self.reason_codes.add('process_maps_unavailable')
        finally:
            self.sample_durations_ns.append(time.monotonic_ns() - started)

    def finish(self):
        hashing_started = time.monotonic_ns()
        if not self.entries:
            self.reason_codes.add('no_shared_object_mappings')
        libraries = []
        for item in self.entries.values():
            record = {key: item[key] for key in ('path', 'size_bytes', 'device', 'inode')}
            try:
                os.lseek(item['fd'], 0, os.SEEK_SET)
                hasher = hashlib.sha256()
                for block in iter(lambda: os.read(item['fd'], 1024 * 1024), b''):
                    hasher.update(block)
                record['sha256'] = hasher.hexdigest()
            except OSError:
                self.reason_codes.add('library_hash_failed')
                record['sha256'] = None
            finally:
                os.close(item['fd'])
            libraries.append(record)
        return {'status': 'complete' if libraries and not self.reason_codes else 'incomplete',
                'scope': 'file-backed shared objects observed in /proc maps; model mappings and static code excluded',
                'sample_count': self.samples, 'sample_durations_ns': self.sample_durations_ns,
                'hash_duration_ns': time.monotonic_ns() - hashing_started,
                'unique_library_count': len(libraries), 'total_pinned_bytes': self.total_bytes,
                'reason_codes': sorted(self.reason_codes), 'libraries': libraries}


class MappedArtifactAudit:
    """Observe whether the checked artifact inodes are mapped by one child."""
    def __init__(self, pid, start_ticks, artifacts, artifact_checks):
        self.pid, self.start_ticks = pid, start_ticks
        self.entries, self.reason_codes, self.sample_durations_ns = {}, set(), []
        self.samples = 0
        for item, checked in zip(artifacts, artifact_checks):
            fd = None
            try:
                path = Path(item['path']).resolve(strict=True)
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_size != item['size_bytes']
                        or info.st_dev != checked['actual_device']
                        or info.st_ino != checked['actual_inode']):
                    os.close(fd)
                    fd = None
                    self.reason_codes.add('artifact_identity_changed')
                    continue
                self.entries[item['role']] = {'fd': fd, 'device': info.st_dev,
                                              'inode': info.st_ino, 'size_bytes': info.st_size,
                                              'mtime_ns': info.st_mtime_ns,
                                              'ctime_ns': info.st_ctime_ns, 'mapping_count': 0}
                fd = None
            except OSError:
                self.reason_codes.add('artifact_unavailable')
            finally:
                if fd is not None:
                    os.close(fd)

    def sample(self):
        started = time.monotonic_ns()
        self.samples += 1
        try:
            fields = Path(f'/proc/{self.pid}/stat').read_text().rpartition(')')[2].split()
            if len(fields) <= 19 or int(fields[19]) != self.start_ticks:
                self.reason_codes.add('process_identity_changed')
                return
            with open(f'/proc/{self.pid}/maps', 'rb') as stream:
                raw = stream.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                self.reason_codes.add('maps_limit_exceeded')
                return
            mapped = set()
            for line in raw.splitlines():
                parts = line.split(None, 5)
                if len(parts) < 5:
                    continue
                try:
                    major, minor = (int(value, 16) for value in parts[3].split(b':', 1))
                    inode = int(parts[4])
                except (ValueError, IndexError):
                    continue
                if inode:
                    mapped.add((major, minor, inode))
            for item in self.entries.values():
                if (os.major(item['device']), os.minor(item['device']), item['inode']) in mapped:
                    item['mapping_count'] += 1
        except (OSError, ValueError):
            self.reason_codes.add('process_maps_unavailable')
        finally:
            self.sample_durations_ns.append(time.monotonic_ns() - started)

    def finish(self):
        records = []
        for role, item in self.entries.items():
            try:
                info = os.fstat(item['fd'])
                stable = (info.st_dev == item['device'] and info.st_ino == item['inode']
                          and info.st_size == item['size_bytes'] and info.st_mtime_ns == item['mtime_ns']
                          and info.st_ctime_ns == item['ctime_ns'])
            except OSError:
                stable = False
            try:
                os.close(item['fd'])
            except OSError:
                stable = False
            if not stable:
                self.reason_codes.add('artifact_changed_during_run')
            records.append({'role': role, 'device': item['device'], 'inode': item['inode'],
                            'size_bytes': item['size_bytes'], 'mapping_observations': item['mapping_count'],
                            'status': 'observed' if item['mapping_count'] else 'not_observed',
                            'stable_file_identity': stable})
        return {'status': 'complete' if records and not self.reason_codes else 'incomplete',
                'scope': 'device/inode mappings observed in /proc maps; content hash is the separate prelaunch artifact check',
                'sample_count': self.samples, 'sample_durations_ns': self.sample_durations_ns,
                'reason_codes': sorted(self.reason_codes), 'artifacts': records}


def supervise(command, directory, *, timeout_s, grace_s=1, cancel=None,
              max_output_bytes=8 * 1024 * 1024, max_telemetry_bytes=1024 * 1024,
              disk_reserve_bytes=0, telemetry=None, cadence_s=.1, run_identity=None,
              campaign_root=None, campaign_limit_bytes=None, env=None,
              telemetry_command=None, telemetry_timeout_s=.25, cleanup_timeout_s=1,
              measurement_telemetry_command=None, measurement_telemetry_timeout_s=.25,
              telemetry_startup_timeout_s=None, metric_adapter=None,
              audit_loaded_libraries=False, expected_artifacts=None,
              audit_mapped_artifacts=False, trial_identity=None):
    """Capture a local child with bounded pipes and optional preflight/telemetry.

    telemetry(pid_or_none, elapsed_s, captured_bytes) returns a JSON object with
    optional `rejection` (reason or None). It runs before launch and on cadence.
    Callbacks must be bounded; a potentially blocking provider belongs in the
    explicit telemetry_command worker, whose JSON-pipe response deadline is also
    capped by remaining workload time. Optional startup allowance applies only to
    preflight, before model launch; it does not extend running response deadlines.
    Cleanup waits for owned leaders are bounded;
    pending cleanup is retained and must block subsequent trials.
    An optional post-cleanup `metric_adapter(result, telemetry_records)` may
    return named complete inventories; validated metrics/provenance are persisted
    in result.json. Adapter failures never change workload status and produce
    explicit missing metrics plus the exception class only.
    The caps cover captured pipes and telemetry, not files the child writes itself.
    A private working directory and child umask contain ordinary child artifacts.
    No GPU, sensor calibration or workload authorization is implied by this API.
    """
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (timeout_s, cadence_s)):
        raise ValueError('positive finite timeout and cadence required')
    if type(grace_s) not in (int, float) or not math.isfinite(grace_s) or grace_s < 0:
        raise ValueError('finite nonnegative grace required')
    if not positive_integer(max_output_bytes) or not positive_integer(max_telemetry_bytes):
        raise ValueError('positive capture limits required')
    if type(disk_reserve_bytes) is not int or disk_reserve_bytes < 0:
        raise ValueError('nonnegative disk reserve required')
    if env is not None and (not isinstance(env, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())):
        raise ValueError('explicit child environment must be a string map')
    if telemetry is not None and telemetry_command is not None:
        raise ValueError('choose a callback or a collector process, not both')
    if metric_adapter is not None and not callable(metric_adapter):
        raise ValueError('metric adapter must be callable')
    if type(audit_loaded_libraries) is not bool:
        raise ValueError('loaded-library audit switch must be boolean')
    if type(audit_mapped_artifacts) is not bool or (audit_mapped_artifacts and not expected_artifacts):
        raise ValueError('mapped-artifact audit requires expected artifacts and a boolean switch')
    expected_artifacts = validate_expected_artifacts(expected_artifacts)
    if trial_identity is not None:
        if not isinstance(trial_identity, dict):
            raise ValueError('trial identity must be a mapping')
        canonical(trial_identity)
    for value in (telemetry_timeout_s, cleanup_timeout_s):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('positive finite collector/cleanup deadline required')
    startup_timeout = telemetry_timeout_s if telemetry_startup_timeout_s is None else telemetry_startup_timeout_s
    if type(startup_timeout) not in (int, float) or not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError('positive finite collector startup deadline required')
    has_telemetry = telemetry is not None or telemetry_command is not None
    if measurement_telemetry_command is not None and not has_telemetry:
        raise ValueError('measurement collector requires an active safety telemetry channel')
    if measurement_telemetry_command is not None and telemetry_command is not None and telemetry is not None:
        raise ValueError('safety telemetry must use one callback or collector process')
    if measurement_telemetry_command is not None and (not isinstance(measurement_telemetry_command, list)
            or not measurement_telemetry_command or any(not isinstance(v, str) for v in measurement_telemetry_command)):
        raise ValueError('explicit measurement collector command list required')
    if (type(measurement_telemetry_timeout_s) not in (int, float)
            or not math.isfinite(measurement_telemetry_timeout_s) or measurement_telemetry_timeout_s <= 0):
        raise ValueError('positive finite measurement collector deadline required')
    directory = Path(directory).resolve()
    if (campaign_root is None) != (campaign_limit_bytes is None):
        raise ValueError('campaign root and limit must be supplied together')
    if campaign_root is not None:
        campaign_root = Path(campaign_root)
        campaign_usage(campaign_root)
        campaign_root = campaign_root.resolve()
        if not directory.is_relative_to(campaign_root):
            raise ValueError('capture must be inside its campaign')
        if not positive_integer(campaign_limit_bytes):
            raise ValueError('positive campaign limit required')
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_private_json(directory / 'manifest.json', {
        'version': VERSION, 'command': command, 'timeout_s': timeout_s,
        'grace_s': grace_s, 'cleanup_timeout_s': cleanup_timeout_s,
        'status': 'incomplete', 'run_identity': run_identity,
        'telemetry_command': telemetry_command, 'telemetry_timeout_s': telemetry_timeout_s,
        'measurement_telemetry_enabled': measurement_telemetry_command is not None,
        'measurement_telemetry_timeout_s': measurement_telemetry_timeout_s,
        'telemetry_startup_timeout_s': startup_timeout,
        'collector_grace_s': .05 if telemetry_command is not None else None,
        'collector_max_response_bytes': min(max_telemetry_bytes, 65536) if telemetry_command is not None else None,
        'collector_max_stderr_bytes': min(max_output_bytes, 65536) if telemetry_command is not None else None,
        'max_output_bytes': max_output_bytes, 'max_telemetry_bytes': max_telemetry_bytes,
        'disk_reserve_bytes': disk_reserve_bytes, 'cadence_s': cadence_s,
        'campaign_root': str(campaign_root) if campaign_root else None,
        'campaign_limit_bytes': campaign_limit_bytes,
        'environment_mode': 'explicit' if env is not None else 'inherited; not frozen',
        'environment_sha256': digest(env) if env is not None else None,
        'metric_adapter_enabled': metric_adapter is not None,
        'audit_loaded_libraries': audit_loaded_libraries,
        'audit_mapped_artifacts': audit_mapped_artifacts,
        'expected_artifacts': expected_artifacts,
        'trial_identity': trial_identity,
        'metrics': missing_metrics('no inference metric adapter attached')})
    artifact_checks = ([check_expected_artifact(item) for item in expected_artifacts]
                       if expected_artifacts is not None else None)
    artifact_failure = artifact_checks is not None and any(
        item['status'] != 'matched' for item in artifact_checks)
    start, process, reason, returncode = time.monotonic_ns(), None, (
        'artifact_identity_check_failed' if artifact_failure else None), None
    status, captured, telemetry_bytes, cycles, telemetry_records = 'failed', 0, 0, [], []
    measurement_bytes, measurement_cycles = 0, []
    child_identity, child_executable_fd, supervision_end_ns = None, None, None
    library_audit, library_identity = None, None
    mapped_audit, mapped_identity = None, None
    next_sample, collector, measurement_collector, target_cleanup = start, None, None, None

    def stop_target():
        nonlocal target_cleanup
        if process is not None and target_cleanup is None:
            target_cleanup = terminate_group(process, grace_s, cleanup_timeout_s)

    try:
        with ExitStack() as stack:
            outputs = {name: stack.enter_context(open_private(directory / (name + '.txt')))
                       for name in ('stdout', 'stderr')}
            samples = stack.enter_context(open_private(directory / 'telemetry.jsonl')) if has_telemetry else None
            measurement_samples = (stack.enter_context(open_private(directory / 'measurement-telemetry.jsonl'))
                                   if measurement_telemetry_command is not None else None)
            selector = stack.enter_context(selectors.DefaultSelector())

            def observe(pid):
                nonlocal telemetry_bytes
                before = time.monotonic_ns()
                elapsed = (before - start) / 1e9
                capture_total = (captured + telemetry_bytes + measurement_bytes
                                 + (collector.stderr_bytes if collector else 0)
                                 + (measurement_collector.stderr_bytes if measurement_collector else 0))
                if collector is not None:
                    snapshot = collector.sample(pid, elapsed, capture_total,
                                                timeout_s=min(startup_timeout if pid is None else telemetry_timeout_s,
                                                              timeout_s - elapsed))
                else:
                    snapshot = telemetry(pid, elapsed, capture_total)
                if not isinstance(snapshot, dict) or (snapshot.get('rejection') is not None
                                                     and (not isinstance(snapshot['rejection'], str) or not snapshot['rejection'].strip())):
                    raise ValueError('invalid telemetry record')
                record = {'timestamp_ns': before, 'stage': 'preflight' if pid is None else 'running',
                          'collection_ns': time.monotonic_ns() - before, 'snapshot': snapshot}
                if collector is not None and collector.last_transport is not None:
                    record['collector_transport'] = collector.last_transport
                raw = canonical(record) + b'\n'
                if len(raw) > max_telemetry_bytes - telemetry_bytes - measurement_bytes:
                    return 'telemetry capture limit reached'
                samples.write(raw)
                samples.flush()
                telemetry_bytes += len(raw)
                if metric_adapter is not None:
                    telemetry_records.append(record)
                cycles.append(time.monotonic_ns() - before)
                return snapshot.get('rejection')

            def observe_measurement(pid):
                nonlocal measurement_bytes
                before = time.monotonic_ns()
                elapsed = (before - start) / 1e9
                capture_total = (captured + telemetry_bytes + measurement_bytes
                                 + (collector.stderr_bytes if collector else 0)
                                 + (measurement_collector.stderr_bytes if measurement_collector else 0))
                snapshot = measurement_collector.sample(
                    pid, elapsed, capture_total,
                    timeout_s=min(measurement_telemetry_timeout_s, timeout_s - elapsed))
                if not isinstance(snapshot, dict) or 'rejection' in snapshot:
                    raise ValueError('measurement collector response must be data without safety rejection')
                record = {'timestamp_ns': before, 'stage': 'preflight' if pid is None else 'running',
                          'collection_ns': time.monotonic_ns() - before, 'snapshot': snapshot}
                if measurement_collector.last_transport is not None:
                    record['collector_transport'] = measurement_collector.last_transport
                raw = canonical(record) + b'\n'
                if len(raw) > max_telemetry_bytes - measurement_bytes - telemetry_bytes:
                    return 'measurement telemetry capture limit reached'
                measurement_samples.write(raw)
                measurement_samples.flush()
                measurement_bytes += len(raw)
                measurement_cycles.append(time.monotonic_ns() - before)
                return None

            if not reason and campaign_root is not None:
                reason = campaign_rejection(campaign_usage(campaign_root),
                                            max_output_bytes + max_telemetry_bytes +
                                            max_telemetry_bytes // 2 + 65536 +
                                            (min(max_output_bytes, 65536) + min(max_telemetry_bytes, 65536)
                                            if telemetry_command is not None else 0) +
                                            (min(max_output_bytes, 65536) + min(max_telemetry_bytes, 65536)
                                             if measurement_telemetry_command is not None else 0) +
                                            (2 * 1024 * 1024 if audit_loaded_libraries else 0),
                                            campaign_limit_bytes,
                                            shutil.disk_usage(directory).free, disk_reserve_bytes)
            if not reason and shutil.disk_usage(directory).free < disk_reserve_bytes:
                reason = 'disk reserve breach'
            if not reason and telemetry_command is not None:
                from collector_worker import ProcessCollector
                collector = ProcessCollector(
                    telemetry_command, directory, env=env, timeout_s=max(startup_timeout, telemetry_timeout_s),
                    max_response_bytes=min(max_telemetry_bytes, 65536),
                    max_stderr_bytes=min(max_output_bytes, 65536),
                    cleanup_timeout_s=cleanup_timeout_s)
                stack.enter_context(collector)
            if not reason and has_telemetry:
                if measurement_telemetry_command is not None:
                    from collector_worker import ProcessCollector
                    measurement_collector = ProcessCollector(
                        measurement_telemetry_command, directory, env=env,
                        timeout_s=measurement_telemetry_timeout_s,
                        max_response_bytes=min(max_telemetry_bytes, 65536),
                        max_stderr_bytes=min(max_output_bytes, 65536),
                        cleanup_timeout_s=cleanup_timeout_s, artifact_prefix='measurement-')
                    stack.enter_context(measurement_collector)
                reason = observe(None)
                if not reason and measurement_collector is not None:
                    reason = observe_measurement(None)
                next_sample = time.monotonic_ns() + int(cadence_s * 1e9)
            if not reason and (time.monotonic_ns() - start) / 1e9 >= timeout_s:
                reason = 'timeout during preflight'
            if not reason:
                process = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, start_new_session=True, umask=0o077, env=env)
                child_identity, child_executable_fd = open_child_executable_identity(process)
                if audit_loaded_libraries:
                    library_audit = LoadedLibraryAudit(
                        process.pid, child_identity['start_ticks'] or 0)
                    library_audit.sample()
                if audit_mapped_artifacts and child_identity['start_ticks'] is not None:
                    mapped_audit = MappedArtifactAudit(process.pid, child_identity['start_ticks'],
                                                       expected_artifacts, artifact_checks)
                    mapped_audit.sample()
                for name in outputs:
                    pipe = getattr(process, name)
                    stack.callback(pipe.close)
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
                # Stop the workload before waiting for collector or pipe cleanup.
                stack.callback(stop_target)
                while selector.get_map() or process.poll() is None:
                    now = time.monotonic_ns()
                    reason = cancel() if cancel else None
                    if not reason and (now - start) / 1e9 >= timeout_s:
                        reason = 'timeout'
                    if not reason and shutil.disk_usage(directory).free < disk_reserve_bytes:
                        reason = 'disk reserve breach'
                    if not reason and now >= next_sample:
                        if campaign_root is not None:
                            reason = campaign_rejection(campaign_usage(campaign_root), 0,
                                                        campaign_limit_bytes,
                                                        shutil.disk_usage(directory).free, disk_reserve_bytes)
                        if not reason and has_telemetry and process.poll() is None:
                            reason = observe(process.pid)
                        if not reason and measurement_collector is not None and process.poll() is None:
                            reason = observe_measurement(process.pid)
                        if not reason and library_audit is not None and process.poll() is None:
                            library_audit.sample()
                        if not reason and mapped_audit is not None and process.poll() is None:
                            mapped_audit.sample()
                        next_sample = time.monotonic_ns() + int(cadence_s * 1e9)
                    if reason:
                        break
                    for key, _ in selector.select(.01):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = max_output_bytes - captured
                        outputs[key.data].write(data[:remaining])
                        captured += min(len(data), remaining)
                        if len(data) > remaining:
                            reason = 'output capture limit reached'
                            break
                    if reason:
                        break
            if reason:
                status = 'incomplete'
            elif process is not None:
                returncode = process.wait()
                status = 'successful' if returncode == 0 else 'failed'
                reason = None if returncode == 0 else 'child returned nonzero'
    except BaseException as error:
        reason = type(error).__name__
        raise
    finally:
        stop_target()
        if target_cleanup is not None:
            returncode = target_cleanup['returncode']
            if target_cleanup['pending']:
                status, reason = 'incomplete', 'workload cleanup pending'
        collector_summary = collector.summary if collector is not None else None
        measurement_collector_summary = (measurement_collector.summary
                                         if measurement_collector is not None else None)
        if collector_summary and collector_summary['cleanup'] and collector_summary['cleanup']['pending']:
            status, reason = 'incomplete', 'collector cleanup pending'
        elif collector_summary and collector_summary['failure'] and status == 'successful':
            status, reason = 'incomplete', collector_summary['failure']
        if (measurement_collector_summary and measurement_collector_summary['cleanup']
                and measurement_collector_summary['cleanup']['pending']):
            status, reason = 'incomplete', 'measurement collector cleanup pending'
        elif measurement_collector_summary and measurement_collector_summary['failure'] and status == 'successful':
            status, reason = 'incomplete', measurement_collector_summary['failure']
        supervision_end_ns = time.monotonic_ns()
        if child_identity is not None:
            child_identity = finish_child_executable_identity(child_identity, child_executable_fd)
            child_executable_fd = None
        if audit_loaded_libraries:
            if library_audit is not None:
                library_identity = library_audit.finish()
            else:
                library_identity = {'status': 'not_run', 'scope': 'no child process launched',
                                    'sample_count': 0, 'sample_durations_ns': [],
                                    'hash_duration_ns': 0, 'unique_library_count': 0,
                                    'total_pinned_bytes': 0, 'reason_codes': ['workload_not_launched'],
                                    'libraries': []}
        if mapped_audit is not None:
            mapped_identity = mapped_audit.finish()
        elif audit_mapped_artifacts:
            mapped_identity = {'status': 'not_run',
                               'scope': ('child process identity unavailable' if process is not None
                                         else 'no child process launched'),
                               'sample_count': 0, 'sample_durations_ns': [],
                               'reason_codes': (['child_process_identity_unavailable'] if process is not None
                                                else ['workload_not_launched']),
                               'artifacts': []}
        result = {'status': status, 'reason': reason, 'returncode': returncode,
                  'launched': process is not None, 'captured_output_bytes': captured,
                  'child_process_identity': child_identity,
                  'artifact_checks': artifact_checks,
                  'loaded_library_identity': library_identity,
                  'mapped_artifact_identity': mapped_identity,
                  'workload_cleanup': target_cleanup, 'collector': collector_summary,
                  'measurement_telemetry_enabled': measurement_telemetry_command is not None,
                  'measurement_collector': measurement_collector_summary,
                  'measurement_telemetry_bytes': measurement_bytes,
                  'measurement_collector_cycles': len(measurement_cycles),
                  'measurement_collector_cycle_durations_ns': measurement_cycles,
                  'measurement_collector_cycle_p95_ns': percentile(measurement_cycles, .95) if measurement_cycles else None,
                  'telemetry_bytes': telemetry_bytes, 'collector_cycles': len(cycles),
                  'collector_cycle_durations_ns': cycles,
                  'collector_cycle_p95_ns': percentile(cycles, .95) if cycles else None,
                  'elapsed_s': (supervision_end_ns - start) / 1e9}
        result['metrics'] = missing_metrics('no inference metric adapter attached')
        result['metric_sources'] = {}
        if metric_adapter is not None:
            try:
                from benchmark_metrics import merge_metric_sources
                adapted = metric_adapter(result, telemetry_records)
                metadata = {}
                if isinstance(adapted, dict) and set(adapted) == {'sources', 'metadata'}:
                    metadata = adapted['metadata']
                    if not isinstance(metadata, dict):
                        raise ValueError('metric adapter metadata must be a mapping')
                    canonical(metadata)
                    adapted = adapted['sources']
                merged = merge_metric_sources(adapted)
                result['metrics'] = merged['metrics']
                result['metric_sources'] = merged['metric_sources']
                if metadata:
                    result['metric_metadata'] = metadata
            except Exception as error:
                # Analysis failures never rewrite process status or discard the
                # captured run; every metric remains explicit and unavailable.
                result['metrics'] = missing_metrics('run-level metric adapter failed')
                result['metric_adapter_error'] = type(error).__name__
        write_private_json(directory / 'result.json', result)
        index = {}
        for path in directory.iterdir():
            # Child-created files are not trusted checksum inputs; do not follow
            # symlinks or index arbitrary/large artifacts supplied by the child.
            if path.name not in {'manifest.json', 'stdout.txt', 'stderr.txt',
                                 'telemetry.jsonl', 'result.json', 'collector-stderr.txt',
                                 'collector-result.json', 'collector-invalid-response.txt'}:
                if path.name in {'measurement-telemetry.jsonl', 'measurement-collector-stderr.txt',
                                 'measurement-collector-result.json', 'measurement-collector-invalid-response.txt'}:
                    pass
                else:
                    continue
            hasher = hashlib.sha256()
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError('capture index requires regular files')
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(block)
            index[path.name] = hasher.hexdigest()
        write_private_json(directory / 'checksums.json', index)
    return result


def open_private(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb')
