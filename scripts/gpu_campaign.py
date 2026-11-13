#!/usr/bin/env python3
"""Supervised reference-server sessions for Epic 2 GPU baselines.

One session = one private llama-server process bound to a UNIX socket inside
the session directory, a fixed ordered list of token-exact /completion requests,
then shutdown. Three subcommands:

  run SESSION.json        preflight + supervise(drive) with the telemetry collector
  drive SESSION.json      (child) start server, wait for health, send requests
  collect SESSION.json    (child) bounded safety/telemetry collector process

The server never listens on TCP, so no port is allocated and no operational
endpoint is touched. Model load has its own deadline separate from each
request deadline (cold page-cache loads can take most of a minute). Foreign
llama-server processes (e.g. the production router and its sleeping model
children) are watched; growth of any of them cancels the session so the
production service always wins. Nothing here changes router configuration,
evicts another client or claims calibrated device counters.
"""
import hashlib
import http.client
import math
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_core import digest, supervise, write_private_json  # noqa: E402
from benchmark_resources import PagingMonitor, meminfo, vmstat  # noqa: E402

SCHEMA = 'rosevine-gpu-session-v1'
REQUIRED = {'schema', 'label', 'server_argv', 'requests', 'load_timeout_s',
            'request_timeout_s', 'session_timeout_s', 'limits', 'device', 'sensors'}
LIMITS = {'ram_reserve_bytes', 'vram_reserve_bytes', 'disk_reserve_bytes',
          'foreign_rss_growth_bytes', 'max_output_bytes', 'max_telemetry_bytes'}


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__('localhost', timeout=timeout)
        self.unix_path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_path)
        self.sock = sock


# Router-only keys and chat-endpoint keys. Requests are token-exact /completion
# calls with explicit greedy sampling, so these do not affect the measured path;
# dropping them keeps sessions independent of production router files.
ROUTER_ONLY = {'load-on-startup', 'stop-timeout'}
CHAT_ONLY = {'jinja', 'chat-template-file', 'reasoning', 'reasoning-format',
             'temp', 'top-p', 'top-k', 'min-p', 'repeat-penalty', 'n-predict'}
FLAG_TRUE = {'no-mmproj-offload', 'kv-unified'}


def preset_argv(binary, preset, *, overrides=None, remove=()):
    """Translate a captured router preset section into a server argv.

    `overrides` replaces/adds keys (one-factor controls); `remove` drops keys
    (e.g. the MTP draft or projector for a named text-only control). Returns
    (argv, deviations) where deviations lists every dropped preset key.
    """
    merged = {**preset, **(overrides or {})}
    deviations = sorted(k for k in merged if k in ROUTER_ONLY | CHAT_ONLY | set(remove))
    argv = [binary]
    for key, value in merged.items():
        if key in ROUTER_ONLY or key in CHAT_ONLY or key in remove:
            continue
        if key in FLAG_TRUE:
            if value != 'true':
                raise ValueError(f'unexpected flag value for {key}')
            argv.append(f'--{key}')
        else:
            argv += [f'--{key}', str(value)]
    return argv, deviations


def load_session(path):
    session = json.loads(Path(path).read_text())
    if not isinstance(session, dict) or set(session) - (REQUIRED | {'notes', 'collector_enabled', 'sample_clocks', 'cpu_affinity', 'mode', 'power_profile'}) or REQUIRED - set(session):
        raise ValueError('session schema invalid')
    if session['schema'] != SCHEMA or set(session['limits']) != LIMITS:
        raise ValueError('session schema/limits invalid')
    if session.get('mode', 'server') not in ('server', 'command'):
        raise ValueError('mode must be server or command')
    if 'power_profile' in session and session['power_profile'] not in POWER_PROFILES:
        raise ValueError(f'power_profile must be one of {sorted(POWER_PROFILES)}')
    if session.get('mode') == 'command':
        if session['requests']:
            raise ValueError('command sessions take no requests')
        return session
    if not session['requests'] or any(set(r) != {'id', 'workload', 'n_predict', 'role'}
                                      or r['role'] not in ('warmup', 'measured')
                                      for r in session['requests']):
        raise ValueError('request list invalid')
    if len({r['id'] for r in session['requests']}) != len(session['requests']):
        raise ValueError('duplicate request id')
    if '--host' in session['server_argv'] or '--port' in session['server_argv']:
        raise ValueError('driver owns the listening socket')
    return session


def load_workload(path):
    record = json.loads(Path(path).read_text())
    ids = record['token_ids']
    if digest(ids) != record['tokens_sha256'] or len(ids) != record['occupied_input_tokens']:
        raise ValueError(f'workload token identity mismatch: {path}')
    return record


def request_body(workload, n_predict):
    """Greedy, uncached, token-exact request matching the frozen workload policy."""
    policy = workload['policy']
    if policy['sampler_order'] != ['temperature'] or policy['sampler_parameters'] != {'temperature': 0}:
        raise ValueError('only the frozen greedy control policy is supported')
    return {'prompt': workload['token_ids'], 'n_predict': n_predict, 'temperature': 0,
            'samplers': ['temperature'], 'seed': policy['seed'], 'cache_prompt': False,
            'return_tokens': True, 'stream': False}


def http_json(path, method, url, body, timeout):
    conn = UnixHTTPConnection(path, timeout)
    try:
        payload = None if body is None else json.dumps(body).encode()
        headers = {} if body is None else {'Content-Type': 'application/json'}
        conn.request(method, url, payload, headers)
        response = conn.getresponse()
        raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError('response too large')
        return response.status, json.loads(raw) if raw else None
    finally:
        conn.close()


def wait_healthy(sock, process, deadline_ns):
    while time.monotonic_ns() < deadline_ns:
        if process.poll() is not None:
            return f'server exited during load ({process.returncode})'
        if os.path.exists(sock):
            try:
                status, _ = http_json(sock, 'GET', '/health', None, 2)
                if status == 200:
                    return None
            except (OSError, ValueError, http.client.HTTPException):
                pass
        time.sleep(.1)
    return 'load deadline exceeded'


def summarize_response(status, body, expected_prompt_n):
    if status != 200 or not isinstance(body, dict):
        return {'ok': False, 'reason': f'http {status}'}
    timings = body.get('timings') or {}
    tokens = body.get('tokens') or []
    record = {'ok': True, 'timings': timings, 'stop_type': body.get('stop_type'),
              'tokens_predicted': body.get('tokens_predicted'),
              'tokens_evaluated': body.get('tokens_evaluated'),
              'output_token_ids': tokens, 'output_sha256': digest(tokens)}
    if timings.get('prompt_n') != expected_prompt_n:
        record['ok'] = False
        record['reason'] = f'prompt_n {timings.get("prompt_n")} != {expected_prompt_n} (cache or BOS mismatch)'
    return record


def _terminated(signum, frame):
    raise SystemExit(128 + signum)


def drive(session_path):
    # Supervisor cancellation sends SIGTERM; convert it so `finally` still stops
    # the server and writes driver.json for the retained failure record.
    signal.signal(signal.SIGTERM, _terminated)
    session = load_session(session_path)
    root = Path(session_path).resolve().parent
    if session.get('mode') == 'command':
        return drive_command(session, root)
    sock = str(root / 'server.sock')
    workloads = {}
    for request in session['requests']:
        if request['workload'] not in workloads:
            workloads[request['workload']] = load_workload(request['workload'])
    argv = [*session['server_argv'], '--host', sock]
    log = open(root / 'server.log', 'wb')
    started = time.monotonic_ns()
    # Optional CPU pinning of the server (e.g. one thread per physical core) to
    # control scheduler placement; recorded in the session file.
    affinity = session.get('cpu_affinity')
    pin = (lambda: os.sched_setaffinity(0, affinity)) if affinity else None
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               preexec_fn=pin)
    summary = {'server_pid': process.pid, 'argv_sha256': digest(argv),
               'server_affinity': sorted(os.sched_getaffinity(process.pid)) if process.poll() is None else None}
    out = open(root / 'requests.jsonl', 'a')
    try:
        failure = wait_healthy(sock, process, started + int(session['load_timeout_s'] * 1e9))
        summary['load_s'] = (time.monotonic_ns() - started) / 1e9
        if failure:
            summary['failure'] = failure
            return 2
        for request in session['requests']:
            workload = workloads[request['workload']]
            body = request_body(workload, request['n_predict'])
            sent = time.monotonic_ns()
            try:
                status, response = http_json(sock, 'POST', '/completion', body, session['request_timeout_s'])
                record = summarize_response(status, response, workload['occupied_input_tokens'])
            except (OSError, ValueError, http.client.HTTPException) as error:
                record = {'ok': False, 'reason': type(error).__name__}
            record.update({'id': request['id'], 'role': request['role'],
                           'workload_manifest_sha256': workload['manifest_sha256'],
                           'fixture_id': workload['fixture_id'],
                           'occupied_input_tokens': workload['occupied_input_tokens'],
                           'client_sent_ns': sent, 'client_received_ns': time.monotonic_ns()})
            out.write(json.dumps(record, sort_keys=True) + '\n')
            out.flush()
            os.fsync(out.fileno())
            if not record['ok']:
                summary['failure'] = f'request {request["id"]}: {record.get("reason")}'
                return 3
            if process.poll() is not None:
                summary['failure'] = f'server exited ({process.returncode})'
                return 4
        return 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(5)
        summary['server_returncode'] = process.returncode
        summary['session_s'] = (time.monotonic_ns() - started) / 1e9
        # The server does not unlink its socket; dead sockets break tree copies.
        if os.path.exists(sock):
            os.unlink(sock)
        write_private_json(root / 'driver.json', summary)
        out.close()
        log.close()


def drive_command(session, root):
    """Run one bounded reference tool (e.g. llama-perplexity) to completion under
    the same supervisor, collector and router protection as server sessions."""
    affinity = session.get('cpu_affinity')
    pin = (lambda: os.sched_setaffinity(0, affinity)) if affinity else None
    started = time.monotonic_ns()
    summary = {'argv_sha256': digest(session['server_argv'])}
    with open(root / 'command.log', 'wb') as log:
        process = subprocess.Popen(session['server_argv'], stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, preexec_fn=pin, cwd=root)
        try:
            summary['returncode'] = process.wait()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(5)
            summary['returncode'] = process.returncode
            summary['wall_s'] = (time.monotonic_ns() - started) / 1e9
            write_private_json(root / 'driver.json', summary)
    return 0 if summary['returncode'] == 0 else 6


def foreign_servers(own_pids=()):
    """PID -> RSS bytes for llama-server processes not owned by this session."""
    found = {}
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit() or int(entry.name) in own_pids:
            continue
        try:
            name = Path(f'/proc/{entry.name}/comm').read_text().strip()
            if not name.startswith('llama-server'):
                continue
            found[int(entry.name)] = int(Path(f'/proc/{entry.name}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            continue
    return found


def own_tree(pid):
    """The supervised driver and its descendants (the private server)."""
    tree, frontier = {pid}, [pid]
    while frontier:
        current = frontier.pop()
        try:
            for task in os.listdir(f'/proc/{current}/task'):
                for child in Path(f'/proc/{current}/task/{task}/children').read_text().split():
                    if int(child) not in tree:
                        tree.add(int(child))
                        frontier.append(int(child))
        except OSError:
            continue
    return tree


def thermal_rejection(streaks, temps, sensors):
    """Per-sensor stop. `ceiling_c` must hold for `sustain_samples` consecutive
    samples (default 1 = immediate); `immediate_c`, if set, stops on any single
    sample. `streaks` is mutated: label -> consecutive samples at/above ceiling.
    Debouncing exists for die sensors that report single-sample load spikes;
    it never raises the ceiling itself."""
    hits = []
    for sensor in sensors:
        label, value = sensor['label'], temps[sensor['label']]
        streaks[label] = streaks.get(label, 0) + 1 if value >= sensor['ceiling_c'] else 0
        if streaks[label] >= sensor.get('sustain_samples', 1) or value >= sensor.get('immediate_c', math.inf):
            hits.append(label)
    return 'thermal stop: ' + ','.join(hits) if hits else None


def collect(session_path):
    from collector_worker import serve
    session = load_session(session_path)
    root = Path(session_path).resolve().parent
    limits = session['limits']
    device = Path(session['device'])
    if (device / 'vendor').read_text().strip() != '0x1002':
        raise ValueError('unexpected GPU vendor')
    sensors = session['sensors']
    page = os.sysconf('SC_PAGE_SIZE')
    paging = PagingMonitor(limits['ram_reserve_bytes'])
    baseline = foreign_servers()
    state = {'scan_at': 0.0, 'foreign': dict(baseline)}
    streaks = {}
    # amdgpu exposes current shader/memory clocks as hwmon freq inputs; a falling
    # shader clock under load is the data-quality signal for thermal throttling.
    clock_paths = {}
    # Optional: reading amdgpu clock inputs queries the SMU firmware; sessions may
    # disable it to measure that interference (default on for compatibility).
    for hwmon in sorted((device / 'hwmon').glob('hwmon*')) if session.get('sample_clocks', True) else ():
        for index in (1, 2):
            label = hwmon / f'freq{index}_label'
            if label.exists():
                clock_paths[label.read_text().strip()] = hwmon / f'freq{index}_input'

    def sample(pid, elapsed_s, captured_bytes):
        memory = meminfo(Path('/proc/meminfo').read_text())
        swap = vmstat(Path('/proc/vmstat').read_text(), page)
        used = int((device / 'mem_info_vram_used').read_text())
        total = int((device / 'mem_info_vram_total').read_text())
        busy = int((device / 'gpu_busy_percent').read_text())
        temps = {s['label']: int(Path(s['path']).read_text()) / 1000 for s in sensors}
        clocks = {}
        for name, clock_path in clock_paths.items():
            try:
                clocks[name] = int(clock_path.read_text())
            except (OSError, ValueError):
                clocks[name] = None
        rejected = thermal_rejection(streaks, temps, sensors)
        if rejected:
            pass
        elif memory['MemAvailable'] < limits['ram_reserve_bytes']:
            rejected = 'RAM reserve breach'
        elif limits['vram_reserve_bytes'] and total - used < limits['vram_reserve_bytes']:
            # 0 disables: allocation failure is handled by the server itself and
            # the production router is protected by the foreign-growth stop.
            rejected = 'VRAM reserve breach'
        elif shutil.disk_usage(root).free < limits['disk_reserve_bytes']:
            rejected = 'disk reserve breach'
        # Production router protection: re-read known foreign RSS every cycle,
        # rescan for new foreign servers every second.
        foreign_growth = {}
        if elapsed_s - state['scan_at'] >= 1 or pid is None:
            state['scan_at'] = elapsed_s
            state['foreign'] = foreign_servers(own_tree(pid) if pid else ())
        for fpid, rss in state['foreign'].items():
            try:
                rss = int(Path(f'/proc/{fpid}/statm').read_text().split()[1]) * page
            except (OSError, ValueError, IndexError):
                continue
            growth = rss - baseline.get(fpid, 0)
            if growth > limits['foreign_rss_growth_bytes']:
                foreign_growth[fpid] = growth
        if foreign_growth and rejected is None:
            rejected = 'production llama-server activity'
        event = paging.update(elapsed_s, swap['pswpin'], swap['pswpout'], memory['MemAvailable'])
        rejected = rejected or event['rejection']
        return {'rejection': rejected, 'paging': event, 'memory': memory, 'swap': swap,
                'temperatures_c': temps, 'vram_used_bytes': used, 'vram_total_bytes': total,
                'gpu_busy_percent': busy, 'gpu_clocks_hz': clocks,
                'foreign_server_growth_bytes': foreign_growth,
                'device_health': {'value': None, 'reason': 'no calibrated reset/hang detector; firmware-adjacent thermal stops only, quality flags in analysis'}}

    serve(sample)


# Benchmark-window GPU power state (maintainer decision rv-19d): expected
# power_dpm_force_performance_level and active pp_power_profile_mode name. The
# runner sets it (root) and restores it; this tool only reads and verifies.
POWER_PROFILES = {'auto': ('auto', 'BOOTUP_DEFAULT'), 'compute': ('manual', 'COMPUTE')}


def power_state(device):
    """(performance level, active profile name), or None where unreadable."""
    device = Path(device)
    try:
        level = (device / 'power_dpm_force_performance_level').read_text().strip()
    except OSError:
        level = None
    try:
        active = [line for line in (device / 'pp_power_profile_mode').read_text().splitlines() if '*' in line]
        profile = active[0].split()[1].rstrip('*:') if active else None
    except (OSError, IndexError):
        profile = None
    return level, profile


def preflight(session):
    """Refuse to start unless the GPU is idle, every foreign server is unloaded
    and any expected power profile is active."""
    device = Path(session['device'])
    level, profile = power_state(device)
    used = int((device / 'mem_info_vram_used').read_text())
    busy = int((device / 'gpu_busy_percent').read_text())
    foreign = foreign_servers()
    loaded = {pid: rss for pid, rss in foreign.items() if rss > 1024 ** 3}
    problems = []
    if used > 2 * 1024 ** 3:
        problems.append(f'VRAM already in use: {used / 2**30:.2f} GiB')
    if busy > 20:
        problems.append(f'GPU busy {busy}%')
    if loaded:
        problems.append(f'foreign llama-server with model resident: {sorted(loaded)}')
    expected = session.get('power_profile')
    if expected and (level, profile) != POWER_PROFILES[expected]:
        problems.append(f'power profile {level}/{profile} is not the expected {expected}')
    return {'vram_used_bytes': used, 'gpu_busy_percent': busy,
            'foreign_servers': len(foreign), 'power_level': level, 'power_profile': profile,
            'problems': problems}


def run(session_path):
    session = load_session(session_path)
    root = Path(session_path).resolve().parent
    if (root / 'supervision').exists():
        raise SystemExit('session already run; sessions are single-shot, create a new one')
    check = preflight(session)
    write_private_json(root / 'preflight.json', check)
    if check['problems']:
        print(json.dumps({'status': 'not started', 'problems': check['problems']}))
        return 5
    me = str(Path(__file__).resolve())
    limits = session['limits']
    collector = [sys.executable, '-u', me, 'collect', str(Path(session_path).resolve())]
    enabled = session.get('collector_enabled', True)
    identity = {'schema': SCHEMA, 'session_sha256': hashlib.sha256(Path(session_path).read_bytes()).hexdigest(),
                'label': session['label'], 'collector_enabled': enabled}
    result = supervise([sys.executable, '-u', me, 'drive', str(Path(session_path).resolve())],
                       root / 'supervision', timeout_s=session['session_timeout_s'], grace_s=10,
                       max_output_bytes=limits['max_output_bytes'],
                       max_telemetry_bytes=limits['max_telemetry_bytes'],
                       disk_reserve_bytes=limits['disk_reserve_bytes'], cadence_s=.1 if enabled else 1.0,
                       run_identity=identity, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'LANG': 'C', 'TERM': 'dumb',
                                                   'HOME': os.environ.get('HOME', '/')},
                       telemetry_command=collector, telemetry_timeout_s=1, telemetry_startup_timeout_s=2,
                       cleanup_timeout_s=5)
    summary = {'label': session['label'], 'status': result['status'], 'reason': result['reason'],
               'returncode': result['returncode'], 'elapsed_s': result['elapsed_s'],
               'cleanup_pending': bool(result['workload_cleanup'] and result['workload_cleanup']['pending'])}
    write_private_json(root / 'run-summary.json', summary)
    print(json.dumps(summary))
    return 0 if result['status'] == 'successful' else 1


def request_quality(window, flags, sustain=5):
    """Data-quality flag for one request: any flagged sensor at/above its
    `flag_c` for `sustain` consecutive 100 ms samples. Thermal state is recorded,
    not used to discard data silently; analyses report both views."""
    streaks, sustained = {}, set()
    for snap in window:
        for label, limit in flags.items():
            streaks[label] = streaks.get(label, 0) + 1 if snap['temperatures_c'][label] >= limit else 0
            if streaks[label] >= sustain:
                sustained.add(label)
    sclk = sorted(c for c in (s.get('gpu_clocks_hz', {}).get('sclk') for s in window) if c)
    return {'samples': len(window), 'flagged': bool(sustained), 'sustained_over_flag': sorted(sustained),
            'max_temperature_c': {label: max((s['temperatures_c'][label] for s in window), default=None)
                                  for label in (window[0]['temperatures_c'] if window else {})},
            'sclk_median_hz': sclk[len(sclk) // 2] if sclk else None}


def session_summary(directory):
    """Per-session reduction from retained raw files. Rates are server-reported
    (prompt/predicted ms exclude HTTP transport); wall is client round trip."""
    import statistics
    directory = Path(directory)
    session = json.loads((directory / 'session.json').read_text())
    run_summary = json.loads((directory / 'run-summary.json').read_text())
    driver_path = directory / 'driver.json'
    driver = json.loads(driver_path.read_text()) if driver_path.exists() else {}
    rows = [json.loads(line) for line in (directory / 'requests.jsonl').read_text().splitlines()] \
        if (directory / 'requests.jsonl').exists() else []
    telemetry = [json.loads(line) for line in (directory / 'supervision' / 'telemetry.jsonl').read_text().splitlines()]
    running = [r['snapshot'] for r in telemetry if r['stage'] == 'running' and 'vram_total_bytes' in r['snapshot']]
    measured = [r for r in rows if r['role'] == 'measured' and r['ok']]
    stamped = [(r['timestamp_ns'], r['snapshot']) for r in telemetry
               if r['stage'] == 'running' and 'temperatures_c' in r['snapshot']]
    flags = {s['label']: s['flag_c'] for s in session['sensors'] if 'flag_c' in s}
    for row in measured:
        window = [snap for t, snap in stamped if row['client_sent_ns'] <= t <= row['client_received_ns']]
        row['quality'] = request_quality(window, flags)
    by_length = {}
    for row in measured:
        t = row['timings']
        entry = by_length.setdefault(row['occupied_input_tokens'], {'decode_tps': [], 'prefill_tps': [], 'wall_s': [],
                                                                  'predicted_n': [], 'draft_n': 0, 'draft_accepted': 0})
        entry.setdefault('unflagged_decode_tps', [])
        entry.setdefault('flagged', 0)
        if row['quality']['flagged']:
            entry['flagged'] += 1
        else:
            entry['unflagged_decode_tps'].append(t['predicted_per_second'])
        entry['decode_tps'].append(t['predicted_per_second'])
        entry['prefill_tps'].append(t['prompt_per_second'])
        entry['wall_s'].append((row['client_received_ns'] - row['client_sent_ns']) / 1e9)
        entry['predicted_n'].append(t['predicted_n'])
        entry['draft_n'] += t.get('draft_n', 0)
        entry['draft_accepted'] += t.get('draft_n_accepted', 0)
    reduced = {}
    for length, entry in by_length.items():
        reduced[length] = {
            'n': len(entry['decode_tps']),
            'decode_tps_median': statistics.median(entry['decode_tps']),
            'decode_tps_min': min(entry['decode_tps']), 'decode_tps_max': max(entry['decode_tps']),
            'prefill_tps_median': statistics.median(entry['prefill_tps']),
            'wall_s_median': statistics.median(entry['wall_s']),
            'predicted_n_total': sum(entry['predicted_n']),
            'eos_before_limit': sum(1 for n in entry['predicted_n'] if n < 512),
            'draft_acceptance': entry['draft_accepted'] / entry['draft_n'] if entry['draft_n'] else None,
            'thermal_flagged': entry['flagged'],
            'unflagged_decode_tps_median': statistics.median(entry['unflagged_decode_tps'])
            if entry['unflagged_decode_tps'] else None}
    temps = {}
    for snap in running:
        for label, value in snap['temperatures_c'].items():
            temps[label] = max(temps.get(label, value), value)
    return {'label': session['label'], 'notes': session.get('notes'), 'status': run_summary['status'],
            'reason': run_summary['reason'], 'load_s': driver.get('load_s'),
            'collector_enabled': session.get('collector_enabled', True),
            'requests_ok': sum(1 for r in rows if r['ok']), 'requests_total': len(session['requests']),
            'by_length': reduced,
            'min_free_vram_gib': min((s['vram_total_bytes'] - s['vram_used_bytes']) / 2**30 for s in running) if running else None,
            'min_mem_available_gib': min(s['memory']['MemAvailable'] / 2**30 for s in running) if running else None,
            'max_temperature_c': temps,
            'paging_warnings': sum(1 for s in running if (s.get('paging') or {}).get('warning')),
            'outputs': {r['id']: r['output_sha256'] for r in rows if r['ok']}}


def output_agreement(summaries):
    """Fraction of shared request ids whose greedy outputs are token-identical."""
    first, *rest = summaries
    result = []
    for other in rest:
        shared = sorted(set(first['outputs']) & set(other['outputs']))
        same = sum(1 for key in shared if first['outputs'][key] == other['outputs'][key])
        result.append({'reference': first['label'], 'other': other['label'], 'shared': len(shared), 'identical': same})
    return result


def overhead(directories, seed=20261112):
    """Pair collector-on and collector-off sessions by schedule order and match
    requests by fixture; compares server decode rate and client wall time."""
    from benchmark_core import paired_improvement
    sessions = [Path(d) for d in directories]
    on = [d for d in sessions if json.loads((d / 'session.json').read_text())['collector_enabled']]
    off = [d for d in sessions if not json.loads((d / 'session.json').read_text())['collector_enabled']]
    if len(on) != len(off):
        raise ValueError('unbalanced overhead design')
    pairs = list(zip(sorted(on), sorted(off)))
    decode_off, decode_on, wall_off, wall_on, cycles = [], [], [], [], []
    for d_on, d_off in pairs:
        rows = {}
        for d, tag in ((d_on, 'on'), (d_off, 'off')):
            for line in (d / 'requests.jsonl').read_text().splitlines():
                r = json.loads(line)
                if r['role'] == 'measured' and r['ok']:
                    rows.setdefault(r['id'], {})[tag] = r
        for pair in rows.values():
            if set(pair) != {'on', 'off'}:
                continue
            decode_on.append(pair['on']['timings']['predicted_per_second'])
            decode_off.append(pair['off']['timings']['predicted_per_second'])
            wall_on.append((pair['on']['client_received_ns'] - pair['on']['client_sent_ns']) / 1e9)
            wall_off.append((pair['off']['client_received_ns'] - pair['off']['client_sent_ns']) / 1e9)
        result = json.loads((d_on / 'supervision' / 'result.json').read_text())
        cycles.extend(result['collector_cycle_durations_ns'])
    from benchmark_core import percentile
    return {'session_pairs': len(pairs), 'matched_requests': len(decode_on),
            'decode_rate_change_on_vs_off': paired_improvement(decode_off, decode_on, higher_is_better=True, seed=seed),
            'wall_time_benefit_on_vs_off': paired_improvement(wall_off, wall_on, higher_is_better=False, seed=seed),
            'collector_cycle_p95_ns': percentile(cycles, .95), 'collector_cycles': len(cycles),
            'design': 'on = 100 ms measurement+safety collector; off = 1 s safety-only collector; ABBA session order'}


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'summarize':
        print(json.dumps([session_summary(d) for d in sys.argv[2:]], indent=1))
    elif command == 'overhead':
        print(json.dumps(overhead(sys.argv[2:]), indent=1))
    else:
        sys.exit({'run': run, 'drive': drive, 'collect': collect}[command](sys.argv[2]) or 0)
