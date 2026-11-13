import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import gpu_campaign  # noqa: E402
from benchmark_core import digest  # noqa: E402

FAKE_SERVER = textwrap.dedent('''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import socketserver
    mode = os.environ.get('FAKE_MODE', 'ok')
    host = sys.argv[sys.argv.index('--host') + 1]
    if mode == 'never-ready':
        time.sleep(60)
    class Server(socketserver.UnixStreamServer, HTTPServer):
        def server_bind(self):
            socketserver.UnixStreamServer.server_bind(self)
            self.server_name, self.server_port = 'fake', 0
    class Handler(BaseHTTPRequestHandler):
        def address_string(self):
            return 'unix'
        def log_message(self, *args):
            pass
        def reply(self, body):
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            self.reply({'status': 'ok'})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt_n = len(body['prompt']) - (1 if mode == 'cached' else 0)
            tokens = list(range(body['n_predict']))
            self.reply({'tokens': tokens, 'stop_type': 'limit', 'tokens_predicted': len(tokens),
                        'tokens_evaluated': prompt_n,
                        'timings': {'prompt_n': prompt_n, 'prompt_ms': 1.0,
                                    'predicted_n': len(tokens), 'predicted_ms': 2.0}})
    Server(host, Handler).serve_forever()
''')


def workload(path, ids, temperature=0):
    record = {'token_ids': ids, 'tokens_sha256': digest(ids), 'occupied_input_tokens': len(ids),
              'policy': {'sampler_order': ['temperature'], 'sampler_parameters': {'temperature': temperature},
                         'seed': 7, 'stops': [], 'thinking': True},
              'manifest_sha256': 'm' * 64, 'fixture_id': 'f'}
    path.write_text(json.dumps(record))
    return str(path)


class GpuCampaignTests(unittest.TestCase):
    def session(self, root, mode='ok', load_timeout=10):
        server = root / 'fake_server.py'
        server.write_text(FAKE_SERVER)
        w = workload(root / 'w.json', [2, 5, 9])
        session = {'schema': gpu_campaign.SCHEMA, 'label': 't',
                   'server_argv': ['/usr/bin/env', f'FAKE_MODE={mode}', sys.executable, str(server)],
                   'requests': [{'id': 'a', 'workload': w, 'n_predict': 4, 'role': 'warmup'},
                                {'id': 'b', 'workload': w, 'n_predict': 4, 'role': 'measured'}],
                   'load_timeout_s': load_timeout, 'request_timeout_s': 5, 'session_timeout_s': 30,
                   'limits': {k: 1 for k in gpu_campaign.LIMITS}, 'device': '/nonexistent', 'sensors': []}
        path = root / 'session.json'
        path.write_text(json.dumps(session))
        return path

    def test_drive_records_every_request_and_stops_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(gpu_campaign.drive(self.session(root)), 0)
            rows = [json.loads(line) for line in (root / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in rows], ['a', 'b'])
            self.assertTrue(all(r['ok'] and r['output_token_ids'] == [0, 1, 2, 3] for r in rows))
            driver = json.loads((root / 'driver.json').read_text())
            self.assertIn('load_s', driver)
            self.assertIsNotNone(driver['server_returncode'])
            self.assertFalse((root / 'server.sock').exists())

    def test_prompt_count_mismatch_fails_and_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(gpu_campaign.drive(self.session(root, mode='cached')), 3)
            row = json.loads((root / 'requests.jsonl').read_text().splitlines()[0])
            self.assertFalse(row['ok'])
            self.assertIn('prompt_n', row['reason'])

    def test_load_deadline_is_separate_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(gpu_campaign.drive(self.session(root, mode='never-ready', load_timeout=1)), 2)
            driver = json.loads((root / 'driver.json').read_text())
            self.assertEqual(driver['failure'], 'load deadline exceeded')

    def test_rejects_tcp_listener_and_non_greedy_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.session(root)
            session = json.loads(path.read_text())
            session['server_argv'] += ['--port', '8080']
            path.write_text(json.dumps(session))
            with self.assertRaises(ValueError):
                gpu_campaign.load_session(path)
            record = json.loads(Path(workload(root / 'hot.json', [1], temperature=1)).read_text())
            with self.assertRaises(ValueError):
                gpu_campaign.request_body(record, 4)

    def test_thermal_debounce_ignores_single_spike_but_not_sustained_or_critical(self):
        sensors = [{'label': 'ccd', 'ceiling_c': 90, 'sustain_samples': 5, 'immediate_c': 95},
                   {'label': 'ctl', 'ceiling_c': 90}]
        streaks = {}
        check = gpu_campaign.thermal_rejection
        self.assertIsNone(check(streaks, {'ccd': 91.9, 'ctl': 70}, sensors))  # one spike
        self.assertIsNone(check(streaks, {'ccd': 70, 'ctl': 70}, sensors))    # streak resets
        for _ in range(4):
            self.assertIsNone(check(streaks, {'ccd': 90, 'ctl': 70}, sensors))
        self.assertEqual(check(streaks, {'ccd': 90, 'ctl': 70}, sensors), 'thermal stop: ccd')
        self.assertEqual(check({}, {'ccd': 95, 'ctl': 70}, sensors), 'thermal stop: ccd')
        self.assertEqual(check({}, {'ccd': 50, 'ctl': 90}, sensors), 'thermal stop: ctl')

    def test_request_quality_flags_sustained_heat_not_spikes(self):
        def snap(t, sclk=2_400_000_000):
            return {'temperatures_c': {'j': t}, 'gpu_clocks_hz': {'sclk': sclk}}
        spike = [snap(80)] * 3 + [snap(99)] + [snap(80)] * 3
        result = gpu_campaign.request_quality(spike, {'j': 95})
        self.assertFalse(result['flagged'])
        self.assertEqual(result['max_temperature_c'], {'j': 99})
        hot = [snap(96, sclk=1_800_000_000)] * 5
        result = gpu_campaign.request_quality(hot, {'j': 95})
        self.assertTrue(result['flagged'])
        self.assertEqual(result['sustained_over_flag'], ['j'])
        self.assertEqual(result['sclk_median_hz'], 1_800_000_000)
        self.assertFalse(gpu_campaign.request_quality([], {'j': 95})['flagged'])

    def test_command_mode_runs_tool_and_records_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.session(root)
            session = json.loads(path.read_text())
            session.update({'mode': 'command', 'requests': [],
                            'server_argv': [sys.executable, '-c', 'open("out.txt","w").write("ok")']})
            path.write_text(json.dumps(session))
            self.assertEqual(gpu_campaign.drive(path), 0)
            self.assertEqual((root / 'out.txt').read_text(), 'ok')
            self.assertEqual(json.loads((root / 'driver.json').read_text())['returncode'], 0)
            session['server_argv'] = [sys.executable, '-c', 'raise SystemExit(3)']
            path.write_text(json.dumps(session))
            (root / 'driver.json').unlink()  # sessions are single-shot; reset for the failure case
            self.assertEqual(gpu_campaign.drive(path), 6)
            session['requests'] = [{'id': 'x', 'workload': 'w', 'n_predict': 1, 'role': 'warmup'}]
            path.write_text(json.dumps(session))
            with self.assertRaises(ValueError):
                gpu_campaign.load_session(path)

    def test_power_profile_is_recorded_and_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = Path(tmp)
            (device / 'mem_info_vram_used').write_text('0')
            (device / 'gpu_busy_percent').write_text('0')
            (device / 'power_dpm_force_performance_level').write_text('auto\n')
            modes = 'PROFILE_INDEX(NAME) CLOCK_TYPE(NAME)\n 0 BOOTUP_DEFAULT{0}:\n    0( GFXCLK) 0\n 5        COMPUTE{1}:\n'
            (device / 'pp_power_profile_mode').write_text(modes.format('*', ' '))
            session = {'device': tmp, 'power_profile': 'compute'}
            check = gpu_campaign.preflight(session)
            self.assertEqual((check['power_level'], check['power_profile']), ('auto', 'BOOTUP_DEFAULT'))
            self.assertTrue(any('power profile' in p for p in check['problems']))
            (device / 'power_dpm_force_performance_level').write_text('manual\n')
            (device / 'pp_power_profile_mode').write_text(modes.format(' ', '*'))
            check = gpu_campaign.preflight(session)
            self.assertFalse(any('power profile' in p for p in check['problems']))
            self.assertEqual(gpu_campaign.power_state(device / 'missing'), (None, None))

    def test_tampered_workload_tokens_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(workload(Path(tmp) / 'w.json', [1, 2]))
            record = json.loads(path.read_text())
            record['token_ids'] = [1, 3]
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                gpu_campaign.load_workload(path)


if __name__ == '__main__':
    unittest.main()
