"""Explicit resource-series reduction, unit preservation and incomplete sources."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_metrics as M
import benchmark_resources as R
import benchmark_core as B
from benchmark_core import missing_metrics


def process_stat(*, minor, major, user, system, start):
    fields = ['S'] + ['0'] * 19
    for index, value in [(7, minor), (9, major), (11, user), (12, system), (19, start)]:
        fields[index] = str(value)
    return '123 (synthetic process) ' + ' '.join(fields)


def readings(index, *, process_start=900, swap_in=None, rss_kb=None, busy=None, power=None):
    swap_in = [10, 14, 20][index] if swap_in is None else swap_in
    rss_kb = [100, 200, 300][index] if rss_kb is None else rss_kb
    busy = [20, 40, 50][index] if busy is None else busy
    power = [100_000_000, 200_000_000, 100_000_000][index] if power is None else power
    memory = f'MemAvailable: {[900, 800, 700][index]} kB\nSwapFree: 500 kB\nSwapTotal: 1000 kB'
    swap = f'pswpin {swap_in}\npswpout {[2, 3, 5][index]}'
    status = (f'VmRSS: {rss_kb} kB\nVmHWM: {rss_kb + 20} kB\n'
              f'RssAnon: {rss_kb // 2} kB\nRssFile: {rss_kb // 2} kB')
    return R.linux_metrics(memory=memory, swap=swap, status=status,
                           smaps=f'Pss: {[80, 150, 220][index]} kB',
                           process_stat=process_stat(minor=10 + 5 * index, major=1 + index,
                                                     user=100 + 50 * index, system=20 + 10 * index,
                                                     start=process_start),
                           page_size=4096, ticks_per_second=100,
                           gpu={'vram_used_bytes': [100, 200, 250][index],
                                'vram_total_bytes': 1000,
                                'gpu_busy_percent': busy,
                                'temperature_millicelsius': [50000, 55000, 60000][index],
                                'clock_hz': [100, 200, 300][index],
                                'power_microwatts': power})


def series(*, starts=(900, 900, 900), swap=None, remove=None):
    result = []
    for index, timestamp in enumerate((0, 1_000_000_000, 2_000_000_000)):
        metrics = readings(index, process_start=starts[index],
                           swap_in=swap[index] if swap else None)
        if remove == (index, 'pss_bytes'):
            metrics['pss_bytes'] = {'value': None, 'reason': 'source not supplied', 'unit': 'bytes'}
        result.append({'timestamp_ns': timestamp, 'metrics': metrics})
    return result


class BenchmarkMetricTests(unittest.TestCase):
    def test_run_aggregates_have_defined_units_and_exact_denominators(self):
        samples = series()
        idle = [{'timestamp_ns': i * 100_000_000,
                 'metrics': dict(readings(i, busy=10, power=0), vram_used_bytes={
                     'value': used, 'reason': None, 'unit': 'bytes'})}
                for i, used in ((0, 80), (1, 100))]
        summary = M.summarize_resource_metrics(samples, idle_samples=idle,
                                              steady_start_ns=1_000_000_000,
                                              committed_output_tokens=2,
                                              max_weighted_gap_s=1)
        metrics = summary['metrics']
        self.assertEqual(set(metrics), set(__import__('benchmark_core').METRICS))
        self.assertEqual(metrics['vram_idle_bytes']['value'], 90)
        self.assertEqual(metrics['vram_peak_bytes']['value'], 250)
        self.assertEqual(metrics['vram_steady_p50_bytes']['value'], 225)
        self.assertEqual(metrics['vram_steady_p95_bytes']['value'], 247.5)
        self.assertEqual(metrics['rss_bytes']['value'], 300 * 1024)
        self.assertEqual(metrics['memory_growth_bytes']['value'], 200 * 1024)
        self.assertEqual(metrics['mem_available_bytes']['value'], 700 * 1024)
        self.assertEqual(metrics['page_faults']['value'], 12)
        self.assertAlmostEqual(metrics['cpu_core_seconds']['value'], 1.2)
        self.assertEqual(metrics['swap_in_bytes']['value'], 10 * 4096)
        self.assertEqual(metrics['swap_out_bytes']['value'], 3 * 4096)
        self.assertEqual(metrics['temperature_start_c']['value'], 50)
        self.assertEqual(metrics['temperature_end_c']['value'], 60)
        self.assertAlmostEqual(metrics['gpu_busy_fraction']['value'], .375)
        self.assertAlmostEqual(metrics['clock_hz']['value'], 200)
        self.assertEqual(metrics['board_energy_j']['value'], 300)
        self.assertEqual(metrics['energy_j_per_committed_token']['value'], 150)
        self.assertEqual(summary['coverage']['maximum_sample_gap_s'], 1)
        self.assertEqual(summary['coverage']['energy_coverage_s'], 2)
        self.assertEqual(summary['coverage']['energy_max_gap_s'], 1)
        self.assertTrue(all(entry['unit'] and (entry['value'] is not None or entry['reason'])
                            for entry in metrics.values()))

    def test_missing_sources_identity_changes_and_counter_resets_stay_explicit(self):
        incomplete = M.summarize_resource_metrics(series(remove=(1, 'pss_bytes')))
        self.assertIsNone(incomplete['metrics']['pss_bytes']['value'])
        self.assertIn('unavailable at sample 1', incomplete['metrics']['pss_bytes']['reason'])
        changed = M.summarize_resource_metrics(series(starts=(900, 901, 901)))
        self.assertIsNone(changed['metrics']['page_faults']['value'])
        self.assertIsNone(changed['metrics']['cpu_core_seconds']['value'])
        self.assertIsNone(changed['metrics']['memory_growth_bytes']['value'])
        reset = M.summarize_resource_metrics(series(swap=(10, 3, 20)))
        self.assertIn('counter reset', reset['metrics']['swap_in_bytes']['reason'])
        no_window = M.summarize_resource_metrics(series())
        self.assertIn('boundary not supplied', no_window['metrics']['vram_steady_p95_bytes']['reason'])
        self.assertIn('not supplied', no_window['metrics']['vram_idle_bytes']['reason'])

    def test_sample_gap_and_missing_power_do_not_invent_weighted_metrics(self):
        samples = series()
        gap = M.summarize_resource_metrics(samples, max_weighted_gap_s=.5)
        self.assertIsNone(gap['metrics']['gpu_busy_fraction']['value'])
        self.assertIn('exceeds declared', gap['metrics']['clock_hz']['reason'])
        missing_power = [dict(sample, metrics=dict(sample['metrics'], board_power_w={
            'value': None, 'reason': 'source not supplied', 'unit': 'W'})) for sample in samples]
        result = M.summarize_resource_metrics(missing_power, committed_output_tokens=3)
        self.assertIsNone(result['metrics']['board_energy_j']['value'])
        self.assertIsNone(result['metrics']['energy_j_per_committed_token']['value'])

    def test_invalid_sample_order_and_denominators_reject(self):
        samples = series()
        samples[2]['timestamp_ns'] = samples[1]['timestamp_ns']
        with self.assertRaises(ValueError):
            M.summarize_resource_metrics(samples)
        with self.assertRaises(ValueError):
            M.summarize_resource_metrics(series(), committed_output_tokens=True)

    def test_resource_guard_rows_adapt_running_process_fields_and_hide_sensor_names(self):
        rows = []
        for i, timestamp in enumerate((10, 20, 30)):
            rows.append({
                'timestamp_ns': timestamp, 'stage': 'running',
                'snapshot': {
                    'memory': {'MemAvailable': 1000 - i * 100},
                    'swap': {'pswpin': i * 4096, 'pswpout': i * 8192},
                    'process': {
                        'memory': {'rss_bytes': 100 + i, 'pss_bytes': 90 + i,
                                   'anonymous_bytes': 50 + i, 'file_backed_bytes': 50,
                                   'high_water_rss_bytes': 120 + i},
                        'counters': {'page_faults': i * 3, 'cpu_core_seconds': i * .5,
                                     'process_start_ticks': 777},
                        'pss_bytes': {'value': 80 + i, 'reason': None},
                    },
                    'vram_used_bytes': 500 + i, 'gpu_busy_percent': i * 10,
                    'temperatures_c': {'private sensor name': 40 + i},
                },
            })
        rows.insert(0, {'timestamp_ns': 1, 'stage': 'preflight', 'snapshot': {}})
        summary = M.summarize_resource_guard_telemetry(rows, max_weighted_gap_s=1e-8)
        metrics = summary['metrics']
        self.assertEqual(summary['coverage']['sample_count'], 3)
        self.assertEqual(metrics['rss_bytes']['value'], 102)
        self.assertEqual(metrics['page_faults']['value'], 6)
        self.assertEqual(metrics['cpu_core_seconds']['value'], 1.0)
        self.assertEqual(metrics['swap_in_bytes']['value'], 8192)
        self.assertEqual(metrics['temperature_end_c']['value'], 42)
        self.assertIsNone(metrics['board_energy_j']['value'])
        self.assertNotIn('private sensor name', repr(summary))
        self.assertEqual(summary['coverage']['temperature_scope'],
                         'hottest selected sensor per sample; start/end may refer to different sensors')

    def test_metric_sources_merge_only_complete_unit_valid_nonconflicting_values(self):
        resource = missing_metrics('resource source unavailable')
        native = missing_metrics('native source unavailable')
        delivery = M.delivery_metric_inventory(1_000_000_000, 4_000_000_000,
                                               [2_000_000_000, 2_500_000_000, 3_000_000_000])
        native['prefill_tokens_per_s'] = {'value': 100.0, 'reason': None, 'unit': 'tokens/s'}
        resource['rss_bytes'] = {'value': 512, 'reason': None, 'unit': 'bytes'}
        merged = M.merge_metric_sources({'resource': resource, 'native': native,
                                         'delivery': delivery})
        self.assertEqual(merged['metrics']['prefill_tokens_per_s']['value'], 100.0)
        self.assertEqual(merged['metrics']['ttft_s']['value'], 1)
        self.assertEqual(merged['metrics']['steady_output_tokens_per_s']['value'], 2)
        self.assertEqual(merged['metrics']['rss_bytes']['value'], 512)
        self.assertEqual(merged['metric_sources']['prefill_tokens_per_s'], ['native'])
        self.assertIsNone(merged['metrics']['target_decode_tokens_per_s']['value'])
        conflict = dict(native)
        conflict['rss_bytes'] = {'value': 1024, 'reason': None, 'unit': 'bytes'}
        with self.assertRaisesRegex(ValueError, 'conflicting measured values'):
            M.merge_metric_sources({'resource': resource, 'native': conflict})
        invalid = dict(native)
        invalid['prefill_tokens_per_s'] = {'value': 100, 'reason': None, 'unit': 'seconds'}
        with self.assertRaisesRegex(ValueError, 'schema or unit'):
            M.merge_metric_sources({'native': invalid})

    def test_resource_guard_adapter_persists_telemetry_and_additional_timing_sources(self):
        native = missing_metrics('native diagnostic did not provide this metric')
        native['prefill_tokens_per_s'] = {'value': 25.0, 'reason': None, 'unit': 'tokens/s'}
        delivery = M.delivery_metric_inventory(1_000_000_000, 3_000_000_000,
                                               [1_500_000_000, 2_000_000_000])
        adapter = M.resource_guard_metric_adapter(
            additional_sources={'native': native, 'delivery': delivery})

        def telemetry(_pid, _elapsed, _size):
            return {
                'memory': {'MemAvailable': 4096}, 'swap': {'pswpin': 8192, 'pswpout': 4096},
                'process': {
                    'memory': {'rss_bytes': 128, 'anonymous_bytes': 64,
                               'file_backed_bytes': 64, 'high_water_rss_bytes': 160},
                    'pss_bytes': {'value': 96, 'reason': None},
                    'counters': {'page_faults': 3, 'cpu_core_seconds': .25,
                                 'process_start_ticks': 123},
                },
                'vram_used_bytes': 512, 'gpu_busy_percent': 10,
                'temperatures_c': {'private test sensor': 45.0},
            }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'integrated-run'
            result = B.supervise([sys.executable, '-c', 'import time; time.sleep(.02)'], path,
                                 timeout_s=2, cadence_s=.003, telemetry=telemetry,
                                 metric_adapter=adapter)
            persisted = json.loads((path / 'result.json').read_text())
        self.assertEqual(result['status'], 'successful')
        self.assertEqual(persisted['metrics']['rss_bytes']['value'], 128)
        self.assertEqual(persisted['metrics']['prefill_tokens_per_s']['value'], 25)
        self.assertEqual(persisted['metrics']['ttft_s']['value'], .5)
        self.assertIn('resource_guard', persisted['metric_sources']['rss_bytes'])
        self.assertEqual(persisted['metric_sources']['prefill_tokens_per_s'], ['native'])
        self.assertGreaterEqual(persisted['metric_metadata']['resource_guard_coverage']['sample_count'], 1)
        self.assertNotIn('private test sensor', repr(persisted))


if __name__ == '__main__':
    unittest.main()
