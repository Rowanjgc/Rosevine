"""Synthetic units, unsafe readings and resource cancellation decisions."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('benchmark_resources', Path(__file__).resolve().parents[2] / 'scripts/benchmark_resources.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ResourceTests(unittest.TestCase):
    def test_linux_counter_units(self):
        self.assertEqual(R.meminfo('MemAvailable: 4096 kB\nSwapTotal: 8192 kB\nSwapFree: 4096 kB')['MemAvailable'], 4194304)
        self.assertEqual(R.vmstat('pswpin 2\npswpout 3', 4096), {'pswpin': 8192, 'pswpout': 12288})
        for text in ['MemAvailable: 3 MB', 'MemAvailable: -1 kB', '']:
            with self.assertRaises(ValueError):
                R.meminfo(text)
        with self.assertRaises(ValueError):
            R.vmstat('pswpin 1\npswpin 2\npswpout 0', 4096)

    def test_budget_boundaries_and_missing_readings(self):
        limits = dict(ram_reserve_bytes=100, vram_reserve_bytes=100, disk_reserve_bytes=100,
                      max_temperature_c=70, max_elapsed_s=60, max_capture_bytes=1000)
        snapshot = dict(mem_available_bytes=100, vram_free_bytes=100, disk_free_bytes=100,
                        temperature_c=50, elapsed_s=1, capture_bytes=0,
                        swap_in_bytes=0, swap_out_bytes=0, device_healthy=True)
        initial = dict(swap_in_bytes=0, swap_out_bytes=0)
        self.assertIsNone(R.budget_rejection(snapshot, limits, initial))
        for key, value in [('vram_free_bytes', 99), ('mem_available_bytes', None),
                           ('temperature_c', 70), ('elapsed_s', 60), ('capture_bytes', 1000),
                           ('device_healthy', False)]:
            with self.subTest(key=key):
                self.assertIsNotNone(R.budget_rejection(dict(snapshot, **{key: value}), limits, initial))
        with self.assertRaises(ValueError):
            R.budget_rejection(snapshot, dict(limits, max_temperature_c=None), initial)
        self.assertIsNone(R.budget_rejection(dict(snapshot, swap_in_bytes=4096), limits, initial))
        unknown = dict(snapshot, device_healthy=None)
        self.assertIsNotNone(R.budget_rejection(unknown, limits, initial))
        self.assertIsNone(R.budget_rejection(unknown, limits, initial, require_device_health=False))
        self.assertEqual(R.budget_rejection(dict(snapshot, device_healthy=False), limits, initial,
                                           require_device_health=False), 'device health failure')

    def test_process_units_and_identity_discard(self):
        memory = R.process_status('Name:\tsynthetic\nVmRSS: 12 kB\nVmHWM: 16 kB\nRssAnon: 4 kB\nRssFile: 8 kB')
        self.assertEqual(memory, {'rss_bytes': 12288, 'high_water_rss_bytes': 16384,
                                  'anonymous_bytes': 4096, 'file_backed_bytes': 8192})
        self.assertEqual(R.process_pss('Pss: 7 kB\nPrivate_Dirty: 5 kB'), 7168)
        fields = ['S'] + ['0'] * 19
        for index, value in [(7, 11), (9, 2), (11, 150), (12, 50), (19, 900)]:
            fields[index] = str(value)
        counters = R.process_counters('123 (a name) with parens) ' + ' '.join(fields), 100)
        self.assertEqual(counters, {'page_faults': 13, 'minor_faults': 11, 'major_faults': 2,
                                    'cpu_core_seconds': 2, 'process_start_ticks': 900})
        with self.assertRaises(ValueError):
            R.process_counters('123 invalid', 100)

    def test_snapshot_adapter_missing_invalid_and_gpu_units(self):
        metrics = R.linux_metrics(gpu={'gpu_busy_percent': 25, 'vram_used_bytes': 1024,
                                       'vram_total_bytes': 2048, 'temperature_millicelsius': 65000,
                                       'power_microwatts': 120000000, 'clock_hz': 2000000000})
        self.assertEqual(metrics['gpu_busy_fraction']['value'], .25)
        self.assertEqual(metrics['temperature_c']['value'], 65)
        self.assertEqual(metrics['board_power_w']['value'], 120)
        self.assertIsNone(metrics['rss_bytes']['value'])
        self.assertTrue(metrics['rss_bytes']['reason'])
        bad = R.linux_metrics(status='VmRSS: bad kB', gpu={'vram_used_bytes': 3, 'vram_total_bytes': 2})
        self.assertEqual(bad['rss_bytes']['reason'], 'invalid source counters')
        self.assertIsNone(bad['vram_used_bytes']['value'])
        with self.assertRaises(ValueError):
            R.linux_metrics(gpu={'power_cap': 100})

    def test_sampled_board_energy_and_gap_reporting(self):
        result = R.sampled_energy([(0, 100), (1000000000, 200), (3000000000, 100)])
        self.assertEqual(result['energy_j'], 450)
        self.assertEqual(result['max_gap_s'], 2)
        self.assertEqual(result['coverage_s'], 3)
        for samples in [[(0, 1)], [(0, 1), (0, 2)], [(0, 1), (1, float('nan'))]]:
            with self.assertRaises(ValueError):
                R.sampled_energy(samples)

    def test_resource_guard_preserves_preflight_and_successive_swap_deltas(self):
        limits = dict(ram_reserve_bytes=100, vram_reserve_bytes=100, disk_reserve_bytes=100,
                      max_temperature_c=70, max_elapsed_s=60, max_capture_bytes=1000)
        readings = dict(mem_available_bytes=100, vram_free_bytes=100, disk_free_bytes=100,
                        temperature_c=50, swap_in_bytes=10, swap_out_bytes=20, device_healthy=True)
        guard = R.ResourceGuard(lambda pid: dict(readings), limits)
        self.assertIsNone(guard(None, 0, 0)['rejection'])
        self.assertIsNone(guard(123, 1, 100)['rejection'])
        readings['swap_out_bytes'] = 21
        observation = guard(123, 2, 200)
        self.assertIsNone(observation['rejection'])
        self.assertEqual(observation['paging']['delta_bytes'], 1)
        readings['swap_out_bytes'] = 20
        observation = guard(123, 3, 200)
        self.assertIsNone(observation['rejection'])
        self.assertIn('reset', observation['paging']['warning'])
        readings['device_healthy'] = None
        self.assertEqual(guard(None, 0, 0)['rejection'], 'missing or invalid safety reading')

    def test_paging_small_events_and_one_large_burst_do_not_cancel(self):
        monitor = R.PagingMonitor(2 * 1024**3)
        for second, count in enumerate([0, 4096, 64 * 1024**2, 64 * 1024**2, 64 * 1024**2]):
            result = monitor.update(second, count, 0, 3 * 1024**3)
            self.assertIsNone(result['rejection'])
        self.assertEqual(result['pressure_duration_s'], 0)

    def test_sustained_paging_requires_rate_duration_and_memory_pressure(self):
        rate = 16 * 1024**2
        for available, measured_rate, should_stop in [(3 * 1024**3, rate, True),
                                                      (5 * 1024**3, rate, False),
                                                      (3 * 1024**3, rate - 1, False)]:
            with self.subTest(available=available, rate=measured_rate):
                monitor = R.PagingMonitor(2 * 1024**3)
                for second in range(4):
                    result = monitor.update(second, second * measured_rate, 0, available)
                    if second < 3:
                        self.assertIsNone(result['rejection'])
                self.assertEqual(result['rejection'] is not None, should_stop)

    def test_paging_reset_gap_and_recovery_cannot_manufacture_duration(self):
        rate = 16 * 1024**2
        monitor = R.PagingMonitor(2 * 1024**3)
        for second in range(3):
            monitor.update(second, second * rate, 0, 3 * 1024**3)
        result = monitor.update(3, 0, 0, 3 * 1024**3)
        self.assertIn('reset', result['warning'])
        self.assertEqual(result['pressure_duration_s'], 0)
        result = monitor.update(5, rate, 0, 3 * 1024**3)
        self.assertIn('gap', result['warning'])
        self.assertEqual(result['pressure_duration_s'], 0)
        monitor.update(6, 2 * rate, 0, 3 * 1024**3)
        result = monitor.update(7, 3 * rate, 0, 5 * 1024**3)
        self.assertEqual(result['pressure_duration_s'], 0)
        with self.assertRaises(ValueError):
            monitor.update(7, 3 * rate, 0, 5 * 1024**3)
        with self.assertRaises(ValueError):
            monitor.update(8, -1, 0, 5 * 1024**3)

    def test_resource_guard_paging_and_hard_reserve_have_distinct_stops(self):
        gib = 1024**3
        limits = dict(ram_reserve_bytes=2*gib, vram_reserve_bytes=gib, disk_reserve_bytes=gib,
                      max_temperature_c=80, max_elapsed_s=60, max_capture_bytes=1000)
        readings = dict(mem_available_bytes=3*gib, vram_free_bytes=2*gib, disk_free_bytes=2*gib,
                        temperature_c=50, swap_in_bytes=0, swap_out_bytes=0, device_healthy=None)
        guard = R.ResourceGuard(lambda pid: dict(readings), limits, require_device_health=False)
        for second in range(4):
            readings['swap_out_bytes'] = second * 16 * 1024**2
            result = guard(None if second == 0 else 123, second, 0)
        self.assertEqual(result['rejection'], 'sustained paging under memory pressure')
        readings['mem_available_bytes'] = 2*gib - 1
        self.assertEqual(guard(123, 4, 0)['rejection'], 'mem_available_bytes reserve breach')
