#!/usr/bin/env python3
"""Run one command inside a COMPUTE power-profile benchmark window.

Maintainer decision (rv-19d): benchmark windows use the amdgpu COMPUTE power
profile, which removed the bimodal shader-clock state of automatic power
management. This wrapper sets the profile through `sudo -n` (it never prompts),
verifies it, runs the command, and always restores automatic power management
(BOOTUP_DEFAULT profile, `auto` level), including on SIGINT/SIGTERM.

Usage: gpu_power_window.py [--device /sys/class/drm/cardN/device] -- COMMAND...
"""
import argparse
import signal
import subprocess
import sys
from pathlib import Path


def profile_index(device, name):
    for line in (device / 'pp_power_profile_mode').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].rstrip('*:') == name:
            return int(fields[0])
    raise SystemExit(f'power profile {name} not offered by {device}')


def write(device, name, value):
    subprocess.run(['sudo', '-n', 'tee', str(device / name)], input=f'{value}\n', text=True,
                   stdout=subprocess.DEVNULL, check=True)


def state(device):
    level = (device / 'power_dpm_force_performance_level').read_text().strip()
    active = [line.split()[1].rstrip('*:') for line in (device / 'pp_power_profile_mode').read_text().splitlines()
              if '*' in line]
    return level, active[0] if active else None


def set_compute(device):
    write(device, 'power_dpm_force_performance_level', 'manual')
    write(device, 'pp_power_profile_mode', profile_index(device, 'COMPUTE'))
    if state(device) != ('manual', 'COMPUTE'):
        raise SystemExit(f'COMPUTE profile not active: {state(device)}')


def restore(device):
    write(device, 'power_dpm_force_performance_level', 'manual')
    write(device, 'pp_power_profile_mode', profile_index(device, 'BOOTUP_DEFAULT'))
    write(device, 'power_dpm_force_performance_level', 'auto')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', default='/sys/class/drm/card1/device', type=Path)
    parser.add_argument('command', nargs='+')
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    before = state(args.device)
    if before != ('auto', 'BOOTUP_DEFAULT'):
        raise SystemExit(f'refusing: GPU power state is {before}, not automatic; restore it first')
    try:
        set_compute(args.device)
        print(f'power window: {state(args.device)}', file=sys.stderr, flush=True)
        return subprocess.run(args.command).returncode
    finally:
        restore(args.device)
        print(f'power window restored: {state(args.device)}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    sys.exit(main())
