# H1 environment and measurement access

Investigation date: 2026-11-11. H1 (`rv-e01-898.1`) characterizes access;
it does not certify backend correctness or authorize a benchmark window.
The [baseline](hardware-and-inference-baseline.md) gives the hardware inventory;
[Epic 1 evidence](epic-1-evidence.md) gives independent model/API observations.

## Inventory and provenance

Repeated CPU/package/kernel probes agree with the baseline: Ryzen 5 7600X,
6 cores / 12 threads, AVX2/FMA/F16C and advertised AVX-512 including VNNI/BF16;
Fedora 43, kernel 7.2.9-101.fc43.x86_64; HIP 6.4.2, clang 19, Mesa 25.3.6,
Vulkan loader 1.4.341. The private `environment.json` records command exit codes,
selected outputs, capture time, memory availability and swap counters.
`vulkan-summary.txt`, `rocminfo.txt` and `telemetry-access.json` retain independent
API enumeration and attribute access evidence. No environment or process-command
line dump was taken. Production settings were unchanged.

RAM capacity/channel/rate evidence remains the planning DMI observation:
two 16 GB single-rank DDR5 modules on A/B, configured at 4800 MT/s. No new
privileged probe was attempted. Live DRAM timings have no verified readable
interface in this investigation. Effective RAM and PCIe bandwidth require
separate approved workload measurements; neither is inferred from ratings.
PCIe sysfs reports current/max 16 GT/s, width 16 for the discrete GPU.

## Device selection and measurement support

| Facility | Observed access / selection | Limit and follow-up |
| --- | --- | --- |
| Discrete GPU | DRM PCI device ID 0x747e; Vulkan names RX 7800 XT/RADV NAVI32; HSA names RX 7800 XT/gfx1101 | Select the device by correlated identity; this session's adjacent llama tools call it Vulkan0 |
| Integrated GPU | PCI 0x164e; RADV RAPHAEL and HSA gfx1036 | Exclude from discrete measurements; also exclude llvmpipe CPU Vulkan |
| Device memory | Readable total/used VRAM sysfs; total 17163091968 bytes | Sampled device-wide values include desktop/other workloads; not process allocation high-water marks |
| GPU activity | Readable gpu_busy_percent | Cadence/attribution not calibrated; nonzero activity invalidates an idle assumption |
| Temperatures/clocks | Readable amdgpu hwmon temperature inputs/critical limits and frequency inputs/labels | Units and sensor mapping need validation before a thermal stopping threshold is approved |
| Board power | Power label/cap readable; no power input captured by this probe | No calibrated power/energy metric; cap is not consumption; recheck sensor availability before energy work |
| RAM/swap | MemAvailable, swap capacity and pswpin/pswpout readable | Require deltas during pilot; existing swap occupancy is not evidence of current paging |
| Vulkan timings | Enumeration succeeds | Timestamp availability, valid bits and known-workload timing remain backend-epic checks |
| HIP events/kernels | HSA enumeration and compiler version available | No kernel/event execution; gfx1101 compatibility is unproven |
| Detailed GPU counters | rocprof/rocprofv3 absent; profiler SDK package absent | No installed counter path validated; do not install or claim gfx1101 counter support |
| CPU performance counters | perf_event_paranoid=2; perf absent from PATH in planning | Counter accessibility is unverified; no permissions changed |

Card numbers and API ordinals are session-local. Repeat DRM/Vulkan/HSA identity
correlation before a workload and reject ambiguous selection. No UUID, serial,
or private machine inventory is needed in public results.

## Executable follow-up and safe short-test proposal

Private `telemetry-overhead.py` accepts explicitly selected readable sysfs files,
reads them 100 times with a monotonic clock and reports median/max read cost.
It ran against GPU-busy and VRAM-used nodes only. The private result establishes
that bounded reads execute, not that telemetry is accurate or free of observer
effects. To repeat after re-identifying the card:

```sh
python3 .private/epic1/telemetry-overhead.py "$GPU_BUSY_FILE" "$VRAM_USED_FILE"
```

The script and exact private paths are intentionally outside public tooling.
For a future collector, start with 100 ms sampling; require p95 collector-cycle
cost below 1 ms and less than 1% paired workload-time perturbation in alternating
collector-off/on trials. The current 100-read median/max check is only a
preliminary host-cost check and does not meet that workload-overhead criterion.
B1 owns that validation and cancellation instrumentation.

Proposed pilot budget for human review: isolated offline process, one sequence,
context at most 4096, at most 32 generated tokens per smoke request, one request,
60 s wall timeout with 5 s termination grace, no assistant/projector/GPU allocation
until the applicable identity and thermal checks pass. Reserve at least 2 GiB
MemAvailable and 1 GiB free VRAM including display headroom; do not reduce reserves
to accommodate a preset. Keep evidence below 100 MiB for the pilot and below the
protocol's 10 GiB campaign cap. This conservative proposal is unvalidated and
does not establish that any model configuration fits.

Before activation, B1 must implement a supervisor that checks these thresholds
before launch and during a run, checks pswpin/pswpout deltas over successive
samples, cancels on budget breach/reset/allocation failure/increasing paging,
and records failed/incomplete output. Exercise it first with synthetic telemetry
at each threshold, a sleeping child and a child ignoring termination; verify
forced kill after grace and evidence preservation. Then use an agreed idle window
for the smallest actual workload, before any historical preset reproduction.
No workload may start until a reviewer approves sensor-specific thermal limits,
display reserve, the run window and the concrete invocation. Clock/power settings,
services, drivers and the router remain unchanged.

## Unknown owners and impact

H1's inventory fields have observations or explicit access gaps. RAM timings and
bandwidth are owned by environment follow-up; they block bandwidth conclusions,
not descriptor inspection. Counter/event accuracy is owned by backend evaluation;
it blocks counter attribution, not host timing. B1 owns supervisor and observer
overhead validation; until then no benchmark is authorized. The maintainer owns
idle-window, display and thermal-budget review. H2 owns model/template/build
identity; Gate 0 cannot claim GO while those identity gaps remain unresolved.
