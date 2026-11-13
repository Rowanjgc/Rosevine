# Gate 1 baseline and fidelity review

Date: 2026-11-12
Bead: rv-e02-22u.4
Status: Reviewed
Outcome: GO (restricted) for candidate performance comparisons against the
Vulkan llama.cpp reference, with the conditions below
Reviewer: agent evidence review; run window, GPU envelope and stop rules approved
by the maintainer on 2026-11-12. No ADR, backend, runtime or representation is adopted.

## Scope

[Gate 0](gate-0-review.md) was updated to a restricted GO with this campaign.
Its tooling is [gpu_campaign.py](../scripts/gpu_campaign.py). Each session is a
private `llama-server` on a UNIX socket with token-exact greedy requests and
supervised telemetry. The production router stays untouched. The maintainer chose
this minimal Gate 1 scope:

- the four historical presets, three independent server sessions each;
- one-factor controls around the fast 64k preset;
- a teacher-forced KL correctness control.

Interaction cells, the 100-task and NLL quality evaluation, and MTP sweeps are
deferred (see the end of this record).

Every session ran 2 warmups and 24 measured requests. The measured requests were
the eight frozen fixtures at 1900 occupied tokens (twice each) and at 29000 tokens
(once each), all with a 512-output limit in seeded shuffled order. Each run
used the RX 7800 XT through Vulkan, the pinned v0.9.0 installed reference and the
hashed local target, MTP assistant and projector. The host CPU is a 6-core Zen 4
desktop part.

## Results

Decode and prefill rates are server-reported tokens/s. The table gives the median
of per-session medians. "Spread" is (max − min) / median across sessions at 1900
tokens. "Shader clock" is the range of per-request median GPU clocks during
1900-token requests.

| Configuration | Sessions | Decode 1.9k | Spread | Decode 29k | Prefill 1.9k | Prefill 29k | MTP accept | Min free VRAM | Shader clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast preset (64k, n-cpu-moe 1, MTP 3) | 3 | 142.7 | 5.0% | 106.4 | 2446 | 1918 | 0.76 | 0.27 GiB | 2326–2434 MHz |
| Default preset (64k, n-cpu-moe 2, MTP 3) | 3 | 101.0 | 35.7% | 86.7 | 2163 | 1899 | 0.79 | 0.22 GiB | 1524–1681 MHz |
| 128k preset (n-cpu-moe 3) | 3 | 79.8 | 1.6% | 68.8 | 2121 | 1884 | — | 0.76 GiB | 2380–2536 MHz |
| 256k preset (n-cpu-moe 8) | 3 | 46.3 | 17.5% | 40.5 | 1583 | 1807 | — | 1.11 GiB | 1276–1772 MHz |
| Fast, MTP off | 1 | 94.6 | — | 79.9 | 2368 | 1907 | — | 0.35 GiB | 2580–2610 MHz |
| 64k, n-cpu-moe 3, MTP | 1 | 80.9 | — | 68.3 | 2030 | 1883 | 0.76 | 0.43 GiB | 1263–1285 MHz |
| 64k, n-cpu-moe 8, MTP | 1 | 64.0 | — | 49.6 | 1559 | 1805 | 0.74 | 2.50 GiB | 1279–1306 MHz |
| 64k, n-cpu-moe 8, MTP off | 1 | 46.2 | — | 40.3 | 1565 | 1804 | — | 2.91 GiB | 1278–1282 MHz |
| 128k, n-cpu-moe 8, MTP off | 1 | 46.3 | — | 40.3 | 1582 | 1805 | — | 2.12 GiB | 1278–1282 MHz |
| 128k, n-cpu-moe 1, MTP (feasibility) | 1 | 104.9 | — | 84.9 | 741 | 515 | 0.76 | 0.01 GiB | 2414–2442 MHz |
| Fast, text-only (projector omitted) | 1 | 108.5 | — | 87.4 | 800 | 719 | 0.76 | 0.32 GiB | 2411–2439 MHz |

All 19 completed sessions returned 26/26 requests. No request crossed a thermal
data-quality flag. MTP acceptance is for this greedy policy; it is not the
historical temperature-1.0 policy.

One-factor findings at fixed capacity:

- MTP: about +51% decode at fast placement (94.6 → 142.7).
- CPU-placed MoE layers: each one costs substantially (MTP on: 142.7 → 101.0 →
  80.9 → 64.0 for 1, 2, 3 and 8 layers). Part of the 2- and 3-layer cost is the
  clock-state effect below.
- Capacity at fixed placement costs nothing measurable (n-cpu-moe 8, MTP off:
  46.2 at 64k, 46.3 at 128k, 46.3 for the 256k preset).
- Occupied length: decode at 29000 tokens is 13–25% below 1900 tokens; prefill
  rate is nearly flat.

Historical anchors are reproduced in direction and roughly in size. The fast
preset gave 142.7 vs 131 tokens/s and the 256k preset 46.3 vs 47. The fast-preset
difference is not attributed: greedy MTP acceptance and the request policy differ
from the historical requests.

## Repeatability envelope

Per-request decode rates are stable to about ±1% while the GPU stays in one clock
state. Four default-preset sessions gave 100.3, 100.5, 100.7 and 100.2; six
n-cpu-moe-8 sessions gave 45.0 / 44.8 / 45.0 pinned and 46.3 / 46.3 / 46.3 unpinned.

**Bimodal GPU clock state (root cause identified).** Configurations that keep
the GPU only partly busy (about 70%) settle into one of two sticky shader-clock
levels under amdgpu's automatic power management. Within one 256k session, with
identical server and prompts, the rate was 46.3 tokens/s at a 1280 MHz median
clock and 54.4 at 1772 MHz. This explains the 17.5% (256k) and 35.7% (default)
session spreads. GPU-bound configurations (fast, 128k) sit near 2400 MHz and
repeat within 1.6–5%. ABBA probes ruled out two alternatives:

- 100 ms clock sampling: off 100.3/100.2, on 100.5/100.7 tokens/s.
- CPU pinning to physical cores: pinned 45.0/44.8/45.0, free 46.3 ×3.

The GPU uses the `auto` performance level with the boot-default power profile.
Changing it is a power-setting change outside agent authority.

**Desktop VRAM headroom.** Other clients' VRAM grew from 0.76 to 1.31 GiB partway
through the campaign. Two later controls then show collapsed prefill (741–800 vs
about 2400 tokens/s) consistent with driver spill: the text-only control and
128k at n-cpu-moe 1. The text-only result is therefore contaminated and is **not**
evidence about projector cost. The 128k / n-cpu-moe 1 cell does not fit without
spill (minimum free VRAM 0.01 GiB). The fast and default presets themselves run
with only 0.22–0.27 GiB free.

**Collector effect.** Three ABBA pairs (24 matched requests) put the effect of
100 ms versus 1 s collection at +0.07% on decode rate (95% CI −0.14% to +0.13%).

## Correctness controls

**Greedy output identity is not a usable control on Vulkan.** Repeated greedy
requests matched in only 1–5 of 24 pairs across sessions, and 1–2 of 8 within a
session. Outputs first diverged between tokens 9 and 510, with or without MTP.

**Teacher-forced KL is usable.** `llama-perplexity --kl-divergence` ran over 1792
scored positions (7 × 512 tokens of calibration-split text). The plan called for
4096 positions; the calibration split holds only 4082 tokens, and the holdout
split was not used.

| Comparison | Mean KLD | 99.9% KLD | Same top token | PPL ratio (Q/base) |
| --- | --- | --- | --- | --- |
| CPU reference repeat vs CPU reference | 0.000000 | 0.000046 | 100.0% | — |
| Vulkan fast placement repeat vs Vulkan | 0.000000 | 0.000046 | 100.0% | — |
| Vulkan fast, q8_0 KV, vs CPU (two runs, identical) | 0.0558 | 1.76 | 93.4% | 1.007 ± 0.013 |
| Vulkan fast, f16 KV, vs CPU | 0.0569 | 2.57 | 93.5% | 1.021 ± 0.014 |
| Vulkan n-cpu-moe 8 vs CPU | identical to fast | | | |

What this shows:

- Both references are deterministic under teacher forcing.
- Vulkan and the CPU reference (`--no-repack`, f16 KV, flash attention off)
  **disagree materially**, and KV precision does not explain it. Neither side is
  established as correct; Vulkan's perplexity is slightly lower.
- With 512-token batches, CPU-resident expert weights are computed on the GPU, so
  this control does not exercise CPU expert kernels. The single-token decode path
  that placement changes affect is untested by KL.

## Metric inventory

| Metric | Status |
| --- | --- |
| Prefill tokens/s | Measured (server `prompt_ms`; no prompt cache: `cache_prompt` false and `prompt_n` verified) |
| Decode tokens/s | Measured. Speculative decode when MTP is on; target-only when off |
| MTP proposed/accepted | Measured as aggregate per request. Per-position acceptance and draft/verify time are null: not exposed by the server |
| End-to-end latency | Measured (client monotonic round trip over the UNIX socket) |
| TTFT, steady output rate | Null: non-streaming requests deliver no per-token timestamps |
| VRAM | Sampled at 100 ms: idle (preflight), minimum free and peak used. Per-process allocation is null: not exposed |
| System RAM | System MemAvailable and swap in/out measured. Server-process RSS/PSS is null: the collector samples the supervised driver, not its child |
| Utilization | GPU busy measured. CPU core-time is null: not collected |
| Clocks and temperatures | GPU shader/memory clocks and five temperature sensors per request |
| PCIe traffic, power/energy | Null: no validated counters; board power is not sampled |
| Load time | Measured separately from request timing (5–8 s with a warm page cache) |

## Decision

GO (restricted). Rerunning the same manifest repeats within a documented envelope
(about ±1% within a clock state), correctness controls exist and are usable, and
every missing metric is labelled. Conditions on later candidate comparisons:

1. **Clock state.** Record per-request GPU clocks. Compare within one clock state,
   or under a fixed power profile set by the maintainer. A difference smaller than
   the 17–36% clock-state spread is not evidence when states differ.
2. **VRAM headroom.** Record idle desktop VRAM and minimum free VRAM. Flag any
   comparison where either configuration runs below about 0.3 GiB free or where
   desktop VRAM changed between arms.
3. **Usefulness threshold.** Keep the 5% median-benefit threshold with a paired
   95% interval above zero, applied within a clock-state stratum.
4. **Correctness.** Compare candidates against the Vulkan reference by
   teacher-forced KL; its noise floor is 99.9% KLD 0.000046 with 100% top-1
   agreement. A candidate must report its divergence from both Vulkan and the CPU
   reference. No candidate may claim fidelity better than the unresolved
   Vulkan–CPU gap until that gap is explained.
5. Greedy output identity is not a correctness criterion on Vulkan.

## Open items and deferrals

- **Vulkan–CPU logit divergence** (KLD 0.056, 6.5% top-1 disagreement): diagnose
  with the independent scalar oracles before Gate 3 reference-execution claims.
- **GPU power profile:** a maintainer decision on a fixed compute profile for
  benchmark windows would remove the clock-state bimodality.
- **Deferred from the minimal scope:** interaction cells; the 100-task and NLL
  quality evaluation on the holdout split; MTP draft-length sweeps; the sampled
  (temperature 1.0) policy; full 4096-position and per-layer intermediate captures;
  CPU-expert decode-path correctness.
- **Thermal note:** the earlier 95 °C CPU stop was an error on the agent's part.
  This Zen 4 CPU regulates to 95 °C under sustained multi-core load, so the stop
  is now 100 °C with a 95 °C flag
  ([rule audit](development/rule-audit.md#gpu-baseline-revision-2026-11-12)).

All attempts, including stopped sessions, are retained privately with their raw
telemetry, request records and server logs. No performance claim beyond this
local reference configuration is made.
