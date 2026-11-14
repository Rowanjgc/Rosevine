# Static placement, CPU SIMD and memory economics (M2)

Date: 2026-11-13. Bead: rv-e06-k0i.2 (RV-06 M2). Status: measured. These are
diagnostics on one host. They make no Gate 4 claim and authorize no policy
change. Raw captures, commands and hashes stay private.

## Setup

| Item | Value |
| --- | --- |
| Host | RX 7800 XT 16 GiB, Ryzen 5 7600X (6 cores / 12 threads), Fedora 43, ROCm 6.4, COMPUTE power window |
| Workload | Parity fixture (real text), P = 512 prefill, then D = 128 single-token decode steps, Q8_0 KV, context 768. Each process runs one warmup repetition and one measured repetition. |
| Reference | llama.cpp `cd959bae`; replay adapter with batch/ubatch 512, FA on, CPU repack on, 6 threads unless noted |
| Rosevine | `65412de` (R3a.4): HIP, batch 64, 12 threads, Q8_0 CPU expert activations unless noted |
| Design | 17 cells × 3 sessions = 51 supervised single-shot processes. Cell order was rotated and reversed across sessions. All 51 succeeded. |
| Statistics | Median of 3 sessions with min–max. Run-to-run spread is below 1% except where noted. With 3 sessions there are no paired intervals, so differences under about 2% are not claimed. |

The production router's sleeping model was unloaded through the router API
before the runs (maintainer-approved), and the EngineeringMemory embedding
service was stopped. No router setting changed.

## Placement semantics, verified independently

- **llama.cpp:** `research/parity_replay.cpp` now accepts an explicit layer list
  (`L:a,b,c`). Its config row reports which layers the loader's own
  per-tensor override notices name. In every cell those notices list exactly the
  requested layers, with 3 tensors per layer.
- **Bytes:** device buffers fall by **408.4 MiB per CPU layer** in both engines.
  llama.cpp: 12,166 → 10,124 → 8,082 → 6,040 MiB for N = 5/10/15/20. Rosevine:
  `device_bytes` 12.28 → 10.14 → 8.00 GB. That equals 128 experts × 3.1904 MiB,
  so it matches the expected tensor bytes.
- **Not supported:**
  - Rosevine supports first-N placement only. Noncontiguous placement is
    unsupported there.
  - Neither engine can place individual experts statically. The experts are
    merged per layer into one tensor, and llama.cpp's override regex matches
    whole tensors. Per-expert economics remain the derived S3 result in
    [optimization research](optimization-research.md).

## Results

| Cell | Decode tok/s | Prefill tok/s | Engine TTFT p512 (ms) | Top-1 vs cell baseline | Device MiB | CPU cores |
| --- | --- | --- | --- | --- | --- | --- |
| vk first 0–9 (baseline) | 58.33 (57.5–58.4) | 811 | 631 | 1.000 | 10,124 | 2.8 |
| vk 10–19 | 58.16 | **852** | 601 | 0.961 | 10,125 | 2.8 |
| vk 20–29 | 58.61 | 809 | 633 | 0.938 | 10,125 | 2.8 |
| vk every third (0, 3, …, 27) | 57.58 | 794 | 645 | 0.953 | 10,124 | 3.3 |
| vk first10, 12 threads (SMT) | **37.34 (35.5–37.8)** | 816 | 627 | 1.000 | 10,124 | 6.4 |
| vk first10, AVX2-only CPU library | 58.08 | 814 | 629 | 1.000 | 10,124 | 2.8 |
| vk first10, CPU repack off | 58.23 | 817 | 627 | 1.000 | 10,124 | 2.8 |
| vk first5 / first15 / first20 | 77.19 / 47.07 / 39.59 | 1257 / 628 / 497 | 407 / 815 / 1031 | 0.95–0.97 | 12,166 / 8,082 / 6,040 | 2.3 / 3.3 / 3.6 |
| HIP first10 | 49.48 | 1182 | 433 | 0.953 vs vk | 10,130 | 2.2 |
| Rosevine Q8, 12 threads (baseline) | 32.69 | 106.3 | 4,816 | 1.000 | 9,670 | 5.3 |
| Rosevine Q8, 6 threads | 31.38 | 104.2 | 4,913 | 1.000 | 9,670 | 3.2 |
| Rosevine F32, AVX-512 / AVX2 | 22.13 / 21.61 | 46.3 / 44.0 | 11,053 / 11,627 | 0.930 | 9,670 | 8.2 / 8.3 |
| Rosevine Q8, first5 / first15 | 37.88 / 28.88 | 164.1 / 78.6 | 3,120 / 6,512 | 0.97 / 0.99 | 11,713 / 7,629 | 3.8 / 6.3 |

Notes:
- **TTFT** is the engine-only prefill time for 512 tokens. User-visible TTFT
  remains unsupported (see the [parity contract](parity-contract.md)).
- **Device MiB:** llama.cpp values come from its non-CPU buffer lines; Rosevine
  values come from `device_bytes`.
- **CPU cores** are process core-equivalents.
- **Top-1** is per-step decode argmax agreement against the same engine's
  baseline cell. Different placements run different layers on CPU kernels, so
  agreement of 0.94–0.97 is expected numerical drift, not a failure.
- **Memory and paging:**
  - High-water RSS is 13.45 GiB in every llama.cpp cell and 12.9–13.3 GiB for
    Rosevine.
  - Peak VRAM over the idle desktop follows device bytes.
  - Paging warnings were 0–8 samples per run, with no pattern by cell. They are
    recorded and are not a resource stop.
- **Thermal:** thermal-flag samples are 0 for llama.cpp, 9–64 for Rosevine Q8 and
  about 180 for Rosevine F32.

## Findings

1. **Which layers go to the CPU barely matters at equal budget.**
   - Decode changes by under 2% across first, middle, last and strided
     placements. Strided is about 1% slower.
   - Prefill is 5% faster with layers 10–19 on the CPU, and 2% slower strided.
   - First-N is therefore a fair baseline policy. A placement search at fixed
     budget is not worth pursuing on this model and host.
2. **The budget is what matters, and its cost is linear.**
   - Decode step time fits `step = c + 0.82 ms × N_cpu` in both engines:
     - llama.cpp Vulkan: slope 0.820 ms/layer, intercept 8.90 ms.
     - Rosevine Q8: slope 0.823 ms/layer, intercept 22.31 ms.
   - Residuals are at most 0.05 ms.
   - With R3a.4, Rosevine's marginal CPU-layer cost now **equals** llama.cpp's.
     The remaining 13.4 ms decode gap at every budget is in the GPU-side
     intercept: attention, GEMV, transfers and synchronization. It is not in CPU
     experts. This updates [parity plan](parity-plan.md) record G3.
   - Each CPU layer frees 408.4 MiB of VRAM at 0.82 ms per token. On VRAM that
     is free, the cheapest win is the smallest N that still fits the budget.
3. **SMT oversubscription hurts llama.cpp sharply.**
   - At 12 threads llama.cpp Vulkan decode drops 36% (58.3 → 37.3) and becomes
     noisy (35.5–37.8).
   - Rosevine's pool gains only 4% from 12 threads over 6 (31.4 → 32.7) and
     costs 5.3 cores instead of 3.2.
   - At equal throughput per core, **6 threads is the efficient Rosevine
     setting**. 12 threads buys 1.3 tok/s for about 2 cores.
4. **CPU ISA is not on llama.cpp's critical path here.**
   - The AVX2-only `libggml-cpu`, built privately at the pin with no `zmm` code,
     matches the native AVX-512 build within noise (58.08 against 58.33).
   - CPU repack off also matches (58.23).
   - For Rosevine F32, AVX2 is 2.3% slower in decode and 5% slower in prefill
     than AVX-512. Rosevine Q8 already uses the AVX2 routine.
5. **The cost model does not favor transfers.**
   - Measured PCIe (S1/S3): one expert layer upload takes 14.7 ms (28.4 GB/s);
     one expert takes 131 µs pinned and 150 µs pageable; a small blocking copy
     takes 18–31 µs.
   - Streaming a whole CPU layer per token (14.7 ms) costs about 18× more than
     executing it on the CPU (0.82 ms).
   - Uploading only the 8 selected experts per token and layer takes about
     1.05 ms plus synchronization. That is already slower than CPU execution
     before any slot cache, eviction or prediction cost. This is consistent with
     S3's NO-GO on per-expert transfer.
   - Execute-in-place on the CPU remains the right static policy for host
     experts.

## Limits

- Three sessions per cell give descriptive ranges, not paired confidence
  intervals. Only differences above about 2% are interpreted.
- Only P = 512 and D = 128 were tested. Depth changes the intercept, not the
  CPU-layer slope.
- Thread pinning, SMT siblings and the memory-bandwidth counters of individual
  cores were not measured. Only process core-equivalents are reported.
- MTP and batch > 1 decode were not covered (see M1, `rv-e06-k0i.1`).
- The fidelity metric is top-1 agreement within each engine. Full KL per
  placement was not run, because placement changes no numeric format.

## Reproduction

`research/parity_replay.cpp` with `CPU_MOE` given as `N` or `L:a,b,c`, and
`rosevine-inspect replay … --cpu-moe N --threads T --expert-activation q8_0|f32
--expert-kernel avx512|avx2`. Run each process as a single-shot `gpu_campaign`
command session inside the run lock and the COMPUTE window. For the reference
ISA control, use a private AVX2-only `libggml-cpu` build at the pin
(`GGML_NATIVE=OFF`, AVX/AVX2/FMA/F16C/BMI2 on) placed first on
`LD_LIBRARY_PATH`. `ldd` confirmed that it resolves.
