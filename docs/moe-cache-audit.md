# Pinned llama.cpp MoE cache audit and routing locality (M1)

Date: 2026-11-13. Bead: rv-e06-k0i.1 (RV-06 M1). Status: measured. This audits
the **existing** experimental cache at the pin and the locality of held-out
traces. It designs and implements no Rosevine cache (D1: no dynamic cache
yet). These are diagnostics, not Gate 4 claims. Raw captures stay private.

## Source audit (llama.cpp `cd959bae`)

| Claim | Source | Local execution |
| --- | --- | --- |
| Flag `--moe-cache-mib N` (env `LLAMA_ARG_MOE_CACHE_MIB`), default 0 = off, marked experimental | `common/arg.cpp`, `include/llama.h` (`moe_cache_size`) | Set through `llama_context_params.moe_cache_size` by the replay adapter |
| Only layers whose expert tensors are **all** host-resident weights are cached, on the GPU that owns the layer | `src/llama-moe-cache.cpp` (`llama_moe_cache_is_host_weight`, `dev_layer`) | Log line: "Vulkan0 MoE cache size = 2045.07 MiB for 4083.75 MiB of host experts" (10 host layers) |
| Layers with the same layout on one device share **one SLRU** and one slot bank (probation/protected halves; hits promote; experts selected in the current ubatch are never evicted) | `moe_cache_slru` | Reproduced by an independent simulator (below) |
| Slots = budget / expert bytes, minus one spare slot for MMQ overread | `alloc_size` | 2,048 MiB gives 640 slots (50% of 10 × 128 experts); 1,024 MiB gives 319 |
| Used only when `n_tokens ≤ 32` and `min(n_tokens × n_used, n_expert) ≤ slots`; larger ubatches bypass it | `get_slot_map`, `max_batch = 32` | ubatch 512 prefill: no SLRU traffic; ubatch 32: SLRU traffic logged as "ubatch > 8" |
| Large batches only **read** the cache (`copy_experts`), so prefill does not evict decode experts | `copy_experts` | "large batches: N experts copied from the cache" |
| Each cached layer does a **synchronous** readback of the selected IDs, then asynchronous uploads of misses (consecutive slots merged), then runs the experts **on the GPU** from the bank | `copy()`: `ggml_backend_tensor_get_async` + `ggml_backend_synchronize` | CPU core-equivalents fall from 2.8 to 0.8 with the cache on |
| The cache disables pipeline parallelism and is never used by the speculative (draft/MTP) context | `llama-context.cpp`, `common/speculative.cpp` (`moe_cache_size = 0`) | Not executed (single GPU; real MTP not run) |
| Counters (hits, misses, uploaded MiB) are logged only at teardown, split at ubatch ≤ 8 / > 8 | `log_stats` | Captured per process |
| Newer upstream CPU-share features | — | **Untested** (needs a separate pinned control) |

## Routing locality on held-out traces

The traces are the S2 traces: Rosevine HIP routing over the frozen-corpus prose
and code holdout (2,048 tokens each), a 560-token generated chat and the
214-token calibration prompt. They cover all 30 layers.

| Domain | Entropy per layer (bits; uniform = 7) | Unique experts per 1 / 4 / 32 tokens | Reuse distance median (p90) | Reuses within 8 tokens |
| --- | --- | --- | --- | --- |
| Prose | 6.19 | 8 / 22.9 / 68.9 | 4 (30) | 70% |
| Code | 6.05 | 8 / 22.4 / 61.1 | 3 (26) | 74% |
| Chat (generated) | 5.57 | 8 / 19.4 / 51.4 | 2 (19) | 80% |
| Calibration | 5.96 | 8 / 21.7 / 65.3 | 3 (22) | 75% |

- **Usage spread:** usage is spread widely, at 0.6–1.4 bits below uniform.
- **Reuse:** short-range reuse is strong. Chat is the most local of the four
  domains.
- **No fitted parameters:** an LRU cache fits nothing to the data, so no
  train/holdout leakage is possible here.
- **Static frequency plans:** a frequency plan fitted on prose generalizes poorly
  to other domains. That is the S3 result in
  [optimization research](optimization-research.md).

## Independent recomputation

A Python SLRU simulator mirrors `moe_cache_slru`: shared slots, probation and
protected segments, sorted miss fills, and the current-selection guard. It ran
on these traces for layers 0–9 and its predictions were then compared with the
live counters. The live run uses llama.cpp's own routing on the parity
fixture, which is a different text, so agreement means the same range rather
than equal values.

| Setting | Simulated decode hit rate, by domain | Live hit rate |
| --- | --- | --- |
| 1,024 MiB, 1 token per call | 0.67–0.76 | 0.681 |
| 2,048 MiB, 1 token | 0.85–0.94 | 0.853 |
| 2,048 MiB, 4 tokens | 0.79–0.91 | 0.794 |

- **Bytes:** live byte accounting is exact. Misses × 3.1904 MiB equals the logged
  upload in every cell; for example, 3,011 × 3.1904 = 9,606 MiB.
- **Lookups:** hits plus misses equal reps × tokens × 10 layers × 8 for k = 1.
- **Determinism:** counters were identical in all three sessions of every cell.

## Live cache on/off (Vulkan unless noted, layers 0–9 on the CPU, 6 threads, P = 512, D = 128)

The design is 13 cells × 3 rotated sessions, 39 supervised processes, all
successful. "Warm" is the second repetition in the same process; the cache is
not cleared by `llama_memory_clear`.

| Cell | Decode tok/s cold → warm | Prefill tok/s cold → warm | Hit rate (uploaded) | VRAM over idle | CPU cores |
| --- | --- | --- | --- | --- | --- |
| Cache off | 58.2 → 58.6 | 761 → 814 | — | 9.93 GiB | 2.8 |
| 1,024 MiB | 52.7 → **53.7 (−8%)** | 761 → 1,160 | 68.1% (20.8 GiB / 256 tokens) | 10.93 | 0.8 |
| 2,048 MiB | 68.0 → **72.7 (+24%)** | 763 → 1,441 | 85.3% (9.6 GiB) | 11.93 | 0.8 |
| **Equal memory: layers 0–4 on the CPU, no cache** | 77.4 → **77.3** | 1,134 → 1,252 | — | 11.93 | 2.2 |
| 2,048 MiB, 1 repetition (cold only) | 67.9 | 758 | 83.4% | 11.93 | 0.8 |
| 4-token verification chunks, cache off | 107.8 → 107.8 | — | — | 9.93 | 2.2 |
| 4-token chunks, 2,048 MiB | 118.4 → **134.2 (+24%)** | — | 79.4% | 11.93 | 0.9 |
| ubatch 32, cache off | 58.4 | 135 | — | 9.66 | 2.0 |
| ubatch 32, 2,048 MiB | 73.0 | 196 | decode 87.4%; prefill 46% (35.6 GiB) | 11.43 | 0.8 |
| P = 2048, cache off | 54.9 → 55.1 | 785 → 801 | — | 10.12 | 2.2 |
| P = 2048, 2,048 MiB | 73.4 → **80.6 (+46%)** | 786 → 1,364 | 93.3% | 12.12 | 0.7 |
| HIP, cache off | 49.1 → 49.6 | 859 → 1,183 | — | 10.39 | 2.2 |
| HIP, 2,048 MiB | 58.8 → **61.4 (+24%)** | 860 → 1,342 | 85.2% | 12.38 | 1.2 |

Fidelity: top-1 agreement with the cache-off cell is 0.93–0.95. With the cache
on, host experts run on GPU kernels instead of CPU kernels, which is the same
kind of numerical drift as a placement change.

## Findings

1. **At equal VRAM, static placement beats the cache for decode.**
   - The 2 GiB cache uses exactly the VRAM of 5 more GPU-resident layers.
   - It decodes 6% slower than layers 0–4 on the CPU with no cache (72.7
     against 77.3 tok/s), even fully warm.
   - Hit-driven uploads (37.5 MiB per token, about 1.3 ms at the measured
     28 GB/s) plus 10 synchronous ID readbacks per token cost more than they save.
2. **An undersized cache is a net loss.**
   - At 1 GiB the hit rate is 68%, the upload is about 81 MiB per token, and
     decode falls 8% below cache-off.
   - High hit rates are necessary but not sufficient, as the bead warned.
3. **Where the cache does help:**
   - **Warm prefill**, through read-only copies: 1,441 tok/s against 1,252
     for the equal-memory static layout.
   - **Locality-rich, longer contexts:** at P = 2048 the hit rate is 93% and
     decode is +46% over cache-off.
   - **MTP-shaped 4-token batches:** +24% over cache-off, at a lower hit rate
     (79%) because 4 tokens select about 22 unique experts per layer.
   - **Cold behaviour:** the cache warms within one 128-token decode (83% cold
     against 87% in the second repetition). Cold prefill gains nothing.
4. **Prefill does not pollute the cache.**
   - With 512-token ubatches, prefill bypasses the SLRU by design.
   - With 32-token ubatches it does fill the cache (46% hit rate,
     35.6 GiB uploaded), but the decode hit rate afterwards is *higher*
     (87.4%), not lower.
   - Small-ubatch prefill is itself slow (135–196 tok/s), so that
     configuration is not useful anyway.
5. **CPU relief:** with the cache on, host experts run on the GPU, and CPU use
   falls from 2.8 to 0.8 core-equivalents.

For Rosevine, which executes host experts on the CPU at the same 0.82 ms per
layer as llama.cpp (see [placement economics](placement-economics.md)), the
measured cache does not justify a dynamic cache for decode at equal memory.
`rv-e06-k0i.3` (cache benefit bounds and overlap) should start from this
baseline: a static layout with N reduced by the cache's VRAM, plus the
prefill-only benefit.

## Limits

- **MTP not run.** Real MTP with the assistant model was not run. The 4-token
  chunks are fixed tokens, all accepted, and measure only the batch-shape
  effect. The pin never uses the cache in the draft context.
- **Statistics.** Three sessions give ranges, not paired intervals. Cache
  counters are deterministic.
- **Not covered.** Only P = 512 and P = 2048 were tested, with 128 decode tokens
  and one fixture. Domain switching within a session and the newer upstream
  control were not tested.

## Reproduction

`research/parity_replay.cpp` takes two optional final arguments,
`MOE_CACHE_MIB DECODE_CHUNK`. It passes through the cache's sizing and teardown
counter lines on stderr. A `DECODE_CHUNK` of k evaluates k fixed tokens per
call with logits at every position. `decode_ns` attributes each chunk's time
evenly to its tokens, and `chunk_ns` keeps the raw spans. Run each process as a
single-shot `gpu_campaign` session inside the run lock and the COMPUTE window.
