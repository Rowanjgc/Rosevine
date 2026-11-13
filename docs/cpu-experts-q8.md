# Q8_0-activation CPU experts (R3a.4)

Date: 2026-11-13. Bead: rv-e06-k0i.6 (maintainer decision D3). Parent:
[R3a.3 F32 CPU experts](cpu-experts.md). Status: **kept as an opt-in numerics
option**. It is faster beyond noise and stays inside the frozen KL band. F32
remains the default and the reference path. This is an internal same-binary
ablation, not a Gate 4 claim. Raw captures, commands and hashes stay private.

## What changed

- `runtime/src/expert_kernel.rs`:
  - New `Activation { F32, Q8_0 }` setting.
  - New `Kernel::dot_q8`, which multiplies one Q4_0 weight row by one Q8_0
    activation row. Each block's integer sum `sumi` is exact; the block then adds
    `sumi * d_w * d_x` in f32, as ggml's `ggml_vec_dot_q4_0_q8_0` does.
  - Two routines:
    - **Scalar:** ggml's generic order.
    - **AVX2+FMA:** sign/`maddubs`/`madd`, with no saturation because
      |w| ≤ 8 and |x| ≤ 128.
  - The AVX-512 kernel setting also uses the AVX2 Q8 routine. The VNNI intrinsic
    (`_mm256_dpbusd_epi32`) still fails to compile on the current toolchain
    ("intrinsic signature mismatch"), so no VNNI path exists. The routine that
    actually runs is printed (`routine=avx2-q8`).
- `runtime/src/expert_pool.rs`:
  - `set_activation` reserves `max_batch × n_experts_used` quantized rows up
    front. Evaluation still never allocates.
  - A job that needs more rows than reserved is rejected; the buffer is never grown.
  - The caller thread quantizes each activation row once per job, using the
    existing `ops::quantize_q8_0` (amax/127, f16 scale, round half away from zero).
- Sessions expose `set_expert_activation`. A HIP session without host expert
  layers accepts only F32.
- `rosevine-inspect`:
  - New flag `--expert-activation f32|q8_0` on `generate`, `replay`, `kld` and
    `dump`. The default is `f32`.
  - The `replay` config line now reports the actual activation in
    `cpu_expert_activation`. Before this change, that field was hard-coded to `f32`.

Upstream note: ggml's AVX2 quantizer rounds half to even (`_MM_ROUND_NEAREST`) and
multiplies by `127/amax`. Rosevine uses the reference quantizer's half-away-from-zero
rounding and `1/d`. Ties are rare; the difference is recorded, not hidden.

## Exact-shape tests

- Integer reference: `q8_kernels_match_exact_integer_reference` compares every
  ISA with an f64 reference over the same quantized operands. It covers
  k ∈ {32, 96, 704, 2816} with 256 rows each, including zero, subnormal and
  negative scales and extreme quants (weights −8, activations ±127). The bound is
  `4e-6 × Σ|w·x|·d`, because SIMD lanes round partial block sums. The scalar
  routine is bitwise identical across kernel settings.
- F32 comparison: the Q8 dot stays within one activation quantization step of
  the F32 dot at k = 2816. A truncated activation row panics before any load.
- Pool test: `q8_activation_matches_kernel_dot_on_every_segment` checks that
  odd segments and multi-row segments match `dot_q8` exactly. It also checks
  that a job larger than the reservation is rejected, and that the pool switches
  back to F32.

## Fidelity (correctness class N)

Same binary, HIP, first-N = 10 CPU expert layers, 12 threads, batch 64, run
under the run lock, the COMPUTE window and `gpu_campaign` supervision.

| Frozen reference | Activation | Mean KL | Top-1 | PPL (ref) |
| --- | --- | --- | --- | --- |
| CPU | F32 | 0.064073 | 92.61% | 11.94 (12.03) |
| CPU | Q8_0 | 0.070897 | 92.49% | 12.23 (12.03) |
| Vulkan fast | F32 | 0.070023 | 92.94% | 11.94 (11.87) |
| Vulkan fast | Q8_0 | 0.067854 | 93.67% | 12.23 (11.87) |

- **Band:** both Q8_0 rows pass the existing band (mean KL at most about 0.08,
  top-1 at least about 92%). No threshold was changed.
- **Reproducibility:** the F32 rows reproduce R3a.3 exactly, with the same
  logits digest. The Q8_0 logits digest is identical against both references.
- **Direction:** against the CPU reference, Q8_0 is slightly further away than
  F32 (KL +0.007). Against Vulkan, it is slightly closer (KL −0.002). It does not
  uniformly move Rosevine toward llama.cpp.

**Layer 0 and selection**, on the 214-token R2 dump prompt, against the pinned
CPU reference dump:

| Tensor | F32 vs ref | Q8_0 vs ref | Q8_0 vs F32 |
| --- | --- | --- | --- |
| `ffn_moe-0` (relative) | 1.394e-2 | 1.363e-2 | 5.65e-3 |
| `l_out-0` (relative) | 6.13e-3 | 6.01e-3 | 2.26e-3 |
| Layer-0 expert sets identical | 97.66% | 97.66% | 100% |
| Final logits KL (ref ‖ ours) | 0.0026 | 0.0010 | 0.0007 |

- **Layer-0 gap:** Q8_0 narrows the layer-0 gap to the reference by about 2%
  relative. That is far too little to explain the gap. The hypothesis that
  activation precision causes most of it is **not supported**.
- **Selection flips against F32:** none at layer 0. Matching expert sets then
  fall with depth: 97.7% at layer 1, 88.8% at layer 10 (the first GPU layer)
  and 67–74% in the last ten layers. This is routing amplification, as
  R2 measured for HIP against CPU. Against the reference, the per-layer
  matching-set curves of F32 and Q8_0 are similar (0.61–0.98 against 0.58–0.98).
- **Decode argmax:** on the parity fixture, the per-step decode argmax agrees
  between Q8_0 and F32 on 0.930 of steps, in all 7 sessions.

## Performance (paired, same binary)

7 sessions alternate F32/Q8_0 order. Each runs a P = 512 prefill and D = 128
single-token decode steps on the parity fixture, with one warmup repetition and
one measured repetition per process. Ratios are Q8_0 / F32, median with a 95%
percentile bootstrap (10,000 resamples).

| Metric | F32 | Q8_0 | Ratio (95% CI) |
| --- | --- | --- | --- |
| Decode tok/s (d512) | 22.07 | 32.82 | **1.484 (1.480–1.488)** |
| Prefill tok/s (p512) | 46.27 | 106.18 | **2.294 (2.290–2.296)** |
| CPU core-equivalents | 8.17 | 5.28 | 0.65 |
| Thermal-flag telemetry samples per run | 148–182 | 23–35 | about 0.18 |
| Peak VRAM / RSS | unchanged | unchanged | 1.00 |

Run-to-run CV is below 0.3% in both arms.

**Attribution** (R3a.1 sync mode, 3 runs per arm, d512, ms/token):

| Span | F32 | Q8_0 |
| --- | --- | --- |
| `cpu_experts` (10 layers) | 21.0 | 8.4 |
| Whole step | 47.2 | 32.8 |

- **CPU expert time:** falls to 0.84 ms per layer. That beats R3a.3's 1.5 ms/layer
  objective and nearly meets the about 0.8 ms/layer that
  [parity plan](parity-plan.md) record G3 derived for Vulkan P90 at d512.
- **Whole step:** the remaining step is dominated by GPU spans, which this
  change does not touch.

## Limits and follow-ups

- **Parity pairs not re-measured.** Using `--expert-activation q8_0` turns
  Rosevine/llama.cpp pairs into machinery comparisons. The parity cells have not
  been re-run with it, so no parity ratio is claimed here.
- **Depth not tested.** Only d512/p512 were timed. CPU expert work does not
  depend on depth, but the share it represents does.
- **Caller-thread quantization.** Rows are quantized serially on the caller
  thread. Moving this into the workers is a possible prefill follow-up.
- **No VNNI.** VNNI and AVX-512 Q8 routines remain untested ideas.
- **Thermal flags.** CPU thermal flags still occur in Q8_0 runs, though about
  5× less often.
- **Default unchanged.** F32 remains the reference and the default. Changing the
  default numerics profile is a separate maintainer decision.

## Reproduction

Build with `--features hip`, then run under the lock, the power window and
`gpu_campaign` supervision:

- `rosevine-inspect kld MODEL --base PRIVATE_REFERENCE --chunks 7 --batch 64 --backend hip --cpu-moe 10 --threads 12 --expert-activation q8_0`
- `rosevine-inspect dump MODEL --out DIR --backend hip --cpu-moe 10 --expert-activation q8_0 < TOKENS`
- `rosevine-inspect replay MODEL --backend hip --n-prompt 512 --n-decode 128 --reps 2 --ctx 768 --cpu-moe 10 --threads 12 --expert-activation q8_0 [--instr sync] < FIXTURE`
