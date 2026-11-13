# Runtime R2: target-only Gemma 4 text execution

Date: 2026-11-13
Bead: rv-e04-398.2 (R2). Follows [R1](runtime-r1-contract.md) and
[ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md).
Status: Implemented and verified against the frozen criteria below. Gate 3 (rv-e04-398.3) makes the gate decision.

R2 runs the target model: prefill and one-token decode, with no MTP. There are two
executors with one session contract:

- `exec::Session` is a multi-threaded **CPU executor**. It is the correctness baseline
  and also provides the host expert path.
- `hip_exec::HipSession` is the **HIP executor** (feature `hip`). It runs dense work on
  the RX 7800 XT and uses **static first-N CPU expert placement**: the routed experts of
  layers `0..N` stay in host memory and run on the CPU expert path.

Both use a Q8_0 KV cache. Neither depends on the MTP assistant.

## Graph

Each step follows the pinned `src/models/gemma4.cpp` and `build_moe_ffn`.

| Step | Semantics |
| --- | --- |
| Input | `x = dequant(tok_embd[token]) * sqrt(2816)` |
| Attention input | `cur = rms_norm(x) * attn_norm` |
| Q | `wq . cur`, then per-head RMS norm times `attn_q_norm`, then NEOX RoPE |
| K | `wk . cur`, then per-head RMS norm times `attn_k_norm`, then NEOX RoPE |
| V | `wv . cur` on local layers. **Global layers have no `wv`: V is the K projection taken before the K norm.** Per-head RMS norm with no weight, and no RoPE. |
| RoPE | Local layers: base 1e4 over 256 dimensions. Global layers: base 1e6 over 512 dimensions with `rope_freqs` (proportional RoPE). The angle accumulates multiplicatively, as in ggml. |
| Attention | Scale 1.0 (Gemma 4 has no pre-attention scaling) and causal. Local layers add the standard sliding mask: a key at `p` is visible from query `q` iff `q - p < 1024`. GQA: 16 query heads over 8 (local) or 2 (global) KV heads. |
| Attention output | `attn_out = rms_norm(wo . attn) * post_attention_norm + x` |
| Shared MLP | `rms_norm(attn_out) * ffn_norm`, then `gelu(gate) * up` (tanh GELU), then `down`, then `* post_ffw_norm_1` |
| Router | `logits = gate_inp(F32) . (rms_norm(attn_out) / sqrt(2816) * gate_inp_scale)`, then softmax over 128 experts, then top 8. Weights are `p / max(sum p, 2^-14)`. |
| Experts | Input is `rms_norm(attn_out) * pre_ffw_norm_2`. Each expert computes `gate_up` (rows 0–703 are the gate, rows 704–1407 are up), then `gelu(gate) * up`, then `down`. The result is multiplied by `down_exps_scale[e]` and then by the routing weight. Experts are summed in slot order, then multiplied by `post_ffw_norm_2`. |
| Layer output | `(rms_norm(mlp + moe) * post_ffw_norm + attn_out) * layer_output_scale` |
| Head | `rms_norm * output_norm`, then the tied `tok_embd` (Q4_0), then `tanh(x / 30) * 30` |

All RMS norms sum squares in f64, as ggml does, and use eps 1e-6. Activations stay
f32 throughout. The KV cache is Q8_0, quantized as `quantize_row_q8_0_ref` does.

## Ownership and lifetimes

- **Weights** (`model::Weights`) borrow the read-only mapping: quantized tensors are
  `&[u8]` slices into it, and only the small f32 vectors are copied. A session borrows
  the weights, so the mapping outlives every session by construction.
- **All memory is allocated up front.** `Session::new` and `HipSession::new` allocate
  the whole KV cache and scratch, sized by `max_ctx` and `max_batch`. The HIP
  session's allocation also includes the device weights. Each checks the total first,
  against `max_bytes`, or against `max_vram` plus a free-VRAM reserve. Evaluation never
  allocates cache or scratch. A test asserts that the held bytes equal the precomputed
  need and do not change across evaluations or a reset.
- **Device memory** is owned by `hip::DevBuf` and freed on drop. Every kernel wrapper
  checks each range a kernel touches against its owning allocation before the FFI call.
  The C side validates scalars only.
- **Sliding layers** use a ring of `window + max_batch` slots; global layers use
  `max_ctx` slots. A batch writes its own K/V before attending, so the ring holds the
  window behind the batch's first query plus the batch itself.
- **Evaluation is atomic.** `n_past` advances only when the whole batch succeeds. A
  cancelled or failed batch may leave written slots at positions `>= n_past`, but
  those are overwritten before anything reads them. The ring sizing guarantees they
  never displace a slot that a later query can see.
- **Rewind** to `n` is allowed while the ring still holds every key that position `n`
  can see: up to `max_batch + 1` positions behind the highest write. A longer rewind
  fails and leaves the session unchanged.
- **Cancellation** is checked between layers. The HIP session synchronizes before it
  returns `Cancelled`.

## Placement and backend

- **Kernels.** The HIP kernels (`runtime/hip/rvhip.hip`) sit behind a C ABI
  (`rvhip.h`) and are built by `build.rs` with the system `hipcc`. They favour
  correctness over speed: one simple kernel per operation and no tuning.
- **FMA contraction is off**, so elementwise kernels follow the CPU's operation order.
  As a result:
  - embedding rows, KV quantization and the expert combine are bit-identical to the
    CPU executor;
  - matmuls differ only in summation order.
- **Routing runs on the host.** The GPU computes the router logits, and the host runs
  softmax, top-k and the routing-weight normalization. That is the same code the CPU
  executor runs, so expert selection is shared rather than reimplemented.
- **Static first-N placement.** For layers `0..N` the expert input is downloaded, the
  CPU expert path runs, and the result is uploaded. For the other layers, the GPU
  gathers each expert's tokens and runs one matmul per selected expert.
- **The R2 runs used N = 10.** The device memory this needs is given under
  Resources below.

## Verification

Raw captures, dumps, prompts and logs are private (`.private/r2-v1`). The committed
tests use synthetic weights only.

### Synthetic (committed, `./scripts/check.sh`)

| Test | Result |
| --- | --- |
| Batched vs stepped vs mixed-batch evaluation | Bitwise identical |
| Window boundary at the real window (1024), with a wrapping ring | With one sliding layer, changing token 0 changes logits at 1022 and 1023 and nothing at 1024, 1025 or 1029. A global layer still sees it at 1023, 1024 and 1025. |
| Cancel mid-batch, then restart and resume | `n_past` unchanged, `Cancelled` returned; the resumed logits are bitwise equal to an uncancelled run |
| Rewind | Replays bitwise; rewinding past the ring or past `n_past` fails and leaves state intact |
| Resource limits | Oversize batch, empty batch, out-of-range token, full context, `max_bytes` one byte short, `max_ctx` above the model limit: all rejected without side effects |
| Scratch reuse | Held bytes equal the precomputed need and do not change across evaluations and reset; a reset session reproduces its logits bitwise |
| Observer | Every layer's output, router logits, top-k and the final norm are emitted |
| `matmul_q4_0_multi` | Bitwise equal to separate `matmul_q4_0` calls at 1, 2 and 7 threads |

### HIP kernel conformance (GPU, `cargo test --features hip --test hip_ops -- --ignored`)

All checks are at the target's exact shapes.

| Kernel | Result |
| --- | --- |
| Q4_0 matmul: all six projection shapes, 1, 7 and 64 tokens | Relative error < 1e-5 vs CPU; each token's row is bitwise equal to its single-token result |
| Router (f32), per-head norm and RoPE (local, and global with frequency factors) at position 1021 | Relative error < 1e-5 |
| Embedding rows, Q8_0 KV store over a wrapping 1028-slot ring | Bitwise |
| Attention through the wrapped ring at positions 1026–1029 | Relative error < 1e-5; an out-of-ring key range is refused |
| GEGLU split | Relative error < 1e-6 |
| Expert combine | Bitwise |
| Out-of-allocation range | Refused before launch |

### Real artifact

**Artifact identity.** The artifact is the H2-hashed target GGUF. The pinned reference
is llama.cpp v0.9.0 on the CPU device only: no repack, f16 KV, flash attention off.

**Teacher-forced KL.** The method is that of [Gate 1](gate-1-review.md): the stored
llama-perplexity base files are scored with the tool's own per-row formulas, over
7 chunks × 255 positions. The frozen band
([divergence review](reference-divergence-review.md#updated-tolerances-for-gate-3-and-later-candidates))
is mean KLD ≤ about 0.08 and top-1 ≥ about 92%, against both references.

| Executor | Reference | Mean KLD (se) | p99.9 KLD | Same top-1 | PPL / base |
| --- | --- | --- | --- | --- | --- |
| CPU | CPU | 0.0620 (0.0052) | 2.98 | 93.4% | 11.96 / 12.03 |
| CPU | Vulkan | 0.0671 (0.0050) | 2.34 | 94.0% | 11.96 / 11.87 |
| HIP, N = 10 | CPU | 0.0664 (0.0057) | 2.77 | 92.3% | 12.19 / 12.03 |
| HIP, N = 10 | Vulkan | 0.0744 (0.0060) | 2.80 | 92.9% | 12.19 / 11.87 |
| *Vulkan vs CPU (Gate 1, for scale)* | | *0.0558* | *1.76* | *93.4%* | |

All four results are inside the band. The HIP executor's top-1 against the CPU
reference (92.3%) sits near the band's floor. Per-chunk means range from 0.024 to
0.175, and chunk 5 is the highest for every executor and reference. As the band's
definition says, falling inside it is not evidence of correctness by itself; the
op-level and layer-0 checks below carry that weight.

**Decode.** HIP stepped decode was teacher-forced one token at a time through chunks
0 and 1: 1,024 target-only decode steps, 510 of them scored. Its per-chunk SHA-256 of
the scored logits **equals** the 512-token-batch run's, so decode is bitwise identical
to batched prefill on the real artifact. Its KL is the batched result: mean 0.0422
and 93.9% top-1 over those 510 rows.

**Intermediates.** One 214-token prompt from the calibration text was dumped in full
from the reference (eval callback) and from both executors, at every layer.

| Comparison | Result |
| --- | --- |
| Executors vs reference, layer 0 | Input embedding bitwise; `attn_norm` 2e-8; Q/K/V 3–5e-3; layer-0 output **6.1e-3** relative for both executors (tolerance about 1e-2; C1's HIP build measured 7.4e-3). Expert sets agree for 97.7% of tokens. |
| Executors vs reference, all 30 layers | Divergence grows smoothly with depth as routing flips, to about 5e-2 at layer 29. The last-token top-1 matches, with KL 0.0017 for each executor. |
| HIP vs CPU executor | Layer 0 is 2.5e-5 relative. The difference grows through routing amplification alone, with no step at the placement boundary: the median MoE error is 7.06e-3 at layer 9 and 6.95e-3 at layer 10. |
| HIP with N = 10 vs HIP with N = 20 | Bitwise identical through layer 9. At layer 10 the inputs are identical and only the expert location differs: selection is identical, the expert output differs by 7.2e-7, and the layer output by 2.5e-7. **Placement does not change semantics.** |

**Window boundary on the real artifact.** HIP prefilled 1,030 calibration tokens,
which wraps the sliding ring. KL against the reference at positions 1023, 1024 and
1025 is 0.00018, 0.00008 and 0.00010, and top-1 agrees at every position from 1020
to 1029. As a sensitivity control, the same run with the window disabled is bitwise
identical to the real run through 1023 and differs from 1024 on. This control is
weak, because the token leaving the window is BOS. The synthetic test is the sharp
one.

**Controlled generation.** A text-only chat prompt (40 tokens) was continued greedily
for 520 target-only tokens by HIP and by the reference. The two agree on the first
93 tokens, then diverge into equally coherent essays with the same section
structure. Output identity is not a criterion: Gate 1 found greedy outputs are not
stable even within Vulkan.

**Resources.** The HIP session needs 9.6 GiB of device memory at a 214-token batch
and 10.0 GiB at a 512-token batch (the logits scratch grows with the batch). An
11 GiB cap left at least 2 GiB free. During these runs:

- peak junction temperature was 82 °C;
- the production router started nothing;
- VRAM returned to its 1.55 GiB baseline after every run.

`max_vram`, the VRAM reserve and an out-of-range N are each rejected before any
allocation. The CPU executor takes about 140 s per 512-token chunk; HIP takes
24–28 s per chunk and decodes at 4.1 tokens/s.

**Correction (2026-11-13).** The 4.1 tokens/s figure came from a run that
probably overlapped a CPU reference generation. Same-work reruns produce identical
tokens and decode at 13.4–14.0 tokens/s with N = 10. See the
[optimization research follow-up](optimization-research.md#s1-decode-discrepancy-environmental-not-a-methodology-difference-measured).

## Limits

- **Speed is not a goal here.** The CPU executor uses scalar-friendly Rust loops, about
  140 s per 512-token chunk on 10 threads. The HIP kernels are untuned, and decode
  runs at about 13–14 tokens/s with 10 CPU expert layers (corrected; see Resources). Specialization is later work.
- **Reference-side approximations are not reproduced.** The pinned CPU backend
  quantizes activations to Q8_0 before every Q4_0 matmul and looks GELU up in an f16
  table. R2 keeps f32 activations and the exact tanh GELU. The layer-0 gap is
  consistent with the activation quantization: Q8_0 quantization of the reference's
  layer-0 normed input alone is 7.5e-3 relative. This is consistent with the gap,
  not a proof of its cause.
- **KL coverage** is the Gate 1 calibration set: 7 chunks × 255 scored positions =
  1,785 positions from one text domain. llama-perplexity scores `n_ctx/2 .. n_ctx-2`,
  so earlier documents' "1792" counts 256 per chunk.
- **Long context** beyond about 1,030 positions is not exercised on the real artifact
  (Q2 scope).
- **The image path** is still the R1 boundary only; R2 is text-only.
