# Vulkan versus CPU reference divergence (rv-dse)

Date: 2026-11-12
Bead: rv-dse (blocks Gate 3, rv-e04-398.3)
Status: Diagnosed and bounded. No backend is shown to be incorrect.

## Question

[Gate 1](gate-1-review.md) found that the Vulkan reference and the CPU reference
disagree under teacher forcing. Over 1792 positions, mean KLD was 0.056 and top-1
agreement was 93.4%. Both backends are bit-deterministic on repeat (99.9% KLD
floor 0.000046). The question is whether a backend op is wrong or whether the gap
is something else.

## Evidence

All runs used the pinned v0.9.0 reference, the hashed local target and the
Gate 1 KL text (7 × 512 calibration tokens). Each session was supervised under
the Gate 1 stop rules. Raw logs and captures stay private.

### 1. Op conformance at the model's exact shapes

`test-export-graph-ops` exported the 105 unique operations of the target graph
for a 512-token batch and for single-token decode. `test-backend-ops --test-file`
then compared Vulkan0 with the CPU backend on each of them. All 105 passed
llama.cpp's per-op error thresholds.

Random data is not model data, and passing per-op thresholds does not make the
ops exact. The result does rule out a gross Vulkan op error at these shapes.

### 2. One-factor KL matrix

Each run changes one factor from a Gate 1 base. It is scored against both the
CPU base (no repack, f16 KV, flash attention off) and the Vulkan base (fast
placement, flash attention on, q8_0 KV).

| Run (one change) | vs CPU base: mean KLD / top-1 | vs Vulkan base: mean KLD / top-1 |
| --- | --- | --- |
| Gate 1: Vulkan base | 0.0558 / 93.4% | 0 (repeat floor) |
| **CPU, flash attention on** | **0.0610 / 94.0%** | 0.0657 / 93.3% |
| Vulkan, flash attention off, f16 KV | 0.0651 / 93.3% | 0.0659 / 93.3% |
| Vulkan, no cooperative-matrix matmul | 0.0647 / 92.9% | 0.0604 / 93.2% |
| Vulkan, no integer-dot (q8_1) matmul | 0.0725 / 92.4% | 0.0586 / 93.6% |
| Vulkan, no op fusion | 0.0647 / 92.4% | 0.0581 / 93.7% |
| Vulkan, CPU expert kernels executed (n-cpu-moe 8, no op offload) | 0.0722 / 92.3% | 0.0569 / 94.2% |

Some runs reproduced their base bit for bit, which shows those switches had no
effect on this path:
- `GGML_VK_DISABLE_F16`
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE`
- `GGML_VK_FA_SPARSE_DISABLE`
- all experts on the GPU
- repack versus no repack for CPU experts.

The key observation is that every effective change lands in the same band:
mean KLD 0.056–0.073 and top-1 agreement 92.3–94.2%. This holds whichever
backend made the change and whichever base it is scored against. A change made
entirely within the CPU backend (flash attention on) moves the CPU reference as
far as switching to Vulkan does.

### 3. Where the difference grows

`llama-eval-callback` captured every named tensor for a 214-token prompt. Three
runs were captured: CPU with flash attention off, CPU with flash attention on, and
Vulkan. Each tensor's 36 printed sample values were compared.

- **Layer 0.** Q, K and V projections agree to 3e-5–2e-4 relative. Attention and
  the expert projections add differences of about 3e-3–7e-3. The layer output
  differs by 2e-3 (CPU flash attention change) to 9e-3 (CPU vs Vulkan).
- **Through the network.** Residual differences grow to about 1e-1 relative by
  layers 6–16, then stay between 4e-2 and 2e-1. The final logits differ by 4e-2
  to 8e-2.
- **Same shape for all three pairs.** The CPU-only pair grows the same way as the
  CPU–Vulkan pairs.
- **Expert routing changes.**
  - Layer 0: routing is identical in all pairs.
  - From layer 1 to 5 onward: the sampled tokens' visible top-8 expert sets
    differ in 53–60 of 175 sampled token-layer positions (about 30%), for every
    pair.

This comparison is approximate. Only 6 of the 8 slots are printed per token, so a
swap between a visible and a hidden slot also counts.

## Conclusion

The Vulkan–CPU gap is not a Vulkan-specific defect we could find. Every pair of
legitimate implementations tested sits in the same band. The best-supported
mechanism is amplification inside the model:
1. Op-level rounding differences of about 1e-4 are normal for quantized
   activations and different attention kernels.
2. They flip near-tied top-8 expert choices from the first few layers onward.
3. Each flip makes a discontinuous change in the residual stream, which compounds
   with depth.

The repeat-determinism floor and this implementation band are separate
quantities. Gate 1 condition 4 had treated the band as a defect to explain.

Neither backend is established as more correct. The CPU reference has its own
documented approximations, including the FP16 GELU lookup
([native-operator-findings](native-operator-findings.md)) and Q8_0 activation
quantization.

## Updated tolerances for Gate 3 and later candidates

These replace Gate 1 condition 4.

1. **Repeat checks** stay bit-exact: 99.9% KLD 0.000046 and 100% top-1 agreement
   against the same backend and settings.
2. **Implementation band.** A candidate whose teacher-forced divergence from both
   references falls within the observed band is indistinguishable from a
   legitimate implementation difference: mean KLD at most about 0.075 and top-1
   agreement at least about 92% at 1792 positions. Divergence above the band is a
   defect signal. Divergence inside the band is not evidence of correctness,
   because routing amplification masks small op errors.
3. **Op-level checks.** Because of (2), Gate 3 must also check correctness where
   amplification has not yet happened:
   - exact-shape op conformance against the CPU backend (the
     `test-export-graph-ops` / `test-backend-ops --test-file` method above);
   - layer-0 intermediate agreement before any routing difference. The observed
     layer-0 output difference across legitimate implementations is at most about
     1e-2 relative.
4. A candidate may not claim fidelity "better than" one reference based on lower
   KLD inside the band.

## Update from C1 (2026-11-13)

A third implementation, the HIP build, gave mean KLD 0.0798 against the CPU
reference and 0.0690 against the Vulkan reference. Its layer-0 output agrees with
the CPU reference to 7.4e-3, within the tolerance above. The band's upper
estimate therefore moves to about 0.08.

HIP's layer-0 `kqv_out` dump matches Vulkan's to about 1e-3, so the token-0
anomaly comes from the GPU capture path and is not Vulkan-specific. See
[backend evaluation](backend-evaluation-c1-c2.md).

## Limits

- **KL coverage.** 1792 scored positions from one text domain.
- **Capture coverage.** One 214-token prompt, with only the printed sample values
  per tensor.
- **Capture changes execution.** The eval callback disables backend fusion and
  forces per-node synchronization.
- **Unresolved Vulkan dump.** The `kqv_out` dump for Vulkan layer 0 does not equal
  token 0's V vector, as causal attention requires. The CPU dump does. The
  downstream post-attention norm agrees with the CPU to 0.7%, so this looks like a
  capture artifact (buffer reuse) rather than a compute error, but it is
  unexplained.
- **Untested sources.** The integrated GPU (a third Vulkan device), the sampled
  decode policy and longer contexts were not tested.
- **Routing comparison.** The routing-flip rate is an approximate visible-set
  comparison, not a full top-8 capture. A full routing capture (the B2 MoE
  adapter) would turn the mechanism into a measured quantity.
- **CPU_REPACK on the full model.** A CPU_REPACK run of the whole model stalled the
  telemetry collector during its 13.5 GiB allocation and is retained as
  incomplete. Repacked CPU experts were exercised only at n-cpu-moe 8, where they
  matched the non-repacked kernels exactly.
