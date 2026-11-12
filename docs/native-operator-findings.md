# Native CPU synthetic operator findings

Date: 2026-11-11. Status: investigation evidence, not a correctness GO.
The [proposed correctness plan](reference-correctness-plan.md) and its existing
mathematical FP32 criterion remain unchanged. No model weights were evaluated.

## Scope and identities

[correctness_oracle.py](../scripts/correctness_oracle.py) now defines independent
binary64 RMS normalization, tanh GELU, softmax, one-head masked attention and a
small gated FFN in addition to the scalar dequantization/matvec/error helpers.
The [synthetic fixture definitions](../research/fixtures/scalar-operators-v1.json)
contain inputs, matrix-row conventions, epsilon, masks and the analytic attention
answer. These are mathematical references, not FP32 rounding emulation.

A private one-thread helper built tiny native GGML graphs against the installed
CPU/base libraries and exact-revision headers. It uses a 32 MiB owned context,
checks compute status/finite output and frees the context after output capture.
The source revision is `cd959bae99ff4bed22bef2604f45241c09bbb8c8`; the selected
native library hashes match Epic 1's resolved-library record. Compiler flags,
source/binary/library hashes, capture outputs and comparisons stay private under
`.private/epic2-operator-reference-v1`, `v2` and `v3`. The owner-executable helper
artifacts stay under ignored `.tools`; captured data/source files are 0600 in
0700 directories. No global package/configuration changed. This validation helper
does not choose C++ as an inference runtime or approve a backend ADR.

## Capture materialization failure and correction

The first helper expanded saved tensors into the graph but did not mark every
saved tensor as an output. Its RMSNorm intermediate remained unmaterialized,
while the terminal weighted RMSNorm was computed. Pinned source's CPU
`RMS_NORM + MUL` fusion and `ggml_can_fuse_subgraph_ext` explain the behavior:
intermediate tensors can be skipped when not protected as requested outputs.
A pointer and plausible shape did not prove a tensor was populated.

The second helper marks each saved tensor with `ggml_set_output`. RMSNorm then
matches the independent mathematical reference within proposed criterion A,
and the terminal weighted result agrees with the original fused computation on
this fixture. The failed first capture is retained. Output marking can alter
fusion/performance, so future all-layer captures must record instrumentation
and compare appropriate terminal outputs with the uninstrumented reference.
Do not use instrumented timing as an unchanged baseline without validation.

## GELU precision difference retained as a failure

With corrected materialization, the mathematical FP32 check still fails for
GELU and its dependent gated-FFN outputs. The pinned CPU `vec.h` enables
`GGML_GELU_FP16`; for inputs between -10 and 10 it rounds the input to FP16 and
uses an FP16 GELU table value, then returns FP32. Outside that range it takes
explicit zero/identity branches. FP32 tensor storage therefore does not establish
mathematical FP32 evaluation of this activation.

A separate diagnostic using independently rounded FP16 input/output around the
tanh GELU formula matches the selected native GELU/FFN fixtures. This diagnostic
does not replace the failed mathematical check. The third helper also exercises
`ggml_geglu_split`, used by the target's shared-FFN graph helper; on the synthetic
fixture its output matches the decomposed native GELU product and has the same
mathematical-reference mismatch. Source evidence and these finite fixtures do
not establish behavior of every possible value or a complete model graph.

That precision diagnostic has since been made explicit as
`gelu_cpu_fp16_lookup` in [correctness_oracle.py](../scripts/correctness_oracle.py).
It reconstructs the pinned CPU table-generation expression with FP32 operations,
the source compiler's fused multiply-add for the inner polynomial, `tanhf`, and
FP16 input/output conversion. Compared with the installed CPU library's initialized
`ggml_table_gelu_f16`, it produced bitwise-identical outputs for all 63,488 finite
binary16 inputs expanded to FP32, including the source's +/-10 shortcut behavior.
The pinned revision and installed-library hash are retained in the private audit
`.private/epic2-gelu-f16-exhaustive-v1/result.json`. Nonfinite inputs remain
rejected by the independent oracle.

This exhaustively validates the lookup reconstruction against that installed
CPU table, not all possible FP32 inputs, other CPUs/build flags, model outputs,
or a quality tolerance. The original mathematical-FP32 criterion still fails on
the selected GELU fixtures, and no alternate acceptance threshold or precision
exception has been approved.

Across 15 saved outputs (44 elements), the corrected CPU fixture meets criterion
A for RMSNorm/weighted/zero norms, softmax, gate/up projections and the analytic
attention cases. Five GELU-dependent outputs fail it. No threshold was loosened,
no FP32-math success is asserted, and no mixed-precision exception was approved.
The small empirical set is not a calibrated operator envelope.

## Consequences and next discriminating checks

The reference contract must name activation precision/lookup behavior as well as
input/output dtypes. A review can define separate precision-specific references
before candidate testing; it cannot silently reclassify this failure as a pass.
Determine applicable CPU/other-backend behavior and measure downstream logits,
routing and quality before claiming placement equivalence. The installed CPU
observation alone does not establish GPU activation behavior or a production
model defect. Full graph/intermediate/logit captures remain unrun.

Before freezing Gate 1 tolerances, retain both the independent mathematical and
immutable native decoding references, establish active-node materialization and
instrumentation effects, and explicitly resolve this precision stratum. Synthetic
unit checks and this bounded native CPU experiment do not lift Gate 0's NO-GO for
performance experiments or approve a model run envelope.

## Target-shaped routed Q4_0 probe

On 2026-11-12, the private target descriptor was inspected for expert weight
dimensions. Its routed down-expert tensor is Q4_0 with GGML axes
`[704, 2816, 128]`. A model-free `ggml_mul_mat_id` probe then used those exact
first two axes, four deterministic synthetic experts, activation width 704,
selected expert IDs 0 and 3, one activation row and one CPU thread. The probe
allocated the same logical tensor axes in ordinary CPU and CPU_REPACK buffers;
the repacked tensor had initialized CPU repack traits (`q4_0_8x8`). All 5,632
outputs were bitwise identical between the two buffers.

The first attempt mapped the matrix axes backwards and failed GGML's shape
assertion before computation. The corrected probe uses the descriptor's stored
axis order and passes the runtime shape contract. This correction and the
private source/build/outputs are retained under
`.private/epic2-q4-target-shape-v1`; the executable is under ignored `.tools`.
The generated weights and activations are synthetic. Four experts stand in for
the descriptor's 128; only one row is evaluated; this does not include the
model's real weights, router, upstream activations, batch, all routed operators,
or accumulated layer effects. It therefore rules out only a simple shape-triggered
ordinary-versus-repack difference for this specific deterministic routed
down-projection. It does not explain or invalidate the reproducible model-level
extra-buffer discrepancy, identify which model path is correct, establish a
general CPU correctness result, or alter any numerical criterion.

The remaining useful discriminator is direct intermediate comparison in the
real model under matched execution settings (including pre/post routed
projection tensors, expert IDs and weights), or a synthetic probe populated
from explicitly authorized model tensor blocks and matched activation rows.
Neither is covered by this result. B2 remains in progress.
