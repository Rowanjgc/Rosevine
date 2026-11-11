# Rosevine research charter

Planning snapshot: 2026-11-11. Status: proposed research program; no backend,
language, layout or placement decision is accepted. See the
[roadmap](research-roadmap.md) for executable Beads and decision gates.

## Mission and boundary

Determine whether specialization for Gemma 4 26B A4B QAT instruction tuning on
one RX 7800 XT can improve on a mature llama.cpp reference, under which
conditions, and why. A measured negative result is a successful research outcome.
Rosevine is independent of the production llama.cpp router and BlackroseAI.
Repository contributions retain GPL-3.0; model and dependency licenses need
separate provenance review before redistribution.

The immediate deliverable is research planning. No runtime, inference kernel,
GPU benchmark, installation or model transformation is part of this assignment.
The direct handoff authorizes research-roadmap creation and private local Beads
adoption after Project Zero. Its storage and privacy controls continue to apply.

## Reference target

Use the existing file named `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` unchanged,
with the existing F32 CPU image projector, Q8_0 K/V cache and mmap loading.
The filename is an identity label: inspection found Q4_0/F32 tensors, rather
than Q4_K tensors. Preserve the file; resolve provenance before kernel design.
The [baseline](hardware-and-inference-baseline.md) records actual descriptors.

The intended prompt contract is the canonical Gemma template and recommended
sampler. The configured template and embedded template have different hashes;
canonical equivalence remains unverified. Initially reproduce the configured
reference, then test canonical parity as a separately named control. Do not
silently substitute either template. The configured sampling values are
temperature 1.0, top-p 0.95, top-k 64, min-p 0 and repeat penalty 1.0.

The target workstation is Fedora 43 with a 16 GB RX 7800 XT and two observed
16 GB DDR5 DIMMs configured at 4800 MT/s. CPU, device enumeration and toolchain
observations are documented in the baseline. Display operation is X11; bspwm is user-reported.
Do not extrapolate from Instinct or NVIDIA measurements to this workstation.

## Independent objectives

Maintain separate measurements of decode throughput, prefill throughput, time
to first token, end-to-end latency, VRAM, system RAM, practical occupied context,
model fidelity, task performance, energy and sustained stability. Preserve
latency distributions, variance, failures and unavailable measurements.

Build Pareto sets within comparable workload and fidelity strata. Never combine
these objectives into a weighted score. A faster but lower-quality variant
belongs to an explicitly identified quality stratum. Keep 64k, 128k and 256k
capacity targets separate from occupied sequence length and from speculation.

The initial performance anchors are user-reported: 131 tokens/s near 1.9k
occupied context for the fast 64k preset and 47 tokens/s near 1.9k for the 256k
preset. They are historical observations with incomplete experimental metadata,
not independently reproduced findings or mandatory implementation targets.

## Success and stopping rules

1. Establish model/software identity, safe measurement access and reproducible
   llama.cpp controls before interpreting differences.
2. Compare HIP, Vulkan and feasible kernel frameworks using exact tensor shapes
   and complete-model trials. Unsupported candidates may be ruled out with a
   reproducible failure; a successful installation alone does not choose a backend.
3. Establish correct target-only text execution before MTP or specialized kernels.
4. For an optimization claim, pass the preregistered fidelity checks and demonstrate
   a paired improvement beyond noise. The proposed minimum useful improvement is
   5% in a named primary metric, with a paired 95% interval excluding zero;
   report every other objective and regression. This threshold needs Gate 1 review.
5. At the final gate, compare integrated configurations against matched llama.cpp
   controls across capacities and occupied lengths. Publish conclusions only after
   human privacy review. NO-GO and PIVOT retain their evidence and fallback plans.

## Operational constraints

Read model files and the router configuration only for authorized inspection.
Never modify those files, production settings, other projects, system services,
GPU drivers, global environments or global Git configuration. Do not stop or
evict operational workloads. New tests use isolated executables, outputs and
endpoints in an agreed idle window, with a cancellable resource budget.

No system installs, large model downloads, driver changes, long GPU tests,
pushes, publication, merges or remote changes are authorized by this plan.
Subsequent consequential architecture choices require a human-accepted ADR.
Private raw captures, prompts, generations and all Beads state remain outside
the public artifact set under the [privacy policy](development/privacy-policy.md).
Full model hashes and detailed profiling are future scheduled work; planning
inspection read headers without reading the complete weight payload.

## Ownership of evidence

Every result identifies its hypothesis, controlled variables, exact artifacts,
method, uncertainty and falsification condition. Distinguish local observation,
source interpretation, user report, hypothesis and untested proposal. Use the
[experiment protocol](experiment-protocol.md) and maintain the
[open questions](open-research-questions.md). No unrun experiment is marked passed.
