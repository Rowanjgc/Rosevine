# Gate 2 backend and runtime review

Date: 2026-11-13
Bead: rv-e03-mmf.4
Status: Evidence complete. ADR awaiting human review.
Outcome: pending human acceptance of [ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md)
Reviewer: agent evidence review. The ADR decision is the maintainer's.

## GO condition check

From the [roadmap](research-roadmap.md#decision-gates), Gate 2 GO requires correct
viable backend coverage, paired shape and full-model evidence, and a
human-accepted runtime ADR.

| Requirement | State | Evidence |
| --- | --- | --- |
| Correct viable backend coverage | Met for HIP and Vulkan | HIP: 103/105 exact model shapes, and the 2 harness-faulted ops pass llama.cpp's standard suites; Vulkan: 105/105; both within the KL band and layer-0 tolerance ([C1/C2](backend-evaluation-c1-c2.md), [C3](backend-evaluation-c3.md)) |
| Paired exact-shape evidence | Met | Per-op HIP and Vulkan timings at model shapes (C1/C2); Triton at 7 shapes (C3) |
| Paired full-model evidence: prefill, target-only, MTP, MoE | Met | 8 ABBA sessions per backend for the phase matrix; 8 per backend for MTP; MoE op timings (C1/C2) |
| Compilation, tooling and complexity | Met | Registers and spills, startup, subgroup test (C1/C2); Triton compile and ISA, and C++/Rust/Python FFI (C3) |
| Unsupported or failed cells | Recorded | No hardware profiler on either backend; rocBLAS not timed separately; one placement; Vulkan host-language cost not measured |
| Human-accepted ADR | **Pending** | ADR 0001 is Proposed |

## Evidence trace

Every row of the ADR's evidence table cites C1, C2 or C3. All runs used:
- the Gate 1 protocol and supervisor;
- the COMPUTE power profile (rv-19d);
- pinned llama.cpp v0.9.0 for both backends;
- the hashed local target.

The shared reference code is the unmodified pinned llama.cpp source. Fixtures for
the Triton and FFI work are synthetic Q4_0 data checked against float64
references.

## Outcome

To be recorded after review: GO with ADR 0001 accepted (and its host-language
choice), or NO-GO/PIVOT with the reason. A GO does not by itself start R1
(rv-e04-398.1).
