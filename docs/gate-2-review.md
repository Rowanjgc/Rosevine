# Gate 2 backend and runtime review

Date: 2026-11-13
Bead: rv-e03-mmf.4
Status: Reviewed
Outcome: **GO.** [ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md) accepted
Reviewer: agent evidence review; the maintainer accepted ADR 0001 on 2026-11-13.

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
| Human-accepted ADR | Met | ADR 0001 accepted by the maintainer, 2026-11-13 |

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

**GO.** The maintainer accepted ADR 0001 as proposed:
- HIP is the primary backend for the Gate 3 reference runtime;
- the host is Rust, calling HIP C++ kernels through a narrow C ABI;
- the pinned llama.cpp Vulkan reference is the performance baseline and fallback,
  with no hybrid;
- Triton is an offline kernel generator only;
- Python is for automation only.

This GO does not start R1 (rv-e04-398.1) by itself. Runtime implementation needs
its own go-ahead.

**Next discriminating actions:**
- R1/R2 toward Gate 3, once authorized;
- check that rocprofv3 works on gfx1101 before K1 (installing it needs approval);
- the parallel tracks M1, M2 and S1.
