# rosevine-runtime

Rust reference runtime for Rosevine ([ADR 0001](../docs/adr/0001-primary-backend-and-runtime-strategy.md)).
It has no third-party dependencies.

- **R1:** artifact and text-input contracts. See [the R1 contract](../docs/runtime-r1-contract.md).
- **R2:** target-only Gemma 4 execution. There is a CPU executor (`exec`) and a HIP executor
  (`hip_exec`, behind the `hip` feature) with static first-N CPU expert placement and a Q8_0 KV
  cache. See [R2 execution](../docs/runtime-r2-execution.md).
- **R3a.3:** persistent, parked CPU expert workers with runtime-selected AVX2/AVX-512
  F32-activation kernels and the scalar reference retained. See
  [implementation, numerical checks and measurements](../docs/cpu-experts.md).

```sh
cargo test --offline --locked          # also run by ../scripts/check.sh
cargo build --offline --locked --release [--features hip]
target/release/rosevine-inspect validate <target.gguf> [--sha256 <hex>]
target/release/rosevine-inspect generate <target.gguf> --n 64 [--backend hip --cpu-moe 10]  # stdin: token ids
cargo test --offline --features hip --test hip_ops -- --ignored   # GPU kernel conformance (needs a GPU)
```

The `hip` feature needs ROCm's `hipcc`. It targets gfx1101 by default; set `ROSEVINE_HIP_ARCH`
to override.

`generate`, `kld` and `dump` accept `--expert-kernel auto|scalar|avx2|avx512`
and `--expert-wait sleep|hybrid|spin`. Defaults are auto and sleep. Hybrid polls
for at most 20 microseconds, spin for at most 100 microseconds, then workers park.
AVX-512 requires Rust 1.89+ at build time; older supported compilers retain AVX2/scalar.
