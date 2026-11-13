# rosevine-runtime

Rust reference runtime for Rosevine ([ADR 0001](../docs/adr/0001-primary-backend-and-runtime-strategy.md)).
Current scope is R1, artifact and text-input contracts; see
[the R1 contract](../docs/runtime-r1-contract.md). It has no third-party dependencies.

```sh
cargo test --offline --locked          # also run by ../scripts/check.sh
cargo build --offline --locked --release
target/release/rosevine-inspect validate <target.gguf> [--sha256 <hex>]
```
