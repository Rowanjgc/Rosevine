# Runtime R1: artifact and text-input contracts

Date: 2026-11-13
Bead: rv-e04-398.1 (R1). Follows [Gate 2](gate-2-review.md) and
[ADR 0001](adr/0001-primary-backend-and-runtime-strategy.md).
Status: Implemented and verified. Model execution is R2 (rv-e04-398.2).

R1 is the first code of the reference runtime: a Rust crate, `runtime/`. It takes
the hashed target GGUF and the text messages, and produces exactly the token IDs
the pinned llama.cpp v0.9.0 reference produces. It does not run the model and
has no GPU code yet.

## Crate

| Module | Responsibility |
| --- | --- |
| `mmap` | Read-only private mapping (`PROT_READ`, `MAP_PRIVATE`), unmapped on drop. Records file identity (device, inode, size, mtime, ctime) and rechecks it on demand. |
| `gguf` | Bounded, zero-copy GGUF v3 parser. All sizes use checked arithmetic. Every count and length is capped before allocation. Accepts F32 and Q4_0 tensors only. |
| `contract` | The exact target tensor table (658 tensors) and the typed architecture configuration, both validated against the pinned artifact |
| `tokenizer` | The Gemma 4 BPE tokenizer, reproducing the pinned `llama-vocab.cpp` |
| `chat` | Native renderer for the text-only subset of the embedded chat template |
| `input` | The validated text-prompt boundary and the defined, not implemented, image boundary |
| `sha256` | FIPS 180-4 digest, cancellable, for identity checks |
| `rosevine-inspect` | Private verification CLI. Prompts cross stdin and stdout hex-encoded, so they never appear in argv or logs. |

**Language and dependencies:** Rust 2024 with the standard library only, no
third-party crates. There is one hand-declared FFI surface (`mmap`/`munmap`);
every `unsafe` block carries a `SAFETY` comment, which clippy enforces.
`./scripts/check.sh` runs rustfmt, clippy with warnings as errors, and the tests,
all offline and locked to the committed lockfile.

## Artifact contract

- **Container:**
  - GGUF version 3, little-endian;
  - 1 to 65,536 tensors and at most 65,536 metadata keys;
  - arrays of at most 4,194,304 elements;
  - strings of at most 16 MiB, valid UTF-8;
  - no nested arrays and no duplicate keys or tensor names;
  - alignment a power of two up to 1 MiB (default 32).
- **Tensors:**
  - rank 1–4 with no zero dimension;
  - the first dimension a multiple of the block size;
  - offsets aligned;
  - ranges must not overlap and must lie inside the file;
  - element counts below 2^62.
- **Target:** exactly the 658 tensors of the
  [shape contract](../scripts/gemma4_shapes.py) with matching shapes and types.
  The architecture is `gemma4` with these parameters:
  - 30 layers, embedding width 2816, 16 heads;
  - KV heads 8 on local layers and 2 on global layers (every sixth layer is
    global);
  - head dimensions 256 local and 512 global;
  - 128 experts with 8 used, expert width 704, dense width 2112;
  - sliding window 1024;
  - no shared-KV or per-layer input embeddings.
- **Immutability:**
  - this process cannot write the original bytes (read-only private mapping);
  - identity is checked at open and before exit;
  - a truncated mapped file would fault on access, so immutability while mapped
    is a documented precondition.

## Text-input contract

- **Rendering.** `chat::render` reproduces the GGUF's embedded template for:
  - `system`/`developer` (first message only), `user` and `assistant` roles;
  - string content;
  - assistant `reasoning_content`, emitted only after the last user turn;
  - `enable_thinking`, where absent means false;
  - `add_generation_prompt`.

  `trim` strips C-locale whitespace bytes only, matching the pinned Jinja engine.
- **Tokenization.** The steps follow the pinned `llama-vocab.cpp` for Gemma 4:
  1. Special tokens are split out longest first. Control and unknown tokens are
     matched only when special parsing is on.
  2. Spaces become U+2581, with no prefix space.
  3. Text is split into runs of newlines and runs of everything else.
  4. Rank-ordered BPE merges are applied (lowest rank first, leftmost on ties).
  5. Anything left unmatched falls back to `<0xXX>` byte tokens.
- **BOS.** llama.cpp forces BOS on for Gemma 4 even though the metadata says
  false, and the tokenizer reports that default. Chat prompts are tokenized with
  BOS off, because the rendered text already begins with `<bos>`, and with
  special parsing on, as in H2.
- **Rejected input:**
  - tools and tool messages;
  - multi-part or non-text content (image, audio, video);
  - reasoning on a non-assistant message;
  - an empty message list;
  - prompts over 262,144 tokens;
  - **content containing the literal text of a control token.** The reference
    accepts this, but tokenizing it would turn user text into turn structure.
    This is the one deliberate departure from the reference.
- **Image boundary (V1).** An image is the `<|image|>` placeholder, which the
  reference expands into slots. Each slot receives one 2816-wide projector
  embedding row. Until V1, non-text parts are rejected before tokenization.

## Verification

| Check | Result |
| --- | --- |
| Target artifact: descriptor and architecture contract | Pass: 658 tensors, 50 keys, data offset 15,824,000 |
| Full-file SHA-256 against the H2 record (14.25 GB) | Match, 60 s; repeated after all R1 runs |
| H2 render and token fixtures (embedded template; plain, system, history and reasoning; thinking omitted, true and false) | 12/12 identical rendered bytes and token IDs |
| H2 image fixtures | 3/3 rejected at the modality boundary |
| Epic 2 corpus documents against recorded native token IDs (BOS, special parsing) | 550/550 identical (104,736 tokens) |
| Edge cases against live pinned `llama-tokenize`, special parsing on and off: whitespace, newline runs, CRLF, scripts, emoji, combining marks, astral and private-use characters, literal and partial special tokens, byte-token text, long repeats | 108/108 identical |
| Synthetic malformed and boundary GGUFs | Every truncation prefix of a valid file rejected; 30+ named violations rejected with the expected error |
| Resource cleanup | The mapping appears in `/proc/self/maps` while owned and is gone after drop; replacing the file is detected |
| Cancellation | The digest checks a cancellation flag every 64 MiB (unit-tested). On the real artifact, a 2 s deadline stopped the hash at 2.22 s. |
| Cost | Parse 17 ms; parse plus contract plus tokenizer 0.09 s; 87 MB RSS without the hash. The hash pass maps all 14 GB as reclaimable page cache. |

**Bug caught by parity.** An early build dropped every merge whose left side began
with a multi-byte character (U+2581). Rendered bytes still matched, but every
fixture tokenized differently. A regression test now loads such a merge through
the real GGUF path.

Fixtures, parity harnesses and outputs are private (`.private/r1-v1`). Committed
tests use only synthetic data.

## Limits

- **Tie order.** The order of equal-length special tokens is ID-based. Upstream's
  order is unspecified (`std::sort`); this matters only if two different
  equal-length special texts overlap in one input, which no test exercises.
- **Detokenization** is not implemented (R2/S1 need it for outputs).
- **Model scope.** Only the target model is covered. The MTP assistant and the
  projector contracts are later work (S2, V1).
- **Template scope.** The configured (personal) template variant is out of scope.
  Only the artifact's embedded template is reproduced.
