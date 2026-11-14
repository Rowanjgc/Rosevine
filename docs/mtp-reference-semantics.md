# MTP reference semantics (S1)

Date: 2026-11-13. Bead: rv-e07-ktl.1 (S1). Status: **source audit and reference
trials complete**; Rosevine implements no MTP (S2/S3 stay deferred). All source
references are to pinned llama.cpp v0.9.0 (`cd959bae9`), unmodified. Raw logs,
commands and hashes stay private.

## Components

| Part | Reference fact | Source |
| --- | --- | --- |
| Assistant model | Separate `gemma4-assistant` GGUF: 4 layers, width 1024, 16 query heads, FFN 8192 (GELU, gated), context metadata 131,072. Layers 0–2 are sliding-window (head dim 256, window 1024); layer 3 is global (head dim 512, proportional RoPE base 1e6). | Assistant GGUF metadata; `src/models/gemma4-assistant.cpp` |
| Assistant weights | Q/O projections, Q norm, attention and FFN norms, a per-layer output scale. **No K or V projection.** Input projection `nextn_proj_pre` (2 × 2816 → 1024); output projection `nextn_proj_post` (1024 → 2816); its own 1024-wide token embedding, which also serves as the tied output head. | Same |
| Input row | `concat(target_embedding(token) × sqrt(2816), h)`. The token embedding is the **target's**, read through `ctx_other`; `h` is a 2816-wide target-space hidden row. | `gemma4-assistant.cpp` graph |
| `h` from the target | The target's post-final-norm hidden state (the LM-head input, `h_nextn`), exported per output row. | `src/models/gemma4.cpp` (`t_h_nextn`) |
| `h` from the assistant | `nextn_proj_post(final_norm(x))`, fed back as the next draft step's `h`. | `gemma4-assistant.cpp` |
| KV | **Shared, read-only.** Each assistant sliding-window layer attends to the target's KV of layer `n_layer − 2` (layer 28, the last local layer). The global layer attends to layer `n_layer − 1` (layer 29, the last global layer). The assistant writes no KV; `seq_rm` on its cache is a no-op. | `src/llama-model.cpp` (`share` lambda); `llama_kv_cache::seq_rm` (`other`) |
| Flash attention | Follows the target context's setting, because the KV is shared. | `src/llama-context.cpp` |

## State machine (one slot, greedy server path)

State per sequence: the target KV through position `p − 1`, the last committed
token `t` (not yet in the target KV), and `pending_h`, the `h` row that pairs
with `t`.

1. **Draft** (`common_speculative_impl_draft_mtp::draft`). Step 0 decodes the
   assistant on `(t, pending_h)` at position `p`. Each later step decodes the
   previous draft token with the `h` row the assistant just produced. **Every
   draft step uses the same position `p`** (the Gemma 4 shared-memory rule). The
   assistant attends to target KV cells at positions up to `p − 1`. The draft
   token is the argmax of a top-k 10 distribution (`probabilistic = false`
   default). Drafting stops at `n_max` tokens or, if `p_min > 0`, below that
   confidence; the presets set `n_max 3`, `n_min 0`, `p_min 0`. The draft cap is
   also limited by context room and remaining `n_predict`
   (`get_n_draft_max`), so **a draft can be shorter than 3**, and with no room
   the step is plain decode.
2. **Verify** (`handle_last_sampled_token`). One target batch of
   `1 + n_draft` rows `[t, d1, …, dk]` at positions `p … p + k`, all with
   logits and `h` rows. Width is 1 + draft length (2–4 with `n_max 3`), not a
   fixed 3 rows.
3. **Accept** (`common_sampler_sample_and_accept_n`). Walk the rows: sample the
   target at row `i` (greedy at temperature 0; the request sampler otherwise).
   Accept `d(i+1)` if it equals that sample; on the first mismatch keep the
   target's sample as the correction token and stop. If all `k` drafts match,
   sample one bonus token from row `k`. Committed tokens per verification are
   `accepted + 1`. An EOG match before the last row also stops acceptance.
   With temperature > 0 and the default greedy drafts, verification is still
   sample-and-match. That is exact for the target distribution. Rejection
   sampling is used only with `--spec-draft-sampling probabilistic`.
4. **Rollback** (`update_slots`, after accept). `seq_rm(p + accepted + 1, ∞)`
   removes the KV of the rejected rows. The Gemma 4 interleaved cache supports
   partial removal, so **no speculative checkpoint is taken** (checkpoints are
   only for contexts that cannot remove a tail). Sliding-window cells are
   reused only when masked for the pre-batch position, so rollback never drops
   a cell a later token still needs.
5. **Carry `h`** (`accept`). `pending_h` becomes the target `h` row at index
   `min(accepted, rows − 1)`: the row of the last kept position, which pairs
   with the new last committed token (the correction or bonus token).
6. **Prompt.** Prefill exports `h` for the last row only (masked mode). The
   first draft after prefill pairs the sampled token with that row. The
   assistant needs no catch-up decode, because it reads the target KV directly.

Checkpoint and slot-save paths store `pending_h` as the speculative state
(`get_state`/`set_state`), so a restored slot resumes drafting correctly.

## Metric definitions

- **Verification:** one target batch that contains at least one draft token.
- **Draft length** `k`: drafts proposed in that verification (0–3).
- **Accepted** `a`: drafts that matched, 0 ≤ `a` ≤ `k`.
- **Committed per verification:** `a + 1`.
- **Acceptance rate:** Σa / Σk, the server's `draft acceptance`.
- **Mean acceptance length:** 1 + Σa / verifications (the server's `mean len`).
- **Per-position acceptance:** the fraction of verifications with `a > i`.
- **Target-only decode:** the same request with speculation off.

Synthetic acceptance (`--spec-synth-rates`) replaces the match with a random
draw. It is **cost evidence only**: its output tokens are not generation
evidence.

## Reference trials

Pinned llama.cpp v0.9.0 Vulkan server, the Gate 1 fast preset without the CPU
projector, greedy driver (temperature 0, no prompt cache), trace logging for
per-verification counts. Six supervised single-shot sessions, all successful,
under the run lock and the COMPUTE window. Measured requests decode 256 tokens.

**Deviation.** The first attempt used the preset unchanged (64k context, expert
weights of the first layer on the CPU). It oversubscribed VRAM. The idle
desktop now holds about 1.8 GB against 0.8 GB at Gate 1, peak use reached
15.8 GiB, and prefill ran about 3× slower (750 against 2,300 tok/s). Those
sessions are retained as invalid and excluded. The trials use a 32,768 context
(enough for the 29,000-token workloads) and expert weights of the first **4**
layers on the CPU, leaving 1.09–1.59 GiB free with no fit warning. Absolute
rates are therefore below Gate 1's fast preset and are not comparable to it.

| Session | Policy | Requests |
| --- | --- | --- |
| `nat-mtp` | MTP, `n_max 3`, `p_min 0`, natural greedy acceptance | 8 domains at 1,900 tokens, 4 at 29,000 |
| `nat-off` | Same requests, speculation off (target-only) | Same |
| `synth-0` … `synth-3` | Synthetic acceptance forcing 0, 1, 2 or 3 accepted drafts | 2 requests at 1,900 tokens, 1 at 29,000 |

### Natural greedy acceptance

| Depth | Verifications | Accepted 0 / 1 / 2 / 3 | Per-position acceptance | Acceptance rate | Committed per verification |
| --- | --- | --- | --- | --- | --- |
| 1,900 | 641 | 91 / 91 / 74 / 385 | 0.858, 0.716, 0.601 | 0.725 | 3.17 |
| 29,000 | 304 | 33 / 36 / 29 / 206 | 0.891, 0.773, 0.678 | 0.781 | 3.34 |

Per request, committed tokens per verification range from 2.44 (prose
continuation) to 3.68 (structured JSON). Every draft has 3 tokens except at
the end of a request, where `n_predict` limits it to 2 or 1. These are
greedy-policy figures on 12 fixed requests, not a general acceptance estimate.

### Costs (ms, medians over requests)

| Depth | Target-only per token | Draft (3 assistant steps) per verification | Verification step, a = 0 / 1 / 2 / 3 forced | Natural MTP decode | Target-only decode |
| --- | --- | --- | --- | --- | --- |
| 1,900 | 13.2 | 4.0–4.2 | 30.4 / 30.9 / 31.2 / 31.5 | 106.2 tok/s | 75.6 tok/s |
| 29,000 | 15.3 | 4.0–4.2 | 38.6 / 39.1 / 39.4 / 39.7 | 84.6 tok/s | 65.5 tok/s |

- The draft time comes from the server's own speculative timers: 4,152 ms over
  987 draft calls in `nat-mtp`, and 3.98 to 4.21 ms per call in the synthetic
  sessions.
- A verification step is the server's decode time divided by verifications. It
  includes the draft, the 4-row target batch, sampling and rollback.
- Subtracting the draft leaves about 27 ms (1,900) and 35 ms (29,000) for the
  4-row target batch, 2.0–2.3× a single-token target step. That ratio is an
  inference from these medians, not a direct timer.
- The verification step barely depends on how many drafts are accepted
  (+1.1 ms from 0 to 3). Rollback is a cheap tail `seq_rm`, as the source
  says.
- End to end, MTP is 1.41× target-only at 1,900 and 1.29× at 29,000, under this
  placement.

### Correctness checks

- **Greedy equivalence.** In exact arithmetic, greedy MTP output equals
  target-only output. Here 3 of 12 requests match for all 256 tokens. The
  other 9 first differ at token 6–118. Gate 1 found that greedy output identity
  is not a usable control on Vulkan: repeated identical requests matched in only
  1–5 of 24 pairs, with or without MTP. This comparison therefore neither
  supports nor refutes equivalence. Batch-shape numerics (a 4-row verification
  against a 1-row decode) are a plausible cause, **not shown**. S2 needs
  teacher-forced logit checks instead.
- **Rosevine rollback (planned S2 contract, tested now).**
  `verification_rows_and_rollback_match_sequential_decode` checks a 4-row
  verification batch on the synthetic GPU model. Each row's logits match
  sequential decode (relative error below 1e-4). After rejecting 2 or 3 rows
  with the bounded rewind, the next 10 decode steps match sequential decode.

## Transition evidence and test plan

| Transition | Reference evidence | Correctness test for S2 |
| --- | --- | --- |
| Prefill → first draft | Source: masked `h` export of the last row; `begin` needs no catch-up for shared KV | Assistant step-0 logits against reference captures for a fixed prompt |
| Draft step (same position, `h` feedback) | Source (`draft`, the shared-memory branch); KV sharing confirmed by the server log (assistant layers 0–2 → target 28, 3 → 29) | Per-step assistant logits and `h` rows against reference captures at depths 1,900 and 29,000 |
| Verify batch `1 + k` rows | Source (`handle_last_sampled_token`); trial draft lengths 3, or 1–2 at request end | Row logits against sequential decode (Rosevine test above); width follows the draft length |
| Accept / correction / bonus | Source (`sample_and_accept_n`); forced 0–3 fixtures give exactly `a` accepted (the synthetic EOG rule aside) | Table-driven unit test over draft/target pairs, including EOG before the last row |
| Rollback | Source (partial `seq_rm`, no checkpoint for this cache); flat verify cost across forced `a` | Rosevine rewind test above, plus a sliding-window case at the window boundary |
| Carry `h` | Source (`accept`: row `min(a, rows − 1)`) | Compare the next draft's step-0 input with the reference after each forced `a` |
| Sampling (temperature > 0) | Source: greedy drafts, sample-and-match | Distribution test: committed-token frequencies equal target-only sampling (χ² over a fixed seed set) |

## Context support

| Context | Status |
| --- | --- |
| Up to 32,768 (tested here, occupied 29,000) | Supported; natural acceptance 0.78 at 29,000 |
| 65,536 and 131,072 | Ran in Gate 1 (feasibility at 128k with acceptance 0.76 at small occupied depth); acceptance at high occupied depth not measured |
| Above 131,072 (target metadata 262,144) | **Unsupported / unvalidated.** The assistant's training context is 131,072. Above it llama.cpp only warns (`n_ctx_seq > n_ctx_train`); it applies no scaling or cap. Assistant behavior there is untested. |

## Outcome

S1 acceptance:
- Every transition has source evidence, trial evidence where a trial can show
  it, and a test plan.
- Draft, verification and end-to-end costs are separated.
- Neither "max 3" nor "3-row verification" is assumed: drafts are up to 3 and
  verification is `1 + k` rows.
- Contexts above 131,072 are marked unsupported.

Not covered: sampled (temperature > 0) acceptance and correction trials, and
reference logit captures. These are S2 test-plan items.
