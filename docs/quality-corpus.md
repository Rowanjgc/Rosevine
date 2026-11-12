# Candidate quality fixtures and held-out corpus

Version: v1 scoped diagnostic fixture, reviewed 2026-11-12. Inputs and splits are
frozen for narrow, descriptive reference diagnostics; they are not approved as a
broad quality standard. No model NLL, task accuracy or noninferiority result has
been measured. The
[experiment protocol](experiment-protocol.md) and
[reference capture plan](reference-correctness-plan.md) remain authoritative about
proposed limits and the distinction between fixture construction and execution.

## Original scored tasks

[quality_tasks.py](../scripts/quality_tasks.py) defines 20 calibration and 100
holdout tasks, with five equally represented categories: arithmetic, structured
transformation, algorithm tracing, text instructions and repeated-key retrieval.
The [public recipe](../research/fixtures/quality-tasks-v1.json) freezes seeds,
canonical split hashes, generator, license and scoring. All task definitions are
original Rosevine synthetic fixtures under GPL-3.0-only. Calibration/holdout exact
prompts and IDs are disjoint; task templates are shared. Holdout tasks must not be
used for candidate/tolerance calibration. This is parameter-level holdout, not
unseen task-family coverage or proof of exclusion from model pretraining. Review
outcome: suitable for fixed-fixture correctness/diagnostic comparisons with the
stated scope. It is underpowered for the proposed 3 percentage-point task
noninferiority margin even with no observed discordances, and shared templates
make iid bounds optimistic. Do not use it to accept broad task-quality
noninferiority; a later acceptance suite needs more independent task families or
a reviewed cluster-aware design.

The scorer accepts visible response content only and requires exactly the
requested JSON object, keys, types and values. It rejects duplicate keys,
nonfinite values, integer/boolean substitutions, markdown wrappers and extra text.
Reasoning must be retained separately, not stripped from arbitrary output until
a convenient JSON answer is found. Invalid expected-answer data fails as an
oracle error, rather than recording a model failure. No generated code is executed.

All 120 prompts were rendered/tokenized twice using the installed native text
reference, embedded template and explicit thinking=true. A fresh full-model hash
matched the original reference. A separate audit verified task/render/token/file
hashes, split counts, token counts and 0600 file modes for every record. Exact
native records remain in `.private/epic2-quality-tokens-v1`; no inference occurred.
Direct-Jinja text replay does not establish server preprocessing parity.

Paired task diagnostics use task-wise wins/losses with a conservative >=95%
accuracy-drop upper bound from Bonferroni-combined one-sided 97.5%
Clopper-Pearson bounds. The best-case zero-discordance upper bound for 100 tasks
is `1 - 0.025**(1/100) = 3.62` percentage points. The
`minimum_zero_discordance_tasks` calculation in `quality_tasks.py` finds that at
least 122 independent paired tasks with zero wins and zero losses are needed to
put this bound at or below the proposed 3-point margin (the bound at 121 is
3.003%; at 122 it is 2.978%). Thus the frozen 100-task suite cannot prove the
proposed margin even in the best observed case. This 122-task result is only an
iid planning lower bound: the current tasks share five templates, so clustered
task dependence needs broader task families or a cluster-aware method before any
noninferiority claim. Do not manufacture a zero uncertainty interval from
all-equal bootstrap samples.

## Prose/code NLL corpus

[fidelity_corpus.py](../scripts/fidelity_corpus.py) built a separate private corpus
from immutable primary-source data. The
[corpus recipe](../research/fixtures/fidelity-corpus-v1.json) records source
revisions/hashes, license provenance, selection, native-reference provenance and
the resulting corpus hash. Exact model/tokenizer identities remain private. No source code from these datasets was executed.

| Stratum | Source and license evidence | Local separation and scope |
| --- | --- | --- |
| Problem prose/reasoning | GSM8K test data at `3101c7d5072418e28b9008a6636bde82a006892c`; [MIT license](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE), [dataset description](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/README.md) | Document text hashes assign approximately 20% to local calibration and the rest to holdout; human math-word-problem prose/solutions, preserving calculation annotations |
| Python code | MBPP at `f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81`; [repository dataset license notice](https://github.com/google-research/google-research/blob/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/README.md) specifies CC BY 4.0 for datasets; [MBPP description/splits](https://github.com/google-research/google-research/blob/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/mbpp/README.md) | Official validation IDs 511–600 are calibration; test IDs 11–510 are holdout. Instruction comments, reference Python solutions, setup text and standard tests are data, not executed programs |

MBPP is attributed to Austin et al., *Program Synthesis with Large Language
Models* (2021); GSM8K to Cobbe et al., *Training Verifiers to Solve Math Word
Problems* (2021), as identified in the linked primary repositories. License and
notice text is retained with the private source capture. No raw external corpus
is added to the public tree; redistribution would require preserving the
applicable license/attribution requirements and publication review. The project
license remains GPL-3.0.

The pinned source notices support the recorded MIT license for GSM8K and CC BY
4.0 dataset license for MBPP. MBPP's official test IDs 11–510 remain holdout and
validation IDs 511–600 provide calibration; GSM8K test documents use the fixed
document-hash partition described in the corpus recipe. This review confirms
source provenance, recorded notices and the local split rule, not a legal opinion,
pretraining exclusion or semantic deduplication. Those unknowns remain explicit.

After calibration-first exact text/code-body deduplication, a fixed SHA-256 ID
ranking selects whole documents until each stratum meets its quota. No padding,
repeated documents, character-count token estimates or truncation fills a quota.
A source shortage fails. Selected documents use raw native text tokenization,
automatic BOS and no escape interpretation, with reset KV per document. They
are distinct from chat-formatted performance/scored-task fixtures.

| Split | Problem-prose scored tokens | Python-code scored tokens |
| --- | --- | --- |
| Calibration | 2021 | 2037 |
| Holdout | 50085 | 50043 |

The 550 selected documents provide 100128 held-out scored next-token positions.
Every selected token stream passed native replay. All documents fit the frozen
4096-token per-document ceiling; token zero supplies context, and logits at input
j predict token j+1. A separate audit checked source/corpus/document/token hashes,
source-derived text and IDs, exact deduplication, scored counts and private modes.
Raw source files, license evidence, selected text/IDs and verification remain in
`.private/epic2-corpus-v1`.

This is a problem-prose/Python stratum, not coverage of general narrative prose,
other languages or other programming languages. No pretraining-exclusion or
semantic-near-duplicate claim is made. The fixed source styles and split are
accepted only for this narrow v1 diagnostic use; other domains need an expanded
version before broad quality claims. Do not infer a broad fidelity result from a
pass on this corpus. Corpus and task changes create a new version before
candidate comparison, with affected references recaptured.

## Execution and analysis still required

No NLL or task generations have been collected. The native capture path now
reduces target log probability during teacher-forced CPU evaluation and retains
per-token FP64 NLL values, exact token hashes and document boundaries at the
post-softcap stage. Private calibration/holdout bundles cover 4,058 and 100,128
targets. The pinned-header API-double check verified synthetic reduction and
target alignment; a separate strict native-linked helper build passed
`--validate-only` on both bundles. Its previously recorded llama/ggml library
hashes were unchanged, and the explicit loader environment resolved 11 hashed
shared libraries. This input-only path initialized no backend and loaded no
model. Corpus evaluation is still outstanding. The separately planned
4096-position correctness oracle continues to retain full-vocabulary logits.

[nll_compare.py](../scripts/nll_compare.py) now implements stable full-vocabulary
target NLL and complete document-paired analysis. Captures bind corpus/run identity,
exact token hashes, target counts and pre/post-softcap stage; incomplete or
mismatched documents, split/stage or nonfinite/negative values fail. It preserves
N-1 target denominators, resamples whole paired documents within strata, and keeps
the frozen token-weighted stratum mixture. It reports mean NLL, perplexity with an
explicit overflow reason, a seeded percentile interval and coverage; no quality
acceptance follows automatically. Analytic/unit tests and a separate serialized
CLI fixture validate the method on synthetic data. An independent C++ long-double
full-vocabulary reducer was also compared against the Python binary64 reducer on
four retained rows (two rows in each of two existing captures, vocabulary 262144).
The maximum observed arithmetic difference stayed within the declared 2e-12-nat
cross-check bound. This checks only reduction agreement on those rows; it does not
validate corpus NLL, model quality or acceptance. The native full-vocabulary
helper now also supports `RVNLL001`: during teacher-forced CPU evaluation it
reduces each post-softcap row to one stable long-double target NLL and writes
little-endian FP64 scalars. [nll_capture.py](../scripts/nll_capture.py) prepares
the exact private corpus token streams and finalizes only complete, successful,
reaped captures with exact document/target alignment. Private calibration and
holdout input bundles now cover 4,058 and 100,128 targets respectively; the
native parser accepted both in `--validate-only` mode. The pinned-header API-double
check passed synthetic full-vocabulary reductions against the Python reference.
No native model NLL run has occurred; loaded-build provenance, corpus NLL,
reference variability, model quality and acceptance remain unmeasured.

Analysis must report inconclusive margins honestly. Apply the protocol's proposed NLL/accuracy limits only after review and
reference-variability calibration. Context boundaries, stochastic-distribution
checks and MTP correctness still need their separately specified fixtures and
execution; the 100 scored tasks and 100k corpus are not substitutes for them.

## Sampled-generation distribution comparison

[`stochastic_capture.py`](../scripts/stochastic_capture.py) freezes an exact
prompt/seed plan and finalizes its bundle map. Plan v2 pins the model, sampler
policy and sampler-observer binary hashes. It retains one record for every planned
seed, including missing, failed and invalid attempts; it emits an analyzer capture
only when every sample has a successful, identity-matched event stream. It never
launches inference.

[`capture_sampled_ids.cpp`](../research/capture_sampled_ids.cpp) is a process-local
observer for the pinned `common_sampler_sample` ABI. Build it as an `LD_PRELOAD`
shared object in a fresh private 0700 directory with the existing C++ compiler:

```sh
c++ -std=c++17 -O2 -fPIC -shared -fno-fast-math -ffp-contract=off \
  research/capture_sampled_ids.cpp -ldl -pthread \
  -o "$PRIVATE_DIR/sampler-observer.so"
```

Include its hash as
`sampler_observer_sha256` in the frozen plan, run identity and matched
`sample_observer` expected-artifact check. It forwards each native sampled token
unchanged and writes indexed `ROSEVINE_SAMPLED_TOKEN_V1` events to stderr. The
event stream is checksummed by the supervisor with the native output. Finalization
requires one ready/done pair, contiguous IDs, a complete event count and either
EOS or the declared output limit; missing or reordered events invalidate the
sample. Unsupported symbol/version or write failures do not become successful
receipts. The added per-token I/O makes this a correctness diagnostic, not a
performance measurement. A synthetic shared-library test checks forwarding and
event capture. The installed completion binary also resolved the observer on its
`--help` path and exited successfully; it made zero sampling calls and loaded no
model. That confirms ABI availability only. No sampled model outputs have been
collected.

[`stochastic_distribution.py`](../scripts/stochastic_distribution.py) accepts two
paired private `rosevine-stochastic-output-v3` captures. Each prompt must have at
least 20 distinct planned seeds, identical prompt and sampler-policy hashes across
arms, and one successful output for every seed; failed or unplanned samples cannot
be silently dropped. The analyzer reports per-position empirical total-variation
and Jensen-Shannon distances, token-support sizes and EOS rates. It pads positions
after EOS with EOS so early stopping remains visible. Results are equal-weighted
across prompts and have no candidate-derived tolerance or acceptance decision.
These are position-wise marginals, not a sequence-level distribution test. Each
seed must link to a successful private supervisor bundle whose checksummed stderr
contains the exact token events and whose run identity matches the prompt, seed,
sampler policy, observer and model hash. The bundle must also contain matched
prelaunch target-model and observer hash checks; the model check still does not
prove which bytes the runtime loaded.
All bundle paths must remain inside the private capture directory. Synthetic tests
cover matching/disjoint support, EOS behavior, seed completeness, failed outcomes,
prompt/sampler identity, event binding and path confinement. No sampled model
outputs have been collected or compared.
