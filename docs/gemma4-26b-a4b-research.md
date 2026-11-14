# Gemma 4 26B A4B — model research notes

> **Purpose:** Informational reference for Rosevine contributors and coding agents (including Claude). This is **not** a task list, specification, architectural decision record, or instruction to modify the implementation.
>
> **Research snapshot:** 2026-11-13. Check upstream revisions before treating any model detail as immutable.

## Sources and evidence levels

1. **Google model card (primary):** https://ai.google.dev/gemma/docs/core/model_card_4
2. **Google instruction-tuned model card/config (primary):** https://huggingface.co/google/gemma-4-26B-A4B-it
3. **Google base model (primary):** https://huggingface.co/google/gemma-4-26B-A4B
4. **Google technical report (primary):** https://arxiv.org/abs/2607.02770
5. **Google architecture visual guide (primary explanatory material):** https://github.com/google-gemma/docs/blob/main/docs/tutorials/visual-guides/gemma-4.md
6. **Independent comparative study:** Manik & Wang, *Gemma 4, Phi-4, and Qwen3: Accuracy-Efficiency Tradeoffs in Dense and MoE Reasoning Language Models*, https://arxiv.org/abs/2604.07035

**Evidence limitation:** These notes were prepared from the official model cards, published architecture material, and available paper abstracts/results. The full body of Google's technical report was **not independently inspected** during this research pass. Do not attribute unverified architectural claims to that report; inspect its full text and the exact model config when implementation-level precision matters.

## Model identity and architecture

| Property | Gemma 4 26B A4B |
| --- | --- |
| Family | Gemma 4, mixture-of-experts (MoE) |
| Total parameters | ~25.2 billion |
| Active parameters per token | ~3.8 billion |
| Transformer layers | 30 |
| Context length | Up to 256K tokens |
| Vocabulary | ~262K tokens |
| Routed experts | 128 |
| Routed experts selected per token | 8 |
| Shared expert | 1 (in addition to routed selection) |
| Local sliding attention window | 1,024 tokens |
| Modalities | Text, images, video represented as frames; output is text |
| Audio | Not a native input modality for this 26B variant |
| License | Apache 2.0 |

**Total versus active parameters:** ~25.2B describes the model's stored weight footprint, whereas ~3.8B describes approximate computation-active parameters per token. The active count does **not** imply that only 3.8B parameters must be resident somewhere in memory. All experts must remain addressable, and selected experts may differ by layer and token.

**Expert routing:** Each MoE layer routes a token to eight of 128 routed experts and also uses a shared expert. The precise router weighting, expert layout, and placement of MoE layers should be confirmed against the exact upstream configuration and tensor names, rather than inferred from this overview. In particular, avoid assuming the same experts are selected across tokens or layers.

**Attention:** Gemma 4 combines sliding-window local attention with global attention; Google's architectural material describes a global final layer and specialized global-attention features including unified key/value representations and proportional RoPE (p-RoPE). These details matter for KV-cache layout, positional encoding, attention kernels, and context scaling. The 1,024-token sliding window is **not** the model's overall context limit. Confirm the precise local/global layer schedule and per-layer KV dimensions against the actual checkpoint configuration before implementing them.

**Multimodality:** The family includes vision processing; the visual encoder is approximately 550M parameters according to the published material. For the 26B variant, image and video-frame input are relevant, but video support should not be mistaken for a standalone video-decoding or audio-understanding capability. A text-only inference path may not load the vision projector/encoder, depending on packaging.

## Published model-card benchmarks

Instruction-tuned variants; values are the model-card's published scores, **not Rosevine measurements**. Benchmark settings and evaluation harnesses differ from local quantized deployments.

| Benchmark | 26B A4B | 31B dense |
| --- | ---: | ---: |
| MMLU-Pro | 82.6 | 85.2 |
| AIME 2026 (no tools) | 88.3 | 89.2 |
| LiveCodeBench v6 | 77.1 | 80.0 |
| GPQA Diamond | 82.3 | 84.3 |
| MMMU-Pro | 73.8 | 76.9 |
| Tau2 | 68.2 | 76.9 |
| Humanity's Last Exam (no tools) | 8.7 | 19.5 |
| MRCR v2, 8-needle, 128K | 44.1 | 66.4 |

**Interpretation:** The MoE model approaches the larger dense model on several general, math, and coding tests despite much lower active compute. The gap is more substantial in the reported agentic, very difficult reasoning, and long-context retrieval tests. Neither benchmark results nor the 256K context *capacity* guarantee equal accuracy throughout the context window.

## Independent study

Manik & Wang (arXiv:2604.07035) compare Gemma 4, Phi-4, and Qwen3 dense/MoE models. Their abstract reports seven models and 8,400 model–dataset–prompt evaluations. Selected reported figures:

| Model | Reported accuracy | Reported mean VRAM |
| --- | ---: | ---: |
| Gemma 4 26B A4B | 0.663 | 48.1 GB |
| Gemma 4 E4B | 0.675 | 14.9 GB |

These are **study-specific experimental observations**, not universal model rankings, intrinsic memory requirements, or measurements of a quantized Rosevine configuration. The 48.1 GB observation does **not** mean the 26B A4B requires 48.1 GB of VRAM in every deployment. Inspect the paper's precision, runtime, batching, context, and memory-accounting methodology before making hardware comparisons.

## Memory arithmetic and inference implications

For 25.2B total parameters, idealized raw weight storage (excluding metadata and runtime memory) is approximately:

| Precision | Idealized weight bytes |
| --- | ---: |
| 16-bit | 50.4 GB |
| 8-bit | 25.2 GB |
| 5-bit | 15.75 GB |
| 4-bit | 12.6 GB |
| 3-bit | 9.45 GB |

These are **decimal GB**, not GiB. Actual GGUF or other quantized formats can exceed these figures because of mixed-precision tensors, quantization block metadata, embeddings, alignment, and auxiliary weights. Runtime allocations include KV cache, compute scratch, routing buffers, and possibly vision components. Context length, batch size, and offload policy materially change memory requirements.

On a 16 GB VRAM device, a roughly 4-bit quantized model **might** be able to keep most or all relevant weights on the GPU at a constrained context size; that must be tested with the actual artifact and runtime. Conversely, CPU/GPU hybrid expert execution can save VRAM while introducing per-token/per-layer transfer, synchronization, and CPU execution costs.

A useful conceptual latency decomposition is:

`decode time ≈ attention + shared/dense work + selected-expert compute + transfers + synchronization + scheduling overhead`

This is an **engineering model**, not a claim that these terms are perfectly additive or directly observable without instrumentation. The active-parameter count alone cannot predict throughput if routed expert weights are frequently moved across PCIe.

## Questions this research raises for Rosevine (not directives)

- How does measured decode latency differ between maximum practical GPU residency and hybrid CPU/GPU expert execution for the **same quantized artifact, context, and workload**?
- How often do selected expert IDs recur per layer across adjacent tokens? Is there useful locality for a bounded expert cache, and does that outweigh cache-management overhead?
- What are the actual host-to-device bytes, CPU expert times, GPU expert times, and sync stalls per generated token?
- Are local and global attention layers using the correct cache limits, RoPE variant, and KV shapes for the checkpoint?
- Does memory pressure from longer contexts change the preferred weight residency or expert placement?
- Are multimodal weights loaded only when required, and are image tokens/projector paths handled correctly?
- Do chat-template, system-message, thinking, and tool-call conventions match the selected instruction-tuned checkpoint?

These are investigation prompts only. They do **not** supersede existing Rosevine architecture documents, measured evidence, acceptance criteria, or Claude's active implementation plan.

## Prompting and interface details worth verifying

Google's published instruction-tuned guidance includes a default sampler around **temperature 1.0, top-p 0.95, top-k 64**; these are recommendations, not immutable correctness requirements. The model supports system messages and function-calling conventions. Published guidance discusses enabling thinking with a `<|think|>` token in the system prompt and cautions against indiscriminately replaying completed thinking traces in subsequent turns (tool-call continuations are a special case). Confirm these rules against the specific checkpoint's current chat template before changing prompt assembly.

The published vision guidance describes image token budgets of **70, 140, 280, 560, or 1120**, with images generally placed before accompanying text. Treat these as model-interface notes requiring verification against the deployed processor and projector, not as a universal tokenization formula.

## Boundaries and provenance

- **Official model-card numbers** are not measurements of Rosevine, llama.cpp, or the user's machine.
- **Independent-paper results** reflect that study's methodology; they do not establish quantized inference speed or minimum VRAM.
- **Theoretical memory arithmetic** is not a substitute for allocator traces or actual peak-residency measurements.
- **Potential optimization ideas** (expert caching, prefetch, hybrid execution, KV specialization) are hypotheses until benchmarked.
- **No implementation decision is made here.** This document exists so Claude and other contributors can cite source-backed model facts without confusing them with project requirements.

### Related Rosevine documentation

See `docs/cpu-experts.md`, `docs/cpu-experts-q8.md`, `docs/decode-attribution.md`, `docs/decode-instrumentation.md`, and `docs/cache-benefit-bounds.md` for project-specific implementation and measurement context. Those documents, not this research summary, are the appropriate sources for claims about what Rosevine has actually implemented or measured.
