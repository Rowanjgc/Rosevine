# Rosevine
A custom LLM inference engine tailored for Gemma4 26B A4B QAT on a 7800XT GPU.

Project Zero establishes privacy and engineering guardrails before inference
research. Backend and host language are set by
[ADR 0001](docs/adr/0001-primary-backend-and-runtime-strategy.md). Licensed under GPL-3.0;
see [LICENSE](LICENSE). This project does not replace the production llama.cpp router.

```sh
./scripts/bootstrap.sh
./scripts/check.sh
```

Read [CONTRIBUTING](CONTRIBUTING.md), [agent instructions](AGENTS.md),
[privacy policy](docs/development/privacy-policy.md),
[development workflow](docs/development/engineering-workflow.md) and
[hardening status](docs/development/hardening-report.md) before contributing.
Report vulnerabilities using [SECURITY](SECURITY.md).

Research planning now has a [charter](docs/rosevine-research-charter.md),
[observed baseline](docs/hardware-and-inference-baseline.md),
[architecture options](docs/architecture-options.md),
[Beads roadmap](docs/research-roadmap.md),
[experiment protocol](docs/experiment-protocol.md) and
[open questions](docs/open-research-questions.md). Backend and language choices
remain proposed; inference implementation has not begun. The local Beads tracker
and raw research evidence remain private.
