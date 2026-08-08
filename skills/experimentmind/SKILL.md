---
name: experimentmind
description: Run, inspect, and explain the ExperimentMind synthetic e-commerce experiment while preserving its deterministic evidence, verification, and decision boundaries. Use when working in the ExperimentMind repository to execute the demo, interpret its report, inspect experiment evidence, troubleshoot the analysis pipeline, or make scoped changes governed by SPEC.md and CLEAN_ROOM.md.
---

# ExperimentMind

Use the repository's Python implementation as the sole computational authority. Keep this skill as workflow guidance; do not reproduce statistics, thresholds, classifications, verification, or decision logic in the skill.

## Establish scope

1. Locate the repository root containing `SPEC.md`, `CLEAN_ROOM.md`, and `pyproject.toml`.
2. Read `SPEC.md` and `CLEAN_ROOM.md` before changing project behavior.
3. Treat `SPEC.md` as the only project-specific source of truth.
4. Flag conflicts between the user's request and the specification instead of silently expanding scope.
5. Never use employer, proprietary, or remembered internal implementations, data, prompts, or documentation.

## Run the deterministic workflow

Install the local package with test dependencies when needed:

```bash
python -m pip install -e ".[test]"
```

Run the offline demo by default:

```bash
python -m experimentmind.demo --seed 42
```

Treat the resulting `Evidence` values and deterministic recommendation as authoritative. Explain treatment effects as treatment minus control. Distinguish computed evidence, deterministic classification, policy output, AI-generated findings, and verifier verdicts.

## Use the optional model path

Invoke a live model only when the user explicitly requests AI-generated findings and supplies or confirms both an API key and model name:

```bash
python -m experimentmind.demo --seed 42 --model "MODEL_NAME"
```

Warn that this command makes a paid external API request. Never choose a model implicitly. Never present generated findings as computed facts or allow them to replace the deterministic recommendation.

## Inspect or modify the project

- Prefer the smallest direct change that satisfies `SPEC.md`.
- Preserve the typed `Evidence` object as the quantitative source of truth.
- Use mature public statistical libraries rather than reimplementing statistical algorithms.
- Keep deterministic calculations, LLM interpretation, verification, and reporting visibly separated.
- Do not add services, agents, databases, RAG, memory, or infrastructure unless the specification and current user request explicitly place them in scope.
- Add or update focused tests for behavior changes.

Run the default verification suite after changes:

```bash
python -W error -m pytest
```

The live API smoke test is opt-in. Do not run it unless the user explicitly authorizes a real request.

## Report results

State whether execution was offline or live, identify the seed and sample size, summarize the primary and guardrail evidence, and report the deterministic recommendation separately from any AI findings. Surface statistical assumptions and limitations when they affect interpretation.
