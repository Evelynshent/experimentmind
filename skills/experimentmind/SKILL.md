---
name: experimentmind
description: Investigate controlled experiments with ExperimentMind while preserving deterministic evidence and decision boundaries. Use when running or interpreting ExperimentMind scenarios, assessing whether top-line evidence is sufficient, selecting among supported segmentation or revenue-decomposition analyses, comparing competing hypotheses, verifying evidence-backed claims, or making scoped repository changes governed by SPEC.md and CLEAN_ROOM.md.
---

# ExperimentMind

Use Python output as the sole quantitative authority. Use AI judgment to frame unresolved questions, rank only valid candidate analyses, and express competing hypotheses—not to calculate evidence or choose policy outcomes.

## Investigate scientifically

1. Read `SPEC.md` and `CLEAN_ROOM.md` before changing behavior.
2. State the experiment hypothesis, primary metric, supporting metrics, guardrails, and pre-specified effect modifiers.
3. Inspect statistical and practical significance together.
4. Stop when evidence is sufficient; do not investigate for appearance or completeness.
5. When evidence is conflicting or insufficient, obtain valid candidates from deterministic code.
6. If one candidate exists, run it directly. If several exist, rank only those candidates by likely decision relevance.
7. Execute segmentation or revenue decomposition through the repository's Python functions. Never calculate results in prose or invent another tool.
8. If the first analysis leaves material uncertainty, run the remaining valid candidate at most once.
9. Compare explanations using supporting, contradicting, and missing evidence.
10. Label mechanisms as hypotheses. Treat `CONSISTENT_WITH_EVIDENCE` as non-contradiction, never causal proof.
11. Present unresolved material uncertainty and the deterministic recommendation separately.

## Run the workflow

Run the flagship investigation offline by default:

```bash
python -m experimentmind.demo --scenario hidden_heterogeneity --seed 42
```

Other supported scenarios are `clear_win` and `shipping_tradeoff`. Omit
`--scenario` to run the preserved V1 demonstration.

Use a live model only when the user explicitly authorizes API usage and names a model:

```bash
python -m experimentmind.demo --scenario hidden_heterogeneity --seed 42 --model "MODEL_NAME"
```

Warn that the live command can incur cost. Never choose a model implicitly.

## Preserve boundaries

- Segment only by dimensions declared before observing results.
- Do not infer that heterogeneous effects validate targeting.
- Do not override `VALIDATE_HETEROGENEITY`, `TRADEOFF`, or another policy result.
- Do not present hypotheses as verified or causal.
- Do not introduce employer or proprietary implementation material.
- Do not add agents, RAG, memory, databases, arbitrary SQL, causal graphs, or infrastructure unless an approved specification explicitly changes scope.

After repository changes, run:

```bash
python -W error -m pytest
```

Do not run the opt-in live API smoke test without explicit authorization.
