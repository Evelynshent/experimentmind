# ExperimentMind

ExperimentMind explores how AI can participate in scientific investigation
without becoming the source of quantitative truth. It keeps computed facts,
AI reasoning, and product decisions visibly separate:

- Statistics and practical-significance classifications are computed by code.
- A deterministic policy produces the recommendation.
- An optional LLM produces typed findings without calculation authority.
- A deterministic verifier checks structured claims against Evidence.
- A Markdown report labels each layer for human review.

V2 adds a bounded investigation loop. The system can recognize that top-line
evidence is insufficient, identify a finite set of scientifically valid next
analyses, run the selected analysis with deterministic Python, and form
competing hypotheses without presenting a mechanism as proven.

The AI is not just writing the report. When multiple valid next analyses exist,
it may rank only those pre-approved candidates by expected decision relevance.
It cannot invent metrics, dimensions, Python, SQL, or analytical tools.

## Quick start

Install the package and test dependencies:

```bash
python -m pip install -e ".[test]"
pytest
```

Run the complete deterministic path without a network connection or API key:

```bash
experimentmind --seed 42
# Equivalent: python -m experimentmind.demo --seed 42
```

Run the flagship V2 investigation offline:

```bash
experimentmind --scenario hidden_heterogeneity --seed 42
```

The top-line revenue result is uncertain. Deterministic planning identifies two
valid analyses—tenure segmentation and revenue decomposition—and the offline
fallback selects the pre-specified tenure analysis. It reveals a clearly
positive effect for new users and a clearly negative effect for existing
users, changing the final policy output to `VALIDATE_HETEROGENEITY`.

That recommendation blocks a global rollout. It does not claim that targeting
is ready; it calls for confirming the interaction before evaluating a targeted
treatment.

The offline report contains Evidence and the deterministic recommendation, with
no AI findings. To include live structured findings, set an API key and choose
a Structured-Outputs-capable model explicitly:

```bash
export OPENAI_API_KEY="your-api-key"
experimentmind --seed 42 --model "your-model"
```

The live command can incur API usage. ExperimentMind never selects a model
implicitly.

For V2 scenarios, `--model` additionally allows the model to rank candidates
when more than one valid analysis exists and to generate structured competing
hypotheses. Candidate validation, analysis execution, hypothesis checks, and
the recommendation remain deterministic.

## V2 investigation scenarios

| Scenario | Top-line pattern | Investigation behavior | Final decision |
|---|---|---|---|
| `clear_win` | Primary metric clearly positive | Stop; additional analysis is unnecessary | `SHIP` |
| `shipping_tradeoff` | Conversion improves while shipping cost worsens | Decompose revenue into conversion × value per conversion | `TRADEOFF` |
| `hidden_heterogeneity` | Overall revenue is inconclusive | Segment by pre-specified user tenure | `VALIDATE_HETEROGENEITY` |

The only supported investigation tools are segmentation by a pre-specified
dimension and one transparent revenue identity:

```text
revenue per session
= conversion rate × revenue per converted session
```

Evidence sufficiency is classified as `SUFFICIENT`, `INSUFFICIENT`, or
`CONFLICTING`. With multiple valid candidates, the optional model can rank
them; with one candidate, deterministic logic selects it directly.
If the first analysis leaves the decision unresolved, the remaining valid
candidate may run once as a deterministic follow-up.

Hypotheses cite top-line or investigation evidence and receive one of three
limited statuses: `CONSISTENT_WITH_EVIDENCE`, `INSUFFICIENT_EVIDENCE`, or
`CONTRADICTED`. Consistency means only that observed directions do not reject
the explanation. It is not causal verification.

## Synthetic experiment

The fictional experiment lowers an e-commerce free-shipping threshold from
$50 to $35. The treatment is designed to increase conversion while reducing
order values and increasing subsidized shipping. These opposing mechanisms
make revenue per session uncertain rather than making the experiment an
obvious success or failure.

The generator produces balanced, user-session-level observations with a fixed
seed. The three decision-relevant metrics are:

- **Revenue per session** (primary)
- **Conversion rate** (secondary)
- **Shipping cost per session** (guardrail)

## Statistical assumptions

Each row is an independently randomized session and every row is analyzed in
its assigned variant (intention to treat). Conversion uses a two-sided
two-proportion z-test and Newcombe confidence interval. Revenue and shipping
cost use two-sided Welch tests and Welch confidence intervals, which allow arm
variances to differ. With roughly 10,000 sessions per arm, inference targets
differences in arm means despite the continuous outcomes being zero-inflated
and right-skewed.

Every confidence interval covers the absolute effect, defined as treatment
minus control. Relative effects are point estimates and are omitted when the
control value is zero. The default significance level is 0.05; significance
itself is not converted directly into a product recommendation.

## Evidence classification

Each metric records whether higher or lower values are favorable and a minimum
meaningful absolute effect. A deterministic function combines that context
with the effect, confidence interval, and p-value:

- **Clearly positive:** statistically significant and favorable beyond the
  meaningful-effect threshold
- **Clearly negative:** statistically significant and unfavorable beyond the
  threshold
- **Negligible:** statistically significant but no larger than the threshold,
  or not significant with the full confidence interval inside the threshold
  bounds
- **Uncertain:** the data cannot rule out a meaningful effect

The synthetic experiment uses thresholds of $0.05 revenue per session, 0.2
percentage points of conversion, and $0.02 shipping cost per session. These
classifications remain deterministic evidence summaries.

## Decision policy

Metrics are labeled primary, secondary, or guardrail. One explicit,
priority-ordered policy converts their classifications into `SHIP`,
`DO_NOT_SHIP`, `COLLECT_MORE_DATA`, or `TRADEOFF`:

1. A clearly negative primary metric means do not ship.
2. A clearly negative guardrail with countervailing positive primary or
   secondary evidence is a tradeoff requiring human judgment.
3. A clearly negative guardrail without positive evidence means do not ship.
4. A clearly positive primary metric with no harmed guardrail means ship.
5. A negligible primary effect means do not ship.
6. An uncertain primary effect means collect more data.

Secondary metrics cannot independently trigger shipping. The seeded experiment
returns `TRADEOFF`: revenue per session is uncertain, conversion is clearly
positive, and shipping cost per session is clearly negative. The policy and
its rationale are computed by code; no LLM is involved.

## Structured findings

An optional LLM analyst can turn Evidence into typed observations and
interpretations. It receives every metric value and the fixed deterministic
recommendation, then returns Pydantic-validated findings through the OpenAI
Responses API's Structured Outputs support. Every finding must name its
Evidence references.

The LLM never computes statistics or changes the recommendation.

## Finding verification

Each observation carries structured metric claims containing the asserted
effect scale, value, significance, and classification. Each interpretation
carries asserted metric directions and a list of introduced concepts. A pure
Python verifier checks those fields against Evidence:

- Observations become `VERIFIED`, `INCORRECT`, or `UNRESOLVED`.
- Interpretations become `CONSISTENT_WITH_EVIDENCE`,
  `CONTRADICTED_BY_EVIDENCE`, or `INSUFFICIENT_EVIDENCE`.

Numerical claims use a documented 1% relative comparison tolerance. An
interpretation is consistent only when its declared metric directions agree
with Evidence and every declared concept corresponds to a measured metric.
Consistency does not establish causation.

The verifier checks structured assertions, not arbitrary prose semantics. This
keeps verification deterministic and avoids fragile claim extraction, but it
also means correctness depends on the structured fields faithfully expressing
the accompanying statement.

```python
from experimentmind import generate_findings, recommend, verify_findings

recommendation = recommend(evidence)
findings = generate_findings(
    evidence,
    recommendation,
    model="your-structured-output-capable-model",
)
verified_findings = verify_findings(findings.findings, evidence)
```

Set `OPENAI_API_KEY` in the environment before making a live API call. Tests
use an injected fake client and never contact the API.

## Markdown report

A pure renderer combines Evidence, verified findings, and the deterministic
recommendation into a Markdown report. Computed facts, AI-generated statements,
verification verdicts, and policy output appear in separate labeled sections.

```python
from experimentmind import render_report

report = render_report(evidence, verified_findings, recommendation)
print(report)
```

Rendering performs no statistical calculations, model calls, or verification.
It only formats the existing authoritative objects for human review.

## Library usage

```python
from experimentmind import (
    analyze_experiment,
    generate_shipping_threshold_experiment,
)

observations = generate_shipping_threshold_experiment(seed=42)
evidence = analyze_experiment(observations)
```

The optional live smoke test is skipped by default. Run it only when you intend
to make a real API request:

```bash
EXPERIMENTMIND_RUN_LIVE_API=1 \
OPENAI_MODEL="your-model" \
pytest -m live_api tests/test_live_api.py
```

See `SPEC.md` for the broader project direction and `CLEAN_ROOM.md` for the
clean-room constraints.

## Codex skill

The repository includes a thin Codex workflow skill at
`skills/experimentmind/SKILL.md`. It teaches Codex how to run and inspect the
project and conduct bounded scientific investigation while preserving the
deterministic evidence boundaries; it does not duplicate the Python statistics
or decision logic.

To make the skill available outside this repository, install the
`skills/experimentmind` directory with Codex's skill installer or copy it into
your Codex skills directory. The repository remains fully usable without the
skill.

Representative skill-evaluation prompts and human-readable behavioral criteria
live in `skills/experimentmind/evals/cases.json`. These cases test the skill's
workflow boundaries; they are not an LLM judge or a replacement for the
deterministic application tests and verifier described in `SPEC.md`.
