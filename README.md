# ExperimentMind

ExperimentMind is a clean-room, independent open-source exploration of
trustworthy experiment analysis. The current deterministic foundation contains
a reproducible synthetic e-commerce experiment, standard statistical tests,
immutable structured Evidence, and practical-significance classification.

No LLM, decision policy, verifier, UI, database, or external integration is
included in this phase.

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

## Run

```bash
python -m pip install -e ".[test]"
pytest
```

```python
from experimentmind import (
    analyze_experiment,
    generate_shipping_threshold_experiment,
)

observations = generate_shipping_threshold_experiment(seed=42)
evidence = analyze_experiment(observations)
```

See `SPEC.md` for the broader project direction and `CLEAN_ROOM.md` for the
clean-room constraints. Components described there but not present here belong
to later phases.
