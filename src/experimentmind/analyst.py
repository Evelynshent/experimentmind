"""One structured LLM call over immutable experiment Evidence."""

import json

from openai import OpenAI

from .evidence import Evidence
from .findings import FindingBatch
from .policy import Recommendation


SYSTEM_INSTRUCTIONS = """You are an experiment analyst.

The supplied Evidence is the only source of quantitative truth. Never calculate,
change, or invent statistics. Produce concise structured findings only.

For observations:
- State only facts explicitly present in Evidence.
- Cite every metric used in evidence_refs.
- Do not claim causation.

For interpretations:
- Use tentative language such as "may" or "could".
- Cite the metrics that make the interpretation plausible.
- Do not present an interpretation as established fact.

The deterministic recommendation is fixed. You may explain its evidence pattern,
but you must not replace or override it.
"""


def format_analysis_input(
    evidence: Evidence, recommendation: Recommendation
) -> str:
    """Serialize authoritative inputs without recomputing or rounding values."""

    payload = {
        "experiment_name": evidence.experiment_name,
        "alpha": evidence.alpha,
        "metrics": [
            {
                "metric_name": metric.metric_name,
                "metric_type": metric.metric_type.value,
                "role": metric.metric_spec.role.value,
                "higher_is_better": metric.metric_spec.higher_is_better,
                "meaningful_effect": metric.metric_spec.meaningful_effect,
                "control_value": metric.control_value,
                "treatment_value": metric.treatment_value,
                "absolute_effect": metric.absolute_effect,
                "relative_effect": metric.relative_effect,
                "confidence_interval": list(metric.confidence_interval),
                "confidence_level": metric.confidence_level,
                "p_value": metric.p_value,
                "sample_size_control": metric.sample_size_control,
                "sample_size_treatment": metric.sample_size_treatment,
                "classification": metric.classification.value,
            }
            for metric in evidence.metrics
        ],
        "recommendation": {
            "decision": recommendation.decision.value,
            "rationale": list(recommendation.rationale),
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def generate_findings(
    evidence: Evidence,
    recommendation: Recommendation,
    *,
    model: str,
    client: OpenAI | None = None,
) -> FindingBatch:
    """Generate schema-constrained findings with one Responses API call."""

    if not model.strip():
        raise ValueError("model must not be empty")

    api_client = client if client is not None else OpenAI()
    response = api_client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": format_analysis_input(evidence, recommendation)},
        ],
        text_format=FindingBatch,
    )
    if response.output_parsed is None:
        raise RuntimeError("model response did not contain parsed findings")
    return response.output_parsed
