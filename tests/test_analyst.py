import json
from types import SimpleNamespace
from typing import Any

import pytest

from experimentmind.analyst import (
    SYSTEM_INSTRUCTIONS,
    format_analysis_input,
    generate_findings,
)
from experimentmind.evidence import EvidenceClassification
from experimentmind.findings import (
    EffectScale,
    Finding,
    FindingBatch,
    FindingType,
    MetricClaim,
)
from experimentmind.policy import recommend
from experimentmind.statistics import analyze_experiment
from experimentmind.synthetic import generate_shipping_threshold_experiment


def authoritative_inputs():
    evidence = analyze_experiment(generate_shipping_threshold_experiment(seed=42))
    return evidence, recommend(evidence)


class FakeResponses:
    def __init__(self, output: FindingBatch | None) -> None:
        self.output = output
        self.arguments: dict[str, Any] | None = None
        self.call_count = 0

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.call_count += 1
        self.arguments = kwargs
        return SimpleNamespace(output_parsed=self.output)


class FakeClient:
    def __init__(self, output: FindingBatch | None) -> None:
        self.responses = FakeResponses(output)


def test_analysis_input_contains_all_authoritative_fields() -> None:
    evidence, recommendation = authoritative_inputs()

    payload = json.loads(format_analysis_input(evidence, recommendation))
    metrics = {metric["metric_name"]: metric for metric in payload["metrics"]}

    assert payload["experiment_name"] == evidence.experiment_name
    assert payload["alpha"] == evidence.alpha
    assert payload["recommendation"]["decision"] == "tradeoff"
    assert set(metrics) == {
        "revenue_per_session",
        "conversion_rate",
        "shipping_cost_per_session",
    }
    conversion = metrics["conversion_rate"]
    source = next(metric for metric in evidence.metrics if metric.metric_name == "conversion_rate")
    assert conversion["control_value"] == source.control_value
    assert conversion["treatment_value"] == source.treatment_value
    assert conversion["absolute_effect"] == source.absolute_effect
    assert conversion["relative_effect"] == source.relative_effect
    assert conversion["confidence_interval"] == list(source.confidence_interval)
    assert conversion["p_value"] == source.p_value
    assert conversion["sample_size_control"] == source.sample_size_control
    assert conversion["sample_size_treatment"] == source.sample_size_treatment
    assert conversion["classification"] == "clearly_positive"
    assert conversion["role"] == "secondary"


def test_generate_findings_makes_one_structured_api_call() -> None:
    evidence, recommendation = authoritative_inputs()
    expected = FindingBatch(
        findings=[
            Finding(
                statement="Revenue remains uncertain.",
                finding_type=FindingType.OBSERVATION,
                evidence_refs=["revenue_per_session"],
                metric_claims=[
                    MetricClaim(
                        metric_name="revenue_per_session",
                        effect_scale=EffectScale.RELATIVE,
                        effect_value=0.01,
                        statistically_significant=False,
                        classification=EvidenceClassification.UNCERTAIN,
                    )
                ],
                directional_claims=[],
                concepts=[],
            )
        ]
    )
    client = FakeClient(expected)
    input_before = format_analysis_input(evidence, recommendation)

    actual = generate_findings(
        evidence, recommendation, model="test-model", client=client  # type: ignore[arg-type]
    )

    assert actual is expected
    assert format_analysis_input(evidence, recommendation) == input_before
    assert client.responses.call_count == 1
    assert client.responses.arguments is not None
    assert client.responses.arguments["model"] == "test-model"
    assert client.responses.arguments["text_format"] is FindingBatch
    assert client.responses.arguments["input"][0] == {
        "role": "system",
        "content": SYSTEM_INSTRUCTIONS,
    }
    user_payload = json.loads(client.responses.arguments["input"][1]["content"])
    assert user_payload["recommendation"]["decision"] == "tradeoff"


def test_generate_findings_rejects_empty_model_without_calling_api() -> None:
    evidence, recommendation = authoritative_inputs()
    client = FakeClient(None)

    with pytest.raises(ValueError, match="model"):
        generate_findings(
            evidence, recommendation, model="  ", client=client  # type: ignore[arg-type]
        )

    assert client.responses.arguments is None
    assert client.responses.call_count == 0


def test_generate_findings_rejects_missing_parsed_output() -> None:
    evidence, recommendation = authoritative_inputs()

    with pytest.raises(RuntimeError, match="parsed findings"):
        generate_findings(
            evidence,
            recommendation,
            model="test-model",
            client=FakeClient(None),  # type: ignore[arg-type]
        )
