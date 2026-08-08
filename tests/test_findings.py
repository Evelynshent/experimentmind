import pytest
from pydantic import ValidationError

from experimentmind.evidence import EvidenceClassification
from experimentmind.findings import (
    Direction,
    DirectionalClaim,
    EffectScale,
    Finding,
    FindingBatch,
    FindingType,
    MetricClaim,
)


def observation_payload() -> dict[str, object]:
    return {
        "statement": "Conversion increased.",
        "finding_type": "observation",
        "evidence_refs": ["conversion_rate"],
        "metric_claims": [
            {
                "metric_name": "conversion_rate",
                "effect_scale": "relative",
                "effect_value": 0.18,
                "statistically_significant": True,
                "classification": "clearly_positive",
            }
        ],
        "directional_claims": [],
        "concepts": [],
    }


def test_finding_batch_validates_structured_findings() -> None:
    batch = FindingBatch(
        findings=[
            Finding(
                statement="Conversion increased.",
                finding_type=FindingType.OBSERVATION,
                evidence_refs=["conversion_rate"],
                metric_claims=[
                    MetricClaim(
                        metric_name="conversion_rate",
                        effect_scale=EffectScale.RELATIVE,
                        effect_value=0.18,
                        statistically_significant=True,
                        classification=EvidenceClassification.CLEARLY_POSITIVE,
                    )
                ],
                directional_claims=[],
                concepts=[],
            )
        ]
    )

    assert batch.findings[0].finding_type is FindingType.OBSERVATION


@pytest.mark.parametrize("field", ["statement", "evidence_refs", "metric_claims"])
def test_empty_required_observation_field_is_rejected(field: str) -> None:
    payload = observation_payload()
    payload[field] = "" if field == "statement" else []

    with pytest.raises(ValidationError):
        Finding.model_validate(payload)


def test_finding_rejects_unexpected_field() -> None:
    payload = observation_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        Finding.model_validate(payload)


def test_observation_rejects_interpretation_fields() -> None:
    payload = observation_payload()
    payload["concepts"] = ["profitability"]

    with pytest.raises(ValidationError, match="observations cannot"):
        Finding.model_validate(payload)


def test_interpretation_requires_directional_claims_and_no_metric_claims() -> None:
    payload = observation_payload()
    payload["finding_type"] = "interpretation"

    with pytest.raises(ValidationError, match="cannot contain metric claims"):
        Finding.model_validate(payload)


def test_structured_claims_must_be_cited() -> None:
    payload = observation_payload()
    payload["evidence_refs"] = ["revenue_per_session"]

    with pytest.raises(ValidationError, match="evidence_refs"):
        Finding.model_validate(payload)


def test_interpretation_shape_is_valid() -> None:
    finding = Finding(
        statement="Lower conversion may reduce revenue.",
        finding_type=FindingType.INTERPRETATION,
        evidence_refs=["conversion_rate", "revenue_per_session"],
        metric_claims=[],
        directional_claims=[
            DirectionalClaim(
                metric_name="conversion_rate", direction=Direction.DECREASED
            )
        ],
        concepts=["conversion_rate", "revenue_per_session"],
    )

    assert finding.finding_type is FindingType.INTERPRETATION


def test_empty_finding_batch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FindingBatch(findings=[])


def test_structured_output_schema_requires_every_finding_field() -> None:
    schema = FindingBatch.model_json_schema()
    finding_schema = schema["$defs"]["Finding"]

    assert set(finding_schema["required"]) == {
        "statement",
        "finding_type",
        "evidence_refs",
        "metric_claims",
        "directional_claims",
        "concepts",
    }
    assert finding_schema["additionalProperties"] is False
