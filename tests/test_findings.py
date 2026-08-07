import pytest
from pydantic import ValidationError

from experimentmind.findings import Finding, FindingBatch, FindingType


def test_finding_batch_validates_structured_findings() -> None:
    batch = FindingBatch(
        findings=[
            Finding(
                statement="Conversion increased.",
                finding_type=FindingType.OBSERVATION,
                evidence_refs=["conversion_rate"],
            )
        ]
    )

    assert batch.findings[0].finding_type is FindingType.OBSERVATION


@pytest.mark.parametrize(
    "payload",
    [
        {"statement": "", "finding_type": "observation", "evidence_refs": ["metric"]},
        {"statement": "Claim", "finding_type": "unsupported", "evidence_refs": ["metric"]},
        {"statement": "Claim", "finding_type": "observation", "evidence_refs": []},
        {
            "statement": "Claim",
            "finding_type": "observation",
            "evidence_refs": ["metric"],
            "unexpected": True,
        },
    ],
)
def test_invalid_finding_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate(payload)


def test_empty_finding_batch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FindingBatch(findings=[])
