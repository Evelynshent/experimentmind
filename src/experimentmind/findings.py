"""Typed, AI-generated findings that cite deterministic Evidence."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import EvidenceClassification


class FindingType(str, Enum):
    """Whether a finding is directly checkable or interpretive."""

    OBSERVATION = "observation"
    INTERPRETATION = "interpretation"


class EffectScale(str, Enum):
    """The Evidence effect estimate asserted by an observation."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class Direction(str, Enum):
    """A metric direction asserted by an interpretation."""

    INCREASED = "increased"
    DECREASED = "decreased"
    UNCHANGED = "unchanged"


class MetricClaim(BaseModel):
    """Machine-checkable quantitative assertion about one metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    effect_scale: EffectScale
    effect_value: float
    statistically_significant: bool
    classification: EvidenceClassification


class DirectionalClaim(BaseModel):
    """Machine-checkable directional assertion about one metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    direction: Direction


class Finding(BaseModel):
    """One structured statement with explicit Evidence references."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    finding_type: FindingType
    evidence_refs: list[str] = Field(min_length=1)
    metric_claims: list[MetricClaim]
    directional_claims: list[DirectionalClaim]
    concepts: list[str]

    @model_validator(mode="after")
    def validate_finding_shape(self) -> "Finding":
        """Require claims appropriate to the declared finding type."""

        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("evidence_refs must be unique")

        if self.finding_type is FindingType.OBSERVATION:
            if not self.metric_claims:
                raise ValueError("observations require at least one metric claim")
            if self.directional_claims or self.concepts:
                raise ValueError(
                    "observations cannot contain directional claims or concepts"
                )
            claim_refs = {claim.metric_name for claim in self.metric_claims}
            claim_keys = {
                (claim.metric_name, claim.effect_scale) for claim in self.metric_claims
            }
            if len(claim_keys) != len(self.metric_claims):
                raise ValueError("observation metric claims must be unique")
        else:
            if self.metric_claims:
                raise ValueError("interpretations cannot contain metric claims")
            if not self.directional_claims:
                raise ValueError("interpretations require a directional claim")
            claim_refs = {claim.metric_name for claim in self.directional_claims}
            if len(claim_refs) != len(self.directional_claims):
                raise ValueError("interpretation directional claims must be unique")

        if not claim_refs.issubset(self.evidence_refs):
            raise ValueError("every structured claim must appear in evidence_refs")
        return self


class FindingBatch(BaseModel):
    """Structured output returned by the LLM analyst."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(min_length=1)
