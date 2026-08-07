"""Typed, AI-generated findings that cite deterministic Evidence."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FindingType(str, Enum):
    """Whether a finding is directly checkable or interpretive."""

    OBSERVATION = "observation"
    INTERPRETATION = "interpretation"


class Finding(BaseModel):
    """One structured statement with explicit Evidence references."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    finding_type: FindingType
    evidence_refs: list[str] = Field(min_length=1)


class FindingBatch(BaseModel):
    """Structured output returned by the LLM analyst."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(min_length=1)
