"""Immutable statistical evidence produced from experiment observations."""

from dataclasses import dataclass
from enum import Enum


class MetricType(str, Enum):
    """The sampling model used to analyze a metric."""

    BINARY = "binary"
    CONTINUOUS = "continuous"


class EvidenceClassification(str, Enum):
    """Deterministic interpretation of statistical and practical evidence."""

    CLEARLY_POSITIVE = "clearly_positive"
    CLEARLY_NEGATIVE = "clearly_negative"
    NEGLIGIBLE = "negligible"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class MetricSpec:
    """Decision context needed to interpret one metric's effect."""

    metric_name: str
    higher_is_better: bool
    meaningful_effect: float


@dataclass(frozen=True)
class MetricEvidence:
    """Authoritative result for one treatment-versus-control comparison.

    The confidence interval is for the absolute effect (treatment minus
    control). ``relative_effect`` is ``None`` when the control value is zero.
    """

    metric_name: str
    metric_type: MetricType
    metric_spec: MetricSpec
    control_value: float
    treatment_value: float
    absolute_effect: float
    relative_effect: float | None
    confidence_interval: tuple[float, float]
    confidence_level: float
    p_value: float
    sample_size_control: int
    sample_size_treatment: int
    classification: EvidenceClassification


@dataclass(frozen=True)
class Evidence:
    """Immutable quantitative source of truth for an experiment."""

    experiment_name: str
    metrics: tuple[MetricEvidence, ...]
    alpha: float
