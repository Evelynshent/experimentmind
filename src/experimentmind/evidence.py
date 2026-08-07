"""Immutable statistical evidence produced from experiment observations."""

from dataclasses import dataclass
from enum import Enum


class MetricType(str, Enum):
    """The sampling model used to analyze a metric."""

    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class MetricEvidence:
    """Authoritative result for one treatment-versus-control comparison.

    The confidence interval is for the absolute effect (treatment minus
    control). ``relative_effect`` is ``None`` when the control value is zero.
    """

    metric_name: str
    metric_type: MetricType
    control_value: float
    treatment_value: float
    absolute_effect: float
    relative_effect: float | None
    confidence_interval: tuple[float, float]
    confidence_level: float
    p_value: float
    sample_size_control: int
    sample_size_treatment: int


@dataclass(frozen=True)
class Evidence:
    """Immutable quantitative source of truth for an experiment."""

    experiment_name: str
    metrics: tuple[MetricEvidence, ...]
    alpha: float
