"""ExperimentMind's deterministic experimentation foundation."""

from .classification import classify_effect
from .evidence import (
    Evidence,
    EvidenceClassification,
    MetricEvidence,
    MetricRole,
    MetricSpec,
    MetricType,
)
from .policy import Decision, Recommendation, recommend
from .statistics import analyze_experiment
from .synthetic import Observations, generate_shipping_threshold_experiment

__all__ = [
    "Evidence",
    "EvidenceClassification",
    "MetricEvidence",
    "MetricRole",
    "MetricSpec",
    "MetricType",
    "Observations",
    "Decision",
    "Recommendation",
    "analyze_experiment",
    "classify_effect",
    "generate_shipping_threshold_experiment",
    "recommend",
]
