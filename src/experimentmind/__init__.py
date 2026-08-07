"""ExperimentMind's deterministic experimentation foundation."""

from .classification import classify_effect
from .evidence import (
    Evidence,
    EvidenceClassification,
    MetricEvidence,
    MetricSpec,
    MetricType,
)
from .statistics import analyze_experiment
from .synthetic import Observations, generate_shipping_threshold_experiment

__all__ = [
    "Evidence",
    "EvidenceClassification",
    "MetricEvidence",
    "MetricSpec",
    "MetricType",
    "Observations",
    "analyze_experiment",
    "classify_effect",
    "generate_shipping_threshold_experiment",
]
