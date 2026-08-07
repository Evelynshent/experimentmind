"""ExperimentMind's deterministic experimentation foundation."""

from .analyst import format_analysis_input, generate_findings
from .classification import classify_effect
from .evidence import (
    Evidence,
    EvidenceClassification,
    MetricEvidence,
    MetricRole,
    MetricSpec,
    MetricType,
)
from .findings import Finding, FindingBatch, FindingType
from .policy import Decision, Recommendation, recommend
from .statistics import analyze_experiment
from .synthetic import Observations, generate_shipping_threshold_experiment

__all__ = [
    "Evidence",
    "EvidenceClassification",
    "Finding",
    "FindingBatch",
    "FindingType",
    "MetricEvidence",
    "MetricRole",
    "MetricSpec",
    "MetricType",
    "Observations",
    "Decision",
    "Recommendation",
    "analyze_experiment",
    "classify_effect",
    "format_analysis_input",
    "generate_shipping_threshold_experiment",
    "generate_findings",
    "recommend",
]
