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
from .findings import (
    Direction,
    DirectionalClaim,
    EffectScale,
    Finding,
    FindingBatch,
    FindingType,
    MetricClaim,
)
from .policy import Decision, Recommendation, recommend
from .statistics import analyze_experiment
from .synthetic import Observations, generate_shipping_threshold_experiment
from .verifier import (
    VerificationStatus,
    VerifiedFinding,
    verify_finding,
    verify_findings,
)

__all__ = [
    "Evidence",
    "EvidenceClassification",
    "Direction",
    "DirectionalClaim",
    "EffectScale",
    "Finding",
    "FindingBatch",
    "FindingType",
    "MetricEvidence",
    "MetricClaim",
    "MetricRole",
    "MetricSpec",
    "MetricType",
    "Observations",
    "Decision",
    "Recommendation",
    "VerificationStatus",
    "VerifiedFinding",
    "analyze_experiment",
    "classify_effect",
    "format_analysis_input",
    "generate_shipping_threshold_experiment",
    "generate_findings",
    "recommend",
    "verify_finding",
    "verify_findings",
]
