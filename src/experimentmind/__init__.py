"""ExperimentMind's deterministic experimentation foundation."""

from .analyses import (
    RevenueDecomposition,
    SegmentationResult,
    decompose_revenue,
    segment_metric,
)
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
from .hypotheses import (
    Hypothesis,
    HypothesisBatch,
    HypothesisStatus,
    VerifiedHypothesis,
    generate_hypotheses,
    verify_hypotheses,
    verify_hypothesis,
)
from .investigation import (
    AnalysisRequest,
    AnalysisType,
    InvestigationResult,
    SufficiencyStatus,
    assess_sufficiency,
    candidate_analyses,
    investigate,
    select_analysis,
)
from .policy import (
    Decision,
    Recommendation,
    recommend,
    recommend_after_investigation,
)
from .report import render_investigation_report, render_report
from .scenarios import (
    ExperimentScenario,
    ScenarioName,
    generate_scenario,
)
from .statistics import analyze_experiment, analyze_scenario
from .synthetic import Observations, generate_shipping_threshold_experiment
from .verifier import (
    VerificationStatus,
    VerifiedFinding,
    verify_finding,
    verify_findings,
)

__all__ = [
    "AnalysisRequest",
    "AnalysisType",
    "Decision",
    "Direction",
    "DirectionalClaim",
    "EffectScale",
    "Evidence",
    "EvidenceClassification",
    "ExperimentScenario",
    "Finding",
    "FindingBatch",
    "FindingType",
    "Hypothesis",
    "HypothesisBatch",
    "HypothesisStatus",
    "InvestigationResult",
    "MetricClaim",
    "MetricEvidence",
    "MetricRole",
    "MetricSpec",
    "MetricType",
    "Observations",
    "Recommendation",
    "RevenueDecomposition",
    "ScenarioName",
    "SegmentationResult",
    "SufficiencyStatus",
    "VerificationStatus",
    "VerifiedFinding",
    "VerifiedHypothesis",
    "analyze_experiment",
    "analyze_scenario",
    "assess_sufficiency",
    "candidate_analyses",
    "classify_effect",
    "decompose_revenue",
    "format_analysis_input",
    "generate_findings",
    "generate_hypotheses",
    "generate_scenario",
    "generate_shipping_threshold_experiment",
    "investigate",
    "recommend",
    "recommend_after_investigation",
    "render_investigation_report",
    "render_report",
    "segment_metric",
    "select_analysis",
    "verify_finding",
    "verify_findings",
    "verify_hypotheses",
    "verify_hypothesis",
]
