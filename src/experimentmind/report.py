"""Deterministic Markdown rendering for experiment analysis results."""

from collections import Counter

from .evidence import Evidence, MetricEvidence
from .policy import Recommendation
from .verifier import VerificationStatus, VerifiedFinding


STATUS_LABELS = {
    VerificationStatus.VERIFIED: "✓ VERIFIED",
    VerificationStatus.INCORRECT: "✗ INCORRECT",
    VerificationStatus.UNRESOLVED: "? UNRESOLVED",
    VerificationStatus.CONSISTENT_WITH_EVIDENCE: "○ CONSISTENT WITH EVIDENCE",
    VerificationStatus.CONTRADICTED_BY_EVIDENCE: "✗ CONTRADICTED BY EVIDENCE",
    VerificationStatus.INSUFFICIENT_EVIDENCE: "? INSUFFICIENT EVIDENCE",
}


def _number(value: float, *, signed: bool = False) -> str:
    return f"{value:+.6g}" if signed else f"{value:.6g}"


def _relative_effect(metric: MetricEvidence) -> str:
    if metric.relative_effect is None:
        return "N/A"
    return f"{metric.relative_effect:+.2%}"


def _p_value(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _evidence_table(evidence: Evidence) -> list[str]:
    lines = [
        "| Metric | Role | Control | Treatment | Absolute effect | Relative effect | CI | p-value | Classification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in evidence.metrics:
        ci_lower, ci_upper = metric.confidence_interval
        interval = f"[{_number(ci_lower, signed=True)}, {_number(ci_upper, signed=True)}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metric.metric_name}`",
                    metric.metric_spec.role.value,
                    _number(metric.control_value),
                    _number(metric.treatment_value),
                    _number(metric.absolute_effect, signed=True),
                    _relative_effect(metric),
                    interval,
                    _p_value(metric.p_value),
                    metric.classification.value,
                ]
            )
            + " |"
        )
    return lines


def _finding_lines(
    verified_findings: tuple[VerifiedFinding, ...]
) -> list[str]:
    if not verified_findings:
        return ["No findings were supplied."]

    lines: list[str] = []
    for index, verified in enumerate(verified_findings, start=1):
        finding = verified.finding
        references = ", ".join(f"`{ref}`" for ref in finding.evidence_refs)
        lines.extend(
            [
                f"### {index}. {STATUS_LABELS[verified.status]}",
                "",
                finding.statement,
                "",
                f"- Type: `{finding.finding_type.value}`",
                f"- Evidence: {references}",
            ]
        )
        lines.extend(f"- Check: {detail}" for detail in verified.details)
        lines.append("")
    return lines[:-1]


def _verification_summary(
    verified_findings: tuple[VerifiedFinding, ...]
) -> list[str]:
    counts = Counter(finding.status for finding in verified_findings)
    return [
        f"- {STATUS_LABELS[status]}: {counts[status]}"
        for status in VerificationStatus
    ]


def render_report(
    evidence: Evidence,
    verified_findings: tuple[VerifiedFinding, ...],
    recommendation: Recommendation,
) -> str:
    """Render facts, checked findings, and policy output as stable Markdown."""

    lines = [
        f"# ExperimentMind — {evidence.experiment_name}",
        "",
        "## Evidence — computed deterministically",
        "",
        *_evidence_table(evidence),
        "",
        f"Significance threshold: `alpha = {_number(evidence.alpha)}`.",
        "",
        "## Findings — AI-generated, structurally verified",
        "",
        *_finding_lines(verified_findings),
        "",
        "## Verification summary",
        "",
        *_verification_summary(verified_findings),
        "",
        "## Recommendation — deterministic policy",
        "",
        f"**{recommendation.decision.value.upper()}**",
        "",
        *(f"- {reason}" for reason in recommendation.rationale),
        "",
        "## Trust legend",
        "",
        "- Evidence table: computed fact",
        "- ✓: observation verified against Evidence",
        "- ○: AI interpretation consistent with Evidence, not causally established",
        "- ✗ or ?: incorrect, contradicted, unresolved, or unsupported claim",
        "- Recommendation: deterministic policy output",
    ]
    return "\n".join(lines) + "\n"
