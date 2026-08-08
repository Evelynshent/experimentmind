"""Deterministic Markdown rendering for experiment analysis results."""

from collections import Counter

from .evidence import Evidence, MetricEvidence
from .hypotheses import VerifiedHypothesis
from .investigation import InvestigationResult
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
        interval = (
            f"[{_number(ci_lower, signed=True)}, {_number(ci_upper, signed=True)}]"
        )
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


def _finding_lines(verified_findings: tuple[VerifiedFinding, ...]) -> list[str]:
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


def _verification_summary(verified_findings: tuple[VerifiedFinding, ...]) -> list[str]:
    counts = Counter(finding.status for finding in verified_findings)
    return [
        f"- {STATUS_LABELS[status]}: {counts[status]}" for status in VerificationStatus
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


def render_investigation_report(
    evidence: Evidence,
    investigation: InvestigationResult,
    hypotheses: tuple[VerifiedHypothesis, ...],
    recommendation: Recommendation,
) -> str:
    """Render the V2 investigation stages without recomputing evidence."""

    lines = [
        f"# ExperimentMind V2 — {evidence.experiment_name}",
        "",
        "## Initial evidence — computed deterministically",
        "",
        *_evidence_table(evidence),
        "",
        "## Evidence sufficiency — deterministic",
        "",
        f"**{investigation.assessment.status.value.upper()}**",
        "",
        *(f"- {reason}" for reason in investigation.assessment.reasons),
        "",
        "## Investigation plan — bounded",
        "",
    ]
    if investigation.candidates:
        lines.extend(
            f"- Candidate `{candidate.request_id}`: {candidate.reason}"
            for candidate in investigation.candidates
        )
    else:
        if investigation.assessment.status.value == "sufficient":
            lines.append("No additional analysis is scientifically necessary.")
        else:
            lines.append("No supported candidate analysis is available.")

    if investigation.plan.selected is not None:
        source = (
            "optional AI ranking"
            if investigation.plan.selected_by_ai
            else "deterministic selection"
        )
        lines.extend(
            [
                "",
                f"Selected `{investigation.plan.selected.request_id}` via {source}.",
                f"Reason: {investigation.plan.selection_reason}",
            ]
        )
        if len(investigation.executed_requests) > 1:
            follow_up = investigation.executed_requests[1]
            lines.append(
                f"Follow-up `{follow_up.request_id}` ran deterministically because "
                "the first analysis did not resolve the decision uncertainty."
            )

    lines.extend(["", "## Additional evidence — computed deterministically", ""])
    rendered_analysis = False
    if investigation.segmentation is not None:
        rendered_analysis = True
        lines.extend(
            [
                f"Segmentation by `{investigation.segmentation.dimension}`:",
                "",
                "| Segment | Control n | Treatment n | Control | Treatment | Effect | CI | p-value | Classification |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for segment in investigation.segmentation.segments:
            metric = segment.metric
            lower, upper = metric.confidence_interval
            lines.append(
                "| "
                + " | ".join(
                    [
                        segment.segment,
                        str(metric.sample_size_control),
                        str(metric.sample_size_treatment),
                        _number(metric.control_value),
                        _number(metric.treatment_value),
                        _number(metric.absolute_effect, signed=True),
                        f"[{_number(lower, signed=True)}, {_number(upper, signed=True)}]",
                        _p_value(metric.p_value),
                        metric.classification.value,
                    ]
                )
                + " |"
            )
    if investigation.decomposition is not None:
        rendered_analysis = True
        decomposition = investigation.decomposition
        if investigation.segmentation is not None:
            lines.extend(["", "Revenue decomposition:", ""])
        lines.extend(
            [
                "`revenue_per_session = conversion_rate × revenue_per_converted_session`",
                "",
                "| Component | Control | Treatment | Effect | Classification |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for metric in (
            decomposition.conversion_rate,
            decomposition.revenue_per_converted_session,
        ):
            classification = metric.classification.value
            if metric.metric_name == "revenue_per_converted_session":
                classification = "descriptive_only"
            lines.append(
                "| "
                + " | ".join(
                    [
                        metric.metric_name,
                        _number(metric.control_value),
                        _number(metric.treatment_value),
                        _number(metric.absolute_effect, signed=True),
                        classification,
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Recomposed control revenue: {_number(decomposition.control_recomposed_revenue)}",
                f"- Recomposed treatment revenue: {_number(decomposition.treatment_recomposed_revenue)}",
                f"- Control arithmetic residual: {_number(decomposition.control_residual, signed=True)}",
                f"- Treatment arithmetic residual: {_number(decomposition.treatment_residual, signed=True)}",
                "- `revenue_per_converted_session` conditions on a post-treatment outcome; its interval and classification are descriptive, not randomized causal evidence.",
            ]
        )
    if not rendered_analysis:
        lines.append("No additional evidence was requested.")

    lines.extend(["", "## Competing hypotheses — AI-generated, not verified facts", ""])
    if hypotheses:
        for index, verified in enumerate(hypotheses, start=1):
            hypothesis = verified.hypothesis
            lines.extend(
                [
                    f"### {index}. {verified.status.value.upper()}",
                    "",
                    hypothesis.statement,
                    "",
                    f"- Supporting evidence: {', '.join(hypothesis.supporting_evidence) or 'none'}",
                    f"- Contradicting evidence: {', '.join(hypothesis.contradicting_evidence) or 'none'}",
                    f"- Evidence needed: {', '.join(hypothesis.evidence_needed)}",
                    *(f"- Check: {detail}" for detail in verified.details),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No hypotheses were supplied.",
                "",
                "A hypothesis that is consistent with evidence is not causally established.",
            ]
        )

    lines.extend(
        [
            "",
            "## Final recommendation — deterministic policy",
            "",
            f"**{recommendation.decision.value.upper()}**",
            "",
            *(f"- {reason}" for reason in recommendation.rationale),
            "",
            "## Epistemic boundary",
            "",
            "- Initial and additional evidence: computed facts",
            "- Analysis candidates: deterministically constrained",
            "- Hypotheses: possible explanations, never causal verification",
            "- Recommendation: explicit deterministic policy",
            "- Consequential decisions: require human review",
        ]
    )
    return "\n".join(lines) + "\n"
