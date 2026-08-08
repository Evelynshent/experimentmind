from dataclasses import replace

from experimentmind.evidence import EvidenceClassification
from experimentmind.findings import (
    Direction,
    DirectionalClaim,
    EffectScale,
    Finding,
    FindingType,
    MetricClaim,
)
from experimentmind.policy import recommend
from experimentmind.report import render_report
from experimentmind.statistics import analyze_experiment
from experimentmind.synthetic import generate_shipping_threshold_experiment
from experimentmind.verifier import VerificationStatus, verify_findings


def report_inputs():
    evidence = analyze_experiment(generate_shipping_threshold_experiment(seed=42))
    conversion = next(
        metric for metric in evidence.metrics if metric.metric_name == "conversion_rate"
    )
    assert conversion.relative_effect is not None
    findings = [
        Finding(
            statement="Conversion increased significantly.",
            finding_type=FindingType.OBSERVATION,
            evidence_refs=["conversion_rate"],
            metric_claims=[
                MetricClaim(
                    metric_name="conversion_rate",
                    effect_scale=EffectScale.RELATIVE,
                    effect_value=conversion.relative_effect,
                    statistically_significant=True,
                    classification=EvidenceClassification.CLEARLY_POSITIVE,
                )
            ],
            directional_claims=[],
            concepts=[],
        ),
        Finding(
            statement="Higher shipping costs may reduce profitability.",
            finding_type=FindingType.INTERPRETATION,
            evidence_refs=["shipping_cost_per_session"],
            metric_claims=[],
            directional_claims=[
                DirectionalClaim(
                    metric_name="shipping_cost_per_session",
                    direction=Direction.INCREASED,
                )
            ],
            concepts=["shipping_cost_per_session", "profitability"],
        ),
    ]
    return evidence, verify_findings(findings, evidence), recommend(evidence)


def test_report_separates_facts_findings_and_decision() -> None:
    evidence, verified, recommendation = report_inputs()

    report = render_report(evidence, verified, recommendation)

    assert report.startswith(f"# ExperimentMind — {evidence.experiment_name}\n")
    assert "## Evidence — computed deterministically" in report
    assert "`conversion_rate` | secondary | 0.0355 | 0.0418 | +0.0063" in report
    assert "+17.75%" in report
    assert "0.0208" in report
    assert "clearly_positive" in report
    assert "## Findings — AI-generated, structurally verified" in report
    assert "### 1. ✓ VERIFIED" in report
    assert "### 2. ? INSUFFICIENT EVIDENCE" in report
    assert "## Recommendation — deterministic policy" in report
    assert "**TRADEOFF**" in report


def test_report_contains_complete_verification_counts() -> None:
    evidence, verified, recommendation = report_inputs()

    report = render_report(evidence, verified, recommendation)

    assert "- ✓ VERIFIED: 1" in report
    assert "- ? INSUFFICIENT EVIDENCE: 1" in report
    for status in (
        VerificationStatus.INCORRECT,
        VerificationStatus.UNRESOLVED,
        VerificationStatus.CONSISTENT_WITH_EVIDENCE,
        VerificationStatus.CONTRADICTED_BY_EVIDENCE,
    ):
        label = status.value.replace("_", " ").upper()
        assert f"{label}: 0" in report


def test_report_displays_undefined_relative_effect_as_not_available() -> None:
    evidence, verified, recommendation = report_inputs()
    revenue = evidence.metrics[0]
    evidence_without_relative = replace(
        evidence,
        metrics=(replace(revenue, relative_effect=None), *evidence.metrics[1:]),
    )

    report = render_report(evidence_without_relative, verified, recommendation)
    revenue_row = next(
        line for line in report.splitlines() if "`revenue_per_session`" in line
    )

    assert "| N/A |" in revenue_row


def test_report_handles_no_findings_and_is_deterministic() -> None:
    evidence, _, recommendation = report_inputs()

    first = render_report(evidence, (), recommendation)
    second = render_report(evidence, (), recommendation)

    assert first == second
    assert "No findings were supplied." in first
    assert "- ✓ VERIFIED: 0" in first
    assert first.endswith("- Recommendation: deterministic policy output\n")
