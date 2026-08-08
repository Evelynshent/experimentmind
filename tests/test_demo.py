from experimentmind.demo import main
from experimentmind.evidence import EvidenceClassification
from experimentmind.findings import (
    EffectScale,
    Finding,
    FindingBatch,
    FindingType,
    MetricClaim,
)


def test_offline_demo_renders_complete_deterministic_report(capsys) -> None:
    result = main(["--seed", "42"])
    output = capsys.readouterr().out

    assert result == 0
    assert output.startswith("# ExperimentMind — Free shipping threshold")
    assert "## Evidence — computed deterministically" in output
    assert "No findings were supplied." in output
    assert "## Recommendation — deterministic policy" in output
    assert "**TRADEOFF**" in output


def test_live_demo_path_verifies_structured_findings(monkeypatch, capsys) -> None:
    def fake_generate_findings(evidence, recommendation, *, model):
        assert recommendation.decision.value == "tradeoff"
        assert model == "test-model"
        conversion = next(
            metric
            for metric in evidence.metrics
            if metric.metric_name == "conversion_rate"
        )
        assert conversion.relative_effect is not None
        return FindingBatch(
            findings=[
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
                )
            ]
        )

    monkeypatch.setattr(
        "experimentmind.demo.generate_findings", fake_generate_findings
    )

    result = main(["--seed", "42", "--model", "test-model"])
    output = capsys.readouterr().out

    assert result == 0
    assert "### 1. ✓ VERIFIED" in output
    assert "Conversion increased significantly." in output
