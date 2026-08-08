"""Explicitly opted-in smoke test that can incur OpenAI API usage."""

import os

import pytest

from experimentmind.analyst import generate_findings
from experimentmind.policy import recommend
from experimentmind.statistics import analyze_experiment
from experimentmind.synthetic import generate_shipping_threshold_experiment
from experimentmind.verifier import verify_findings


@pytest.mark.live_api
@pytest.mark.skipif(
    os.getenv("EXPERIMENTMIND_RUN_LIVE_API") != "1",
    reason="set EXPERIMENTMIND_RUN_LIVE_API=1 to authorize API usage",
)
def test_live_structured_findings_smoke() -> None:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        pytest.fail("OPENAI_MODEL is required for the live API smoke test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required for the live API smoke test")

    evidence = analyze_experiment(generate_shipping_threshold_experiment(seed=42))
    recommendation = recommend(evidence)
    findings = generate_findings(evidence, recommendation, model=model)
    verified = verify_findings(findings.findings, evidence)

    assert findings.findings
    assert len(verified) == len(findings.findings)
