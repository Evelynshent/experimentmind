import json
from types import SimpleNamespace

from experimentmind.findings import Direction
from experimentmind.hypotheses import (
    Hypothesis,
    HypothesisBatch,
    HypothesisDirectionalClaim,
    HypothesisStatus,
    generate_hypotheses,
    verify_hypothesis,
)
from experimentmind.investigation import investigate
from experimentmind.scenarios import ScenarioName, generate_scenario
from experimentmind.statistics import analyze_scenario


def context():
    scenario = generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY)
    evidence = analyze_scenario(scenario)
    return scenario, evidence, investigate(scenario, evidence)


def hypothesis(*, statement: str, ref: str, direction: Direction) -> Hypothesis:
    return Hypothesis(
        statement=statement,
        supporting_evidence=[ref],
        contradicting_evidence=[],
        directional_claims=[
            HypothesisDirectionalClaim(evidence_ref=ref, direction=direction)
        ],
        evidence_needed=["A confirmatory interaction test in a new experiment."],
    )


def test_matching_hypothesis_is_only_consistent_not_verified() -> None:
    _, evidence, investigation = context()
    checked = verify_hypothesis(
        hypothesis(
            statement="The ranking change may work differently for new users.",
            ref="segment:user_tenure:new:revenue_per_session",
            direction=Direction.INCREASED,
        ),
        evidence,
        investigation,
    )

    assert checked.status is HypothesisStatus.CONSISTENT_WITH_EVIDENCE
    assert "does not establish" in checked.details[0]


def test_wrong_direction_is_contradicted() -> None:
    _, evidence, investigation = context()
    checked = verify_hypothesis(
        hypothesis(
            statement="Revenue may decrease for new users.",
            ref="segment:user_tenure:new:revenue_per_session",
            direction=Direction.DECREASED,
        ),
        evidence,
        investigation,
    )

    assert checked.status is HypothesisStatus.CONTRADICTED


def test_missing_reference_has_insufficient_evidence() -> None:
    _, evidence, investigation = context()
    checked = verify_hypothesis(
        hypothesis(
            statement="Profit may explain the response.",
            ref="top_line:profit",
            direction=Direction.INCREASED,
        ),
        evidence,
        investigation,
    )

    assert checked.status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_certain_causal_language_has_insufficient_evidence() -> None:
    _, evidence, investigation = context()
    checked = verify_hypothesis(
        hypothesis(
            statement="Limited preference history causes the new-user effect.",
            ref="segment:user_tenure:new:revenue_per_session",
            direction=Direction.INCREASED,
        ),
        evidence,
        investigation,
    )

    assert checked.status is HypothesisStatus.INSUFFICIENT_EVIDENCE


def test_uncertainty_word_must_match_a_word_boundary() -> None:
    _, evidence, investigation = context()
    checked = verify_hypothesis(
        hypothesis(
            statement="Payment history causes the new-user effect.",
            ref="segment:user_tenure:new:revenue_per_session",
            direction=Direction.INCREASED,
        ),
        evidence,
        investigation,
    )

    assert checked.status is HypothesisStatus.INSUFFICIENT_EVIDENCE


class FakeResponses:
    def __init__(self, output: HypothesisBatch) -> None:
        self.output = output
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(output_parsed=self.output)


class FakeClient:
    def __init__(self, output: HypothesisBatch) -> None:
        self.responses = FakeResponses(output)


def test_hypothesis_generation_receives_intervention_context() -> None:
    scenario, evidence, investigation = context()
    item = hypothesis(
        statement="The ranking may interact with limited preference history.",
        ref="segment:user_tenure:new:revenue_per_session",
        direction=Direction.INCREASED,
    )
    expected = HypothesisBatch(hypotheses=[item, item.model_copy()])
    client = FakeClient(expected)

    actual = generate_hypotheses(
        scenario,
        evidence,
        investigation,
        model="test-model",
        client=client,
    )

    assert actual is expected
    payload = json.loads(client.responses.call["input"][1]["content"])
    assert payload["experiment_hypothesis"] == scenario.hypothesis
    assert payload["treatment_description"] == scenario.treatment_description


def test_decomposition_hypothesis_context_marks_conditional_value_descriptive() -> None:
    scenario = generate_scenario(ScenarioName.SHIPPING_TRADEOFF)
    evidence = analyze_scenario(scenario)
    investigation = investigate(scenario, evidence)
    ref = "decomposition:revenue_per_converted_session"
    item = hypothesis(
        statement="Converted-session value may decrease after treatment.",
        ref=ref,
        direction=Direction.DECREASED,
    )
    client = FakeClient(HypothesisBatch(hypotheses=[item, item.model_copy()]))

    generate_hypotheses(
        scenario,
        evidence,
        investigation,
        model="test-model",
        client=client,
    )

    payload = json.loads(client.responses.call["input"][1]["content"])
    conditional = payload["evidence"][ref]
    assert conditional["classification"] == "descriptive_only"
    assert conditional["interpretation_scope"] == (
        "post_treatment_conditioned_descriptive"
    )
