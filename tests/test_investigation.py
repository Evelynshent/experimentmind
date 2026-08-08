import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from experimentmind.evidence import EvidenceClassification
from experimentmind.investigation import (
    AnalysisType,
    PlannerChoice,
    SufficiencyStatus,
    assess_sufficiency,
    investigate,
    select_analysis,
)
from experimentmind.policy import Decision, recommend_after_investigation
from experimentmind.scenarios import ScenarioName, generate_scenario
from experimentmind.statistics import analyze_scenario


class FakeResponses:
    def __init__(self, choice: PlannerChoice) -> None:
        self.choice = choice
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.choice)


class FakeClient:
    def __init__(self, choice: PlannerChoice) -> None:
        self.responses = FakeResponses(choice)


def run_scenario(name: ScenarioName):
    scenario = generate_scenario(name)
    evidence = analyze_scenario(scenario)
    result = investigate(scenario, evidence)
    recommendation = recommend_after_investigation(
        evidence, segmentation=result.segmentation
    )
    return result, recommendation


def test_clear_win_stops_without_investigation() -> None:
    result, recommendation = run_scenario(ScenarioName.CLEAR_WIN)

    assert result.assessment.status is SufficiencyStatus.SUFFICIENT
    assert result.candidates == ()
    assert result.plan.selected is None
    assert recommendation.decision is Decision.SHIP


def test_positive_primary_and_harmed_guardrail_are_conflicting() -> None:
    scenario = generate_scenario(ScenarioName.CLEAR_WIN)
    evidence = analyze_scenario(scenario)
    metrics = tuple(
        replace(metric, classification=EvidenceClassification.CLEARLY_NEGATIVE)
        if metric.metric_name == "shipping_cost_per_session"
        else metric
        for metric in evidence.metrics
    )

    assessment = assess_sufficiency(replace(evidence, metrics=metrics))

    assert assessment.status is SufficiencyStatus.CONFLICTING


def test_tradeoff_requests_only_revenue_decomposition() -> None:
    result, recommendation = run_scenario(ScenarioName.SHIPPING_TRADEOFF)

    assert result.assessment.status is SufficiencyStatus.CONFLICTING
    assert len(result.candidates) == 1
    assert result.plan.selected is not None
    assert result.plan.selected.analysis_type is AnalysisType.REVENUE_DECOMPOSITION
    assert result.decomposition is not None
    assert recommendation.decision is Decision.TRADEOFF


def test_single_candidate_does_not_call_optional_llm() -> None:
    scenario = generate_scenario(ScenarioName.SHIPPING_TRADEOFF)
    evidence = analyze_scenario(scenario)
    client = FakeClient(
        PlannerChoice(request_id="not-used", reason="This must not be called.")
    )

    result = investigate(scenario, evidence, planner_model="test-model", client=client)

    assert result.plan.selected is not None
    assert result.plan.selected_by_ai is False
    assert client.responses.calls == []


def test_hidden_heterogeneity_requests_segmentation_and_changes_decision() -> None:
    result, recommendation = run_scenario(ScenarioName.HIDDEN_HETEROGENEITY)

    assert result.assessment.status is SufficiencyStatus.INSUFFICIENT
    assert [candidate.analysis_type for candidate in result.candidates] == [
        AnalysisType.SEGMENTATION,
        AnalysisType.REVENUE_DECOMPOSITION,
    ]
    assert result.segmentation is not None
    assert recommendation.decision is Decision.VALIDATE_HETEROGENEITY


def test_optional_llm_can_rank_only_supplied_candidates() -> None:
    scenario = generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY)
    evidence = analyze_scenario(scenario)
    offline = investigate(scenario, evidence)
    decomposition = offline.candidates[1]
    client = FakeClient(
        PlannerChoice(
            request_id=decomposition.request_id,
            reason="The funnel identity directly tests an offsetting mechanism.",
        )
    )

    plan = select_analysis(
        offline.candidates,
        model="test-model",
        client=client,
        evidence=evidence,
        hypothesis=scenario.hypothesis,
    )

    assert plan.selected == decomposition
    assert plan.selected_by_ai is True
    assert len(client.responses.calls) == 1
    payload = json.loads(client.responses.calls[0]["input"][1]["content"])
    assert payload["experiment_hypothesis"] == scenario.hypothesis
    assert len(payload["current_evidence"]) == 3


def test_decomposition_first_continues_to_unresolved_segmentation() -> None:
    scenario = generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY)
    evidence = analyze_scenario(scenario)
    client = FakeClient(
        PlannerChoice(
            request_id="decompose:revenue_per_session",
            reason="Inspect the funnel identity first.",
        )
    )

    result = investigate(scenario, evidence, planner_model="test-model", client=client)
    recommendation = recommend_after_investigation(
        evidence, segmentation=result.segmentation
    )

    assert [request.analysis_type for request in result.executed_requests] == [
        AnalysisType.REVENUE_DECOMPOSITION,
        AnalysisType.SEGMENTATION,
    ]
    assert result.decomposition is not None
    assert result.segmentation is not None
    assert recommendation.decision is Decision.VALIDATE_HETEROGENEITY


def test_optional_llm_cannot_invent_an_analysis() -> None:
    scenario = generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY)
    evidence = analyze_scenario(scenario)
    candidates = investigate(scenario, evidence).candidates
    client = FakeClient(
        PlannerChoice(request_id="run:arbitrary_sql", reason="Try something else.")
    )

    with pytest.raises(ValueError, match="outside the valid candidates"):
        select_analysis(candidates, model="test-model", client=client)
