"""Command-line walkthrough of the complete ExperimentMind pipeline."""

import argparse
from collections.abc import Sequence

from .analyst import generate_findings
from .hypotheses import generate_hypotheses, verify_hypotheses
from .investigation import investigate
from .policy import recommend, recommend_after_investigation
from .report import render_investigation_report, render_report
from .scenarios import ScenarioName, generate_scenario
from .statistics import analyze_experiment, analyze_scenario
from .synthetic import generate_shipping_threshold_experiment
from .verifier import verify_findings


def build_parser() -> argparse.ArgumentParser:
    """Build the small public command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run a synthetic ExperimentMind readout or investigation."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users-per-variant", type=int, default=10_000)
    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in ScenarioName],
        help="Run a V2 investigation scenario. Omit to preserve the V1 demo.",
    )
    parser.add_argument(
        "--model",
        help=(
            "OpenAI model for live structured findings, bounded planning, "
            "and hypotheses. Omit for a fully offline deterministic report."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and print its Markdown report to standard output."""

    args = build_parser().parse_args(argv)
    if args.scenario is not None:
        scenario = generate_scenario(
            args.scenario,
            seed=args.seed,
            users_per_variant=args.users_per_variant,
        )
        evidence = analyze_scenario(scenario)
        investigation = investigate(
            scenario,
            evidence,
            planner_model=args.model,
        )
        recommendation = recommend_after_investigation(
            evidence, segmentation=investigation.segmentation
        )
        verified_hypotheses = ()
        if args.model is not None and investigation.plan.selected is not None:
            hypotheses = generate_hypotheses(
                scenario, evidence, investigation, model=args.model
            )
            verified_hypotheses = verify_hypotheses(
                hypotheses.hypotheses, evidence, investigation
            )
        print(
            render_investigation_report(
                evidence,
                investigation,
                verified_hypotheses,
                recommendation,
            ),
            end="",
        )
        return 0

    observations = generate_shipping_threshold_experiment(
        seed=args.seed, users_per_variant=args.users_per_variant
    )
    evidence = analyze_experiment(observations)
    recommendation = recommend(evidence)

    verified_findings = ()
    if args.model is not None:
        findings = generate_findings(evidence, recommendation, model=args.model)
        verified_findings = verify_findings(findings.findings, evidence)

    print(render_report(evidence, verified_findings, recommendation), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
