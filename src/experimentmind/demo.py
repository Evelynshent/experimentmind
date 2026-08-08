"""Command-line walkthrough of the complete ExperimentMind pipeline."""

import argparse
from collections.abc import Sequence

from .analyst import generate_findings
from .policy import recommend
from .report import render_report
from .statistics import analyze_experiment
from .synthetic import generate_shipping_threshold_experiment
from .verifier import verify_findings


def build_parser() -> argparse.ArgumentParser:
    """Build the small public command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run the synthetic free-shipping-threshold experiment."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users-per-variant", type=int, default=10_000)
    parser.add_argument(
        "--model",
        help=(
            "OpenAI model for live structured findings. Omit for a fully "
            "offline deterministic report."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and print its Markdown report to standard output."""

    args = build_parser().parse_args(argv)
    observations = generate_shipping_threshold_experiment(
        seed=args.seed, users_per_variant=args.users_per_variant
    )
    evidence = analyze_experiment(observations)
    recommendation = recommend(evidence)

    verified_findings = ()
    if args.model is not None:
        findings = generate_findings(
            evidence, recommendation, model=args.model
        )
        verified_findings = verify_findings(findings.findings, evidence)

    print(render_report(evidence, verified_findings, recommendation), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
