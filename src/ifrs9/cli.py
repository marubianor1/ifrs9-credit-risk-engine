"""Command line interface for the IFRS 9 project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ifrs9.development.factory import run_development_sample_build
from ifrs9.ingestion.freddie import run_freddie_ingestion
from ifrs9.mart.loan_month import run_point_in_time_mart_build
from ifrs9.targets.factory import run_target_build
from ifrs9.transformations.freddie_silver import run_freddie_silver_build


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(prog="ifrs9")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-freddie", help="Ingest Freddie Mac samples to Bronze")
    selection = ingest.add_mutually_exclusive_group(required=True)
    selection.add_argument("--year", type=int, help="Single Freddie Mac sample year to ingest")
    selection.add_argument("--all", action="store_true", help="Ingest all discovered sample years")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing successful partitions",
    )

    silver = subparsers.add_parser("build-silver", help="Build Freddie Mac Silver layer")
    silver_selection = silver.add_mutually_exclusive_group(required=True)
    silver_selection.add_argument("--year", type=int, help="Single Freddie Mac year to transform")
    silver_selection.add_argument("--all", action="store_true", help="Transform all Bronze years")
    silver.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing successful Silver partitions",
    )

    mart = subparsers.add_parser("build-mart", help="Build point-in-time Gold mart")
    mart_selection = mart.add_mutually_exclusive_group(required=True)
    mart_selection.add_argument("--year", type=int, help="Single Freddie Mac year to build")
    mart_selection.add_argument("--all", action="store_true", help="Build all Silver years")
    mart.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing successful Gold partitions",
    )

    targets = subparsers.add_parser("build-targets", help="Build default and PD target tables")
    targets_selection = targets.add_mutually_exclusive_group(required=True)
    targets_selection.add_argument("--all", action="store_true", help="Build all target tables")
    targets.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target outputs",
    )

    development = subparsers.add_parser(
        "build-development-sample",
        help="Build temporal development sample metadata",
    )
    development.add_argument("--config", type=Path, help="Path to development sample YAML")
    development.add_argument(
        "--snapshot-frequency",
        choices=["monthly", "quarter_end", "year_end"],
        help="Override configured snapshot frequency",
    )
    development.add_argument(
        "--sampling-strategy",
        choices=["none", "random_nondefault", "stratified_nondefault"],
        help="Override configured non-default sampling strategy",
    )
    development.add_argument(
        "--population",
        choices=["application", "behavioural"],
        help="Override configured population",
    )
    development.add_argument(
        "--loan-disjoint",
        action="store_true",
        help="Enable loan-disjoint sensitivity mode",
    )
    development.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing development sample output",
    )

    scorecard = subparsers.add_parser("train-scorecard", help="Train logistic scorecard")
    scorecard.add_argument("--config", type=Path, help="Path to scorecard YAML")
    scorecard.add_argument(
        "--population",
        choices=["application", "behavioural"],
        required=True,
        help="Scorecard population to train",
    )
    scorecard.add_argument(
        "--snapshot-frequency",
        choices=["monthly", "quarter_end", "year_end"],
        help="Override configured snapshot frequency",
    )
    scorecard.add_argument(
        "--sampling-strategy",
        choices=["none", "random_nondefault", "stratified_nondefault"],
        help="Override configured sampling strategy",
    )
    scorecard.add_argument("--run-id", help="Optional deterministic run ID")

    pd_parser = subparsers.add_parser("build-pd", help="Build calibrated PD framework artifacts")
    pd_parser.add_argument("--config", type=Path, help="Path to PD framework YAML")
    pd_parser.add_argument(
        "--scorecard-run",
        default=None,
        help="Parent scorecard run ID. Defaults to the configured primary run.",
    )
    pd_parser.add_argument("--run-id", help="Optional deterministic PD run ID")
    pd_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing PD run with the same run ID",
    )

    forward = subparsers.add_parser(
        "build-forward-looking",
        help="Build forward-looking PD scenario artifacts",
    )
    forward.add_argument("--config", type=Path, help="Path to forward-looking YAML")
    forward.add_argument(
        "--pd-run",
        default=None,
        help="Parent PD run ID. Defaults to the configured parent PD run.",
    )
    forward.add_argument("--run-id", help="Optional deterministic forward-looking run ID")
    forward.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing forward-looking run with the same run ID",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-freddie":
        run_freddie_ingestion(year=args.year, all_years=args.all, force=args.force)
        return 0
    if args.command == "build-silver":
        run_freddie_silver_build(year=args.year, all_years=args.all, force=args.force)
        return 0
    if args.command == "build-mart":
        run_point_in_time_mart_build(year=args.year, all_years=args.all, force=args.force)
        return 0
    if args.command == "build-targets":
        run_target_build(all_targets=args.all, force=args.force)
        return 0
    if args.command == "build-development-sample":
        run_development_sample_build(
            force=args.force,
            config_path=args.config,
            snapshot_frequency=args.snapshot_frequency,
            sampling_strategy=args.sampling_strategy,
            population=args.population,
            loan_disjoint=True if args.loan_disjoint else None,
        )
        return 0
    if args.command == "train-scorecard":
        from ifrs9.models.scorecard.runner import run_scorecard

        run_scorecard(
            config_path=args.config,
            population=args.population,
            snapshot_frequency=args.snapshot_frequency,
            sampling_strategy=args.sampling_strategy,
            run_id=args.run_id,
        )
        return 0
    if args.command == "build-pd":
        from ifrs9.pd.framework import run_pd_framework

        run_pd_framework(
            config_path=args.config,
            scorecard_run_id=args.scorecard_run,
            run_id=args.run_id,
            force=args.force,
        )
        return 0
    if args.command == "build-forward-looking":
        from ifrs9.macro.forward_looking import run_forward_looking

        run_forward_looking(
            config_path=args.config,
            pd_run_id=args.pd_run,
            run_id=args.run_id,
            force=args.force,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
