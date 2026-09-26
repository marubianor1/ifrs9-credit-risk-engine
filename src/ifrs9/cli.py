"""Command line interface for the IFRS 9 project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ifrs9.ingestion.freddie import run_freddie_ingestion


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-freddie":
        run_freddie_ingestion(year=args.year, all_years=args.all, force=args.force)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
