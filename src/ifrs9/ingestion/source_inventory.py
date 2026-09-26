"""Lightweight inspection utilities for Freddie Mac sample ZIP archives."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from ifrs9.ingestion.schema_registry import find_repo_root

SAMPLE_ZIP_PATTERN = re.compile(r"sample_(?P<year>\d{4})\.zip$")


class SourceInventoryError(ValueError):
    """Raised when source archive metadata is malformed or unsupported."""


@dataclass(frozen=True)
class FileSampleProfile:
    """Sampled structure for one member of a Freddie Mac ZIP archive."""

    member_name: str | None
    sampled_column_count: int | None
    sample_rows_checked: int
    consistent: bool
    observed_column_counts: tuple[int, ...]


@dataclass(frozen=True)
class FreddieSampleArchiveProfile:
    """Lightweight metadata for one Freddie Mac annual sample archive."""

    year: int
    zip_file: Path
    members: tuple[str, ...]
    origination: FileSampleProfile
    performance: FileSampleProfile


def parse_sample_year(path: str | Path) -> int:
    """Parse the vintage year from a Freddie Mac sample ZIP filename."""
    name = Path(path).name
    match = SAMPLE_ZIP_PATTERN.fullmatch(name)
    if match is None:
        msg = f"Unsupported Freddie Mac sample ZIP filename: {name}"
        raise SourceInventoryError(msg)
    return int(match.group("year"))


def split_pipe_row(row: str) -> list[str]:
    """Split one headerless Freddie Mac pipe-delimited record."""
    return row.rstrip("\r\n").split("|")


def validate_column_count(row: str, expected_column_count: int) -> None:
    """Raise when a pipe-delimited row does not match the expected width."""
    observed = len(split_pipe_row(row))
    if observed != expected_column_count:
        msg = f"Expected {expected_column_count} columns, observed {observed}"
        raise SourceInventoryError(msg)


def expected_members_for_year(year: int) -> tuple[str, str]:
    """Return expected origination and performance member names for a sample year."""
    return f"sample_orig_{year}.txt", f"sample_perf_{year}.txt"


def identify_sample_members(members: tuple[str, ...], year: int) -> tuple[str, str]:
    """Find the expected origination and performance members in an archive."""
    orig_member, perf_member = expected_members_for_year(year)
    missing = [member for member in (orig_member, perf_member) if member not in members]
    if missing:
        msg = f"Missing expected ZIP member(s) for {year}: {', '.join(missing)}"
        raise SourceInventoryError(msg)
    return orig_member, perf_member


def _inspect_member(
    archive: zipfile.ZipFile,
    member_name: str,
    sample_size: int,
) -> FileSampleProfile:
    counts: list[int] = []
    with archive.open(member_name) as file:
        for raw_line in file:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            counts.append(len(split_pipe_row(line)))
            if len(counts) >= sample_size:
                break

    unique_counts = tuple(sorted(set(counts)))
    return FileSampleProfile(
        member_name=member_name,
        sampled_column_count=counts[0] if counts else None,
        sample_rows_checked=len(counts),
        consistent=len(unique_counts) <= 1,
        observed_column_counts=unique_counts,
    )


def inspect_sample_zip(zip_path: Path, sample_size: int = 100) -> FreddieSampleArchiveProfile:
    """Inspect a Freddie Mac sample ZIP without extracting it to disk."""
    if sample_size < 1:
        msg = "sample_size must be positive"
        raise SourceInventoryError(msg)

    year = parse_sample_year(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = tuple(archive.namelist())
        orig_member, perf_member = identify_sample_members(members, year)
        origination = _inspect_member(archive, orig_member, sample_size)
        performance = _inspect_member(archive, perf_member, sample_size)

    return FreddieSampleArchiveProfile(
        year=year,
        zip_file=zip_path,
        members=members,
        origination=origination,
        performance=performance,
    )


def load_freddie_source_dir(repo_root: Path | None = None) -> Path:
    """Load the Freddie Mac source directory from config/data.yaml."""
    root = find_repo_root(repo_root)
    config_path = root / "config" / "data.yaml"
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    source_dir = config["source_data"]["freddie_mac"]["source_dir"]
    return root / source_dir


def discover_sample_zips(source_dir: Path) -> list[Path]:
    """Discover Freddie Mac annual sample ZIP files in a source directory."""
    return sorted(
        path for path in source_dir.glob("sample_*.zip") if SAMPLE_ZIP_PATTERN.fullmatch(path.name)
    )


def inspect_available_sample_zips(
    repo_root: Path | None = None,
    sample_size: int = 100,
) -> list[FreddieSampleArchiveProfile]:
    """Inspect all configured Freddie Mac sample ZIP archives."""
    source_dir = load_freddie_source_dir(repo_root)
    return [
        inspect_sample_zip(path, sample_size=sample_size)
        for path in discover_sample_zips(source_dir)
    ]


__all__ = [
    "FileSampleProfile",
    "FreddieSampleArchiveProfile",
    "SourceInventoryError",
    "discover_sample_zips",
    "expected_members_for_year",
    "identify_sample_members",
    "inspect_available_sample_zips",
    "inspect_sample_zip",
    "load_freddie_source_dir",
    "parse_sample_year",
    "split_pipe_row",
    "validate_column_count",
]
