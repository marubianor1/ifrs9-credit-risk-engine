"""Load and validate Freddie Mac source schema metadata."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

DatasetKind = Literal["origination", "performance"]


class SchemaRegistryError(ValueError):
    """Raised when schema metadata cannot be loaded or selected."""


class FieldDefinition(BaseModel):
    """Metadata for one source column in a pipe-delimited Freddie Mac file."""

    position: int = Field(ge=1)
    source_name: str
    canonical_name: str
    description: str
    source_type: str
    target_type: str
    nullable: bool
    missing_codes: list[str] = Field(default_factory=list)
    valid_values: dict[str, str] | list[str] | None = None
    date_format: str | None = None
    category: str
    availability: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def support_legacy_name_key(cls, data: object) -> object:
        """Support schema files that used `name` before `source_name` was adopted."""
        if isinstance(data, dict) and "source_name" not in data and "name" in data:
            data = {**data, "source_name": data["name"]}
        return data

    @property
    def name(self) -> str:
        """Return the Freddie Mac source name for backward-compatible callers."""
        return self.source_name

    @field_validator("canonical_name")
    @classmethod
    def canonical_name_must_be_snake_case(cls, value: str) -> str:
        """Validate that canonical names are analytics-friendly identifiers."""
        if not value.isidentifier() or value.lower() != value:
            msg = f"Invalid canonical_name: {value}"
            raise ValueError(msg)
        return value


class SourceSchema(BaseModel):
    """A versioned schema for one Freddie Mac source file layout."""

    schema_id: str
    dataset: DatasetKind
    source: str
    delimiter: str = "|"
    header: bool = False
    version: str
    expected_column_count: int = Field(ge=1)
    guide_reference: str
    columns: list[FieldDefinition]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_positions_and_count(self) -> SourceSchema:
        """Validate contiguous positions and expected column count."""
        positions = [column.position for column in self.columns]
        expected_positions = list(range(1, len(self.columns) + 1))
        if positions != expected_positions:
            msg = f"{self.schema_id} has non-contiguous field positions: {positions}"
            raise ValueError(msg)
        if len(self.columns) != self.expected_column_count:
            msg = (
                f"{self.schema_id} expected {self.expected_column_count} columns, "
                f"but defines {len(self.columns)}"
            )
            raise ValueError(msg)
        return self


class SchemaVersionMapping(BaseModel):
    """Year-to-schema mapping entry."""

    dataset: DatasetKind
    schema_id: str
    year_start: int
    year_end: int
    observed_column_count: int

    def supports(self, dataset: DatasetKind, year: int) -> bool:
        """Return whether this mapping supports a dataset and year."""
        return self.dataset == dataset and self.year_start <= year <= self.year_end


class SchemaVersions(BaseModel):
    """Version registry for Freddie Mac source schemas."""

    schema_versions: list[SchemaVersionMapping]


def find_repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root by walking upward from a starting path."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "config").is_dir():
            return candidate
    msg = f"Could not locate repository root from {current}"
    raise SchemaRegistryError(msg)


def _read_yaml(path: Path) -> object:
    with path.open() as stream:
        return yaml.safe_load(stream)


@lru_cache(maxsize=32)
def _load_schema_by_id(schema_id: str, repo_root: str) -> SourceSchema:
    schema_path = Path(repo_root) / "config" / "schemas" / f"{schema_id}.yaml"
    if not schema_path.exists():
        msg = f"Schema file not found: {schema_path}"
        raise SchemaRegistryError(msg)
    return SourceSchema.model_validate(_read_yaml(schema_path))


def load_schema(
    dataset: DatasetKind,
    year: int,
    repo_root: Path | None = None,
) -> SourceSchema:
    """Load the schema metadata for a Freddie Mac dataset and vintage year."""
    root = find_repo_root(repo_root)
    versions_path = root / "config" / "schemas" / "freddie_schema_versions.yaml"
    if not versions_path.exists():
        msg = f"Schema version registry not found: {versions_path}"
        raise SchemaRegistryError(msg)

    versions = SchemaVersions.model_validate(_read_yaml(versions_path))
    for mapping in versions.schema_versions:
        if mapping.supports(dataset, year):
            schema = _load_schema_by_id(mapping.schema_id, str(root))
            if schema.dataset != dataset:
                msg = f"Schema {schema.schema_id} is for {schema.dataset}, not {dataset}"
                raise SchemaRegistryError(msg)
            return schema

    msg = f"No Freddie Mac {dataset} schema is registered for year {year}"
    raise SchemaRegistryError(msg)


__all__ = [
    "DatasetKind",
    "FieldDefinition",
    "SchemaRegistryError",
    "SchemaVersionMapping",
    "SchemaVersions",
    "SourceSchema",
    "find_repo_root",
    "load_schema",
]
