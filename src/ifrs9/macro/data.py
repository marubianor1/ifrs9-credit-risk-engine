"""Public macroeconomic data loading and quarterly transformation."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from ifrs9.macro.config import ForwardLookingConfig, MacroSeriesConfig

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def build_macro_dataset(repo_root: Path, config: ForwardLookingConfig) -> pd.DataFrame:
    """Download, cache, transform, and align public macro series to quarters."""
    macro_config = config.macro
    cache_dir = repo_root / macro_config.raw_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    quarterly: list[pd.DataFrame] = []
    for name, series_config in macro_config.series.items():
        raw = _load_fred_series(cache_dir, series_config.fred_id)
        quarterly.append(_transform_series(name, raw, series_config))
    merged = quarterly[0]
    for frame in quarterly[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date").ffill()
    merged = merged[merged["date"] >= pd.Timestamp(macro_config.start_date)]
    merged = _add_derived_features(merged)
    output = repo_root / macro_config.transformed_path
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged.reset_index(drop=True)


def load_macro_dataset(repo_root: Path, config: ForwardLookingConfig) -> pd.DataFrame:
    """Load cached quarterly macro data, building it when needed."""
    path = repo_root / config.macro.transformed_path
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"])
        return frame.sort_values("date").reset_index(drop=True)
    return build_macro_dataset(repo_root, config)


def _load_fred_series(cache_dir: Path, series_id: str) -> pd.DataFrame:
    path = cache_dir / f"{series_id}.csv"
    if not path.exists():
        urlretrieve(FRED_CSV_URL.format(series_id=series_id), path)
    frame = pd.read_csv(path)
    date_column = frame.columns[0]
    value_column = frame.columns[1]
    frame = frame.rename(columns={date_column: "date", value_column: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    return frame.dropna(subset=["value"])


def _transform_series(
    name: str,
    frame: pd.DataFrame,
    config: MacroSeriesConfig,
) -> pd.DataFrame:
    series = frame.set_index("date")["value"].sort_index()
    if config.aggregation == "mean":
        quarterly = series.resample("QE").mean()
    else:
        quarterly = series.resample("QE").last()
    if config.transform == "yoy_pct":
        transformed = quarterly.pct_change(4) * 100
        column = f"{name}_yoy"
    else:
        transformed = quarterly
        column = name
    return transformed.rename(column).reset_index()


def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "unemployment_rate" in output.columns:
        output["unemployment_rate_change"] = output["unemployment_rate"].diff()
    if "mortgage_rate" in output.columns:
        output["mortgage_rate_change"] = output["mortgage_rate"].diff()
    return output
