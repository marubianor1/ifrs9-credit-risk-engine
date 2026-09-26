"""WOE/IV binning utilities for traditional scorecards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinRow:
    """One fitted WOE bin."""

    feature: str
    bin_label: str
    lower: float | None
    upper: float | None
    categories: tuple[str, ...] | None
    missing: bool
    events: float
    non_events: float
    total: float
    bad_rate: float
    woe: float
    iv: float


class WoeBinner:
    """Train-only WOE/IV binning with frozen transform rules."""

    def __init__(
        self,
        *,
        max_bins: int = 10,
        min_bin_pct: float = 0.02,
        enforce_monotonic: bool = True,
        rare_category_pct: float = 0.01,
        smoothing: float = 0.5,
    ) -> None:
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.enforce_monotonic = enforce_monotonic
        self.rare_category_pct = rare_category_pct
        self.smoothing = smoothing
        self.bins_: dict[str, list[BinRow]] = {}
        self.feature_iv_: dict[str, float] = {}
        self.feature_types_: dict[str, str] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> WoeBinner:
        """Fit WOE bins on TRAIN data only."""
        weights = sample_weight if sample_weight is not None else pd.Series(1.0, index=target.index)
        for feature in frame.columns:
            series = frame[feature]
            if pd.api.types.is_numeric_dtype(series):
                rows = self._fit_numeric(feature, series, target, weights)
                self.feature_types_[feature] = "numeric"
            else:
                rows = self._fit_categorical(feature, series, target, weights)
                self.feature_types_[feature] = "categorical"
            self.bins_[feature] = rows
            self.feature_iv_[feature] = float(sum(row.iv for row in rows))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen WOE bins."""
        transformed: dict[str, pd.Series] = {}
        for feature, rows in self.bins_.items():
            if self.feature_types_[feature] == "numeric":
                transformed[feature] = self._transform_numeric(frame[feature], rows)
            else:
                transformed[feature] = self._transform_categorical(frame[feature], rows)
        return pd.DataFrame(transformed, index=frame.index)

    def bin_table(self) -> pd.DataFrame:
        """Return fitted bin statistics."""
        records = []
        for rows in self.bins_.values():
            for row in rows:
                records.append(row.__dict__)
        return pd.DataFrame(records)

    def _fit_numeric(
        self,
        feature: str,
        series: pd.Series,
        target: pd.Series,
        weights: pd.Series,
    ) -> list[BinRow]:
        non_missing = series.notna()
        values = series[non_missing].astype(float)
        if values.nunique(dropna=True) <= 1:
            edges = np.array([-np.inf, np.inf])
        else:
            quantiles = np.linspace(0, 1, self.max_bins + 1)
            edges = np.unique(values.quantile(quantiles).to_numpy())
            edges[0] = -np.inf
            edges[-1] = np.inf
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
        assignments = pd.cut(series.astype(float), bins=edges, include_lowest=True, right=True)
        groups = []
        for interval in assignments[non_missing].cat.categories:
            mask = assignments == interval
            if mask.any():
                groups.append(
                    {
                        "lower": float(interval.left),
                        "upper": float(interval.right),
                        "mask": mask,
                    }
                )
        groups = self._merge_low_population(groups, len(series))
        if self.enforce_monotonic:
            groups = self._merge_until_monotonic(groups, target, weights)
        rows = self._rows_from_groups(feature, groups, target, weights)
        if series.isna().any():
            rows.extend(self._missing_rows(feature, series.isna(), target, weights))
        return rows

    def _fit_categorical(
        self,
        feature: str,
        series: pd.Series,
        target: pd.Series,
        weights: pd.Series,
    ) -> list[BinRow]:
        values = series.fillna("__MISSING__").astype(str)
        min_count = max(1, int(len(series) * self.rare_category_pct))
        counts = values.value_counts()
        grouped = values.where(values.map(counts) >= min_count, "__RARE__")
        groups = []
        for category in sorted(grouped.unique()):
            if category == "__MISSING__":
                continue
            mask = grouped == category
            categories = tuple(sorted(values[mask].unique()))
            groups.append({"categories": categories, "mask": mask, "label": category})
        rows = self._categorical_rows_from_groups(feature, groups, target, weights)
        if (grouped == "__MISSING__").any():
            rows.extend(self._missing_rows(feature, grouped == "__MISSING__", target, weights))
        return rows

    def _merge_low_population(
        self,
        groups: list[dict[str, Any]],
        total_rows: int,
    ) -> list[dict[str, Any]]:
        if not groups:
            return groups
        min_rows = max(1, int(total_rows * self.min_bin_pct))
        merged = groups[:]
        index = 0
        while len(merged) > 1 and index < len(merged):
            if int(merged[index]["mask"].sum()) >= min_rows:
                index += 1
                continue
            merge_index = index - 1 if index == len(merged) - 1 else index + 1
            left = min(index, merge_index)
            right = max(index, merge_index)
            merged[left] = {
                "lower": min(merged[left]["lower"], merged[right]["lower"]),
                "upper": max(merged[left]["upper"], merged[right]["upper"]),
                "mask": merged[left]["mask"] | merged[right]["mask"],
            }
            del merged[right]
            index = 0
        return merged

    def _merge_until_monotonic(
        self,
        groups: list[dict[str, Any]],
        target: pd.Series,
        weights: pd.Series,
    ) -> list[dict[str, Any]]:
        merged = groups[:]
        while len(merged) > 2:
            rates = [self._bad_rate(group["mask"], target, weights) for group in merged]
            increasing = all(left <= right for left, right in zip(rates, rates[1:], strict=False))
            decreasing = all(left >= right for left, right in zip(rates, rates[1:], strict=False))
            if increasing or decreasing:
                break
            diffs = [abs(left - right) for left, right in zip(rates, rates[1:], strict=False)]
            merge_at = int(np.argmin(diffs))
            merged[merge_at] = {
                "lower": min(merged[merge_at]["lower"], merged[merge_at + 1]["lower"]),
                "upper": max(merged[merge_at]["upper"], merged[merge_at + 1]["upper"]),
                "mask": merged[merge_at]["mask"] | merged[merge_at + 1]["mask"],
            }
            del merged[merge_at + 1]
        return merged

    def _rows_from_groups(
        self,
        feature: str,
        groups: list[dict[str, Any]],
        target: pd.Series,
        weights: pd.Series,
    ) -> list[BinRow]:
        totals = self._target_totals(target, weights)
        rows = []
        for index, group in enumerate(groups):
            rows.append(
                self._make_row(
                    feature,
                    f"bin_{index}",
                    group["mask"],
                    target,
                    weights,
                    totals,
                    lower=group["lower"],
                    upper=group["upper"],
                )
            )
        return rows

    def _categorical_rows_from_groups(
        self,
        feature: str,
        groups: list[dict[str, Any]],
        target: pd.Series,
        weights: pd.Series,
    ) -> list[BinRow]:
        totals = self._target_totals(target, weights)
        rows = []
        for group in groups:
            rows.append(
                self._make_row(
                    feature,
                    group["label"] if group["label"] != "__RARE__" else "RARE",
                    group["mask"],
                    target,
                    weights,
                    totals,
                    categories=group["categories"],
                )
            )
        return rows

    def _missing_rows(
        self,
        feature: str,
        mask: pd.Series,
        target: pd.Series,
        weights: pd.Series,
    ) -> list[BinRow]:
        return [
            self._make_row(
                feature,
                "MISSING",
                mask,
                target,
                weights,
                self._target_totals(target, weights),
                missing=True,
            )
        ]

    def _make_row(
        self,
        feature: str,
        label: str,
        mask: pd.Series,
        target: pd.Series,
        weights: pd.Series,
        totals: tuple[float, float],
        *,
        lower: float | None = None,
        upper: float | None = None,
        categories: tuple[str, ...] | None = None,
        missing: bool = False,
    ) -> BinRow:
        events = float(weights[mask & (target == 1)].sum())
        non_events = float(weights[mask & (target == 0)].sum())
        total_events, total_non_events = totals
        event_rate = (events + self.smoothing) / (total_events + self.smoothing * 2)
        non_event_rate = (non_events + self.smoothing) / (total_non_events + self.smoothing * 2)
        woe = float(np.log(non_event_rate / event_rate))
        iv = float((non_event_rate - event_rate) * woe)
        total = events + non_events
        return BinRow(
            feature=feature,
            bin_label=label,
            lower=lower,
            upper=upper,
            categories=categories,
            missing=missing,
            events=events,
            non_events=non_events,
            total=total,
            bad_rate=events / total if total else 0.0,
            woe=woe,
            iv=iv,
        )

    def _transform_numeric(self, series: pd.Series, rows: list[BinRow]) -> pd.Series:
        result = pd.Series(np.nan, index=series.index, dtype=float)
        missing_row = next((row for row in rows if row.missing), None)
        for row in rows:
            if row.missing:
                continue
            mask = (
                series.notna()
                & (series.astype(float) > row.lower)
                & (series.astype(float) <= row.upper)
            )
            result.loc[mask] = row.woe
        if missing_row is not None:
            result.loc[series.isna()] = missing_row.woe
        fallback = rows[0].woe if rows else 0.0
        return result.fillna(fallback)

    def _transform_categorical(self, series: pd.Series, rows: list[BinRow]) -> pd.Series:
        values = series.fillna("__MISSING__").astype(str)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        missing_row = next((row for row in rows if row.missing), None)
        for row in rows:
            if row.missing or row.categories is None:
                continue
            result.loc[values.isin(row.categories)] = row.woe
        if missing_row is not None:
            result.loc[values == "__MISSING__"] = missing_row.woe
        fallback = rows[0].woe if rows else 0.0
        return result.fillna(fallback)

    def _target_totals(self, target: pd.Series, weights: pd.Series) -> tuple[float, float]:
        return float(weights[target == 1].sum()), float(weights[target == 0].sum())

    def _bad_rate(self, mask: pd.Series, target: pd.Series, weights: pd.Series) -> float:
        total = float(weights[mask].sum())
        return float(weights[mask & (target == 1)].sum()) / total if total else 0.0
