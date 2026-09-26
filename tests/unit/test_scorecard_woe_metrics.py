from __future__ import annotations

import numpy as np
import pandas as pd

from ifrs9.models.scorecard.metrics import binary_metrics, psi
from ifrs9.models.scorecard.woe import WoeBinner


def test_woe_iv_missing_bin_and_no_infinite_values() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None, 6.0]})
    target = pd.Series([0, 0, 1, 1, 0, 1])
    weights = pd.Series([1.0] * 6)

    binner = WoeBinner(max_bins=3, min_bin_pct=0.1, enforce_monotonic=False).fit(
        frame,
        target,
        sample_weight=weights,
    )
    table = binner.bin_table()
    transformed = binner.transform(frame)

    assert table["iv"].sum() > 0
    assert table["missing"].any()
    assert np.isfinite(table["woe"]).all()
    assert np.isfinite(transformed["x"]).all()


def test_monotonic_binning_and_frozen_oot_bins() -> None:
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    target = pd.Series([0] * 50 + [1] * 50)
    binner = WoeBinner(max_bins=8, min_bin_pct=0.05, enforce_monotonic=True).fit(frame, target)

    table = binner.bin_table()
    non_missing = table[~table["missing"]].sort_values("lower")
    assert non_missing["bad_rate"].is_monotonic_increasing

    oot = pd.DataFrame({"x": [-10.0, 10.0, 110.0, None]})
    transformed = binner.transform(oot)
    assert len(transformed) == 4
    assert np.isfinite(transformed["x"]).all()


def test_weighted_metrics_and_psi() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.2, 0.5, 0.9])
    w = np.array([1.0, 2.0, 1.0, 2.0])

    result = binary_metrics(y, p, w)

    assert result["roc_auc"] > 0.5
    assert result["ks"] > 0
    assert result["brier"] > 0
    assert psi(np.array([1, 2, 3, 4]), np.array([1, 2, 8, 9])) > 0
