from __future__ import annotations

from app.components.formatting import au_date, bps, percentage, ratio, usd
from app.components.tables import calibration_status
from app.components.theme import (
    RATING_COLOURS,
    SCENARIO_COLOURS,
    STAGE_COLOURS,
    rating_order,
    status_label,
)


def test_currency_formatting_uses_usd_conventions() -> None:
    assert usd(73_245_322_184.51) == "US$73.2bn"
    assert usd(690_088_387.12) == "US$690.1m"
    assert usd(1250) == "US$1,250"


def test_australian_date_formatting() -> None:
    assert au_date("2025-03-01") == "1 Mar 2025"


def test_percentage_bps_and_ratio_formatting() -> None:
    assert percentage(0.38155) == "38.16%"
    assert bps(0.0025) == "25 bps"
    assert ratio(1.234) == "1.23"


def test_semantic_stage_and_scenario_colours() -> None:
    assert STAGE_COLOURS[1] == "#286090"
    assert STAGE_COLOURS[2] == "#B7791F"
    assert STAGE_COLOURS[3] == "#B42318"
    assert SCENARIO_COLOURS["BASE"] == "#286090"
    assert SCENARIO_COLOURS["UPSIDE"] == "#287C8E"
    assert SCENARIO_COLOURS["DOWNSIDE"] == "#B42318"


def test_rating_order_and_progression() -> None:
    assert rating_order() == ["R1", "R2", "R3", "R4", "R5"]
    assert list(RATING_COLOURS) == rating_order()


def test_bad_rate_calibration_status_logic() -> None:
    assert calibration_status(1.0) == "GREEN"
    assert calibration_status(0.7) == "AMBER"
    assert calibration_status(1.4) == "AMBER"
    assert calibration_status(0.5) == "RED"
    assert calibration_status(1.6) == "RED"


def test_monitoring_rag_labels_include_text_marker() -> None:
    assert status_label("GREEN") == "OK - GREEN"
    assert status_label("AMBER") == "WATCH - AMBER"
    assert status_label("RED") == "ACTION - RED"
