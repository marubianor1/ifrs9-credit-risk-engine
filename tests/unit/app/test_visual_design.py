from __future__ import annotations

from pathlib import Path

import pandas as pd
from app.components.advisory import (
    stage_concentration_insight,
    stage_concentration_table,
)
from app.components.charts import (
    calibration_scatter,
    dumbbell_chart,
    oe_bullet_chart,
    scenario_delta_bar,
    scenario_waterfall,
    stage_bar,
    stage_concentration_chart,
)
from app.components.formatting import (
    au_date,
    bps,
    display_label,
    ordered_splits,
    percentage,
    ratio,
    usd,
)
from app.components.tables import bad_rate_table, calibration_status, format_table
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
    assert "..." not in usd(73_245_322_184.51)


def test_australian_date_formatting() -> None:
    assert au_date("2025-03-01") == "1 Mar 2025"


def test_percentage_bps_and_ratio_formatting() -> None:
    assert percentage(0.38155) == "38.16%"
    assert bps(0.0025) == "25 bps"
    assert ratio(1.234) == "1.23"


def test_semantic_stage_and_scenario_colours() -> None:
    assert STAGE_COLOURS[1] == "#2457A6"
    assert STAGE_COLOURS[2] == "#B7791F"
    assert STAGE_COLOURS[3] == "#B42318"
    assert SCENARIO_COLOURS["BASE"] == "#2457A6"
    assert SCENARIO_COLOURS["UPSIDE"] == "#23827A"
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
    assert status_label("GREEN") == "● - GREEN"
    assert status_label("AMBER") == "▲ - AMBER"
    assert status_label("RED") == "■ - RED"


def test_split_order_and_display_label_mapping() -> None:
    assert ordered_splits(["OOT", "TRAIN", "VALIDATION"]) == ["TRAIN", "VALIDATION", "OOT"]
    assert display_label("observed_bad_rate") == "Observed default rate"
    assert display_label("predicted_pd") == "Predicted PD"
    assert display_label("oe_ratio") == "O/E"


def test_bad_rate_table_uses_display_labels_and_status_markers() -> None:
    metrics = pd.DataFrame(
        [
            {"split": "OOT", "observed_bad_rate": 0.006, "predicted_bad_rate": 0.005},
            {"split": "TRAIN", "observed_bad_rate": 0.004, "predicted_bad_rate": 0.004},
            {
                "split": "VALIDATION",
                "observed_bad_rate": 0.018,
                "predicted_bad_rate": 0.005,
            },
        ]
    )
    table = bad_rate_table(metrics)

    assert table.columns.tolist() == [
        "Split",
        "Observed default rate",
        "Predicted PD",
        "Difference",
        "O/E",
        "Calibration status",
    ]
    assert table["Split"].astype(str).tolist() == ["TRAIN", "VALIDATION", "OOT"]
    assert table.loc[0, "Calibration status"].startswith("●")


def test_table_formatters_do_not_treat_text_method_columns_as_rates() -> None:
    frame = pd.DataFrame(
        [
            {
                "calibration_method": "raw",
                "observed_bad_rate": 0.01,
                "predicted_bad_rate": 0.012,
            }
        ]
    )
    styler = format_table(frame)

    assert "raw" in styler.to_html()


def test_table_headers_do_not_expose_snake_case_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "observed_bad_rate": 0.01,
                "predicted_bad_rate": 0.012,
                "oe_ratio": 0.83,
            }
        ]
    )
    styler = format_table(frame)

    assert styler.data.columns.tolist() == ["Observed default rate", "Predicted PD", "O/E"]


def test_percentage_axis_helper_formats_rates() -> None:
    frame = pd.DataFrame(
        [{"stage": 1, "coverage_ratio": 0.01}, {"stage": 2, "coverage_ratio": 0.25}]
    )
    fig = stage_bar(frame, value="coverage_ratio", title="Coverage", yaxis_title="Coverage")

    assert fig.layout.yaxis.tickformat == ".0%"


def test_calibration_chart_uses_scatter_and_reference_line() -> None:
    frame = pd.DataFrame(
        [
            {"split": "TRAIN", "predicted_bad_rate": 0.01, "observed_bad_rate": 0.012},
            {
                "split": "VALIDATION",
                "predicted_bad_rate": 0.02,
                "observed_bad_rate": 0.018,
            },
            {"split": "OOT", "predicted_bad_rate": 0.03, "observed_bad_rate": 0.032},
        ]
    )
    fig = calibration_scatter(frame, title="Calibration")

    assert fig.data[0].mode == "markers"
    assert any(trace.name == "Observed = Predicted" and trace.mode == "lines" for trace in fig.data)


def test_scenario_delta_values_are_vs_base() -> None:
    frame = pd.DataFrame(
        [
            {"scenario": "UPSIDE", "ecl": 90.0},
            {"scenario": "BASE", "ecl": 100.0},
            {"scenario": "DOWNSIDE", "ecl": 115.0},
        ]
    )
    fig = scenario_delta_bar(frame, title="Scenario")
    y_values = sorted(float(value) for trace in fig.data for value in trace.y)

    assert y_values == [-10.0, 0.0, 15.0]


def test_scenario_waterfall_reconciles_to_total_delta() -> None:
    frame = pd.DataFrame(
        [
            {"driver": "PD / macro", "effect": 10.0},
            {"driver": "Stage migration", "effect": 5.0},
            {"driver": "LGD", "effect": -2.0},
        ]
    )
    fig = scenario_waterfall(frame, title="Waterfall")

    assert fig.data[0].measure[-1] == "total"
    assert float(fig.data[0].y[-1]) == 13.0


def test_stage_concentration_table_and_insight_are_deterministic() -> None:
    stage = pd.DataFrame(
        [
            {"stage": 1, "total_ead": 90.0, "weighted_ecl": 10.0, "coverage_ratio": 0.01},
            {"stage": 2, "total_ead": 8.0, "weighted_ecl": 30.0, "coverage_ratio": 0.12},
            {"stage": 3, "total_ead": 2.0, "weighted_ecl": 60.0, "coverage_ratio": 0.40},
        ]
    )

    table = stage_concentration_table(stage)
    insight = stage_concentration_insight(stage, source="unit")

    assert table.loc[0, "% EAD"] == "90.00%"
    assert table.loc[2, "% ECL"] == "60.00%"
    assert insight is not None
    assert "10.0% of portfolio exposure" in insight.text
    assert "90.0% of expected credit loss" in insight.text


def test_stage_concentration_chart_compares_ead_and_ecl_share() -> None:
    stage = pd.DataFrame(
        [
            {"stage": 1, "total_ead": 90.0, "weighted_ecl": 10.0},
            {"stage": 2, "total_ead": 10.0, "weighted_ecl": 90.0},
        ]
    )
    fig = stage_concentration_chart(stage, title="Concentration")

    assert {trace.name for trace in fig.data} == {"EAD share", "ECL share"}
    assert fig.layout.xaxis.tickformat == ".0%"


def test_dumbbell_chart_structure() -> None:
    frame = pd.DataFrame(
        [
            {"split": "TRAIN", "observed": 0.01, "predicted": 0.012},
            {"split": "OOT", "observed": 0.02, "predicted": 0.018},
        ]
    )
    fig = dumbbell_chart(
        frame,
        category="split",
        left_value="predicted",
        right_value="observed",
        left_label="Predicted",
        right_label="Observed",
        title="Dumbbell",
        xaxis_title="Rate",
    )

    assert len(fig.data) == 4
    assert fig.data[-2].name == "Predicted"
    assert fig.data[-1].name == "Observed"


def test_oe_bullet_reference_line_is_one() -> None:
    frame = pd.DataFrame(
        [
            {"metric": "O/E TRAIN", "oe": 1.0},
            {"metric": "O/E OOT", "oe": 0.75},
        ]
    )
    fig = oe_bullet_chart(frame, category="metric", oe_value="oe", title="O/E")

    assert any(
        shape.type == "line" and shape.x0 == 1.0 and shape.x1 == 1.0
        for shape in fig.layout.shapes
    )


def test_app_code_uses_current_streamlit_and_pandas_width_apis() -> None:
    app_root = Path(__file__).resolve().parents[3] / "app"
    source = "\n".join(path.read_text() for path in app_root.rglob("*.py"))

    assert "use_container_width" not in source
    assert ".applymap(" not in source
