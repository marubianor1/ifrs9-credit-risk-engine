"""Scenario Lab page for persisted and custom stress runs."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from app.components.guidance import render_guidance
from app.components.layout import friendly_error, money, page_title, pct
from app.services.runtime import full_mode_message, is_cloud_demo
from app.services.scenario_lab import (
    load_lab_config,
    load_scenario_result,
    run_or_load_scenario,
    scenario_config_from_controls,
    scenario_run_id,
)
from pydantic import ValidationError


@st.cache_data(show_spinner=False)
def _load_result(run_id: str):
    return load_scenario_result(run_id)


def _preset_defaults(preset_key: str) -> dict:
    lab_config = load_lab_config()
    if preset_key == "custom":
        return lab_config.presets["baseline"].model_dump()
    return lab_config.presets[preset_key].model_dump()


def _default_text(value: object) -> str:
    """Return an empty string for unset optional widget defaults."""
    return "" if value is None else str(value)


def _share_default(defaults: dict, key: str) -> float:
    """Return a percentage default from a decimal portfolio share."""
    return float(defaults["portfolio"][key] * 100)


def _render_result(result: dict) -> None:
    summary = result["summary"]
    stage = result["stage"]
    rating = result["rating"]
    waterfall = result["waterfall"]

    st.subheader("Baseline vs Stressed ECL")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline EAD", money(summary["baseline_ead"]))
    k2.metric("Baseline ECL", money(summary["baseline_ecl"]))
    k3.metric("Stressed ECL", money(summary["stressed_ecl"]))
    k4.metric("Delta ECL", money(summary["delta_ecl"]), pct(summary["delta_ecl_pct"]))

    left, right = st.columns(2)
    with left:
        st.subheader("Stage Migration")
        chart = stage.melt(
            id_vars=["stage"],
            value_vars=["baseline_ecl", "stressed_ecl"],
            var_name="case",
            value_name="ecl",
        )
        stage_fig = px.bar(chart, x="stage", y="ecl", color="case", barmode="group")
        st.plotly_chart(stage_fig, use_container_width=True)
        st.dataframe(stage, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Driver Waterfall")
        waterfall_fig = px.bar(waterfall, x="driver", y="effect", color="driver")
        st.plotly_chart(waterfall_fig, use_container_width=True)
        st.dataframe(waterfall, use_container_width=True, hide_index=True)

    st.subheader("ECL Delta by Rating")
    rating_fig = px.bar(rating, x="rating", y="delta_ecl", color="rating")
    st.plotly_chart(rating_fig, use_container_width=True)
    st.dataframe(rating, use_container_width=True, hide_index=True)


def main() -> None:
    page_title("Scenario Lab", "Stress existing ECL outputs without refitting parent models.")
    render_guidance("scenario_lab")
    lab_config = load_lab_config()
    labels = {
        "baseline": "Baseline",
        "mild_deterioration": "Mild deterioration",
        "severe_deterioration": "Severe deterioration",
        "custom": "Custom",
    }
    selected_label = st.selectbox(
        "Preset",
        list(labels.values()),
        index=list(labels).index(st.session_state.get("scenario_preset", "baseline"))
        if st.session_state.get("scenario_preset", "baseline") in labels
        else 0,
    )
    preset_key = next(key for key, value in labels.items() if value == selected_label)
    st.session_state["scenario_preset"] = preset_key
    defaults = _preset_defaults(preset_key)

    st.info(
        "Scenario Lab overlays are stress assumptions for portfolio analysis. They are not "
        "newly fitted PD, LGD, EAD, or staging models."
    )
    if is_cloud_demo():
        st.info(full_mode_message())
        preset_map = {
            "Baseline": "scenario_baseline_v1",
            "Mild deterioration": "scenario_mild_deterioration_v1",
            "Severe deterioration": "scenario_severe_deterioration_v1",
        }
        scenario_summaries = []
        for label, scenario_id in preset_map.items():
            try:
                summary = _load_result(scenario_id)["summary"]
            except Exception as exc:
                friendly_error(exc)
                return
            scenario_summaries.append({"label": label, **summary})
        st.subheader("Preset Scenario Comparison")
        st.dataframe(scenario_summaries, use_container_width=True, hide_index=True)
        selected_cloud = st.selectbox("Scenario result", list(preset_map))
        _render_result(_load_result(preset_map[selected_cloud]))
        return

    with st.form("scenario_controls"):
        st.subheader("Macro / PD")
        c1, c2, c3 = st.columns(3)
        unemployment = c1.number_input(
            "Unemployment shock",
            value=float(defaults["macro"]["unemployment_shock"]),
            step=0.25,
        )
        hpi = c2.number_input("HPI shock", value=float(defaults["macro"]["hpi_shock"]), step=1.0)
        gdp = c3.number_input("GDP shock", value=float(defaults["macro"]["gdp_shock"]), step=0.25)
        c4, c5 = st.columns(2)
        mortgage_rate = c4.number_input(
            "Mortgage-rate shock",
            value=float(defaults["macro"]["mortgage_rate_shock"]),
            step=0.25,
        )
        rho_text = c5.text_input(
            "Rho override",
            _default_text(defaults["macro"]["rho_override"]),
        )
        weights = defaults["macro"].get("scenario_weights") or {
            "UPSIDE": 0.20,
            "BASE": 0.55,
            "DOWNSIDE": 0.25,
        }
        w1, w2, w3 = st.columns(3)
        upside_weight = w1.number_input(
            "Upside weight",
            value=float(weights["UPSIDE"]),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        base_weight = w2.number_input(
            "Base weight",
            value=float(weights["BASE"]),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        downside_weight = w3.number_input(
            "Downside weight",
            value=float(weights["DOWNSIDE"]),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )

        st.subheader("SICR / Staging")
        s1, s2, s3, s4 = st.columns(4)
        relative_pd = s1.number_input(
            "Relative PD threshold",
            value=float(defaults["staging"]["relative_pd_threshold"] or 2.0),
            min_value=0.1,
            step=0.1,
        )
        absolute_pd_text = s2.text_input(
            "Absolute PD threshold",
            _default_text(defaults["staging"]["absolute_pd_threshold"]),
        )
        rating_downgrade = s3.number_input(
            "Rating downgrade notches",
            value=int(defaults["staging"]["rating_downgrade_threshold"] or 3),
            min_value=1,
            step=1,
        )
        dpd_backstop = s4.number_input(
            "DPD backstop months",
            value=int(defaults["staging"]["dpd_backstop"] or 1),
            min_value=1,
            step=1,
        )
        cure_probation = st.number_input(
            "Stage 2 cure probation months",
            value=int(defaults["staging"]["stage2_cure_probation"] or 3),
            min_value=0,
            step=1,
        )

        st.subheader("Portfolio Deterioration")
        p1, p2, p3, p4 = st.columns(4)
        rating_share = p1.number_input(
            "Rating downgrade %",
            value=_share_default(defaults, "rating_downgrade_share"),
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )
        delinquency_1m = p2.number_input(
            "1m delinquency shock %",
            value=_share_default(defaults, "delinquency_1m_share"),
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )
        delinquency_2m = p3.number_input(
            "2m delinquency shock %",
            value=_share_default(defaults, "delinquency_2m_share"),
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )
        delinquency_3m = p4.number_input(
            "3m+ / default shock %",
            value=_share_default(defaults, "delinquency_3m_default_share"),
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )

        st.subheader("LGD / EAD Stress Assumptions")
        l1, l2, l3 = st.columns(3)
        lgd_method = l1.selectbox(
            "LGD method",
            ["structural_base", "downturn_sensitivity"],
            index=["structural_base", "downturn_sensitivity"].index(defaults["lgd"]["method"]),
        )
        lgd_overlay = l2.number_input(
            "Manual LGD overlay pp",
            value=float(defaults["lgd"]["manual_overlay_pp"]),
            step=0.5,
        )
        ead_multiplier = l3.number_input(
            "EAD multiplier",
            value=float(defaults["ead"]["multiplier"]),
            min_value=0.0,
            step=0.01,
        )

        run_clicked = st.form_submit_button("Run Scenario", type="primary")

    if run_clicked:
        try:
            if preset_key == "baseline":
                config = lab_config.presets["baseline"]
            else:
                rho = float(rho_text) if rho_text else None
                absolute_pd = float(absolute_pd_text) if absolute_pd_text else None
                config = scenario_config_from_controls(
                    name=preset_key,
                    description=labels[preset_key],
                    macro={
                        "unemployment_shock": unemployment,
                        "hpi_shock": hpi,
                        "gdp_shock": gdp,
                        "mortgage_rate_shock": mortgage_rate,
                        "rho_override": rho,
                        "scenario_weights": {
                            "UPSIDE": upside_weight,
                            "BASE": base_weight,
                            "DOWNSIDE": downside_weight,
                        },
                    },
                    staging={
                        "relative_pd_threshold": relative_pd,
                        "absolute_pd_threshold": absolute_pd,
                        "rating_downgrade_threshold": rating_downgrade,
                        "dpd_backstop": dpd_backstop,
                        "stage2_cure_probation": cure_probation,
                    },
                    portfolio={
                        "rating_downgrade_share": rating_share / 100.0,
                        "delinquency_1m_share": delinquency_1m / 100.0,
                        "delinquency_2m_share": delinquency_2m / 100.0,
                        "delinquency_3m_default_share": delinquency_3m / 100.0,
                        "seed": defaults["portfolio"]["seed"],
                    },
                    lgd={"method": lgd_method, "manual_overlay_pp": lgd_overlay},
                    ead={"multiplier": ead_multiplier},
                )
        except (ValueError, ValidationError) as exc:
            friendly_error(exc)
            return
        with st.spinner("Running or loading Scenario Lab result..."):
            try:
                run_id, result, cached = run_or_load_scenario(config)
            except Exception as exc:
                friendly_error(exc)
                return
        st.session_state["last_scenario_run_id"] = run_id
        st.session_state["last_scenario_cached"] = cached
        st.success(f"{'Loaded cached' if cached else 'Created'} scenario result: `{run_id}`")

    run_id = st.session_state.get("last_scenario_run_id")
    if not run_id:
        config = lab_config.presets["baseline"]
        run_id = scenario_run_id(config)
        if not st.session_state.get("last_scenario_run_id"):
            st.caption(
                "Run a scenario to populate results. Existing preset artifacts are also "
                "available below."
            )
            run_id = "scenario_baseline_v1"

    try:
        result = _load_result(run_id)
    except Exception:
        result = _load_result("scenario_baseline_v1")

    _render_result(result)


main()
