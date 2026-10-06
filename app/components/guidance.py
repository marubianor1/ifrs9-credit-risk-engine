"""Compact methodology guidance for Streamlit pages."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class MetricGuide:
    """One compact metric guidance entry."""

    metric: str
    calculation: str
    interpretation: str
    caution: str | None = None


GUIDANCE_SECTIONS: dict[str, list[MetricGuide]] = {
    "overview": [
        MetricGuide(
            "Portfolio EAD",
            "Sum of reporting-date EAD across active exposures.",
            "Portfolio exposure subject to the ECL calculation; movements may reflect balance change, amortisation or mix.",
        ),
        MetricGuide(
            "Weighted ECL",
            "Sum of scenario-weighted ECL: scenario weight times scenario ECL.",
            "Accounting expected credit loss under the configured scenario weights.",
            "Higher ECL may reflect higher credit risk, more Stage 2 or Stage 3 exposure, or conservative assumptions.",
        ),
        MetricGuide(
            "Coverage ratio",
            "Weighted ECL divided by EAD.",
            "Allowance relative to exposure. Compare by Stage, rating and period rather than reading the portfolio ratio alone.",
        ),
        MetricGuide(
            "Stage 2 + 3 share",
            "Stage 2 EAD plus Stage 3 EAD, divided by total EAD.",
            "Share of exposure showing significant increase in credit risk or credit impairment.",
        ),
        MetricGuide(
            "Downside impact",
            "Downside ECL less Base ECL.",
            "Incremental ECL under adverse macro assumptions.",
        ),
    ],
    "scoring": [
        MetricGuide(
            "Observed default rate",
            "Defaults within 12 months divided by eligible observations.",
            "Realised portfolio credit risk; a high default rate alone is not poor model performance.",
        ),
        MetricGuide(
            "Predicted PD",
            "Average model-predicted 12-month probability of default for the selected population or split.",
            "Predicted credit risk level before comparing with realised defaults.",
        ),
        MetricGuide(
            "O/E",
            "Observed default rate divided by Predicted PD.",
            "Around 1 indicates aggregate calibration; above 1 under-predicts risk; below 1 over-predicts risk.",
            "RAG status uses configurable project thresholds only.",
        ),
        MetricGuide(
            "Difference",
            "Observed default rate less Predicted PD, expressed in basis points.",
            "Shows calibration difference in an additive scale.",
        ),
    ],
    "scoring_performance": [
        MetricGuide(
            "AUC",
            "Probability that a randomly selected default is ranked riskier than a randomly selected non-default.",
            "0.5 is no discrimination; higher values indicate stronger ranking ability.",
            "No regulatory cut-off is implied.",
        ),
        MetricGuide(
            "Gini",
            "Two times AUC minus 1.",
            "Alternative expression of discriminatory power.",
        ),
        MetricGuide(
            "KS",
            "Maximum gap between cumulative default and non-default score distributions.",
            "Higher values generally indicate stronger separation.",
        ),
        MetricGuide(
            "PR-AUC and Brier score",
            "PR-AUC summarises precision-recall performance; Brier score is mean squared PD error.",
            "PR-AUC helps with low-default populations; lower Brier score is better.",
        ),
    ],
    "scoring_calibration": [
        MetricGuide(
            "Calibration plot",
            "Predicted PD versus observed default rate by risk band.",
            "Points near the 45-degree line indicate good calibration.",
            "Above the line indicates under-prediction; below the line indicates over-prediction.",
        ),
        MetricGuide(
            "PSI",
            "Sum of actual share minus reference share, multiplied by the log of their ratio.",
            "Population or model-input drift against the reference population.",
            "Uses project monitoring thresholds, not regulatory rules.",
        ),
    ],
    "pd": [
        MetricGuide(
            "Raw and calibrated PD",
            "Raw scorecard probabilities are adjusted using the configured calibration method.",
            "Calibration aligns model PDs with observed default levels.",
        ),
        MetricGuide(
            "TTC PD",
            "Long-run default-rate anchor used as a through-the-cycle reference.",
            "Provides a stable reference point before point-in-time adjustments.",
        ),
        MetricGuide(
            "PIT and forward-looking PD",
            "TTC or base PD adjusted for current and forecast macro conditions.",
            "Reflects the implemented forward-looking framework.",
        ),
        MetricGuide(
            "Lifetime PD",
            "Monthly marginal PDs are accumulated over remaining life.",
            "Marginal PD is month-specific default risk; cumulative PD is risk accumulated through time.",
        ),
        MetricGuide(
            "Rating grades and transitions",
            "Grades bucket calibrated PD; transitions show movement between grades or default states.",
            "Higher-risk grades indicate larger expected default probability and are not external agency ratings.",
        ),
    ],
    "lgd": [
        MetricGuide(
            "Realised LGD",
            "Economic loss divided by EAD at default using observed recoveries and costs.",
            "Observed loss severity conditional on default under the implemented cash-flow method.",
        ),
        MetricGuide(
            "Predicted LGD",
            "Expected loss severity conditional on default.",
            "Compared with realised LGD through O/E and segment diagnostics.",
        ),
        MetricGuide(
            "Cure rate",
            "Resolved cured default episodes divided by resolved default episodes.",
            "Higher cure rates generally reduce expected severity, all else equal.",
        ),
        MetricGuide(
            "Cure and non-cure decomposition",
            "Expected LGD combines cure probability, LGD given cure and LGD given non-cure.",
            "Shows whether changes are driven by cure mix or non-cure severity.",
        ),
        MetricGuide(
            "Downturn LGD and challengers",
            "Downturn LGD applies validated conservative factors; rejected challengers are shown for governance.",
            "Downturn LGD is sensitivity, not a fitted macro-LGD model.",
            "v1.3 and forward-looking challengers were rejected due to weaker OOT stability than the structural baseline.",
        ),
    ],
    "ead": [
        MetricGuide(
            "EAD at default",
            "Exposure expected to be outstanding when default occurs.",
            "Used with PD and LGD in the ECL engine.",
        ),
        MetricGuide(
            "EAD ratio",
            "EAD at default divided by current exposure where observable.",
            "Indicates whether exposure grows or amortises before default.",
        ),
        MetricGuide(
            "Contractual amortisation baseline",
            "Future exposure is projected from contractual balance reduction.",
            "Selected baseline for amortising Freddie Mac mortgages.",
        ),
        MetricGuide(
            "O/E, MAE and RMSE",
            "O/E is observed EAD divided by predicted EAD; MAE is mean absolute error; RMSE is square-root mean squared error.",
            "Lower MAE and RMSE are better; RMSE is more sensitive to large errors.",
        ),
        MetricGuide(
            "Lifetime EAD profile",
            "Expected exposure path over future months.",
            "Supports later lifetime ECL calculations.",
        ),
    ],
    "sicr": [
        MetricGuide(
            "Stage 1",
            "12-month ECL for exposures without significant increase in credit risk.",
            "Represents performing exposures under IFRS 9 staging.",
        ),
        MetricGuide(
            "Stage 2",
            "Lifetime ECL after significant increase in credit risk.",
            "Driven by enabled SICR triggers and cure rules.",
        ),
        MetricGuide(
            "Stage 3",
            "Credit-impaired or active default state.",
            "Stage 3 takes precedence over Stage 2 triggers.",
        ),
        MetricGuide(
            "SICR triggers",
            "Relative PD uses current PD divided by reference PD; absolute PD, rating downgrade and DPD backstop apply where configured.",
            "Triggers indicate deterioration from the reference risk state.",
        ),
        MetricGuide(
            "Cure and migration",
            "Stage 2 cure requires configured consecutive non-SICR months; migration shows movement between stages.",
            "Calendar gaps break probation where configured.",
        ),
    ],
    "ecl": [
        MetricGuide(
            "Stage 1 ECL",
            "Sum over months 1 to 12 of marginal PD times LGD times EAD times discount factor.",
            "Captures 12-month expected credit loss.",
        ),
        MetricGuide(
            "Stage 2 ECL",
            "Sum over remaining life of marginal PD times LGD times EAD times discount factor.",
            "Captures lifetime expected credit loss after SICR.",
        ),
        MetricGuide(
            "Stage 3 ECL",
            "PD is treated as 100% and loss is approximated consistently with available LGD and EAD methodology.",
            "Credit-impaired loss estimate under current project data constraints.",
            "Detailed default cash-flow timing is a documented project limitation.",
        ),
        MetricGuide(
            "Scenario-weighted ECL",
            "Scenario ECLs are weighted by configured Base, Upside and Downside probabilities.",
            "Shows the accounting estimate under probability-weighted macro assumptions.",
        ),
        MetricGuide(
            "Rating and downturn sensitivity",
            "Rating view shows expected-loss concentration; downturn sensitivity compares structural LGD with conservative downturn LGD.",
            "Downturn sensitivity is not a fitted scenario LGD model.",
        ),
    ],
    "scenario_lab": [
        MetricGuide(
            "Macro shocks",
            "Unemployment, HPI, GDP and rate shocks feed the existing forward-looking PD framework.",
            "They stress current ECL outputs without refitting models.",
        ),
        MetricGuide(
            "Driver waterfall",
            "Sequentially attributes delta ECL to PD/macro, stage migration, LGD, EAD and residual interaction.",
            "Attribution is order-dependent.",
        ),
        MetricGuide(
            "Stage migration effect",
            "Change in ECL attributable to SICR or stage movements.",
            "Highlights deterioration from Stage 1 to Stage 2 or Stage 3.",
        ),
        MetricGuide(
            "Mild and severe presets",
            "Config-driven illustrative stress scenarios.",
            "They are not regulatory scenarios.",
        ),
    ],
    "monitoring": [
        MetricGuide(
            "Discrimination",
            "AUC, Gini and KS measure ranking of higher-risk exposures above lower-risk exposures.",
            "Weakening values may indicate model-rank deterioration.",
        ),
        MetricGuide(
            "Calibration",
            "Observed versus predicted outcomes and O/E measure whether predicted risk matches realised outcomes.",
            "O/E above 1 indicates realised outcomes exceeded predictions.",
        ),
        MetricGuide(
            "Stability",
            "PSI measures distribution drift relative to the reference population.",
            "Higher PSI indicates more drift under project thresholds.",
        ),
        MetricGuide(
            "LGD and EAD monitoring",
            "LGD compares observed and predicted severity; EAD monitors O/E, MAE and RMSE.",
            "Use alongside population and segment diagnostics.",
        ),
        MetricGuide(
            "RAG status",
            "GREEN, AMBER and RED are configurable project monitoring thresholds.",
            "They are not APRA regulatory thresholds.",
        ),
    ],
    "report_generator": [
        MetricGuide(
            "Validated artifacts",
            "Quantitative values are loaded from persisted, validated run artifacts.",
            "The report generator narrates existing results rather than calculating risk metrics.",
        ),
        MetricGuide(
            "Groq narrative",
            "Groq generates structured commentary from a compact report context.",
            "The LLM does not calculate PD, LGD, EAD, staging or ECL.",
        ),
        MetricGuide(
            "Numerical controls",
            "Unsupported numerical claims are flagged against the supplied context.",
            "A deterministic fallback is available when the provider is unavailable.",
        ),
    ],
}

GLOSSARY: dict[str, str] = {
    "PD": "Probability of default.",
    "LGD": "Loss given default.",
    "EAD": "Exposure at default.",
    "ECL": "Expected credit loss.",
    "SICR": "Significant increase in credit risk.",
    "Stage 1": "Performing exposure measured on 12-month ECL.",
    "Stage 2": "SICR exposure measured on lifetime ECL.",
    "Stage 3": "Credit-impaired or defaulted exposure.",
    "O/E": "Observed outcome divided by expected or predicted outcome.",
    "AUC": "Ranking discrimination measure for default versus non-default observations.",
    "Gini": "Two times AUC minus 1.",
    "KS": "Maximum separation between default and non-default score distributions.",
    "PSI": "Population stability index.",
    "TTC": "Through-the-cycle reference risk level.",
    "PIT": "Point-in-time risk level.",
    "OOT": "Out-of-time validation period.",
}


def render_metric_guide(
    guides: list[MetricGuide],
    *,
    title: str = "How to read this section",
    expanded: bool = False,
) -> None:
    """Render compact methodology guidance in a collapsed expander."""
    with st.expander(title, expanded=expanded):
        for guide in guides:
            st.markdown(f"**{guide.metric}**")
            st.markdown(f"- **Calculation:** {guide.calculation}")
            st.markdown(f"- **Interpretation:** {guide.interpretation}")
            if guide.caution:
                st.markdown(f"- **Watch for:** {guide.caution}")


def render_guidance(section: str, *, expanded: bool = False) -> None:
    """Render a named guidance section."""
    render_metric_guide(GUIDANCE_SECTIONS[section], expanded=expanded)


def render_glossary(*, expanded: bool = False) -> None:
    """Render the compact portfolio glossary."""
    with st.expander("Glossary", expanded=expanded):
        for term, definition in GLOSSARY.items():
            st.markdown(f"- **{term}:** {definition}")
