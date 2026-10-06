"""About and methodology page."""

from __future__ import annotations

import streamlit as st
from app.components.guidance import render_glossary
from app.components.layout import page_title


def main() -> None:
    page_title(
        "About / Methodology",
        "Portfolio/research implementation notes and governance boundaries.",
    )
    st.info(
        "Portfolio / research implementation. This is not a regulatory production system and "
        "does not claim production readiness."
    )
    st.subheader("Project purpose")
    st.write(
        "Demonstrate an end-to-end IFRS 9 credit-risk workflow using public Freddie Mac "
        "mortgage data, with transparent model governance and reproducible artifacts."
    )

    st.subheader("Architecture pipeline")
    pipeline = [
        "Raw",
        "Bronze",
        "Silver",
        "Gold",
        "Scorecard",
        "PD / LGD / EAD",
        "SICR",
        "ECL",
        "Scenario Lab",
        "Reporting",
    ]
    st.markdown(" → ".join(f"**{step}**" for step in pipeline))

    left, right = st.columns(2)
    with left:
        st.subheader("IFRS 9 framework")
        st.markdown(
            """
            - PD estimates default likelihood.
            - LGD estimates severity conditional on default.
            - EAD estimates exposure at default.
            - SICR allocates Stage 1, Stage 2 and Stage 3.
            - ECL combines PD, LGD, EAD, staging, scenarios and discounting.
            """
        )
        st.subheader("Data source")
        st.write(
            "Freddie Mac public loan-level origination and monthly performance data, "
            "transformed into raw, Bronze, Silver and Gold analytical layers."
        )
    with right:
        st.subheader("Model governance")
        st.write(
            "`lgd_v1_2` is retained as the project LGD baseline. Rejected challengers remain "
            "visible for audit, and scenario/report outputs are labelled separately from model "
            "output."
        )
        st.subheader("Forward-looking methodology")
        st.write(
            "Forward-looking PD uses existing macro scenario overlays. LGD remains structural and "
            "macro-neutral in the baseline, with downturn LGD available as sensitivity."
        )

    st.subheader("Limitations")
    st.markdown(
        """
        - Freddie Mac public data omits some contractual cashflow and servicing detail.
        - Stage 3 ECL uses a documented approximation where detailed default cashflow timing is
          unavailable.
        - Monitoring statuses use project thresholds, not regulatory thresholds.
        - Streamlit presents persisted artifacts and does not refit models on page load.
        """
    )
    st.subheader("Reproducibility / tests")
    st.write(
        "The repository includes versioned configuration, persisted run artifacts, service-layer "
        "tests, page-import tests and Streamlit AppTest checks for dashboard stability."
    )
    render_glossary()


main()
