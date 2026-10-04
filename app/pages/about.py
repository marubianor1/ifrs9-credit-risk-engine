"""About and methodology page."""

from __future__ import annotations

import streamlit as st
from app.components.layout import page_title


def main() -> None:
    page_title(
        "About / Methodology",
        "Portfolio/research implementation notes and governance boundaries.",
    )
    st.markdown(
        """
        This application is a professional portfolio and research implementation of an IFRS 9
        credit risk workflow using Freddie Mac Single-Family Loan-Level data. It is not a
        regulatory production system and does not claim production readiness.

        **Data source:** Freddie Mac public loan-level origination and monthly performance data,
        transformed through raw, Bronze, Silver, and Gold analytical layers.

        **Architecture:** backend model and artifact generation lives in `src/ifrs9`; Streamlit is
        a presentation layer over persisted artifacts and explicit backend APIs.

        **IFRS 9 methodology:** the project covers scoring, calibrated PD, forward-looking PD,
        LGD, amortizing mortgage EAD, SICR/staging, ECL, scenario stress testing, monitoring, and
        AI-assisted reporting.

        **Point-in-time controls:** model inputs use safe point-in-time variables and persisted
        run IDs. Streamlit pages do not recalculate validated metrics on load.

        **Model governance:** `lgd_v1_2` is retained as the project LGD baseline; rejected LGD
        challengers remain visible for audit. Scenario and report pages label stress overlays and
        AI-generated commentary separately from model output.

        **Limitations:** Freddie Mac public data omits some contractual cashflow and servicing
        detail. Stage 3 ECL uses an approximation where detailed default cashflow timing is
        unavailable. Monitoring thresholds are project thresholds, not regulatory limits.

        **Security:** no API keys or Freddie Mac source extracts are committed. Public app
        downloads are report-level summaries rather than loan-level datasets.
        """
    )


main()
