from __future__ import annotations

import re
from pathlib import Path

from app.components import guidance


class _Recorder:
    def __init__(self) -> None:
        self.markdown_calls: list[str] = []
        self.expanders: list[tuple[str, bool]] = []

    def expander(self, title: str, *, expanded: bool = False):
        self.expanders.append((title, expanded))
        return self

    def markdown(self, body: str) -> None:
        self.markdown_calls.append(body)

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _all_guidance_text() -> str:
    parts: list[str] = []
    for guides in guidance.GUIDANCE_SECTIONS.values():
        for guide in guides:
            parts.extend(
                [
                    guide.metric,
                    guide.calculation,
                    guide.interpretation,
                    guide.caution or "",
                ]
            )
    parts.extend(guidance.GLOSSARY)
    parts.extend(guidance.GLOSSARY.values())
    return "\n".join(parts)


def test_guidance_component_renders_compact_expander(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(guidance, "st", recorder)

    guidance.render_metric_guide(
        [
            guidance.MetricGuide(
                "O/E",
                "Observed outcome divided by predicted outcome.",
                "Around 1 indicates aggregate calibration.",
                "Use project thresholds only.",
            )
        ],
        expanded=True,
    )

    assert recorder.expanders == [("How to read this section", True)]
    assert (
        "- **Calculation:** Observed outcome divided by predicted outcome."
        in recorder.markdown_calls
    )
    assert (
        "- **Interpretation:** Around 1 indicates aggregate calibration."
        in recorder.markdown_calls
    )
    assert "- **Watch for:** Use project thresholds only." in recorder.markdown_calls


def test_every_main_page_has_guidance_block() -> None:
    page_root = Path("app/pages")
    pages = [
        "overview.py",
        "scoring.py",
        "pd.py",
        "lgd.py",
        "ead.py",
        "sicr.py",
        "ecl.py",
        "scenario_lab.py",
        "monitoring.py",
        "report_generator.py",
        "about.py",
    ]

    missing = []
    for page in pages:
        source = (page_root / page).read_text()
        if "render_guidance(" not in source and "render_glossary(" not in source:
            missing.append(page)

    assert missing == []


def test_guidance_text_has_no_raw_snake_case_labels() -> None:
    text = _all_guidance_text()

    assert re.search(r"\b[a-z]+_[a-z_]+\b", text) is None


def test_guidance_avoids_unsupported_apra_threshold_claims() -> None:
    text = _all_guidance_text()

    assert "APRA requires" not in text
    assert not re.search(r"APRA.*(?:<|>|<=|>=)", text, flags=re.IGNORECASE)


def test_key_formula_guidance_is_present() -> None:
    text = _all_guidance_text()

    assert "Weighted ECL divided by EAD" in text
    assert "Observed default rate divided by Predicted PD" in text
    assert "marginal PD times LGD times EAD times discount factor" in text
    assert "Downside ECL less Base ECL" in text
