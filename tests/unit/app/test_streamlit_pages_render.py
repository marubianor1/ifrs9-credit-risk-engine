from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    "page",
    [
        "pd",
        "lgd",
        "ead",
        "sicr",
        "scenario_lab",
        "report_generator",
        "about",
    ],
)
def test_v2_remaining_pages_render_without_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    page: str,
) -> None:
    monkeypatch.setenv("IFRS9_APP_MODE", "cloud_demo")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    repo_root = Path(__file__).resolve().parents[3]
    page_path = repo_root / "app" / "pages" / f"{page}.py"

    app = AppTest.from_file(str(page_path)).run(timeout=120)

    assert list(app.exception) == []
