from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest


class _Sidebar:
    def __enter__(self) -> _Sidebar:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Navigation:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


def _fake_streamlit() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    module.secrets = {}
    module.session_state = {}
    module.sidebar = _Sidebar()
    module.set_page_config = lambda **kwargs: None
    module.header = lambda *args, **kwargs: None
    module.selectbox = lambda *args, **kwargs: None
    module.divider = lambda *args, **kwargs: None
    module.metric = lambda *args, **kwargs: None
    module.caption = lambda *args, **kwargs: None
    module.write = lambda *args, **kwargs: None
    module.markdown = lambda *args, **kwargs: None
    module.Page = lambda path, title: {"path": path, "title": title}
    module.navigation = lambda pages: _Navigation()
    return module


def test_streamlit_entrypoint_bootstraps_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    app_dir = repo_root / "app"
    entrypoint = app_dir / "streamlit_app.py"

    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit())

    original_paths = [path for path in sys.path if path not in {"", str(repo_root)}]
    monkeypatch.setattr(sys, "path", [str(app_dir), *original_paths])

    runpy.run_path(str(entrypoint), run_name="__main__")

    assert sys.path[0] == str(repo_root)
    assert "app.components.layout" in sys.modules
