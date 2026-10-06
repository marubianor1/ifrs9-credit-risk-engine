from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable, Iterable

import pytest


class _FakeBlock:
    def __enter__(self) -> _FakeBlock:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def button(self, *args: object, **kwargs: object) -> bool:
        return False

    def form_submit_button(self, *args: object, **kwargs: object) -> bool:
        return False

    def toggle(self, *args: object, value: bool = False, **kwargs: object) -> bool:
        return value

    def selectbox(
        self,
        label: str,
        options: Iterable[object],
        *args: object,
        index: int = 0,
        **kwargs: object,
    ) -> object:
        del label, args, kwargs
        values = list(options)
        return values[index] if values else None

    def multiselect(
        self,
        label: str,
        options: Iterable[object],
        *args: object,
        default: Iterable[object] | None = None,
        **kwargs: object,
    ) -> list[object]:
        del label, options, args, kwargs
        return list(default or [])

    def number_input(
        self,
        label: str,
        *args: object,
        value: object = 0,
        **kwargs: object,
    ) -> object:
        del label, args, kwargs
        return value

    def text_input(self, label: str, value: str = "", *args: object, **kwargs: object) -> str:
        del label, args, kwargs
        return value

    def columns(
        self,
        spec: int | Iterable[object],
        *args: object,
        **kwargs: object,
    ) -> list[_FakeBlock]:
        del args, kwargs
        count = spec if isinstance(spec, int) else len(list(spec))
        return [_FakeBlock() for _ in range(count)]

    def tabs(self, labels: Iterable[str], *args: object, **kwargs: object) -> list[_FakeBlock]:
        del args, kwargs
        return [_FakeBlock() for _ in labels]

    def expander(self, *args: object, **kwargs: object) -> _FakeBlock:
        del args, kwargs
        return _FakeBlock()

    def form(self, *args: object, **kwargs: object) -> _FakeBlock:
        del args, kwargs
        return _FakeBlock()

    def spinner(self, *args: object, **kwargs: object) -> _FakeBlock:
        del args, kwargs
        return _FakeBlock()

    def __getattr__(self, name: str) -> Callable[..., None]:
        del name

        def _noop(*args: object, **kwargs: object) -> None:
            del args, kwargs

        return _noop


def _fake_cache_data(*args: object, **kwargs: object) -> Callable:
    del kwargs
    if args and callable(args[0]):
        return args[0]

    def _decorator(func: Callable) -> Callable:
        return func

    return _decorator


def _fake_streamlit() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    block = _FakeBlock()
    module.session_state = {}
    module.secrets = {}
    module.cache_data = _fake_cache_data
    module.columns = block.columns
    module.tabs = block.tabs
    module.expander = block.expander
    module.form = block.form
    module.spinner = block.spinner
    module.selectbox = block.selectbox
    module.multiselect = block.multiselect
    module.number_input = block.number_input
    module.text_input = block.text_input
    module.button = block.button
    module.form_submit_button = block.form_submit_button
    module.toggle = block.toggle
    for name in (
        "title",
        "warning",
        "info",
        "caption",
        "write",
        "markdown",
        "json",
        "error",
        "success",
        "header",
        "subheader",
        "dataframe",
        "plotly_chart",
        "metric",
        "download_button",
    ):
        setattr(module, name, getattr(block, name))
    module.__getattr__ = lambda name: getattr(block, name)
    return module


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit())
    for name in list(sys.modules):
        if name.startswith("app.pages.") or name == "app.components.guidance":
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_scenario_delta_bar_is_exported() -> None:
    from app.components.charts import scenario_delta_bar

    assert callable(scenario_delta_bar)


@pytest.mark.usefixtures("fake_streamlit")
@pytest.mark.parametrize(
    "module_name",
    [
        "app.pages.overview",
        "app.pages.scoring",
        "app.pages.pd",
        "app.pages.lgd",
        "app.pages.ead",
        "app.pages.sicr",
        "app.pages.ecl",
        "app.pages.scenario_lab",
        "app.pages.monitoring",
        "app.pages.report_generator",
        "app.pages.about",
    ],
)
def test_streamlit_page_modules_import_without_missing_helpers(module_name: str) -> None:
    importlib.import_module(module_name)
