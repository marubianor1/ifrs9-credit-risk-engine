"""Runtime mode helpers for the Streamlit app."""

from __future__ import annotations

import os

APP_MODE_ENV = "IFRS9_APP_MODE"
LOCAL_FULL = "local_full"
CLOUD_DEMO = "cloud_demo"


def app_mode() -> str:
    """Return the current app execution mode."""
    mode = os.getenv(APP_MODE_ENV, LOCAL_FULL).strip().lower()
    if mode not in {LOCAL_FULL, CLOUD_DEMO}:
        return LOCAL_FULL
    return mode


def is_cloud_demo() -> bool:
    """Return whether the app is running in public cloud-demo mode."""
    return app_mode() == CLOUD_DEMO


def full_mode_message() -> str:
    """Standard disabled-action copy for cloud mode."""
    return "Full recalculation is available in local/full mode."
