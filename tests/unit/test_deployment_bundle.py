from __future__ import annotations

import os
from pathlib import Path

from ifrs9.deployment import audit_deployment_bundle


def test_deployment_bundle_integrity() -> None:
    result = audit_deployment_bundle(Path("deployment"))

    assert result.file_count > 0
    assert result.bundle_size_bytes > 0
    assert result.loan_level_files_detected == 0
    assert result.absolute_paths_detected == 0
    assert result.secrets_detected == 0


def test_cloud_demo_artifact_loading(monkeypatch) -> None:
    monkeypatch.setenv("IFRS9_APP_MODE", "cloud_demo")
    from app.services.ead import load_ead_artifacts
    from app.services.ecl import load_ecl_artifacts
    from app.services.lgd import load_lgd_artifacts
    from app.services.pd import load_pd_artifacts
    from app.services.scenario_lab import load_scenario_result
    from app.services.scorecard import load_scorecard_artifacts
    from app.services.sicr import load_sicr_artifacts

    from ifrs9.reporting import build_report_context

    assert not load_scorecard_artifacts("behavioural_qe_v1")["metrics"].empty
    assert not load_pd_artifacts("pd_behavioural_qe_v1")["rating_summary"].empty
    assert not load_lgd_artifacts("lgd_v1_2")["backtesting"].empty
    assert not load_ead_artifacts("ead_v1")["profiles"].empty
    assert not load_sicr_artifacts("sicr_v1_1")["stage_distribution"].empty
    assert not load_ecl_artifacts("ecl_v1")["stage"].empty
    assert load_scenario_result("scenario_baseline_v1")["summary"]

    context = build_report_context(
        repo_root=Path.cwd(),
        report_type="Executive IFRS 9 Summary",
        audience="Executive",
        detail="Concise",
    )
    payload = context.model_dump_json()
    assert "loan_id" not in payload
    assert "part-ecl.parquet" not in payload
    assert os.getenv("IFRS9_APP_MODE") == "cloud_demo"
