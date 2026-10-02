import json

from compare_audio.core.recording_analysis import analyze_signal
from compare_audio.core.report_export import SessionInfo, export_report

from .conftest import room


def test_export_writes_all_files(tmp_path, program, tracks_audio):
    result = analyze_signal(room(tracks_audio[0], 1.0), program)
    info = SessionInfo(
        profile_name="測試", started_at="2026-09-28T10:00:00", notes="SN-1"
    )
    html = export_report(result, info, tmp_path / "report", plot_png=b"\x89PNG fake")

    assert html.read_text(encoding="utf-8").count("<tr") >= 3
    document = json.loads((tmp_path / "report" / "result.json").read_text("utf-8"))
    assert document["session"]["notes"] == "SN-1"
    assert document["summary"]["verdict"] == result.summary.verdict.value
    csv_text = (tmp_path / "report" / "events.csv").read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0].startswith("錄音時間")


def test_report_records_selected_sensitivity(tmp_path, program, tracks_audio):
    from dataclasses import replace

    sensitive = replace(
        program, config=program.config.model_copy(update={"sensitivity": 80})
    )
    result = analyze_signal(room(tracks_audio[0], 1.0), sensitive)
    info = SessionInfo(profile_name="測試", started_at="2026-09-30T17:00:00")
    html = export_report(result, info, tmp_path / "report")
    document = json.loads((html.parent / "result.json").read_text("utf-8"))
    assert document["analysis_config"]["sensitivity"] == 80
    assert document["recognition_thresholds"] == result.config.recognition_thresholds()
    assert "辨識靈敏度" in html.read_text("utf-8")
