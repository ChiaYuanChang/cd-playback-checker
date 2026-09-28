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
