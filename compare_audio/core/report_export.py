"""Saving an analysis: JSON (machine readable), CSV (events) and an HTML report."""

import base64
import csv
import html
import json
from pathlib import Path

from pydantic import BaseModel, Field

from compare_audio.core.analysis_models import (
    EVENT_LABELS,
    AnalysisResult,
    Severity,
    TrackVerdict,
    Verdict,
)
from compare_audio.core.time_format import format_clock

VERDICT_TEXT = {
    Verdict.PASS: "PASS 正常",
    Verdict.WARN: "需人工確認",
    Verdict.FAIL: "FAIL 有問題",
}
TRACK_VERDICT_TEXT = {
    TrackVerdict.PASS: "正常",
    TrackVerdict.WARN: "需確認",
    TrackVerdict.FAIL: "有問題",
    TrackVerdict.NOT_HEARD: "未錄到",
}
SEVERITY_TEXT = {Severity.FAIL: "問題", Severity.WARN: "待確認", Severity.INFO: "資訊"}


class SessionInfo(BaseModel):
    """Who/what/when of a test run, shown in reports."""

    profile_name: str = Field(..., description="Test disc name.")
    recording_path: str | None = Field(default=None, description="Saved WAV file.")
    started_at: str = Field(..., description="ISO timestamp of the recording.")
    device: str | None = Field(default=None, description="Microphone used.")
    notes: str = Field(default="", description="Operator notes / serial number.")


def result_document(result: AnalysisResult, info: SessionInfo) -> dict[str, object]:
    return {
        "session": info.model_dump(),
        "summary": result.summary.model_dump(mode="json"),
        "tracks": [t.model_dump(mode="json") for t in result.tracks],
        "events": [e.model_dump(mode="json") for e in result.events],
        "segments": [s.model_dump(mode="json") for s in result.segments],
    }


def write_json(result: AnalysisResult, info: SessionInfo, path: Path) -> Path:
    path.write_text(
        json.dumps(result_document(result, info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_events_csv(result: AnalysisResult, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:  # Excel-friendly
        writer = csv.writer(handle)
        writer.writerow(
            ["錄音時間", "結束", "等級", "類型", "曲目", "曲內位置", "跳動(秒)", "說明"]
        )
        for event in result.events:
            writer.writerow(
                [
                    format_clock(event.rec_start_s, 2),
                    format_clock(event.rec_end_s, 2),
                    SEVERITY_TEXT[event.severity],
                    EVENT_LABELS[event.event_type],
                    "" if event.track_index is None else event.track_index + 1,
                    ""
                    if event.track_position_s is None
                    else format_clock(event.track_position_s, 2),
                    "" if event.jump_s is None else f"{event.jump_s:+.3f}",
                    event.detail,
                ]
            )
    return path


def _row(cells: list[str], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def write_html(
    result: AnalysisResult, info: SessionInfo, path: Path, plot_png: bytes | None
) -> Path:
    s = result.summary
    esc = html.escape
    colors = {Verdict.PASS: "#2e7d32", Verdict.WARN: "#b7791f", Verdict.FAIL: "#c62828"}
    facts = [
        ("測試片", esc(info.profile_name)),
        ("錄音時間", esc(info.started_at)),
        ("麥克風", esc(info.device or "-")),
        ("備註", esc(info.notes or "-")),
        ("錄音長度", format_clock(s.recording_duration_s)),
        (
            "開始播放於",
            "-" if s.playback_start_s is None else format_clock(s.playback_start_s),
        ),
        ("聽到的比例", f"{100 * s.heard_ratio:.1f}%"),
        (
            "訊噪比",
            "-"
            if s.signal_to_noise_db is None
            else f"約 {s.signal_to_noise_db:.0f} dB",
        ),
        ("時脈差", "-" if s.drift_ppm is None else f"{s.drift_ppm:+.0f} ppm"),
        ("錄音檔", esc(info.recording_path or "-")),
    ]
    track_rows = [
        _row(["#", "曲目", "長度", "結果", "聽到", "問題", "待確認"], "th")
    ] + [
        _row(
            [
                str(t.index + 1),
                esc(t.title),
                format_clock(t.duration_s, 0),
                TRACK_VERDICT_TEXT[t.verdict],
                format_clock(t.heard_s, 0),
                str(t.fail_count),
                str(t.warn_count),
            ]
        )
        for t in result.tracks
    ]
    event_rows = [_row(["錄音時間", "等級", "類型", "曲目", "說明"], "th")] + [
        f'<tr class="{e.severity.value}">'
        + "".join(
            f"<td>{cell}</td>"
            for cell in [
                format_clock(e.rec_start_s, 2),
                SEVERITY_TEXT[e.severity],
                EVENT_LABELS[e.event_type],
                "-" if e.track_index is None else str(e.track_index + 1),
                esc(e.detail),
            ]
        )
        + "</tr>"
        for e in result.events
    ]
    image = ""
    if plot_png:
        encoded = base64.b64encode(plot_png).decode("ascii")
        image = (
            f'<h2>對齊圖</h2><img src="data:image/png;base64,{encoded}" alt="對齊圖">'
        )
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>播放測試報告 - {esc(info.profile_name)}</title>
<style>
body {{ font-family: "Microsoft JhengHei", "PingFang TC", sans-serif; margin: 24px;
       color: #222; background: #fff; }}
.verdict {{ font-size: 28px; font-weight: bold; color: white; padding: 12px 20px;
            border-radius: 8px; display: inline-block;
            background: {colors[s.verdict]}; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
th {{ background: #f2f2f2; }}
tr.fail td {{ background: #fdecea; }} tr.warn td {{ background: #fff6e0; }}
tr.info td {{ color: #666; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
</style></head><body>
<h1>CD 播放測試報告</h1>
<div class="verdict">{VERDICT_TEXT[s.verdict]}</div>
<p>問題 {s.fail_count} 個，待確認 {s.warn_count} 個</p>
<table>{"".join(_row([k, v]) for k, v in facts)}</table>
<h2>各曲結果</h2><table>{"".join(track_rows)}</table>
<h2>事件</h2><table>{"".join(event_rows)}</table>
{image}
</body></html>"""
    path.write_text(document, encoding="utf-8")
    return path


def export_report(
    result: AnalysisResult,
    info: SessionInfo,
    folder: Path,
    plot_png: bytes | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    write_json(result, info, folder / "result.json")
    write_events_csv(result, folder / "events.csv")
    return write_html(result, info, folder / "report.html", plot_png)
