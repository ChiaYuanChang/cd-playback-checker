"""Result page: verdict, the three linked plots, events (with A/B listening), tracks."""

from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from compare_audio.audio.clip_player import play_clip, stop_playback
from compare_audio.core.analysis_models import (
    EVENT_LABELS,
    AnalysisResult,
    DetectedEvent,
    Severity,
)
from compare_audio.core.audio_io import read_clip
from compare_audio.core.report_export import (
    SEVERITY_TEXT,
    TRACK_VERDICT_TEXT,
    VERDICT_TEXT,
)
from compare_audio.core.time_format import format_clock
from compare_audio.ui import theme
from compare_audio.ui.plots import TimelinePlots, WaveformControl

_CLIP_MARGIN_S = 1.5


class ResultView(QWidget):
    export_requested = Signal()
    open_folder_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.verdict = QLabel()
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict.setMinimumWidth(180)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 13px; color: #333;")
        export = QPushButton("匯出報告…")
        export.clicked.connect(self.export_requested.emit)
        folder = QPushButton("開啟錄音資料夾")
        folder.clicked.connect(self.open_folder_requested.emit)
        reset = QPushButton("全部顯示")
        reset.clicked.connect(lambda: self.plots.reset_view())
        top = QHBoxLayout()
        top.addWidget(self.verdict)
        top.addWidget(self.summary, 1)
        self.plots = TimelinePlots()
        self.wave_control = WaveformControl(self.plots)
        controls = QHBoxLayout()
        controls.addWidget(self.wave_control)
        controls.addStretch(1)
        for button in (reset, folder, export):
            controls.addWidget(button)

        self.events = QTableWidget(0, 6)
        self.events.setHorizontalHeaderLabels(
            ["錄音時間", "等級", "類型", "曲目", "曲內位置", "說明"]
        )
        header = self.events.horizontalHeader()
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        for col in range(5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.events.verticalHeader().setVisible(False)
        self.events.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.events.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.events.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.events.itemSelectionChanged.connect(self._on_event_selected)

        self.play_rec = QPushButton("▶ 播放錄音")
        self.play_ref = QPushButton("▶ 播放參考")
        stop = QPushButton("■ 停止")
        self.play_rec.clicked.connect(self._play_recording)
        self.play_ref.clicked.connect(self._play_reference)
        stop.clicked.connect(stop_playback)
        self.show_info = QCheckBox("顯示曲間空白")
        self.show_info.setChecked(False)
        self.show_info.toggled.connect(self._fill_events)
        listen = QHBoxLayout()
        listen.addWidget(QLabel("選一個事件來對照試聽："))
        for widget in (self.play_rec, self.play_ref, stop):
            listen.addWidget(widget)
        listen.addStretch(1)
        listen.addWidget(self.show_info)
        events_page = QWidget()
        events_layout = QVBoxLayout(events_page)
        events_layout.setContentsMargins(0, 4, 0, 0)
        events_layout.addWidget(self.events, 1)
        events_layout.addLayout(listen)

        self.tracks = QTableWidget(0, 7)
        self.tracks.setHorizontalHeaderLabels(
            ["#", "曲目", "長度", "結果", "聽到", "問題", "待確認"]
        )
        self.tracks.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.tracks.verticalHeader().setVisible(False)
        self.tracks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tracks.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tracks.itemSelectionChanged.connect(self._on_track_selected)

        self.tabs = QTabWidget()
        self.tabs.addTab(events_page, "事件")
        self.tabs.addTab(self.tracks, "各曲結果")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.plots)
        splitter.addWidget(self.tabs)
        splitter.setSizes([460, 280])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)

        self.result: AnalysisResult | None = None
        self.recording_path: Path | None = None
        self._shown: list[DetectedEvent] = []
        self._set_listen_enabled(None)

    # ------------------------------------------------------------------ content
    def show_result(self, result: AnalysisResult, recording_path: Path | None) -> None:
        self.result = result
        self.recording_path = recording_path
        s = result.summary
        color = theme.VERDICT_COLOR[s.verdict]
        self.verdict.setText(VERDICT_TEXT[s.verdict])
        self.verdict.setStyleSheet(
            f"background: {color}; color: white; font-size: 22px; font-weight: bold;"
            "border-radius: 8px; padding: 10px 16px;"
        )
        headline = f"問題 {s.fail_count} 個　·　待確認 {s.warn_count} 個"
        parts = [
            f"錄音 {format_clock(s.recording_duration_s, 0)}",
            "沒有找到播放"
            if s.playback_start_s is None
            else f"起播 {format_clock(s.playback_start_s)}",
            f"聽到 {100 * s.heard_ratio:.0f}%",
        ]
        if s.signal_to_noise_db is not None:
            parts.append(f"訊噪比 {s.signal_to_noise_db:.0f} dB")
        if s.drift_ppm is not None:
            parts.append(f"時脈差 {s.drift_ppm:+.0f} ppm")
        self.summary.setText(
            f"<div style='font-size:16px; font-weight:bold;'>{headline}</div>"
            f"<div style='color:#555;'>{'　·　'.join(parts)}</div>"
        )
        self.plots.show_result(result)
        self._fill_events()
        self._fill_tracks()
        self.tabs.setCurrentIndex(0)

    def _fill_events(self) -> None:
        if self.result is None:
            return
        show_info = self.show_info.isChecked()
        self._shown = [
            e for e in self.result.events if show_info or e.severity != Severity.INFO
        ]
        self.events.setRowCount(len(self._shown))
        for row, event in enumerate(self._shown):
            color = QColor(theme.SEVERITY_COLOR[event.severity])
            cells = [
                format_clock(event.rec_start_s, 2),
                SEVERITY_TEXT[event.severity],
                EVENT_LABELS[event.event_type],
                "" if event.track_index is None else str(event.track_index + 1),
                ""
                if event.track_position_s is None
                else format_clock(event.track_position_s, 1),
                event.detail,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col in (1, 2):
                    item.setForeground(color)
                self.events.setItem(row, col, item)
        self._set_listen_enabled(None)

    def _fill_tracks(self) -> None:
        tracks = self.result.tracks if self.result else []
        self.tracks.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            cells = [
                str(track.index + 1),
                track.title,
                format_clock(track.duration_s, 0),
                TRACK_VERDICT_TEXT[track.verdict],
                format_clock(track.heard_s, 0),
                str(track.fail_count),
                str(track.warn_count),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 3:
                    item.setForeground(QColor(theme.TRACK_COLOR[track.verdict]))
                self.tracks.setItem(row, col, item)

    # ------------------------------------------------------------------ interaction
    def _selected_event(self) -> DetectedEvent | None:
        rows = {index.row() for index in self.events.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._shown[row] if row < len(self._shown) else None

    def _on_event_selected(self) -> None:
        event = self._selected_event()
        self._set_listen_enabled(event)
        if event is None:
            self.plots.clear_highlight()
            return
        self.plots.focus(event.rec_start_s, event.rec_end_s)
        self.plots.highlight(event.rec_start_s, event.rec_end_s)

    def _on_track_selected(self) -> None:
        rows = {index.row() for index in self.tracks.selectedIndexes()}
        if self.result is None or len(rows) != 1:
            return
        track = self.result.tracks[rows.pop()]
        if track.rec_start_s is not None and track.rec_end_s is not None:
            self.plots.focus(track.rec_start_s, track.rec_end_s, margin_s=3.0)

    def _set_listen_enabled(self, event: DetectedEvent | None) -> None:
        has_recording = (
            self.recording_path is not None and self.recording_path.is_file()
        )
        self.play_rec.setEnabled(event is not None and has_recording)
        self.play_ref.setEnabled(event is not None and event.track_index is not None)

    def _clip_span(self, event: DetectedEvent) -> float:
        return max(3.0, event.rec_end_s - event.rec_start_s + 2 * _CLIP_MARGIN_S)

    def _play_recording(self) -> None:
        event = self._selected_event()
        if event is None or self.recording_path is None:
            return
        data, rate = read_clip(
            self.recording_path,
            event.rec_start_s - _CLIP_MARGIN_S,
            self._clip_span(event),
        )
        play_clip(data, rate)

    def _play_reference(self) -> None:
        event = self._selected_event()
        if event is None or event.track_index is None or self.result is None:
            return
        track = self.result.program_tracks[event.track_index]
        start = (event.track_position_s or 0.0) - _CLIP_MARGIN_S
        data, rate = read_clip(
            track.source_path, max(0.0, start), self._clip_span(event)
        )
        play_clip(data, rate)

    def plot_png(self) -> bytes:
        pixmap = self.plots.grab()
        buffer = QByteArray()
        device = QBuffer(buffer)
        device.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(device, "PNG")
        return bytes(buffer.data())
