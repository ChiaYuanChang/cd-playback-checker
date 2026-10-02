"""What the user sees while recording: where the player is, and early warnings."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from compare_audio.core.analysis_models import (
    EVENT_LABELS,
    DetectedEvent,
    Severity,
    TrackReport,
)
from compare_audio.core.live_analysis import LiveDetection
from compare_audio.core.reference_program import ReferenceProgram
from compare_audio.core.time_format import format_clock
from compare_audio.ui import theme
from compare_audio.ui.plots import TimelinePlots, WaveformControl
from compare_audio.ui.workers import LiveUpdate


class LiveView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.status_label = QLabel("等待音樂開始…")
        self.status_label.setStyleSheet("font-size: 26px; font-weight: bold;")
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #555; font-size: 14px;")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)

        self.tracks = QTableWidget(0, 3)
        self.tracks.setHorizontalHeaderLabels(["曲目", "進度", "狀態"])
        header = self.tracks.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tracks.verticalHeader().setVisible(False)
        self.tracks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.events = QTableWidget(0, 3)
        self.events.setHorizontalHeaderLabels(["錄音時間", "類型", "說明（初步）"])
        events_header = self.events.horizontalHeader()
        events_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        events_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.events.verticalHeader().setVisible(False)
        self.events.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.plots = TimelinePlots()
        self.follow = QCheckBox("自動跟隨")
        self.follow.setChecked(True)
        self.plots.manual_navigation.connect(lambda: self.follow.setChecked(False))
        self.wave_control = WaveformControl(self.plots)
        reset = QPushButton("全部顯示")
        reset.clicked.connect(self.plots.reset_view)
        self.follow.toggled.connect(self._follow_changed)
        controls = QHBoxLayout()
        controls.addWidget(self.follow)
        controls.addWidget(reset)
        controls.addWidget(self.wave_control)
        controls.addStretch(1)
        self.diagnostics = QLabel("辨識資訊：等待收音")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setStyleSheet("font-size: 12px; color: #555;")
        self.diagnostics.setMinimumHeight(36)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self.tracks)
        bottom.addWidget(self.events)
        bottom.setSizes([380, 520])
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.plots)
        splitter.addWidget(bottom)
        splitter.setSizes([420, 260])

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.progress)
        layout.addLayout(controls)
        layout.addWidget(self.diagnostics)
        layout.addWidget(splitter, 1)
        self._program: ReferenceProgram | None = None
        self._bars: list[QProgressBar] = []

    def start(self, program: ReferenceProgram, source_text: str) -> None:
        self._program = program
        self.status_label.setText("等待音樂開始…")
        self.status_label.setStyleSheet("font-size: 26px; font-weight: bold;")
        self.detail_label.setText(
            f"{source_text}。可以去按播放了，程式會自己找到開頭。"
        )
        self.progress.setVisible(False)
        self.follow.setChecked(True)
        self.diagnostics.setText("辨識資訊：等待收音；初步判定每 5 秒更新")
        self.plots.clear_all()
        self.plots.set_tracks(program.tracks)
        self.plots.align.setYRange(0, program.duration_s)
        self.plots.align.setXRange(0, 60)
        self.events.setRowCount(0)
        self.tracks.setRowCount(len(program.tracks))
        self._bars = []
        for track in program.tracks:
            self.tracks.setItem(
                track.index, 0, QTableWidgetItem(f"{track.index + 1}. {track.title}")
            )
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            self.tracks.setCellWidget(track.index, 1, bar)
            self._bars.append(bar)
            self.tracks.setItem(track.index, 2, QTableWidgetItem("等待判定"))

    def update_live(self, update: LiveUpdate) -> None:
        program = self._program
        if program is None:
            return
        status = update.status
        elapsed = format_clock(status.rec_s, 0)
        if status.locked and status.track_index is not None:
            track = program.tracks[status.track_index]
            position = format_clock(status.track_position_s or 0, 0)
            length = format_clock(track.duration_s, 0)
            self.status_label.setText(f"第 {track.index + 1} 首　{position} / {length}")
            self.status_label.setStyleSheet(
                f"font-size: 26px; font-weight: bold; color: {theme.TEXT};"
            )
            self.detail_label.setText(f"{track.title}　·　已錄 {elapsed}")
        elif np.isfinite(update.program_s).any():
            self.status_label.setText("暫時對不上")
            self.status_label.setStyleSheet(
                f"font-size: 26px; font-weight: bold; color: {theme.WARN};"
            )
            self.detail_label.setText(
                f"可能是跳針、停頓、曲間空白或環境噪音　·　已錄 {elapsed}"
            )
        else:
            self.detail_label.setText(f"還沒聽到參考音訊　·　已錄 {elapsed}")
        if update.overflows:
            self.detail_label.setText(
                self.detail_label.text() + f"　·　⚠ 錄音緩衝溢位 {update.overflows} 次"
            )
        span = max(60.0, status.rec_s + 5.0)
        self.plots.set_live(update.rec_s, update.program_s, span)
        if self.follow.isChecked():
            self.plots.align.setXRange(0, span, padding=0)
        self.plots.set_live_level(update.level_t, update.level_db)

    def _follow_changed(self, checked: bool) -> None:
        if checked:
            self.plots.reset_view()

    def set_detection(self, detection: LiveDetection) -> None:
        self.set_events(detection.events)
        self._update_tracks(detection.tracks)
        self.plots.set_envelopes(detection.envelopes, detection.noise_floor_db)
        score = "—" if detection.match_score is None else f"{detection.match_score:.2f}"
        snr = (
            "尚未對齊"
            if detection.signal_to_noise_db is None
            else f"{detection.signal_to_noise_db:.1f} dB"
        )
        sensitivity = self._program.config.sensitivity if self._program else 50
        hint = ""
        if (
            detection.signal_to_noise_db is not None
            and detection.signal_to_noise_db < 12
        ):
            hint = " · 收音接近背景音，可提高靈敏度或將麥克風靠近喇叭"
        elif detection.match_score is not None and self._program is not None:
            if detection.match_score < self._program.config.effective_match_threshold:
                hint = " · 目前匹配較弱，請確認參考曲目與收音"
        self.diagnostics.setText(
            f"靈敏度 {sensitivity} · 匹配分數 {score} · "
            f"背景音 {detection.noise_floor_db:.1f} dB · 訊噪比 {snr}"
            " · 初步判定每 5 秒更新" + hint
        )

    def _update_tracks(self, reports: list[TrackReport]) -> None:
        for report in reports:
            fraction = min(1.0, report.heard_s / max(report.duration_s, 1.0))
            self._bars[report.index].setValue(int(1000 * fraction))
            if report.fail_count:
                text = "有問題"
            elif report.warn_count:
                text = "需確認"
            elif fraction > 0.9:
                text = "已播"
            elif report.heard_s > 0:
                text = "部分"
            else:
                text = "尚未確認"
            self.tracks.item(report.index, 2).setText(text)

    def set_events(self, events: list[DetectedEvent]) -> None:
        shown = [e for e in events if e.severity != Severity.INFO]
        self.events.setRowCount(len(shown))
        for row, event in enumerate(shown):
            color = QColor(theme.SEVERITY_COLOR[event.severity])
            cells = [
                format_clock(event.rec_start_s, 1),
                EVENT_LABELS[event.event_type],
                event.detail,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 1:
                    item.setForeground(color)
                self.events.setItem(row, col, item)
        self.events.scrollToBottom()

    def set_analysing(self, fraction: float | None) -> None:
        self.progress.setVisible(fraction is not None)
        if fraction is not None:
            self.status_label.setText("分析中…")
            self.status_label.setStyleSheet("font-size: 26px; font-weight: bold;")
            self.detail_label.setText("錄音已停止，正在做完整分析")
            self.progress.setValue(int(100 * fraction))
