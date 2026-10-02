"""Left panel: test disc, microphone (with level meter), options, start/stop."""

from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from compare_audio.audio.audio_sources import LevelMonitor
from compare_audio.audio.input_devices import (
    InputDevice,
    find_device,
    list_input_devices,
)
from compare_audio.core.reference_program import ReferenceProgram
from compare_audio.core.time_format import format_clock
from compare_audio.ui import theme
from compare_audio.ui.level_meter import LevelMeter
from compare_audio.ui.sensitivity_control import SensitivityControl

_QUIET_DBFS = -45.0


class SetupPanel(QWidget):
    profile_selected = Signal(object)  # Path | None
    new_profile = Signal()
    edit_profile = Signal()
    delete_profile = Signal()
    import_profile = Signal()
    start_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(340)
        self.setMaximumWidth(420)
        self._devices: list[InputDevice] = []
        self._monitor: LevelMonitor | None = None
        self._recording = False
        self._quiet_ticks = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll_monitor)

        # --- test disc
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_index)
        buttons = QHBoxLayout()
        for text, signal in (
            ("新增", self.new_profile),
            ("編輯", self.edit_profile),
            ("刪除", self.delete_profile),
            ("匯入…", self.import_profile),
        ):
            button = QPushButton(text)
            button.clicked.connect(signal.emit)
            buttons.addWidget(button)
        self.track_list = QListWidget()
        self.track_list.setMinimumHeight(120)
        self.profile_status = QLabel("請先新增或選擇測試片")
        self.profile_status.setWordWrap(True)
        disc = QGroupBox("1. 測試片（參考音訊）")
        disc_layout = QVBoxLayout(disc)
        disc_layout.addWidget(self.profile_combo)
        disc_layout.addLayout(buttons)
        disc_layout.addWidget(self.track_list)
        disc_layout.addWidget(self.profile_status)

        # --- microphone
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_index)
        refresh = QPushButton("重新整理")
        refresh.clicked.connect(self.refresh_devices)
        self.all_apis = QCheckBox("顯示所有驅動程式")
        self.all_apis.toggled.connect(self.refresh_devices)
        self.meter = LevelMeter()
        self.mic_hint = QLabel("")
        self.mic_hint.setWordWrap(True)
        mic = QGroupBox("2. 麥克風")
        mic_layout = QGridLayout(mic)
        mic_layout.addWidget(self.device_combo, 0, 0)
        mic_layout.addWidget(refresh, 0, 1)
        mic_layout.addWidget(self.meter, 1, 0, 1, 2)
        mic_layout.addWidget(self.mic_hint, 2, 0, 1, 2)
        mic_layout.addWidget(self.all_apis, 3, 0, 1, 2)

        # --- options
        self.auto_stop = QCheckBox("最後一首播完自動停止")
        self.auto_stop.setChecked(True)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        browse = QPushButton("…")
        browse.setFixedWidth(36)
        browse.clicked.connect(self._choose_folder)
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("序號 / 備註（會寫進報告）")
        options = QGroupBox("3. 選項")
        options_layout = QGridLayout(options)
        options_layout.addWidget(self.auto_stop, 0, 0, 1, 2)
        options_layout.addWidget(QLabel("錄音存放位置"), 1, 0, 1, 2)
        options_layout.addWidget(self.folder_edit, 2, 0)
        options_layout.addWidget(browse, 2, 1)
        options_layout.addWidget(self.notes_edit, 3, 0, 1, 2)
        self.sensitivity = SensitivityControl()
        options_layout.addWidget(QLabel("辨識靈敏度（當次測試）"), 4, 0, 1, 2)
        options_layout.addWidget(self.sensitivity, 5, 0, 1, 2)

        # --- start / stop
        self.start_button = QPushButton("開始錄音")
        self.start_button.setObjectName("startButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_clicked.emit)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setStyleSheet("color: #555;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 10)
        layout.addWidget(disc)
        layout.addWidget(mic)
        layout.addWidget(options)
        layout.addStretch(1)
        layout.addWidget(self.start_button)
        layout.addWidget(self.elapsed_label)

    # ------------------------------------------------------------------ profiles
    def set_profiles(
        self, entries: list[tuple[str, Path]], selected: Path | None
    ) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for name, path in entries:
            self.profile_combo.addItem(name, path)
        index = next((i for i, (_, p) in enumerate(entries) if p == selected), 0)
        self.profile_combo.setCurrentIndex(index if entries else -1)
        self.profile_combo.blockSignals(False)
        self._on_profile_index(self.profile_combo.currentIndex())

    def current_profile_path(self) -> Path | None:
        data = self.profile_combo.currentData()
        return Path(data) if data else None

    def _on_profile_index(self, index: int) -> None:
        self.profile_selected.emit(self.current_profile_path() if index >= 0 else None)

    def show_program(self, program: ReferenceProgram | None, status: str) -> None:
        self.track_list.clear()
        if program is not None:
            for track in program.tracks:
                length = format_clock(track.duration_s, 0)
                self.track_list.addItem(f"{track.index + 1}. {track.title}  ({length})")
        self.profile_status.setText(status)
        self._update_start()

    def set_profile_status(self, text: str) -> None:
        self.profile_status.setText(text)

    # ------------------------------------------------------------------ devices
    def refresh_devices(self) -> None:
        current = self.current_device()
        self._devices = list_input_devices(all_apis=self.all_apis.isChecked())
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device in self._devices:
            self.device_combo.addItem(device.label(self.all_apis.isChecked()))
        chosen = find_device(self._devices, current.name) if current else None
        chosen = chosen or next((d for d in self._devices if d.is_default), None)
        if chosen is not None:
            self.device_combo.setCurrentIndex(self._devices.index(chosen))
        self.device_combo.blockSignals(False)
        if not self._devices:
            self.mic_hint.setText("找不到任何收音裝置。請接上麥克風後按「重新整理」。")
        self._on_device_index(self.device_combo.currentIndex())

    def select_device_by_name(self, name: str) -> None:
        device = find_device(self._devices, name)
        if device is not None:
            self.device_combo.setCurrentIndex(self._devices.index(device))

    def current_device(self) -> InputDevice | None:
        index = self.device_combo.currentIndex()
        return self._devices[index] if 0 <= index < len(self._devices) else None

    def _on_device_index(self, _: int) -> None:
        self.stop_monitor()
        if not self._recording:
            self.start_monitor()
        self._update_start()

    def start_monitor(self) -> None:
        device = self.current_device()
        if device is None:
            return
        try:
            self._monitor = LevelMonitor(device)
            self._monitor.start()
        except Exception as exc:
            self._monitor = None
            self.mic_hint.setText(f"無法開啟麥克風：{exc}")
            self.mic_hint.setStyleSheet(f"color: {theme.FAIL};")
            return
        self._quiet_ticks = 0
        self.mic_hint.setText(f"{device.default_rate:.0f} Hz，對著喇叭測試一下音量")
        self.mic_hint.setStyleSheet("color: #555;")
        self._timer.start()

    def stop_monitor(self) -> None:
        self._timer.stop()
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        self.meter.reset()

    def _poll_monitor(self) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        clipped = monitor.take_clip()
        self.meter.set_level(monitor.peak_db, monitor.rms_db, clipped)
        self._quiet_ticks = (
            self._quiet_ticks + 1 if monitor.peak_db < _QUIET_DBFS else 0
        )
        if monitor.all_zero:
            self._hint(
                "收到的訊號全是 0：請確認系統允許此程式使用麥克風"
                "（Windows：設定 → 隱私權 → 麥克風）",
                theme.FAIL,
            )
        elif clipped:
            self._hint("音量太大會爆音，請調低麥克風增益或拉開一點距離", theme.FAIL)
        elif self._quiet_ticks > 60:
            self._hint("收音很小聲：麥克風靠近喇叭一點，或調高增益", theme.WARN)
        elif monitor.peak_db > -30:
            self._hint("收音正常", theme.PASS)

    def _hint(self, text: str, color: str) -> None:
        self.mic_hint.setText(text)
        self.mic_hint.setStyleSheet(f"color: {color};")

    def show_session_level(self, peak_db: float, rms_db: float, clipped: bool) -> None:
        self.meter.set_level(peak_db, rms_db, clipped)

    # ------------------------------------------------------------------ options
    def recordings_folder(self) -> Path:
        return Path(self.folder_edit.text())

    def set_recordings_folder(self, folder: Path) -> None:
        self.folder_edit.setText(str(folder))
        self.folder_edit.setToolTip(str(folder))

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "錄音存放位置", self.folder_edit.text()
        )
        if folder:
            self.set_recordings_folder(Path(folder))

    # ------------------------------------------------------------------ state
    def set_program_ready(self, ready: bool) -> None:
        self._program_ready = ready
        self._update_start()

    def _update_start(self) -> None:
        if self._recording:
            return
        ready = (
            getattr(self, "_program_ready", False) and self.current_device() is not None
        )
        self.start_button.setEnabled(ready)

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        self.start_button.setProperty("recording", recording)
        self.start_button.setText("停止" if recording else "開始錄音")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        for widget in (
            self.profile_combo,
            self.device_combo,
            self.all_apis,
            self.track_list,
            self.sensitivity,
        ):
            widget.setEnabled(not recording)
        if recording:
            self.stop_monitor()
            self.start_button.setEnabled(True)
        else:
            self.elapsed_label.setText("")
            self.start_monitor()
            self._update_start()

    def set_busy(self, text: str) -> None:
        """Recording stopped, analysis running."""
        self.start_button.setEnabled(False)
        self.start_button.setText(text)

    def set_elapsed(self, seconds: float) -> None:
        self.elapsed_label.setText(f"已錄 {format_clock(seconds, 0)}")
