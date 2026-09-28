"""Main window: wires the setup panel, the workers and the live/result pages."""

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from compare_audio import __version__
from compare_audio.audio.audio_sources import AudioSource, FileSource, MicrophoneSource
from compare_audio.core.analysis_models import AnalysisResult, DetectedEvent
from compare_audio.core.reference_program import ReferenceProgram
from compare_audio.core.report_export import SessionInfo, export_report, write_json
from compare_audio.core.time_format import format_clock
from compare_audio.profiles.test_profile import (
    ProfileError,
    TestProfile,
    list_profiles,
    load_profile,
    profile_filename,
    save_profile,
)
from compare_audio.ui import app_paths
from compare_audio.ui.live_view import LiveView
from compare_audio.ui.profile_dialog import AUDIO_FILTER, ProfileDialog
from compare_audio.ui.result_view import ResultView
from compare_audio.ui.setup_panel import SetupPanel
from compare_audio.ui.workers import (
    FileAnalysisWorker,
    LiveUpdate,
    ReferenceLoader,
    SessionWorker,
)

WELCOME = """
<h2>CD 播放測試</h2>
<ol style="font-size:14px; line-height:170%;">
<li>左上選擇<b>測試片</b>（第一次請按「新增」，依順序加入那張 CD 的參考音檔）</li>
<li>選擇<b>麥克風</b>，對著喇叭看音量表，確認有收到聲音、沒有爆音</li>
<li>按<b>開始錄音</b>，再去 CD 播放器按播放（不用同時，程式會自己找到開頭）</li>
<li>最後一首播完會自動停止，也可以隨時按<b>停止</b></li>
</ol>
<p style="color:#666; font-size:13px;">
小技巧：麥克風固定在喇叭前 5～10 公分、關掉 Windows 的「音訊增強 / 噪音抑制」，
結果最準。沒有 CD 機也可以用「檔案 → 用錄音檔模擬收音…」試用。
</p>
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"CD 播放測試  v{__version__}")
        self.resize(1360, 860)
        self.settings = QSettings()
        self.profile: TestProfile | None = None
        self.profile_path: Path | None = None
        self.program: ReferenceProgram | None = None
        self.loader: ReferenceLoader | None = None
        self.session: SessionWorker | None = None
        self.file_worker: FileAnalysisWorker | None = None
        self.session_info: SessionInfo | None = None
        self.result: AnalysisResult | None = None
        self._threads: set[QThread] = set()  # keep running workers alive

        self.setup = SetupPanel()
        self.live = LiveView()
        self.results = ResultView()
        welcome = QLabel(WELCOME)
        welcome.setAlignment(Qt.AlignmentFlag.AlignTop)
        welcome.setWordWrap(True)
        welcome.setContentsMargins(30, 20, 30, 20)
        welcome_page = QWidget()
        QVBoxLayout(welcome_page).addWidget(welcome)
        self.pages = QStackedWidget()
        for page in (welcome_page, self.live, self.results):
            self.pages.addWidget(page)
        splitter = QSplitter()
        splitter.addWidget(self.setup)
        splitter.addWidget(self.pages)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])
        self.setCentralWidget(splitter)
        self._build_menu()

        self.setup.profile_selected.connect(self._on_profile_selected)
        self.setup.new_profile.connect(self._new_profile)
        self.setup.edit_profile.connect(self._edit_profile)
        self.setup.delete_profile.connect(self._delete_profile)
        self.setup.import_profile.connect(self._import_profile)
        self.setup.start_clicked.connect(self._on_start_clicked)
        self.results.export_requested.connect(self._export)
        self.results.open_folder_requested.connect(self._open_recording_folder)

        folder = self.settings.value("recordings_dir", "")
        self.setup.set_recordings_folder(
            Path(folder) if folder else app_paths.default_recordings_dir()
        )
        self.setup.auto_stop.setChecked(
            self.settings.value("auto_stop", True, type=bool)
        )
        self.setup.refresh_devices()
        device = self.settings.value("device", "")
        if device:
            self.setup.select_device_by_name(device)
        self._reload_profiles(self._saved_profile_path())
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("檔案")
        actions = [
            ("分析錄音檔…", "Ctrl+O", self._analyze_file),
            ("用錄音檔模擬收音…", None, self._simulate_from_file),
            (None, None, None),
            ("匯出報告…", "Ctrl+E", self._export),
            ("開啟錄音資料夾", None, self._open_recording_folder),
            (None, None, None),
            ("結束", QKeySequence.StandardKey.Quit, self.close),
        ]
        for text, shortcut, slot in actions:
            if text is None:
                file_menu.addSeparator()
                continue
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        help_menu = self.menuBar().addMenu("說明")
        about = QAction("關於", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "關於",
            f"CD 播放測試 v{__version__}\n\n"
            "用麥克風錄下 CD 播放器的聲音，和參考音檔自動對齊，"
            "找出跳過、重播、卡住、停頓與瞬間無聲。",
        )

    # ------------------------------------------------------------------ profiles
    def _saved_profile_path(self) -> Path | None:
        value = self.settings.value("profile", "")
        return Path(value) if value else None

    def _reload_profiles(self, selected: Path | None) -> None:
        entries: list[tuple[str, Path]] = []
        for path in list_profiles(app_paths.profiles_dir()):
            try:
                entries.append((load_profile(path).name, path))
            except ProfileError:
                continue
        self.setup.set_profiles(entries, selected)
        if not entries:
            self.setup.show_program(None, "還沒有測試片，請按「新增」或「匯入…」")

    def _on_profile_selected(self, path: Path | None) -> None:
        self.program = None
        self.setup.set_program_ready(False)
        if path is None:
            self.profile = self.profile_path = None
            return
        try:
            self.profile = load_profile(path)
        except ProfileError as exc:
            self.setup.show_program(None, str(exc))
            return
        self.profile_path = path
        self.settings.setValue("profile", str(path))
        self.setup.show_program(None, "載入參考音檔中…")
        loader = ReferenceLoader(self.profile, app_paths.cache_dir())
        loader.progress.connect(
            lambda f: self.setup.set_profile_status(f"載入參考音檔中… {100 * f:.0f}%")
        )
        loader.loaded.connect(
            lambda program, p=path: self._on_program_loaded(p, program)
        )
        loader.failed.connect(lambda message: self.setup.show_program(None, message))
        self.loader = loader
        self._run(loader)

    def _on_program_loaded(self, path: Path, program: ReferenceProgram) -> None:
        if path != self.profile_path:
            return  # a newer selection is loading
        self.program = program
        total = format_clock(program.duration_s, 0)
        self.setup.show_program(
            program, f"已就緒：{len(program.tracks)} 首，共 {total}"
        )
        self.setup.set_program_ready(True)

    def _new_profile(self) -> None:
        dialog = ProfileDialog(self)
        if dialog.exec():
            profile = dialog.profile()
            path = app_paths.profiles_dir() / profile_filename(profile.name)
            if path.exists() and not self._confirm(
                f"已有同名的測試片「{profile.name}」，要覆蓋嗎？"
            ):
                return
            save_profile(profile, path)
            self._reload_profiles(path)

    def _edit_profile(self) -> None:
        if self.profile is None or self.profile_path is None:
            return
        dialog = ProfileDialog(self, self.profile)
        if dialog.exec():
            save_profile(dialog.profile(), self.profile_path)
            self._reload_profiles(self.profile_path)

    def _delete_profile(self) -> None:
        if self.profile is None or self.profile_path is None:
            return
        if self._confirm(f"刪除測試片「{self.profile.name}」？（參考音檔不會被刪除）"):
            self.profile_path.unlink(missing_ok=True)
            self._reload_profiles(None)

    def _import_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "匯入測試片設定", "", "測試片設定 (*.profile.json *.json)"
        )
        if not path:
            return
        try:
            profile = load_profile(path)
        except ProfileError as exc:
            QMessageBox.warning(self, "無法匯入", str(exc))
            return
        target = app_paths.profiles_dir() / profile_filename(profile.name)
        save_profile(profile, target)
        self._reload_profiles(target)

    def _confirm(self, text: str) -> bool:
        answer = QMessageBox.question(self, "確認", text)
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ recording
    def _on_start_clicked(self) -> None:
        if self.session is not None:
            self.session.request_stop()
            self.setup.set_busy("停止中…")
            return
        device = self.setup.current_device()
        if device is None or self.program is None:
            return
        self._start_session(MicrophoneSource(device), device.name)

    def _simulate_from_file(self) -> None:
        if self.session is not None:
            return
        if self.program is None:
            QMessageBox.information(self, "尚未就緒", "請先選擇測試片並等它載入完成。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇要當作收音的錄音檔", "", AUDIO_FILTER
        )
        if not path:
            return
        speeds = ["1 倍（真實時間）", "4 倍速", "16 倍速"]
        choice, ok = QInputDialog.getItem(
            self, "播放速度", "模擬速度：", speeds, 1, False
        )
        if not ok:
            return
        speed = {0: 1.0, 1: 4.0, 2: 16.0}[speeds.index(choice)]
        self._start_session(FileSource(path, speed), f"模擬：{Path(path).name}")

    def _start_session(self, source: AudioSource, device_label: str) -> None:
        assert self.program is not None and self.profile is not None
        folder = self.setup.recordings_folder()
        stamp = datetime.now()
        safe = profile_filename(self.profile.name).removesuffix(".profile.json")
        wav_path = folder / f"{stamp:%Y%m%d-%H%M%S}_{safe}.wav"
        self.session_info = SessionInfo(
            profile_name=self.profile.name,
            recording_path=str(wav_path),
            started_at=stamp.isoformat(timespec="seconds"),
            device=device_label,
            notes=self.setup.notes_edit.text().strip(),
        )
        worker = SessionWorker(
            self.program, source, wav_path, self.setup.auto_stop.isChecked()
        )
        worker.level.connect(self.setup.show_session_level)
        worker.updated.connect(self._on_live_update)
        worker.provisional.connect(self._on_provisional)
        worker.auto_stopping.connect(
            lambda reason: self.statusBar().showMessage(f"自動停止：{reason}", 8000)
        )
        worker.analysing.connect(self._on_analysing)
        worker.analysis_progress.connect(self.live.set_analysing)
        worker.completed.connect(self._on_session_completed)
        worker.failed.connect(self._on_session_failed)
        worker.finished.connect(self._on_session_thread_finished)
        self.session = worker
        self.settings.setValue(
            "device",
            self.setup.current_device().name if self.setup.current_device() else "",
        )
        self.setup.set_recording(True)
        self.live.start(self.program, f"收音：{device_label}")
        self.pages.setCurrentWidget(self.live)
        self._run(worker)

    def _on_live_update(self, update: LiveUpdate) -> None:
        self.live.update_live(update)
        self.setup.set_elapsed(update.status.rec_s)

    def _on_provisional(self, events: list[DetectedEvent]) -> None:
        self.live.set_events(events)

    def _on_analysing(self) -> None:
        self.setup.set_busy("分析中…")
        self.live.set_analysing(0.0)

    def _on_session_completed(self, result: AnalysisResult, wav_path: str) -> None:
        path = Path(wav_path)
        if self.session_info is not None:
            write_json(result, self.session_info, path.with_suffix(".result.json"))
        self._show_result(result, path)

    def _on_session_failed(self, message: str) -> None:
        QMessageBox.critical(self, "錄音失敗", message)
        self.pages.setCurrentIndex(0)

    def _on_session_thread_finished(self) -> None:
        self.session = None
        self.live.set_analysing(None)
        self.setup.set_recording(False)

    # ------------------------------------------------------------------ files
    def _analyze_file(self) -> None:
        if self.program is None or self.profile is None:
            QMessageBox.information(self, "尚未就緒", "請先選擇測試片並等它載入完成。")
            return
        folder = str(self.setup.recordings_folder())
        path, _ = QFileDialog.getOpenFileName(self, "選擇錄音檔", folder, AUDIO_FILTER)
        if not path:
            return
        progress = QProgressDialog("分析中…", None, 0, 100, self)
        progress.setWindowTitle("分析錄音檔")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        worker = FileAnalysisWorker(self.program, Path(path))
        worker.progress.connect(lambda f: progress.setValue(int(100 * f)))
        worker.completed.connect(
            lambda result, p: self._on_file_analysed(result, Path(p))
        )
        worker.failed.connect(
            lambda message: QMessageBox.critical(self, "分析失敗", message)
        )
        worker.finished.connect(progress.close)
        self.file_worker = worker
        self._run(worker)

    def _on_file_analysed(self, result: AnalysisResult, path: Path) -> None:
        stamp = datetime.fromtimestamp(path.stat().st_mtime)
        self.session_info = SessionInfo(
            profile_name=self.profile.name if self.profile else "",
            recording_path=str(path),
            started_at=stamp.isoformat(timespec="seconds"),
            device="（分析既有錄音檔）",
            notes=self.setup.notes_edit.text().strip(),
        )
        self._show_result(result, path)

    def _show_result(self, result: AnalysisResult, path: Path) -> None:
        self.result = result
        self.results.show_result(result, path)
        self.pages.setCurrentWidget(self.results)
        verdict = result.summary.verdict.value.upper()
        self.statusBar().showMessage(f"分析完成：{verdict}，錄音檔 {path}", 15000)

    def _export(self) -> None:
        if self.result is None or self.session_info is None:
            QMessageBox.information(self, "沒有結果", "還沒有可以匯出的分析結果。")
            return
        base = Path(self.session_info.recording_path or self.setup.recordings_folder())
        suggested = base.with_name(base.stem + "_report") if base.suffix else base
        folder = QFileDialog.getExistingDirectory(
            self, "報告要存到哪個資料夾", str(suggested.parent)
        )
        if not folder:
            return
        target = Path(folder) / suggested.name
        html = export_report(
            self.result, self.session_info, target, self.results.plot_png()
        )
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(html)))

    def _open_recording_folder(self) -> None:
        path = self.setup.recordings_folder()
        if self.session_info and self.session_info.recording_path:
            path = Path(self.session_info.recording_path).parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ lifecycle
    def _run(self, thread: QThread) -> None:
        self._threads.add(thread)
        thread.finished.connect(lambda t=thread: self._threads.discard(t))
        thread.start()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.session is not None:
            if not self._confirm(
                "錄音中，確定要結束程式嗎？（錄音檔會保留，但不會分析）"
            ):
                event.ignore()
                return
            self.session.request_stop()
            self.session.wait()
        self.setup.stop_monitor()
        for thread in list(self._threads):
            thread.wait()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("recordings_dir", str(self.setup.recordings_folder()))
        self.settings.setValue("auto_stop", self.setup.auto_stop.isChecked())
        super().closeEvent(event)
