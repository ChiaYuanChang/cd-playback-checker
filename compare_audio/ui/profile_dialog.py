"""Create or edit a test profile: name and the reference tracks in play order."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.audio_io import audio_duration_s
from compare_audio.core.reference_program import TrackSource
from compare_audio.core.time_format import format_clock
from compare_audio.profiles.test_profile import TestProfile

AUDIO_FILTER = "音訊檔 (*.wav *.flac *.mp3 *.aif *.aiff *.ogg);;所有檔案 (*)"


class ProfileDialog(QDialog):
    def __init__(
        self, parent: QWidget | None, profile: TestProfile | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("編輯測試片" if profile else "新增測試片")
        self.resize(720, 460)
        self._base = profile
        self.name_edit = QLineEdit(profile.name if profile else "")
        self.name_edit.setPlaceholderText("例如：測試片 A（6 首）")
        self.description_edit = QLineEdit(profile.description if profile else "")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["曲名", "長度", "檔案"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(26)

        add, remove = QPushButton("加入音檔…"), QPushButton("移除")
        up, down = QPushButton("上移"), QPushButton("下移")
        add.clicked.connect(self._add_files)
        remove.clicked.connect(self._remove)
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        buttons = QHBoxLayout()
        for button in (add, remove, up, down):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.total_label = QLabel()
        buttons.addWidget(self.total_label)

        form = QFormLayout()
        form.addRow("名稱", self.name_edit)
        form.addRow("說明", self.description_edit)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        hint = QLabel(
            "依播放順序排列曲目。建議使用從同一張 CD 抓下的 WAV / FLAC 無損檔。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        layout.addWidget(hint)
        layout.addWidget(box)
        for track in profile.tracks if profile else []:
            self._append(track.title, track.path)
        self._refresh_total()

    def _append(self, title: str, path: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(title))
        try:
            duration = format_clock(audio_duration_s(path), 0)
        except Exception:
            duration = "無法讀取"
        length = QTableWidgetItem(duration)
        length.setFlags(length.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 1, length)
        item = QTableWidgetItem(path)
        item.setToolTip(path)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 2, item)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "選擇參考音檔", "", AUDIO_FILTER)
        for path in sorted(paths, key=lambda p: Path(p).name.lower()):
            self._append(Path(path).stem, path)
        self._refresh_total()

    def _remove(self) -> None:
        for row in sorted(
            {i.row() for i in self.table.selectedIndexes()}, reverse=True
        ):
            self.table.removeRow(row)
        self._refresh_total()

    def _move(self, step: int) -> None:
        row = self.table.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        for col in range(3):
            a, b = self.table.takeItem(row, col), self.table.takeItem(target, col)
            self.table.setItem(row, col, b)
            self.table.setItem(target, col, a)
        self.table.selectRow(target)

    def _refresh_total(self) -> None:
        total = 0.0
        for row in range(self.table.rowCount()):
            try:
                total += audio_duration_s(self.table.item(row, 2).text())
            except Exception:
                pass
        self.total_label.setText(
            f"共 {self.table.rowCount()} 首，{format_clock(total, 0)}"
        )

    def _accept(self) -> None:
        name = self.name_edit.text().strip()
        tracks = [
            TrackSource(
                title=self.table.item(row, 0).text().strip() or f"第 {row + 1} 首",
                path=self.table.item(row, 2).text(),
            )
            for row in range(self.table.rowCount())
        ]
        if not name:
            QMessageBox.warning(self, "缺少名稱", "請輸入測試片名稱。")
            return
        if not tracks:
            QMessageBox.warning(self, "沒有曲目", "請至少加入一個參考音檔。")
            return
        missing = [t.path for t in tracks if not Path(t.path).is_file()]
        if missing:
            QMessageBox.warning(
                self, "找不到檔案", "以下檔案不存在：\n" + "\n".join(missing)
            )
            return
        self._result = TestProfile(
            name=name,
            description=self.description_edit.text().strip(),
            tracks=tracks,
            analysis=self._base.analysis if self._base else AnalysisConfig(),
        )
        self.accept()

    def profile(self) -> TestProfile:
        return self._result
