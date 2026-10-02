"""A shared recognition sensitivity control for profiles and session setup."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class SensitivityControl(QWidget):
    def __init__(self, value: int = 50, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(value)
        self.slider.setAccessibleName("辨識靈敏度")
        self.value_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(QLabel("保守"))
        row.addWidget(self.slider, 1)
        row.addWidget(QLabel("靈敏"))
        row.addWidget(self.value_label)
        hint = QLabel(
            "小聲收音可提高靈敏度；過高可能對錯位置。\n50 為標準，錄音中不能調整。"
        )
        hint.setWordWrap(True)
        hint.setMinimumHeight(34)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(hint)
        self.slider.valueChanged.connect(self._update_label)
        self._update_label(value)

    def _update_label(self, value: int) -> None:
        self.value_label.setText(str(value))

    def value(self) -> int:
        return self.slider.value()
