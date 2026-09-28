"""Horizontal input level meter with peak hold and a clip lamp."""

import time

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from compare_audio.ui import theme

_FLOOR_DB = -60.0
_HOLD_S = 1.5
_CLIP_S = 1.5


class LevelMeter(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._rms_db = _FLOOR_DB
        self._peak_db = _FLOOR_DB
        self._hold_db = _FLOOR_DB
        self._hold_at = 0.0
        self._clip_at = -10.0

    def set_level(self, peak_db: float, rms_db: float, clipped: bool = False) -> None:
        now = time.monotonic()
        self._peak_db = max(_FLOOR_DB, peak_db)
        self._rms_db = max(_FLOOR_DB, rms_db)
        if self._peak_db >= self._hold_db or now - self._hold_at > _HOLD_S:
            self._hold_db, self._hold_at = self._peak_db, now
        if clipped:
            self._clip_at = now
        self.update()

    def reset(self) -> None:
        self.set_level(_FLOOR_DB, _FLOOR_DB)

    @staticmethod
    def _x(db: float, width: float) -> float:
        return width * (db - _FLOOR_DB) / -_FLOOR_DB

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        lamp = 46
        width, height = self.width() - lamp - 6, self.height()
        painter.setPen(QColor(theme.BORDER))
        painter.setBrush(QColor("#fbfbfc"))
        painter.drawRoundedRect(QRectF(0.5, 0.5, width, height - 1), 4, 4)
        # zones: green up to -18 dBFS, amber to -6, red above
        for low, high, color in (
            (-60, -18, "#4caf50"),
            (-18, -6, "#f2b01e"),
            (-6, 0, "#e53935"),
        ):
            if self._rms_db <= low:
                continue
            x0, x1 = self._x(low, width), self._x(min(high, self._rms_db), width)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(
                QRectF(x0 + 2, 4, max(0.0, x1 - x0 - 2), height - 8), 2, 2
            )
        painter.setPen(QColor("#333333"))
        hold_x = self._x(self._hold_db, width)
        painter.drawLine(int(hold_x), 3, int(hold_x), height - 3)
        for db in (-48, -36, -24, -12, -6):
            x = int(self._x(db, width))
            painter.setPen(QColor("#c7cad0"))
            painter.drawLine(x, height - 6, x, height - 2)
        clipping = time.monotonic() - self._clip_at < _CLIP_S
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.FAIL if clipping else "#e0e2e6"))
        painter.drawRoundedRect(QRectF(width + 6, 2, lamp, height - 4), 4, 4)
        painter.setPen(QColor("#ffffff" if clipping else "#9a9da3"))
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.75))
        painter.setFont(font)
        painter.drawText(
            QRectF(width + 6, 2, lamp, height - 4), Qt.AlignmentFlag.AlignCenter, "CLIP"
        )
