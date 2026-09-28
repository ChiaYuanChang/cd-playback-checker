"""Colours, fonts and the light Fusion palette used everywhere."""

import sys

import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from compare_audio.core.analysis_models import Severity, TrackVerdict, Verdict

FAIL = "#c62828"
WARN = "#b7791f"
INFO = "#3f8f4f"
PASS = "#2e7d32"
ACCENT = "#1f6fd1"
RECORDING = "#6f6f6f"
REFERENCE = "#1f6fd1"
MUTED = "#8a8a8a"
SURFACE = "#ffffff"
PANEL = "#f4f5f7"
BORDER = "#d8dbe0"
TEXT = "#1d1f23"

SEVERITY_COLOR = {Severity.FAIL: FAIL, Severity.WARN: WARN, Severity.INFO: INFO}
VERDICT_COLOR = {Verdict.PASS: PASS, Verdict.WARN: WARN, Verdict.FAIL: FAIL}
TRACK_COLOR = {
    TrackVerdict.PASS: PASS,
    TrackVerdict.WARN: WARN,
    TrackVerdict.FAIL: FAIL,
    TrackVerdict.NOT_HEARD: MUTED,
}

_FONT_FAMILIES = {
    "win32": ["Microsoft JhengHei UI", "Microsoft JhengHei", "Segoe UI"],
    "darwin": ["PingFang TC", "Heiti TC", "Helvetica Neue"],
}


def ui_font(point_size: int = 10) -> QFont:
    font = QFont()
    font.setFamilies(_FONT_FAMILIES.get(sys.platform, ["Noto Sans CJK TC", "Sans"]))
    font.setPointSize(point_size + (3 if sys.platform == "darwin" else 0))
    return font


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(PANEL))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f7f8fa"))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    app.setFont(ui_font())
    app.setStyleSheet(
        f"""
        QGroupBox {{ font-weight: bold; border: 1px solid {BORDER};
                     border-radius: 6px; margin-top: 14px;
                     padding: 10px 8px 8px 8px; background: {SURFACE}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QPushButton {{ padding: 5px 12px; border: 1px solid {BORDER};
                       border-radius: 5px; background: #ffffff; }}
        QPushButton:hover {{ background: #eef3fb; }}
        QPushButton:disabled {{ color: #a0a0a0; }}
        QPushButton#startButton {{ font-size: 16px; font-weight: bold; padding: 12px;
                                   color: white; background: {ACCENT}; border: none; }}
        QPushButton#startButton:hover {{ background: #1a5fb4; }}
        QPushButton#startButton[recording="true"] {{ background: {FAIL}; }}
        QPushButton#startButton:disabled {{ background: #9fb7d9; }}
        QTableWidget {{ gridline-color: #e6e8eb; }}
        QHeaderView::section {{ background: #f2f3f5; border: none;
                                border-bottom: 1px solid {BORDER}; padding: 4px; }}
        """
    )
    pg.setConfigOptions(antialias=True, background=SURFACE, foreground="#444444")
