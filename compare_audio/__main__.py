"""Entry point: ``python -m compare_audio`` (also used by the packaged exe)."""

import sys

from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("CompareAudio")
    app.setApplicationName("CompareAudio")
    app.setApplicationDisplayName("CD 播放測試")

    from compare_audio.ui.main_window import MainWindow
    from compare_audio.ui.theme import apply_theme

    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
