from types import SimpleNamespace

import numpy as np

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.analysis_models import (
    DetectedEvent,
    EnvelopeTrace,
    EventType,
    Severity,
    TrackReport,
    TrackVerdict,
)
from compare_audio.core.live_analysis import LiveDetection, LiveStatus
from compare_audio.core.reference_program import ReferenceProgram, ReferenceTrack
from compare_audio.core.spectral_features import FeatureSet
from compare_audio.ui.live_view import LiveView
from compare_audio.ui.main_window import MainWindow
from compare_audio.ui.plots import EnvelopeCurve, TimelinePlots
from compare_audio.ui.sensitivity_control import SensitivityControl
from compare_audio.ui.workers import LiveUpdate


def _program():
    return ReferenceProgram(
        AnalysisConfig(), [ReferenceTrack(0, "測試", "", 0, 30)], FeatureSet.empty(48)
    )


def test_manual_navigation_survives_live_updates_and_follow_resumes(qt_app):
    view = LiveView()
    view.start(_program(), "模擬")
    view.plots.align.setXRange(10, 20, padding=0)
    view.plots.align.setYRange(10, 15, padding=0)
    view.plots.align.getViewBox().sigRangeChangedManually.emit([True, True])
    assert not view.follow.isChecked()
    update = LiveUpdate(
        LiveStatus(100, True, 25, 0, 25, False),
        np.array([10, 11]),
        np.array([9, 10]),
        np.array([10, 11]),
        np.array([-20, -20]),
        0,
    )
    view.update_live(update)
    np.testing.assert_allclose(view.plots.align.viewRange(), [[10, 20], [10, 15]])
    view.follow.setChecked(True)
    view.update_live(update)
    np.testing.assert_allclose(view.plots.align.viewRange()[0], [0, 105])
    view.close()


def test_detection_drives_both_table_and_waveforms(qt_app):
    view = LiveView()
    view.start(_program(), "模擬")
    env = EnvelopeTrace(
        20,
        np.array([-0.1, -0.2]),
        np.array([0.1, 0.2]),
        np.array([np.nan, -0.15]),
        np.array([np.nan, 0.15]),
        np.array([-40, -20]),
        np.array([np.nan, -20]),
    )
    event = DetectedEvent(
        event_type=EventType.TRACK_INCOMPLETE,
        severity=Severity.WARN,
        rec_start_s=0,
        rec_end_s=2,
        track_index=0,
        detail="開頭需確認",
    )
    report = TrackReport(
        index=0,
        title="測試",
        duration_s=30,
        heard_s=27,
        verdict=TrackVerdict.WARN,
        warn_count=1,
    )
    view.set_detection(LiveDetection([event], [report], env, -50, 30, 0.9))
    assert view.tracks.item(0, 2).text() == "需確認"
    assert view.events.item(0, 2).text() == "開頭需確認"
    assert view._bars[0].value() == 900
    x, y = view.plots._ref_env.getData()
    assert np.isnan(y[1])
    assert y[3] > 0  # overlay uses the same direction
    view.wave_control.mode.setCurrentIndex(1)
    _, split_y = view.plots._ref_env.getData()
    assert split_y[3] < 0
    assert "訊噪比 30.0 dB" in view.diagnostics.text()
    view.close()


def test_unaligned_waveform_bins_stay_blank(qt_app):
    curve = EnvelopeCurve(1, "#000000")
    curve.set_envelope(np.full(10000, np.nan), 200)
    _, y = curve.getData()
    assert np.isnan(y[1::2]).all()
    plots = TimelinePlots()
    assert plots.align.getViewBox().state["mouseEnabled"] == [True, True]
    plots.close()


def test_session_override_uses_shared_cached_features_without_changing_profile(qt_app):
    program = _program()
    control = SensitivityControl()
    control.slider.setValue(85)
    window = SimpleNamespace(
        program=program, setup=SimpleNamespace(sensitivity=control)
    )
    selected = MainWindow._session_program(window)
    assert selected.config.sensitivity == 85
    assert selected.features is program.features
    assert program.config.sensitivity == 50
    control.close()


def test_real_wheel_navigation_disables_follow_in_all_plots(qt_app):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    view = LiveView()
    view.resize(1000, 800)
    view.start(_program(), "模擬")
    view.show()
    qt_app.processEvents()
    for plot in (view.plots.align, view.plots.wave, view.plots.level):
        view.follow.setChecked(True)
        qt_app.processEvents()
        initial = view.plots.align.viewRange()[0]
        center = plot.getViewBox().sceneBoundingRect().center()
        pos = view.plots.mapFromScene(center)
        event = QWheelEvent(
            QPointF(pos),
            QPointF(view.plots.viewport().mapToGlobal(pos)),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(view.plots.viewport(), event)
        qt_app.processEvents()
        assert not view.follow.isChecked()
        changed = view.plots.align.viewRange()[0]
        assert changed[1] - changed[0] < initial[1] - initial[0]
    view.close()


def test_sensitivity_locked_during_recording(qt_app):
    from compare_audio.ui.setup_panel import SetupPanel

    panel = SetupPanel()
    panel.set_recording(True)
    assert not panel.sensitivity.slider.isEnabled()
    panel.set_recording(False)
    assert panel.sensitivity.slider.isEnabled()
    panel.close()
