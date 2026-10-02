"""Exercise real source -> recording worker -> live/result UI with queued signals."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr
from PySide6.QtCore import QEventLoop, QTimer

from compare_audio.audio.audio_sources import FileSource
from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.analysis_models import Verdict
from compare_audio.core.recording_analysis import analyze_file
from compare_audio.core.reference_program import TrackSource, load_reference_program
from compare_audio.ui.live_view import LiveView
from compare_audio.ui.result_view import ResultView
from compare_audio.ui.workers import SessionWorker
from tests.core.conftest import RATE, _tone_melody, room


@pytest.mark.parametrize("input_rate", [RATE, 44100])
def test_record_stop_final_analysis_and_ui_use_selected_thresholds(
    qt_app, tmp_path, input_rate
):
    tracks = [_tone_melody(i, 12) for i in [1, 2]]
    sources = []
    for i, signal in enumerate(tracks):
        path = tmp_path / f"T{i}.wav"
        sf.write(path, signal, RATE, subtype="FLOAT")
        sources.append(TrackSource(title=f"T{i}", path=str(path)))
    base = load_reference_program(sources, AnalysisConfig())
    selected = replace(base, config=base.config.model_copy(update={"sensitivity": 80}))
    input_path = tmp_path / "input.wav"
    input_signal = room(np.concatenate(tracks), 2)
    if input_rate != RATE:
        input_signal = soxr.resample(input_signal, RATE, input_rate).astype(np.float32)
    sf.write(input_path, input_signal, input_rate, subtype="FLOAT")
    output_path = tmp_path / "recording.wav"
    worker = SessionWorker(
        selected, FileSource(input_path, speed=200), output_path, False
    )
    live = LiveView()
    live.start(selected, "檔案模擬")
    results = ResultView()
    failures, completed, detections = [], [], []
    worker.updated.connect(live.update_live)
    worker.provisional.connect(lambda detection: detections.append(detection))
    worker.provisional.connect(
        lambda detection: live.set_detection(detection) if detection else None
    )
    worker.failed.connect(failures.append)
    worker.completed.connect(lambda result, path: completed.append((result, path)))
    worker.completed.connect(
        lambda result, path: results.show_result(result, Path(path))
    )
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(worker.request_stop)
    timeout.timeout.connect(loop.quit)
    timeout.start(15000)
    worker.start()
    loop.exec()
    if worker.isRunning():
        worker.request_stop()
        worker.wait(5000)
    qt_app.processEvents()
    timeout.stop()
    assert not worker.isRunning()
    assert failures == []
    assert len(completed) == 1
    result, path = completed[0]
    assert result.config.sensitivity == 80
    assert result.summary.verdict == Verdict.PASS
    assert result.summary.heard_ratio > 0.95
    assert Path(path) == output_path and output_path.exists()
    assert sf.info(output_path).samplerate == input_rate
    assert detections[-1].tracks[0].heard_s > 11
    assert live.tracks.item(0, 2).text() == "已播"
    assert np.isfinite(live.plots._ref_env._amp).any()
    # Saved PCM16 WAV agrees with streaming analysis despite quantisation.
    offline = analyze_file(output_path, selected)
    assert offline.summary.verdict == result.summary.verdict
    assert abs(offline.summary.heard_ratio - result.summary.heard_ratio) < 0.01
    assert base.config.sensitivity == 50
    live.close()
    results.close()
