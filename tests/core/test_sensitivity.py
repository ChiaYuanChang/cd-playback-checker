from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.analysis_models import EventType, Verdict
from compare_audio.core.live_analysis import LiveAnalyzer
from compare_audio.core.recording_analysis import analyze_signal

from .conftest import RATE, room


def test_neutral_and_bounded_sensitivity():
    normal = AnalysisConfig()
    assert normal.effective_match_threshold == normal.match_threshold
    assert normal.effective_acquire_threshold == normal.acquire_threshold
    assert normal.effective_silence_margin_db == normal.silence_margin_db
    for value in [-1, 101]:
        with pytest.raises(ValidationError):
            AnalysisConfig(sensitivity=value)
    sensitive = AnalysisConfig(sensitivity=100)
    conservative = AnalysisConfig(sensitivity=0)
    for field in normal.recognition_thresholds():
        name = "effective_" + field
        assert (
            getattr(sensitive, name)
            < getattr(normal, name)
            < getattr(conservative, name)
        )
    assert normal.feature_signature() == sensitive.feature_signature()


@pytest.mark.parametrize("sensitivity", [0, 50, 100])
def test_quiet_clean_audio_and_dropout_still_distinguished(
    program, tracks_audio, sensitivity
):
    program = replace(
        program, config=program.config.model_copy(update={"sensitivity": sensitivity})
    )
    clean = room(np.concatenate(tracks_audio), 3)
    # Absolute volume drops 40 dB, but signal-to-noise is unchanged.
    quiet = analyze_signal(clean * 0.01, program)
    assert quiet.summary.verdict == Verdict.PASS, quiet.events
    broken = clean.copy()
    broken[13 * RATE : int(13.2 * RATE)] = 0
    result = analyze_signal(broken * 0.01, program)
    assert any(e.event_type == EventType.DROPOUT for e in result.events)


def test_live_coverage_events_and_waveform_share_detection(program, tracks_audio):
    live = LiveAnalyzer(program)
    live.push(room(tracks_audio[0], 2))
    detection = live.provisional_detection()
    assert detection is not None
    assert detection.tracks[0].heard_s > 27
    assert detection.tracks[1].heard_s == 0
    assert not any(e.event_type == EventType.TRACK_MISSING for e in detection.events)
    assert np.isfinite(detection.envelopes.ref_max).any()
    assert np.isnan(detection.envelopes.ref_max[0])
    assert detection.match_score is not None
    assert detection.signal_to_noise_db is not None


def test_sensitivity_recovers_quiet_audio_without_claiming_dropout_verification(
    program, tracks_audio
):
    rng = np.random.default_rng(2)
    signal = np.concatenate([np.zeros(3 * RATE), *tracks_audio])
    recording = (signal * 0.035 + rng.normal(0, 0.002, len(signal))).astype(np.float32)
    normal = analyze_signal(recording, program)
    sensitive_program = replace(
        program, config=program.config.model_copy(update={"sensitivity": 100})
    )
    sensitive = analyze_signal(recording, sensitive_program)
    assert normal.summary.heard_ratio == 0
    assert sensitive.summary.heard_ratio > 0.98
    assert sensitive.summary.verdict == Verdict.WARN
    assert sensitive.summary.signal_to_noise_db is not None
    assert any("收音不足" in e.detail for e in sensitive.events)


def test_waiting_for_music_is_not_a_missing_track_failure(program):
    live = LiveAnalyzer(program)
    live.push(np.zeros(5 * RATE, np.float32))
    detection = live.provisional_detection()
    assert detection is not None
    assert detection.events == []
    assert all(
        track.heard_s == 0 and track.fail_count == 0 for track in detection.tracks
    )
