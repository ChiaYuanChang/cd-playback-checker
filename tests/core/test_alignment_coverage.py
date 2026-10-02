"""Regression for long sessions whose valid early runs used to be deleted."""

import numpy as np
import pytest

from compare_audio.core.alignment_tracker import AlignmentPoint, PointStatus
from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.analysis_models import EventType
from compare_audio.core.event_detector import RecordingContext, detect_events
from compare_audio.core.reference_program import ReferenceProgram, ReferenceTrack
from compare_audio.core.spectral_features import FeatureSet, frame_levels_db


@pytest.mark.parametrize("drift", [0.000775, -0.000775])
def test_36_minute_drift_with_track_silences_preserves_every_track(drift):
    # Reduced feature rate keeps a full-length detector test small; use actual
    # per-frame similarity and coverage processing, without mocked similarity.
    config = AnalysisConfig(hop=800, n_mels=8, envelope_rate_hz=20)
    rate = config.frame_rate
    duration = 2200.0
    pre_roll = 3.65
    rng = np.random.default_rng(9)
    mel = rng.normal(-25, 5, (int(duration * rate), 8)).astype(np.float32)
    t = (np.arange(len(mel)) + 0.5) / rate
    silent = t % 220 > 218.5
    mel[silent] = -70
    env = np.where(silent, -70.0, -20.0).astype(np.float32)
    amp = 10 ** (env / 20)
    features = FeatureSet(mel, env, -amp, amp)
    program = ReferenceProgram(
        config,
        [ReferenceTrack(i, f"T{i}", "", i * 220.0, 220.0) for i in range(10)],
        features,
    )
    rec_duration = pre_roll + duration / (1 + drift) + 3
    rec_t = (np.arange(int(rec_duration * rate)) + 0.5) / rate
    prog_t = (rec_t - pre_roll) * (1 + drift)
    idx = np.floor(prog_t * rate).astype(int)
    valid = (idx >= 0) & (idx < len(mel))
    rec_mel = np.full((len(rec_t), 8), -70.0, np.float32)
    rec_env = np.full(len(rec_t), -70.0, np.float32)
    rec_mel[valid] = mel[idx[valid]]
    rec_env[valid] = env[idx[valid]]
    rec_amp = 10 ** (rec_env / 20)
    rec_features = FeatureSet(rec_mel, rec_env, -rec_amp, rec_amp)
    points = []
    for rec_s in np.arange(pre_roll + 1, rec_duration - 4, 0.25):
        p = (rec_s - pre_roll) * (1 + drift)
        quiet = p % 220 > 217.75 or p % 220 < 0.75
        points.append(
            AlignmentPoint(
                rec_s,
                np.nan if quiet else p,
                0.955,
                PointStatus.SILENT if quiet else PointStatus.MATCHED,
            )
        )
    context = RecordingContext(
        rec_features, frame_levels_db(rec_mel), -70, -70, rec_duration
    )
    detection = detect_events(points, context, program, config)
    assert len(detection.segments) == 10
    assert all(track.heard_s > 215 for track in detection.tracks)
    assert not any(e.event_type == EventType.TRACK_MISSING for e in detection.events)
    assert detection.segments[0].rec_start_s < 5
    assert detection.segments[-1].rec_end_s > 2195
    assert abs(detection.segments[0].drift_ppm) == pytest.approx(775, abs=1)


def test_correcting_redundant_mapping_keeps_observed_audio(program):
    # Feed a wrongly located early run followed by a correct run. Similarity is
    # computed from real synthetic music, so the neighbour's mapping proves the
    # early recording coverage belongs to the program rather than being deleted.
    config = program.config
    features = program.features
    duration = program.duration_s
    context = RecordingContext(
        features, frame_levels_db(features.log_mel), -100, -100, duration
    )
    points = [
        AlignmentPoint(t, t + 10 if t < 20 else t, 0.9, PointStatus.MATCHED)
        for t in np.arange(1, duration - 1, 0.25)
    ]
    detection = detect_events(points, context, program, config)
    assert detection.segments[0].rec_start_s < 1
    assert detection.tracks[0].heard_s > 29
    assert detection.tracks[1].heard_s > 29
