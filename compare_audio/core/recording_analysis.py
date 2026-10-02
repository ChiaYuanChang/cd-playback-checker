"""Offline analysis of a whole recording (also used after a live session ends)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from compare_audio.core.alignment_tracker import (
    AlignmentTracker,
    PointStatus,
    estimate_noise_floor_db,
)
from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.analysis_models import (
    AlignmentSegment,
    AlignmentTrace,
    AnalysisResult,
    AnalysisSummary,
    EnvelopeTrace,
    Severity,
    Verdict,
)
from compare_audio.core.audio_io import load_mono
from compare_audio.core.event_detector import RecordingContext, detect_events
from compare_audio.core.reference_program import ReferenceProgram
from compare_audio.core.spectral_features import (
    FeatureSet,
    extract_features,
    frame_levels_db,
)

ProgressCallback = Callable[[float], None]

_TRACK_CHUNK_FRAMES = 3000


def analyze_file(
    path: str | Path,
    program: ReferenceProgram,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    signal = load_mono(path, program.config.sample_rate)
    return analyze_signal(signal, program, progress)


def analyze_signal(
    signal: np.ndarray,
    program: ReferenceProgram,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    config = program.config
    features = extract_features(signal, config)
    if progress:
        progress(0.15)
    return analyze_features(
        features, len(signal) / config.sample_rate, program, progress
    )


def analyze_features(
    features: FeatureSet,
    duration_s: float,
    program: ReferenceProgram,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    config = program.config
    levels = frame_levels_db(features.log_mel)
    context = RecordingContext(
        features=features,
        levels_db=levels,
        frame_floor_db=estimate_noise_floor_db(levels),
        env_floor_db=estimate_noise_floor_db(features.band_env_db),
        duration_s=duration_s,
    )
    tracker = AlignmentTracker(program, config, noise_floor_db=context.frame_floor_db)
    n_frames = len(features.log_mel)
    for start in range(0, n_frames, _TRACK_CHUNK_FRAMES):
        tracker.push(features.log_mel[start : start + _TRACK_CHUNK_FRAMES])
        if progress:
            progress(
                0.15 + 0.7 * min(1.0, (start + _TRACK_CHUNK_FRAMES) / max(n_frames, 1))
            )
    detection = detect_events(tracker.points, context, program, config)
    if progress:
        progress(0.95)

    envelopes = build_envelope_trace(
        features,
        program,
        detection.segments,
        detection.gain_db,
        context.env_floor_db,
        config,
    )
    alignment = _alignment_trace(tracker.points)
    heard = sum(track.heard_s for track in detection.tracks)
    fails = sum(1 for e in detection.events if e.severity == Severity.FAIL)
    warns = sum(1 for e in detection.events if e.severity == Severity.WARN)
    verdict = Verdict.FAIL if fails else Verdict.WARN if warns else Verdict.PASS
    summary = AnalysisSummary(
        verdict=verdict,
        recording_duration_s=round(duration_s, 3),
        playback_start_s=detection.segments[0].rec_start_s
        if detection.segments
        else None,
        program_duration_s=round(program.duration_s, 3),
        heard_ratio=round(heard / program.duration_s, 4) if program.duration_s else 0.0,
        fail_count=fails,
        warn_count=warns,
        noise_floor_db=round(context.env_floor_db, 1),
        signal_to_noise_db=_signal_to_noise(envelopes, context.env_floor_db),
        drift_ppm=detection.drift_ppm,
    )
    if progress:
        progress(1.0)
    return AnalysisResult(
        summary=summary,
        events=detection.events,
        tracks=detection.tracks,
        segments=detection.segments,
        alignment=alignment,
        envelopes=envelopes,
        program_tracks=program.tracks,
        config=config,
    )


def _alignment_trace(points) -> AlignmentTrace:
    return AlignmentTrace(
        rec_s=np.array([p.rec_s for p in points], np.float64),
        program_s=np.array([p.program_s for p in points], np.float64),
        score=np.array([p.score for p in points], np.float32),
        matched=np.array([p.status == PointStatus.MATCHED for p in points], bool),
        silent=np.array([p.status == PointStatus.SILENT for p in points], bool),
    )


def build_envelope_trace(
    features: FeatureSet,
    program: ReferenceProgram,
    segments: list[AlignmentSegment],
    gain_db: float | None,
    floor_db: float,
    config: AnalysisConfig,
) -> EnvelopeTrace:
    """Map the reference envelopes onto the recording's time axis."""
    rate = config.envelope_rate_hz
    n = len(features.band_env_db)
    ref_min = np.full(n, np.nan, np.float32)
    ref_max = np.full(n, np.nan, np.float32)
    expected = np.full(n, np.nan, np.float32)
    ref_env = program.features.band_env_db
    for segment in segments:
        j0 = max(0, int(np.ceil(segment.rec_start_s * rate - 0.5)))
        j1 = min(n, int(np.floor(segment.rec_end_s * rate - 0.5)) + 1)
        if j1 <= j0:
            continue
        rec_idx = np.arange(j0, j1)
        offsets = segment.offset_s + segment.drift_ppm * 1e-6 * (
            (rec_idx + 0.5) / rate - segment.rec_start_s
        )
        prog_idx = np.round(rec_idx + offsets * rate).astype(np.int64)
        valid = (prog_idx >= 0) & (prog_idx < len(ref_env))
        rec_idx, prog_idx = rec_idx[valid], prog_idx[valid]
        ref_min[rec_idx] = program.features.wave_min[prog_idx]
        ref_max[rec_idx] = program.features.wave_max[prog_idx]
        if gain_db is not None:
            level = ref_env[prog_idx] + gain_db
            expected[rec_idx] = 10 * np.log10(
                10 ** (level / 10) + 10 ** (floor_db / 10)
            )

    # Scale the reference waveform to the recording's loudness for overlaying.
    mapped = ~np.isnan(ref_max)
    if mapped.any():
        rec_span = (features.wave_max - features.wave_min)[mapped]
        ref_span = (ref_max - ref_min)[mapped]
        loud = ref_span > np.percentile(ref_span, 50)
        if loud.sum() > 10:
            scale = float(np.median(rec_span[loud] / np.maximum(ref_span[loud], 1e-6)))
            ref_min *= scale
            ref_max *= scale
    return EnvelopeTrace(
        rate_hz=float(rate),
        rec_min=features.wave_min,
        rec_max=features.wave_max,
        ref_min=ref_min,
        ref_max=ref_max,
        rec_level_db=features.band_env_db,
        expected_level_db=expected,
    )


def _signal_to_noise(envelopes: EnvelopeTrace, floor_db: float) -> float | None:
    mapped = np.isfinite(envelopes.ref_max)
    if mapped.sum() < 50:
        return None
    return round(float(np.median(envelopes.rec_level_db[mapped])) - floor_db, 1)
