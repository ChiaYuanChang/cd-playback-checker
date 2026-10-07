"""From tracker points to defects.

1. Group trusted points with the same offset into runs (continuous playback).
2. Between two runs, find exactly where the old mapping stops explaining the audio
   (t1) and where the new one starts (t2) with a per-frame similarity change point.
3. Classify the transition from ``u = t2 - t1`` (recording time nobody explains),
   ``a = p2 - p1`` (how far the program moved meanwhile) and whether the
   unexplained audio is silent:

   ====================  ===============  =====================================
   u                     a                meaning
   ====================  ===============  =====================================
   ~0                    > 0              skip (content lost)
   ~0                    < 0              repeat (content played again)
   silent                ~ u              dropout (muted while playing on)
   silent                ~ 0              pause (stalled), or a gap between tracks
   has sound             << u             stuck (looping / not advancing)
   has sound             ~ u              unrecognised audio (noise, distortion)
   ====================  ===============  =====================================

4. Inside runs, look for short mutes in the fine energy envelope and for long
   stretches where the audio stops resembling the reference.
5. Check that every track was heard from its first to its last audible second.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from compare_audio.core.alignment_tracker import AlignmentPoint, PointStatus
from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.analysis_models import (
    AlignmentSegment,
    DetectedEvent,
    EventType,
    Severity,
    TrackReport,
    TrackVerdict,
)
from compare_audio.core.reference_program import (
    ReferenceProgram,
    ReferenceTrack,
    frame_center_s,
)
from compare_audio.core.spectral_features import FeatureSet, frame_levels_db
from compare_audio.core.time_format import format_clock

_SHAPE_SMOOTH_S = 2.0
_SIM_SMOOTH_FRAMES = 5
_MIN_RUN_POINTS = 3
_MIN_RUN_S = 0.75
_REDUNDANT_RATIO = 0.9
_BLIP_MAX_S = 4.0
_SPLIT_SILENT_STEPS = 2
_QUIET_VS_MUSIC_DB = 12.0
_SILENT_U_FRACTION = 0.7
_AUDIBLE_FRACTION = 0.2
_AUDIBLE_MARGIN_DB = 8.0
_LOST_MARGIN_DB = 12.0
_LOST_FRACTION = 0.5
_BOUNDARY_REACH_S = 2.5
_STOP_TAIL_S = 3.0
_REPEAT_MAX_U_S = 0.5
_EDGE_SEARCH_WINDOWS = 4
_LOCAL_FLOOR_MAX_DB = 12.0
_LONG_DROPOUT_S = 0.25
_LOOP_MATCH = 0.38
_DROPOUT_MERGE_S = 0.6
_EDGE_LOOK_S = 3.0
_UNCONFIRMED_EDGE_S = 3.0
_BOUNDARY_BACK_S = 1.0
_GAP_SOUND_S = 0.5
_MIN_REPORTED_GAP_S = 0.1
_SOFT_DB = 10.0
_MAX_DRIFT = 0.01  # sanity limit; a webcam + CD player pair measured 0.36% (3600 ppm)
_ENDING_EVENTS = (EventType.STOPPED, EventType.RECORDING_ENDED, EventType.UNCERTAIN)


@dataclass
class RecordingContext:
    """What the detector needs to know about the recording."""

    features: FeatureSet
    levels_db: np.ndarray  # broadband level per log-mel frame
    frame_floor_db: float
    env_floor_db: float
    duration_s: float


@dataclass
class DetectionOutput:
    events: list[DetectedEvent]
    segments: list[AlignmentSegment]
    tracks: list[TrackReport]
    gain_db: float | None
    drift_ppm: float | None


@dataclass
class _Run:
    rec: list[float] = field(default_factory=list)
    offset: list[float] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    intercept: float = 0.0
    slope: float = 0.0
    t_ref: float = 0.0

    def add(self, point: AlignmentPoint) -> None:
        self.rec.append(point.rec_s)
        self.offset.append(point.program_s - point.rec_s)
        self.score.append(point.score)

    def extend(self, other: "_Run") -> None:
        self.rec.extend(other.rec)
        self.offset.extend(other.offset)
        self.score.extend(other.score)

    @property
    def span_s(self) -> float:
        return self.rec[-1] - self.rec[0]

    @property
    def is_crumb(self) -> bool:
        """Too short for its own offset to mean anything."""
        return len(self.rec) < _MIN_RUN_POINTS or self.span_s < _MIN_RUN_S

    def recent_offset(self) -> float:
        return float(np.median(self.offset[-5:]))

    def offset_at(self, rec_s: float) -> float:
        return float(self.intercept + self.slope * (rec_s - self.t_ref))

    def offsets_at(self, rec_s: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * (rec_s - self.t_ref)

    def program_at(self, rec_s: float) -> float:
        return rec_s + self.offset_at(rec_s)

    def inliers(self, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
        rec, off = np.array(self.rec), np.array(self.offset)
        keep = np.abs(off - np.median(off)) <= 2 * tolerance
        return rec[keep], off[keep]

    def fit(self, slope: float) -> None:
        """Offset line with the clock drift shared by all runs of the recording."""
        rec, off = np.array(self.rec), np.array(self.offset)
        self.t_ref = float(rec[0])
        self.slope = slope
        self.intercept = float(np.median(off - slope * (rec - self.t_ref)))


def detect_events(
    points: list[AlignmentPoint],
    context: RecordingContext,
    program: ReferenceProgram,
    config: AnalysisConfig,
    live: bool = False,
) -> DetectionOutput:
    """``live``: the recording is still running, so nothing is judged about its end
    or about tracks that simply have not been reached yet."""
    return _Detector(points, context, program, config, live).run()


class _Detector:
    def __init__(
        self,
        points: list[AlignmentPoint],
        context: RecordingContext,
        program: ReferenceProgram,
        config: AnalysisConfig,
        live: bool = False,
    ) -> None:
        self.live = live
        self.points = points
        self.ctx = context
        self.program = program
        self.config = config
        self.tolerance = config.offset_tolerance_s
        self.rec_shape = _shape_features(context.features.log_mel, config)
        self.prog_shape = _shape_features(program.features.log_mel, config)
        self.prog_levels = frame_levels_db(program.features.log_mel)
        self.rec_silent = (
            context.levels_db <= context.frame_floor_db + config.silence_margin_db
        )
        self.events: list[DetectedEvent] = []
        self.lost_ranges: list[tuple[float, float]] = []  # program content skipped
        self.gain_db: float | None = None
        self.slope = 0.0

    # ------------------------------------------------------------------ pipeline
    def run(self) -> DetectionOutput:
        runs = self._build_runs()
        if not runs:
            return self._nothing_found()
        self._refine_edges(runs)
        self.gain_db = self._estimate_gain(runs)
        for previous, following in zip(runs, runs[1:], strict=False):
            self._classify_transition(previous, following)
        for run in runs:
            self._scan_dropouts(run)
        self._scan_uncertain(runs)
        if not self.live:
            self._check_ending(runs[-1])
        tracks = self._track_reports(runs)
        self.events.sort(key=lambda event: event.rec_start_s)
        self.events = self._merge_dropouts(self.events)
        segments = [self._segment(run) for run in runs]
        return DetectionOutput(
            self.events, segments, tracks, self.gain_db, self._overall_drift(runs)
        )

    def _nothing_found(self) -> DetectionOutput:
        self.events.append(
            DetectedEvent(
                event_type=EventType.TRACK_MISSING,
                severity=Severity.FAIL,
                rec_start_s=0.0,
                rec_end_s=self.ctx.duration_s,
                detail="錄音中找不到任何參考音軌的內容",
            )
        )
        tracks = [
            TrackReport(
                index=track.index,
                title=track.title,
                duration_s=track.duration_s,
                verdict=TrackVerdict.FAIL,
                heard_s=0.0,
                fail_count=1,
            )
            for track in self.program.tracks
        ]
        return DetectionOutput(self.events, [], tracks, None, None)

    # ------------------------------------------------------------------ runs
    def _build_runs(self) -> list[_Run]:
        """Group trusted points; a silence of half a second or more starts a new run
        even at the same offset, so gaps and mutes are always examined."""
        runs: list[_Run] = []
        current: _Run | None = None
        join_tolerance = 1.5 * self.tolerance
        silent_steps = 0
        for point in self.points:
            if point.status == PointStatus.SILENT:
                silent_steps += 1
                continue
            if point.status != PointStatus.MATCHED:
                continue
            offset = point.program_s - point.rec_s
            if (
                current is not None
                and silent_steps < _SPLIT_SILENT_STEPS
                and abs(offset - current.recent_offset()) <= join_tolerance
            ):
                current.add(point)
            else:
                if current is not None:
                    runs.append(current)
                current = _Run()
                current.add(point)
            silent_steps = 0
        if current is not None:
            runs.append(current)
        self._fit_all(runs)
        return self._drop_redundant(runs)

    def _fit_all(self, runs: list[_Run]) -> None:
        """One clock drift for the whole recording (pooled over all runs)."""
        numerator = denominator = 0.0
        for run in runs:
            rec, off = run.inliers(self.tolerance)
            if len(rec) >= 5:
                centered = rec - rec.mean()
                numerator += float(np.sum(centered * (off - off.mean())))
                denominator += float(np.sum(centered * centered))
        slope = numerator / denominator if denominator > 100.0 else 0.0
        self.slope = float(np.clip(slope, -_MAX_DRIFT, _MAX_DRIFT))
        for run in runs:
            run.fit(self.slope)

    def _drop_redundant(self, runs: list[_Run]) -> list[_Run]:
        """Remove runs that a neighbour's mapping explains just as well.

        If continuing the previous (or next) mapping fits the audio of a run about
        as well as the run's own mapping, there was no audible jump: the tracker
        mis-locked on repeated music or noise. Weak crumbs are dropped as well.
        Neighbours left with the same offset are merged.
        """
        half = self.config.window_frames // 2
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for k, run in enumerate(runs):
                if run.is_crumb and float(np.median(run.score)) < (
                    self.config.acquire_threshold + 0.1
                ):
                    del runs[k]
                    changed = True
                    break
                # A few seconds elsewhere, then back to exactly the old position.
                if (
                    0 < k < len(runs) - 1
                    and run.span_s < _BLIP_MAX_S
                    and abs(
                        runs[k - 1].offset_at(run.rec[0])
                        - runs[k + 1].offset_at(run.rec[0])
                    )
                    <= 1.5 * self.tolerance
                ):
                    del runs[k]
                    changed = True
                    break
                f0 = self._frame(run.rec[0]) - half
                f1 = self._frame(run.rec[-1]) + half
                own = float(np.mean(self._similarity(run, f0, f1)))
                # Only neighbours at a different position can "explain away" a run;
                # a run split off by a silence has the same mapping by definition.
                neighbours = [
                    other
                    for other in (
                        runs[k - 1] if k else None,
                        runs[k + 1] if k + 1 < len(runs) else None,
                    )
                    if other is not None
                    and not other.is_crumb
                    and abs(other.offset_at(run.rec[0]) - run.offset_at(run.rec[0]))
                    > 1.5 * self.tolerance
                ]
                if own > 0 and any(
                    float(np.mean(self._similarity(other, f0, f1)))
                    >= _REDUNDANT_RATIO * own
                    for other in neighbours
                ):
                    del runs[k]
                    changed = True
                    break
            runs = self._merge_equal_neighbours(runs)
        return runs

    def _merge_equal_neighbours(self, runs: list[_Run]) -> list[_Run]:
        merged: list[_Run] = []
        for run in runs:
            if (
                merged
                and abs(merged[-1].offset_at(run.rec[0]) - run.offset_at(run.rec[0]))
                <= (1.5 * self.tolerance)
                and not self._silence_between(merged[-1], run)
            ):
                merged[-1].extend(run)
                merged[-1].fit(self.slope)
            else:
                merged.append(run)
        return merged

    def _silence_between(self, a: _Run, b: _Run) -> bool:
        between = [
            p
            for p in self.points
            if a.rec[-1] < p.rec_s < b.rec[0] and p.status == PointStatus.SILENT
        ]
        return len(between) >= _SPLIT_SILENT_STEPS

    # ------------------------------------------------------------------ similarity
    def _frame(self, rec_s: float) -> int:
        cfg = self.config
        return int(round((rec_s * cfg.sample_rate - cfg.n_fft / 2) / cfg.hop))

    def _time(self, frame: float) -> float:
        return frame_center_s(frame - 0.5, self.config)

    def _mapped_frames(
        self, run: _Run, t0: int, t1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = np.arange(t0, t1)
        times = frame_center_s(frames.astype(np.float64), self.config)
        program = frames + run.offsets_at(times) * self.config.frame_rate
        return frames, np.round(program).astype(np.int64)

    def _similarity(self, run: _Run, t0: int, t1: int) -> np.ndarray:
        """Smoothed per-frame spectral-shape similarity under the run's mapping."""
        t0 = max(0, t0)
        t1 = min(len(self.rec_shape), t1)
        if t1 <= t0:
            return np.zeros(0)
        frames, program = self._mapped_frames(run, t0, t1)
        valid = (program >= 0) & (program < len(self.prog_shape))
        sim = np.zeros(len(frames))
        sim[valid] = np.einsum(
            "ij,ij->i", self.rec_shape[frames[valid]], self.prog_shape[program[valid]]
        )
        sim[self.rec_silent[frames]] = 0.0
        return uniform_filter1d(sim, size=_SIM_SMOOTH_FRAMES, mode="nearest")

    def _null_level(self, run: _Run, t0: int, t1: int) -> float:
        sim = self._similarity(run, t0, t1)
        typical = float(np.median(sim)) if len(sim) else 0.3
        return float(np.clip(0.5 * typical, 0.05, 0.35))

    def _refine_edges(self, runs: list[_Run]) -> None:
        """Find where each run's mapping really starts and stops explaining audio."""
        half = self.config.window_frames // 2
        span = self.config.window_frames
        reach = half + _EDGE_SEARCH_WINDOWS * span
        first, last = runs[0], runs[-1]
        # Leading edge of the first run: extend back while the mapping still fits.
        anchor = self._frame(first.rec[0])
        level = self._null_level(first, anchor, anchor + 2 * span)
        start = max(0, anchor - reach)
        gain = self._similarity(first, start, anchor) - level
        suffix = np.concatenate([np.cumsum(gain[::-1])[::-1], [0.0]])
        first.start_s = self._time(start + int(np.argmax(suffix)))
        # Trailing edge of the last run.
        anchor = self._frame(last.rec[-1])
        level = self._null_level(last, anchor - 2 * span, anchor)
        end = min(len(self.rec_shape), anchor + reach)
        gain = self._similarity(last, anchor, end) - level
        prefix = np.concatenate([[0.0], np.cumsum(gain)])
        last.end_s = self._time(anchor + int(np.argmax(prefix)))
        # Transitions between runs.
        for a, b in zip(runs, runs[1:], strict=False):
            t_a, t_b = self._frame(a.rec[-1]), self._frame(b.rec[0])
            lo = max(self._frame(a.rec[0]), t_a - half)
            hi = max(lo + 1, min(self._frame(b.rec[-1]), t_b + half))
            level_a = self._null_level(a, t_a - 2 * span, t_a)
            level_b = self._null_level(b, t_b, t_b + 2 * span)
            gain_a = self._similarity(a, lo, hi) - level_a
            gain_b = self._similarity(b, lo, hi) - level_b
            prefix = np.concatenate([[0.0], np.cumsum(gain_a)])
            suffix = np.concatenate([np.cumsum(gain_b[::-1])[::-1], [0.0]])
            k1, k2 = int(np.argmax(prefix)), int(np.argmax(suffix))
            if k1 > k2:
                k1 = k2 = int(np.argmax(prefix + suffix))
            a.end_s = self._time(lo + k1)
            b.start_s = self._time(lo + k2)

    # ------------------------------------------------------------------ levels
    def _estimate_gain(self, runs: list[_Run]) -> float | None:
        """Median level difference recording - reference over well aligned audio."""
        diffs: list[np.ndarray] = []
        for run in runs:
            rec_idx, prog_idx = self._env_mapping(run, run.start_s, run.end_s)
            if len(rec_idx) == 0:
                continue
            rec = self.ctx.features.band_env_db[rec_idx]
            ref = self.program.features.band_env_db[prog_idx]
            loud = rec > self.ctx.env_floor_db + 15.0
            if loud.sum() > 20:
                diffs.append((rec - ref)[loud])
        if not diffs:
            return None
        return float(np.median(np.concatenate(diffs)))

    def _env_mapping(
        self, run: _Run, start_s: float, end_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rate = self.config.envelope_rate_hz
        n_rec = len(self.ctx.features.band_env_db)
        j0 = max(0, int(np.ceil(start_s * rate - 0.5)))
        j1 = min(n_rec, int(np.floor(end_s * rate - 0.5)))
        if j1 <= j0:
            empty = np.zeros(0, np.int64)
            return empty, empty
        rec_idx = np.arange(j0, j1)
        offsets = run.offsets_at((rec_idx + 0.5) / rate)
        prog_idx = np.round(rec_idx + offsets * rate).astype(np.int64)
        valid = (prog_idx >= 0) & (prog_idx < len(self.program.features.band_env_db))
        return rec_idx[valid], prog_idx[valid]

    def _local_floor(self, t1: float, t2: float) -> float:
        """Background level in the recording between t1 and t2.

        The global floor comes from the quietest moments; while people talk the
        background is louder, and a fade-out below it is not audible.
        """
        floor = self.ctx.env_floor_db
        rate = self.config.envelope_rate_hz
        env = self.ctx.features.band_env_db
        j0, j1 = max(0, int(t1 * rate)), min(len(env), int(np.ceil(t2 * rate)))
        if j1 - j0 < int(0.2 * rate):
            return floor
        return float(np.clip(np.median(env[j0:j1]), floor, floor + _LOCAL_FLOOR_MAX_DB))

    def _audible_fraction(
        self,
        p1: float,
        p2: float,
        margin_db: float = _AUDIBLE_MARGIN_DB,
        floor_db: float | None = None,
    ) -> float:
        """Share of the program between p1 and p2 that would be audible here."""
        if p2 <= p1:
            return 0.0
        rate = self.config.envelope_rate_hz
        j0 = max(0, int(p1 * rate))
        j1 = min(len(self.program.features.band_env_db), int(np.ceil(p2 * rate)))
        if j1 <= j0:
            return 0.0
        floor = self.ctx.env_floor_db if floor_db is None else floor_db
        expected = self.program.features.band_env_db[j0:j1] + (self.gain_db or 0.0)
        return float(np.mean(expected > floor + margin_db))

    def _silent_fraction(self, t1_s: float, t2_s: float) -> float:
        f1 = max(0, self._frame(t1_s))
        f2 = min(len(self.rec_silent), max(self._frame(t2_s), f1 + 1))
        span = self.rec_silent[f1:f2]
        return float(np.mean(span)) if len(span) else 1.0

    def _player_silent(self, t1: float, t2: float, p1: float, p2: float) -> bool:
        """Did the player output (almost) nothing between t1 and t2?

        Near the noise floor counts as silent; so does audio far quieter than the
        music around the event, which is how talking in the room shows up.
        """
        if self._silent_fraction(t1, t2) >= _SILENT_U_FRACTION:
            return True
        rate = self.config.envelope_rate_hz
        env = self.ctx.features.band_env_db
        j0, j1 = max(0, int(t1 * rate)), min(len(env), int(np.ceil(t2 * rate)))
        if j1 <= j0 or self.gain_db is None:
            return False
        heard = float(np.median(env[j0:j1]))
        program = self.program.features.band_env_db
        around = np.concatenate(
            [
                program[max(0, int((p1 - 1.0) * rate)) : max(0, int(p1 * rate))],
                program[max(0, int(p2 * rate)) : max(0, int((p2 + 1.0) * rate))],
            ]
        )
        if len(around) == 0:
            return False
        expected = float(np.percentile(around, 75)) + self.gain_db
        return heard <= expected - _QUIET_VS_MUSIC_DB

    def _loop_period_s(self, t1_s: float, t2_s: float) -> float | None:
        f1 = max(0, self._frame(t1_s))
        f2 = min(len(self.rec_shape), self._frame(t2_s))
        n = f2 - f1
        if n < 16:
            return None
        segment = self.rec_shape[f1:f2]
        max_lag = min(int(1.5 * self.config.frame_rate), n // 2)
        corr = np.array(
            [
                np.mean(np.einsum("ij,ij->i", segment[:-lag], segment[lag:]))
                for lag in range(1, max_lag + 1)
            ]
        )
        for lag in range(4, max_lag - 1):
            here = corr[lag - 1]
            peak = here >= corr[lag - 2] and here >= corr[lag]
            if peak and here >= 0.5 and here - corr[: lag - 1].min() >= 0.15:
                return lag / self.config.frame_rate
        return None

    # ------------------------------------------------------------------ describing
    def _where(self, program_s: float) -> str:
        index, position = self.program.locate(program_s)
        return f"第 {index + 1} 首 {format_clock(position)}"

    def _whole_tracks_between(self, p1: float, p2: float) -> list[ReferenceTrack]:
        return [t for t in self.program.tracks if p1 <= t.start_s and t.end_s <= p2]

    def _skip_detail(self, p1: float, p2: float, prefix: str) -> str:
        skipped = self._whole_tracks_between(p1, p2)
        whole = ""
        if skipped:
            names = "、".join(str(t.index + 1) for t in skipped)
            whole = f"，第 {names} 首整首沒播"
        return f"{prefix}{whole}（{self._where(p1)} → {self._where(p2)}）"

    def _repeat_detail(self, advance: float, p1: float, p2: float) -> str:
        return f"重播 {-advance:.2f} 秒（{self._where(p1)} 跳回 {self._where(p2)}）"

    def _add(
        self,
        event_type: EventType,
        severity: Severity,
        start_s: float,
        end_s: float,
        program_s: float | None,
        detail: str,
        jump_s: float | None = None,
    ) -> None:
        index, position = (None, None)
        if program_s is not None:
            index, position = self.program.locate(program_s)
        self.events.append(
            DetectedEvent(
                event_type=event_type,
                severity=severity,
                rec_start_s=round(start_s, 3),
                rec_end_s=round(max(end_s, start_s), 3),
                track_index=index,
                track_position_s=None if position is None else round(position, 3),
                jump_s=None if jump_s is None else round(jump_s, 3),
                detail=detail,
            )
        )

    def _add_skip(
        self, start_s: float, end_s: float, p1: float, p2: float, prefix: str
    ) -> None:
        self.lost_ranges.append((p1, p2))
        self._add(
            EventType.SKIP,
            Severity.FAIL,
            start_s,
            end_s,
            p1,
            self._skip_detail(p1, p2, prefix),
            p2 - p1,
        )

    # ------------------------------------------------------------------ transitions
    def _nearby_boundary(self, p1: float, p2: float) -> int | None:
        """Index of a track that starts close to (or between) p1 and p2."""
        lo, hi = min(p1, p2), max(p1, p2)
        for track in self.program.tracks[1:]:
            if lo - _BOUNDARY_REACH_S <= track.start_s <= hi + _BOUNDARY_REACH_S:
                return track.index
        return None

    def _track_gap(
        self,
        boundary: int,
        t1: float,
        t2: float,
        p2: float,
        extra: float,
        advance: float,
    ) -> None:
        gap = max(extra, 0.0)
        silence = self._silent_fraction(t1, t2) * (t2 - t1)
        unexplained = gap - silence
        if gap > self.config.track_gap_max_s:
            self._add(
                EventType.PAUSE,
                Severity.WARN,
                t1,
                t2,
                p2,
                f"第 {boundary} → {boundary + 1} 首之間停頓 {gap:.1f} 秒（過長）",
                advance,
            )
        elif unexplained > _GAP_SOUND_S:
            # The player spent longer here than the silence accounts for: something
            # audible happened around the track change that could not be matched.
            self._add(
                EventType.UNCERTAIN,
                Severity.WARN,
                t1,
                t2,
                p2,
                f"第 {boundary} → {boundary + 1} 首之間多出 {unexplained:.1f} 秒"
                "無法辨認的聲音（請聽聽看）",
                advance,
            )
        elif gap >= _MIN_REPORTED_GAP_S:
            self._add(
                EventType.TRACK_GAP,
                Severity.INFO,
                t1,
                t2,
                p2,
                f"第 {boundary} → {boundary + 1} 首之間空白 {gap:.1f} 秒",
                advance,
            )

    def _classify_transition(self, a: _Run, b: _Run) -> None:
        cfg = self.config
        t1, t2 = a.end_s, max(a.end_s, b.start_s)
        p1, p2 = a.program_at(t1), b.program_at(t2)
        u, advance = t2 - t1, p2 - p1
        extra = u - advance  # recording time during which the program stood still
        tol = self.tolerance

        floor = self._local_floor(t1, t2)
        tol_u = max(tol, 0.1 * u)
        lost_audible = (
            self._audible_fraction(p1, p2, _LOST_MARGIN_DB, floor) >= _LOST_FRACTION
        )
        # Between tracks players insert or drop silence, and a fade-out or a soft
        # intro may not be recognised in a noisy room. That is a normal gap as long
        # as the program did not move further than the recording did (nothing
        # audible was skipped) and did not jump back.
        boundary = self._nearby_boundary(p1, p2)
        if (
            boundary is not None
            and advance >= -_BOUNDARY_BACK_S
            and (advance <= u + tol_u or not lost_audible)
        ):
            self._track_gap(boundary, t1, t2, p2, extra, advance)
            return

        if u < cfg.min_gap_s:
            if advance > tol and lost_audible:
                self._add_skip(t1, t2, p1, p2, f"跳過 {advance:.2f} 秒")
            elif advance < -tol:
                self._add(
                    EventType.REPEAT,
                    Severity.FAIL,
                    t1,
                    t2,
                    p1,
                    self._repeat_detail(advance, p1, p2),
                    advance,
                )
            return

        if self._player_silent(t1, t2, p1, p2):
            self._classify_silent(t1, t2, p1, p2, u, advance, tol_u, lost_audible)
        else:
            self._classify_sounding(t1, t2, p1, p2, u, advance, tol_u, lost_audible)

    def _classify_silent(self, t1, t2, p1, p2, u, advance, tol_u, lost_audible) -> None:
        tol = self.tolerance
        if advance > u + tol_u:
            if lost_audible:
                self._add_skip(
                    t1, t2, p1, p2, f"跳過 {advance:.2f} 秒，期間無聲 {u:.2f} 秒"
                )
        elif abs(advance - u) <= tol_u:
            if self._audible_fraction(p1, p2) >= _AUDIBLE_FRACTION:
                self._add(
                    EventType.DROPOUT,
                    Severity.FAIL,
                    t1,
                    t2,
                    p1,
                    f"無聲 {u:.2f} 秒（{self._where(p1)}）",
                    advance,
                )
        elif advance >= -max(tol, 0.25 * u):
            self._add(
                EventType.PAUSE,
                Severity.FAIL,
                t1,
                t2,
                p1,
                f"停頓 {u - advance:.2f} 秒（無聲，{self._where(p1)}）",
                advance,
            )
        else:
            self._add(
                EventType.REPEAT,
                Severity.FAIL,
                t1,
                t2,
                p1,
                f"重播 {-advance:.2f} 秒，之前無聲 {u:.2f} 秒"
                f"（{self._where(p1)} 跳回 {self._where(p2)}）",
                advance,
            )

    def _loops_recent_music(self, t1: float, t2: float, p1: float) -> bool:
        """Does the unexplained audio consist of the music just before p1?

        A stuck player repeats a fragment it has just played; room noise during a
        pause does not look like that fragment.
        """
        cfg = self.config
        f1, f2 = max(0, self._frame(t1)), min(len(self.rec_shape), self._frame(t2))
        g2 = int(round((p1 * cfg.sample_rate - cfg.n_fft / 2) / cfg.hop))
        g1 = max(0, g2 - int(1.5 * cfg.frame_rate))
        g2 = min(len(self.prog_shape), g2 + 2)
        if f2 - f1 < 4 or g2 - g1 < 4:
            return False
        sounding = ~self.rec_silent[f1:f2]
        if sounding.sum() < 4:
            return False
        sims = self.rec_shape[f1:f2][sounding] @ self.prog_shape[g1:g2].T
        return float(np.median(sims.max(axis=1))) >= _LOOP_MATCH

    def _classify_sounding(
        self, t1, t2, p1, p2, u, advance, tol_u, lost_audible
    ) -> None:
        extra = u - advance
        looping = u > _REPEAT_MAX_U_S and (
            self._loop_period_s(t1, t2) is not None
            or self._loops_recent_music(t1, t2, p1)
        )
        if advance > u + tol_u and lost_audible:
            self._add_skip(
                t1, t2, p1, p2, f"跳過 {advance:.2f} 秒，中間 {u:.2f} 秒聲音不對"
            )
        elif advance < -self.tolerance and not looping:
            self._add(
                EventType.REPEAT,
                Severity.FAIL,
                t1,
                t2,
                p1,
                self._repeat_detail(advance, p1, p2),
                advance,
            )
        elif looping or (advance < -self.tolerance):
            period = self._loop_period_s(t1, t2)
            loop = f"，重複約 {period:.2f} 秒的片段" if period else ""
            self._add(
                EventType.STUCK,
                Severity.FAIL,
                t1,
                t2,
                p1,
                f"卡住 {u:.2f} 秒{loop}（{self._where(p1)}）",
                advance,
            )
        elif extra > max(0.25, 0.5 * u):
            # The program stood still and what we hear is not the music: a pause
            # with room noise (people talking) rather than a loop.
            self._add(
                EventType.PAUSE,
                Severity.FAIL,
                t1,
                t2,
                p1,
                f"停頓 {extra:.2f} 秒（期間只有環境音，{self._where(p1)}）",
                advance,
            )
        elif u >= self.config.uncertain_min_s:
            self._add(
                EventType.UNCERTAIN,
                Severity.WARN,
                t1,
                t2,
                p1,
                f"{u:.1f} 秒對不上參考（可能是噪音或失真，{self._where(p1)}）",
                advance,
            )

    # ------------------------------------------------------------------ inside runs
    def _scan_dropouts(self, run: _Run) -> None:
        """Short mutes that leave the matching intact but empty the energy envelope."""
        cfg = self.config
        guard = 0.05
        rec_idx, prog_idx = self._env_mapping(
            run, run.start_s + guard, run.end_s - guard
        )
        if len(rec_idx) == 0 or self.gain_db is None:
            return
        floor = self.ctx.env_floor_db
        rec = self.ctx.features.band_env_db[rec_idx]
        ref = self.program.features.band_env_db[prog_idx] + self.gain_db
        expected = 10 * np.log10(10 ** (ref / 10) + 10 ** (floor / 10))
        deficit = expected - rec
        # Short mutes need a deep, clear drop; long ones may sit in a soft passage
        # as long as the recording really falls to the noise floor.
        clear = (deficit >= cfg.dropout_depth_db) & (
            expected >= floor + cfg.dropout_depth_db + 3.0
        )
        floored = (rec <= floor + 3.0) & (deficit >= 5.0) & (expected >= floor + 6.0)
        min_blocks = max(1, int(round(cfg.dropout_min_s * cfg.envelope_rate_hz)))
        long_blocks = int(round(_LONG_DROPOUT_S * cfg.envelope_rate_hz))
        spans = [
            (first, last)
            for first, last in _true_spans(clear, max_hole=1)
            if last - first + 1 >= min_blocks
        ] + [
            (first, last)
            for first, last in _true_spans(floored, max_hole=2)
            if last - first + 1 >= long_blocks
        ]
        for first, last in sorted(spans):
            start_s = rec_idx[first] / cfg.envelope_rate_hz
            end_s = (rec_idx[last] + 1) / cfg.envelope_rate_hz
            if self._overlaps_event(start_s, end_s):
                continue
            program_s = run.program_at(start_s)
            depth = float(np.median(deficit[first : last + 1]))
            self._add(
                EventType.DROPOUT,
                Severity.FAIL,
                start_s,
                end_s,
                program_s,
                f"無聲 {1000 * (end_s - start_s):.0f} ms（音量掉 {depth:.0f} dB，"
                f"{self._where(program_s)}）",
            )

    def _scan_uncertain(self, runs: list[_Run]) -> None:
        """Long stretches where the audio stops resembling the reference.

        The position is kept (the same mapping continues afterwards), so this is
        not a jump; it is noise drowning the music or distortion. Only reported
        when the similarity collapses relative to the rest of the run.
        """
        cfg = self.config
        min_frames = int(2 * cfg.uncertain_min_s * cfg.frame_rate)
        smooth = max(3, int(0.5 * cfg.frame_rate))
        loud_program = float(np.percentile(self.prog_levels, 50)) - 25.0
        for run in runs:
            f0, f1 = max(0, self._frame(run.start_s)), self._frame(run.end_s)
            if f1 - f0 < min_frames:
                continue
            sim = uniform_filter1d(
                self._similarity(run, f0, f1), smooth, mode="nearest"
            )
            sounding = ~self.rec_silent[f0 : f0 + len(sim)]
            if sounding.sum() < min_frames:
                continue
            typical = float(np.median(sim[sounding]))
            if typical < 0.1:
                continue
            _, mapped = self._mapped_frames(run, f0, f0 + len(sim))
            mapped = np.clip(mapped, 0, len(self.prog_levels) - 1)
            active = self.prog_levels[mapped] >= loud_program
            low = (sim < max(0.05, 0.25 * typical)) & sounding & active
            for first, last in _true_spans(low, max_hole=smooth):
                if last - first + 1 < min_frames:
                    continue
                start_s, end_s = self._time(f0 + first), self._time(f0 + last + 1)
                if self._overlaps_event(start_s, end_s):
                    continue
                program_s = run.program_at(start_s)
                self._add(
                    EventType.UNCERTAIN,
                    Severity.WARN,
                    start_s,
                    end_s,
                    program_s,
                    f"{end_s - start_s:.1f} 秒聲音和參考不像"
                    f"（可能是噪音或失真，{self._where(program_s)}）",
                )

    def _merge_dropouts(self, events: list[DetectedEvent]) -> list[DetectedEvent]:
        """One mute interrupted by a knock or reverb shows up as several pieces."""
        merged: list[DetectedEvent] = []
        for event in events:
            previous = merged[-1] if merged else None
            if (
                previous is not None
                and event.event_type == previous.event_type == EventType.DROPOUT
                and event.rec_start_s - previous.rec_end_s <= _DROPOUT_MERGE_S
            ):
                start, end = (
                    previous.rec_start_s,
                    max(previous.rec_end_s, event.rec_end_s),
                )
                where = self._where(
                    self.program.tracks[previous.track_index or 0].start_s
                    + (previous.track_position_s or 0.0)
                )
                merged[-1] = previous.model_copy(
                    update={
                        "rec_end_s": end,
                        "detail": f"無聲 {1000 * (end - start):.0f} ms（{where}）",
                    }
                )
            else:
                merged.append(event)
        return merged

    def _overlaps_event(self, start_s: float, end_s: float) -> bool:
        return any(
            event.rec_start_s - 0.1 <= end_s and start_s <= event.rec_end_s + 0.1
            for event in self.events
            if event.severity != Severity.INFO
        )

    # ------------------------------------------------------------------ coverage
    def _is_soft(self, p1: float, p2: float) -> bool:
        """Is the program between p1 and p2 much softer than its track overall?

        Fade-outs and soft intros are hard to recognise under room noise; not
        confirming them is expected and not worth a warning.
        """
        if p2 <= p1:
            return True
        rate = self.config.envelope_rate_hz
        env = self.program.features.band_env_db
        track = self.program.tracks[self.program.track_index_at(p1)]
        body = env[int(track.start_s * rate) : int(track.end_s * rate)]
        part = env[max(0, int(p1 * rate)) : int(np.ceil(p2 * rate))]
        if len(body) == 0 or len(part) == 0:
            return True
        return float(np.percentile(part, 90)) <= (
            float(np.percentile(body, 50)) - _SOFT_DB
        )

    def _track_audible_range(
        self, index: int, floor_db: float | None = None
    ) -> tuple[float, float] | None:
        track = self.program.tracks[index]
        rate = self.config.envelope_rate_hz
        j0, j1 = int(track.start_s * rate), int(track.end_s * rate)
        floor = self.ctx.env_floor_db if floor_db is None else floor_db
        expected = self.program.features.band_env_db[j0:j1] + (self.gain_db or 0.0)
        audible = np.nonzero(expected > floor + _AUDIBLE_MARGIN_DB)[0]
        if len(audible) == 0:
            return None
        return (j0 + audible[0]) / rate, (j0 + audible[-1] + 1) / rate

    def _check_ending(self, last: _Run) -> None:
        cfg = self.config
        p_end = last.program_at(last.end_s)
        floor = self._local_floor(last.end_s, self.ctx.duration_s)
        final = self._track_audible_range(len(self.program.tracks) - 1, floor)
        audible_end = final[1] if final is not None else self.program.duration_s
        if p_end >= audible_end - cfg.edge_tolerance_s:
            return
        tail = self.ctx.duration_s - last.end_s
        where = self._where(p_end)
        if tail < _STOP_TAIL_S:
            self._add(
                EventType.RECORDING_ENDED,
                Severity.WARN,
                last.end_s,
                self.ctx.duration_s,
                p_end,
                f"錄音在節目結束前停止（停在 {where}）",
            )
            return
        track_range = self._track_audible_range(
            self.program.track_index_at(p_end), floor
        )
        if track_range is not None and p_end >= track_range[1] - cfg.edge_tolerance_s:
            return  # the track was finished; the missing tracks are reported per track
        tail_silent = (
            self._silent_fraction(last.end_s, self.ctx.duration_s) >= _SILENT_U_FRACTION
        )
        if not tail_silent and self._is_soft(p_end, audible_end):
            return
        if tail_silent:
            self._add(
                EventType.STOPPED,
                Severity.FAIL,
                last.end_s,
                self.ctx.duration_s,
                p_end,
                f"播放在 {where} 停止",
            )
        else:
            self._add(
                EventType.UNCERTAIN,
                Severity.WARN,
                last.end_s,
                self.ctx.duration_s,
                p_end,
                f"{where} 之後的聲音對不上參考",
            )

    def _lost_covers(self, p_lo: float, p_hi: float) -> bool:
        return any(lo - 1.0 <= p_lo and p_hi <= hi + 1.0 for lo, hi in self.lost_ranges)

    def _track_reports(self, runs: list[_Run]) -> list[TrackReport]:
        recording_ended = self.live or any(
            e.event_type == EventType.RECORDING_ENDED for e in self.events
        )
        last_program = runs[-1].program_at(runs[-1].end_s)
        reports: list[TrackReport] = []
        for track in self.program.tracks:
            spans: list[tuple[float, float, float, float]] = []  # program, rec
            for run in runs:
                p_lo, p_hi = run.program_at(run.start_s), run.program_at(run.end_s)
                lo, hi = max(p_lo, track.start_s), min(p_hi, track.end_s)
                if hi > lo:
                    spans.append(
                        (
                            lo,
                            hi,
                            lo - run.offset_at(run.start_s),
                            hi - run.offset_at(run.end_s),
                        )
                    )
            heard = sum(hi - lo for lo, hi, _, _ in spans)
            if not spans:
                if recording_ended and track.start_s >= last_program:
                    reports.append(self._report(track, TrackVerdict.NOT_HEARD, 0.0))
                    continue
                if not self._lost_covers(track.start_s, track.end_s):
                    near = self._rec_time_near(runs, track.start_s)
                    self._add(
                        EventType.TRACK_MISSING,
                        Severity.FAIL,
                        near,
                        near,
                        track.start_s,
                        f"第 {track.index + 1} 首完全沒有聽到",
                    )
            else:
                self._check_track_edges(track, spans, recording_ended, last_program)
            reports.append(self._track_verdict(track, heard, spans))
        return reports

    def _check_track_edges(
        self,
        track: ReferenceTrack,
        spans: list[tuple[float, float, float, float]],
        recording_ended: bool,
        last_program: float,
    ) -> None:
        """Was the track heard from its first to its last audible second?

        Silence in the recording where the missing part should be means it was
        not played (FAIL); sound there means it probably played but could not be
        confirmed (WARN, and only when longer than a quiet intro/outro would be).
        """
        tol = self.config.edge_tolerance_s
        first_heard = min(lo for lo, _, _, _ in spans)
        last_heard = max(hi for _, hi, _, _ in spans)
        rec_first = min(r for _, _, r, _ in spans)
        rec_last = max(r for _, _, _, r in spans)

        head_floor = self._local_floor(rec_first - _EDGE_LOOK_S, rec_first)
        audible = self._track_audible_range(track.index, head_floor)
        if audible is not None:
            missing = first_heard - audible[0]
            region = (max(0.0, rec_first - missing), rec_first)
            silent = self._silent_fraction(*region) >= 0.7
            soft = not silent and self._is_soft(audible[0], first_heard)
            if (missing > (tol if silent else _UNCONFIRMED_EDGE_S)) and not (
                soft or self._lost_covers(audible[0], first_heard)
            ):
                self._add(
                    EventType.TRACK_INCOMPLETE,
                    Severity.FAIL if silent else Severity.WARN,
                    region[0],
                    region[1],
                    first_heard,
                    f"第 {track.index + 1} 首開頭 {missing:.1f} 秒"
                    + ("沒有播" if silent else "無法確認"),
                )

        ended = any(
            e.event_type in _ENDING_EVENTS and e.track_index == track.index
            for e in self.events
        )
        if ended or (recording_ended and track.end_s >= last_program):
            return
        tail_floor = self._local_floor(rec_last, rec_last + _EDGE_LOOK_S)
        audible = self._track_audible_range(track.index, tail_floor)
        if audible is not None:
            missing = audible[1] - last_heard
            region = (rec_last, rec_last + missing)
            silent = self._silent_fraction(*region) >= 0.7
            soft = not silent and self._is_soft(last_heard, audible[1])
            if (missing > (tol if silent else _UNCONFIRMED_EDGE_S)) and not (
                soft or self._lost_covers(last_heard, audible[1])
            ):
                self._add(
                    EventType.TRACK_INCOMPLETE,
                    Severity.FAIL if silent else Severity.WARN,
                    region[0],
                    region[1],
                    last_heard,
                    f"第 {track.index + 1} 首結尾 {missing:.1f} 秒"
                    + ("沒有播" if silent else "無法確認"),
                )

    def _rec_time_near(self, runs: list[_Run], program_s: float) -> float:
        """Recording time where a program position would have been expected."""
        best = min(runs, key=lambda r: abs(r.program_at(r.start_s) - program_s))
        return max(0.0, program_s - best.offset_at(best.start_s))

    def _track_verdict(
        self,
        track: ReferenceTrack,
        heard: float,
        spans: list[tuple[float, float, float, float]],
    ) -> TrackReport:
        mine = [e for e in self.events if e.track_index == track.index]
        fails = sum(1 for e in mine if e.severity == Severity.FAIL)
        warns = sum(1 for e in mine if e.severity == Severity.WARN)
        if not spans or fails:
            verdict = TrackVerdict.FAIL
        elif warns:
            verdict = TrackVerdict.WARN
        else:
            verdict = TrackVerdict.PASS
        report = self._report(
            track,
            verdict,
            heard,
            min((r for _, _, r, _ in spans), default=None),
            max((r for _, _, _, r in spans), default=None),
        )
        return report.model_copy(update={"fail_count": fails, "warn_count": warns})

    @staticmethod
    def _report(
        track: ReferenceTrack,
        verdict: TrackVerdict,
        heard: float,
        rec_first: float | None = None,
        rec_last: float | None = None,
    ) -> TrackReport:
        return TrackReport(
            index=track.index,
            title=track.title,
            duration_s=round(track.duration_s, 3),
            verdict=verdict,
            heard_s=round(heard, 3),
            rec_start_s=None if rec_first is None else round(rec_first, 3),
            rec_end_s=None if rec_last is None else round(rec_last, 3),
        )

    def _segment(self, run: _Run) -> AlignmentSegment:
        return AlignmentSegment(
            rec_start_s=round(run.start_s, 4),
            rec_end_s=round(run.end_s, 4),
            offset_s=round(run.offset_at(run.start_s), 5),
            drift_ppm=round(run.slope * 1e6, 2),  # offset slope, see offset_at()
            median_score=round(float(np.median(run.score)), 3),
        )

    def _overall_drift(self, runs: list[_Run]) -> float | None:
        """Recording clock vs player clock: + means the recording runs long."""
        if sum(run.span_s for run in runs) < 30.0:
            return None
        return round(-self.slope * 1e6, 1)


def _shape_features(log_mel: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """Spectral shape relative to the local average, unit length per frame."""
    size = max(3, int(_SHAPE_SMOOTH_S * config.frame_rate) | 1)
    trend = uniform_filter1d(
        log_mel.astype(np.float32), size=size, axis=0, mode="nearest"
    )
    shape = log_mel - trend
    shape -= shape.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(shape, axis=1, keepdims=True)
    return (shape / np.maximum(norm, 1e-6)).astype(np.float32)


def _true_spans(mask: np.ndarray, max_hole: int = 0) -> list[tuple[int, int]]:
    """Inclusive index spans where ``mask`` is true, bridging holes up to max_hole."""
    spans: list[tuple[int, int]] = []
    indices = np.nonzero(mask)[0]
    if len(indices) == 0:
        return spans
    start = prev = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - prev > max_hole + 1:
            spans.append((start, prev))
            start = index
        prev = index
    spans.append((start, prev))
    return spans
