"""Streaming alignment: where in the program is each window of the recording?

Every ``step`` the tracker takes the latest ``window`` of recording features and
looks for it in the program:

- locked: search only around the predicted position (fast, and not fooled by a
  chorus that appears several times)
- lost or not yet locked: search the whole program, then confirm on the next step

The same class serves live recording (features pushed as they arrive) and offline
analysis (the whole recording pushed at once).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.growing_array import GrowingArray
from compare_audio.core.reference_program import ReferenceProgram, frame_center_s
from compare_audio.core.spectral_features import frame_levels_db
from compare_audio.core.window_matcher import LocalSearch, ProgramMatcher, WindowMatch

_SILENT_FRACTION = 0.8
_ONLINE_FLOOR_PERCENTILE = 5.0
_FLOOR_REFRESH_STEPS = 20
_MAX_GLOBAL_BACKOFF = 4
_MOVE_MAX_SILENT = 0.05
_CONTINUITY_MARGIN = 0.03
_JUMP_MARGIN = 0.05
_MAX_EXTRA_RADIUS_STEPS = 40


class PointStatus(StrEnum):
    MATCHED = "matched"
    WEAK = "weak"  # sound present but no reliable match
    SILENT = "silent"  # at the recording's noise floor


@dataclass(frozen=True, slots=True)
class AlignmentPoint:
    rec_s: float  # recording time of the window centre
    program_s: float  # matched program position of the window centre (nan if none)
    score: float
    status: PointStatus


def estimate_noise_floor_db(levels_db: np.ndarray) -> float:
    """Level of the quietest stretches (pre-roll, gaps between tracks)."""
    if len(levels_db) == 0:
        return -100.0
    kernel = min(len(levels_db), 15)
    smoothed = np.convolve(levels_db, np.ones(kernel) / kernel, mode="valid")
    return float(np.percentile(smoothed, 2.0))


class AlignmentTracker:
    def __init__(
        self,
        program: ReferenceProgram,
        config: AnalysisConfig,
        noise_floor_db: float | None = None,
    ) -> None:
        self._program = program
        self._config = config
        self._window = config.window_frames
        self._step = config.step_frames
        self._radius = config.seconds_to_frames(config.search_radius_s)
        self._tolerance = config.offset_tolerance_s * config.frame_rate
        self._matcher = ProgramMatcher(
            program.features.log_mel, self._window, config.coarse_factor
        )
        self._frames = GrowingArray(config.n_mels)
        self._levels = GrowingArray(None)
        self._fixed_floor = noise_floor_db
        self._floor = noise_floor_db if noise_floor_db is not None else -100.0
        self._next_start = 0
        self._step_count = 0
        self._offset: float | None = None  # program frame - recording frame
        self._pending_offset: float | None = None
        self._last_offset: float | None = None
        self._misses = 0
        self._quiet_steps = 0
        self._global_backoff = 0
        self._global_skip = 0
        self.points: list[AlignmentPoint] = []

    # ------------------------------------------------------------------ state
    @property
    def is_locked(self) -> bool:
        return self._offset is not None

    @property
    def noise_floor_db(self) -> float:
        return self._floor

    def current_program_s(self) -> float | None:
        """Program position at the newest recording frame, if locked."""
        if self._offset is None:
            return None
        return frame_center_s(self._frames.size - 1 + self._offset, self._config)

    # ------------------------------------------------------------------ input
    def push(self, log_mel_frames: np.ndarray) -> list[AlignmentPoint]:
        if len(log_mel_frames):
            self._frames.extend(log_mel_frames)
            self._levels.extend(frame_levels_db(log_mel_frames))
        new_points: list[AlignmentPoint] = []
        while self._next_start + self._window <= self._frames.size:
            new_points.append(self._process_window(self._next_start))
            self._next_start += self._step
        self.points.extend(new_points)
        return new_points

    # ------------------------------------------------------------------ steps
    def _update_floor(self) -> None:
        if self._fixed_floor is not None:
            return
        if self._step_count % _FLOOR_REFRESH_STEPS == 0 or self._step_count < 8:
            levels = self._levels.view()
            if len(levels):
                self._floor = float(np.percentile(levels, _ONLINE_FLOOR_PERCENTILE))

    def _silent_fraction(self, start: int) -> float:
        levels = self._levels.view()[start : start + self._window]
        return float(
            np.mean(levels <= self._floor + self._config.effective_silence_margin_db)
        )

    def _loudness_db(self, start: int) -> float:
        """How far the louder part of the window stands above the noise floor."""
        levels = self._levels.view()[start : start + self._window]
        return float(np.percentile(levels, 75)) - self._floor

    def _process_window(self, start: int) -> AlignmentPoint:
        self._update_floor()
        self._step_count += 1
        query = self._frames.view()[start : start + self._window]
        silent_fraction = self._silent_fraction(start)
        match, trusted = self._match(query, start, silent_fraction)
        rec_s = frame_center_s(start + self._window / 2, self._config)
        score = match.score if match is not None else 0.0
        if (
            trusted
            and match is not None
            and score >= self._config.effective_match_threshold
        ):
            program_s = frame_center_s(match.lag + self._window / 2, self._config)
            return AlignmentPoint(rec_s, program_s, score, PointStatus.MATCHED)
        silent = silent_fraction >= _SILENT_FRACTION
        status = PointStatus.SILENT if silent else PointStatus.WEAK
        return AlignmentPoint(rec_s, float("nan"), score, status)

    def _local(
        self, query: np.ndarray, predicted: float, radius: int, prefer: bool = True
    ) -> LocalSearch | None:
        center = int(round(predicted))
        return self._matcher.local_search(
            query,
            center - radius,
            center + radius,
            prefer_lag=predicted,
            prefer_radius=int(np.ceil(self._tolerance)),
            prefer_margin=_CONTINUITY_MARGIN if prefer else 0.0,
        )

    def _match(
        self, query: np.ndarray, start: int, silent_fraction: float
    ) -> tuple[WindowMatch | None, bool]:
        """Best match for this window and whether it is consistent with a lock.

        Only trusted matches become MATCHED points: an unconfirmed jump or a fresh
        global hit is reported as WEAK until the next window agrees with it.
        """
        config = self._config
        silent = silent_fraction >= _SILENT_FRACTION
        # A window that is partly silence is dominated by the silence->music step,
        # which matches any such step in the program; only windows fully inside
        # music may acquire or move the lock.
        clean = silent_fraction <= _MOVE_MAX_SILENT
        fallback: WindowMatch | None = None
        if silent:
            self._quiet_steps += 1
        if self._offset is not None:
            # After silence (a gap between tracks, a pause) the position has most
            # likely moved: widen the search and drop the continuity preference,
            # which would otherwise keep a stale lock on slowly changing music.
            after_silence = self._quiet_steps > 0 and not silent
            radius = self._radius + min(
                self._quiet_steps * self._step, _MAX_EXTRA_RADIUS_STEPS * self._step
            )
            result = self._local(
                query, start + self._offset, radius, prefer=not after_silence
            )
            if (
                result is not None
                and result.best.score >= config.effective_match_threshold
            ):
                if not silent:
                    self._quiet_steps = 0
                return self._follow(result, start, clean, after_silence)
            if result is not None:
                fallback = result.at_preferred or result.best
            if not silent:
                self._misses += 1
            if self._misses < config.lost_steps:
                return fallback, False
            # Lost: drop the lock and re-acquire (with confirmation) as at the start.
            self._offset = None
            self._pending_offset = None
        elif self._pending_offset is not None:
            if not clean:
                return fallback, False
            result = self._local(query, start + self._pending_offset, self._radius // 2)
            if (
                result is not None
                and result.best.score >= config.effective_acquire_threshold
            ):
                offset = result.best.lag - start
                if abs(offset - self._pending_offset) <= self._tolerance:
                    self._lock(offset)
                    return result.best, True
            self._pending_offset = None

        if not clean:
            return fallback, False
        if self._global_skip > 0:
            self._global_skip -= 1
            return fallback, False
        found = self._pick_candidate(
            self._matcher.global_search(query, config.global_candidates), start
        )
        if found is None or found.score < config.effective_acquire_threshold:
            self._global_backoff = min(_MAX_GLOBAL_BACKOFF, self._global_backoff + 1)
            self._global_skip = self._global_backoff
            return fallback, False
        self._global_backoff = 0
        self._pending_offset = found.lag - start
        return found, False

    def _follow(
        self, result: LocalSearch, start: int, clean: bool, after_silence: bool
    ) -> tuple[WindowMatch | None, bool]:
        """Handle a good local match while locked."""
        assert self._offset is not None
        self._misses = 0
        offset = result.best.lag - start
        if abs(offset - self._offset) <= self._tolerance:
            self._offset = 0.5 * self._offset + 0.5 * offset
            self._pending_offset = None
            return result.best, True
        # A different position. It moves the lock only when two consecutive
        # windows agree, each one clearly better than the predicted position and
        # loud enough: quiet windows (fades, soft passages) give biased lags, and a
        # real jump there is still caught by the global search once the lock is lost.
        near = result.at_preferred
        near_score = near.score if near is not None else -1.0
        margin = 0.0 if after_silence else _JUMP_MARGIN
        strong = (
            clean
            and result.best.score >= self._config.effective_acquire_threshold
            and result.best.score >= near_score + margin
            and self._loudness_db(start) >= self._config.effective_jump_min_loudness_db
        )
        if (
            strong
            and self._pending_offset is not None
            and abs(offset - self._pending_offset) <= self._tolerance
        ):
            self._lock(offset)
            return result.best, True
        self._pending_offset = offset if strong else None
        trusted = (
            near is not None and near.score >= self._config.effective_match_threshold
        )
        return near, trusted

    def _pick_candidate(
        self, candidates: list[WindowMatch], start: int
    ) -> WindowMatch | None:
        """Best global candidate; near-ties go to the one closest to the last lock."""
        if not candidates:
            return None
        reference = self._offset if self._offset is not None else self._last_offset
        if reference is None:
            return candidates[0]
        top = candidates[0].score
        close = [c for c in candidates if c.score >= top - _CONTINUITY_MARGIN]
        predicted = start + reference
        return min(close, key=lambda c: abs(c.lag - predicted))

    def _lock(self, offset: float) -> None:
        self._offset = offset
        self._last_offset = offset
        self._pending_offset = None
        self._misses = 0
        self._quiet_steps = 0
        self._global_backoff = 0
        self._global_skip = 0
