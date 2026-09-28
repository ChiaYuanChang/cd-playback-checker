"""Analysis while recording: features and tracking as audio arrives.

The live view shows where the player is and provisional problems; when the session
ends, ``finish()`` runs the full offline analysis on the same features, so the final
report is identical to analysing the saved recording afterwards.
"""

from dataclasses import dataclass

import numpy as np

from compare_audio.core.alignment_tracker import (
    AlignmentPoint,
    AlignmentTracker,
    estimate_noise_floor_db,
)
from compare_audio.core.analysis_models import AnalysisResult, DetectedEvent
from compare_audio.core.event_detector import RecordingContext, detect_events
from compare_audio.core.growing_array import GrowingArray
from compare_audio.core.recording_analysis import ProgressCallback, analyze_features
from compare_audio.core.reference_program import ReferenceProgram
from compare_audio.core.spectral_features import (
    FeatureExtractor,
    FeatureSet,
    frame_levels_db,
)


@dataclass(frozen=True)
class LiveStatus:
    rec_s: float
    locked: bool
    program_s: float | None
    track_index: int | None
    track_position_s: float | None
    last_track_done: bool


class LiveAnalyzer:
    def __init__(self, program: ReferenceProgram) -> None:
        self.program = program
        self.config = program.config
        self._extractor = FeatureExtractor(self.config)
        self._tracker = AlignmentTracker(program, self.config)
        self._log_mel = GrowingArray(self.config.n_mels)
        self._band_env = GrowingArray(None)
        self._wave_min = GrowingArray(None)
        self._wave_max = GrowingArray(None)
        self.n_samples = 0

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.config.sample_rate

    @property
    def points(self) -> list[AlignmentPoint]:
        return self._tracker.points

    def push(self, samples: np.ndarray) -> list[AlignmentPoint]:
        """Feed mono audio at the analysis rate; returns the new tracker points."""
        self.n_samples += len(samples)
        features = self._extractor.push(samples)
        self._log_mel.extend(features.log_mel)
        self._band_env.extend(features.band_env_db)
        self._wave_min.extend(features.wave_min)
        self._wave_max.extend(features.wave_max)
        return self._tracker.push(features.log_mel)

    def features(self) -> FeatureSet:
        """Everything extracted so far (views, no copying)."""
        return FeatureSet(
            log_mel=self._log_mel.view(),
            band_env_db=self._band_env.view(),
            wave_min=self._wave_min.view(),
            wave_max=self._wave_max.view(),
        )

    def status(self) -> LiveStatus:
        program_s = self._tracker.current_program_s()
        index = position = None
        done = False
        if program_s is not None:
            index, position = self.program.locate(program_s)
            last = self.program.tracks[-1]
            done = index == last.index and program_s >= last.end_s - 1.0
        return LiveStatus(
            rec_s=self.duration_s,
            locked=self._tracker.is_locked,
            program_s=program_s,
            track_index=index,
            track_position_s=position,
            last_track_done=done,
        )

    def provisional_events(self) -> list[DetectedEvent]:
        """Events found so far (the newest seconds may still change)."""
        features = self.features()
        if len(features.log_mel) == 0:
            return []
        levels = frame_levels_db(features.log_mel)
        context = RecordingContext(
            features=features,
            levels_db=levels,
            frame_floor_db=estimate_noise_floor_db(levels),
            env_floor_db=estimate_noise_floor_db(features.band_env_db),
            duration_s=self.duration_s,
        )
        detection = detect_events(
            self.points, context, self.program, self.config, live=True
        )
        return detection.events

    def finish(self, progress: ProgressCallback | None = None) -> AnalysisResult:
        features = self.features()
        owned = FeatureSet(
            log_mel=features.log_mel.copy(),
            band_env_db=features.band_env_db.copy(),
            wave_min=features.wave_min.copy(),
            wave_max=features.wave_max.copy(),
        )
        return analyze_features(owned, self.duration_s, self.program, progress)
