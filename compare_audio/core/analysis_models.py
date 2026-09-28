"""Contracts produced by the analysis and consumed by the UI and the reports."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.reference_program import ReferenceTrack


class EventType(StrEnum):
    SKIP = "skip"
    REPEAT = "repeat"
    STUCK = "stuck"
    PAUSE = "pause"
    DROPOUT = "dropout"
    UNCERTAIN = "uncertain"
    TRACK_GAP = "track_gap"
    TRACK_MISSING = "track_missing"
    TRACK_INCOMPLETE = "track_incomplete"
    STOPPED = "stopped"
    RECORDING_ENDED = "recording_ended"


EVENT_LABELS: dict[EventType, str] = {
    EventType.SKIP: "跳過",
    EventType.REPEAT: "重播",
    EventType.STUCK: "卡住",
    EventType.PAUSE: "停頓",
    EventType.DROPOUT: "瞬間無聲",
    EventType.UNCERTAIN: "待確認",
    EventType.TRACK_GAP: "曲間空白",
    EventType.TRACK_MISSING: "整首沒播",
    EventType.TRACK_INCOMPLETE: "沒播完整",
    EventType.STOPPED: "中途停止",
    EventType.RECORDING_ENDED: "錄音提早結束",
}


class Severity(StrEnum):
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


class Verdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class TrackVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_HEARD = "not_heard"


class DetectedEvent(BaseModel):
    event_type: EventType = Field(..., description="What went wrong (or happened).")
    severity: Severity = Field(..., description="fail = defect, warn = check by ear.")
    rec_start_s: float = Field(..., description="Recording time where it starts.")
    rec_end_s: float = Field(..., description="Recording time where it ends.")
    track_index: int | None = Field(default=None, description="Track it belongs to.")
    track_position_s: float | None = Field(
        default=None, description="Position inside that track where it happened."
    )
    jump_s: float | None = Field(
        default=None,
        description="Program position change: + content skipped, - content replayed.",
    )
    detail: str = Field(..., description="Human readable description (zh-TW).")

    @property
    def label(self) -> str:
        return EVENT_LABELS[self.event_type]


class AlignmentSegment(BaseModel):
    """A stretch of the recording that follows the program continuously."""

    rec_start_s: float = Field(..., description="Recording time where it starts.")
    rec_end_s: float = Field(..., description="Recording time where it ends.")
    offset_s: float = Field(
        ..., description="Program time minus recording time at the segment start."
    )
    drift_ppm: float = Field(..., description="Clock drift measured over the segment.")
    median_score: float = Field(..., description="Median matching score (0-1).")

    def offset_at(self, rec_s: float) -> float:
        return self.offset_s + self.drift_ppm * 1e-6 * (rec_s - self.rec_start_s)

    def program_at(self, rec_s: float) -> float:
        return rec_s + self.offset_at(rec_s)


class TrackReport(BaseModel):
    index: int = Field(..., description="0-based track index.")
    title: str = Field(..., description="Track title.")
    duration_s: float = Field(..., description="Reference duration.")
    verdict: TrackVerdict = Field(..., description="Overall result for the track.")
    heard_s: float = Field(
        ..., description="Seconds of the track found in the recording."
    )
    rec_start_s: float | None = Field(
        default=None, description="Where it was heard first."
    )
    rec_end_s: float | None = Field(
        default=None, description="Where it was heard last."
    )
    fail_count: int = Field(default=0, description="Number of defects.")
    warn_count: int = Field(default=0, description="Number of spots to check by ear.")


class AnalysisSummary(BaseModel):
    verdict: Verdict = Field(..., description="PASS / WARN / FAIL for the whole run.")
    recording_duration_s: float = Field(..., description="Length of the recording.")
    playback_start_s: float | None = Field(
        default=None, description="Recording time where the first track was found."
    )
    program_duration_s: float = Field(..., description="Total reference duration.")
    heard_ratio: float = Field(..., description="Fraction of the program found.")
    fail_count: int = Field(..., description="Number of defects.")
    warn_count: int = Field(..., description="Number of spots to check by ear.")
    noise_floor_db: float = Field(..., description="Recording noise floor (dB).")
    signal_to_noise_db: float | None = Field(
        default=None, description="Typical music level above the noise floor (dB)."
    )
    drift_ppm: float | None = Field(
        default=None, description="Clock drift between player and sound card."
    )


@dataclass
class AlignmentTrace:
    """Per-window tracker output, for the alignment plot."""

    rec_s: np.ndarray
    program_s: np.ndarray  # nan where not matched
    score: np.ndarray
    matched: np.ndarray  # bool
    silent: np.ndarray  # bool


@dataclass
class EnvelopeTrace:
    """Fine envelopes on the recording's time axis (reference mapped onto it)."""

    rate_hz: float
    rec_min: np.ndarray
    rec_max: np.ndarray
    ref_min: np.ndarray  # nan where the recording is not aligned
    ref_max: np.ndarray
    rec_level_db: np.ndarray
    expected_level_db: np.ndarray  # nan where not aligned


@dataclass
class AnalysisResult:
    summary: AnalysisSummary
    events: list[DetectedEvent]
    tracks: list[TrackReport]
    segments: list[AlignmentSegment]
    alignment: AlignmentTrace
    envelopes: EnvelopeTrace
    program_tracks: list[ReferenceTrack]
    config: AnalysisConfig
