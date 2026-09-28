"""Build the audio a faulty CD player would output, plus the ground truth."""

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

_JOIN_FADE_S = 0.002


class DefectKind(StrEnum):
    SKIP = "skip"  # jump forward, content lost
    REPEAT = "repeat"  # jump back, content played again
    STUCK = "stuck"  # a short fragment loops several times
    DROPOUT = "dropout"  # muted while playback continues
    PAUSE = "pause"  # silence, then playback resumes where it stopped
    STOP = "stop"  # playback stops for good
    TRUNCATE = "truncate"  # the rest of the track is skipped
    MISSING = "missing"  # the whole track is skipped


class Defect(BaseModel):
    kind: DefectKind = Field(..., description="Type of fault to inject.")
    track: int = Field(..., description="0-based track index.")
    position_s: float = Field(default=0.0, description="Position inside the track.")
    size_s: float = Field(default=0.0, description="Jump length, mute or loop length.")
    count: int = Field(default=1, description="Extra loop repetitions (stuck).")


class TruthEvent(BaseModel):
    kind: DefectKind = Field(..., description="Injected fault.")
    track: int = Field(..., description="0-based track index.")
    position_s: float = Field(..., description="Position inside the track.")
    size_s: float = Field(..., description="Fault size in seconds.")
    count: int = Field(default=1, description="Loop repetitions (stuck).")
    played_s: float = Field(..., description="Time in the player output.")
    rec_s: float = Field(default=0.0, description="Time in the recording.")


class PlaybackPlan(BaseModel):
    gaps_s: list[float] = Field(
        ..., description="Silence before each track (index 0 unused)."
    )
    defects: list[Defect] = Field(default_factory=list, description="Faults to inject.")


def _faded(audio: np.ndarray, rate: int) -> np.ndarray:
    n = min(len(audio) // 2, int(_JOIN_FADE_S * rate))
    if n < 2:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def render_playback(
    tracks: list[np.ndarray], plan: PlaybackPlan, rate: int
) -> tuple[np.ndarray, list[TruthEvent]]:
    """Concatenate the tracks as the player would output them, faults included."""
    pieces: list[np.ndarray] = []
    truth: list[TruthEvent] = []
    produced = 0
    channels = tracks[0].shape[1]

    def emit(audio: np.ndarray) -> None:
        nonlocal produced
        pieces.append(audio)
        produced += len(audio)

    def silence(seconds: float) -> None:
        emit(np.zeros((int(round(seconds * rate)), channels), np.float32))

    for index, track in enumerate(tracks):
        defects = sorted(
            (d for d in plan.defects if d.track == index), key=lambda d: d.position_s
        )
        if any(d.kind == DefectKind.MISSING for d in defects):
            truth.append(
                TruthEvent(
                    kind=DefectKind.MISSING,
                    track=index,
                    position_s=0.0,
                    size_s=len(track) / rate,
                    played_s=produced / rate,
                )
            )
            continue
        if index > 0 and plan.gaps_s[index] > 0:
            silence(plan.gaps_s[index])
        cursor = 0
        stopped = False
        for defect in defects:
            position = int(round(defect.position_s * rate))
            size = int(round(defect.size_s * rate))
            if position < cursor or position >= len(track):
                continue
            emit(_faded(track[cursor:position], rate))
            truth.append(
                TruthEvent(
                    kind=defect.kind,
                    track=index,
                    position_s=defect.position_s,
                    size_s=defect.size_s,
                    count=defect.count,
                    played_s=produced / rate,
                )
            )
            if defect.kind == DefectKind.SKIP:
                cursor = min(len(track), position + size)
            elif defect.kind == DefectKind.REPEAT:
                cursor = max(0, position - size)
            elif defect.kind == DefectKind.STUCK:
                fragment = _faded(track[max(0, position - size) : position], rate)
                for _ in range(defect.count):
                    emit(fragment)
                cursor = position
            elif defect.kind == DefectKind.DROPOUT:
                silence(defect.size_s)
                cursor = min(len(track), position + size)
            elif defect.kind == DefectKind.PAUSE:
                silence(defect.size_s)
                cursor = position
            elif defect.kind == DefectKind.TRUNCATE:
                cursor = len(track)
            elif defect.kind == DefectKind.STOP:
                stopped = True
                break
        if stopped:
            break
        emit(_faded(track[cursor:], rate))
    return np.concatenate(pieces, axis=0), truth
