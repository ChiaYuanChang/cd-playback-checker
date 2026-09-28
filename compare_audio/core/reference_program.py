"""The reference "program": all tracks of a test disc laid end to end.

Tracks are pushed through one streaming extractor, so the feature timeline is exactly
the timeline of the concatenated audio. A gapless transition on the disc is then
continuous here too, and a pause the player inserts between tracks shows up as a
negative jump that the event detector recognises at track boundaries.
"""

import bisect
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from pydantic import BaseModel, Field

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.spectral_features import FeatureExtractor, FeatureSet

_READ_BLOCK = 1 << 16
_CACHE_VERSION = 1

ProgressCallback = Callable[[float], None]


class TrackSource(BaseModel):
    """One reference track as listed in a test profile."""

    title: str = Field(..., description="Display name of the track.")
    path: str = Field(
        ..., description="Audio file path (absolute or profile-relative)."
    )


@dataclass(frozen=True)
class ReferenceTrack:
    index: int
    title: str
    source_path: str
    start_s: float  # program time where the track begins
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass
class ReferenceProgram:
    config: AnalysisConfig
    tracks: list[ReferenceTrack]
    features: FeatureSet

    @property
    def duration_s(self) -> float:
        return self.tracks[-1].end_s if self.tracks else 0.0

    @property
    def n_frames(self) -> int:
        return len(self.features.log_mel)

    def track_index_at(self, program_s: float) -> int:
        starts = [track.start_s for track in self.tracks]
        return max(
            0, min(len(self.tracks) - 1, bisect.bisect_right(starts, program_s) - 1)
        )

    def locate(self, program_s: float) -> tuple[int, float]:
        """(track index, seconds into that track) for a program position."""
        index = self.track_index_at(program_s)
        return index, program_s - self.tracks[index].start_s


def frame_center_s(frame: float, config: AnalysisConfig) -> float:
    return (frame * config.hop + config.n_fft / 2) / config.sample_rate


def _cache_key(sources: list[TrackSource], config: AnalysisConfig) -> str:
    stamp: list[str] = [config.feature_signature(), str(_CACHE_VERSION)]
    for source in sources:
        path = Path(source.path).resolve()
        stat = path.stat()
        stamp.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha1("\n".join(stamp).encode("utf-8")).hexdigest()[:20]


def _clamp_track_floors(
    log_mel: np.ndarray, tracks: list[ReferenceTrack], config: AnalysisConfig
) -> None:
    """Clamp very quiet detail the microphone could never hear (in place)."""
    for track in tracks:
        first = max(
            0,
            int(
                np.ceil(
                    (track.start_s * config.sample_rate - config.n_fft / 2) / config.hop
                )
            ),
        )
        last = min(
            len(log_mel),
            int(
                np.floor(
                    (track.end_s * config.sample_rate - config.n_fft / 2) / config.hop
                )
            )
            + 1,
        )
        if last <= first:
            continue
        segment = log_mel[first:last]
        np.maximum(segment, segment.max() - config.ref_floor_db, out=segment)


def _decode_into(
    path: str,
    extractor: FeatureExtractor,
    config: AnalysisConfig,
    on_block: Callable[[int, int], None],
) -> tuple[int, list[FeatureSet]]:
    """Stream one file through the extractor; returns (n_samples, features)."""
    parts: list[FeatureSet] = []
    produced = 0
    with sf.SoundFile(path) as source:
        total = max(1, source.frames)
        resampler = (
            None
            if source.samplerate == config.sample_rate
            else soxr.ResampleStream(
                source.samplerate, config.sample_rate, 1, dtype="float32", quality="HQ"
            )
        )
        done = 0
        while True:
            block = source.read(_READ_BLOCK, dtype="float32", always_2d=True)
            last = len(block) == 0
            mono = block.mean(axis=1) if len(block) else np.zeros(0, np.float32)
            if resampler is not None:
                mono = resampler.resample_chunk(mono, last=last)
            if len(mono):
                produced += len(mono)
                parts.append(extractor.push(mono))
            done += len(block)
            on_block(done, total)
            if last:
                break
    return produced, parts


def load_reference_program(
    sources: list[TrackSource],
    config: AnalysisConfig,
    cache_dir: Path | None = None,
    progress: ProgressCallback | None = None,
) -> ReferenceProgram:
    if not sources:
        raise ValueError("A test profile needs at least one reference track.")
    for source in sources:
        if not Path(source.path).is_file():
            raise FileNotFoundError(f"找不到參考音檔：{source.path}")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"ref-{_cache_key(sources, config)}.npz"
        if cache_path.is_file():
            cached = _load_cached(cache_path, sources, config)
            if cached is not None:
                if progress:
                    progress(1.0)
                return cached

    extractor = FeatureExtractor(config)
    parts: list[FeatureSet] = []
    tracks: list[ReferenceTrack] = []
    start_samples = 0
    for index, source in enumerate(sources):

        def report(done: int, total: int, index: int = index) -> None:
            if progress:
                progress((index + done / total) / len(sources))

        n_samples, track_parts = _decode_into(source.path, extractor, config, report)
        parts.extend(track_parts)
        tracks.append(
            ReferenceTrack(
                index=index,
                title=source.title,
                source_path=str(source.path),
                start_s=start_samples / config.sample_rate,
                duration_s=n_samples / config.sample_rate,
            )
        )
        start_samples += n_samples

    features = FeatureSet.concatenate(parts, config.n_mels)
    _clamp_track_floors(features.log_mel, tracks, config)
    program = ReferenceProgram(config=config, tracks=tracks, features=features)
    if cache_path is not None:
        _save_cached(cache_path, program)
    return program


def _save_cached(path: Path, program: ReferenceProgram) -> None:
    meta = [
        {
            "title": t.title,
            "path": t.source_path,
            "start_s": t.start_s,
            "duration_s": t.duration_s,
        }
        for t in program.tracks
    ]
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        log_mel=program.features.log_mel,
        band_env_db=program.features.band_env_db,
        wave_min=program.features.wave_min,
        wave_max=program.features.wave_max,
        meta=np.array(json.dumps(meta)),
    )
    tmp.replace(path)


def _load_cached(
    path: Path, sources: list[TrackSource], config: AnalysisConfig
) -> ReferenceProgram | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            features = FeatureSet(
                log_mel=data["log_mel"],
                band_env_db=data["band_env_db"],
                wave_min=data["wave_min"],
                wave_max=data["wave_max"],
            )
    except (OSError, KeyError, ValueError):
        return None
    if len(meta) != len(sources):
        return None
    tracks = [
        ReferenceTrack(
            index=i,
            title=source.title,
            source_path=str(source.path),
            start_s=float(entry["start_s"]),
            duration_s=float(entry["duration_s"]),
        )
        for i, (entry, source) in enumerate(zip(meta, sources, strict=True))
    ]
    return ReferenceProgram(config=config, tracks=tracks, features=features)
