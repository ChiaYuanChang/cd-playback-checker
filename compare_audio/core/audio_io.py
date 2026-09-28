"""Reading audio into the analysis format (mono float32 at a fixed rate).

Every read goes through ``open_audio``: sound files via libsndfile, and ``.cda``
CD tracks straight from the disc (Windows).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import soxr

_BLOCK_FRAMES = 1 << 16


class AudioReader(Protocol):
    samplerate: int
    frames: int
    channels: int

    def read(self, n: int) -> np.ndarray:
        """Next ``n`` frames (fewer at the end) as float32, shape (frames, channels)."""
        ...

    def seek(self, frame: int) -> None: ...

    def close(self) -> None: ...


class _SoundFileReader:
    def __init__(self, path: str | Path) -> None:
        self._file = sf.SoundFile(str(path))
        self.samplerate = self._file.samplerate
        self.frames = self._file.frames
        self.channels = self._file.channels

    def read(self, n: int) -> np.ndarray:
        return self._file.read(n, dtype="float32", always_2d=True)

    def seek(self, frame: int) -> None:
        self._file.seek(frame)

    def close(self) -> None:
        self._file.close()


@contextmanager
def open_audio(path: str | Path) -> Iterator[AudioReader]:
    if Path(path).suffix.lower() == ".cda":
        from compare_audio.audio import cd_audio  # Windows drive access, lazily

        reader: AudioReader = cd_audio.open_cda_track(path)
    else:
        reader = _SoundFileReader(path)
    try:
        yield reader
    finally:
        reader.close()


def audio_duration_s(path: str | Path) -> float:
    if Path(path).suffix.lower() == ".cda":
        from compare_audio.audio import cd_audio

        return cd_audio.parse_cda(path).duration_s  # from the header, no disc access
    info = sf.info(str(path))
    return info.frames / info.samplerate


def load_mono(path: str | Path, target_rate: int) -> np.ndarray:
    """Decode block by block, downmix to mono and resample.

    Reading in blocks keeps memory at the size of the (small) output even for a
    30 minute stereo file.
    """
    with open_audio(path) as source:
        resampler = (
            None
            if source.samplerate == target_rate
            else soxr.ResampleStream(
                source.samplerate, target_rate, 1, dtype="float32", quality="HQ"
            )
        )
        chunks: list[np.ndarray] = []
        while True:
            block = source.read(_BLOCK_FRAMES)
            if len(block) == 0:
                break
            mono = block.mean(axis=1)
            chunks.append(mono if resampler is None else resampler.resample_chunk(mono))
        if resampler is not None:
            chunks.append(resampler.resample_chunk(np.zeros(0, np.float32), last=True))
    if not chunks:
        return np.zeros(0, np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def resample_mono(
    signal: np.ndarray, source_rate: float, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(signal, dtype=np.float32)
    return soxr.resample(
        np.asarray(signal, dtype=np.float32), source_rate, target_rate, quality="HQ"
    ).astype(np.float32)


def read_clip(
    path: str | Path, start_s: float, duration_s: float
) -> tuple[np.ndarray, int]:
    """Read a short excerpt at the file's native rate (for listening)."""
    with open_audio(path) as source:
        rate = source.samplerate
        start = min(max(0, int(round(start_s * rate))), source.frames)
        source.seek(start)
        frames = max(0, min(int(round(duration_s * rate)), source.frames - start))
        data = source.read(frames)
    return data[:, : min(2, data.shape[1])], rate
