"""Reading audio files into the analysis format (mono float32 at a fixed rate)."""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

_BLOCK_FRAMES = 1 << 16


def audio_duration_s(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def load_mono(path: str | Path, target_rate: int) -> np.ndarray:
    """Decode a file block by block, downmix to mono and resample.

    Reading in blocks keeps memory at the size of the (small) output even for a
    30 minute stereo file.
    """
    with sf.SoundFile(str(path)) as source:
        resampler = (
            None
            if source.samplerate == target_rate
            else soxr.ResampleStream(
                source.samplerate, target_rate, 1, dtype="float32", quality="HQ"
            )
        )
        chunks: list[np.ndarray] = []
        while True:
            block = source.read(_BLOCK_FRAMES, dtype="float32", always_2d=True)
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
    with sf.SoundFile(str(path)) as source:
        rate = source.samplerate
        start = max(0, int(round(start_s * rate)))
        start = min(start, source.frames)
        source.seek(start)
        frames = max(0, min(int(round(duration_s * rate)), source.frames - start))
        data = source.read(frames, dtype="float32", always_2d=True)
    return data[:, : min(2, data.shape[1])], rate
