"""Small synthetic programs shared by the core tests (a few seconds of audio)."""

import numpy as np
import pytest

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.reference_program import ReferenceProgram, ReferenceTrack
from compare_audio.core.spectral_features import extract_features

RATE = 16000


def _tone_melody(seed: int, seconds: float) -> np.ndarray:
    """Random notes with harmonics and clear onsets: easy to match, not periodic."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE), np.float32)
    position = 0.3
    while position < seconds - 0.3:
        length = float(rng.choice([0.15, 0.25, 0.4]))
        freq = 220.0 * 2 ** (rng.integers(0, 24) / 12)
        n = int(length * RATE)
        t = np.arange(n) / RATE
        note = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 6))
        note *= np.exp(-t / (0.5 * length))
        start = int(position * RATE)
        out[start : start + n] += 0.2 * note[: len(out) - start]
        position += length
    return out


@pytest.fixture(scope="session")
def config() -> AnalysisConfig:
    return AnalysisConfig()


@pytest.fixture(scope="session")
def tracks_audio() -> list[np.ndarray]:
    return [_tone_melody(1, 30.0), _tone_melody(2, 30.0)]


@pytest.fixture(scope="session")
def program(config: AnalysisConfig, tracks_audio: list[np.ndarray]) -> ReferenceProgram:
    signal = np.concatenate(tracks_audio)
    starts = np.cumsum([0] + [len(t) for t in tracks_audio])[:-1] / RATE
    tracks = [
        ReferenceTrack(i, f"T{i + 1}", "", float(starts[i]), len(t) / RATE)
        for i, t in enumerate(tracks_audio)
    ]
    return ReferenceProgram(config, tracks, extract_features(signal, config))


def room(signal: np.ndarray, pre_roll_s: float, seed: int = 0) -> np.ndarray:
    """Quieter copy with a little noise and a noise-only lead-in."""
    rng = np.random.default_rng(seed)
    body = np.concatenate([np.zeros(int(pre_roll_s * RATE), np.float32), 0.4 * signal])
    return (body + rng.normal(0, 0.002, len(body))).astype(np.float32)
