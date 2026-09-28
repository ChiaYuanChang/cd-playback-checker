"""Listening to short excerpts (recording vs reference) on the default output."""

import numpy as np
import sounddevice as sd


def play_clip(data: np.ndarray, rate: float) -> None:
    sd.stop()
    if len(data):
        peak = float(np.max(np.abs(data)))
        if peak > 0:
            data = data * min(1.0, 0.9 / peak) if peak > 0.9 else data
        sd.play(data, int(rate))


def stop_playback() -> None:
    sd.stop()
