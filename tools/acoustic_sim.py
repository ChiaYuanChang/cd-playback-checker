"""Speaker -> room -> microphone -> sound card, with noise and clock drift."""

import numpy as np
import soxr
from pydantic import BaseModel, Field
from scipy.signal import butter, oaconvolve, sosfilt

from tools.defect_injection import TruthEvent

MIC_RATE = 48000


class RoomCondition(BaseModel):
    name: str = Field(..., description="Short label for reports.")
    snr_db: float = Field(..., description="Music vs noise power in 150-6000 Hz.")
    drr_db: float = Field(
        ..., description="Direct-to-reverberant ratio (mic distance)."
    )
    rt60_s: float = Field(..., description="Reverberation time of the room.")
    drift_ppm: float = Field(default=0.0, description="Player vs sound card clock.")
    noise_kind: str = Field(default="pink", description="pink | babble | mixed")
    pre_roll_s: float = Field(default=6.0, description="Noise before playback starts.")
    post_roll_s: float = Field(default=4.0, description="Noise after playback ends.")


def _peaking(freq: float, gain_db: float, q: float, rate: int) -> np.ndarray:
    """RBJ peaking EQ biquad as one SOS row."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / rate
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return np.array(
        [
            [
                b[0] / den[0],
                b[1] / den[0],
                b[2] / den[0],
                1.0,
                den[1] / den[0],
                den[2] / den[0],
            ]
        ]
    )


def _speaker(x: np.ndarray, rate: int, rng: np.random.Generator) -> np.ndarray:
    """Small speaker: no bass, rolled-off treble, a couple of resonances."""
    sos = np.vstack(
        [
            butter(2, 150.0, btype="highpass", fs=rate, output="sos"),
            butter(2, 9000.0, btype="lowpass", fs=rate, output="sos"),
            _peaking(rng.uniform(600, 1200), rng.uniform(2, 5), 1.2, rate),
            _peaking(rng.uniform(2500, 4500), rng.uniform(-5, -2), 1.5, rate),
        ]
    )
    y = sosfilt(sos, x)
    peak = np.max(np.abs(y)) + 1e-9
    return np.tanh(1.2 * y / peak) * peak / np.tanh(1.2)


def room_impulse_response(cond: RoomCondition, rng: np.random.Generator) -> np.ndarray:
    n = int((cond.rt60_s * 1.2 + 0.03) * MIC_RATE)
    ir = np.zeros(n)
    direct = 12
    ir[direct] = 1.0
    reverb = np.zeros(n)
    for _ in range(10):  # early reflections
        at = direct + int(rng.uniform(0.002, 0.03) * MIC_RATE)
        reverb[at] += rng.choice([-1, 1]) * rng.uniform(0.2, 0.6)
    t = np.arange(n) / MIC_RATE
    tail_start = direct + int(0.005 * MIC_RATE)
    noise = rng.standard_normal(n)
    low = sosfilt(butter(2, 2000, btype="lowpass", fs=MIC_RATE, output="sos"), noise)
    high = noise - low
    late = low * np.exp(-6.91 * t / cond.rt60_s) + high * np.exp(
        -6.91 * t / (0.6 * cond.rt60_s)
    )
    late[:tail_start] = 0
    reverb += late * 0.5
    energy = np.sum(reverb**2)
    reverb *= np.sqrt(10 ** (-cond.drr_db / 10) / energy)
    return ir + reverb


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / MIC_RATE)
    spectrum /= np.sqrt(np.maximum(freqs, 20.0))
    return np.fft.irfft(spectrum, n)


def _babble(n: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-like noise in talk spurts (people talking nearby)."""
    carrier = sosfilt(
        butter(2, (250, 3500), btype="bandpass", fs=MIC_RATE, output="sos"),
        rng.standard_normal(n),
    )
    syllables = sosfilt(
        butter(2, 5.0, btype="lowpass", fs=MIC_RATE, output="sos"),
        rng.standard_normal(n),
    )
    envelope = np.maximum(0, syllables) * 40
    active = np.zeros(n)
    position = int(rng.uniform(0, 3) * MIC_RATE)
    while position < n:
        length = int(rng.uniform(1.0, 6.0) * MIC_RATE)
        active[position : position + length] = 1.0
        position += length + int(rng.uniform(2.0, 10.0) * MIC_RATE)
    return carrier * envelope * active


def _knocks(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    position = int(rng.uniform(5, 20) * MIC_RATE)
    length = int(0.15 * MIC_RATE)
    t = np.arange(length) / MIC_RATE
    while position + length < n:
        freq = rng.uniform(150, 900)
        out[position : position + length] += (
            np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.03) * rng.uniform(2, 6)
        )
        position += int(rng.uniform(8, 40) * MIC_RATE)
    return out


def _band_power(x: np.ndarray) -> float:
    band = sosfilt(
        butter(2, (150, 6000), btype="bandpass", fs=MIC_RATE, output="sos"), x
    )
    return float(np.mean(band**2))


def simulate_recording(
    played: np.ndarray,
    played_rate: int,
    cond: RoomCondition,
    truth: list[TruthEvent],
    rng: np.random.Generator,
) -> np.ndarray:
    """Mono float32 recording at MIC_RATE; fills ``rec_s`` of the truth events."""
    mono = played.mean(axis=1) if played.ndim == 2 else played
    speaker = _speaker(mono.astype(np.float64), played_rate, rng)
    # Played at the player's clock, captured at the sound card's clock.
    captured = soxr.resample(
        speaker.astype(np.float32), played_rate, MIC_RATE * (1 + cond.drift_ppm * 1e-6)
    ).astype(np.float64)
    wet = oaconvolve(captured, room_impulse_response(cond, rng), mode="full")
    wet = sosfilt(butter(1, 80, btype="highpass", fs=MIC_RATE, output="sos"), wet)

    pre, post = int(cond.pre_roll_s * MIC_RATE), int(cond.post_roll_s * MIC_RATE)
    music = np.concatenate([np.zeros(pre), wet, np.zeros(post)])
    n = len(music)
    if cond.noise_kind == "babble":
        noise = _babble(n, rng) + 0.15 * _pink(n, rng)
    elif cond.noise_kind == "mixed":
        noise = _pink(n, rng) + 0.8 * _babble(n, rng) / 3 + 0.02 * _knocks(n, rng)
    else:
        noise = _pink(n, rng)
    active = np.abs(music) > 1e-4
    music_power = _band_power(music[active]) if active.any() else 1.0
    noise *= np.sqrt(music_power / (_band_power(noise) * 10 ** (cond.snr_db / 10)))
    recording = music + noise + rng.normal(0, 1e-5, n)
    recording *= 0.4 / (np.max(np.abs(recording)) + 1e-9)
    recording = np.round(recording * 32767) / 32767  # 16-bit sound card

    stretch = 1 + cond.drift_ppm * 1e-6
    for event in truth:
        event.rec_s = cond.pre_roll_s + event.played_s * stretch + 12 / MIC_RATE
    return recording.astype(np.float32)
