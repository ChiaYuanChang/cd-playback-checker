"""Feature extraction shared by the reference and the recording.

Frame conventions (identical for reference and recording, so constant offsets cancel):
- log-mel frame ``i`` covers samples ``[i*hop, i*hop + n_fft)``
- envelope block ``j`` covers samples ``[j*H, (j+1)*H)`` with ``H = envelope_hop``

An offset of ``d`` seconds between recording and program therefore maps index ``i``
to ``i + d * frame_rate`` (log-mel) and ``j`` to ``j + d * envelope_rate`` (envelopes).
"""

from dataclasses import dataclass

import numpy as np
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, get_window, sosfilt

from compare_audio.core.analysis_config import AnalysisConfig

_POWER_EPS = 1e-10
_FRAMES_PER_CHUNK = 2048
_BATCH_SAMPLES = 1 << 18


@dataclass
class FeatureSet:
    """Features of a signal (or of the newly processed part of a stream)."""

    log_mel: np.ndarray  # (n_frames, n_mels) float32, dB
    band_env_db: np.ndarray  # (n_blocks,) float32, band-limited power in dB
    wave_min: np.ndarray  # (n_blocks,) float32
    wave_max: np.ndarray  # (n_blocks,) float32

    @staticmethod
    def empty(n_mels: int) -> "FeatureSet":
        return FeatureSet(
            log_mel=np.zeros((0, n_mels), np.float32),
            band_env_db=np.zeros(0, np.float32),
            wave_min=np.zeros(0, np.float32),
            wave_max=np.zeros(0, np.float32),
        )

    @staticmethod
    def concatenate(parts: list["FeatureSet"], n_mels: int) -> "FeatureSet":
        if not parts:
            return FeatureSet.empty(n_mels)
        return FeatureSet(
            log_mel=np.concatenate([p.log_mel for p in parts], axis=0),
            band_env_db=np.concatenate([p.band_env_db for p in parts]),
            wave_min=np.concatenate([p.wave_min for p in parts]),
            wave_max=np.concatenate([p.wave_max for p in parts]),
        )


def mel_filterbank(config: AnalysisConfig) -> np.ndarray:
    """Triangular HTK-mel filters, area normalised. Shape ``(n_fft//2+1, n_mels)``."""

    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    bin_hz = np.arange(config.n_fft // 2 + 1) * config.sample_rate / config.n_fft
    edges_mel = np.linspace(
        hz_to_mel(np.array(config.fmin_hz)),
        hz_to_mel(np.array(config.fmax_hz)),
        config.n_mels + 2,
    )
    edges_hz = mel_to_hz(edges_mel)
    bank = np.zeros((len(bin_hz), config.n_mels), np.float64)
    for m in range(config.n_mels):
        lower, center, upper = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rising = (bin_hz - lower) / (center - lower)
        falling = (upper - bin_hz) / (upper - center)
        bank[:, m] = np.maximum(0.0, np.minimum(rising, falling)) * (
            2.0 / (upper - lower)
        )
    return bank.astype(np.float32)


class FeatureExtractor:
    """Streaming feature extractor; batch extraction feeds it in chunks.

    Holding the partial-frame remainders between calls makes the output identical
    whether a signal is pushed at once or in arbitrary pieces.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config
        self._bank = mel_filterbank(config)
        self._window = get_window("hann", config.n_fft, fftbins=True).astype(np.float32)
        self._stft_pending = np.zeros(0, np.float32)
        self._block = config.envelope_hop
        self._block_pending = np.zeros(0, np.float32)
        self._filtered_pending = np.zeros(0, np.float32)
        self._sos = butter(
            4,
            config.envelope_band_hz,
            btype="bandpass",
            fs=config.sample_rate,
            output="sos",
        )
        self._filter_state = np.zeros((self._sos.shape[0], 2), np.float64)
        self.n_samples = 0

    def push(self, samples: np.ndarray) -> FeatureSet:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return FeatureSet.empty(self._config.n_mels)
        self.n_samples += len(x)
        log_mel = self._push_stft(x)
        filtered, self._filter_state = sosfilt(self._sos, x, zi=self._filter_state)
        band_env_db = self._push_band_env(filtered.astype(np.float32))
        wave_min, wave_max = self._push_wave_blocks(x)
        return FeatureSet(log_mel, band_env_db, wave_min, wave_max)

    def _push_stft(self, x: np.ndarray) -> np.ndarray:
        n_fft, hop = self._config.n_fft, self._config.hop
        buffer = np.concatenate([self._stft_pending, x])
        n_frames = 0 if len(buffer) < n_fft else 1 + (len(buffer) - n_fft) // hop
        if n_frames == 0:
            self._stft_pending = buffer
            return np.zeros((0, self._config.n_mels), np.float32)
        frames = sliding_window_view(buffer, n_fft)[::hop][:n_frames]
        out = np.empty((n_frames, self._config.n_mels), np.float32)
        for start in range(0, n_frames, _FRAMES_PER_CHUNK):
            chunk = frames[start : start + _FRAMES_PER_CHUNK] * self._window
            spectrum = scipy.fft.rfft(chunk, axis=1, workers=-1)
            power = spectrum.real**2 + spectrum.imag**2
            out[start : start + len(chunk)] = 10.0 * np.log10(
                power @ self._bank + _POWER_EPS
            )
        self._stft_pending = buffer[n_frames * hop :].copy()
        return out

    def _push_band_env(self, filtered: np.ndarray) -> np.ndarray:
        buffer = np.concatenate([self._filtered_pending, filtered])
        n_blocks = len(buffer) // self._block
        self._filtered_pending = buffer[n_blocks * self._block :].copy()
        if n_blocks == 0:
            return np.zeros(0, np.float32)
        blocks = buffer[: n_blocks * self._block].reshape(n_blocks, self._block)
        power = np.mean(blocks.astype(np.float64) ** 2, axis=1)
        return (10.0 * np.log10(power + _POWER_EPS)).astype(np.float32)

    def _push_wave_blocks(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        buffer = np.concatenate([self._block_pending, x])
        n_blocks = len(buffer) // self._block
        self._block_pending = buffer[n_blocks * self._block :].copy()
        if n_blocks == 0:
            empty = np.zeros(0, np.float32)
            return empty, empty
        blocks = buffer[: n_blocks * self._block].reshape(n_blocks, self._block)
        return blocks.min(axis=1), blocks.max(axis=1)


def extract_features(signal: np.ndarray, config: AnalysisConfig) -> FeatureSet:
    extractor = FeatureExtractor(config)
    parts = [
        extractor.push(signal[start : start + _BATCH_SAMPLES])
        for start in range(0, len(signal), _BATCH_SAMPLES)
    ]
    return FeatureSet.concatenate(parts, config.n_mels)


def frame_levels_db(log_mel: np.ndarray) -> np.ndarray:
    """Broadband level of each frame: power mean over the mel bands, in dB."""
    if len(log_mel) == 0:
        return np.zeros(0, np.float32)
    peak = log_mel.max(axis=1, keepdims=True)
    mean_power = np.mean(10.0 ** ((log_mel - peak) / 10.0), axis=1)
    return (peak[:, 0] + 10.0 * np.log10(mean_power)).astype(np.float32)
