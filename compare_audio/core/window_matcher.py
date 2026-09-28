"""Normalised cross-correlation (NCC) of a feature window against the program.

The score removes each band's mean inside every window before correlating, so a
constant per-band gain (speaker/room/mic colouration, volume knob) cancels exactly.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

_FLAT_VARIANCE = 1e-3  # mean squared deviation (dB^2) below which a window is "flat"


@dataclass(frozen=True, slots=True)
class WindowMatch:
    lag: float  # program frame where the query window starts (sub-frame)
    score: float


@dataclass(frozen=True, slots=True)
class LocalSearch:
    best: WindowMatch
    at_preferred: (
        WindowMatch | None
    )  # best peak near the preferred lag, if one was given


def centered_window_energy(features: np.ndarray, window: int) -> np.ndarray:
    """Sum over bands of the within-window variance * window, for every window start."""
    n = len(features)
    if n < window:
        return np.zeros(0, np.float64)
    data = features.astype(np.float64)
    c1 = np.zeros((n + 1, data.shape[1]), np.float64)
    c2 = np.zeros((n + 1, data.shape[1]), np.float64)
    np.cumsum(data, axis=0, out=c1[1:])
    np.cumsum(data * data, axis=0, out=c2[1:])
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    return np.maximum((s2 - s1 * s1 / window).sum(axis=1), 0.0)


def ncc_scores(
    query: np.ndarray, reference: np.ndarray, reference_energy: np.ndarray
) -> np.ndarray:
    """Score of ``query`` (W x B) against every W-long slice of ``reference``."""
    window, bands = query.shape
    centered = query - query.mean(axis=0, keepdims=True)
    query_norm = float(np.sqrt(np.sum(centered.astype(np.float64) ** 2)))
    n_lags = len(reference) - window + 1
    if n_lags <= 0 or query_norm < np.sqrt(_FLAT_VARIANCE * window * bands):
        return np.zeros(max(n_lags, 0), np.float64)
    numerator = fftconvolve(
        reference.astype(np.float32),
        centered[::-1].astype(np.float32),
        mode="valid",
        axes=0,
    ).sum(axis=1)
    flat = reference_energy < _FLAT_VARIANCE * window * bands
    denominator = query_norm * np.sqrt(np.where(flat, 1.0, reference_energy))
    scores = numerator / denominator
    scores[flat] = 0.0
    return np.clip(scores, -1.0, 1.0)


def refine_peak(scores: np.ndarray, index: int) -> float:
    """Parabolic interpolation of a discrete peak; returns a fractional index."""
    if 0 < index < len(scores) - 1:
        left, mid, right = scores[index - 1], scores[index], scores[index + 1]
        curvature = left - 2.0 * mid + right
        if curvature < 0:
            return index + float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(index)


def decimate_frames(features: np.ndarray, factor: int) -> np.ndarray:
    usable = (len(features) // factor) * factor
    if usable == 0:
        return np.zeros((0, features.shape[1]), features.dtype)
    return features[:usable].reshape(-1, factor, features.shape[1]).mean(axis=1)


class ProgramMatcher:
    """Pre-computed program statistics for fast local and global window searches."""

    def __init__(
        self, program_features: np.ndarray, window: int, coarse_factor: int
    ) -> None:
        # Removing the global mean keeps the cumulative sums well conditioned; the
        # score is invariant to it.
        self.features = (program_features - program_features.mean()).astype(np.float32)
        self.window = window
        self.energy = centered_window_energy(self.features, window)
        self.coarse_factor = coarse_factor
        self.coarse = decimate_frames(self.features, coarse_factor)
        self.coarse_window = max(2, window // coarse_factor)
        self.coarse_energy = centered_window_energy(self.coarse, self.coarse_window)

    @property
    def max_lag(self) -> int:
        return len(self.features) - self.window

    def local_search(
        self,
        query: np.ndarray,
        lag_lo: int,
        lag_hi: int,
        prefer_lag: float | None = None,
        prefer_radius: int = 0,
        prefer_margin: float = 0.0,
    ) -> LocalSearch | None:
        """Best lag in ``[lag_lo, lag_hi]``.

        With ``prefer_lag`` the peak near the predicted position wins unless another
        peak beats it by more than ``prefer_margin``: in music that repeats bar for
        bar, several lags score almost the same and continuity is the tie-breaker.
        """
        lag_lo = max(0, lag_lo)
        lag_hi = min(self.max_lag, lag_hi)
        if lag_hi < lag_lo:
            return None
        scores = ncc_scores(
            query,
            self.features[lag_lo : lag_hi + self.window],
            self.energy[lag_lo : lag_hi + 1],
        )
        best = int(np.argmax(scores))
        preferred: WindowMatch | None = None
        if prefer_lag is not None:
            center = int(round(prefer_lag)) - lag_lo
            lo = max(0, center - prefer_radius)
            hi = min(len(scores), center + prefer_radius + 1)
            if hi > lo:
                near = lo + int(np.argmax(scores[lo:hi]))
                preferred = self._match_at(scores, near, lag_lo)
                if scores[near] >= scores[best] - prefer_margin:
                    best = near
        return LocalSearch(
            best=self._match_at(scores, best, lag_lo), at_preferred=preferred
        )

    @staticmethod
    def _match_at(scores: np.ndarray, index: int, lag_lo: int) -> WindowMatch:
        return WindowMatch(
            lag=lag_lo + refine_peak(scores, index), score=float(scores[index])
        )

    def global_search(self, query: np.ndarray, n_candidates: int) -> list[WindowMatch]:
        """Coarse search over the whole program, then refine the best peaks."""
        coarse_query = decimate_frames(query, self.coarse_factor)[: self.coarse_window]
        if (
            len(coarse_query) < self.coarse_window
            or len(self.coarse) < self.coarse_window
        ):
            return []
        scores = ncc_scores(coarse_query, self.coarse, self.coarse_energy)
        separation = max(1, self.window // self.coarse_factor)
        candidates: list[WindowMatch] = []
        working = scores.copy()
        for _ in range(n_candidates):
            peak = int(np.argmax(working))
            if working[peak] <= 0.05:
                break
            lag = peak * self.coarse_factor
            refined = self.local_search(
                query, lag - 2 * self.coarse_factor, lag + 2 * self.coarse_factor
            )
            if refined is not None:
                candidates.append(refined.best)
            working[max(0, peak - separation) : peak + separation + 1] = -np.inf
        return sorted(candidates, key=lambda match: match.score, reverse=True)
