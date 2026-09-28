"""Append-only numpy buffer for data that grows while recording."""

import numpy as np


class GrowingArray:
    """Append-only buffer (1-D or 2-D rows) with amortised growth."""

    def __init__(self, width: int | None) -> None:
        shape = (4096,) if width is None else (4096, width)
        self._data = np.zeros(shape, np.float32)
        self.size = 0

    def extend(self, rows: np.ndarray) -> None:
        needed = self.size + len(rows)
        if needed > len(self._data):
            grown = np.zeros(
                (max(needed, 2 * len(self._data)), *self._data.shape[1:]), np.float32
            )
            grown[: self.size] = self._data[: self.size]
            self._data = grown
        self._data[self.size : needed] = rows
        self.size = needed

    def view(self) -> np.ndarray:
        return self._data[: self.size]
