"""Where recording audio comes from: a microphone, or a file played in real time.

Both deliver mono float32 blocks through a queue. The PortAudio callback only copies
data into the queue; everything heavier happens in the session thread.
"""

import queue
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd
import soundfile as sf

from compare_audio.audio.input_devices import InputDevice

_BLOCK_S = 0.05


class AudioSource(Protocol):
    sample_rate: float

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self, timeout: float) -> np.ndarray | None: ...

    @property
    def overflows(self) -> int: ...

    @property
    def finished(self) -> bool: ...


class MicrophoneSource:
    def __init__(self, device: InputDevice) -> None:
        self.device = device
        self.sample_rate = device.default_rate
        self._channels = max(1, min(2, device.channels))
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._overflows = 0

    @property
    def overflows(self) -> int:
        return self._overflows

    @property
    def finished(self) -> bool:
        return False

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        if status.input_overflow:
            self._overflows += 1
        block = indata[:, 0] if self._channels == 1 else indata.mean(axis=1)
        self._queue.put(block.astype(np.float32, copy=True))

    def start(self) -> None:
        # WASAPI shared mode only accepts the device's own rate, so ask for that
        # first; the analysis resamples anyway. Some drivers insist on their own
        # channel count or a standard rate, hence the fallbacks.
        attempts = [
            (self._channels, self.device.default_rate),
            (1, self.device.default_rate),
            (self.device.channels, self.device.default_rate),
            (self._channels, 48000.0),
            (self._channels, 44100.0),
        ]
        error: Exception | None = None
        for channels, rate in dict.fromkeys(attempts):
            try:
                stream = sd.InputStream(
                    device=self.device.index,
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=int(rate * _BLOCK_S),
                    callback=self._callback,
                )
            except sd.PortAudioError as exc:
                error = exc
                continue
            self._channels, self.sample_rate = channels, rate
            self._stream = stream
            stream.start()
            return
        raise RuntimeError(f"{self.device.name}：{error}")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class FileSource:
    """Plays a recording into the pipeline as if it came from a microphone.

    Used to try the app without a CD player (e.g. with the synthetic test set);
    ``speed`` > 1 replays faster than real time.
    """

    def __init__(self, path: str | Path, speed: float = 1.0) -> None:
        self.path = Path(path)
        self.speed = speed
        self.sample_rate = float(sf.info(str(self.path)).samplerate)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def overflows(self) -> int:
        return 0

    @property
    def finished(self) -> bool:
        return self._done.is_set() and self._queue.empty()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        block = int(self.sample_rate * _BLOCK_S)
        started = time.monotonic()
        sent = 0
        with sf.SoundFile(str(self.path)) as source:
            while not self._stop.is_set():
                data = source.read(block, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break
                self._queue.put(data.mean(axis=1).astype(np.float32))
                sent += len(data)
                due = started + sent / self.sample_rate / self.speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        self._done.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class LevelMonitor:
    """Input level of a microphone while not recording (for the setup meter)."""

    def __init__(self, device: InputDevice) -> None:
        self.device = device
        self.peak_db = -120.0
        self.rms_db = -120.0
        self.clipped = False
        self.all_zero = False
        self._zero_blocks = 0
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        peak = float(np.max(np.abs(indata))) if len(indata) else 0.0
        rms = (
            float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
            if len(indata)
            else 0.0
        )
        self.peak_db = 20 * np.log10(max(peak, 1e-6))
        self.rms_db = 20 * np.log10(max(rms, 1e-6))
        self.clipped = self.clipped or peak >= 0.99
        self._zero_blocks = self._zero_blocks + 1 if peak == 0.0 else 0
        self.all_zero = self._zero_blocks >= 20

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device.index,
            samplerate=self.device.default_rate,
            channels=max(1, min(2, self.device.channels)),
            dtype="float32",
            blocksize=int(self.device.default_rate * _BLOCK_S),
            callback=self._callback,
        )
        self._stream.start()

    def take_clip(self) -> bool:
        clipped, self.clipped = self.clipped, False
        return clipped

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
