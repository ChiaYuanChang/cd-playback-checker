"""Background threads. Signals carry results back to the GUI thread (queued)."""

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from PySide6.QtCore import QThread, Signal

from compare_audio.audio.audio_sources import AudioSource
from compare_audio.audio.cd_audio import CdaCancelled, parse_cda, rip_track
from compare_audio.core.audio_io import load_mono
from compare_audio.core.live_analysis import LiveAnalyzer, LiveStatus
from compare_audio.core.recording_analysis import analyze_signal
from compare_audio.core.reference_program import (
    ReferenceProgram,
    load_reference_program,
)
from compare_audio.profiles.test_profile import TestProfile

_LEVEL_EVERY_S = 0.05
_UPDATE_EVERY_S = 0.25
_EVENTS_EVERY_S = 5.0
_AUTO_STOP_AFTER_S = 3.0
_LIVE_LEVEL_RATE_HZ = 20


class ReferenceLoader(QThread):
    progress = Signal(float)
    loaded = Signal(object)  # ReferenceProgram
    failed = Signal(str)

    def __init__(self, profile: TestProfile, cache_dir: Path) -> None:
        super().__init__()
        self.profile = profile
        self.cache_dir = cache_dir

    def run(self) -> None:
        try:
            program = load_reference_program(
                self.profile.tracks,
                self.profile.analysis,
                cache_dir=self.cache_dir,
                progress=self.progress.emit,
            )
        except Exception as exc:  # shown to the user, never crash the thread
            self.failed.emit(str(exc))
            return
        self.loaded.emit(program)


@dataclass(frozen=True)
class LiveUpdate:
    status: LiveStatus
    rec_s: np.ndarray
    program_s: np.ndarray  # nan where not matched
    level_t: np.ndarray
    level_db: np.ndarray
    overflows: int


class SessionWorker(QThread):
    """Records from a source into a WAV file while analysing live."""

    level = Signal(float, float, bool)  # peak dBFS, rms dBFS, clipped
    updated = Signal(object)  # LiveUpdate
    provisional = Signal(list)  # list[DetectedEvent]
    auto_stopping = Signal(str)  # reason
    analysing = Signal()
    analysis_progress = Signal(float)
    completed = Signal(object, str)  # AnalysisResult, wav path
    failed = Signal(str)

    def __init__(
        self,
        program: ReferenceProgram,
        source: AudioSource,
        wav_path: Path,
        auto_stop: bool,
    ) -> None:
        super().__init__()
        self.program = program
        self.source = source
        self.wav_path = wav_path
        self.auto_stop = auto_stop
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self.source.start()
        except Exception as exc:
            self.failed.emit(f"無法開啟收音裝置：{exc}")
            return
        try:
            live = self._record()
        except Exception as exc:
            self.source.stop()
            self.failed.emit(f"錄音中斷：{exc}")
            return
        self.analysing.emit()
        try:
            result = live.finish(progress=self.analysis_progress.emit)
        except Exception as exc:
            self.failed.emit(f"分析失敗：{exc}")
            return
        self.completed.emit(result, str(self.wav_path))

    def _record(self) -> LiveAnalyzer:
        config = self.program.config
        rate = self.source.sample_rate
        resampler = soxr.ResampleStream(rate, config.sample_rate, 1, dtype="float32")
        live = LiveAnalyzer(self.program)
        max_duration = self.program.duration_s * 1.3 + 120.0
        self.wav_path.parent.mkdir(parents=True, exist_ok=True)
        last_level = last_update = last_events = 0.0
        done_at: float | None = None
        clipped = False
        with sf.SoundFile(
            str(self.wav_path),
            "w",
            samplerate=int(round(rate)),
            channels=1,
            subtype="PCM_16",
        ) as wav:
            while not self._stop.is_set():
                block = self.source.read(timeout=0.1)
                if block is None:
                    if self.source.finished:
                        self.auto_stopping.emit("模擬音檔已播完")
                        break
                    continue
                wav.write(block)
                peak = float(np.max(np.abs(block))) if len(block) else 0.0
                clipped = clipped or peak >= 0.99
                live.push(resampler.resample_chunk(block))
                now = time.monotonic()
                if now - last_level >= _LEVEL_EVERY_S:
                    rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
                    self.level.emit(
                        20 * np.log10(max(peak, 1e-6)),
                        20 * np.log10(max(rms, 1e-6)),
                        clipped,
                    )
                    clipped = False
                    last_level = now
                if now - last_update >= _UPDATE_EVERY_S:
                    update = self._snapshot(live)
                    self.updated.emit(update)
                    last_update = now
                    if self.auto_stop:
                        if update.status.last_track_done:
                            done_at = done_at or now
                            if now - done_at >= _AUTO_STOP_AFTER_S:
                                self.auto_stopping.emit("最後一首已播完")
                                break
                        if live.duration_s > max_duration:
                            self.auto_stopping.emit("錄音已超過節目長度")
                            break
                if now - last_events >= _EVENTS_EVERY_S:
                    self.provisional.emit(live.provisional_events())
                    last_events = time.monotonic()
        self.source.stop()
        live.push(resampler.resample_chunk(np.zeros(0, np.float32), last=True))
        self.updated.emit(self._snapshot(live))
        return live

    def _snapshot(self, live: LiveAnalyzer) -> LiveUpdate:
        points = live.points
        rec = np.array([p.rec_s for p in points], np.float64)
        program = np.array(
            [p.program_s if p.status == "matched" else np.nan for p in points],
            np.float64,
        )
        env = live.features().band_env_db
        rate = self.program.config.envelope_rate_hz
        factor = max(1, int(rate // _LIVE_LEVEL_RATE_HZ))
        usable = (len(env) // factor) * factor
        level = env[:usable].reshape(-1, factor).max(axis=1)
        t = (np.arange(len(level)) + 0.5) * factor / rate
        return LiveUpdate(live.status(), rec, program, t, level, self.source.overflows)


class CdRipWorker(QThread):
    """Copies audio CD tracks (.cda) to WAV files so the disc is not needed later."""

    progress = Signal(int, float)  # index in the list, fraction of that track
    ripped = Signal(list)  # list[tuple[str, str]]: (title, wav path)
    failed = Signal(str)

    def __init__(self, cda_paths: list[str], out_dir: Path) -> None:
        super().__init__()
        self.cda_paths = cda_paths
        self.out_dir = out_dir
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        tracks: list[tuple[str, str]] = []
        try:
            for index, path in enumerate(self.cda_paths):
                number = parse_cda(path).track_number
                target = rip_track(
                    path,
                    self.out_dir / f"Track{number:02d}.wav",
                    progress=lambda f, i=index: self.progress.emit(i, f),
                    cancelled=self._cancel.is_set,
                )
                tracks.append((f"第 {number} 軌", str(target)))
        except CdaCancelled:
            return
        except Exception as exc:  # shown to the user, never crash the thread
            self.failed.emit(str(exc))
            return
        self.ripped.emit(tracks)


class FileAnalysisWorker(QThread):
    progress = Signal(float)
    completed = Signal(object, str)  # AnalysisResult, path
    failed = Signal(str)

    def __init__(self, program: ReferenceProgram, path: Path) -> None:
        super().__init__()
        self.program = program
        self.path = path

    def run(self) -> None:
        try:
            self.progress.emit(0.02)
            signal = load_mono(self.path, self.program.config.sample_rate)
            result = analyze_signal(signal, self.program, progress=self.progress.emit)
        except Exception as exc:
            self.failed.emit(f"無法分析 {self.path.name}：{exc}")
            return
        self.completed.emit(result, str(self.path))
