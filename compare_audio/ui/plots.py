"""pyqtgraph views sharing the recording-time axis.

- alignment: recording time -> reference position (a straight line when all is well)
- waveform: recording above the axis, aligned reference mirrored below it
- level: recording level vs the level expected from the reference
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from compare_audio.core.analysis_models import (
    AlignmentSegment,
    AnalysisResult,
    DetectedEvent,
    Severity,
)
from compare_audio.core.reference_program import ReferenceTrack
from compare_audio.core.time_format import format_clock
from compare_audio.ui import theme

_MAX_COLUMNS = 2500
_MIN_EVENT_WIDTH_S = 0.15


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        decimals = 0 if spacing >= 1 else 1 if spacing >= 0.1 else 2
        return [format_clock(v, decimals) if v >= 0 else "" for v in values]


class ProgramAxis(pg.AxisItem):
    """Left axis of the alignment plot: track starts, then time inside the track."""

    _STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tracks: list[ReferenceTrack] = []

    def tickValues(self, minVal, maxVal, size):  # noqa: N802,N803 (pyqtgraph API)
        if not self.tracks:
            return super().tickValues(minVal, maxVal, size)
        span = max(maxVal - minVal, 1e-6)
        starts = [t.start_s for t in self.tracks if minVal <= t.start_s <= maxVal]
        wanted = max(2, int(size / 40))
        step = next((s for s in self._STEPS if span / s <= wanted), self._STEPS[-1])
        minor: list[float] = []
        for track in self.tracks:
            if track.end_s < minVal or track.start_s > maxVal:
                continue
            k = max(1, int(np.ceil((minVal - track.start_s) / step)))
            while track.start_s + k * step < min(track.end_s - step / 3, maxVal):
                minor.append(track.start_s + k * step)
                k += 1
        return [(span, starts), (step, minor)]

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        labels = []
        for value in values:
            starting = next(
                (t for t in self.tracks if abs(value - t.start_s) < 1e-6), None
            )
            if starting is not None:
                labels.append(f"第{starting.index + 1}首")
                continue
            track = next(
                (t for t in reversed(self.tracks) if value >= t.start_s - 1e-6), None
            )
            labels.append(
                "" if track is None else format_clock(value - track.start_s, 0)
            )
        return labels


class EnvelopeCurve(pg.PlotDataItem):
    """Amplitude envelope drawn as vertical bars, re-binned to the visible range."""

    def __init__(self, sign: float, color: str) -> None:
        super().__init__(pen=pg.mkPen(color, width=1), connect="pairs")
        self.rate_hz = 200.0
        self._sign = sign
        self._amp = np.zeros(0, np.float32)

    def set_envelope(self, amplitude: np.ndarray, rate_hz: float) -> None:
        self.rate_hz = rate_hz
        self._amp = np.nan_to_num(np.asarray(amplitude, np.float32), nan=0.0)
        self.refresh(None)

    def refresh(self, x_range: tuple[float, float] | None) -> None:
        n = len(self._amp)
        if n == 0:
            self.setData([], [])
            return
        if x_range is None:
            lo, hi = 0, n
        else:
            lo = max(0, int(x_range[0] * self.rate_hz) - 1)
            hi = min(n, int(x_range[1] * self.rate_hz) + 2)
        if hi <= lo:
            self.setData([], [])
            return
        span = self._amp[lo:hi]
        factor = max(1, int(np.ceil(len(span) / _MAX_COLUMNS)))
        usable = (len(span) // factor) * factor
        columns = span[:usable].reshape(-1, factor).max(axis=1) if usable else span
        factor = factor if usable else 1
        x = (lo + (np.arange(len(columns)) + 0.5) * factor) / self.rate_hz
        ys = np.zeros(2 * len(columns), np.float32)
        ys[1::2] = self._sign * columns
        self.setData(np.repeat(x, 2), ys)


def _event_brush(event: DetectedEvent) -> QColor:
    color = QColor(theme.SEVERITY_COLOR[event.severity])
    color.setAlpha(60 if event.severity == Severity.INFO else 90)
    return color


class TimelinePlots(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None, include_wave: bool = True) -> None:
        super().__init__(parent)
        self.setBackground(theme.SURFACE)
        self._program_axis = ProgramAxis(orientation="left")
        self.align = self.addPlot(
            row=0,
            col=0,
            axisItems={"bottom": TimeAxis("bottom"), "left": self._program_axis},
        )
        self.wave = pg.PlotItem(axisItems={"bottom": TimeAxis("bottom")})
        row = 1
        if include_wave:
            self.addItem(self.wave, row=row, col=0)
            row += 1
        self.level = self.addPlot(
            row=row, col=0, axisItems={"bottom": TimeAxis("bottom")}
        )
        layout = self.ci.layout
        for index, stretch in enumerate((5, 3, 2) if include_wave else (5, 2)):
            layout.setRowStretchFactor(index, stretch)
        for plot in (self.wave, self.level):
            plot.setXLink(self.align)
        for plot in (self.align, self.wave, self.level):
            plot.showGrid(x=True, y=True, alpha=0.15)
            axis = plot.getAxis("left")
            axis.setWidth(72)
            axis.enableAutoSIPrefix(False)
            plot.setMouseEnabled(x=True, y=False)
        self.align.setMouseEnabled(x=True, y=True)
        self.align.setLabel("left", "參考位置")
        self.wave.setLabel("left", "波形")
        self.wave.getAxis("left").setTicks([[]])
        self.level.setLabel("left", "dB")
        self.level.setLabel("bottom", "錄音時間")

        self._dots = pg.ScatterPlotItem(
            size=3, pen=None, brush=pg.mkBrush(120, 120, 120, 90)
        )
        self.align.addItem(self._dots)
        self._line = self.align.plot(
            [], [], pen=pg.mkPen(theme.TEXT, width=2), connect="finite"
        )
        self._segments: list[AlignmentSegment] = []
        self._rec_env = EnvelopeCurve(1.0, theme.RECORDING)
        self._ref_env = EnvelopeCurve(-1.0, theme.REFERENCE)
        wave_legend = self.wave.addLegend(offset=(8, 2), labelTextSize="9pt")
        self.wave.addItem(self._rec_env)
        self.wave.addItem(self._ref_env)
        wave_legend.addItem(self._rec_env, "上：錄音")
        wave_legend.addItem(self._ref_env, "下：參考（已對齊）")
        level_legend = self.level.addLegend(offset=(8, 2), labelTextSize="9pt")
        self._rec_level = self.level.plot(
            [], [], pen=pg.mkPen(theme.RECORDING, width=1), name="錄音音量"
        )
        self._expected = self.level.plot(
            [], [], pen=pg.mkPen(theme.REFERENCE, width=1), name="依參考預期"
        )
        for legend in (wave_legend, level_legend):
            legend.setBrush(pg.mkBrush(255, 255, 255, 215))
            legend.setColumnCount(2)
        for item in (self._rec_level, self._expected):
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
        self._decorations: list[tuple[pg.PlotItem, pg.GraphicsObject]] = []
        self._highlight: list[tuple[pg.PlotItem, pg.LinearRegionItem]] = []
        self.align.sigXRangeChanged.connect(self._on_range)
        self._duration = 0.0

    # ------------------------------------------------------------------ data
    def clear_all(self) -> None:
        for plot, item in self._decorations:
            plot.removeItem(item)
        self._decorations = []
        self.clear_highlight()
        self._segments = []
        self._dots.setData([], [])
        self._line.setData([], [])
        self._rec_env.set_envelope(np.zeros(0), 200.0)
        self._ref_env.set_envelope(np.zeros(0), 200.0)
        self._rec_level.setData([], [])
        self._expected.setData([], [])

    def _decorate(self, plot: pg.PlotItem, item: pg.GraphicsObject) -> None:
        plot.addItem(item)
        self._decorations.append((plot, item))

    def set_tracks(self, tracks: list[ReferenceTrack]) -> None:
        self._program_axis.tracks = tracks
        edges = [t.start_s for t in tracks] + ([tracks[-1].end_s] if tracks else [])
        for edge in edges:
            line = pg.InfiniteLine(pos=edge, angle=0, pen=pg.mkPen("#c9ccd2", width=1))
            self._decorate(self.align, line)

    def show_result(self, result: AnalysisResult) -> None:
        self.clear_all()
        self.set_tracks(result.program_tracks)
        # Refined segments put each break exactly where the playback jumped; the
        # raw window matches (plotted at window centres) stay as faint dots.
        trace = result.alignment
        self._dots.setData(trace.rec_s[trace.matched], trace.program_s[trace.matched])
        self._segments = list(result.segments)
        xs: list[float] = []
        ys: list[float] = []
        for segment in self._segments:
            xs += [segment.rec_start_s, segment.rec_end_s, np.nan]
            ys += [
                segment.program_at(segment.rec_start_s),
                segment.program_at(segment.rec_end_s),
                np.nan,
            ]
        self._line.setData(np.array(xs), np.array(ys))
        env = result.envelopes
        rec_amp = np.maximum(np.abs(env.rec_min), np.abs(env.rec_max))
        ref_amp = np.maximum(np.abs(env.ref_min), np.abs(env.ref_max))
        self._rec_env.set_envelope(rec_amp, env.rate_hz)
        self._ref_env.set_envelope(ref_amp, env.rate_hz)
        t = np.arange(len(env.rec_level_db)) / env.rate_hz
        self._rec_level.setData(t, env.rec_level_db)
        self._expected.setData(t, env.expected_level_db, connect="finite")
        peak = float(np.nanpercentile(rec_amp, 99.5)) if len(rec_amp) else 1.0
        self.wave.setYRange(-peak * 1.1, peak * 1.1, padding=0)
        if len(env.rec_level_db):
            top = float(np.nanmax(env.rec_level_db)) + 3
            self.level.setYRange(result.summary.noise_floor_db - 5, top, padding=0)
        for event in result.events:
            start = event.rec_start_s
            end = max(event.rec_end_s, start + _MIN_EVENT_WIDTH_S)
            for plot in (self.align, self.wave, self.level):
                region = pg.LinearRegionItem(
                    (start, end),
                    movable=False,
                    brush=_event_brush(event),
                    pen=pg.mkPen(None),
                )
                region.setZValue(-10)
                self._decorate(plot, region)
        self._duration = result.summary.recording_duration_s
        self.reset_view()

    def set_live(self, rec_s: np.ndarray, program_s: np.ndarray, span_s: float) -> None:
        self._line.setData(rec_s, program_s)
        self._duration = span_s

    def set_live_level(self, t: np.ndarray, level_db: np.ndarray) -> None:
        self._rec_level.setData(t, level_db)

    # ------------------------------------------------------------------ view
    def _on_range(self, _, x_range) -> None:
        self._rec_env.refresh(tuple(x_range))
        self._ref_env.refresh(tuple(x_range))

    def reset_view(self) -> None:
        self.align.enableAutoRange(axis="y")
        self.align.setXRange(0, max(self._duration, 1.0), padding=0.01)

    def focus(self, start_s: float, end_s: float, margin_s: float = 4.0) -> None:
        lo, hi = start_s - margin_s, end_s + margin_s
        self.align.setXRange(lo, hi, padding=0)
        values = [
            segment.program_at(t)
            for segment in self._segments
            if segment.rec_end_s >= lo and segment.rec_start_s <= hi
            for t in (max(lo, segment.rec_start_s), min(hi, segment.rec_end_s))
        ]
        if values:
            self.align.setYRange(min(values) - 2, max(values) + 2, padding=0)

    def clear_highlight(self) -> None:
        for plot, region in self._highlight:
            plot.removeItem(region)
        self._highlight = []

    def highlight(self, start_s: float, end_s: float) -> None:
        self.clear_highlight()
        end_s = max(end_s, start_s + _MIN_EVENT_WIDTH_S)
        for plot in (self.align, self.wave, self.level):
            region = pg.LinearRegionItem(
                (start_s, end_s),
                movable=False,
                brush=QColor(31, 111, 209, 50),
                pen=pg.mkPen(theme.ACCENT, width=1, style=Qt.PenStyle.DashLine),
            )
            region.setZValue(-5)
            plot.addItem(region)
            self._highlight.append((plot, region))
