"""Procedural music for testing: drums, bass, chords, melody, repeated sections.

The goal is not beauty but the properties that matter to the aligner: onsets,
sustained harmonic notes, quiet passages, fades, and choruses that repeat (the
electronic style even repeats whole bars exactly, the hardest case for matching).
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_PROGRESSIONS = ((0, 4, 5, 3), (5, 3, 0, 4), (0, 5, 3, 4), (0, 3, 4, 3), (5, 4, 3, 4))
STYLES = ("pop", "ballad", "electronic")


@dataclass(frozen=True)
class Section:
    kind: str
    first_bar: int
    n_bars: int
    energy: float


def _midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


class _Mixer:
    def __init__(self, n_samples: int, rng: np.random.Generator) -> None:
        self.left = np.zeros(n_samples, np.float64)
        self.right = np.zeros(n_samples, np.float64)
        self.rng = rng

    def add(
        self, at_s: float, sound: np.ndarray, gain: float, pan: float = 0.0
    ) -> None:
        start = int(round((at_s + self.rng.normal(0, 0.004)) * SAMPLE_RATE))
        start = max(0, start)
        end = min(len(self.left), start + len(sound))
        if end <= start:
            return
        piece = sound[: end - start] * gain
        self.left[start:end] += piece * np.sqrt(0.5 * (1 - pan))
        self.right[start:end] += piece * np.sqrt(0.5 * (1 + pan))


def _envelope(
    n: int, attack_s: float, decay_s: float, sustain: float, release_s: float
):
    t = np.arange(n) / SAMPLE_RATE
    env = np.minimum(1.0, t / max(attack_s, 1e-4))
    env *= sustain + (1 - sustain) * np.exp(
        -np.maximum(0, t - attack_s) / max(decay_s, 1e-4)
    )
    release = int(release_s * SAMPLE_RATE)
    if release > 0 and n > release:
        env[-release:] *= np.linspace(1, 0, release) ** 2
    return env


def _additive(
    freq: float,
    dur_s: float,
    amps: list[float],
    vibrato: float = 0.0,
    decay_per_harmonic: float = 0.0,
    rng: np.random.Generator | None = None,
):
    n = max(1, int(dur_s * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    wobble = vibrato * np.sin(2 * np.pi * 5.2 * t) if vibrato else 0.0
    phase0 = 2 * np.pi * t * freq * (1 + wobble)
    out = np.zeros(n)
    for h, amp in enumerate(amps, start=1):
        if freq * h > SAMPLE_RATE / 2.2:
            break
        partial = amp * np.sin(h * phase0 + (rng.uniform(0, 6.28) if rng else 0.0))
        if decay_per_harmonic:
            partial *= np.exp(-t * decay_per_harmonic * h)
        out += partial
    return out


def _kick() -> np.ndarray:
    n = int(0.35 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    freq = 45 + 75 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * np.exp(-t / 0.11)
    click = np.zeros(n)
    click[:80] = np.hanning(160)[80:] * 0.6
    return body + click


def _noise_hit(rng, dur_s: float, band: tuple[float, float], tau: float) -> np.ndarray:
    n = int(dur_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    sos = butter(2, band, btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * np.exp(-t / tau)


def _snare(rng) -> np.ndarray:
    noise = _noise_hit(rng, 0.25, (900, 7000), 0.06) * 1.6
    t = np.arange(len(noise)) / SAMPLE_RATE
    return noise + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.045)


def _plan_sections(n_bars: int, style: str, rng: np.random.Generator) -> list[Section]:
    template = [
        ("intro", 4, 0.35),
        ("verse", 8, 0.6),
        ("chorus", 8, 0.95),
        ("verse", 8, 0.62),
        ("chorus", 8, 1.0),
        ("bridge", 8, 0.45),
        ("chorus", 8, 1.0),
        ("chorus", 8, 1.0),
        ("outro", 4, 0.5),
    ]
    if style == "ballad":
        template = [
            ("intro", 2, 0.3),
            ("verse", 8, 0.45),
            ("chorus", 8, 0.75),
            ("verse", 8, 0.5),
            ("chorus", 8, 0.8),
            ("bridge", 4, 0.3),
            ("chorus", 8, 0.85),
            ("outro", 4, 0.35),
        ]
    sections: list[Section] = []
    bar = 0
    while bar < n_bars:
        for kind, length, energy in template:
            if bar >= n_bars:
                break
            length = min(length, n_bars - bar)
            sections.append(Section(kind, bar, length, energy * rng.uniform(0.9, 1.05)))
            bar += length
        template = template[1:-1] or template  # repeat the middle when the song is long
    return sections


def _phrase(
    rng: np.random.Generator, beats: float, busy: float
) -> list[tuple[float, float, int]]:
    """(start_beat, length_beats, scale step relative to the chord) for one phrase."""
    notes: list[tuple[float, float, int]] = []
    position, step = 0.0, int(rng.integers(0, 5))
    choices = (
        [0.5, 0.5, 1.0, 1.0, 1.5, 2.0] if busy > 0.5 else [1.0, 1.5, 2.0, 2.0, 3.0]
    )
    while position < beats - 0.25:
        length = float(min(rng.choice(choices), beats - position))
        if rng.random() < 0.15:
            position += length  # rest
            continue
        step = int(np.clip(step + rng.integers(-2, 3), -2, 9))
        notes.append((position, length, step))
        position += length
    return notes


def render_song(seed: int, duration_s: float, style: str | None = None) -> np.ndarray:
    """Stereo float32 song of about ``duration_s`` seconds (starts and ends quietly)."""
    rng = np.random.default_rng(seed)
    style = style or str(rng.choice(STYLES))
    tempo = {
        "pop": rng.uniform(96, 128),
        "ballad": rng.uniform(64, 78),
        "electronic": rng.uniform(118, 130),
    }[style]
    beat = 60.0 / tempo
    bar = 4 * beat
    lead_in, fade_out = 0.4, min(6.0, duration_s * 0.1)
    n_bars = max(4, int((duration_s - lead_in) / bar))
    total_s = lead_in + n_bars * bar + 2.5
    mixer = _Mixer(int(total_s * SAMPLE_RATE), rng)

    root = int(rng.integers(45, 55))
    scale = _MINOR if rng.random() < 0.4 else _MAJOR
    progression = _PROGRESSIONS[int(rng.integers(len(_PROGRESSIONS)))]
    chorus_progression = _PROGRESSIONS[int(rng.integers(len(_PROGRESSIONS)))]
    sections = _plan_sections(n_bars, style, rng)
    phrases = {
        kind: _phrase(rng, 8.0, 0.7 if style != "ballad" else 0.3)
        for kind in ("verse", "chorus", "bridge", "intro", "outro")
    }
    lead_amps = [1.0, 0.55, 0.35, 0.22, 0.12, 0.08]
    pad_amps = [1.0 / h**1.6 for h in range(1, 11)]
    bass_amps = [1.0 / h for h in range(1, 9)]
    kick, arp_cache = _kick(), {}

    def degree_note(degree: int, octave_base: int) -> int:
        octave, index = divmod(degree, len(scale))
        return octave_base + scale[index] + 12 * octave

    for section in sections:
        chords = chorus_progression if section.kind == "chorus" else progression
        motif = phrases.get(section.kind, phrases["verse"])
        for bar_index in range(section.n_bars):
            absolute_bar = section.first_bar + bar_index
            bar_start = lead_in + absolute_bar * bar
            degree = chords[bar_index % len(chords)]
            energy = section.energy
            chord_notes = [degree_note(degree + k, root + 12) for k in (0, 2, 4)]

            # pad / chords
            if style != "electronic" or section.kind != "intro":
                for note in chord_notes:
                    tone = _additive(_midi_hz(note), bar + 0.6, pad_amps, rng=rng)
                    tone *= _envelope(
                        len(tone),
                        0.25 if style != "ballad" else 0.02,
                        2.5 if style == "ballad" else 1.0,
                        0.2 if style == "ballad" else 0.7,
                        0.5,
                    )
                    mixer.add(bar_start, tone, 0.07 * energy, rng.uniform(-0.5, 0.5))

            # bass
            if section.kind not in ("intro",) and not (
                style == "ballad" and energy < 0.5
            ):
                bass_note = degree_note(degree, root - 12)
                pattern = (0, 2) if style == "ballad" else (0, 1.5, 2, 3, 3.5)
                for beat_pos in pattern:
                    length = 0.9 * beat if style != "ballad" else 1.8 * beat
                    tone = _additive(_midi_hz(bass_note), length, bass_amps, rng=rng)
                    tone *= _envelope(len(tone), 0.005, 0.25, 0.4, 0.05)
                    mixer.add(bar_start + beat_pos * beat, tone, 0.22 * energy)

            # drums
            drums_on = style != "ballad" or section.kind == "chorus"
            if (
                drums_on
                and section.kind not in ("intro",)
                and not (section.kind == "bridge" and style == "pop" and bar_index % 2)
            ):
                kick_beats = (
                    (0, 1, 2, 3)
                    if style == "electronic"
                    else (0, 2.5)
                    if bar_index % 2
                    else (0, 2)
                )
                for beat_pos in kick_beats:
                    mixer.add(bar_start + beat_pos * beat, kick, 0.55 * energy)
                for beat_pos in (1, 3):
                    mixer.add(
                        bar_start + beat_pos * beat, _snare(rng), 0.28 * energy, 0.1
                    )
                hat_step = 0.25 if section.kind == "chorus" else 0.5
                for k in range(int(4 / hat_step)):
                    hat = _noise_hit(rng, 0.05, (6500, 16000), 0.012)
                    velocity = 0.7 + 0.3 * rng.random()
                    mixer.add(
                        bar_start + k * hat_step * beat,
                        hat,
                        0.16 * energy * velocity,
                        -0.3,
                    )
                if bar_index == section.n_bars - 1 and section.kind != "outro":
                    for k in range(4):  # fill
                        mixer.add(
                            bar_start + (3 + k * 0.25) * beat, _snare(rng), 0.2 * energy
                        )

            # electronic arpeggio: identical every bar of a section (exact loop)
            if style == "electronic" and section.kind in (
                "intro",
                "verse",
                "chorus",
                "bridge",
            ):
                key = (degree, section.kind)
                if key not in arp_cache:
                    arp = np.zeros(int((bar + 0.3) * SAMPLE_RATE))
                    for k in range(16):
                        note = chord_notes[k % 3] + (12 if k % 4 == 3 else 0)
                        pluck = _additive(
                            _midi_hz(note), 0.18, [1, 0.6, 0.4, 0.3, 0.2], rng=rng
                        )
                        pluck *= _envelope(len(pluck), 0.002, 0.06, 0.1, 0.03)
                        at = int(k * beat / 4 * SAMPLE_RATE)
                        arp[at : at + len(pluck)] += pluck[: len(arp) - at]
                    arp_cache[key] = arp
                start = int(bar_start * SAMPLE_RATE)
                piece = arp_cache[key][: len(mixer.left) - start] * 0.09 * energy
                mixer.left[start : start + len(piece)] += piece * 0.8
                mixer.right[start : start + len(piece)] += piece * 0.6

            # lead melody (motif repeats every two bars, varied in the second half)
            if section.kind != "intro" or style == "ballad":
                half = bar_index % 2
                for start_beat, length, step in motif:
                    if not (half * 4 <= start_beat < half * 4 + 4):
                        continue
                    if (
                        bar_index >= 4
                        and section.kind == "verse"
                        and rng.random() < 0.25
                    ):
                        step += int(rng.integers(-1, 2))
                    note = degree_note(degree + step, root + 24)
                    dur = length * beat * 0.92
                    if style == "ballad":
                        tone = _additive(
                            _midi_hz(note),
                            dur + 0.8,
                            lead_amps,
                            decay_per_harmonic=1.2,
                            rng=rng,
                        )
                        tone *= _envelope(len(tone), 0.004, 0.9, 0.0, 0.3)
                    else:
                        tone = _additive(
                            _midi_hz(note), dur, lead_amps, vibrato=0.003, rng=rng
                        )
                        tone *= _envelope(len(tone), 0.012, 0.12, 0.65, 0.06)
                    mixer.add(
                        bar_start + (start_beat - half * 4) * beat,
                        tone,
                        0.12 * (0.6 + 0.4 * energy),
                        0.15,
                    )

    mix = np.stack([mixer.left, mixer.right], axis=1)
    peak = np.max(np.abs(mix)) + 1e-9
    mix = np.tanh(1.3 * mix / peak) / np.tanh(1.3)
    mix *= 0.89
    end = int(min(total_s, lead_in + n_bars * bar + 1.5) * SAMPLE_RATE)
    mix = mix[:end]
    fade = int(fade_out * SAMPLE_RATE)
    mix[-fade:] *= (np.linspace(1, 0, fade) ** 2)[:, None]
    mix[: int(0.02 * SAMPLE_RATE)] *= np.linspace(0, 1, int(0.02 * SAMPLE_RATE))[
        :, None
    ]
    return mix.astype(np.float32)
