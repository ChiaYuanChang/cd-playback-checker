"""Generate synthetic reference tracks, a test profile and faulty recordings.

    uv run python -m tools.make_test_set --out sample_data

The output folder can be opened directly in the app: the profile lists the
reference tracks and ``recordings/`` holds simulated microphone captures with a
``.truth.json`` next to each describing the injected faults.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_audio.core.reference_program import TrackSource
from compare_audio.profiles.test_profile import TestProfile, save_profile
from tools.acoustic_sim import MIC_RATE, RoomCondition, simulate_recording
from tools.defect_injection import Defect, DefectKind, PlaybackPlan, render_playback
from tools.synth_music import SAMPLE_RATE, render_song

TRACK_STYLES = ("pop", "ballad", "electronic", "pop", "electronic", "ballad")

CONDITIONS = (
    RoomCondition(
        name="near-quiet",
        snr_db=25,
        drr_db=12,
        rt60_s=0.35,
        drift_ppm=60,
        noise_kind="pink",
        pre_roll_s=5.0,
    ),
    RoomCondition(
        name="mid-office",
        snr_db=15,
        drr_db=5,
        rt60_s=0.5,
        drift_ppm=-90,
        noise_kind="mixed",
        pre_roll_s=9.0,
    ),
    RoomCondition(
        name="near-talking",
        snr_db=14,
        drr_db=10,
        rt60_s=0.4,
        drift_ppm=-40,
        noise_kind="babble",
        pre_roll_s=4.0,
    ),
    RoomCondition(
        name="far-noisy",
        snr_db=8,
        drr_db=0,
        rt60_s=0.7,
        drift_ppm=120,
        noise_kind="babble",
        pre_roll_s=14.0,
    ),
)

_SIZES = {
    DefectKind.SKIP: (0.12, 0.25, 0.5, 1.0, 3.0, 8.0),
    DefectKind.REPEAT: (0.12, 0.25, 0.5, 1.0, 3.0),
    DefectKind.STUCK: (0.1, 0.15, 0.25, 0.4),
    DefectKind.DROPOUT: (0.03, 0.06, 0.1, 0.3, 1.0),
    DefectKind.PAUSE: (0.5, 1.0, 2.5),
}
_KIND_WEIGHTS = {
    DefectKind.SKIP: 3,
    DefectKind.REPEAT: 3,
    DefectKind.STUCK: 2,
    DefectKind.DROPOUT: 3,
    DefectKind.PAUSE: 2,
}


def random_plan(
    durations: list[float], rng: np.random.Generator, faulty: bool
) -> PlaybackPlan:
    gaps = [0.0] + [float(rng.choice([0.0, 0.5, 2.0, 2.0])) for _ in durations[1:]]
    if not faulty:
        return PlaybackPlan(gaps_s=gaps)
    kinds = list(_KIND_WEIGHTS)
    weights = np.array([_KIND_WEIGHTS[k] for k in kinds], float)
    defects: list[Defect] = []
    for track, duration in enumerate(durations):
        n = int(rng.choice([0, 1, 1, 2]))
        positions: list[float] = []
        for _ in range(n * 4):
            if len(positions) == n:
                break
            position = float(rng.uniform(6.0, duration - 9.0))
            if all(abs(position - other) > 9.0 for other in positions):
                positions.append(position)
        for position in sorted(positions):
            kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
            size = float(rng.choice(_SIZES[kind]))
            count = int(rng.choice([3, 6, 12])) if kind == DefectKind.STUCK else 1
            defects.append(
                Defect(
                    kind=kind,
                    track=track,
                    position_s=round(position, 2),
                    size_s=size,
                    count=count,
                )
            )
    if rng.random() < 0.25:
        track = int(rng.integers(0, len(durations) - 1))
        at = durations[track] * float(rng.uniform(0.6, 0.8))
        defects = [
            d for d in defects if not (d.track == track and d.position_s > at - 9)
        ]
        defects.append(
            Defect(kind=DefectKind.TRUNCATE, track=track, position_s=round(at, 2))
        )
    if rng.random() < 0.15:
        track = int(rng.integers(1, len(durations)))
        defects = [d for d in defects if d.track != track]
        defects.append(Defect(kind=DefectKind.MISSING, track=track))
    if rng.random() < 0.2:
        track = len(durations) - 1
        at = durations[track] * float(rng.uniform(0.3, 0.7))
        defects = [
            d for d in defects if not (d.track == track and d.position_s > at - 9)
        ]
        defects.append(
            Defect(kind=DefectKind.STOP, track=track, position_s=round(at, 2))
        )
    return PlaybackPlan(gaps_s=gaps, defects=defects)


def make_reference(
    out: Path, n_tracks: int, seconds: float, song_seed: int = 100
) -> list[Path]:
    ref_dir = out / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(n_tracks):
        style = TRACK_STYLES[index % len(TRACK_STYLES)]
        path = ref_dir / f"{index + 1:02d}_{style}.flac"
        if not path.exists():
            audio = render_song(seed=song_seed + index, duration_s=seconds, style=style)
            sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("sample_data"))
    parser.add_argument("--tracks", type=int, default=6)
    parser.add_argument("--track-seconds", type=float, default=60.0)
    parser.add_argument("--faulty-per-condition", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--song-seed", type=int, default=100)
    args = parser.parse_args()

    paths = make_reference(args.out, args.tracks, args.track_seconds, args.song_seed)
    profile = TestProfile(
        name="合成測試片",
        description="程序化產生的 6 首測試音樂（tools/make_test_set.py）",
        tracks=[TrackSource(title=p.stem, path=str(p)) for p in paths],
    )
    save_profile(profile, args.out / "synthetic_disc.profile.json")
    tracks = [sf.read(p, dtype="float32", always_2d=True)[0] for p in paths]
    durations = [len(t) / SAMPLE_RATE for t in tracks]

    rec_dir = args.out / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for cond in CONDITIONS:
        for case in range(args.faulty_per_condition + 1):
            faulty = case > 0
            name = f"{cond.name}_{'faults' + str(case) if faulty else 'clean'}"
            plan = random_plan(durations, rng, faulty)
            played, truth = render_playback(tracks, plan, SAMPLE_RATE)
            recording = simulate_recording(played, SAMPLE_RATE, cond, truth, rng)
            sf.write(rec_dir / f"{name}.wav", recording, MIC_RATE, subtype="PCM_16")
            meta = {
                "condition": cond.model_dump(),
                "plan": plan.model_dump(mode="json"),
                "truth": [event.model_dump(mode="json") for event in truth],
            }
            (rec_dir / f"{name}.truth.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{name}: {len(recording) / MIC_RATE:6.1f} s, {len(truth)} faults")


if __name__ == "__main__":
    main()
