"""Run the analysis on a generated test set and score it against the truth.

uv run python -m tools.evaluate_detection --data sample_data [--plots]
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_audio.core.analysis_models import (
    AnalysisResult,
    DetectedEvent,
    EventType,
    Severity,
)
from compare_audio.core.recording_analysis import analyze_file
from compare_audio.core.reference_program import load_reference_program
from compare_audio.profiles.test_profile import load_profile
from tools.defect_injection import DefectKind, TruthEvent

COMPATIBLE: dict[DefectKind, set[EventType]] = {
    DefectKind.SKIP: {EventType.SKIP},
    DefectKind.REPEAT: {EventType.REPEAT, EventType.STUCK},
    DefectKind.STUCK: {EventType.STUCK, EventType.REPEAT},
    DefectKind.DROPOUT: {EventType.DROPOUT},
    DefectKind.PAUSE: {EventType.PAUSE},
    DefectKind.STOP: {EventType.STOPPED},
    DefectKind.TRUNCATE: {EventType.SKIP, EventType.TRACK_INCOMPLETE},
    DefectKind.MISSING: {EventType.TRACK_MISSING, EventType.SKIP},
}
TIME_TOLERANCE_S = 1.0


def truth_label(event: TruthEvent) -> str:
    if event.kind == DefectKind.STUCK:
        return f"stuck {event.size_s:g}s x{event.count}"
    if event.kind in (DefectKind.MISSING, DefectKind.TRUNCATE, DefectKind.STOP):
        return event.kind.value
    return f"{event.kind.value} {event.size_s:g}s"


Pairing = tuple[TruthEvent, DetectedEvent | None, bool]


def match_events(
    truth: list[TruthEvent], detected: list[DetectedEvent]
) -> tuple[list[Pairing], list[DetectedEvent]]:
    """Pair truth with detections: (truth, detection, correct type) + leftovers."""
    unused = [e for e in detected if e.severity != Severity.INFO]
    pairs: list[tuple[TruthEvent, DetectedEvent | None, bool]] = []
    for event in truth:
        tolerance = (
            2.0
            if event.kind in (DefectKind.TRUNCATE, DefectKind.MISSING)
            else TIME_TOLERANCE_S
        )

        def near(d: DetectedEvent, event=event, tolerance=tolerance) -> bool:
            if event.kind == DefectKind.MISSING:
                return (
                    d.track_index == event.track
                    or abs(d.rec_start_s - event.rec_s) <= tolerance
                )
            return d.rec_start_s - tolerance <= event.rec_s <= d.rec_end_s + tolerance

        candidates = [d for d in unused if near(d)]
        correct = [d for d in candidates if d.event_type in COMPATIBLE[event.kind]]
        chosen = min(
            correct or candidates,
            key=lambda d: abs(d.rec_start_s - event.rec_s),
            default=None,
        )
        if chosen is not None:
            unused.remove(chosen)
        pairs.append((event, chosen, bool(correct)))
    return pairs, unused


def plot_case(
    path: Path, result: AnalysisResult, truth: list[TruthEvent], title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    trace = result.alignment
    ax.plot(
        trace.rec_s[trace.matched],
        trace.program_s[trace.matched],
        ".",
        ms=2,
        color="0.3",
    )
    for track in result.program_tracks:
        ax.axhline(track.start_s, color="0.85", lw=0.8)
    for event in truth:
        ax.axvline(event.rec_s, color="tab:blue", lw=0.8, ls="--")
        ax.text(
            event.rec_s,
            ax.get_ylim()[1] if False else 0,
            truth_label(event),
            rotation=90,
            fontsize=7,
            color="tab:blue",
            va="bottom",
        )
    colors = {
        Severity.FAIL: "tab:red",
        Severity.WARN: "tab:orange",
        Severity.INFO: "tab:green",
    }
    for event in result.events:
        ax.axvspan(
            event.rec_start_s,
            max(event.rec_end_s, event.rec_start_s + 0.3),
            color=colors[event.severity],
            alpha=0.3,
        )
    ax.set_ylabel("program position (s)")
    ax.set_title(title)
    env = result.envelopes
    t = np.arange(len(env.rec_level_db)) / env.rate_hz
    ax2.plot(t, env.rec_level_db, lw=0.4, color="0.4", label="recording")
    ax2.plot(t, env.expected_level_db, lw=0.4, color="tab:purple", label="expected")
    ax2.set_ylabel("level (dB)")
    ax2.set_xlabel("recording time (s)")
    ax2.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("sample_data"))
    parser.add_argument("--out", type=Path, default=Path("eval_output"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--only", default="", help="substring filter on case names")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="override an AnalysisConfig field, e.g. --set match_threshold=0.4",
    )
    args = parser.parse_args()

    profile = load_profile(next(args.data.glob("*.profile.json")))
    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.set)}
    config = profile.analysis.model_copy(update=overrides)
    program = load_reference_program(
        profile.tracks, config, cache_dir=args.data / "cache"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    per_kind: dict[str, list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )  # truth, correct, flagged
    per_condition: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    time_errors: list[float] = []
    size_errors: list[float] = []
    lines: list[str] = []
    for truth_path in sorted((args.data / "recordings").glob("*.truth.json")):
        name = truth_path.name.removesuffix(".truth.json")
        if args.only and args.only not in name:
            continue
        meta = json.loads(truth_path.read_text(encoding="utf-8"))
        truth = [TruthEvent.model_validate(e) for e in meta["truth"]]
        started = time.time()
        result = analyze_file(truth_path.with_name(f"{name}.wav"), program)
        elapsed = time.time() - started
        pairs, leftovers = match_events(truth, result.events)
        condition = meta["condition"]["name"]
        stats = per_condition[condition]
        lines.append(
            f"\n## {name}  ({result.summary.verdict.value}, {elapsed:.1f}s, "
            f"SNR~{result.summary.signal_to_noise_db} dB, "
            f"drift {result.summary.drift_ppm} ppm)"
        )
        for event, detection, correct in pairs:
            key = event.kind.value
            per_kind[key][0] += 1
            stats["truth"] += 1
            if detection is not None and correct:
                per_kind[key][1] += 1
                stats["correct"] += 1
                time_errors.append(abs(detection.rec_start_s - event.rec_s))
                if (
                    event.kind in (DefectKind.SKIP, DefectKind.REPEAT)
                    and detection.jump_s is not None
                ):
                    size_errors.append(abs(abs(detection.jump_s) - event.size_s))
                mark = "OK "
            elif detection is not None:
                per_kind[key][2] += 1
                stats["flagged"] += 1
                mark = "~~ "
            else:
                stats["missed"] += 1
                mark = "-- "
            found = (
                f"{detection.event_type.value} {detection.detail}"
                if detection
                else "(missed)"
            )
            lines.append(
                f"  {mark} {truth_label(event):22s} @ {event.rec_s:7.2f}s -> {found}"
            )
        for event in leftovers:
            stats["false_" + event.severity.value] += 1
            lines.append(
                f"  !! extra {event.severity.value}: {event.event_type.value} "
                f"@ {event.rec_start_s:.2f}s {event.detail}"
            )
        if args.verbose:
            for event in result.events:
                if event.severity == Severity.INFO:
                    lines.append(f"     info @ {event.rec_start_s:.2f}s {event.detail}")
        if args.plots:
            plot_case(args.out / f"{name}.png", result, truth, name)

    report = [
        "# Detection evaluation",
        "",
        "| kind | truth | correct | flagged other | recall |",
        "|---|---|---|---|---|",
    ]
    for kind, (n, ok, flagged) in sorted(per_kind.items()):
        report.append(f"| {kind} | {n} | {ok} | {flagged} | {ok / n:.0%} |")
    report += [
        "",
        "| condition | truth | correct | flagged | missed | extra FAIL | extra WARN |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, stats in per_condition.items():
        report.append(
            f"| {condition} | {stats['truth']} | {stats['correct']} | "
            f"{stats['flagged']} | {stats['missed']} | {stats['false_fail']} | "
            f"{stats['false_warn']} |"
        )
    if time_errors:
        report.append(
            f"\nmedian time error {np.median(time_errors) * 1000:.0f} ms, "
            f"p90 {np.percentile(time_errors, 90) * 1000:.0f} ms"
        )
    if size_errors:
        report.append(f"median jump size error {np.median(size_errors) * 1000:.0f} ms")
    text = "\n".join(report + lines)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
