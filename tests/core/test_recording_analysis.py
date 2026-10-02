"""End-to-end checks of the analysis on short synthetic programs."""

import numpy as np
import pytest

from compare_audio.core.analysis_models import EventType, Severity, Verdict
from compare_audio.core.live_analysis import LiveAnalyzer
from compare_audio.core.recording_analysis import analyze_signal

from .conftest import RATE, room


def _fails(result):
    return [e for e in result.events if e.severity == Severity.FAIL]


def _played(tracks, edit):
    """Concatenate both tracks after applying ``edit`` to the first one."""
    return np.concatenate([edit(tracks[0]), np.zeros(RATE, np.float32), tracks[1]])


def test_clean_playback_passes_with_unknown_start(program, tracks_audio):
    recording = room(_played(tracks_audio, lambda t: t), pre_roll_s=4.3)
    result = analyze_signal(recording, program)
    assert result.summary.verdict == Verdict.PASS, result.events
    assert result.summary.playback_start_s == pytest.approx(4.3, abs=0.6)
    gaps = [e for e in result.events if e.event_type == EventType.TRACK_GAP]
    assert len(gaps) == 1
    assert all(track.heard_s > 25 for track in result.tracks)


@pytest.mark.parametrize(
    ("edit", "expected", "jump"),
    [
        (
            lambda t: np.concatenate([t[: 12 * RATE], t[int(12.5 * RATE) :]]),
            EventType.SKIP,
            0.5,
        ),
        (
            lambda t: np.concatenate([t[: 12 * RATE], t[int(11.0 * RATE) :]]),
            EventType.REPEAT,
            -1.0,
        ),
    ],
)
def test_jumps_are_found_and_measured(program, tracks_audio, edit, expected, jump):
    recording = room(_played(tracks_audio, edit), pre_roll_s=3.0)
    result = analyze_signal(recording, program)
    fails = _fails(result)
    assert [e.event_type for e in fails] == [expected], result.events
    assert fails[0].jump_s == pytest.approx(jump, abs=0.03)
    assert fails[0].rec_start_s == pytest.approx(15.0, abs=0.15)
    assert fails[0].track_index == 0


def test_dropout_is_found(program, tracks_audio):
    def mute(track):
        out = track.copy()
        out[int(10 * RATE) : int(10.2 * RATE)] = 0
        return out

    result = analyze_signal(room(_played(tracks_audio, mute), 2.0), program)
    fails = _fails(result)
    assert [e.event_type for e in fails] == [EventType.DROPOUT], result.events
    assert fails[0].rec_start_s == pytest.approx(12.0, abs=0.1)


def test_stuck_loop_is_found(program, tracks_audio):
    def loop(track):
        at = int(14 * RATE)
        fragment = track[at - int(0.2 * RATE) : at]
        return np.concatenate([track[:at], *[fragment] * 8, track[at:]])

    result = analyze_signal(room(_played(tracks_audio, loop), 2.0), program)
    fails = _fails(result)
    assert fails and fails[0].event_type in (EventType.STUCK, EventType.REPEAT)
    assert fails[0].rec_start_s == pytest.approx(16.0, abs=0.3)


def test_missing_track_and_stop(program, tracks_audio):
    recording = room(
        np.concatenate([tracks_audio[0][: 20 * RATE], np.zeros(6 * RATE, np.float32)]),
        2.0,
    )
    result = analyze_signal(recording, program)
    types = {e.event_type for e in _fails(result)}
    assert EventType.STOPPED in types
    assert result.summary.verdict == Verdict.FAIL


def test_live_analyzer_matches_offline(program, tracks_audio):
    recording = room(_played(tracks_audio, lambda t: t), pre_roll_s=2.0)
    live = LiveAnalyzer(program)
    for start in range(0, len(recording), 4000):
        live.push(recording[start : start + 4000])
    status = live.status()
    assert status.locked and status.track_index == 1
    final = live.finish()
    offline = analyze_signal(recording, program)
    assert final.summary.verdict == offline.summary.verdict == Verdict.PASS
    assert len(final.segments) == len(offline.segments)


def test_repeat_does_not_inflate_played_coverage(program, tracks_audio):
    repeated = np.concatenate([tracks_audio[0], tracks_audio[0], tracks_audio[1]])
    result = analyze_signal(room(repeated, 2), program)
    assert any(e.event_type == EventType.REPEAT for e in result.events)
    assert result.summary.heard_ratio <= 1
    assert all(track.heard_s <= track.duration_s for track in result.tracks)
