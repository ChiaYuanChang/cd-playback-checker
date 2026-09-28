import numpy as np

from compare_audio.core.window_matcher import ProgramMatcher, ncc_scores, refine_peak


def test_ncc_is_gain_and_colouration_invariant():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 5, (400, 8)).astype(np.float32)
    query = reference[120:170] + np.arange(8) * 3.0 - 20.0  # per-band offsets (dB)
    from compare_audio.core.window_matcher import centered_window_energy

    scores = ncc_scores(query, reference, centered_window_energy(reference, 50))
    assert int(np.argmax(scores)) == 120
    assert scores[120] > 0.999


def test_refine_peak_interpolates():
    scores = np.array([0.0, 0.5, 1.0, 0.9, 0.1])
    assert 2.0 < refine_peak(scores, 2) < 2.5


def test_global_search_finds_position():
    rng = np.random.default_rng(1)
    program = rng.normal(0, 5, (5000, 12)).astype(np.float32)
    matcher = ProgramMatcher(program, window=150, coarse_factor=4)
    found = matcher.global_search(program[3210:3360] + 1.0, n_candidates=3)
    assert abs(found[0].lag - 3210) < 1.0
    assert found[0].score > 0.95
