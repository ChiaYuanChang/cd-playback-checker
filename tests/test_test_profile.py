import pytest

from compare_audio.core.reference_program import TrackSource
from compare_audio.profiles.test_profile import (
    ProfileError,
    TestProfile,
    list_profiles,
    load_profile,
    save_profile,
)


def test_roundtrip_keeps_relative_paths(tmp_path):
    (tmp_path / "audio").mkdir()
    track = tmp_path / "audio" / "01.flac"
    track.write_bytes(b"")
    profile = TestProfile(
        name="測試片", tracks=[TrackSource(title="一", path=str(track))]
    )
    path = save_profile(profile, tmp_path / "disc.profile.json")

    assert '"audio/01.flac"' in path.read_text(encoding="utf-8")
    loaded = load_profile(path)
    assert loaded.name == "測試片"
    assert loaded.tracks[0].path == str(track.resolve())
    assert list_profiles(tmp_path) == [path]


def test_broken_profile_raises(tmp_path):
    bad = tmp_path / "bad.profile.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(bad)


def test_sensitivity_roundtrip_and_old_profile_default(tmp_path):
    from compare_audio.core.analysis_config import AnalysisConfig

    profile = TestProfile(
        name="靈敏度",
        tracks=[TrackSource(title="一", path="01.wav")],
        analysis=AnalysisConfig(sensitivity=80),
    )
    path = save_profile(profile, tmp_path / "sensitive.profile.json")
    assert load_profile(path).analysis.sensitivity == 80
    old = tmp_path / "old.profile.json"
    old.write_text('{"name":"old","tracks":[{"title":"T1","path":"01.wav"}]}')
    assert load_profile(old).analysis.sensitivity == 50
