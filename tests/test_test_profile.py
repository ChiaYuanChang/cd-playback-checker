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
