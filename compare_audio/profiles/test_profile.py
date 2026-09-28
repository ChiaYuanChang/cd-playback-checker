"""A test profile names one test disc: its tracks in play order and its settings.

Profiles are JSON files. Track paths are stored relative to the profile file when the
audio sits next to it, so a folder with a profile and its tracks can be copied to
another PC as is.
"""

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.reference_program import TrackSource

PROFILE_SUFFIX = ".profile.json"


class TestProfile(BaseModel):
    __test__ = False  # not a pytest test class

    name: str = Field(
        ..., min_length=1, description="Display name, e.g. the disc title."
    )
    tracks: list[TrackSource] = Field(
        ..., min_length=1, description="Tracks in play order."
    )
    description: str = Field(default="", description="Free notes about the disc.")
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig, description="Detection settings for this disc."
    )


class ProfileError(Exception):
    """A profile file could not be read."""


def load_profile(path: str | Path) -> TestProfile:
    path = Path(path)
    try:
        profile = TestProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ProfileError(f"無法讀取測試片設定 {path.name}：{exc}") from exc
    base = path.parent
    resolved = [
        track.model_copy(update={"path": str((base / track.path).resolve())})
        if not Path(track.path).is_absolute()
        else track
        for track in profile.tracks
    ]
    return profile.model_copy(update={"tracks": resolved})


def save_profile(profile: TestProfile, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()
    stored = []
    for track in profile.tracks:
        track_path = Path(track.path).resolve()
        try:
            relative = track_path.relative_to(base)
            stored.append(track.model_copy(update={"path": relative.as_posix()}))
        except ValueError:
            stored.append(track.model_copy(update={"path": str(track_path)}))
    data = profile.model_copy(update={"tracks": stored}).model_dump(
        mode="json", exclude_defaults=False
    )
    data["analysis"] = profile.analysis.model_dump(mode="json", exclude_defaults=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def list_profiles(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.rglob(f"*{PROFILE_SUFFIX}"), key=lambda p: p.name.lower())


def profile_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name).strip()
    return f"{safe or 'profile'}{PROFILE_SUFFIX}"
