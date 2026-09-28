"""Per-user folders for profiles, feature cache and recordings."""

from pathlib import Path

from PySide6.QtCore import QStandardPaths


def _location(kind: QStandardPaths.StandardLocation) -> Path:
    path = Path(QStandardPaths.writableLocation(kind))
    path.mkdir(parents=True, exist_ok=True)
    return path


def profiles_dir() -> Path:
    path = _location(QStandardPaths.StandardLocation.AppDataLocation) / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = _location(QStandardPaths.StandardLocation.CacheLocation) / "reference"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_recordings_dir() -> Path:
    base = _location(QStandardPaths.StandardLocation.DocumentsLocation)
    path = base / "CompareAudio" / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path
