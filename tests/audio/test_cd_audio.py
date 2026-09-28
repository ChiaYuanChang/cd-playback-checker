"""CD track support, with a fake disc standing in for the Windows drive."""

import struct
import sys

import numpy as np
import pytest
import soundfile as sf

from compare_audio.audio import cd_audio
from compare_audio.audio.cd_audio import (
    CD_SAMPLE_RATE,
    FRAMES_PER_SECTOR,
    SECTOR_BYTES,
    CdaCancelled,
    CdaError,
    CdaTrackReader,
    parse_cda,
    rip_track,
)
from compare_audio.core.analysis_config import AnalysisConfig
from compare_audio.core.audio_io import audio_duration_s, load_mono, read_clip
from compare_audio.core.reference_program import TrackSource, load_reference_program

START_LBA = 4500
LENGTH = 450  # sectors = 6 seconds


def _msf(lba: int) -> bytes:
    minutes, rest = divmod(lba, 75 * 60)
    seconds, frames = divmod(rest, 75)
    return bytes([frames, seconds, minutes, 0])


def write_cda(path, track=3, serial=0x1234ABCD, start=START_LBA, length=LENGTH):
    header = b"RIFF" + struct.pack("<I", 36) + b"CDDA" + b"fmt " + struct.pack("<I", 24)
    body = struct.pack("<HHIII", 1, track, serial, start, length)
    path.write_bytes(header + body + _msf(start + 150) + _msf(length))
    return path


def expected_pcm(first_frame: int, n: int) -> np.ndarray:
    """What the fake disc holds at absolute frames first_frame .. first_frame+n."""
    g = np.arange(first_frame, first_frame + n, dtype=np.int64)
    left = (g * 37) % 60000 - 30000
    return np.stack([left, -left], axis=1).astype(np.int16)


class FakeDisc:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int]] = []
        self.closed = False

    def read_sectors(self, lba: int, count: int) -> bytes:
        self.requests.append((lba, count))
        pcm = expected_pcm(lba * FRAMES_PER_SECTOR, count * FRAMES_PER_SECTOR)
        return pcm.astype("<i2").tobytes()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def cda_file(tmp_path):
    return write_cda(tmp_path / "Track03.cda")


@pytest.fixture
def fake_disc(monkeypatch):
    disc = FakeDisc()
    monkeypatch.setattr(cd_audio, "_open_drive_for", lambda path, track: disc)
    return disc


def test_parse_cda_reads_header(cda_file):
    track = parse_cda(cda_file)
    assert (track.track_number, track.disc_serial) == (3, 0x1234ABCD)
    assert (track.start_lba, track.length_sectors) == (START_LBA, LENGTH)
    assert track.duration_s == pytest.approx(
        LENGTH * FRAMES_PER_SECTOR / CD_SAMPLE_RATE
    )
    assert audio_duration_s(cda_file) == pytest.approx(6.0)


def test_parse_cda_rejects_other_files(tmp_path):
    fake = tmp_path / "Track01.cda"
    fake.write_bytes(b"RIFF" + b"\0" * 60)
    with pytest.raises(CdaError):
        parse_cda(fake)


def test_reader_returns_exact_samples_across_sectors(cda_file):
    disc = FakeDisc()
    reader = CdaTrackReader(parse_cda(cda_file), disc)
    base = START_LBA * FRAMES_PER_SECTOR
    pieces = [reader.read_pcm16(n) for n in (1, 587, 1000, 30000, 10**9)]
    got = np.concatenate(pieces)
    np.testing.assert_array_equal(got, expected_pcm(base, LENGTH * FRAMES_PER_SECTOR))
    assert len(reader.read_pcm16(10)) == 0  # end of track
    assert all(count <= 20 for _, count in disc.requests)
    assert disc.requests[0][0] == START_LBA

    reader.seek(12345)
    np.testing.assert_array_equal(
        reader.read_pcm16(100), expected_pcm(base + 12345, 100)
    )
    floats = reader.read(10)
    assert floats.dtype == np.float32 and np.abs(floats).max() <= 1.0


def test_audio_helpers_read_cda_tracks(cda_file, fake_disc):
    mono = load_mono(cda_file, 16000)
    assert len(mono) == pytest.approx(6.0 * 16000, abs=2)
    clip, rate = read_clip(cda_file, 0.5, 0.25)
    assert rate == CD_SAMPLE_RATE and clip.shape == (11025, 2)
    start = START_LBA * FRAMES_PER_SECTOR + 22050
    np.testing.assert_allclose(clip, expected_pcm(start, 11025) / 32768.0)
    assert fake_disc.closed


def test_rip_track_is_bit_exact(cda_file, fake_disc, tmp_path):
    fractions: list[float] = []
    target = rip_track(cda_file, tmp_path / "out" / "Track03.wav", fractions.append)
    data, rate = sf.read(target, dtype="int16")
    assert rate == CD_SAMPLE_RATE
    np.testing.assert_array_equal(
        data, expected_pcm(START_LBA * FRAMES_PER_SECTOR, LENGTH * FRAMES_PER_SECTOR)
    )
    assert fractions[-1] == pytest.approx(1.0)
    assert sorted(p.name for p in target.parent.iterdir()) == ["Track03.wav"]


def test_cancelled_rip_leaves_nothing(cda_file, fake_disc, tmp_path):
    with pytest.raises(CdaCancelled):
        rip_track(cda_file, tmp_path / "Track03.wav", cancelled=lambda: True)
    assert list(tmp_path.glob("*.wav")) == []


def test_reference_program_accepts_cda(cda_file, fake_disc):
    program = load_reference_program(
        [TrackSource(title="第 3 軌", path=str(cda_file))], AnalysisConfig()
    )
    assert program.tracks[0].duration_s == pytest.approx(6.0, abs=0.01)
    assert len(program.features.log_mel) > 500


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows message")
def test_explains_that_cda_needs_windows(cda_file):
    with pytest.raises(CdaError, match="Windows"):
        cd_audio.open_cda_track(cda_file)


def test_sector_size_constants():
    assert SECTOR_BYTES == FRAMES_PER_SECTOR * 4 == 2352
