"""Audio CD tracks on Windows: ``.cda`` files and raw CD-DA sector reads.

A ``.cda`` file (``Track01.cda``) contains no audio. It is a 44-byte pointer that
Windows shows for every track of an audio CD: track number, disc serial number,
first sector and length. The audio itself stays on the disc and is read here
sector by sector with ``IOCTL_CDROM_RAW_READ`` (16-bit stereo 44.1 kHz PCM).

The Windows calls go through ctypes, so nothing extra has to be installed. Other
platforms can parse ``.cda`` files but cannot read the disc through them (on
macOS an audio CD simply shows up with AIFF tracks, which open like any file).
"""

import ctypes
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

CD_SAMPLE_RATE = 44100
SECTOR_BYTES = 2352  # one CD-DA sector = 588 stereo 16-bit frames (1/75 s)
FRAMES_PER_SECTOR = SECTOR_BYTES // 4
_CDA_SIZE = 44
_SECTORS_PER_READ = 20  # stays below the usual 64 KB transfer limit
_RETRIES = 3
_RIP_CHUNK_FRAMES = CD_SAMPLE_RATE * 2

_IOCTL_CDROM_RAW_READ = 0x0002403E
_TRACK_MODE_CDDA = 2
_COOKED_SECTOR_BYTES = 2048  # RAW_READ_INFO.DiskOffset is LBA * 2048, even for audio
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ_WRITE = 0x1 | 0x2
_OPEN_EXISTING = 3
_DRIVE_CDROM = 5


class CdaError(Exception):
    """A CD track could not be used; the message is shown to the user."""


class CdaCancelled(CdaError):
    """The user stopped reading the disc."""


@dataclass(frozen=True)
class CdaTrack:
    track_number: int
    disc_serial: int
    start_lba: int  # first sector of the track (LBA 0 = 00:02:00 on the disc)
    length_sectors: int

    @property
    def frames(self) -> int:
        return self.length_sectors * FRAMES_PER_SECTOR

    @property
    def duration_s(self) -> float:
        return self.frames / CD_SAMPLE_RATE


def is_cda(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".cda"


def parse_cda(path: str | Path) -> CdaTrack:
    """Read the RIFF/CDDA header of a ``.cda`` file."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise CdaError(
            f"無法讀取 {Path(path).name}：請確認光碟還在光碟機裡（{exc}）"
        ) from exc
    if len(data) < _CDA_SIZE or data[0:4] != b"RIFF" or data[8:12] != b"CDDA":
        raise CdaError(f"{Path(path).name} 不是有效的 CD 音軌（.cda）檔")
    _, track, serial, start, length = struct.unpack_from("<HHIII", data, 20)
    if length == 0:
        raise CdaError(f"{Path(path).name} 的音軌長度是 0")
    return CdaTrack(track, serial, start, length)


class SectorSource(Protocol):
    def read_sectors(self, lba: int, count: int) -> bytes: ...

    def close(self) -> None: ...


class CdaTrackReader:
    """Reads one track like a sound file: 44.1 kHz stereo, seekable."""

    samplerate = CD_SAMPLE_RATE
    channels = 2

    def __init__(self, track: CdaTrack, source: SectorSource) -> None:
        self.track = track
        self.frames = track.frames
        self._source = source
        self._position = 0

    def seek(self, frame: int) -> None:
        self._position = max(0, min(int(frame), self.frames))

    def read_pcm16(self, n: int) -> np.ndarray:
        """Next ``n`` frames (fewer at the end) as int16, shape (frames, 2)."""
        n = min(int(n), self.frames - self._position)
        if n <= 0:
            return np.zeros((0, 2), np.int16)
        first = self._position // FRAMES_PER_SECTOR
        last = (self._position + n - 1) // FRAMES_PER_SECTOR
        pieces: list[bytes] = []
        for sector in range(first, last + 1, _SECTORS_PER_READ):
            count = min(_SECTORS_PER_READ, last + 1 - sector)
            pieces.append(
                self._source.read_sectors(self.track.start_lba + sector, count)
            )
        pcm = np.frombuffer(b"".join(pieces), dtype="<i2").reshape(-1, 2)
        offset = self._position - first * FRAMES_PER_SECTOR
        out = pcm[offset : offset + n]
        self._position += len(out)
        return out

    def read(self, n: int) -> np.ndarray:
        return self.read_pcm16(n).astype(np.float32) / 32768.0

    def close(self) -> None:
        self._source.close()


class _RawReadInfo(ctypes.Structure):
    _fields_ = [
        ("DiskOffset", ctypes.c_int64),
        ("SectorCount", ctypes.c_uint32),
        ("TrackMode", ctypes.c_int32),
    ]


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, pointer = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        dword,
        dword,
        pointer,
        dword,
        dword,
        handle,
    ]
    kernel32.CreateFileW.restype = handle
    kernel32.DeviceIoControl.argtypes = [
        handle,
        dword,
        pointer,
        dword,
        pointer,
        dword,
        ctypes.POINTER(dword),
        pointer,
    ]
    kernel32.DeviceIoControl.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [handle]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.GetLogicalDrives.argtypes = []
    kernel32.GetLogicalDrives.restype = dword
    kernel32.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetDriveTypeW.restype = ctypes.c_uint
    kernel32.GetVolumeInformationW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        dword,
        ctypes.POINTER(dword),
        ctypes.POINTER(dword),
        ctypes.POINTER(dword),
        ctypes.c_wchar_p,
        dword,
    ]
    kernel32.GetVolumeInformationW.restype = ctypes.c_int
    return kernel32


class WindowsCdDrive:
    """Raw audio sectors from a CD drive (``\\\\.\\D:``)."""

    def __init__(self, letter: str) -> None:
        self.letter = letter.upper()
        self._kernel32 = _kernel32()
        handle = self._kernel32.CreateFileW(
            f"\\\\.\\{self.letter}:",
            _GENERIC_READ,
            _FILE_SHARE_READ_WRITE,
            None,
            _OPEN_EXISTING,
            0,
            None,
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            code = ctypes.get_last_error()
            raise CdaError(f"無法開啟光碟機 {self.letter}:（錯誤碼 {code}）")
        self._handle = handle

    def read_sectors(self, lba: int, count: int) -> bytes:
        size = count * SECTOR_BYTES
        info = _RawReadInfo(lba * _COOKED_SECTOR_BYTES, count, _TRACK_MODE_CDDA)
        buffer = ctypes.create_string_buffer(size)
        returned = ctypes.c_uint32(0)
        code = 0
        for _ in range(_RETRIES):
            ok = self._kernel32.DeviceIoControl(
                self._handle,
                _IOCTL_CDROM_RAW_READ,
                ctypes.byref(info),
                ctypes.sizeof(info),
                buffer,
                size,
                ctypes.byref(returned),
                None,
            )
            if ok and returned.value == size:
                return buffer.raw
            code = ctypes.get_last_error()
        if count > 1:  # find the one bad sector, report exactly where it is
            return b"".join(self.read_sectors(lba + i, 1) for i in range(count))
        position = (lba * FRAMES_PER_SECTOR) / CD_SAMPLE_RATE
        raise CdaError(
            f"光碟讀取失敗（光碟位置約 {int(position // 60)}:{position % 60:04.1f}，"
            f"錯誤碼 {code}）：光碟可能有刮傷或髒污"
        )

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _cd_drives(kernel32) -> list[str]:
    mask = kernel32.GetLogicalDrives()
    letters = [chr(ord("A") + i) for i in range(26) if mask & (1 << i)]
    return [d for d in letters if kernel32.GetDriveTypeW(f"{d}:\\") == _DRIVE_CDROM]


def _volume_serial(kernel32, letter: str) -> int | None:
    serial = ctypes.c_uint32(0)
    ok = kernel32.GetVolumeInformationW(
        f"{letter}:\\", None, 0, ctypes.byref(serial), None, None, None, 0
    )
    return serial.value if ok else None


def _open_drive_for(path: str | Path, track: CdaTrack) -> SectorSource:
    """The drive holding the disc this ``.cda`` file describes."""
    if sys.platform != "win32":
        raise CdaError(
            "CDA 檔只是 Windows 為光碟音軌建立的捷徑，聲音還在光碟上，"
            "需要在 Windows 上、光碟放在光碟機裡才能讀取。"
            "（macOS 請直接選光碟裡的 AIFF 音軌）"
        )
    kernel32 = _kernel32()
    drives = _cd_drives(kernel32)
    if not drives:
        raise CdaError("這台電腦找不到光碟機")
    own = Path(path).drive[:1].upper()
    if own in drives:  # picked straight from the disc: Windows made it for this disc
        return WindowsCdDrive(own)
    for letter in drives:  # a copied .cda file: find the disc by its serial number
        if _volume_serial(kernel32, letter) == track.disc_serial:
            return WindowsCdDrive(letter)
    raise CdaError(
        "找不到放著這張光碟的光碟機。請把光碟放進光碟機，"
        "再直接從光碟機裡選 Track01.cda 等檔案。"
    )


def open_cda_track(path: str | Path) -> CdaTrackReader:
    track = parse_cda(path)
    return CdaTrackReader(track, _open_drive_for(path, track))


def rip_track(
    path: str | Path,
    target: str | Path,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Copy one CD track to a 16-bit WAV (bit for bit, as read from the disc)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".part.wav")
    reader = open_cda_track(path)
    try:
        with sf.SoundFile(
            str(partial), "w", samplerate=CD_SAMPLE_RATE, channels=2, subtype="PCM_16"
        ) as out:
            done = 0
            while True:
                if cancelled is not None and cancelled():
                    raise CdaCancelled("已取消讀取光碟")
                pcm = reader.read_pcm16(_RIP_CHUNK_FRAMES)
                if len(pcm) == 0:
                    break
                out.write(pcm)
                done += len(pcm)
                if progress is not None:
                    progress(done / max(reader.frames, 1))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        reader.close()
    partial.replace(target)
    return target
