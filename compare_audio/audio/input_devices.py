"""Microphone listing on top of PortAudio (sounddevice)."""

import sys
from dataclasses import dataclass

import sounddevice as sd

# One API per platform, otherwise every microphone shows up three or four times
# (MME, DirectSound, WASAPI, WDM-KS on Windows).
_PREFERRED_HOST_API = {"win32": "Windows WASAPI", "darwin": "Core Audio"}


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    channels: int
    default_rate: float
    is_default: bool

    def label(self, with_api: bool = False) -> str:
        text = self.name + ("（預設）" if self.is_default else "")
        return f"{text}  [{self.host_api}]" if with_api else text


def list_input_devices(all_apis: bool = False) -> list[InputDevice]:
    try:
        devices = sd.query_devices()
        apis = sd.query_hostapis()
        default_input = sd.default.device[0]
    except sd.PortAudioError:
        return []
    preferred = _PREFERRED_HOST_API.get(sys.platform)
    found: list[InputDevice] = []
    for index, info in enumerate(devices):
        if info["max_input_channels"] <= 0:
            continue
        api = apis[info["hostapi"]]["name"]
        found.append(
            InputDevice(
                index=index,
                name=str(info["name"]),
                host_api=str(api),
                channels=int(info["max_input_channels"]),
                default_rate=float(info["default_samplerate"]),
                is_default=index == default_input
                or index == apis[info["hostapi"]].get("default_input_device"),
            )
        )
    if all_apis or preferred is None:
        return found
    filtered = [d for d in found if d.host_api == preferred]
    return filtered or found


def find_device(devices: list[InputDevice], name: str) -> InputDevice | None:
    return next((d for d in devices if d.name == name), None)
