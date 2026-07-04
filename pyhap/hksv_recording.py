"""TLV codecs for classic HomeKit Secure Video recording configuration.

The Camera Recording Management service advertises the container, video and
audio configurations the accessory supports and receives the controller's
selection. These structures predate the 17.99 preview and use the standard
HAP TLV8 layout (repeated items separated by a zero-length type-0 TLV).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import struct
from typing import List

from pyhap import tlv

_SEPARATOR = b"\x00\x00"


class EventTrigger(IntEnum):
    MOTION = 0x01
    DOORBELL = 0x02


class MediaContainerType(IntEnum):
    FRAGMENTED_MP4 = 0x00


class VideoCodecType(IntEnum):
    H264 = 0x00
    H265 = 0x01


class AudioCodecType(IntEnum):
    AAC_LC = 0x00
    AAC_ELD = 0x01


class AudioSampleRate(IntEnum):
    KHZ_8 = 0x00
    KHZ_16 = 0x01
    KHZ_24 = 0x02
    KHZ_32 = 0x03
    KHZ_44_1 = 0x04
    KHZ_48 = 0x05


class BitRateMode(IntEnum):
    VARIABLE = 0x00
    CONSTANT = 0x01


def _u8(value: int) -> bytes:
    return struct.pack("<B", value)


def _u16(value: int) -> bytes:
    return struct.pack("<H", value)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _int(data: bytes) -> int:
    return int.from_bytes(data, "little")


def _join(items: List[bytes]) -> bytes:
    return _SEPARATOR.join(items)


def _iter(data: bytes):
    offset = 0
    while offset < len(data):
        tag = data[offset]
        length = data[offset + 1]
        value = data[offset + 2 : offset + 2 + length]
        offset += 2 + length
        while length == 255 and offset < len(data) and data[offset] == tag:
            length = data[offset + 1]
            value += data[offset + 2 : offset + 2 + length]
            offset += 2 + length
        yield tag, value


def _decode(data: bytes) -> dict:
    out: dict = {}
    for tag, value in _iter(data):
        out.setdefault(tag, value)
    return out


def _split(data: bytes) -> list:
    items = []
    current = b""
    for tag, value in _iter(data):
        if tag == 0 and not value:
            if current:
                items.append(current)
            current = b""
            continue
        current += tlv.encode(_u8(tag), value)
    if current:
        items.append(current)
    return items


@dataclass
class MediaContainerConfiguration:
    fragment_length_ms: int = 4000
    container_type: MediaContainerType = MediaContainerType.FRAGMENTED_MP4

    def encode(self) -> bytes:
        params = tlv.encode(b"\x01", _u32(self.fragment_length_ms))
        return tlv.encode(b"\x01", _u8(self.container_type), b"\x02", params)

    @classmethod
    def decode(cls, data: bytes) -> "MediaContainerConfiguration":
        d = _decode(data)
        params = _decode(d[2])
        return cls(
            container_type=MediaContainerType(_int(d[1])),
            fragment_length_ms=_int(params[1]),
        )


@dataclass
class SupportedRecordingConfiguration:
    prebuffer_length_ms: int = 4000
    event_triggers: int = EventTrigger.MOTION
    media_containers: List[MediaContainerConfiguration] = field(
        default_factory=lambda: [MediaContainerConfiguration()]
    )

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u32(self.prebuffer_length_ms),
            b"\x02", struct.pack("<Q", self.event_triggers),
            b"\x03", _join([c.encode() for c in self.media_containers]),
        )

    @classmethod
    def decode(cls, data: bytes) -> "SupportedRecordingConfiguration":
        d = _decode(data)
        return cls(
            prebuffer_length_ms=_int(d[1]),
            event_triggers=_int(d[2]),
            media_containers=[
                MediaContainerConfiguration.decode(i) for i in _split(d[3])
            ],
        )


@dataclass
class VideoAttributes:
    width: int
    height: int
    frame_rate: int

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u16(self.width),
            b"\x02", _u16(self.height),
            b"\x03", _u8(self.frame_rate),
        )

    @classmethod
    def decode(cls, data: bytes) -> "VideoAttributes":
        d = _decode(data)
        return cls(width=_int(d[1]), height=_int(d[2]), frame_rate=_int(d[3]))


@dataclass
class VideoCodecConfiguration:
    codec_type: VideoCodecType
    profile: int
    level: int
    bitrate_kbps: int
    iframe_interval_ms: int
    attributes: List[VideoAttributes]

    def encode(self) -> bytes:
        params = tlv.encode(
            b"\x01", _u8(self.profile),
            b"\x02", _u8(self.level),
            b"\x03", _u32(self.bitrate_kbps),
            b"\x04", _u32(self.iframe_interval_ms),
        )
        return tlv.encode(
            b"\x01", _u8(self.codec_type),
            b"\x02", params,
            b"\x03", _join([a.encode() for a in self.attributes]),
        )

    @classmethod
    def decode(cls, data: bytes) -> "VideoCodecConfiguration":
        d = _decode(data)
        params = _decode(d[2])
        return cls(
            codec_type=VideoCodecType(_int(d[1])),
            profile=_int(params[1]),
            level=_int(params[2]),
            bitrate_kbps=_int(params[3]),
            iframe_interval_ms=_int(params[4]),
            attributes=[VideoAttributes.decode(i) for i in _split(d[3])],
        )


def encode_supported_video(configs: List[VideoCodecConfiguration]) -> bytes:
    return tlv.encode(b"\x01", _join([c.encode() for c in configs]))


def decode_supported_video(data: bytes) -> List[VideoCodecConfiguration]:
    return [VideoCodecConfiguration.decode(i) for i in _split(_decode(data)[1])]


@dataclass
class AudioCodecConfiguration:
    codec_type: AudioCodecType = AudioCodecType.AAC_LC
    channels: int = 1
    bitrate_mode: BitRateMode = BitRateMode.VARIABLE
    sample_rate: AudioSampleRate = AudioSampleRate.KHZ_32
    max_audio_bitrate_kbps: int = 64

    def encode(self) -> bytes:
        params = tlv.encode(
            b"\x01", _u8(self.channels),
            b"\x02", _u8(self.bitrate_mode),
            b"\x03", _u8(self.sample_rate),
            b"\x04", _u32(self.max_audio_bitrate_kbps),
        )
        return tlv.encode(b"\x01", _u8(self.codec_type), b"\x02", params)

    @classmethod
    def decode(cls, data: bytes) -> "AudioCodecConfiguration":
        d = _decode(data)
        params = _decode(d[2])
        return cls(
            codec_type=AudioCodecType(_int(d[1])),
            channels=_int(params[1]),
            bitrate_mode=BitRateMode(_int(params[2])),
            sample_rate=AudioSampleRate(_int(params[3])),
            max_audio_bitrate_kbps=_int(params[4]),
        )


def encode_supported_audio(configs: List[AudioCodecConfiguration]) -> bytes:
    return tlv.encode(b"\x01", _join([c.encode() for c in configs]))


def decode_supported_audio(data: bytes) -> List[AudioCodecConfiguration]:
    return [AudioCodecConfiguration.decode(i) for i in _split(_decode(data)[1])]


@dataclass
class SelectedRecordingConfiguration:
    """The controller's chosen recording configuration."""

    recording: SupportedRecordingConfiguration
    video: VideoCodecConfiguration
    audio: AudioCodecConfiguration

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", self.recording.encode(),
            b"\x02", self.video.encode(),
            b"\x03", self.audio.encode(),
        )

    @classmethod
    def decode(cls, data: bytes) -> "SelectedRecordingConfiguration":
        d = _decode(data)
        return cls(
            recording=SupportedRecordingConfiguration.decode(d[1]),
            video=VideoCodecConfiguration.decode(d[2]),
            audio=AudioCodecConfiguration.decode(d[3]),
        )
