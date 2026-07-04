"""TLV8 codecs for the HomeKit Secure Video Open Source Compatibility Guide.

Payload encoders/decoders for the camera services introduced by Apple's
"HomeKit Secure Video Open Source Compatibility Guide" (Developer Preview,
2026-06-03, spec version "17.99"). This module is protocol-only: it converts
between python values and the characteristic TLV8 payloads. Session state
machines and media handling live with the accessory implementation.

Repeated TLV lists are encoded as sub-TLV items joined by a zero-length TLV of
type 0 (controllers overwrite duplicate top-level types and split lists on
type 0).
"""

from dataclasses import dataclass, field
from enum import IntEnum
import struct
from typing import Optional
from uuid import UUID

from pyhap import tlv

TLV_SEPARATOR = b"\x00\x00"


class VideoCodecType(IntEnum):
    H264 = 1
    H265 = 2


class VideoQuality(IntEnum):
    HIGHEST = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class AudioCodecType(IntEnum):
    OPUS = 3


class AudioSampleRate(IntEnum):
    KHZ_16 = 1
    KHZ_24 = 2
    KHZ_32 = 3
    KHZ_48 = 4


class AudioBitDepth(IntEnum):
    BIT_8 = 1
    BIT_16 = 2
    BIT_24 = 3


class SensorType(IntEnum):
    UNKNOWN = 0
    PRIMARY = 1
    GENERIC = 255


class SensorIntent(IntEnum):
    UNKNOWN = 0
    MAIN = 1
    PACKAGE = 2
    GENERIC = 255


class BufferCommand(IntEnum):
    START = 1
    START_AND_STOP = 2
    STOP = 3


class BufferStopAction(IntEnum):
    PAUSE = 1
    FINALIZE = 2


class BufferActivity(IntEnum):
    SHOULD_RECORD = 1
    SHOULD_NOT_RECORD = 2


class BufferEventCommandType(IntEnum):
    QUERY = 1
    ACKNOWLEDGE = 2


class BufferEventType(IntEnum):
    CMAF_SESSION_START = 1
    CMAF_SESSION_STOP = 2
    MOTION = 3
    CMAF_ERROR = 4


class CMAFError(IntEnum):
    NONE = 0
    UNKNOWN = 1
    CANNOT_FIND_HOST = 2
    CERT_CONNECTION_FAILURE = 3
    CANNOT_CERTIFY = 4
    INVALID_STATE = 5
    REQUIRES_RETRY = 6
    NO_RESPONSE = 7
    MAX_SESSION_TIME_EXCEEDED = 8
    CANCELED = 9
    MP4_ERROR = 10
    CONNECTION_FAILED = 11
    TIMEOUT = 12
    OUT_OF_RESOURCES = 13
    INVALID_DATA = 14
    HTTP_BAD_REQUEST = 15
    HTTP_INVALID_TOKEN = 16
    HTTP_CAMERA_ZONE_DISABLED = 17
    HTTP_MISMATCHED_TOKEN = 18
    HTTP_NOT_FOUND = 19
    HTTP_INIT_MISSING = 20
    HTTP_UNSUPPORTED_MEDIA_TYPE = 21
    HTTP_BLOCKED = 22
    HTTP_CERTIFICATE_EXPIRED = 23
    HTTP_INTERNAL_SERVER_ERROR = 24
    HTTP_SERVICE_UNAVAILABLE = 25
    HTTP_CAMERA_ZONE_DOES_NOT_EXIST = 26


class ZoneApplicationMethod(IntEnum):
    NORMAL = 1
    INVERTED = 2


class RTPStreamingCommand(IntEnum):
    END = 1
    START = 2


class RTPStreamingStatus(IntEnum):
    SUCCESS = 0
    UNKNOWN_SESSION_IDENTIFIER = 1
    NO_SUCH_STREAM = 2
    BUSY = 3
    ERROR = 4


class WebRTCStreamingStatus(IntEnum):
    SUCCESS = 0
    UNKNOWN_SESSION_IDENTIFIER = 1
    BUSY = 2
    ERROR = 3


class WebRTCOfferStatus(IntEnum):
    SUCCESS = 0
    PRIVACY_MODE_ACTIVE = 1
    ERROR = 2


def _u8(value: int) -> bytes:
    return struct.pack("<B", value)


def _u16(value: int) -> bytes:
    return struct.pack("<H", value)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _int(data: bytes) -> int:
    return int.from_bytes(data, "little")


def _iter_tlv(data: bytes):
    """Yield (type, value) pairs from a TLV8 buffer, merging fragmented values."""
    offset = 0
    while offset < len(data):
        tag = data[offset]
        length = data[offset + 1]
        value = data[offset + 2 : offset + 2 + length]
        offset += 2 + length
        # Values longer than 255 bytes are fragmented into consecutive TLVs of
        # the same type.
        while length == 255 and offset < len(data) and data[offset] == tag:
            length = data[offset + 1]
            value += data[offset + 2 : offset + 2 + length]
            offset += 2 + length
        yield tag, value


def _decode(data: bytes) -> dict:
    """Decode a TLV8 buffer into {type: value}; repeated types collect into lists."""
    out: dict = {}
    for tag, value in _iter_tlv(data):
        if tag in out:
            existing = out[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[tag] = [existing, value]
        else:
            out[tag] = value
    return out


def _split_items(data: bytes) -> list:
    """Split a repeated-TLV list value on zero-length type-0 separators."""
    items = []
    current = b""
    for tag, value in _iter_tlv(data):
        if tag == 0 and not value:
            if current:
                items.append(current)
            current = b""
            continue
        current += tlv.encode(_u8(tag), value)
    if current:
        items.append(current)
    return items


def _join_items(items: list) -> bytes:
    return TLV_SEPARATOR.join(items)


@dataclass
class VideoStreamTier:
    identifier: int
    quality: VideoQuality
    average_bitrate_kbps: int
    width: int
    height: int
    frame_rate: int

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u32(self.identifier),
            b"\x02", _u8(self.quality),
            b"\x03", _u32(self.average_bitrate_kbps),
            b"\x04", _u16(self.width),
            b"\x05", _u16(self.height),
            b"\x06", _u8(self.frame_rate),
        )

    @classmethod
    def decode(cls, data: bytes) -> "VideoStreamTier":
        d = _decode(data)
        return cls(
            identifier=_int(d[1]),
            quality=VideoQuality(_int(d[2])),
            average_bitrate_kbps=_int(d[3]),
            width=_int(d[4]),
            height=_int(d[5]),
            frame_rate=_int(d[6]),
        )


@dataclass
class SupportedVideoStreamTiers:
    codec: VideoCodecType
    payload_type: int
    tiers: list

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u8(self.codec),
            b"\x02", _u8(self.payload_type),
            b"\x03", _join_items([t.encode() for t in self.tiers]),
        )

    @classmethod
    def decode(cls, data: bytes) -> "SupportedVideoStreamTiers":
        d = _decode(data)
        return cls(
            codec=VideoCodecType(_int(d[1])),
            payload_type=_int(d[2]),
            tiers=[VideoStreamTier.decode(i) for i in _split_items(d[3])],
        )


@dataclass
class AudioStreamTier:
    identifier: int
    target_average_bitrate: int
    sample_rate: AudioSampleRate
    bit_depth: AudioBitDepth
    packet_time_ms: int = 20
    number_of_channels: int = 1

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u32(self.identifier),
            b"\x02", _u32(self.target_average_bitrate),
            b"\x03", _u8(self.sample_rate),
            b"\x04", _u8(self.bit_depth),
            b"\x05", _u8(self.packet_time_ms),
            b"\x06", _u8(self.number_of_channels),
        )

    @classmethod
    def decode(cls, data: bytes) -> "AudioStreamTier":
        d = _decode(data)
        return cls(
            identifier=_int(d[1]),
            target_average_bitrate=_int(d[2]),
            sample_rate=AudioSampleRate(_int(d[3])),
            bit_depth=AudioBitDepth(_int(d[4])),
            packet_time_ms=_int(d[5]),
            number_of_channels=_int(d[6]),
        )


@dataclass
class SupportedAudioStreamTiers:
    codec: AudioCodecType
    payload_type: int
    tiers: list

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u8(self.codec),
            b"\x02", _u8(self.payload_type),
            b"\x03", _join_items([t.encode() for t in self.tiers]),
        )

    @classmethod
    def decode(cls, data: bytes) -> "SupportedAudioStreamTiers":
        d = _decode(data)
        return cls(
            codec=AudioCodecType(_int(d[1])),
            payload_type=_int(d[2]),
            tiers=[AudioStreamTier.decode(i) for i in _split_items(d[3])],
        )


@dataclass
class VideoStreamCapability:
    identifier: bytes
    video_quality: VideoQuality
    width: int
    height: int
    frames_per_second: int
    average_bit_rate_kbps: int
    peak_bit_rate_kbps: int

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", self.identifier,
            b"\x02", _u8(self.video_quality),
            b"\x03", _u16(self.width),
            b"\x04", _u16(self.height),
            b"\x05", _u8(self.frames_per_second),
            b"\x06", _u32(self.average_bit_rate_kbps),
            b"\x07", _u32(self.peak_bit_rate_kbps),
        )

    @classmethod
    def decode(cls, data: bytes) -> "VideoStreamCapability":
        d = _decode(data)
        return cls(
            identifier=d[1],
            video_quality=VideoQuality(_int(d[2])),
            width=_int(d[3]),
            height=_int(d[4]),
            frames_per_second=_int(d[5]),
            average_bit_rate_kbps=_int(d[6]),
            peak_bit_rate_kbps=_int(d[7]),
        )


@dataclass
class SensorConfiguration:
    width: int
    height: int
    sensor_uuid: bytes
    sensor_type: SensorType = SensorType.PRIMARY
    sensor_intent: SensorIntent = SensorIntent.MAIN
    video_stream_capabilities: list = field(default_factory=list)

    def encode(self) -> bytes:
        dimensions = tlv.encode(b"\x01", _u16(self.width), b"\x02", _u16(self.height))
        return tlv.encode(
            b"\x01", dimensions,
            b"\x02", self.sensor_uuid,
            b"\x03", _u8(self.sensor_type),
            b"\x04", _u8(self.sensor_intent),
            b"\x05", _join_items(
                [c.encode() for c in self.video_stream_capabilities]
            ),
        )

    @classmethod
    def decode(cls, data: bytes) -> "SensorConfiguration":
        d = _decode(data)
        dims = _decode(d[1])
        return cls(
            width=_int(dims[1]),
            height=_int(dims[2]),
            sensor_uuid=d[2],
            sensor_type=SensorType(_int(d[3])),
            sensor_intent=SensorIntent(_int(d[4])),
            video_stream_capabilities=[
                VideoStreamCapability.decode(i) for i in _split_items(d[5])
            ],
        )


@dataclass
class CameraCapabilities:
    sensors: list
    version: int = 1

    def encode(self) -> bytes:
        sensors = tlv.encode(
            b"\x01", _join_items([s.encode() for s in self.sensors])
        )
        return tlv.encode(b"\x01", _u8(self.version), b"\x02", sensors)

    @classmethod
    def decode(cls, data: bytes) -> "CameraCapabilities":
        d = _decode(data)
        inner = _decode(d[2])
        return cls(
            version=_int(d[1]),
            sensors=[SensorConfiguration.decode(i) for i in _split_items(inner[1])],
        )


@dataclass
class ContributingSensors:
    sensor_uuids: list

    def encode(self) -> bytes:
        items = [tlv.encode(b"\x01", uuid) for uuid in self.sensor_uuids]
        return tlv.encode(b"\x01", _join_items(items))

    @classmethod
    def decode(cls, data: bytes) -> "ContributingSensors":
        d = _decode(data)
        return cls(
            sensor_uuids=[_decode(i)[1] for i in _split_items(d[1])]
        )


@dataclass
class CameraKey:
    key: bytes
    key_number: int

    def encode(self) -> bytes:
        return tlv.encode(b"\x01", self.key, b"\x02", _u64(self.key_number))

    @classmethod
    def decode(cls, data: bytes) -> "CameraKey":
        d = _decode(data)
        return cls(key=d[1], key_number=_int(d[2]))


def encode_camera_key_id(key_id: int) -> bytes:
    return tlv.encode(b"\x01", _u64(key_id))


def decode_camera_key_id(data: bytes) -> int:
    return _int(_decode(data)[1])


@dataclass
class BufferUploadCommand:
    session_id: int
    command: BufferCommand
    start: Optional[int] = None
    stop: Optional[int] = None
    stop_action: Optional[BufferStopAction] = None

    @classmethod
    def decode(cls, data: bytes) -> "BufferUploadCommand":
        d = _decode(data)
        return cls(
            session_id=_int(d[1]),
            command=BufferCommand(_int(d[2])),
            start=_int(d[3]) if 3 in d else None,
            stop=_int(d[4]) if 4 in d else None,
            stop_action=BufferStopAction(_int(d[5])) if 5 in d else None,
        )

    def encode(self) -> bytes:
        args = [b"\x01", _u64(self.session_id), b"\x02", _u8(self.command)]
        if self.start is not None:
            args += [b"\x03", _u64(self.start)]
        if self.stop is not None:
            args += [b"\x04", _u64(self.stop)]
        if self.stop_action is not None:
            args += [b"\x05", _u8(self.stop_action)]
        return tlv.encode(*args)


def encode_buffer_upload_response(clip_id: int) -> bytes:
    return tlv.encode(b"\x01", _u64(clip_id))


def decode_buffer_upload_response(data: bytes) -> int:
    return _int(_decode(data)[1])


@dataclass
class BufferActivityCommand:
    start: int
    duration_ms: int
    activity: BufferActivity

    @classmethod
    def decode(cls, data: bytes) -> "BufferActivityCommand":
        d = _decode(data)
        return cls(
            start=_int(d[1]),
            duration_ms=_int(d[2]),
            activity=BufferActivity(_int(d[3])),
        )

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", _u64(self.start),
            b"\x02", _u64(self.duration_ms),
            b"\x03", _u8(self.activity),
        )


@dataclass
class BufferEventCommand:
    command: BufferEventCommandType
    sequence_number: int
    limit: Optional[int] = None

    @classmethod
    def decode(cls, data: bytes) -> "BufferEventCommand":
        d = _decode(data)
        return cls(
            command=BufferEventCommandType(_int(d[1])),
            sequence_number=_int(d[2]),
            limit=_int(d[3]) if 3 in d else None,
        )

    def encode(self) -> bytes:
        args = [b"\x01", _u8(self.command), b"\x02", _u64(self.sequence_number)]
        if self.limit is not None:
            args += [b"\x03", _u64(self.limit)]
        return tlv.encode(*args)


@dataclass
class BufferEvent:
    sequence_number: int
    type: BufferEventType
    cmaf_session_id: Optional[int] = None
    motion_active: Optional[bool] = None
    cmaf_error: Optional[CMAFError] = None

    def encode(self) -> bytes:
        args = [b"\x01", _u64(self.sequence_number), b"\x02", _u8(self.type)]
        if self.type is BufferEventType.CMAF_SESSION_START:
            args += [b"\x03", tlv.encode(b"\x01", _u64(self.cmaf_session_id))]
        elif self.type is BufferEventType.CMAF_SESSION_STOP:
            args += [b"\x04", tlv.encode(b"\x01", _u64(self.cmaf_session_id))]
        elif self.type is BufferEventType.MOTION:
            args += [
                b"\x05",
                tlv.encode(b"\x01", _u8(1 if self.motion_active else 0)),
            ]
        elif self.type is BufferEventType.CMAF_ERROR:
            args += [
                b"\x06",
                tlv.encode(
                    b"\x01", _u64(self.cmaf_session_id),
                    b"\x02", _u8(self.cmaf_error),
                ),
            ]
        return tlv.encode(*args)

    @classmethod
    def decode(cls, data: bytes) -> "BufferEvent":
        d = _decode(data)
        event = cls(
            sequence_number=_int(d[1]),
            type=BufferEventType(_int(d[2])),
        )
        if event.type is BufferEventType.CMAF_SESSION_START:
            event.cmaf_session_id = _int(_decode(d[3])[1])
        elif event.type is BufferEventType.CMAF_SESSION_STOP:
            event.cmaf_session_id = _int(_decode(d[4])[1])
        elif event.type is BufferEventType.MOTION:
            event.motion_active = bool(_int(_decode(d[5])[1]))
        elif event.type is BufferEventType.CMAF_ERROR:
            error = _decode(d[6])
            event.cmaf_session_id = _int(error[1])
            event.cmaf_error = CMAFError(_int(error[2]))
        return event


def encode_buffer_events_response(events: list) -> bytes:
    return tlv.encode(b"\x01", _join_items([e.encode() for e in events]))


def decode_buffer_events_response(data: bytes) -> list:
    d = _decode(data)
    return [BufferEvent.decode(i) for i in _split_items(d[1])]


@dataclass
class PublishingPoint:
    url: str
    server_ca_certificates: list

    def encode(self) -> bytes:
        certs = [tlv.encode(b"\x01", c) for c in self.server_ca_certificates]
        return tlv.encode(
            b"\x01", self.url.encode(),
            b"\x02", _join_items(certs),
        )

    @classmethod
    def decode(cls, data: bytes) -> "PublishingPoint":
        d = _decode(data)
        return cls(
            url=d[1].decode(),
            server_ca_certificates=[_decode(i)[1] for i in _split_items(d[2])],
        )


@dataclass
class ZonePolygon:
    identifier: bytes
    vertices: list

    def encode(self) -> bytes:
        packed = b"".join(_u16(x) + _u16(y) for x, y in self.vertices)
        return tlv.encode(b"\x01", self.identifier, b"\x03", packed)

    @classmethod
    def decode(cls, data: bytes) -> "ZonePolygon":
        d = _decode(data)
        raw = d[3]
        vertices = [
            (struct.unpack_from("<H", raw, i)[0], struct.unpack_from("<H", raw, i + 2)[0])
            for i in range(0, len(raw), 4)
        ]
        return cls(identifier=d[1], vertices=vertices)


@dataclass
class CameraZones:
    method: ZoneApplicationMethod
    polygons: list
    zone_data_version: int = 2

    def encode(self) -> bytes:
        zone_data = tlv.encode(
            b"\x01", _u8(self.method),
            b"\x03", _join_items([p.encode() for p in self.polygons]),
        )
        return tlv.encode(b"\x01", _u8(self.zone_data_version), b"\x02", zone_data)

    @classmethod
    def decode(cls, data: bytes) -> "CameraZones":
        d = _decode(data)
        version = _int(d[1])
        inner = _decode(d[2])
        return cls(
            zone_data_version=version,
            method=ZoneApplicationMethod(_int(inner[1])),
            polygons=[ZonePolygon.decode(i) for i in _split_items(inner[3])],
        )


@dataclass
class RTPStreamingControlWrite:
    session_identifier: bytes
    command: RTPStreamingCommand
    video_tier: Optional[int] = None
    video_ssrc: Optional[int] = None
    audio_tier: Optional[int] = None
    audio_ssrc: Optional[int] = None

    @classmethod
    def decode(cls, data: bytes) -> "RTPStreamingControlWrite":
        d = _decode(data)
        return cls(
            session_identifier=d[1],
            command=RTPStreamingCommand(_int(d[2])),
            video_tier=_int(d[3]) if 3 in d else None,
            video_ssrc=_int(d[4]) if 4 in d else None,
            audio_tier=_int(d[5]) if 5 in d else None,
            audio_ssrc=_int(d[6]) if 6 in d else None,
        )

    def encode(self) -> bytes:
        args = [b"\x01", self.session_identifier, b"\x02", _u8(self.command)]
        if self.video_tier is not None:
            args += [b"\x03", _u32(self.video_tier)]
        if self.video_ssrc is not None:
            args += [b"\x04", _u32(self.video_ssrc)]
        if self.audio_tier is not None:
            args += [b"\x05", _u32(self.audio_tier)]
        if self.audio_ssrc is not None:
            args += [b"\x06", _u32(self.audio_ssrc)]
        return tlv.encode(*args)


def encode_rtp_streaming_status(
    session_identifier: bytes, status: RTPStreamingStatus
) -> bytes:
    return tlv.encode(b"\x01", session_identifier, b"\x02", _u8(status))


@dataclass
class SFrameKeyData:
    key: bytes
    kid: int

    def encode(self) -> bytes:
        return tlv.encode(b"\x01", self.key, b"\x02", _u64(self.kid))

    @classmethod
    def decode(cls, data: bytes) -> "SFrameKeyData":
        d = _decode(data)
        return cls(key=d[1], kid=_int(d[2]))


@dataclass
class WebRTCICECandidate:
    candidate: str
    sdp_mid: Optional[str] = None
    sdp_mline_index: Optional[int] = None

    def encode(self) -> bytes:
        args = [b"\x01", self.candidate.encode()]
        if self.sdp_mid is not None:
            args += [b"\x02", self.sdp_mid.encode()]
        if self.sdp_mline_index is not None:
            args += [b"\x03", _u16(self.sdp_mline_index)]
        return tlv.encode(*args)

    @classmethod
    def decode(cls, data: bytes) -> "WebRTCICECandidate":
        d = _decode(data)
        return cls(
            candidate=d[1].decode(),
            sdp_mid=d[2].decode() if 2 in d else None,
            sdp_mline_index=_int(d[3]) if 3 in d else None,
        )


@dataclass
class WebRTCSolicitOfferWrite:
    sframe_enabled: bool = False

    @classmethod
    def decode(cls, data: bytes) -> "WebRTCSolicitOfferWrite":
        d = _decode(data)
        options = _decode(d[1]) if 1 in d else {}
        return cls(sframe_enabled=bool(_int(options[1])) if 1 in options else False)

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", tlv.encode(b"\x01", _u8(1 if self.sframe_enabled else 0))
        )


@dataclass
class WebRTCSolicitOfferResponse:
    session_identifier: bytes
    status: WebRTCOfferStatus
    sdp_offer: Optional[str] = None
    additional_candidates: list = field(default_factory=list)
    sframe_configuration: Optional[SFrameKeyData] = None

    def encode(self) -> bytes:
        args = [b"\x01", self.session_identifier]
        if self.sdp_offer is not None:
            args += [b"\x02", self.sdp_offer.encode()]
        if self.additional_candidates:
            args += [
                b"\x03",
                _join_items([c.encode() for c in self.additional_candidates]),
            ]
        args += [b"\x04", _u8(self.status)]
        if self.sframe_configuration is not None:
            args += [b"\x05", self.sframe_configuration.encode()]
        return tlv.encode(*args)

    @classmethod
    def decode(cls, data: bytes) -> "WebRTCSolicitOfferResponse":
        d = _decode(data)
        return cls(
            session_identifier=d[1],
            sdp_offer=d[2].decode() if 2 in d else None,
            additional_candidates=[
                WebRTCICECandidate.decode(i) for i in _split_items(d[3])
            ]
            if 3 in d
            else [],
            status=WebRTCOfferStatus(_int(d[4])),
            sframe_configuration=SFrameKeyData.decode(d[5]) if 5 in d else None,
        )


@dataclass
class WebRTCProvideAnswerWrite:
    session_identifier: bytes
    sdp_answer: str
    additional_candidates: list = field(default_factory=list)

    @classmethod
    def decode(cls, data: bytes) -> "WebRTCProvideAnswerWrite":
        d = _decode(data)
        return cls(
            session_identifier=d[1],
            sdp_answer=d[2].decode(),
            additional_candidates=[
                WebRTCICECandidate.decode(i) for i in _split_items(d[3])
            ]
            if 3 in d
            else [],
        )

    def encode(self) -> bytes:
        args = [b"\x01", self.session_identifier, b"\x02", self.sdp_answer.encode()]
        if self.additional_candidates:
            args += [
                b"\x03",
                _join_items([c.encode() for c in self.additional_candidates]),
            ]
        return tlv.encode(*args)


def encode_webrtc_status(
    session_identifier: bytes, status: WebRTCStreamingStatus
) -> bytes:
    return tlv.encode(b"\x01", session_identifier, b"\x02", _u8(status))


def decode_webrtc_status(data: bytes) -> tuple:
    d = _decode(data)
    return d[1], WebRTCStreamingStatus(_int(d[2]))


@dataclass
class WebRTCReofferWrite:
    session_identifier: bytes
    sdp_offer: str
    sframe_enabled: bool = False

    @classmethod
    def decode(cls, data: bytes) -> "WebRTCReofferWrite":
        d = _decode(data)
        options = _decode(d[3]) if 3 in d else {}
        return cls(
            session_identifier=d[1],
            sdp_offer=d[2].decode(),
            sframe_enabled=bool(_int(options[1])) if 1 in options else False,
        )

    def encode(self) -> bytes:
        return tlv.encode(
            b"\x01", self.session_identifier,
            b"\x02", self.sdp_offer.encode(),
            b"\x03", tlv.encode(b"\x01", _u8(1 if self.sframe_enabled else 0)),
        )


@dataclass
class WebRTCReofferResponse:
    session_identifier: bytes
    sdp_answer: str
    status: WebRTCStreamingStatus
    sframe_configuration: Optional[SFrameKeyData] = None

    def encode(self) -> bytes:
        args = [
            b"\x01", self.session_identifier,
            b"\x02", self.sdp_answer.encode(),
            b"\x03", _u8(self.status),
        ]
        if self.sframe_configuration is not None:
            args += [b"\x04", self.sframe_configuration.encode()]
        return tlv.encode(*args)


@dataclass
class WebRTCUpdateSessionWrite:
    session_identifier: bytes
    receive_keys_to_add: list = field(default_factory=list)
    receive_kids_to_remove: list = field(default_factory=list)

    @classmethod
    def decode(cls, data: bytes) -> "WebRTCUpdateSessionWrite":
        d = _decode(data)
        return cls(
            session_identifier=d[1],
            receive_keys_to_add=[
                SFrameKeyData.decode(i) for i in _split_items(d[2])
            ]
            if 2 in d
            else [],
            receive_kids_to_remove=[
                _int(_decode(i)[1]) for i in _split_items(d[3])
            ]
            if 3 in d
            else [],
        )

    def encode(self) -> bytes:
        args = [b"\x01", self.session_identifier]
        if self.receive_keys_to_add:
            args += [
                b"\x02",
                _join_items([k.encode() for k in self.receive_keys_to_add]),
            ]
        if self.receive_kids_to_remove:
            args += [
                b"\x03",
                _join_items(
                    [tlv.encode(b"\x01", _u64(k)) for k in self.receive_kids_to_remove]
                ),
            ]
        return tlv.encode(*args)


def decode_client_csr_write(data: bytes) -> bytes:
    """Return the 32-byte nonce from a Camera Client CSR write."""
    return _decode(data)[1]


def encode_client_csr_response(csr_der: bytes, nonce_signature: bytes) -> bytes:
    return tlv.encode(b"\x01", csr_der, b"\x02", nonce_signature)


@dataclass
class ClientCertificate:
    certificate: bytes
    ca: bytes

    @classmethod
    def decode(cls, data: bytes) -> "ClientCertificate":
        d = _decode(data)
        return cls(certificate=d[1], ca=d[2])

    def encode(self) -> bytes:
        return tlv.encode(b"\x01", self.certificate, b"\x02", self.ca)


def encode_certificate_status(needs_update: bool) -> bytes:
    return tlv.encode(b"\x01", _u8(1 if needs_update else 0))


def decode_certificate_status(data: bytes) -> bool:
    return bool(_int(_decode(data)[1]))


def sensor_uuid_bytes(sensor_uuid: UUID) -> bytes:
    """Encode a sensor UUID for the Sensor UUID data characteristic."""
    return sensor_uuid.bytes
