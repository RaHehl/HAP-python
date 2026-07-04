"""Contains the Camera accessory and related.

When a HAP client (e.g. iOS) wants to start a video stream it does the following:
[0. Read supported RTP configuration]
[0. Read supported video configuration]
[0. Read supported audio configuration]
[0. Read the current streaming status]
1. Sets the SetupEndpoints characteristic to notify the camera about its IP address,
selected security parameters, etc.
2. The camera responds to the above by setting the SetupEndpoints with its IP address,
etc.
3. The client sets the SelectedRTPStreamConfiguration characteristic to notify the
camera of its prefered audio and video configuration and to initiate the start of the
streaming.
4. The camera starts the streaming with the above configuration.
[5. At some point the client can reconfigure or stop the stream similarly to step 3.]
"""

import asyncio
import functools
import ipaddress
import logging
import os
import struct
import sys
from uuid import UUID

from pyhap import RESOURCE_DIR, hksv, tlv
from pyhap.accessory import Accessory
from pyhap.const import CATEGORY_CAMERA
from pyhap.util import base64_to_bytes, byte_bool, to_base64_str

if sys.version_info >= (3, 11):
    from asyncio import timeout as async_timeout
else:
    from async_timeout import timeout as async_timeout

SETUP_TYPES = {
    "SESSION_ID": b"\x01",
    "STATUS": b"\x02",
    "ADDRESS": b"\x03",
    "VIDEO_SRTP_PARAM": b"\x04",
    "AUDIO_SRTP_PARAM": b"\x05",
    "VIDEO_SSRC": b"\x06",
    "AUDIO_SSRC": b"\x07",
}


SETUP_STATUS = {
    "SUCCESS": b"\x00",
    "BUSY": b"\x01",
    "ERROR": b"\x02",
}


SETUP_IPV = {
    "IPV4": b"\x00",
    "IPV6": b"\x01",
}


SETUP_ADDR_INFO = {
    "ADDRESS_VER": b"\x01",
    "ADDRESS": b"\x02",
    "VIDEO_RTP_PORT": b"\x03",
    "AUDIO_RTP_PORT": b"\x04",
}


SETUP_SRTP_PARAM = {
    "CRYPTO": b"\x01",
    "MASTER_KEY": b"\x02",
    "MASTER_SALT": b"\x03",
}


STREAMING_STATUS = {
    "AVAILABLE": b"\x00",
    "STREAMING": b"\x01",
    "BUSY": b"\x02",
}


RTP_CONFIG_TYPES = {
    "CRYPTO": b"\x02",
}


SRTP_CRYPTO_SUITES = {
    "AES_CM_128_HMAC_SHA1_80": b"\x00",
    "AES_CM_256_HMAC_SHA1_80": b"\x01",
    "NONE": b"\x02",
}


VIDEO_TYPES = {
    "CODEC": b"\x01",
    "CODEC_PARAM": b"\x02",
    "ATTRIBUTES": b"\x03",
    "RTP_PARAM": b"\x04",
}


VIDEO_CODEC_TYPES = {
    "H264": b"\x00",
}


VIDEO_CODEC_PARAM_TYPES = {
    "PROFILE_ID": b"\x01",
    "LEVEL": b"\x02",
    "PACKETIZATION_MODE": b"\x03",
    "CVO_ENABLED": b"\x04",
    "CVO_ID": b"\x05",
}


VIDEO_CODEC_PARAM_CVO_TYPES = {
    "UNSUPPORTED": b"\x01",
    "SUPPORTED": b"\x02",
}


VIDEO_CODEC_PARAM_PROFILE_ID_TYPES = {
    "BASELINE": b"\x00",
    "MAIN": b"\x01",
    "HIGH": b"\x02",
}


VIDEO_CODEC_PARAM_LEVEL_TYPES = {
    "TYPE3_1": b"\x00",
    "TYPE3_2": b"\x01",
    "TYPE4_0": b"\x02",
}


VIDEO_CODEC_PARAM_PACKETIZATION_MODE_TYPES = {
    "NON_INTERLEAVED": b"\x00",
}


VIDEO_ATTRIBUTES_TYPES = {
    "IMAGE_WIDTH": b"\x01",
    "IMAGE_HEIGHT": b"\x02",
    "FRAME_RATE": b"\x03",
}


SUPPORTED_VIDEO_CONFIG_TAG = b"\x01"


SELECTED_STREAM_CONFIGURATION_TYPES = {
    "SESSION": b"\x01",
    "VIDEO": b"\x02",
    "AUDIO": b"\x03",
}


RTP_PARAM_TYPES = {
    "PAYLOAD_TYPE": b"\x01",
    "SYNCHRONIZATION_SOURCE": b"\x02",
    "MAX_BIT_RATE": b"\x03",
    "RTCP_SEND_INTERVAL": b"\x04",
    "MAX_MTU": b"\x05",
    "COMFORT_NOISE_PAYLOAD_TYPE": b"\x06",
}


AUDIO_TYPES = {
    "CODEC": b"\x01",
    "CODEC_PARAM": b"\x02",
    "RTP_PARAM": b"\x03",
    "COMFORT_NOISE": b"\x04",
}


AUDIO_CODEC_TYPES = {
    "PCMU": b"\x00",
    "PCMA": b"\x01",
    "AACELD": b"\x02",
    "OPUS": b"\x03",
}


AUDIO_CODEC_PARAM_TYPES = {
    "CHANNEL": b"\x01",
    "BIT_RATE": b"\x02",
    "SAMPLE_RATE": b"\x03",
    "PACKET_TIME": b"\x04",
}


AUDIO_CODEC_PARAM_BIT_RATE_TYPES = {
    "VARIABLE": b"\x00",
    "CONSTANT": b"\x01",
}


AUDIO_CODEC_PARAM_SAMPLE_RATE_TYPES = {
    "KHZ_8": b"\x00",
    "KHZ_16": b"\x01",
    "KHZ_24": b"\x02",
}


SUPPORTED_AUDIO_CODECS_TAG = b"\x01"
SUPPORTED_COMFORT_NOISE_TAG = b"\x02"
SUPPORTED_AUDIO_CONFIG_TAG = b"\x02"
SET_CONFIG_REQUEST_TAG = b"\x02"
SESSION_ID = b"\x01"


NO_SRTP = b"\x01\x01\x02\x02\x00\x03\x00"
"""Configuration value for no SRTP."""


# ---------------------------------------------------------------------------
# HomeKit Secure Video / multi-tier streaming (spec v1.0, 2026-06-03, R17+).
#
# This is a *separate* streaming model from the legacy CameraRTPStreamManagement
# above. HEVC is only available here. Note that the video codec enumeration is
# renumbered relative to the legacy ``VIDEO_CODEC_TYPES`` (legacy H264 == 0x00).
# ---------------------------------------------------------------------------

#: Camera Capabilities characteristic (spec 4.5).
CAMERA_CAPABILITIES_SERVICE_VERSION = "17.99"

# Canonical TLV definitions for the 17.99 spec live in pyhap.hksv.
TLV_SEPARATOR = hksv.TLV_SEPARATOR


FFMPEG_CMD = (
    "ffmpeg -re -f avfoundation -framerate {fps} -i 0:0 -threads 0 "
    "-vcodec libx264 -an -pix_fmt yuv420p -r {fps} -f rawvideo -tune zerolatency "
    "-vf scale={width}:{height} -b:v {v_max_bitrate}k -bufsize {v_max_bitrate}k "
    "-payload_type 99 -ssrc {v_ssrc} -f rtp "
    "-srtp_out_suite AES_CM_128_HMAC_SHA1_80 -srtp_out_params {v_srtp_key} "
    "srtp://{address}:{v_port}?rtcpport={v_port}&"
    "localrtcpport={v_port}&pkt_size=1378"
)
"""Template for the ffmpeg command."""

logger = logging.getLogger(__name__)


class Camera(Accessory):
    """An Accessory that can negotiated camera stream settings with iOS and start a
    stream.
    """

    category = CATEGORY_CAMERA

    @staticmethod
    def get_supported_rtp_config(support_srtp):
        """Return a tlv representation of the RTP configuration we support.

        SRTP support allows only the AES_CM_128_HMAC_SHA1_80 cipher for now.

        :param support_srtp: True if SRTP is supported, False otherwise.
        :type support_srtp: bool
        """
        if support_srtp:
            crypto = SRTP_CRYPTO_SUITES["AES_CM_128_HMAC_SHA1_80"]
        else:
            crypto = SRTP_CRYPTO_SUITES["NONE"]
        return tlv.encode(RTP_CONFIG_TYPES["CRYPTO"], crypto, to_base64=True)

    @staticmethod
    def get_supported_video_stream_config(video_params):
        """Return a tlv representation of the supported video stream configuration.

        Expected video parameters:
            - codec
            - resolutions

        :param video_params: Supported video configurations
        :type video_params: dict
        """
        codec_params_tlv = tlv.encode(
            VIDEO_CODEC_PARAM_TYPES["PACKETIZATION_MODE"],
            VIDEO_CODEC_PARAM_PACKETIZATION_MODE_TYPES["NON_INTERLEAVED"],
        )

        codec_params = video_params["codec"]
        for profile in codec_params["profiles"]:
            codec_params_tlv += tlv.encode(
                VIDEO_CODEC_PARAM_TYPES["PROFILE_ID"], profile
            )

        for level in codec_params["levels"]:
            codec_params_tlv += tlv.encode(VIDEO_CODEC_PARAM_TYPES["LEVEL"], level)

        attr_tlv = b""
        for resolution in video_params["resolutions"]:
            res_tlv = tlv.encode(
                VIDEO_ATTRIBUTES_TYPES["IMAGE_WIDTH"],
                struct.pack("<H", resolution[0]),
                VIDEO_ATTRIBUTES_TYPES["IMAGE_HEIGHT"],
                struct.pack("<H", resolution[1]),
                VIDEO_ATTRIBUTES_TYPES["FRAME_RATE"],
                struct.pack("<H", resolution[2]),
            )
            attr_tlv += tlv.encode(VIDEO_TYPES["ATTRIBUTES"], res_tlv)

        config_tlv = tlv.encode(
            VIDEO_TYPES["CODEC"],
            VIDEO_CODEC_TYPES["H264"],
            VIDEO_TYPES["CODEC_PARAM"],
            codec_params_tlv,
        )

        return tlv.encode(
            SUPPORTED_VIDEO_CONFIG_TAG, config_tlv + attr_tlv, to_base64=True
        )

    @staticmethod
    def get_supported_audio_stream_config(audio_params):
        """Return a tlv representation of the supported audio stream configuration.

        iOS supports only AACELD and OPUS

        Expected audio parameters:
        - codecs
        - comfort_noise

        :param audio_params: Supported audio configurations
        :type audio_params: dict
        """
        has_supported_codec = False
        configs = b""
        for codec_param in audio_params["codecs"]:
            param_type = codec_param["type"]
            if param_type == "OPUS":
                has_supported_codec = True
                codec = AUDIO_CODEC_TYPES["OPUS"]
                bitrate = AUDIO_CODEC_PARAM_BIT_RATE_TYPES["VARIABLE"]
            elif param_type == "AAC-eld":
                has_supported_codec = True
                codec = AUDIO_CODEC_TYPES["AACELD"]
                bitrate = AUDIO_CODEC_PARAM_BIT_RATE_TYPES["VARIABLE"]
            else:
                logger.warning("Unsupported codec %s", param_type)
                continue

            param_samplerate = codec_param["samplerate"]
            if param_samplerate == 8:
                samplerate = AUDIO_CODEC_PARAM_SAMPLE_RATE_TYPES["KHZ_8"]
            elif param_samplerate == 16:
                samplerate = AUDIO_CODEC_PARAM_SAMPLE_RATE_TYPES["KHZ_16"]
            elif param_samplerate == 24:
                samplerate = AUDIO_CODEC_PARAM_SAMPLE_RATE_TYPES["KHZ_24"]
            else:
                logger.warning("Unsupported sample rate %s", param_samplerate)
                continue

            param_tlv = tlv.encode(
                AUDIO_CODEC_PARAM_TYPES["CHANNEL"],
                b"\x01",
                AUDIO_CODEC_PARAM_TYPES["BIT_RATE"],
                bitrate,
                AUDIO_CODEC_PARAM_TYPES["SAMPLE_RATE"],
                samplerate,
            )
            config_tlv = tlv.encode(
                AUDIO_TYPES["CODEC"], codec, AUDIO_TYPES["CODEC_PARAM"], param_tlv
            )
            configs += tlv.encode(SUPPORTED_AUDIO_CODECS_TAG, config_tlv)

        if not has_supported_codec:
            logger.warning("Client does not support any audio codec that iOS supports.")

            codec = AUDIO_CODEC_TYPES["OPUS"]
            bitrate = AUDIO_CODEC_PARAM_BIT_RATE_TYPES["VARIABLE"]
            samplerate = AUDIO_CODEC_PARAM_SAMPLE_RATE_TYPES["KHZ_24"]

            param_tlv = tlv.encode(
                AUDIO_CODEC_PARAM_TYPES["CHANNEL"],
                b"\x01",
                AUDIO_CODEC_PARAM_TYPES["BIT_RATE"],
                bitrate,
                AUDIO_CODEC_PARAM_TYPES["SAMPLE_RATE"],
                samplerate,
            )

            config_tlv = tlv.encode(
                AUDIO_TYPES["CODEC"], codec, AUDIO_TYPES["CODEC_PARAM"], param_tlv
            )

            configs = tlv.encode(SUPPORTED_AUDIO_CODECS_TAG, config_tlv)

        comfort_noise = byte_bool(audio_params.get("comfort_noise", False))
        audio_config = to_base64_str(
            configs + tlv.encode(SUPPORTED_COMFORT_NOISE_TAG, comfort_noise)
        )
        return audio_config

    # --- HomeKit Secure Video / multi-tier streaming (spec v1.0) ---

    @staticmethod
    def get_supported_video_stream_tiers(video_tiers, payload_type=99):
        """Return a tlv representation of the Supported Video Stream Tiers (spec 4.3).

        Advertises a single video codec (all tiers must share it) with one or more
        tiers. Each tier is a dict with keys ``id``, ``quality`` (HIGHEST/HIGH/
        MEDIUM/LOW), ``avg_bitrate`` (kbps), ``width``, ``height`` and ``fps``.
        The codec is read from the first tier's ``codec`` (H264/H265, default H265).
        """
        codec = video_tiers[0].get("codec", "H265")
        value = hksv.SupportedVideoStreamTiers(
            codec=hksv.VideoCodecType[codec],
            payload_type=payload_type,
            tiers=[
                hksv.VideoStreamTier(
                    identifier=tier["id"],
                    quality=hksv.VideoQuality[tier["quality"]],
                    average_bitrate_kbps=tier["avg_bitrate"],
                    width=tier["width"],
                    height=tier["height"],
                    frame_rate=tier["fps"],
                )
                for tier in video_tiers
            ],
        ).encode()
        return to_base64_str(value)

    @staticmethod
    def get_supported_audio_stream_tiers(audio_tiers, payload_type=110):
        """Return a tlv representation of the Supported Audio Stream Tiers (spec 4.4).

        Only Opus is defined. Exactly one tier is allowed per this spec revision.
        Each tier is a dict with keys ``id``, ``avg_bitrate`` (bits/s),
        ``sample_rate`` (kHz: 16/24/32/48), ``bit_depth`` (8/16/24), ``packet_time``
        (ms, must be 20) and ``channels`` (must be 1).
        """
        value = hksv.SupportedAudioStreamTiers(
            codec=hksv.AudioCodecType.OPUS,
            payload_type=payload_type,
            tiers=[
                hksv.AudioStreamTier(
                    identifier=tier["id"],
                    target_average_bitrate=tier["avg_bitrate"],
                    sample_rate=hksv.AudioSampleRate(
                        {16: 1, 24: 2, 32: 3, 48: 4}[tier.get("sample_rate", 24)]
                    ),
                    bit_depth=hksv.AudioBitDepth(
                        {8: 1, 16: 2, 24: 3}[tier.get("bit_depth", 16)]
                    ),
                    packet_time_ms=tier.get("packet_time", 20),
                    number_of_channels=tier.get("channels", 1),
                )
                for tier in audio_tiers
            ],
        ).encode()
        return to_base64_str(value)

    @staticmethod
    def get_camera_capabilities(sensors):
        """Return a tlv representation of the Camera Capabilities (spec 4.5).

        ``sensors`` is a list of dicts, each with ``uuid`` (16 raw bytes),
        ``width``, ``height`` (sensor dimensions in pixels) and ``video_caps`` -
        a list of dicts (``id`` 16 raw bytes, ``quality``, ``width``, ``height``,
        ``fps``, ``avg_bitrate`` kbps, ``peak_bitrate`` kbps).
        """
        value = hksv.CameraCapabilities(
            sensors=[
                hksv.SensorConfiguration(
                    width=sensor["width"],
                    height=sensor["height"],
                    sensor_uuid=sensor["uuid"],
                    video_stream_capabilities=[
                        hksv.VideoStreamCapability(
                            identifier=cap["id"],
                            video_quality=hksv.VideoQuality[cap["quality"]],
                            width=cap["width"],
                            height=cap["height"],
                            frames_per_second=cap["fps"],
                            average_bit_rate_kbps=cap["avg_bitrate"],
                            peak_bit_rate_kbps=cap["peak_bitrate"],
                        )
                        for cap in sensor["video_caps"]
                    ],
                )
                for sensor in sensors
            ]
        ).encode()
        return to_base64_str(value)

    def __init__(self, options, *args, **kwargs):
        """Initialize a camera accessory with the given options.

        :param options: Describes the supported video and audio configuration
            of this camera. Expected values are video, audio, srtp and address.
            Example configuration:

            .. code-block:: python

            {
                "video": {
                    "codec": {
                        "profiles": [
                            camera.VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["BASELINE"],
                        ],
                        "levels": [
                            camera.VIDEO_CODEC_PARAM_LEVEL_TYPES['TYPE3_1'],
                        ],
                    },
                    "resolutions": [
                        [320, 240, 15],  # Width, Height, framerate
                        [1024, 768, 30],
                        [640, 480, 30],
                        [640, 360, 30],
                        [480, 360, 30],
                        [480, 270, 30],
                        [320, 240, 30],
                        [320, 180, 30],
                    ],
                },
                "audio": {
                    "codecs": [
                        {
                            'type': 'OPUS',
                            'samplerate': 24,
                        },
                        {
                            'type': 'AAC-eld',
                            'samplerate': 16
                        }
                    ],
                },
                "address": "192.168.1.226",  # Address from which the camera will stream
            }

            Additional optional values are:
            - srtp - boolean, defaults to False. Whether the camera supports SRTP.
            - start_stream_cmd - string specifying the command to be executed to start
                the stream. The string can contain the keywords, corresponding to the
                video and audio configuration that was negotiated between the camera
                and the client. See the ``start`` method for a full list of parameters.

        :type options: ``dict``
        """
        self.has_srtp = options.get("srtp", False)
        self.start_stream_cmd = options.get("start_stream_cmd", FFMPEG_CMD)

        self.stream_address = options["address"]
        try:
            ipaddress.IPv4Address(self.stream_address)
            self.stream_address_isv6 = b"\x00"
        except ValueError:
            self.stream_address_isv6 = b"\x01"
        self.sessions = {}

        super().__init__(*args, **kwargs)

        self.add_preload_service("Microphone")
        self._streaming_status = []
        self._management = []
        self._multi_tier = bool(options.get("video_tiers"))
        self._multi_tier_idx = None
        # The legacy CameraRTPStreamManagement service and the new HKSV multi-tier
        # service can coexist on one accessory (legacy H.264 as the baseline that
        # iOS recognises, plus the new HEVC tiers as an enhancement). Set up
        # whichever the options describe.
        if options.get("video"):
            self._setup_stream_management(options)
        if self._multi_tier:
            self._video_tiers = {t["id"]: t for t in options["video_tiers"]}
            self._audio_tiers = {t["id"]: t for t in options.get("audio_tiers", [])}
            self._status_active_char = None
            self._setup_multi_tier_management(options)
        self._webrtc_sessions = {}
        self._webrtc_service = None
        self._webrtc_active_sessions_char = None
        if options.get("webrtc"):
            self._setup_webrtc_management(options)
        self._buffer_events = []
        self._buffer_event_seq = 0
        self._buffer_service = None
        self._buffer_event_seq_char = None
        self._publishing_point = None
        if options.get("buffer_management"):
            self._setup_buffer_management()

    @property
    def streaming_status(self):
        """For backwards compatibility."""
        return self._streaming_status[0]

    def _setup_stream_management(self, options):
        """Create stream management."""
        stream_count = options.get("stream_count", 1)
        for stream_idx in range(stream_count):
            self._management.append(self._create_stream_management(stream_idx, options))
            self._streaming_status.append(STREAMING_STATUS["AVAILABLE"])

    def _create_stream_management(self, stream_idx, options):
        """Create a stream management service."""
        management = self.add_preload_service(
            "CameraRTPStreamManagement", unique_id=stream_idx
        )
        management.configure_char(
            "StreamingStatus",
            getter_callback=lambda: self._get_streaming_status(stream_idx),
        )
        management.configure_char(
            "SupportedRTPConfiguration",
            value=self.get_supported_rtp_config(options.get("srtp", False)),
        )
        management.configure_char(
            "SupportedVideoStreamConfiguration",
            value=self.get_supported_video_stream_config(options["video"]),
        )
        management.configure_char(
            "SupportedAudioStreamConfiguration",
            value=self.get_supported_audio_stream_config(options["audio"]),
        )
        management.configure_char(
            "SelectedRTPStreamConfiguration",
            setter_callback=self.set_selected_stream_configuration,
        )
        management.configure_char(
            "SetupEndpoints",
            setter_callback=lambda value: self.set_endpoints(
                value, stream_idx=stream_idx
            ),
        )
        return management

    # --- HomeKit Secure Video / multi-tier streaming setup (spec v1.0) ---

    def _setup_multi_tier_management(self, options):
        """Create the Camera Capabilities, Global Operating Mode and Multi-Tier
        RTP Stream Management services (spec 3.1, 3.2, 3.6)."""
        stream_idx = len(self._management)
        self._multi_tier_idx = stream_idx
        sensors = options.get("sensors") or self._default_sensors(options)

        capabilities = self.add_preload_service("CameraCapabilities", chars=["Version"])
        capabilities.configure_char(
            "Version", value=CAMERA_CAPABILITIES_SERVICE_VERSION
        )
        capabilities.configure_char(
            "CameraCapabilities", value=self.get_camera_capabilities(sensors)
        )

        operating_mode = self.add_preload_service("CameraGlobalOperatingMode")
        operating_mode.configure_char("HomeKitCameraActive", value=True)
        operating_mode.configure_char("StreamingEnabled", value=True)
        operating_mode.configure_char("CameraOperatingModeIndicator", value=False)

        # Declare HKSV recording explicitly OFF so the controller does not try to
        # build a clip library (which loops on CameraClipsLibraryError.noZoneName
        # when the recording services aren't implemented).
        recording = self.add_preload_service("CameraRecordingManagement")
        recording.configure_char("Active", value=0)
        recording.configure_char("RecordingAudioActive", value=0)

        management = self.add_preload_service("CameraMultiTierRTPStreamManagement")
        management.configure_char("StreamingEnabled", value=True)
        self._status_active_char = management.configure_char("StatusActive", value=True)
        management.configure_char(
            "SupportedVideoStreamTiers",
            value=self.get_supported_video_stream_tiers(options["video_tiers"]),
        )
        management.configure_char(
            "SupportedAudioStreamTiers",
            value=self.get_supported_audio_stream_tiers(
                options.get("audio_tiers")
                or [{"id": 1, "avg_bitrate": 24000, "sample_rate": 24}]
            ),
        )
        management.configure_char(
            "SupportedRTPConfiguration",
            value=self.get_supported_rtp_config(options.get("srtp", True)),
        )
        management.configure_char(
            "SetupEndpoints",
            setter_callback=lambda value: self.set_endpoints(
                value, stream_idx=stream_idx
            ),
        )
        management.configure_char(
            "RTPStreamingControl",
            setter_callback=lambda value: self.set_rtp_streaming_control(value),
        )
        management.configure_char("SensorUUID", value=to_base64_str(sensors[0]["uuid"]))
        self._management.append(management)
        self._streaming_status.append(STREAMING_STATUS["AVAILABLE"])

    @staticmethod
    def _default_sensors(options):
        """Derive a single-sensor Camera Capabilities payload from the tiers."""
        tiers = options["video_tiers"]
        largest = max(tiers, key=lambda t: t["width"] * t["height"])
        # A fixed namespace UUID so the sensor identity is stable across restarts.
        sensor_uuid = UUID(int=0xC0FFEE).bytes
        video_caps = [
            {
                "id": UUID(int=t["id"]).bytes,
                "quality": t["quality"],
                "width": t["width"],
                "height": t["height"],
                "fps": t["fps"],
                "avg_bitrate": t["avg_bitrate"],
                "peak_bitrate": t.get("peak_bitrate", int(t["avg_bitrate"] * 1.1)),
            }
            for t in tiers
        ]
        return [
            {
                "uuid": sensor_uuid,
                "width": largest["width"],
                "height": largest["height"],
                "video_caps": video_caps,
            }
        ]

    def set_rtp_streaming_control(self, value):
        """Handle a write to the RTP Streaming Control characteristic (spec 4.16).

        The controller writes a Start/End command referencing a session (set up via
        Setup Endpoints) and a video/audio tier identifier. The same characteristic
        is then read back for the command status (Write Response).
        """
        try:
            control = hksv.RTPStreamingControlWrite.decode(base64_to_bytes(value))
        except (KeyError, ValueError):
            logger.error("Bad RTP Streaming Control write")
            return
        session_id = UUID(bytes=control.session_identifier)
        status = hksv.RTPStreamingStatus.SUCCESS

        session_info = self.sessions.get(session_id)

        if control.command == hksv.RTPStreamingCommand.START:
            if session_info is None:
                status = hksv.RTPStreamingStatus.UNKNOWN_SESSION_IDENTIFIER
            elif session_info.get("tier_streaming"):
                # Already streaming for this session - idempotent ack, don't
                # spawn a second encoder (iOS may repeat START).
                status = hksv.RTPStreamingStatus.SUCCESS
            else:
                tier = self._video_tiers.get(control.video_tier)
                if tier is None:
                    status = hksv.RTPStreamingStatus.NO_SUCH_STREAM
                else:
                    session_info["tier_streaming"] = True
                    self._start_tier_stream(session_info, control, tier)

        elif control.command == hksv.RTPStreamingCommand.END:
            if session_info is None or not session_info.get("tier_streaming"):
                status = hksv.RTPStreamingStatus.NO_SUCH_STREAM
            else:
                self.driver.add_job(self._stop_tier_stream, session_id)
        else:
            # Other commands (e.g. iOS's 0x00 status poll / keepalive): report the
            # session's current health rather than ERROR, which iOS reads as a
            # failed stream and tears the session down.
            if session_info is None:
                status = hksv.RTPStreamingStatus.UNKNOWN_SESSION_IDENTIFIER
            else:
                status = hksv.RTPStreamingStatus.SUCCESS

        response = to_base64_str(
            hksv.encode_rtp_streaming_status(control.session_identifier, status)
        )
        self._management[self._multi_tier_idx].get_characteristic(
            "RTPStreamingControl"
        ).set_value(response)

    def _start_tier_stream(self, session_info, control, tier):
        """Resolve a tier to encoder parameters and start the stream."""
        opts = dict(session_info)
        opts["v_ssrc"] = control.video_ssrc
        if control.audio_ssrc is not None:
            opts["a_ssrc"] = control.audio_ssrc

        # Resolve the audio tier (audio parameters are advertised up-front in the
        # new model rather than negotiated per-session like the legacy path).
        audio_tier = self._audio_tiers.get(control.audio_tier) or next(
            iter(self._audio_tiers.values()), None
        )
        if audio_tier:
            opts["a_max_bitrate"] = max(1, audio_tier["avg_bitrate"] // 1000)
            opts["a_sample_rate"] = audio_tier.get("sample_rate", 24)
            opts["a_packet_time"] = audio_tier.get("packet_time", 20)

        opts["width"] = tier["width"]
        opts["height"] = tier["height"]
        opts["fps"] = tier["fps"]
        opts["v_max_bitrate"] = tier["avg_bitrate"]
        opts["v_codec_name"] = tier.get("codec", "H265").lower()
        opts["v_payload_type"] = 99
        # Default the H.264 profile so the legacy start_stream paths do not KeyError.
        opts.setdefault("v_profile_id", VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["HIGH"])

        async def _runner():
            success = await self.start_stream(session_info, opts)
            stream_idx = session_info["stream_idx"]
            self._streaming_status[stream_idx] = (
                STREAMING_STATUS["STREAMING"]
                if success
                else STREAMING_STATUS["AVAILABLE"]
            )

        self.driver.add_job(_runner)

    async def _stop_tier_stream(self, session_id):
        """Stop a multi-tier stream and free the session."""
        session_info = self.sessions.pop(session_id, None)
        if session_info is None:
            return
        await self.stop_stream(session_info)
        self._streaming_status[session_info["stream_idx"]] = STREAMING_STATUS["AVAILABLE"]

    def _setup_webrtc_management(self, options):
        """Create the Camera WebRTC Stream Management service (spec 3.7).

        The signaling chars are wired to a session registry; the actual WebRTC
        stack is supplied by overriding the ``webrtc_*`` hooks.
        """
        sensors = options.get("sensors") or self._default_sensors(options)
        service = self.add_preload_service("CameraWebRTCStreamManagement")
        service.configure_char("StreamingEnabled", value=True)
        service.configure_char(
            "WebRTCSupportedVideoStreamTiers",
            value=self.get_supported_video_stream_tiers(options["video_tiers"]),
        )
        service.configure_char(
            "WebRTCSupportedAudioStreamTiers",
            value=self.get_supported_audio_stream_tiers(
                options.get("audio_tiers")
                or [{"id": 1, "avg_bitrate": 24000, "sample_rate": 24}]
            ),
        )
        service.configure_char("SensorUUID", value=to_base64_str(sensors[0]["uuid"]))
        self._webrtc_active_sessions_char = service.configure_char(
            "WebRTCNumberOfActiveSessions", value=0
        )
        service.configure_char(
            "WebRTCSolicitOffer", setter_callback=self.set_webrtc_solicit_offer
        )
        service.configure_char(
            "WebRTCProvideAnswer", setter_callback=self.set_webrtc_provide_answer
        )
        service.configure_char(
            "WebRTCStreamingControl",
            setter_callback=self.set_webrtc_streaming_control,
        )
        service.configure_char(
            "WebRTCReoffer", setter_callback=self.set_webrtc_reoffer
        )
        service.configure_char(
            "WebRTCUpdateSession", setter_callback=self.set_webrtc_update_session
        )
        self._webrtc_service = service

    def _webrtc_respond(self, char_name, response):
        self._webrtc_service.get_characteristic(char_name).set_value(
            to_base64_str(response)
        )

    def _webrtc_update_active_sessions(self):
        active = sum(
            1 for s in self._webrtc_sessions.values() if s.get("state") == "active"
        )
        self._webrtc_active_sessions_char.set_value(active)

    def set_webrtc_solicit_offer(self, value):
        """Handle a write to WebRTC Solicit Offer (spec 4.17)."""
        write = hksv.WebRTCSolicitOfferWrite.decode(base64_to_bytes(value))
        offer = self.webrtc_create_offer(write.sframe_enabled)
        if isinstance(offer, hksv.WebRTCOfferStatus):
            response = hksv.WebRTCSolicitOfferResponse(
                session_identifier=b"\x00" * 16, status=offer
            )
        elif offer is None:
            response = hksv.WebRTCSolicitOfferResponse(
                session_identifier=b"\x00" * 16,
                status=hksv.WebRTCOfferStatus.ERROR,
            )
        else:
            self._webrtc_sessions[offer.session_identifier] = {"state": "offered"}
            response = offer
        self._webrtc_respond("WebRTCSolicitOffer", response.encode())

    def set_webrtc_provide_answer(self, value):
        """Handle a write to WebRTC Provide Answer (spec 4.18)."""
        write = hksv.WebRTCProvideAnswerWrite.decode(base64_to_bytes(value))
        session = self._webrtc_sessions.get(write.session_identifier)
        if session is None:
            status = hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER
        elif self.webrtc_apply_answer(
            write.session_identifier, write.sdp_answer, write.additional_candidates
        ):
            session["state"] = "active"
            self._webrtc_update_active_sessions()
            status = hksv.WebRTCStreamingStatus.SUCCESS
        else:
            status = hksv.WebRTCStreamingStatus.ERROR
        self._webrtc_respond(
            "WebRTCProvideAnswer",
            hksv.encode_webrtc_status(write.session_identifier, status),
        )

    def set_webrtc_streaming_control(self, value):
        """Handle a write to WebRTC Streaming Control (spec 4.19): End a session."""
        d = hksv._decode(base64_to_bytes(value))
        session_identifier = d[1]
        if self._webrtc_sessions.pop(session_identifier, None) is None:
            status = hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER
        else:
            self.webrtc_end_session(session_identifier)
            self._webrtc_update_active_sessions()
            status = hksv.WebRTCStreamingStatus.SUCCESS
        self._webrtc_respond(
            "WebRTCStreamingControl",
            hksv.encode_webrtc_status(session_identifier, status),
        )

    def set_webrtc_reoffer(self, value):
        """Handle a write to WebRTC Reoffer (spec 4.21)."""
        write = hksv.WebRTCReofferWrite.decode(base64_to_bytes(value))
        session = self._webrtc_sessions.get(write.session_identifier)
        result = None
        if session is not None:
            result = self.webrtc_reoffer(
                write.session_identifier, write.sdp_offer, write.sframe_enabled
            )
        if session is None:
            response = hksv.WebRTCReofferResponse(
                session_identifier=write.session_identifier,
                sdp_answer="",
                status=hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER,
            )
        elif result is None:
            response = hksv.WebRTCReofferResponse(
                session_identifier=write.session_identifier,
                sdp_answer="",
                status=hksv.WebRTCStreamingStatus.ERROR,
            )
        else:
            sdp_answer, sframe_configuration = result
            response = hksv.WebRTCReofferResponse(
                session_identifier=write.session_identifier,
                sdp_answer=sdp_answer,
                status=hksv.WebRTCStreamingStatus.SUCCESS,
                sframe_configuration=sframe_configuration,
            )
        self._webrtc_respond("WebRTCReoffer", response.encode())

    def set_webrtc_update_session(self, value):
        """Handle a write to WebRTC Update Session (spec 4.22): SFrame key updates."""
        write = hksv.WebRTCUpdateSessionWrite.decode(base64_to_bytes(value))
        if write.session_identifier not in self._webrtc_sessions:
            status = hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER
        elif self.webrtc_update_session(
            write.session_identifier,
            write.receive_keys_to_add,
            write.receive_kids_to_remove,
        ):
            status = hksv.WebRTCStreamingStatus.SUCCESS
        else:
            status = hksv.WebRTCStreamingStatus.ERROR
        self._webrtc_respond(
            "WebRTCUpdateSession",
            hksv.encode_webrtc_status(write.session_identifier, status),
        )

    def webrtc_create_offer(self, sframe_enabled):
        """Produce a WebRTC offer for a new session.

        Override to return a :class:`pyhap.hksv.WebRTCSolicitOfferResponse` with a
        fresh session identifier, the SDP offer and optional ICE candidates /
        SFrame configuration. Return a :class:`pyhap.hksv.WebRTCOfferStatus` for a
        gated state (e.g. privacy mode) or ``None`` when no stack is available.
        """
        return None

    def webrtc_apply_answer(self, session_identifier, sdp_answer, candidates):
        """Apply the controller's SDP answer; return success. Override."""
        return False

    def webrtc_reoffer(self, session_identifier, sdp_offer, sframe_enabled):
        """Renegotiate a session; return ``(sdp_answer, sframe_config)``. Override."""
        return None

    def webrtc_update_session(self, session_identifier, keys_to_add, kids_to_remove):
        """Apply SFrame receive-key updates; return success. Override."""
        return False

    def webrtc_end_session(self, session_identifier):
        """Tear down a session's WebRTC resources. Override."""

    def _setup_buffer_management(self):
        """Create the Camera Buffer Management service (spec 3.5).

        The lib owns the Camera Event Queue (sequence numbers, query and
        acknowledge); buffering, uploads and CMAF ingest are supplied by
        overriding the ``buffer_*`` hooks and calling ``queue_buffer_event``.
        """
        service = self.add_preload_service("CameraBufferManagement")
        service.configure_char(
            "BufferUploadCommand", setter_callback=self.set_buffer_upload_command
        )
        service.configure_char(
            "BufferActivityCommand", setter_callback=self.set_buffer_activity_command
        )
        service.configure_char(
            "BufferEventCommand", setter_callback=self.set_buffer_event_command
        )
        self._buffer_event_seq_char = service.configure_char(
            "BufferEventSequenceNumber", value=0
        )
        service.configure_char(
            "CameraRecordingPublishingPoint",
            setter_callback=self.set_recording_publishing_point,
        )
        self._buffer_service = service

    @property
    def publishing_point(self):
        """The CMAF publishing point written by the controller, if any."""
        return self._publishing_point

    def queue_buffer_event(
        self, event_type, cmaf_session_id=None, motion_active=None, cmaf_error=None
    ):
        """Queue a Camera Event and notify its sequence number.

        Returns the assigned sequence number. The controller fetches the queue
        via Buffer Event Command and trims it with an Acknowledge.
        """
        self._buffer_event_seq += 1
        event = hksv.BufferEvent(
            sequence_number=self._buffer_event_seq,
            type=event_type,
            cmaf_session_id=cmaf_session_id,
            motion_active=motion_active,
            cmaf_error=cmaf_error,
        )
        self._buffer_events.append(event)
        self._buffer_event_seq_char.set_value(self._buffer_event_seq)
        return self._buffer_event_seq

    def set_buffer_upload_command(self, value):
        """Handle a write to Buffer Upload Command (spec 4.9)."""
        command = hksv.BufferUploadCommand.decode(base64_to_bytes(value))
        clip_id = self.buffer_upload(command)
        self._buffer_service.get_characteristic("BufferUploadCommand").set_value(
            to_base64_str(hksv.encode_buffer_upload_response(clip_id or 0))
        )

    def set_buffer_activity_command(self, value):
        """Handle a write to Buffer Activity Command (spec 4.10)."""
        self.buffer_activity(
            hksv.BufferActivityCommand.decode(base64_to_bytes(value))
        )

    def set_buffer_event_command(self, value):
        """Handle a write to Buffer Event Command (spec 4.11): query/acknowledge."""
        command = hksv.BufferEventCommand.decode(base64_to_bytes(value))
        if command.command is hksv.BufferEventCommandType.ACKNOWLEDGE:
            self._buffer_events = [
                e
                for e in self._buffer_events
                if e.sequence_number > command.sequence_number
            ]
            events = []
        else:
            events = [
                e
                for e in self._buffer_events
                if e.sequence_number >= command.sequence_number
            ]
            if command.limit is not None:
                events = events[: command.limit]
        self._buffer_service.get_characteristic("BufferEventCommand").set_value(
            to_base64_str(hksv.encode_buffer_events_response(events))
        )

    def set_recording_publishing_point(self, value):
        """Handle a write to Camera Recording Publishing Point (spec 4.13)."""
        self._publishing_point = hksv.PublishingPoint.decode(base64_to_bytes(value))
        self.publishing_point_updated(self._publishing_point)

    def buffer_upload(self, command):  # pylint: disable=unused-argument
        """Upload a clip from the buffer; return the clip id. Override."""
        return 0

    def buffer_activity(self, command):
        """Apply a should-record window to the buffer. Override."""

    def publishing_point_updated(self, point):
        """React to a new CMAF publishing point (URL + server CAs). Override."""

    async def _start_stream(self, objs, reconfigure):  # pylint: disable=unused-argument
        """Start or reconfigure video streaming for the given session.

        Schedules ``self.start_stream`` or ``self.reconfigure``.

        No support for reconfigure currently.

        :param objs: TLV-decoded SelectedRTPStreamConfiguration
        :type objs: ``dict``

        :param reconfigure: Whether the stream should be reconfigured instead of
            started.
        :type reconfigure: bool
        """
        video_tlv = objs.get(SELECTED_STREAM_CONFIGURATION_TYPES["VIDEO"])
        audio_tlv = objs.get(SELECTED_STREAM_CONFIGURATION_TYPES["AUDIO"])

        opts = {}

        if video_tlv:
            video_objs = tlv.decode(video_tlv)

            video_codec_params = video_objs.get(VIDEO_TYPES["CODEC_PARAM"])
            if video_codec_params:
                video_codec_param_objs = tlv.decode(video_codec_params)
                opts["v_profile_id"] = video_codec_param_objs[
                    VIDEO_CODEC_PARAM_TYPES["PROFILE_ID"]
                ]
                opts["v_level"] = video_codec_param_objs[
                    VIDEO_CODEC_PARAM_TYPES["LEVEL"]
                ]

            video_attrs = video_objs.get(VIDEO_TYPES["ATTRIBUTES"])
            if video_attrs:
                video_attr_objs = tlv.decode(video_attrs)
                opts["width"] = struct.unpack(
                    "<H", video_attr_objs[VIDEO_ATTRIBUTES_TYPES["IMAGE_WIDTH"]]
                )[0]
                opts["height"] = struct.unpack(
                    "<H", video_attr_objs[VIDEO_ATTRIBUTES_TYPES["IMAGE_HEIGHT"]]
                )[0]
                opts["fps"] = struct.unpack(
                    "<B", video_attr_objs[VIDEO_ATTRIBUTES_TYPES["FRAME_RATE"]]
                )[0]

            video_rtp_param = video_objs.get(VIDEO_TYPES["RTP_PARAM"])
            if video_rtp_param:
                video_rtp_param_objs = tlv.decode(video_rtp_param)
                if RTP_PARAM_TYPES["SYNCHRONIZATION_SOURCE"] in video_rtp_param_objs:
                    opts["v_ssrc"] = struct.unpack(
                        "<I",
                        video_rtp_param_objs.get(
                            RTP_PARAM_TYPES["SYNCHRONIZATION_SOURCE"]
                        ),
                    )[0]
                if RTP_PARAM_TYPES["PAYLOAD_TYPE"] in video_rtp_param_objs:
                    opts["v_payload_type"] = video_rtp_param_objs.get(
                        RTP_PARAM_TYPES["PAYLOAD_TYPE"]
                    )
                if RTP_PARAM_TYPES["MAX_BIT_RATE"] in video_rtp_param_objs:
                    opts["v_max_bitrate"] = struct.unpack(
                        "<H", video_rtp_param_objs.get(RTP_PARAM_TYPES["MAX_BIT_RATE"])
                    )[0]
                if RTP_PARAM_TYPES["RTCP_SEND_INTERVAL"] in video_rtp_param_objs:
                    opts["v_rtcp_interval"] = struct.unpack(
                        "<f",
                        video_rtp_param_objs.get(RTP_PARAM_TYPES["RTCP_SEND_INTERVAL"]),
                    )[0]
                if RTP_PARAM_TYPES["MAX_MTU"] in video_rtp_param_objs:
                    opts["v_max_mtu"] = video_rtp_param_objs.get(
                        RTP_PARAM_TYPES["MAX_MTU"]
                    )

        if audio_tlv:
            audio_objs = tlv.decode(audio_tlv)

            opts["a_codec"] = audio_objs[AUDIO_TYPES["CODEC"]]
            audio_codec_param_objs = tlv.decode(audio_objs[AUDIO_TYPES["CODEC_PARAM"]])
            audio_rtp_param_objs = tlv.decode(audio_objs[AUDIO_TYPES["RTP_PARAM"]])
            opts["a_comfort_noise"] = audio_objs[AUDIO_TYPES["COMFORT_NOISE"]]

            opts["a_channel"] = audio_codec_param_objs[
                AUDIO_CODEC_PARAM_TYPES["CHANNEL"]
            ][0]
            opts["a_bitrate"] = struct.unpack(
                "?", audio_codec_param_objs[AUDIO_CODEC_PARAM_TYPES["BIT_RATE"]]
            )[0]
            opts["a_sample_rate"] = 8 * (
                1 + audio_codec_param_objs[AUDIO_CODEC_PARAM_TYPES["SAMPLE_RATE"]][0]
            )
            opts["a_packet_time"] = struct.unpack(
                "<B", audio_codec_param_objs[AUDIO_CODEC_PARAM_TYPES["PACKET_TIME"]]
            )[0]

            opts["a_ssrc"] = struct.unpack(
                "<I", audio_rtp_param_objs[RTP_PARAM_TYPES["SYNCHRONIZATION_SOURCE"]]
            )[0]
            opts["a_payload_type"] = audio_rtp_param_objs[
                RTP_PARAM_TYPES["PAYLOAD_TYPE"]
            ]
            opts["a_max_bitrate"] = struct.unpack(
                "<H", audio_rtp_param_objs[RTP_PARAM_TYPES["MAX_BIT_RATE"]]
            )[0]
            opts["a_rtcp_interval"] = struct.unpack(
                "<f", audio_rtp_param_objs[RTP_PARAM_TYPES["RTCP_SEND_INTERVAL"]]
            )[0]
            opts["a_comfort_payload_type"] = audio_rtp_param_objs[
                RTP_PARAM_TYPES["COMFORT_NOISE_PAYLOAD_TYPE"]
            ]

        session_objs = tlv.decode(objs[SELECTED_STREAM_CONFIGURATION_TYPES["SESSION"]])
        session_id = UUID(bytes=session_objs[SETUP_TYPES["SESSION_ID"]])
        session_info = self.sessions[session_id]
        stream_idx = session_info["stream_idx"]

        opts.update(session_info)
        success = (
            await self.reconfigure_stream(session_info, opts)
            if reconfigure
            else await self.start_stream(session_info, opts)
        )

        if success:
            self._streaming_status[stream_idx] = STREAMING_STATUS["STREAMING"]
        else:
            logger.error(
                "[%s] Failed to start/reconfigure stream, deleting session.", session_id
            )
            del self.sessions[session_id]
            self._streaming_status[stream_idx] = STREAMING_STATUS["AVAILABLE"]

    def _get_streaming_status(self, stream_idx):
        """Get the streaming status in TLV format.

        Called when iOS reads the StreaminStatus ``Characteristic``.
        """
        return tlv.encode(b"\x01", self._streaming_status[stream_idx], to_base64=True)

    async def _stop_stream(self, objs):
        """Stop the stream for the specified session.

        Schedules ``self.stop_stream``.

        :param objs: TLV-decoded SelectedRTPStreamConfiguration value.
        :param objs: ``dict``
        """
        session_objs = tlv.decode(objs[SELECTED_STREAM_CONFIGURATION_TYPES["SESSION"]])
        session_id = UUID(bytes=session_objs[SETUP_TYPES["SESSION_ID"]])

        session_info = self.sessions.get(session_id)
        if not session_info:
            logger.error(
                "Requested to stop stream for session %s, but no "
                "such session was found",
                session_id,
            )
            return

        stream_idx = session_info["stream_idx"]
        await self.stop_stream(session_info)
        del self.sessions[session_id]

        self._streaming_status[stream_idx] = STREAMING_STATUS["AVAILABLE"]

    def set_selected_stream_configuration(self, value):
        """Set the selected stream configuration.

        Called from iOS to set the SelectedRTPStreamConfiguration ``Characteristic``.

        This method schedules a stream for the session in ``value`` to be start, stopped
        or reconfigured, depending on the request.

        :param value: base64-encoded selected configuration in TLV format
        :type value: ``str``
        """
        logger.info("==> iOS WROTE SelectedRTPStreamConfiguration (legacy H.264 path)")
        logger.debug("set_selected_stream_config - value - %s", value)

        objs = tlv.decode(value, from_base64=True)
        if SELECTED_STREAM_CONFIGURATION_TYPES["SESSION"] not in objs:
            logger.error("Bad request to set selected stream configuration.")
            return

        session = tlv.decode(objs[SELECTED_STREAM_CONFIGURATION_TYPES["SESSION"]])

        request_type = session[b"\x02"][0]
        logger.debug("Set stream config request: %d", request_type)
        if request_type == 1:
            job = functools.partial(self._start_stream, reconfigure=False)
        elif request_type == 0:
            job = self._stop_stream
        elif request_type == 4:
            job = functools.partial(self._start_stream, reconfigure=True)
        else:
            logger.error("Unknown request type %d", request_type)
            return

        self.driver.add_job(job, objs)

    def set_streaming_available(self, stream_idx):
        """Send an update to the controller that streaming is available."""
        self._streaming_status[stream_idx] = STREAMING_STATUS["AVAILABLE"]
        if stream_idx == self._multi_tier_idx:
            # The multi-tier service has no StreamingStatus characteristic; the
            # controller learns about teardown through the RTP session itself.
            return
        self._management[stream_idx].get_characteristic("StreamingStatus").notify()

    def set_endpoints(self, value, stream_idx=None):
        """Configure streaming endpoints.

        Called when iOS sets the SetupEndpoints ``Characteristic``. The endpoint
        information for the camera should be set as the current value of SetupEndpoints.

        :param value: The base64-encoded stream session details in TLV format.
        :param value: ``str``
        """
        if stream_idx is None:
            stream_idx = 0

        objs = tlv.decode(value, from_base64=True)
        session_id = UUID(bytes=objs[SETUP_TYPES["SESSION_ID"]])

        # Extract address info
        address_tlv = objs[SETUP_TYPES["ADDRESS"]]
        address_info_objs = tlv.decode(address_tlv)
        is_ipv6 = struct.unpack("?", address_info_objs[SETUP_ADDR_INFO["ADDRESS_VER"]])[
            0
        ]
        address = address_info_objs[SETUP_ADDR_INFO["ADDRESS"]].decode("utf8")
        target_video_port = struct.unpack(
            "<H", address_info_objs[SETUP_ADDR_INFO["VIDEO_RTP_PORT"]]
        )[0]
        target_audio_port = struct.unpack(
            "<H", address_info_objs[SETUP_ADDR_INFO["AUDIO_RTP_PORT"]]
        )[0]

        # Video SRTP Params
        video_srtp_tlv = objs[SETUP_TYPES["VIDEO_SRTP_PARAM"]]
        video_info_objs = tlv.decode(video_srtp_tlv)
        video_crypto_suite = video_info_objs[SETUP_SRTP_PARAM["CRYPTO"]][0]
        video_master_key = video_info_objs[SETUP_SRTP_PARAM["MASTER_KEY"]]
        video_master_salt = video_info_objs[SETUP_SRTP_PARAM["MASTER_SALT"]]

        # Audio SRTP Params
        audio_srtp_tlv = objs[SETUP_TYPES["AUDIO_SRTP_PARAM"]]
        audio_info_objs = tlv.decode(audio_srtp_tlv)
        audio_crypto_suite = audio_info_objs[SETUP_SRTP_PARAM["CRYPTO"]][0]
        audio_master_key = audio_info_objs[SETUP_SRTP_PARAM["MASTER_KEY"]]
        audio_master_salt = audio_info_objs[SETUP_SRTP_PARAM["MASTER_SALT"]]

        logger.debug(
            "Received endpoint configuration:"
            "\nsession_id: %s\naddress: %s\nis_ipv6: %s"
            "\ntarget_video_port: %s\ntarget_audio_port: %s"
            "\nvideo_crypto_suite: %s\nvideo_srtp: %s"
            "\naudio_crypto_suite: %s\naudio_srtp: %s",
            session_id,
            address,
            is_ipv6,
            target_video_port,
            target_audio_port,
            video_crypto_suite,
            to_base64_str(video_master_key + video_master_salt),
            audio_crypto_suite,
            to_base64_str(audio_master_key + audio_master_salt),
        )

        # Configure the SetupEndpoints response

        if self.has_srtp:
            video_srtp_tlv = tlv.encode(
                SETUP_SRTP_PARAM["CRYPTO"],
                SRTP_CRYPTO_SUITES["AES_CM_128_HMAC_SHA1_80"],
                SETUP_SRTP_PARAM["MASTER_KEY"],
                video_master_key,
                SETUP_SRTP_PARAM["MASTER_SALT"],
                video_master_salt,
            )

            audio_srtp_tlv = tlv.encode(
                SETUP_SRTP_PARAM["CRYPTO"],
                SRTP_CRYPTO_SUITES["AES_CM_128_HMAC_SHA1_80"],
                SETUP_SRTP_PARAM["MASTER_KEY"],
                audio_master_key,
                SETUP_SRTP_PARAM["MASTER_SALT"],
                audio_master_salt,
            )
        else:
            video_srtp_tlv = NO_SRTP
            audio_srtp_tlv = NO_SRTP

        video_ssrc = int.from_bytes(os.urandom(3), byteorder="big")
        audio_ssrc = int.from_bytes(os.urandom(3), byteorder="big")

        res_address_tlv = tlv.encode(
            SETUP_ADDR_INFO["ADDRESS_VER"],
            self.stream_address_isv6,
            SETUP_ADDR_INFO["ADDRESS"],
            self.stream_address.encode("utf-8"),
            SETUP_ADDR_INFO["VIDEO_RTP_PORT"],
            struct.pack("<H", target_video_port),
            SETUP_ADDR_INFO["AUDIO_RTP_PORT"],
            struct.pack("<H", target_audio_port),
        )

        response_tlv = tlv.encode(
            SETUP_TYPES["SESSION_ID"],
            session_id.bytes,
            SETUP_TYPES["STATUS"],
            SETUP_STATUS["SUCCESS"],
            SETUP_TYPES["ADDRESS"],
            res_address_tlv,
            SETUP_TYPES["VIDEO_SRTP_PARAM"],
            video_srtp_tlv,
            SETUP_TYPES["AUDIO_SRTP_PARAM"],
            audio_srtp_tlv,
            SETUP_TYPES["VIDEO_SSRC"],
            struct.pack("<I", video_ssrc),
            SETUP_TYPES["AUDIO_SSRC"],
            struct.pack("<I", audio_ssrc),
            to_base64=True,
        )

        self.sessions[session_id] = {
            "id": session_id,
            "stream_idx": stream_idx,
            "address": address,
            "v_port": target_video_port,
            "v_srtp_key": to_base64_str(video_master_key + video_master_salt),
            "v_ssrc": video_ssrc,
            "a_port": target_audio_port,
            "a_srtp_key": to_base64_str(audio_master_key + audio_master_salt),
            "a_ssrc": audio_ssrc,
        }

        self._management[stream_idx].get_characteristic("SetupEndpoints").set_value(
            response_tlv
        )

    async def stop(self):
        """Stop all streaming sessions."""
        await asyncio.gather(
            *(self.stop_stream(session_info) for session_info in self.sessions.values())
        )

    # ### For client extensions ###

    async def start_stream(self, session_info, stream_config):
        """Start a new stream with the given configuration.

        This method can be implemented to start a new stream. Any specific information
        about the started stream can be persisted in the ``session_info`` argument.
        The same will be passed to ``stop_stream`` when the stream for this session
        needs to be stopped.

        The default implementation starts a new process with the command in
        ``self.start_stream_cmd``, formatted with the ``stream_config``.

        :param session_info: Contains information about the current session. Can be used
            for session storage. Available keys:
            - id - The session ID.
        :type session_info: ``dict``
        :param stream_config: Stream configuration, as negotiated with the HAP client.
            Implementations can only use part of these. Available keys:
            General configuration:
                - address - The IP address from which the camera will stream
                - v_port - Remote port to which to stream video
                - v_srtp_key - Base64-encoded key and salt value for the
                    AES_CM_128_HMAC_SHA1_80 cipher to use when streaming video.
                    The key and the salt are concatenated before encoding
                - a_port - Remote audio port to which to stream audio
                - a_srtp_key - As v_srtp_params, but for the audio stream.
            Video configuration:
                - v_profile_id - The profile ID for the H.264 codec, e.g. baseline.
                    Refer to ``VIDEO_CODEC_PARAM_PROFILE_ID_TYPES``.
                - v_level - The level in the profile ID, e.g. 3:1.
                    Refer to ``VIDEO_CODEC_PARAM_LEVEL_TYPES``.
                - width - Video width
                - height - Video height
                - fps - Video frame rate
                - v_ssrc - Video synchronisation source
                - v_payload_type - Type of the video codec
                - v_max_bitrate - Maximum bit rate generated by the codec in kbps
                    and averaged over 1 second
                - v_rtcp_interval - Minimum RTCP interval in seconds
                - v_max_mtu - MTU that the IP camera must use to transmit
                    Video RTP packets.
            Audio configuration:
                - a_bitrate - Whether the bitrate is variable or constant
                - a_codec - Audio codec
                - a_comfort_noise - Wheter to use a comfort noise codec
                - a_channel - Number of audio channels
                - a_sample_rate - Audio sample rate in KHz
                - a_packet_time - Length of time represented by the media in a packet
                - a_ssrc - Audio synchronisation source
                - a_payload_type - Type of the audio codec
                - a_max_bitrate - Maximum bit rate generated by the codec in kbps
                    and averaged over 1 second
                - a_rtcp_interval - Minimum RTCP interval in seconds
                - a_comfort_payload_type - The type of codec for comfort noise

        :return: True if and only if starting the stream command was successful.
        :rtype: ``bool``
        """
        logger.debug(
            "[%s] Starting stream with the following parameters: %s",
            session_info["id"],
            stream_config,
        )

        cmd = self.start_stream_cmd.format(**stream_config).split()
        logger.debug('Executing start stream command: "%s"', " ".join(cmd))
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                limit=1024,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to start streaming process because of error: %s", e)
            return False

        session_info["process"] = process

        logger.info(
            "[%s] Started stream process - PID %d", session_info["id"], process.pid
        )

        return True

    async def stop_stream(self, session_info):
        """Stop the stream for the given ``session_id``.

        This method can be implemented if custom stop stream commands are needed. The
        default implementation gets the ``process`` value from the ``session_info``
        object and terminates it (assumes it is a ``subprocess.Popen`` object).

        :param session_info: The session info object. Available keys:
            - id - The session ID.
        :type session_info: ``dict``
        """
        session_id = session_info["id"]
        ffmpeg_process = session_info.get("process")
        if ffmpeg_process:
            logger.info("[%s] Stopping stream.", session_id)
            try:
                ffmpeg_process.terminate()
                async with async_timeout(2.0):
                    _, stderr = await ffmpeg_process.communicate()
                logger.debug("Stream command stderr: %s", stderr)
            except asyncio.TimeoutError:
                logger.error(
                    "Timeout while waiting for the stream process "
                    "to terminate. Trying with kill."
                )
                ffmpeg_process.kill()
                await ffmpeg_process.wait()
            logger.debug("Stream process stopped.")
        else:
            logger.warning("No process for session ID %s", session_id)

    async def reconfigure_stream(self, session_info, stream_config):
        """Reconfigure the stream so that it uses the given ``stream_config``.

        :param session_info: The session object for the session that needs to
            be reconfigured. Available keys:
            - id - The session id.
        :type session_id: ``dict``

        :return: True if and only if the reconfiguration is successful.
        :rtype: ``bool``
        """
        await self.start_stream(session_info, stream_config)

    def get_snapshot(self, image_size):  # pylint: disable=unused-argument
        """Return a jpeg of a snapshot from the camera.

        Overwrite to implement getting snapshots from your camera.

        :param image_size: ``dict`` describing the requested image size. Contains the
            keys "image-width" and "image-height"
        """
        with open(os.path.join(RESOURCE_DIR, "snapshot.jpg"), "rb") as fp:
            return fp.read()
