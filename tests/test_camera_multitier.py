"""Tests for the multi-tier (HKSV 17.99) camera paths."""

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from pyhap import hksv
from pyhap.accessory_driver import AccessoryDriver
from pyhap import camera as camera_module
from pyhap.camera import Camera
from pyhap.util import base64_to_bytes, to_base64_str


VIDEO_TIERS = [
    {"id": 1, "quality": "HIGH", "avg_bitrate": 4500, "width": 3840, "height": 2160,
     "fps": 30, "codec": "H265"},
    {"id": 2, "quality": "MEDIUM", "avg_bitrate": 1700, "width": 1920, "height": 1080,
     "fps": 30},
    {"id": 3, "quality": "LOW", "avg_bitrate": 180, "width": 640, "height": 360,
     "fps": 15},
]
AUDIO_TIERS = [{"id": 1, "avg_bitrate": 24000, "sample_rate": 24}]
SENSORS = [{
    "uuid": UUID(int=0xC0FFEE).bytes, "width": 3840, "height": 2160,
    "video_caps": [
        {"id": UUID(int=1).bytes, "quality": "HIGH", "width": 3840, "height": 2160,
         "fps": 30, "avg_bitrate": 4500, "peak_bitrate": 5000},
        {"id": UUID(int=3).bytes, "quality": "LOW", "width": 640, "height": 360,
         "fps": 15, "avg_bitrate": 180, "peak_bitrate": 198},
    ],
}]

# Golden vectors captured from the wire format validated live against iOS 26.5.
GOLDEN_VIDEO_TIERS = (
    "AQECAgFjA1IBBAEAAAACAQIDBJQRAAAEAgAPBQJwCAYBHgAAAQQCAAAAAgEDAwSkBgAABAKABwUC"
    "OAQGAR4AAAEEAwAAAAIBBAMEtAAAAAQCgAIFAmgBBgEP"
)
GOLDEN_AUDIO_TIERS = "AQEDAgFuAxgBBAEAAAACBMBdAAADAQIEAQIFARQGAQE="
GOLDEN_CAPABILITIES = (
    "AQEBAoABfgEIAQIADwICcAgCEAAAAAAAAAAAAAAAAADA/+4DAQEEAQEFWgEQAAAAAAAAAAAAAAAA"
    "AAAAAQIBAgMCAA8EAnAIBQEeBgSUEQAABwSIEwAAAAABEAAAAAAAAAAAAAAAAAAAAAMCAQQDAoAC"
    "BAJoAQUBDwYEtAAAAAcExgAAAA=="
)


def test_supported_video_stream_tiers_golden():
    assert Camera.get_supported_video_stream_tiers(VIDEO_TIERS) == GOLDEN_VIDEO_TIERS
    decoded = hksv.SupportedVideoStreamTiers.decode(
        base64_to_bytes(GOLDEN_VIDEO_TIERS)
    )
    assert decoded.codec is hksv.VideoCodecType.H265
    assert [t.identifier for t in decoded.tiers] == [1, 2, 3]


def test_supported_audio_stream_tiers_golden():
    assert Camera.get_supported_audio_stream_tiers(AUDIO_TIERS) == GOLDEN_AUDIO_TIERS
    decoded = hksv.SupportedAudioStreamTiers.decode(
        base64_to_bytes(GOLDEN_AUDIO_TIERS)
    )
    assert decoded.codec is hksv.AudioCodecType.OPUS
    assert decoded.tiers[0].sample_rate is hksv.AudioSampleRate.KHZ_24


def test_camera_capabilities_golden():
    assert Camera.get_camera_capabilities(SENSORS) == GOLDEN_CAPABILITIES
    decoded = hksv.CameraCapabilities.decode(base64_to_bytes(GOLDEN_CAPABILITIES))
    assert decoded.version == 1
    assert decoded.sensors[0].sensor_intent is hksv.SensorIntent.MAIN
    assert len(decoded.sensors[0].video_stream_capabilities) == 2


@pytest.fixture(name="multi_tier_camera")
def multi_tier_camera_fixture():
    with patch(
        "pyhap.accessory_driver.AccessoryDriver.persist"
    ), patch("pyhap.accessory_driver.AccessoryDriver.load"):
        driver = AccessoryDriver(loop=MagicMock(), listen_address="127.0.0.1")
        options = {
            "video": {
                "codec": {
                    "profiles": [
                        camera_module.VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["BASELINE"]
                    ],
                    "levels": [camera_module.VIDEO_CODEC_PARAM_LEVEL_TYPES["TYPE3_1"]],
                },
                "resolutions": [[640, 360, 15]],
            },
            "audio": {"codecs": [{"type": "OPUS", "samplerate": 24}]},
            "srtp": True,
            "address": "127.0.0.1",
            "video_tiers": VIDEO_TIERS,
            "audio_tiers": AUDIO_TIERS,
        }
        yield Camera(options, driver, "Cam")


def test_multi_tier_service_setup(multi_tier_camera):
    camera = multi_tier_camera

    management = camera.get_service("CameraMultiTierRTPStreamManagement")
    assert management.get_characteristic("StreamingEnabled").get_value() is True
    tiers_value = management.get_characteristic("SupportedVideoStreamTiers").get_value()
    assert tiers_value == GOLDEN_VIDEO_TIERS

    operating_mode = camera.get_service("CameraGlobalOperatingMode")
    assert operating_mode.get_characteristic("HomeKitCameraActive").get_value() is True

    # Recording is declared off; without the service the controller loops on
    # CameraClipsLibraryError.noZoneName.
    recording = camera.get_service("CameraRecordingManagement")
    assert recording.get_characteristic("Active").get_value() == 0
    assert recording.get_characteristic("RecordingAudioActive").get_value() == 0

    capabilities = camera.get_service("CameraCapabilities")
    assert capabilities.get_characteristic("Version").get_value() == "17.99"


def _control_response(camera):
    value = (
        camera.get_service("CameraMultiTierRTPStreamManagement")
        .get_characteristic("RTPStreamingControl")
        .get_value()
    )
    d = hksv._decode(base64_to_bytes(value))
    return UUID(bytes=d[1]), hksv.RTPStreamingStatus(hksv._int(d[2]))


def test_rtp_streaming_control_unknown_session(multi_tier_camera):
    camera = multi_tier_camera
    session_id = UUID(int=42)
    write = hksv.RTPStreamingControlWrite(
        session_identifier=session_id.bytes,
        command=hksv.RTPStreamingCommand.START,
        video_tier=1,
        video_ssrc=1234,
    )
    camera.set_rtp_streaming_control(to_base64_str(write.encode()))
    response_session, status = _control_response(camera)
    assert response_session == session_id
    assert status is hksv.RTPStreamingStatus.UNKNOWN_SESSION_IDENTIFIER


def test_rtp_streaming_control_start_and_idempotent_repeat(multi_tier_camera):
    camera = multi_tier_camera
    session_id = UUID(int=7)
    camera.sessions[session_id] = {"stream_idx": camera._multi_tier_idx}
    camera._start_tier_stream = MagicMock()

    write = to_base64_str(
        hksv.RTPStreamingControlWrite(
            session_identifier=session_id.bytes,
            command=hksv.RTPStreamingCommand.START,
            video_tier=1,
            video_ssrc=1234,
            audio_tier=1,
            audio_ssrc=5678,
        ).encode()
    )
    camera.set_rtp_streaming_control(write)
    _, status = _control_response(camera)
    assert status is hksv.RTPStreamingStatus.SUCCESS
    assert camera._start_tier_stream.call_count == 1
    assert camera.sessions[session_id]["tier_streaming"] is True

    # iOS may repeat START for a running session: ack, no second encoder.
    camera.set_rtp_streaming_control(write)
    _, status = _control_response(camera)
    assert status is hksv.RTPStreamingStatus.SUCCESS
    assert camera._start_tier_stream.call_count == 1


def test_rtp_streaming_control_unknown_tier(multi_tier_camera):
    camera = multi_tier_camera
    session_id = UUID(int=8)
    camera.sessions[session_id] = {"stream_idx": camera._multi_tier_idx}

    write = to_base64_str(
        hksv.RTPStreamingControlWrite(
            session_identifier=session_id.bytes,
            command=hksv.RTPStreamingCommand.START,
            video_tier=99,
            video_ssrc=1234,
        ).encode()
    )
    camera.set_rtp_streaming_control(write)
    _, status = _control_response(camera)
    assert status is hksv.RTPStreamingStatus.NO_SUCH_STREAM


def test_rtp_streaming_control_keepalive_reports_health(multi_tier_camera):
    camera = multi_tier_camera
    session_id = UUID(int=9)

    # Unknown command (iOS status poll) on an unknown session.
    poll = to_base64_str(
        hksv.tlv.encode(b"\x01", session_id.bytes, b"\x02", b"\x00")
    )
    camera.set_rtp_streaming_control(poll)
    _, status = _control_response(camera)
    assert status is hksv.RTPStreamingStatus.UNKNOWN_SESSION_IDENTIFIER

    # Same poll with a live session reports success instead of ERROR.
    camera.sessions[session_id] = {"stream_idx": camera._multi_tier_idx}
    camera.set_rtp_streaming_control(poll)
    _, status = _control_response(camera)
    assert status is hksv.RTPStreamingStatus.SUCCESS
