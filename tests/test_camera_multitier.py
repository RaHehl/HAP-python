"""Tests for the multi-tier (HKSV 17.99) camera paths."""

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from pyhap import hksv, tlv
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


@pytest.fixture(name="webrtc_camera")
def webrtc_camera_fixture():
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
            "webrtc": True,
        }
        yield Camera(options, driver, "Cam")


def _webrtc_response(camera, char_name, decoder):
    value = (
        camera.get_service("CameraWebRTCStreamManagement")
        .get_characteristic(char_name)
        .get_value()
    )
    return decoder(base64_to_bytes(value))


def test_webrtc_service_setup(webrtc_camera):
    camera = webrtc_camera
    service = camera.get_service("CameraWebRTCStreamManagement")
    assert service.get_characteristic("StreamingEnabled").get_value() is True
    assert (
        service.get_characteristic("WebRTCSupportedVideoStreamTiers").get_value()
        == GOLDEN_VIDEO_TIERS
    )
    assert service.get_characteristic("WebRTCNumberOfActiveSessions").get_value() == 0


def test_webrtc_solicit_offer_without_stack_errors(webrtc_camera):
    camera = webrtc_camera
    write = to_base64_str(hksv.WebRTCSolicitOfferWrite().encode())
    camera.set_webrtc_solicit_offer(write)
    response = _webrtc_response(
        camera, "WebRTCSolicitOffer", hksv.WebRTCSolicitOfferResponse.decode
    )
    assert response.status is hksv.WebRTCOfferStatus.ERROR
    assert not camera._webrtc_sessions


def test_webrtc_solicit_offer_privacy_mode(webrtc_camera):
    camera = webrtc_camera
    camera.webrtc_create_offer = (
        lambda sframe_enabled: hksv.WebRTCOfferStatus.PRIVACY_MODE_ACTIVE
    )
    camera.set_webrtc_solicit_offer(
        to_base64_str(hksv.WebRTCSolicitOfferWrite().encode())
    )
    response = _webrtc_response(
        camera, "WebRTCSolicitOffer", hksv.WebRTCSolicitOfferResponse.decode
    )
    assert response.status is hksv.WebRTCOfferStatus.PRIVACY_MODE_ACTIVE


def test_webrtc_full_session_lifecycle(webrtc_camera):
    camera = webrtc_camera
    session_id = UUID(int=21).bytes
    camera.webrtc_create_offer = lambda sframe_enabled: (
        hksv.WebRTCSolicitOfferResponse(
            session_identifier=session_id,
            status=hksv.WebRTCOfferStatus.SUCCESS,
            sdp_offer="v=0\r\n",
            sframe_configuration=(
                hksv.SFrameKeyData(key=b"\x01" * 16, kid=1) if sframe_enabled else None
            ),
        )
    )
    camera.webrtc_apply_answer = lambda sid, answer, candidates: True
    camera.webrtc_update_session = lambda sid, add, remove: True
    ended = []
    camera.webrtc_end_session = ended.append

    # Solicit (with SFrame)
    camera.set_webrtc_solicit_offer(
        to_base64_str(hksv.WebRTCSolicitOfferWrite(sframe_enabled=True).encode())
    )
    offer = _webrtc_response(
        camera, "WebRTCSolicitOffer", hksv.WebRTCSolicitOfferResponse.decode
    )
    assert offer.status is hksv.WebRTCOfferStatus.SUCCESS
    assert offer.sframe_configuration.kid == 1
    assert camera._webrtc_sessions[session_id]["state"] == "offered"

    # Provide answer -> active, session count notifies 1
    camera.set_webrtc_provide_answer(
        to_base64_str(
            hksv.WebRTCProvideAnswerWrite(
                session_identifier=session_id, sdp_answer="v=0\r\n"
            ).encode()
        )
    )
    _, status = _webrtc_response(
        camera, "WebRTCProvideAnswer", hksv.decode_webrtc_status
    )
    assert status is hksv.WebRTCStreamingStatus.SUCCESS
    service = camera.get_service("CameraWebRTCStreamManagement")
    assert service.get_characteristic("WebRTCNumberOfActiveSessions").get_value() == 1

    # SFrame key update
    camera.set_webrtc_update_session(
        to_base64_str(
            hksv.WebRTCUpdateSessionWrite(
                session_identifier=session_id,
                receive_keys_to_add=[hksv.SFrameKeyData(key=b"\x02" * 16, kid=2)],
            ).encode()
        )
    )
    _, status = _webrtc_response(
        camera, "WebRTCUpdateSession", hksv.decode_webrtc_status
    )
    assert status is hksv.WebRTCStreamingStatus.SUCCESS

    # End -> teardown hook, count back to 0
    camera.set_webrtc_streaming_control(
        to_base64_str(
            hksv.tlv.encode(b"\x01", session_id, b"\x02", b"\x01")
        )
    )
    _, status = _webrtc_response(
        camera, "WebRTCStreamingControl", hksv.decode_webrtc_status
    )
    assert status is hksv.WebRTCStreamingStatus.SUCCESS
    assert ended == [session_id]
    assert service.get_characteristic("WebRTCNumberOfActiveSessions").get_value() == 0
    assert not camera._webrtc_sessions


def test_webrtc_unknown_session_paths(webrtc_camera):
    camera = webrtc_camera
    unknown = UUID(int=99).bytes

    camera.set_webrtc_provide_answer(
        to_base64_str(
            hksv.WebRTCProvideAnswerWrite(
                session_identifier=unknown, sdp_answer="v=0\r\n"
            ).encode()
        )
    )
    _, status = _webrtc_response(
        camera, "WebRTCProvideAnswer", hksv.decode_webrtc_status
    )
    assert status is hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER

    camera.set_webrtc_streaming_control(
        to_base64_str(hksv.tlv.encode(b"\x01", unknown, b"\x02", b"\x01"))
    )
    _, status = _webrtc_response(
        camera, "WebRTCStreamingControl", hksv.decode_webrtc_status
    )
    assert status is hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER

    camera.set_webrtc_reoffer(
        to_base64_str(
            hksv.WebRTCReofferWrite(
                session_identifier=unknown, sdp_offer="v=0\r\n"
            ).encode()
        )
    )
    reoffer = _webrtc_response(
        camera, "WebRTCReoffer", hksv.WebRTCReofferResponse.decode
    )
    assert reoffer.status is hksv.WebRTCStreamingStatus.UNKNOWN_SESSION_IDENTIFIER


def test_webrtc_reoffer_success(webrtc_camera):
    camera = webrtc_camera
    session_id = UUID(int=5).bytes
    camera._webrtc_sessions[session_id] = {"state": "active"}
    camera.webrtc_reoffer = lambda sid, offer, sframe: ("v=0\r\nanswer", None)

    camera.set_webrtc_reoffer(
        to_base64_str(
            hksv.WebRTCReofferWrite(
                session_identifier=session_id, sdp_offer="v=0\r\n"
            ).encode()
        )
    )
    reoffer = _webrtc_response(
        camera, "WebRTCReoffer", hksv.WebRTCReofferResponse.decode
    )
    assert reoffer.status is hksv.WebRTCStreamingStatus.SUCCESS
    assert reoffer.sdp_answer == "v=0\r\nanswer"


@pytest.fixture(name="buffer_camera")
def buffer_camera_fixture():
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
            "buffer_management": True,
        }
        yield Camera(options, driver, "Cam")


def _buffer_char(camera, name):
    return camera.get_service("CameraBufferManagement").get_characteristic(name)


def test_buffer_event_queue_notify_query_acknowledge(buffer_camera):
    camera = buffer_camera
    assert _buffer_char(camera, "BufferEventSequenceNumber").get_value() == 0

    seq1 = camera.queue_buffer_event(
        hksv.BufferEventType.CMAF_SESSION_START, cmaf_session_id=5
    )
    seq2 = camera.queue_buffer_event(hksv.BufferEventType.MOTION, motion_active=True)
    seq3 = camera.queue_buffer_event(
        hksv.BufferEventType.CMAF_ERROR,
        cmaf_session_id=5,
        cmaf_error=hksv.CMAFError.TIMEOUT,
    )
    assert (seq1, seq2, seq3) == (1, 2, 3)
    assert _buffer_char(camera, "BufferEventSequenceNumber").get_value() == 3

    # Query from seq 2, unlimited
    camera.set_buffer_event_command(
        to_base64_str(
            hksv.BufferEventCommand(
                command=hksv.BufferEventCommandType.QUERY, sequence_number=2
            ).encode()
        )
    )
    events = hksv.decode_buffer_events_response(
        base64_to_bytes(_buffer_char(camera, "BufferEventCommand").get_value())
    )
    assert [e.sequence_number for e in events] == [2, 3]
    assert events[1].cmaf_error is hksv.CMAFError.TIMEOUT

    # Query with limit
    camera.set_buffer_event_command(
        to_base64_str(
            hksv.BufferEventCommand(
                command=hksv.BufferEventCommandType.QUERY, sequence_number=1, limit=1
            ).encode()
        )
    )
    events = hksv.decode_buffer_events_response(
        base64_to_bytes(_buffer_char(camera, "BufferEventCommand").get_value())
    )
    assert [e.sequence_number for e in events] == [1]

    # Acknowledge trims everything up to seq 2
    camera.set_buffer_event_command(
        to_base64_str(
            hksv.BufferEventCommand(
                command=hksv.BufferEventCommandType.ACKNOWLEDGE, sequence_number=2
            ).encode()
        )
    )
    assert [e.sequence_number for e in camera._buffer_events] == [3]


def test_buffer_upload_command_hook(buffer_camera):
    camera = buffer_camera
    received = []

    def _upload(command):
        received.append(command)
        return 77

    camera.buffer_upload = _upload
    camera.set_buffer_upload_command(
        to_base64_str(
            hksv.BufferUploadCommand(
                session_id=5,
                command=hksv.BufferCommand.START_AND_STOP,
                start=1000,
                stop=5000,
                stop_action=hksv.BufferStopAction.FINALIZE,
            ).encode()
        )
    )
    assert received[0].stop_action is hksv.BufferStopAction.FINALIZE
    clip_id = hksv.decode_buffer_upload_response(
        base64_to_bytes(_buffer_char(camera, "BufferUploadCommand").get_value())
    )
    assert clip_id == 77


def test_buffer_activity_and_publishing_point(buffer_camera):
    camera = buffer_camera
    seen = []
    camera.buffer_activity = seen.append
    camera.set_buffer_activity_command(
        to_base64_str(
            hksv.BufferActivityCommand(
                start=1000,
                duration_ms=30000,
                activity=hksv.BufferActivity.SHOULD_NOT_RECORD,
            ).encode()
        )
    )
    assert seen[0].activity is hksv.BufferActivity.SHOULD_NOT_RECORD

    points = []
    camera.publishing_point_updated = points.append
    point = hksv.PublishingPoint(
        url="https://hub.local:8443/ingest/",
        server_ca_certificates=[b"\x30\x82\x01\x00"],
    )
    camera.set_recording_publishing_point(to_base64_str(point.encode()))
    assert camera.publishing_point == point
    assert points == [point]


def test_camera_key_management(buffer_camera):
    camera = buffer_camera
    received = []
    camera.camera_key_received = received.append

    camera.set_camera_key(
        to_base64_str(hksv.CameraKey(key=b"\x42" * 32, key_number=7).encode())
    )
    assert received[0].key_number == 7
    assert camera._camera_keys[7].key == b"\x42" * 32

    key_id_value = (
        camera.get_service("CameraKeyManagement")
        .get_characteristic("CameraKeyID")
        .get_value()
    )
    assert hksv.decode_camera_key_id(base64_to_bytes(key_id_value)) == 7


def test_client_csr_default_implementation(buffer_camera):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    camera = buffer_camera
    nonce = b"\x11" * 32
    camera.set_camera_client_csr(to_base64_str(tlv.encode(b"\x01", nonce)))

    response = base64_to_bytes(
        camera.get_service("CameraClientCertificateManagement")
        .get_characteristic("CameraClientCSR")
        .get_value()
    )
    d = hksv._decode(response)
    csr = x509.load_der_x509_csr(d[1])
    assert csr.is_signature_valid
    assert len(d[2]) <= 128
    # The nonce signature verifies against the CSR's public key.
    csr.public_key().verify(d[2], nonce, ec.ECDSA(hashes.SHA256()))


def test_client_certificate_provisioning(buffer_camera):
    camera = buffer_camera
    provisioned = []
    camera.client_certificate_received = provisioned.append

    camera.set_certificate_needs_update(True)
    status_char = camera.get_service(
        "CameraClientCertificateManagement"
    ).get_characteristic("CameraClientCertificateStatus")
    assert hksv.decode_certificate_status(base64_to_bytes(status_char.get_value()))

    cert = hksv.ClientCertificate(certificate=b"\x30\x82CERT", ca=b"\x30\x82CA")
    camera.set_camera_client_certificate(to_base64_str(cert.encode()))
    assert camera.client_certificate == cert
    assert provisioned == [cert]
    # Provisioning a certificate clears the needs-update flag.
    assert not hksv.decode_certificate_status(base64_to_bytes(status_char.get_value()))


def test_recording_management_default_inactive(multi_tier_camera):
    from pyhap import hksv_recording as hr

    camera = multi_tier_camera
    service = camera.get_service("CameraRecordingManagement")
    assert service.get_characteristic("Active").get_value() == 0

    supported = hr.SupportedRecordingConfiguration.decode(
        base64_to_bytes(
            service.get_characteristic(
                "SupportedCameraRecordingConfiguration"
            ).get_value()
        )
    )
    assert supported.media_containers[0].container_type is (
        hr.MediaContainerType.FRAGMENTED_MP4
    )
    video = hr.decode_supported_video(
        base64_to_bytes(
            service.get_characteristic(
                "SupportedVideoRecordingConfiguration"
            ).get_value()
        )
    )
    assert video[0].codec_type is hr.VideoCodecType.H265
    assert video[0].attributes[0].width == 3840


def test_selected_recording_configuration_hook(multi_tier_camera):
    from pyhap import hksv_recording as hr

    camera = multi_tier_camera
    selected_seen = []
    camera.recording_configuration_selected = selected_seen.append

    selected = hr.SelectedRecordingConfiguration(
        recording=hr.SupportedRecordingConfiguration(),
        video=hr.VideoCodecConfiguration(
            codec_type=hr.VideoCodecType.H265,
            profile=2,
            level=2,
            bitrate_kbps=2000,
            iframe_interval_ms=4000,
            attributes=[hr.VideoAttributes(1920, 1080, 30)],
        ),
        audio=hr.AudioCodecConfiguration(),
    )
    camera.set_selected_recording_configuration(to_base64_str(selected.encode()))
    assert selected_seen == [selected]
    assert camera.selected_recording_configuration == selected


def test_setter_receives_client_addr():
    """A two-arg setter opts in to the sender's client address."""
    from pyhap.characteristic import _setter_wants_client_addr

    assert _setter_wants_client_addr(lambda value: None) is False
    assert _setter_wants_client_addr(lambda value, addr: None) is True
    assert _setter_wants_client_addr(lambda *args: None) is False


@pytest.fixture(name="recording_camera")
def recording_camera_fixture():
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
        }
        camera = Camera(options, driver, "Cam")
        camera.driver = driver
        return camera


def test_data_stream_transport_service_present(recording_camera):
    service = recording_camera.get_service("DataStreamTransportManagement")
    assert service.get_characteristic("Version").get_value() == "1.0"
    assert (
        service.get_characteristic(
            "SupportedDataStreamTransportConfiguration"
        ).get_value()
        is not None
    )


def test_setup_data_stream_transport_without_session_errors(recording_camera):
    from pyhap import hds
    from pyhap.util import base64_to_bytes

    camera = recording_camera
    # No session key for this client, and no listener started.
    setup = hds.encode_setup_response  # noqa: F841
    request = hds.tlv.encode(
        hds.SETUP_TYPES["CONTROLLER_KEY_SALT"], b"\x11" * 32
    )
    camera.set_data_stream_transport(
        to_base64_str(request), sender_client_addr=("10.0.0.9", 5000)
    )
    value = (
        camera.get_service("DataStreamTransportManagement")
        .get_characteristic("SetupDataStreamTransport")
        .get_value()
    )
    objs = hds.tlv.decode(base64_to_bytes(value))
    assert objs[hds.SETUP_RESPONSE_TYPES["STATUS"]] == hds.SETUP_STATUS_GENERIC_ERROR


@pytest.mark.asyncio
async def test_setup_data_stream_transport_with_session(recording_camera):
    import os

    from pyhap import hds
    from pyhap.util import base64_to_bytes

    camera = recording_camera
    await camera._ensure_hds_listener()
    try:
        client = ("10.0.0.9", 5000)
        camera.driver.session_shared_keys[client] = os.urandom(32)
        request = hds.tlv.encode(
            hds.SETUP_TYPES["CONTROLLER_KEY_SALT"], os.urandom(32)
        )
        camera.set_data_stream_transport(
            to_base64_str(request), sender_client_addr=client
        )
        value = (
            camera.get_service("DataStreamTransportManagement")
            .get_characteristic("SetupDataStreamTransport")
            .get_value()
        )
        objs = hds.tlv.decode(base64_to_bytes(value))
        assert objs[hds.SETUP_RESPONSE_TYPES["STATUS"]] == hds.SETUP_STATUS_SUCCESS
        assert objs[hds.SETUP_RESPONSE_TYPES["ACCESSORY_KEY_SALT"]]
    finally:
        await camera.stop()
