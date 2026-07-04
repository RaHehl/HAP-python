"""Tests for the HKSV 17.99 TLV codecs."""

from uuid import UUID

from pyhap import hksv, tlv


SENSOR_UUID = UUID("12345678-1234-5678-1234-567812345678").bytes


def test_video_stream_tiers_roundtrip():
    tiers = hksv.SupportedVideoStreamTiers(
        codec=hksv.VideoCodecType.H265,
        payload_type=99,
        tiers=[
            hksv.VideoStreamTier(1, hksv.VideoQuality.HIGH, 4500, 3840, 2160, 30),
            hksv.VideoStreamTier(2, hksv.VideoQuality.MEDIUM, 1700, 1920, 1080, 30),
            hksv.VideoStreamTier(3, hksv.VideoQuality.LOW, 180, 640, 360, 15),
        ],
    )
    decoded = hksv.SupportedVideoStreamTiers.decode(tiers.encode())
    assert decoded == tiers
    assert len(decoded.tiers) == 3
    assert decoded.tiers[0].width == 3840


def test_audio_stream_tiers_roundtrip():
    tiers = hksv.SupportedAudioStreamTiers(
        codec=hksv.AudioCodecType.OPUS,
        payload_type=110,
        tiers=[
            hksv.AudioStreamTier(
                identifier=1,
                target_average_bitrate=32000,
                sample_rate=hksv.AudioSampleRate.KHZ_48,
                bit_depth=hksv.AudioBitDepth.BIT_16,
            )
        ],
    )
    decoded = hksv.SupportedAudioStreamTiers.decode(tiers.encode())
    assert decoded == tiers
    assert decoded.tiers[0].packet_time_ms == 20
    assert decoded.tiers[0].number_of_channels == 1


def test_camera_capabilities_roundtrip():
    caps = hksv.CameraCapabilities(
        sensors=[
            hksv.SensorConfiguration(
                width=3840,
                height=2160,
                sensor_uuid=SENSOR_UUID,
                sensor_intent=hksv.SensorIntent.MAIN,
                video_stream_capabilities=[
                    hksv.VideoStreamCapability(
                        identifier=UUID(int=1).bytes,
                        video_quality=hksv.VideoQuality.HIGH,
                        width=3840,
                        height=2160,
                        frames_per_second=30,
                        average_bit_rate_kbps=4500,
                        peak_bit_rate_kbps=5000,
                    ),
                    hksv.VideoStreamCapability(
                        identifier=UUID(int=2).bytes,
                        video_quality=hksv.VideoQuality.LOW,
                        width=640,
                        height=360,
                        frames_per_second=15,
                        average_bit_rate_kbps=180,
                        peak_bit_rate_kbps=190,
                    ),
                ],
            ),
            hksv.SensorConfiguration(
                width=1600,
                height=1200,
                sensor_uuid=UUID(int=3).bytes,
                sensor_intent=hksv.SensorIntent.PACKAGE,
            ),
        ]
    )
    decoded = hksv.CameraCapabilities.decode(caps.encode())
    assert decoded == caps
    assert decoded.sensors[1].sensor_intent is hksv.SensorIntent.PACKAGE


def test_contributing_sensors_roundtrip():
    sensors = hksv.ContributingSensors(sensor_uuids=[SENSOR_UUID, UUID(int=9).bytes])
    assert hksv.ContributingSensors.decode(sensors.encode()) == sensors


def test_camera_key_roundtrip():
    key = hksv.CameraKey(key=b"\x42" * 32, key_number=7)
    assert hksv.CameraKey.decode(key.encode()) == key
    assert hksv.decode_camera_key_id(hksv.encode_camera_key_id(7)) == 7


def test_buffer_upload_command_roundtrip():
    cmd = hksv.BufferUploadCommand(
        session_id=123456789,
        command=hksv.BufferCommand.START_AND_STOP,
        start=1700000000000,
        stop=1700000004000,
        stop_action=hksv.BufferStopAction.FINALIZE,
    )
    assert hksv.BufferUploadCommand.decode(cmd.encode()) == cmd

    minimal = hksv.BufferUploadCommand(
        session_id=1, command=hksv.BufferCommand.STOP
    )
    decoded = hksv.BufferUploadCommand.decode(minimal.encode())
    assert decoded.start is None
    assert decoded.stop_action is None

    assert hksv.decode_buffer_upload_response(
        hksv.encode_buffer_upload_response(42)
    ) == 42


def test_buffer_activity_command_roundtrip():
    cmd = hksv.BufferActivityCommand(
        start=1700000000000,
        duration_ms=30000,
        activity=hksv.BufferActivity.SHOULD_RECORD,
    )
    assert hksv.BufferActivityCommand.decode(cmd.encode()) == cmd


def test_buffer_events_roundtrip():
    events = [
        hksv.BufferEvent(
            sequence_number=1,
            type=hksv.BufferEventType.CMAF_SESSION_START,
            cmaf_session_id=5,
        ),
        hksv.BufferEvent(
            sequence_number=2,
            type=hksv.BufferEventType.MOTION,
            motion_active=True,
        ),
        hksv.BufferEvent(
            sequence_number=3,
            type=hksv.BufferEventType.CMAF_ERROR,
            cmaf_session_id=5,
            cmaf_error=hksv.CMAFError.HTTP_CERTIFICATE_EXPIRED,
        ),
        hksv.BufferEvent(
            sequence_number=4,
            type=hksv.BufferEventType.CMAF_SESSION_STOP,
            cmaf_session_id=5,
        ),
    ]
    decoded = hksv.decode_buffer_events_response(
        hksv.encode_buffer_events_response(events)
    )
    assert decoded == events
    assert decoded[2].cmaf_error is hksv.CMAFError.HTTP_CERTIFICATE_EXPIRED

    cmd = hksv.BufferEventCommand(
        command=hksv.BufferEventCommandType.QUERY, sequence_number=1, limit=10
    )
    assert hksv.BufferEventCommand.decode(cmd.encode()) == cmd


def test_publishing_point_roundtrip():
    point = hksv.PublishingPoint(
        url="https://hub.local:8443/ingest/",
        server_ca_certificates=[b"\x30\x82\x01\x00", b"\x30\x82\x02\x00"],
    )
    assert hksv.PublishingPoint.decode(point.encode()) == point


def test_camera_zones_roundtrip():
    zones = hksv.CameraZones(
        method=hksv.ZoneApplicationMethod.NORMAL,
        polygons=[
            hksv.ZonePolygon(
                identifier=UUID(int=1).bytes,
                vertices=[(0, 0), (3840, 0), (3840, 2160), (0, 2160)],
            ),
            hksv.ZonePolygon(
                identifier=UUID(int=2).bytes,
                vertices=[(100, 100), (200, 100), (150, 200)],
            ),
        ],
    )
    decoded = hksv.CameraZones.decode(zones.encode())
    assert decoded == zones
    assert decoded.zone_data_version == 2
    assert decoded.polygons[0].vertices[2] == (3840, 2160)


def test_zone_vertices_little_endian():
    polygon = hksv.ZonePolygon(identifier=UUID(int=1).bytes, vertices=[(0x1234, 0x5678)])
    encoded = polygon.encode()
    raw = hksv._decode(encoded)[3]
    assert raw == b"\x34\x12\x78\x56"


def test_rtp_streaming_control_roundtrip():
    write = hksv.RTPStreamingControlWrite(
        session_identifier=UUID(int=7).bytes,
        command=hksv.RTPStreamingCommand.START,
        video_tier=1,
        video_ssrc=0xDEADBEEF,
        audio_tier=1,
        audio_ssrc=0xC0FFEE,
    )
    assert hksv.RTPStreamingControlWrite.decode(write.encode()) == write

    end = hksv.RTPStreamingControlWrite(
        session_identifier=UUID(int=7).bytes,
        command=hksv.RTPStreamingCommand.END,
    )
    assert hksv.RTPStreamingControlWrite.decode(end.encode()).video_tier is None

    status = hksv.encode_rtp_streaming_status(
        UUID(int=7).bytes, hksv.RTPStreamingStatus.SUCCESS
    )
    decoded = hksv._decode(status)
    assert decoded[1] == UUID(int=7).bytes
    assert hksv.RTPStreamingStatus(hksv._int(decoded[2])) is (
        hksv.RTPStreamingStatus.SUCCESS
    )


def test_webrtc_solicit_offer_roundtrip():
    write = hksv.WebRTCSolicitOfferWrite(sframe_enabled=True)
    assert hksv.WebRTCSolicitOfferWrite.decode(write.encode()).sframe_enabled

    response = hksv.WebRTCSolicitOfferResponse(
        session_identifier=UUID(int=11).bytes,
        status=hksv.WebRTCOfferStatus.SUCCESS,
        sdp_offer="v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n",
        additional_candidates=[
            hksv.WebRTCICECandidate("candidate:1 1 UDP 1 10.0.0.1 5000 typ host", "0", 0),
            hksv.WebRTCICECandidate("candidate:2 1 TCP 2 10.0.0.1 5001 typ host"),
        ],
        sframe_configuration=hksv.SFrameKeyData(key=b"\x01" * 16, kid=1),
    )
    decoded = hksv.WebRTCSolicitOfferResponse.decode(response.encode())
    assert decoded == response
    assert decoded.additional_candidates[1].sdp_mid is None

    privacy = hksv.WebRTCSolicitOfferResponse(
        session_identifier=UUID(int=12).bytes,
        status=hksv.WebRTCOfferStatus.PRIVACY_MODE_ACTIVE,
    )
    assert (
        hksv.WebRTCSolicitOfferResponse.decode(privacy.encode()).status
        is hksv.WebRTCOfferStatus.PRIVACY_MODE_ACTIVE
    )


def test_webrtc_provide_answer_roundtrip():
    write = hksv.WebRTCProvideAnswerWrite(
        session_identifier=UUID(int=11).bytes,
        sdp_answer="v=0\r\n",
        additional_candidates=[
            hksv.WebRTCICECandidate("candidate:1 1 UDP 1 10.0.0.2 6000 typ host")
        ],
    )
    assert hksv.WebRTCProvideAnswerWrite.decode(write.encode()) == write

    session, status = hksv.decode_webrtc_status(
        hksv.encode_webrtc_status(
            UUID(int=11).bytes, hksv.WebRTCStreamingStatus.SUCCESS
        )
    )
    assert session == UUID(int=11).bytes
    assert status is hksv.WebRTCStreamingStatus.SUCCESS


def test_webrtc_reoffer_and_update_session_roundtrip():
    reoffer = hksv.WebRTCReofferWrite(
        session_identifier=UUID(int=11).bytes,
        sdp_offer="v=0\r\n",
        sframe_enabled=True,
    )
    assert hksv.WebRTCReofferWrite.decode(reoffer.encode()) == reoffer

    update = hksv.WebRTCUpdateSessionWrite(
        session_identifier=UUID(int=11).bytes,
        receive_keys_to_add=[
            hksv.SFrameKeyData(key=b"\x02" * 16, kid=2),
            hksv.SFrameKeyData(key=b"\x03" * 16, kid=3),
        ],
        receive_kids_to_remove=[1],
    )
    assert hksv.WebRTCUpdateSessionWrite.decode(update.encode()) == update


def test_client_certificate_roundtrip():
    nonce = b"\x11" * 32
    assert hksv.decode_client_csr_write(tlv.encode(b"\x01", nonce)) == nonce

    response = hksv.encode_client_csr_response(b"\x30\x82CSR", b"\x30\x45SIG")
    d = hksv._decode(response)
    assert d[1] == b"\x30\x82CSR"
    assert d[2] == b"\x30\x45SIG"

    cert = hksv.ClientCertificate(certificate=b"\x30\x82CERT", ca=b"\x30\x82CA")
    assert hksv.ClientCertificate.decode(cert.encode()) == cert

    assert hksv.decode_certificate_status(hksv.encode_certificate_status(True)) is True
    assert hksv.decode_certificate_status(hksv.encode_certificate_status(False)) is False


def test_long_value_fragmentation_roundtrip():
    """SDP payloads exceed 255 bytes and must survive TLV fragmentation."""
    sdp = "v=0\r\n" + "a=candidate:%s\r\n" % ("x" * 400)
    response = hksv.WebRTCSolicitOfferResponse(
        session_identifier=UUID(int=1).bytes,
        status=hksv.WebRTCOfferStatus.SUCCESS,
        sdp_offer=sdp,
    )
    assert hksv.WebRTCSolicitOfferResponse.decode(response.encode()).sdp_offer == sdp


def test_rtp_streaming_control_unknown_command_passthrough():
    """Undocumented controller commands (iOS 0x00 status poll) decode raw."""
    raw = tlv.encode(b"\x01", UUID(int=7).bytes, b"\x02", b"\x00")
    decoded = hksv.RTPStreamingControlWrite.decode(raw)
    assert decoded.command == 0
    assert not isinstance(decoded.command, hksv.RTPStreamingCommand)


def test_recording_supported_config_roundtrip():
    from pyhap import hksv_recording as hr

    config = hr.SupportedRecordingConfiguration(
        prebuffer_length_ms=4000,
        event_triggers=hr.EventTrigger.MOTION | hr.EventTrigger.DOORBELL,
        media_containers=[hr.MediaContainerConfiguration(fragment_length_ms=4000)],
    )
    decoded = hr.SupportedRecordingConfiguration.decode(config.encode())
    assert decoded == config
    assert decoded.event_triggers == 3


def test_recording_supported_video_roundtrip():
    from pyhap import hksv_recording as hr

    configs = [
        hr.VideoCodecConfiguration(
            codec_type=hr.VideoCodecType.H265,
            profile=2,
            level=2,
            bitrate_kbps=2000,
            iframe_interval_ms=4000,
            attributes=[
                hr.VideoAttributes(3840, 2160, 30),
                hr.VideoAttributes(1920, 1080, 30),
            ],
        ),
        hr.VideoCodecConfiguration(
            codec_type=hr.VideoCodecType.H264,
            profile=0,
            level=0,
            bitrate_kbps=800,
            iframe_interval_ms=4000,
            attributes=[hr.VideoAttributes(1280, 720, 24)],
        ),
    ]
    decoded = hr.decode_supported_video(hr.encode_supported_video(configs))
    assert decoded == configs
    assert decoded[0].attributes[0].width == 3840


def test_recording_supported_audio_roundtrip():
    from pyhap import hksv_recording as hr

    configs = [
        hr.AudioCodecConfiguration(
            codec_type=hr.AudioCodecType.AAC_LC,
            channels=1,
            bitrate_mode=hr.BitRateMode.VARIABLE,
            sample_rate=hr.AudioSampleRate.KHZ_32,
            max_audio_bitrate_kbps=64,
        ),
        hr.AudioCodecConfiguration(
            codec_type=hr.AudioCodecType.AAC_ELD,
            sample_rate=hr.AudioSampleRate.KHZ_24,
        ),
    ]
    decoded = hr.decode_supported_audio(hr.encode_supported_audio(configs))
    assert decoded == configs


def test_recording_selected_config_roundtrip():
    from pyhap import hksv_recording as hr

    selected = hr.SelectedRecordingConfiguration(
        recording=hr.SupportedRecordingConfiguration(
            prebuffer_length_ms=4000, event_triggers=hr.EventTrigger.MOTION
        ),
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
    decoded = hr.SelectedRecordingConfiguration.decode(selected.encode())
    assert decoded == selected
