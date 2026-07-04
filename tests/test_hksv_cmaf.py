"""Tests for the CMAF ingest orchestration."""

import pytest

from pyhap import hksv_cmaf
from pyhap.hksv import BufferEventType, CMAFError


class _FakeCamera:
    def __init__(self):
        self.events = []
        self._seq = 0

    def queue_buffer_event(
        self, event_type, cmaf_session_id=None, motion_active=None, cmaf_error=None
    ):
        self._seq += 1
        self.events.append(
            (event_type, cmaf_session_id, cmaf_error)
        )
        return self._seq


class _RecordingUploader:
    def __init__(self, fail_on=None, reason=None, raise_generic=False):
        self.segments = []
        self.closed = False
        self._fail_on = fail_on
        self._reason = reason
        self._raise_generic = raise_generic

    async def upload_segment(self, segment, is_initialization):
        self.segments.append((segment, is_initialization))
        if self._fail_on is not None and len(self.segments) == self._fail_on:
            if self._raise_generic:
                raise RuntimeError("socket blew up")
            raise hksv_cmaf.CMAFIngestError(self._reason)

    async def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, CMAFError.NONE),
        (204, CMAFError.NONE),
        (400, CMAFError.HTTP_BAD_REQUEST),
        (401, CMAFError.HTTP_INVALID_TOKEN),
        (404, CMAFError.HTTP_NOT_FOUND),
        (415, CMAFError.HTTP_UNSUPPORTED_MEDIA_TYPE),
        (503, CMAFError.HTTP_SERVICE_UNAVAILABLE),
        (418, CMAFError.UNKNOWN),
    ],
)
def test_http_status_mapping(status, expected):
    assert hksv_cmaf.map_http_status_to_error(status) is expected


@pytest.mark.asyncio
async def test_successful_session_lifecycle():
    camera = _FakeCamera()
    uploader = _RecordingUploader()
    session = hksv_cmaf.CMAFIngestSession(camera, uploader, session_id=42)

    session.start()
    assert await session.push(b"init", is_initialization=True) is True
    assert await session.push(b"frag1") is True
    assert await session.push(b"frag2") is True
    await session.stop()

    assert uploader.segments == [(b"init", True), (b"frag1", False), (b"frag2", False)]
    assert uploader.closed is True
    assert session.error is None
    assert camera.events == [
        (BufferEventType.CMAF_SESSION_START, 42, None),
        (BufferEventType.CMAF_SESSION_STOP, 42, None),
    ]


@pytest.mark.asyncio
async def test_session_start_is_idempotent():
    camera = _FakeCamera()
    session = hksv_cmaf.CMAFIngestSession(camera, _RecordingUploader(), 1)
    session.start()
    session.start()
    assert len(camera.events) == 1


@pytest.mark.asyncio
async def test_upload_failure_ends_session_with_reason():
    camera = _FakeCamera()
    uploader = _RecordingUploader(fail_on=2, reason=CMAFError.HTTP_CERTIFICATE_EXPIRED)
    session = hksv_cmaf.CMAFIngestSession(camera, uploader, session_id=7)

    session.start()
    assert await session.push(b"init", is_initialization=True) is True
    assert await session.push(b"frag1") is False  # fails here
    # Further pushes are refused.
    assert await session.push(b"frag2") is False

    assert session.error is CMAFError.HTTP_CERTIFICATE_EXPIRED
    assert uploader.closed is True
    assert camera.events == [
        (BufferEventType.CMAF_SESSION_START, 7, None),
        (BufferEventType.CMAF_ERROR, 7, CMAFError.HTTP_CERTIFICATE_EXPIRED),
    ]
    # A later stop does not emit a second terminal event.
    await session.stop()
    assert len(camera.events) == 2


@pytest.mark.asyncio
async def test_generic_transport_failure_maps_to_connection_failed():
    camera = _FakeCamera()
    uploader = _RecordingUploader(fail_on=1, raise_generic=True)
    session = hksv_cmaf.CMAFIngestSession(camera, uploader, session_id=3)
    session.start()
    assert await session.push(b"init", is_initialization=True) is False
    assert session.error is CMAFError.CONNECTION_FAILED
    assert camera.events[-1] == (
        BufferEventType.CMAF_ERROR, 3, CMAFError.CONNECTION_FAILED
    )
