"""CMAF ingest orchestration for the 17.99 recording data path.

In the preview spec a camera records by pushing CMAF (fragmented-MP4) segments
to a publishing point over HTTPS with mutual TLS: the client certificate comes
from the certificate provisioning flow, the server is trusted via the CAs in the
publishing point, and progress and failures are reported back to the controller
through the camera buffer event queue.

Following the same split as the rest of this package, pyhap owns the protocol
orchestration - session lifecycle, HTTP-status-to-error-reason mapping and the
buffer events - while the concrete HTTPS/mTLS transport (aiohttp, httpx, ...) is
supplied by the accessory as a :class:`CMAFUploader`. This keeps the library
dependency-free and lets the integration reuse its own HTTP stack.
"""

from __future__ import annotations

from typing import Optional, Protocol

from pyhap.hksv import BufferEventType, CMAFError


def map_http_status_to_error(status: int) -> CMAFError:
    """Map an HTTP response status to the CMAF error reported to the controller."""
    if 200 <= status < 300:
        return CMAFError.NONE
    return {
        400: CMAFError.HTTP_BAD_REQUEST,
        401: CMAFError.HTTP_INVALID_TOKEN,
        403: CMAFError.HTTP_BLOCKED,
        404: CMAFError.HTTP_NOT_FOUND,
        409: CMAFError.HTTP_MISMATCHED_TOKEN,
        415: CMAFError.HTTP_UNSUPPORTED_MEDIA_TYPE,
        500: CMAFError.HTTP_INTERNAL_SERVER_ERROR,
        503: CMAFError.HTTP_SERVICE_UNAVAILABLE,
    }.get(status, CMAFError.UNKNOWN)


class CMAFIngestError(Exception):
    """Raised by a :class:`CMAFUploader` to report a specific CMAF failure."""

    def __init__(self, reason: CMAFError) -> None:
        super().__init__(reason.name)
        self.reason = reason


class CMAFUploader(Protocol):
    """The concrete HTTPS/mTLS transport that pushes CMAF segments.

    Implementations open a connection to the publishing point (trusting its
    server CAs and presenting the provisioned client certificate) and upload
    each segment. Failures raise :class:`CMAFIngestError` with a CMAF reason;
    HTTP responses may instead use :func:`map_http_status_to_error`.
    """

    async def upload_segment(self, segment: bytes, is_initialization: bool) -> None:
        """Upload one CMAF segment; raise CMAFIngestError on failure."""

    async def close(self) -> None:
        """Release any connection resources."""


class CMAFIngestSession:
    """Drive one CMAF ingest recording session and its buffer events.

    Emits a CMAF-session-start event when the session opens, pushes segments
    through the uploader, and emits a CMAF-error or CMAF-session-stop event when
    it ends - the sequence numbers and delivery are handled by the camera's
    buffer event queue.
    """

    def __init__(self, camera, uploader: CMAFUploader, session_id: int) -> None:
        self._camera = camera
        self._uploader = uploader
        self._session_id = session_id
        self._started = False
        self._closed = False
        self._error: Optional[CMAFError] = None

    @property
    def error(self) -> Optional[CMAFError]:
        """The CMAF error that ended the session, if any."""
        return self._error

    def start(self) -> None:
        """Announce the CMAF session to the controller."""
        if self._started:
            return
        self._started = True
        self._camera.queue_buffer_event(
            BufferEventType.CMAF_SESSION_START, cmaf_session_id=self._session_id
        )

    async def push(self, segment: bytes, is_initialization: bool = False) -> bool:
        """Upload one segment; on failure end the session with a CMAF error.

        Returns ``True`` when the segment was accepted and ``False`` when the
        session has failed and should no longer be used.
        """
        if self._closed:
            return False
        try:
            await self._uploader.upload_segment(segment, is_initialization)
        except CMAFIngestError as err:
            await self._fail(err.reason)
            return False
        except Exception:  # noqa: BLE001 - any transport failure ends the session
            await self._fail(CMAFError.CONNECTION_FAILED)
            return False
        return True

    async def stop(self) -> None:
        """Finish the session normally and emit the stop event."""
        if self._closed:
            return
        self._closed = True
        await self._uploader.close()
        self._camera.queue_buffer_event(
            BufferEventType.CMAF_SESSION_STOP, cmaf_session_id=self._session_id
        )

    async def _fail(self, reason: CMAFError) -> None:
        if self._closed:
            return
        self._closed = True
        self._error = reason
        await self._uploader.close()
        self._camera.queue_buffer_event(
            BufferEventType.CMAF_ERROR,
            cmaf_session_id=self._session_id,
            cmaf_error=reason,
        )
