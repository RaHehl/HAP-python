"""Classic HomeKit Secure Video recording transfer over HDS ``dataSend``.

Once a controller has set up an HDS connection and completed the ``control``
handshake, it opens a ``dataSend`` stream of type ``ipcamera.recording``. The
accessory answers and then streams fragmented-MP4 recording fragments as
``dataSend``/``data`` events: the first fragment is the MP4 initialization
segment (``moov``), the rest are media fragments. Fragments larger than a chunk
are split across several events and reassembled by the controller using the
per-chunk metadata.

The fragments themselves come from a delegate: an async generator yielding
:class:`RecordingPacket` objects. This module owns the protocol; producing the
MP4 bytes is the accessory's job.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import IntEnum
import logging
from typing import AsyncIterator, Callable, Optional

from pyhap.hds_protocol import Message

logger = logging.getLogger("pyhap.hds")

_DATA_SEND = "dataSend"
_TYPE_RECORDING = "ipcamera.recording"
_TARGET_CONTROLLER = "controller"

# The controller reassembles chunks; this only bounds a single HDS frame.
DEFAULT_CHUNK_SIZE = 0x40000

MEDIA_INITIALIZATION = "mediaInitialization"
MEDIA_FRAGMENT = "mediaFragment"


class RecordingReason(IntEnum):
    NORMAL = 0
    NOT_ALLOWED = 1
    BUSY = 2
    CANCELLED = 3
    UNEXPECTED_FAILURE = 4
    TIMEOUT = 5
    BAD_DATA = 6
    PROTOCOL_ERROR = 7
    INVALID_CONFIGURATION = 8


@dataclass
class RecordingPacket:
    """One recording fragment produced by the delegate."""

    data: bytes
    is_last: bool = False


# A delegate is called with the stream id and yields RecordingPackets. The first
# yielded packet is the MP4 initialization segment.
RecordingDelegate = Callable[[int], AsyncIterator[RecordingPacket]]


class RecordingStreamManager:
    """Handle ``dataSend`` recording streams on one HDS connection."""

    def __init__(
        self,
        connection,
        delegate: RecordingDelegate,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._connection = connection
        self._delegate = delegate
        self._chunk_size = chunk_size
        self._stream_id: Optional[int] = None
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        connection.add_request_handler(_DATA_SEND, "open", self._handle_open)
        connection.add_request_handler(_DATA_SEND, "close", self._handle_close)
        connection.add_request_handler(_DATA_SEND, "ack", self._handle_ack)

    def _handle_open(self, message: Message):
        body = message.message
        if (
            body.get("target") != _TARGET_CONTROLLER
            or body.get("type") != _TYPE_RECORDING
        ):
            return RecordingReason.UNEXPECTED_FAILURE, {}
        if self._task is not None and not self._task.done():
            return RecordingReason.BUSY, {}
        self._stream_id = body["streamId"]
        self._closed = False
        self._task = asyncio.get_event_loop().create_task(self._stream())
        return RecordingReason.NORMAL, {}

    def _handle_close(self, message: Message):
        self._stop()
        return RecordingReason.NORMAL, {}

    def _handle_ack(self, message: Message):
        return RecordingReason.NORMAL, {}

    def _stop(self) -> None:
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _stream(self) -> None:
        try:
            sequence_number = 1
            async for packet in self._delegate(self._stream_id):
                if self._closed:
                    break
                self._send_fragment(
                    packet.data,
                    sequence_number,
                    initialization=sequence_number == 1,
                    is_last=packet.is_last,
                )
                if packet.is_last:
                    break
                sequence_number += 1
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("Recording stream %s failed", self._stream_id, exc_info=True)

    def _send_fragment(
        self, fragment: bytes, sequence_number: int, initialization: bool, is_last: bool
    ) -> None:
        offset = 0
        chunk_sequence_number = 1
        total = len(fragment)
        # A zero-length fragment still needs a single event to carry its metadata.
        while True:
            chunk = fragment[offset : offset + self._chunk_size]
            offset += len(chunk)
            last_chunk = offset >= total
            metadata = {
                "dataType": MEDIA_INITIALIZATION if initialization else MEDIA_FRAGMENT,
                "dataSequenceNumber": sequence_number,
                "dataChunkSequenceNumber": chunk_sequence_number,
                "isLastDataChunk": last_chunk,
            }
            if chunk_sequence_number == 1:
                metadata["dataTotalSize"] = total
            event = {
                "streamId": self._stream_id,
                "packets": [{"data": chunk, "metadata": metadata}],
            }
            if last_chunk and is_last:
                event["endOfStream"] = True
            self._connection.send_event(_DATA_SEND, "data", event)
            chunk_sequence_number += 1
            if last_chunk:
                break
