"""Tests for the HDS dataSend recording transfer."""

import asyncio
import os

import pytest

from pyhap import hds, hds_recording, hds_server
from pyhap.hds_protocol import EVENT, HDSStatus, REQUEST, RESPONSE, Message


def _paired(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    accessory_salt = listener.register_transport(secret, salt)
    acc_read, acc_write = hds.derive_keys(secret, salt, accessory_salt)
    return hds.HDSCrypto(acc_write, acc_read)


class _CaptureTransport(asyncio.Transport):
    def __init__(self):
        super().__init__()
        self.buffer = bytearray()

    def write(self, data):
        self.buffer += data

    def close(self):
        pass


def _drain_events(controller, transport):
    """Decrypt and return every buffered accessory->controller message."""
    buffer = bytearray(transport.buffer)
    transport.buffer = bytearray()
    messages = []
    while True:
        payload = controller.decrypt_frame(buffer)
        if payload is None:
            break
        messages.append(Message.decode(payload))
    return messages


async def _run_recording(delegate, chunk_size=hds_recording.DEFAULT_CHUNK_SIZE):
    listener = hds_server.HDSListener()
    controller = _paired(listener)
    connection = hds_server.HDSConnection(listener)
    transport = _CaptureTransport()
    connection.connection_made(transport)
    hds_recording.RecordingStreamManager(connection, delegate, chunk_size=chunk_size)

    # Bind with the control handshake.
    connection.data_received(
        controller.encrypt_frame(Message("control", REQUEST, "hello", {}, id=1).encode())
    )
    _drain_events(controller, transport)

    # Open the recording data stream.
    connection.data_received(
        controller.encrypt_frame(
            Message(
                "dataSend", REQUEST, "open",
                {"streamId": 99, "type": "ipcamera.recording", "target": "controller"},
                id=2,
            ).encode()
        )
    )
    # Let the streaming task run.
    await asyncio.sleep(0.05)
    return controller, transport


@pytest.mark.asyncio
async def test_recording_stream_open_and_fragments():
    async def delegate(stream_id):
        assert stream_id == 99
        yield hds_recording.RecordingPacket(b"moov-init-segment")
        yield hds_recording.RecordingPacket(b"fragment-1")
        yield hds_recording.RecordingPacket(b"fragment-2", is_last=True)

    controller, transport = await _run_recording(delegate)
    messages = _drain_events(controller, transport)

    # First message is the open response, then three data events.
    assert messages[0].kind == RESPONSE
    assert messages[0].topic == "open"
    assert messages[0].status == HDSStatus.SUCCESS

    data_events = [m for m in messages if m.kind == EVENT and m.topic == "data"]
    assert len(data_events) == 3
    metas = [e.message["packets"][0]["metadata"] for e in data_events]
    assert metas[0]["dataType"] == hds_recording.MEDIA_INITIALIZATION
    assert metas[1]["dataType"] == hds_recording.MEDIA_FRAGMENT
    assert [m["dataSequenceNumber"] for m in metas] == [1, 2, 3]
    assert all(m["isLastDataChunk"] for m in metas)
    assert data_events[-1].message["endOfStream"] is True
    assert data_events[0].message["packets"][0]["data"] == b"moov-init-segment"


@pytest.mark.asyncio
async def test_recording_fragment_chunking():
    fragment = os.urandom(1000)

    async def delegate(stream_id):
        yield hds_recording.RecordingPacket(b"init")
        yield hds_recording.RecordingPacket(fragment, is_last=True)

    controller, transport = await _run_recording(delegate, chunk_size=256)
    messages = _drain_events(controller, transport)
    data_events = [m for m in messages if m.kind == EVENT and m.topic == "data"]

    # init (1 chunk) + fragment split into ceil(1000/256)=4 chunks.
    fragment_events = [
        e
        for e in data_events
        if e.message["packets"][0]["metadata"]["dataSequenceNumber"] == 2
    ]
    assert len(fragment_events) == 4
    metas = [e.message["packets"][0]["metadata"] for e in fragment_events]
    assert [m["dataChunkSequenceNumber"] for m in metas] == [1, 2, 3, 4]
    # dataTotalSize only on the first chunk.
    assert metas[0]["dataTotalSize"] == 1000
    assert "dataTotalSize" not in metas[1]
    assert [m["isLastDataChunk"] for m in metas] == [False, False, False, True]
    # Reassembling the chunks reproduces the fragment.
    reassembled = b"".join(e.message["packets"][0]["data"] for e in fragment_events)
    assert reassembled == fragment
    # endOfStream only on the very last chunk.
    assert fragment_events[-1].message["endOfStream"] is True
    assert "endOfStream" not in fragment_events[0].message


@pytest.mark.asyncio
async def test_recording_open_rejects_wrong_type():
    async def delegate(stream_id):
        yield hds_recording.RecordingPacket(b"x", is_last=True)

    listener = hds_server.HDSListener()
    controller = _paired(listener)
    connection = hds_server.HDSConnection(listener)
    transport = _CaptureTransport()
    connection.connection_made(transport)
    hds_recording.RecordingStreamManager(connection, delegate)

    connection.data_received(
        controller.encrypt_frame(Message("control", REQUEST, "hello", {}, id=1).encode())
    )
    _drain_events(controller, transport)

    connection.data_received(
        controller.encrypt_frame(
            Message(
                "dataSend", REQUEST, "open",
                {"streamId": 1, "type": "wrong", "target": "controller"},
                id=2,
            ).encode()
        )
    )
    await asyncio.sleep(0.02)
    response = _drain_events(controller, transport)[0]
    assert response.status == HDSStatus.PROTOCOL_SPECIFIC_ERROR
    assert response.message["status"] == hds_recording.RecordingReason.UNEXPECTED_FAILURE


@pytest.mark.asyncio
async def test_recording_close_stops_stream():
    started = asyncio.Event()
    release = asyncio.Event()

    async def delegate(stream_id):
        yield hds_recording.RecordingPacket(b"init")
        started.set()
        await release.wait()  # block until the controller closes
        yield hds_recording.RecordingPacket(b"never", is_last=True)

    listener = hds_server.HDSListener()
    controller = _paired(listener)
    connection = hds_server.HDSConnection(listener)
    transport = _CaptureTransport()
    connection.connection_made(transport)
    hds_recording.RecordingStreamManager(connection, delegate)

    connection.data_received(
        controller.encrypt_frame(Message("control", REQUEST, "hello", {}, id=1).encode())
    )
    _drain_events(controller, transport)
    connection.data_received(
        controller.encrypt_frame(
            Message(
                "dataSend", REQUEST, "open",
                {"streamId": 7, "type": "ipcamera.recording", "target": "controller"},
                id=2,
            ).encode()
        )
    )
    await started.wait()
    _drain_events(controller, transport)

    connection.data_received(
        controller.encrypt_frame(
            Message("dataSend", REQUEST, "close", {"streamId": 7}, id=3).encode()
        )
    )
    await asyncio.sleep(0.02)
    messages = _drain_events(controller, transport)
    assert messages[0].topic == "close"
    assert messages[0].status == HDSStatus.SUCCESS
    # The blocked "never" fragment was never sent.
    assert not [m for m in messages if m.kind == EVENT]
