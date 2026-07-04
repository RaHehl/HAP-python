"""Tests for the HDS TCP listener and connection dispatch."""

import asyncio
import os

import pytest

from pyhap import hds, hds_server
from pyhap.hds_protocol import EVENT, HDSStatus, REQUEST, RESPONSE, Message


class _CaptureTransport(asyncio.Transport):
    """An in-memory transport capturing everything written to it."""

    def __init__(self):
        super().__init__()
        self.buffer = bytearray()
        self._closed = False

    def write(self, data):
        self.buffer += data

    def close(self):
        self._closed = True

    def take(self):
        data = bytes(self.buffer)
        self.buffer = bytearray()
        return data


def _controller_side(listener, shared_secret, controller_salt):
    """Register a transport and return the controller-side crypto for it."""
    accessory_salt = listener.register_transport(shared_secret, controller_salt)
    acc_read, acc_write = hds.derive_keys(
        shared_secret, controller_salt, accessory_salt
    )
    # The controller reads what the accessory writes and vice versa.
    return hds.HDSCrypto(acc_write, acc_read)


def _connect(listener):
    connection = hds_server.HDSConnection(listener)
    transport = _CaptureTransport()
    connection.connection_made(transport)
    return connection, transport


def _feed(connection, controller, message: Message):
    connection.data_received(controller.encrypt_frame(message.encode()))


def _read(controller, transport):
    buffer = bytearray(transport.take())
    payload = controller.decrypt_frame(buffer)
    return Message.decode(payload) if payload is not None else None


@pytest.fixture(name="listener")
def listener_fixture():
    return hds_server.HDSListener()


def test_control_hello_handshake(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    controller = _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)

    _feed(connection, controller, Message("control", REQUEST, "hello", {}, id=1))
    response = _read(controller, transport)
    assert response.kind == RESPONSE
    assert response.topic == "hello"
    assert response.id == 1
    assert response.status == 0


def test_unknown_request_returns_unsupported(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    controller = _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)
    # Bind first with a hello.
    _feed(connection, controller, Message("control", REQUEST, "hello", {}, id=1))
    _read(controller, transport)  # consume the hello response, keep counters in sync

    _feed(connection, controller, Message("dataSend", REQUEST, "mystery", {}, id=2))
    response = _read(controller, transport)
    assert response.status == HDSStatus.PROTOCOL_SPECIFIC_ERROR


def test_custom_request_handler(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    controller = _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)
    _feed(connection, controller, Message("control", REQUEST, "hello", {}, id=1))
    _read(controller, transport)  # consume the hello response, keep counters in sync

    seen = []

    def _handler(message):
        seen.append(message.message)
        return 0, {"streamId": 5}

    connection.add_request_handler("dataSend", "open", _handler)
    _feed(
        connection,
        controller,
        Message("dataSend", REQUEST, "open", {"type": "ipcamera.recording"}, id=7),
    )
    response = _read(controller, transport)
    assert seen == [{"type": "ipcamera.recording"}]
    assert response.status == 0
    assert response.message == {"streamId": 5}
    assert response.id == 7


def test_accessory_sends_events(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    controller = _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)
    _feed(connection, controller, Message("control", REQUEST, "hello", {}, id=1))
    _read(controller, transport)  # consume the hello response, keep counters in sync

    connection.send_event("dataSend", "data", {"data": b"fragment", "eos": False})
    event = _read(controller, transport)
    assert event.kind == EVENT
    assert event.topic == "data"
    assert event.message["data"] == b"fragment"


@pytest.mark.asyncio
async def test_accessory_request_resolves_on_response(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    controller = _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)
    _feed(connection, controller, Message("control", REQUEST, "hello", {}, id=1))
    _read(controller, transport)  # consume the hello response, keep counters in sync

    future = connection.send_request("dataSend", "open", {"streamId": 1})
    sent = _read(controller, transport)
    assert sent.kind == REQUEST
    assert sent.id is not None

    _feed(
        connection,
        controller,
        Message("dataSend", RESPONSE, "open", {"ok": True}, id=sent.id, status=0),
    )
    resolved = await asyncio.wait_for(future, 1)
    assert resolved.message == {"ok": True}


def test_binding_rejects_foreign_keys(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)

    # A controller with an unrelated secret cannot bind.
    foreign = hds.HDSCrypto(os.urandom(32), os.urandom(32))
    connection.data_received(
        foreign.encrypt_frame(Message("control", REQUEST, "hello", {}, id=1).encode())
    )
    assert connection._crypto is None
    assert transport.take() == b""


def test_partial_first_frame_waits(listener):
    secret, salt = os.urandom(32), os.urandom(32)
    controller = _controller_side(listener, secret, salt)
    connection, transport = _connect(listener)

    frame = controller.encrypt_frame(
        Message("control", REQUEST, "hello", {}, id=1).encode()
    )
    connection.data_received(frame[:5])
    assert connection._crypto is None
    connection.data_received(frame[5:])
    assert connection._crypto is not None
    assert _read(controller, transport).status == 0


@pytest.mark.asyncio
async def test_real_loopback_roundtrip():
    listener = hds_server.HDSListener()
    port = await listener.start("127.0.0.1")
    try:
        secret, salt = os.urandom(32), os.urandom(32)
        accessory_salt = listener.register_transport(secret, salt)
        acc_read, acc_write = hds.derive_keys(secret, salt, accessory_salt)
        controller = hds.HDSCrypto(acc_write, acc_read)

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            controller.encrypt_frame(
                Message("control", REQUEST, "hello", {}, id=1).encode()
            )
        )
        await writer.drain()

        data = await asyncio.wait_for(reader.read(4096), 2)
        buffer = bytearray(data)
        response = Message.decode(controller.decrypt_frame(buffer))
        assert response.topic == "hello"
        assert response.status == 0

        writer.close()
        await writer.wait_closed()
    finally:
        await listener.stop()
