"""Tests for the HDS payload codec and message layer."""

import pytest

from pyhap import hds_protocol as hp


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        -1,
        0,
        38,
        39,
        127,
        128,
        -128,
        -129,
        32767,
        70000,
        -(2**40),
        3.5,
        "",
        "hello",
        "x" * 32,
        "x" * 33,
        "ünïcödé " * 50,
        b"",
        b"\x00\x01\x02",
        b"d" * 32,
        b"d" * 500,
        [],
        [1, 2, 3],
        list(range(14)),
        list(range(50)),
        {},
        {"a": 1, "b": "two"},
        {str(i): i for i in range(14)},
        {str(i): i for i in range(40)},
    ],
)
def test_roundtrip(value):
    assert hp.decode(hp.encode(value)) == value


def test_integer_boundary_tags():
    # 0..38 encode inline in a single byte.
    assert hp.encode(0) == bytes([0x08])
    assert hp.encode(38) == bytes([0x2E])
    # 39 spills to int8.
    assert hp.encode(39)[0] == 0x30
    assert hp.encode(-1) == bytes([0x07])


def test_nested_structure_roundtrip():
    value = {
        "protocol": "dataSend",
        "chunk": {
            "data": b"\xde\xad\xbe\xef" * 100,
            "offset": 4096,
            "final": False,
            "tiers": [1, 2, 3],
        },
    }
    assert hp.decode(hp.encode(value)) == value


def test_request_message_roundtrip():
    msg = hp.request("dataSend", "open", {"target": "controller", "type": "ipcamera.recording"}, 7)
    decoded = hp.Message.decode(msg.encode())
    assert decoded.protocol == "dataSend"
    assert decoded.kind == hp.REQUEST
    assert decoded.topic == "open"
    assert decoded.id == 7
    assert decoded.status is None
    assert decoded.message["type"] == "ipcamera.recording"


def test_response_message_roundtrip():
    msg = hp.response("dataSend", "open", {"status": 0}, request_id=7, status=0)
    decoded = hp.Message.decode(msg.encode())
    assert decoded.kind == hp.RESPONSE
    assert decoded.id == 7
    assert decoded.status == 0


def test_event_message_roundtrip():
    msg = hp.event("dataSend", "data", {"data": b"fragment", "endOfStream": True})
    decoded = hp.Message.decode(msg.encode())
    assert decoded.kind == hp.EVENT
    assert decoded.topic == "data"
    assert decoded.id is None
    assert decoded.message["endOfStream"] is True


def test_message_header_length_prefix():
    payload = hp.request("control", "hello", {}, 1).encode()
    header_length = payload[0]
    # The header is a self-contained dictionary right after the length byte.
    assert hp.decode(payload[1 : 1 + header_length])["protocol"] == "control"


def test_hds_transport_carries_message():
    """A message encodes, encrypts through HDSCrypto and decodes end to end."""
    import os

    from pyhap import hds

    secret = os.urandom(32)
    csalt = os.urandom(32)
    asalt = hds.new_key_salt()
    acc_read, acc_write = hds.derive_keys(secret, csalt, asalt)
    accessory = hds.HDSCrypto(acc_read, acc_write)
    controller = hds.HDSCrypto(acc_write, acc_read)

    outgoing = hp.event("dataSend", "data", {"data": b"x" * 5000, "endOfStream": False})
    frame = accessory.encrypt_frame(outgoing.encode())
    buffer = bytearray(frame)
    payload = controller.decrypt_frame(buffer)
    decoded = hp.Message.decode(payload)
    assert decoded.message["data"] == b"x" * 5000
