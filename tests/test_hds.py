"""Tests for the HomeKit Data Stream transport primitives."""

import os
import struct

import pytest

from pyhap import hds, tlv


def test_setup_request_decode():
    salt = os.urandom(32)
    data = tlv.encode(
        hds.SETUP_TYPES["SESSION_COMMAND_TYPE"], hds.SESSION_COMMAND_START,
        hds.SETUP_TYPES["TRANSPORT_TYPE"], hds.TRANSPORT_TYPE_TCP,
        hds.SETUP_TYPES["CONTROLLER_KEY_SALT"], salt,
    )
    request = hds.SetupRequest.decode(data)
    assert request.controller_key_salt == salt
    assert request.transport_type == hds.TRANSPORT_TYPE_TCP


def test_setup_response_advertises_port():
    salt = os.urandom(32)
    response = hds.encode_setup_response(52345, salt)
    objs = tlv.decode(response)
    assert objs[hds.SETUP_RESPONSE_TYPES["STATUS"]] == hds.SETUP_STATUS_SUCCESS
    assert objs[hds.SETUP_RESPONSE_TYPES["ACCESSORY_KEY_SALT"]] == salt
    params = tlv.decode(
        objs[hds.SETUP_RESPONSE_TYPES["TRANSPORT_TYPE_SESSION_PARAMETERS"]]
    )
    port = struct.unpack(
        "<H", params[hds.TRANSPORT_SESSION_PARAM_TCP_LISTENING_PORT]
    )[0]
    assert port == 52345


def test_setup_response_error_omits_params():
    response = hds.encode_setup_response(0, b"", status=hds.SETUP_STATUS_BUSY)
    objs = tlv.decode(response)
    assert objs[hds.SETUP_RESPONSE_TYPES["STATUS"]] == hds.SETUP_STATUS_BUSY
    assert hds.SETUP_RESPONSE_TYPES["ACCESSORY_KEY_SALT"] not in objs


def test_derive_keys_deterministic_and_directional():
    secret = os.urandom(32)
    controller_salt = os.urandom(32)
    accessory_salt = os.urandom(32)
    read_a, write_a = hds.derive_keys(secret, controller_salt, accessory_salt)
    read_b, write_b = hds.derive_keys(secret, controller_salt, accessory_salt)
    assert (read_a, write_a) == (read_b, write_b)
    assert read_a != write_a
    # A different shared secret yields different keys.
    other_read, _ = hds.derive_keys(os.urandom(32), controller_salt, accessory_salt)
    assert other_read != read_a


def _paired_endpoints():
    secret = os.urandom(32)
    controller_salt = os.urandom(32)
    accessory_salt = hds.new_key_salt()
    acc_read, acc_write = hds.derive_keys(secret, controller_salt, accessory_salt)
    accessory = hds.HDSCrypto(acc_read, acc_write)
    # The controller's read is the accessory's write and vice versa.
    controller = hds.HDSCrypto(acc_write, acc_read)
    return accessory, controller


def test_frame_roundtrip_between_endpoints():
    accessory, controller = _paired_endpoints()

    payload = b"fragment-mp4-data" * 100
    frame = accessory.encrypt_frame(payload)
    buffer = bytearray(frame)
    assert controller.decrypt_frame(buffer) == payload
    assert not buffer

    # And the other direction.
    reply = controller.encrypt_frame(b"ack")
    buffer = bytearray(reply)
    assert accessory.decrypt_frame(buffer) == b"ack"


def test_frame_counter_advances():
    accessory, controller = _paired_endpoints()
    buffer = bytearray()
    for i in range(5):
        buffer += accessory.encrypt_frame(b"chunk-%d" % i)
    for i in range(5):
        assert controller.decrypt_frame(buffer) == b"chunk-%d" % i
    assert not buffer


def test_partial_frame_returns_none_without_consuming():
    accessory, controller = _paired_endpoints()
    frame = accessory.encrypt_frame(b"hello world")

    # Header only.
    buffer = bytearray(frame[:4])
    assert controller.decrypt_frame(buffer) is None
    assert len(buffer) == 4

    # Header plus part of the body.
    buffer = bytearray(frame[:-3])
    assert controller.decrypt_frame(buffer) is None

    # Completing the frame decrypts it.
    buffer += frame[-3:]
    assert controller.decrypt_frame(buffer) == b"hello world"


def test_tampered_frame_fails():
    accessory, controller = _paired_endpoints()
    frame = bytearray(accessory.encrypt_frame(b"secret"))
    frame[-1] ^= 0xFF
    with pytest.raises(Exception):
        controller.decrypt_frame(frame)


def test_frame_type_preserved_in_header():
    accessory, _ = _paired_endpoints()
    frame = accessory.encrypt_frame(b"x", frame_type=2)
    assert frame[0] == 2
