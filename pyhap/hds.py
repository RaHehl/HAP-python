"""HomeKit Data Stream (HDS) transport primitives.

HDS is the TCP side channel HomeKit Secure Video uses to move fMP4 recording
fragments (and other bulk payloads) off the HAP control connection. A controller
sets up the transport with SetupDataStreamTransport, connects to the advertised
TCP port and exchanges frames encrypted with keys derived from the HAP session's
shared secret plus a per-transport salt.

This module is protocol-only: key derivation, frame encryption/decryption and
the setup-request/response TLV shapes. The listener and the higher-level HDS
protocol (control/dataSend topics) build on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import struct
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from pyhap import tlv
from pyhap.hap_crypto import hap_hkdf

# --- Setup Data Stream Transport (HAP R17 section 12) ---

TRANSPORT_TYPE_TCP = b"\x00"

# SetupDataStreamTransport write (Transfer Transport Configuration).
SETUP_TYPES = {
    "SESSION_COMMAND_TYPE": b"\x01",
    "TRANSPORT_TYPE": b"\x02",
    "CONTROLLER_KEY_SALT": b"\x03",
}

SESSION_COMMAND_START = b"\x00"

# SetupDataStreamTransport response.
SETUP_RESPONSE_TYPES = {
    "STATUS": b"\x01",
    "TRANSPORT_TYPE_SESSION_PARAMETERS": b"\x02",
    "ACCESSORY_KEY_SALT": b"\x03",
}

TRANSPORT_SESSION_PARAM_TCP_LISTENING_PORT = b"\x01"

SETUP_STATUS_SUCCESS = b"\x00"
SETUP_STATUS_GENERIC_ERROR = b"\x01"
SETUP_STATUS_BUSY = b"\x02"

# HDS derives its two directional keys from the HAP session shared secret salted
# with both key salts; the info strings pick the direction.
_KEY_SALT_INFO_READ = b"HDS-Read-Encryption-Key"
_KEY_SALT_INFO_WRITE = b"HDS-Write-Encryption-Key"

_NONCE_LENGTH = 12
_TAG_LENGTH = 16
_FRAME_HEADER_LENGTH = 4
_MAX_PAYLOAD_LENGTH = 0x1000000 - 1  # 24-bit length field


@dataclass
class SetupRequest:
    """Decoded SetupDataStreamTransport controller request."""

    controller_key_salt: bytes
    transport_type: bytes = TRANSPORT_TYPE_TCP
    session_command: bytes = SESSION_COMMAND_START

    @classmethod
    def decode(cls, data: bytes) -> "SetupRequest":
        objs = tlv.decode(data)
        return cls(
            controller_key_salt=objs[SETUP_TYPES["CONTROLLER_KEY_SALT"]],
            transport_type=objs.get(
                SETUP_TYPES["TRANSPORT_TYPE"], TRANSPORT_TYPE_TCP
            ),
            session_command=objs.get(
                SETUP_TYPES["SESSION_COMMAND_TYPE"], SESSION_COMMAND_START
            ),
        )


def encode_setup_response(
    listening_port: int,
    accessory_key_salt: bytes,
    status: bytes = SETUP_STATUS_SUCCESS,
) -> bytes:
    """Encode a SetupDataStreamTransport response advertising the TCP port."""
    if status != SETUP_STATUS_SUCCESS:
        return tlv.encode(SETUP_RESPONSE_TYPES["STATUS"], status)
    session_params = tlv.encode(
        TRANSPORT_SESSION_PARAM_TCP_LISTENING_PORT,
        struct.pack("<H", listening_port),
    )
    return tlv.encode(
        SETUP_RESPONSE_TYPES["STATUS"],
        status,
        SETUP_RESPONSE_TYPES["TRANSPORT_TYPE_SESSION_PARAMETERS"],
        session_params,
        SETUP_RESPONSE_TYPES["ACCESSORY_KEY_SALT"],
        accessory_key_salt,
    )


def derive_keys(shared_secret, controller_key_salt, accessory_key_salt):
    """Derive the (read, write) HDS keys from the HAP session shared secret.

    ``read`` decrypts controller->accessory frames; ``write`` encrypts
    accessory->controller frames. The salt is the concatenation of the two key
    salts, matching the controller's derivation.
    """
    salt = controller_key_salt + accessory_key_salt
    read_key = hap_hkdf(shared_secret, salt, _KEY_SALT_INFO_READ)
    write_key = hap_hkdf(shared_secret, salt, _KEY_SALT_INFO_WRITE)
    return read_key, write_key


def new_key_salt() -> bytes:
    """Return a fresh 32-byte accessory key salt."""
    return os.urandom(32)


def _nonce(counter: int) -> bytes:
    # 96-bit nonce: 64-bit little-endian counter at offset 0, then four zero
    # bytes (HDS convention, distinct from the right-justified HAP control nonce).
    return struct.pack("<Q", counter) + b"\x00\x00\x00\x00"


class HDSCrypto:
    """Frame encryption/decryption for one HDS transport.

    Each 4-byte frame header (a 1-byte type followed by a 24-bit little-endian
    payload length) is authenticated as additional data over an encrypted
    payload, with a per-direction monotonically increasing nonce counter.
    """

    def __init__(self, read_key: bytes, write_key: bytes) -> None:
        self._read_cipher = ChaCha20Poly1305(read_key)
        self._write_cipher = ChaCha20Poly1305(write_key)
        self._read_count = 0
        self._write_count = 0

    def encrypt_frame(self, payload: bytes, frame_type: int = 1) -> bytes:
        """Encrypt a payload into a full HDS frame (header + ciphertext + tag)."""
        if len(payload) > _MAX_PAYLOAD_LENGTH:
            raise ValueError("HDS payload too large for a single frame")
        header = bytes([frame_type]) + len(payload).to_bytes(3, "big")
        nonce = _nonce(self._write_count)
        self._write_count += 1
        ciphertext = self._write_cipher.encrypt(nonce, payload, header)
        return header + ciphertext

    def decrypt_frame(self, buffer: bytearray) -> Optional[bytes]:
        """Decrypt one complete frame from the front of ``buffer``.

        Returns the plaintext payload and removes the frame from ``buffer``, or
        ``None`` when a full frame is not yet buffered (leaving ``buffer``
        untouched so the caller can retry once more data arrives).
        """
        if len(buffer) < _FRAME_HEADER_LENGTH:
            return None
        header = bytes(buffer[:_FRAME_HEADER_LENGTH])
        payload_length = int.from_bytes(header[1:4], "big")
        frame_length = _FRAME_HEADER_LENGTH + payload_length + _TAG_LENGTH
        if len(buffer) < frame_length:
            return None
        ciphertext = bytes(buffer[_FRAME_HEADER_LENGTH:frame_length])
        nonce = _nonce(self._read_count)
        plaintext = self._read_cipher.decrypt(nonce, ciphertext, header)
        self._read_count += 1
        del buffer[:frame_length]
        return plaintext

    @property
    def frame_type_of(self):
        """Expose the frame type byte reader for callers inspecting headers."""
        return lambda header: header[0]
