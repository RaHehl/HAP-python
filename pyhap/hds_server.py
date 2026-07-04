"""HomeKit Data Stream TCP listener and connection dispatch.

Ties the HDS transport crypto (:mod:`pyhap.hds`) and message codec
(:mod:`pyhap.hds_protocol`) into an asyncio TCP server. A controller sets up a
transport over HAP (SetupDataStreamTransport), connects to the advertised port
and completes a ``control``/``hello`` handshake; from there both sides exchange
request/response/event messages on named protocols (e.g. ``dataSend`` for
recording fragment transfer).

The listener does not know the HAP session shared secret on its own; the camera
accessory registers a transport with the secret when it answers
SetupDataStreamTransport, and the first frame from the controller is matched
against the registered transports by trial decryption.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Awaitable, Callable, Dict, Optional, Tuple

from pyhap import hds
from pyhap.hds_protocol import EVENT, REQUEST, RESPONSE, Message

logger = logging.getLogger("pyhap.hds")

RequestHandler = Callable[[Message], Tuple[int, dict]]
EventHandler = Callable[[Message], None]


@dataclass
class _PendingTransport:
    read_key: bytes
    write_key: bytes


class HDSConnection(asyncio.Protocol):
    """One HDS TCP connection: key binding, framing and message dispatch."""

    def __init__(self, listener: "HDSListener") -> None:
        self._listener = listener
        self._buffer = bytearray()
        self._crypto: Optional[hds.HDSCrypto] = None
        self._transport: Optional[asyncio.Transport] = None
        self._request_handlers: Dict[Tuple[str, str], RequestHandler] = {}
        self._event_handlers: Dict[Tuple[str, str], EventHandler] = {}
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._next_request_id = 1
        self.on_close: Optional[Callable[["HDSConnection"], None]] = None
        # The mandatory control handshake is answered by default.
        self.add_request_handler("control", "hello", lambda message: (0, {}))

    # --- registration API ---

    def add_request_handler(
        self, protocol: str, topic: str, handler: RequestHandler
    ) -> None:
        """Register a handler returning ``(status, response_message)``."""
        self._request_handlers[(protocol, topic)] = handler

    def add_event_handler(
        self, protocol: str, topic: str, handler: EventHandler
    ) -> None:
        """Register a handler for a fire-and-forget event."""
        self._event_handlers[(protocol, topic)] = handler

    # --- send API ---

    def send_event(self, protocol: str, topic: str, message: dict) -> None:
        """Send an event message (no response expected)."""
        self._send(Message(protocol, EVENT, topic, message))

    def send_request(
        self, protocol: str, topic: str, message: dict
    ) -> "asyncio.Future[Message]":
        """Send a request; the future resolves with the controller's response."""
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future
        self._send(Message(protocol, REQUEST, topic, message, id=request_id))
        return future

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    # --- asyncio.Protocol ---

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]

    def data_received(self, data: bytes) -> None:
        self._buffer += data
        if self._crypto is None and not self._bind():
            return
        self._drain_frames()

    def connection_lost(self, exc: Optional[Exception]) -> None:
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        self._listener._connections.discard(self)
        if self.on_close is not None:
            self.on_close(self)

    # --- internals ---

    def _bind(self) -> bool:
        """Identify the transport by trial-decrypting the first frame."""
        if len(self._buffer) < 4:
            return False
        for pending in list(self._listener._pending):
            trial = hds.HDSCrypto(pending.read_key, pending.write_key)
            probe = bytearray(self._buffer)
            try:
                payload = trial.decrypt_frame(probe)
            except Exception:  # noqa: BLE001 - wrong keys fail the auth tag
                continue
            if payload is None:
                # Right keys but the frame is not complete yet; wait for more.
                return False
            self._crypto = trial
            self._buffer = probe
            self._listener._pending.remove(pending)
            self._listener._connections.add(self)
            self._dispatch(payload)
            return True
        return False

    def _drain_frames(self) -> None:
        while True:
            try:
                payload = self._crypto.decrypt_frame(self._buffer)
            except Exception:  # noqa: BLE001
                logger.warning("Dropping HDS connection on frame decrypt failure")
                self.close()
                return
            if payload is None:
                return
            self._dispatch(payload)

    def _dispatch(self, payload: bytes) -> None:
        try:
            message = Message.decode(payload)
        except (ValueError, IndexError):
            logger.warning("Ignoring malformed HDS message")
            return
        if message.kind == REQUEST:
            self._handle_request(message)
        elif message.kind == RESPONSE:
            future = self._pending_requests.pop(message.id, None)
            if future is not None and not future.done():
                future.set_result(message)
        elif message.kind == EVENT:
            handler = self._event_handlers.get((message.protocol, message.topic))
            if handler is not None:
                handler(message)

    def _handle_request(self, message: Message) -> None:
        handler = self._request_handlers.get((message.protocol, message.topic))
        if handler is None:
            status, response_message = 2, {}  # unsupported
        else:
            status, response_message = handler(message)
        self._send(
            Message(
                message.protocol,
                RESPONSE,
                message.topic,
                response_message,
                id=message.id,
                status=status,
            )
        )

    def _send(self, message: Message) -> None:
        if self._transport is None or self._crypto is None:
            raise RuntimeError("HDS connection is not ready to send")
        self._transport.write(self._crypto.encrypt_frame(message.encode()))


class HDSListener:
    """An asyncio TCP server accepting HDS connections."""

    def __init__(self) -> None:
        self._server: Optional[asyncio.AbstractServer] = None
        self._pending = []
        self._connections = set()
        self.on_connection: Optional[Callable[[HDSConnection], None]] = None

    async def start(self, host: str = "0.0.0.0") -> int:
        """Bind the server to an ephemeral port and return it."""
        loop = asyncio.get_event_loop()
        self._server = await loop.create_server(self._make_connection, host, 0)
        return self.port

    def _make_connection(self) -> HDSConnection:
        connection = HDSConnection(self)
        if self.on_connection is not None:
            self.on_connection(connection)
        return connection

    @property
    def port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    def register_transport(
        self, shared_secret: bytes, controller_key_salt: bytes
    ) -> bytes:
        """Register an expected transport, returning the accessory key salt.

        The camera calls this when it answers SetupDataStreamTransport, passing
        the HAP session shared secret and the controller's key salt. The
        returned accessory key salt goes into the setup response.
        """
        accessory_key_salt = hds.new_key_salt()
        read_key, write_key = hds.derive_keys(
            shared_secret, controller_key_salt, accessory_key_salt
        )
        self._pending.append(_PendingTransport(read_key, write_key))
        return accessory_key_salt

    async def stop(self) -> None:
        for connection in list(self._connections):
            connection.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
