"""Wire format between the RL trainer and the LIBERO simulator process.

The simulator runs in the ``libero`` pixi environment, which deliberately does
**not** have this package (or torch) installed, so this module must stay
importable on its own: stdlib + numpy only. ``scripts/libero_server.py`` loads
this exact file by its path, which is why nothing here imports
``surgicalvla``.

Framing is newline-delimited JSON, one message per line. Arrays would be far
too large as JSON numbers -- a 128x128x3 frame is 49 kB of uint8 but ~1 MB of
JSON text -- so :func:`encode` converts numpy arrays into a
``{"__ndarray__": ...}`` envelope holding a base64 of the zlib-compressed raw
bytes plus dtype and shape. That keeps a frame at ~10-40 kB and keeps the
protocol usable over a socket.
"""

from __future__ import annotations

import base64
import json
import socket
import zlib

import numpy as np

#: Printed once by the server on stdout when it is ready to accept connections.
READY_PREFIX = "@@READY@@"

_ARRAY_KEY = "__ndarray__"


def encode_array(array: np.ndarray) -> dict:
    a = np.ascontiguousarray(array)
    raw = zlib.compress(a.tobytes(), 6)
    return {
        _ARRAY_KEY: True,
        "dtype": a.dtype.str,
        "shape": list(a.shape),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(payload: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    a = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return a.reshape(payload["shape"]).copy()


def encode(obj):
    """Recursively convert numpy arrays/scalars into JSON-friendly values."""
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def decode(obj):
    if isinstance(obj, dict):
        if obj.get(_ARRAY_KEY):
            return decode_array(obj)
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def dumps(message: dict) -> bytes:
    return (json.dumps(encode(message), separators=(",", ":")) + "\n").encode("utf-8")


def loads(line: bytes | str) -> dict:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return decode(json.loads(line))


class Channel:
    """Newline-delimited JSON over a connected socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._file = sock.makefile("rb")

    def send(self, message: dict) -> None:
        self.sock.sendall(dumps(message))

    def recv(self) -> dict | None:
        """Next message, or ``None`` when the peer closed the connection."""
        line = self._file.readline()
        if not line:
            return None
        return loads(line)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            try:
                self.sock.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect(host: str, port: int, timeout: float | None = 600.0) -> Channel:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return Channel(sock)
