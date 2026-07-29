"""MessagePack framing helpers shared by the STT and TTS servers.

Real `moshi-server` (Rust) serializes its `OutMsg`/`InMsg` enums with `rmp_serde`
using `.with_human_readable().with_struct_map()` — i.e. msgpack *maps* with a
string `"type"` tag field, not array-encoded structs. `unmute`'s Python client
mirrors this with `msgpack.packb(..., use_bin_type=True, use_single_float=True)`
on the way out and plain `msgpack.unpackb(...)` on the way in
(`unmute/stt/speech_to_text.py`, `unmute/tts/text_to_speech.py`).

We match that: outgoing messages are packed as msgpack maps with single-precision
(float32) floats — cheap, matches the audio/VAD data's native precision, and is
indistinguishable to any msgpack-conformant client since msgpack floats decode to
the same Python `float` regardless of wire width. Incoming messages are decoded
with the default (raw=False) unpacker, which accepts both widths.
"""

from __future__ import annotations

from typing import Any

import msgpack
from pydantic import BaseModel


def pack_message(message: BaseModel) -> bytes:
    """Serialize an outgoing protocol message to a msgpack binary WebSocket frame."""
    payload = message.model_dump(mode="python")
    return msgpack.packb(payload, use_bin_type=True, use_single_float=True)


def unpack_message(data: bytes) -> dict[str, Any]:
    """Decode an incoming msgpack binary WebSocket frame into a plain dict.

    Raises `msgpack.exceptions.UnpackException` (or a `ValueError`/`TypeError` from
    a malformed payload) on invalid input; callers are expected to turn that into a
    protocol `Error` message rather than letting it propagate as an unhandled
    exception, per the moshi-server contract that a bad frame closes only the
    offending connection.
    """
    unpacked = msgpack.unpackb(data, raw=False)
    if not isinstance(unpacked, dict):
        raise TypeError(f"expected a msgpack map, got {type(unpacked).__name__}")
    return unpacked
