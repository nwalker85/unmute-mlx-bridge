"""End-to-end protocol conformance for `TtsServer` over a real WebSocket, using a
deterministic fake engine (see `test_stt_server_conformance.py` for the rationale
and the injection seam this relies on).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import msgpack
import pytest
import websockets
from websockets.asyncio.server import serve

from unmute_mlx_bridge.config import TtsConfig
from unmute_mlx_bridge.observability import build_process_request
from unmute_mlx_bridge.tts.engine import TtsStepEvent
from unmute_mlx_bridge.tts.server import PROTOCOL_PATH, TtsServer


class FakeBundle:
    """Stands in for `TtsModelBundle`; carries no MLX state."""


@dataclass
class FakeSession:
    """Deterministic fake of `TtsSession`: one Audio + one Text event per word,
    a final flush on Eos — enough to exercise the full message sequencing without
    touching MLX.
    """

    bundle: object
    voice: str | None
    max_gen_length: int
    _step: int = field(default=0, init=False)

    def push_text(self, text: str) -> list[TtsStepEvent]:
        events = []
        for word in text.split():
            self._step += 1
            events.append(TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3]))
            events.append(
                TtsStepEvent(
                    kind="word", text=word, start_s=self._step - 1, stop_s=self._step
                )
            )
        return events

    def push_eos(self) -> list[TtsStepEvent]:
        self._step += 1
        return [TtsStepEvent(kind="audio", pcm=[0.0, 0.0])]


@pytest.fixture
async def tts_server():
    config = TtsConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        voice_repo="unused",
        default_voice="unused.wav",
        quantize_bits=None,
        max_gen_length=1000,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = TtsServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield server, port


async def _connect(port: int, query: str = "format=PcmMessagePack"):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}{PROTOCOL_PATH}?{query}",
        additional_headers={"kyutai-api-key": "public_token"},
    )


async def _recv_message(ws) -> dict:
    raw = await ws.recv(decode=False)
    return msgpack.unpackb(raw)


async def test_connect_receives_ready(tts_server):
    _server, port = tts_server
    async with await _connect(port) as ws:
        message = await _recv_message(ws)
        assert message == {"type": "Ready"}


async def test_unsupported_format_gets_error_and_close(tts_server):
    _server, port = tts_server
    async with await _connect(port, query="format=OggOpus") as ws:
        message = await _recv_message(ws)
        assert message["type"] == "Error"
        assert "PcmMessagePack" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await ws.recv()


async def test_text_produces_audio_then_text_event(tts_server):
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": "hello world"}))
        audio1 = await _recv_message(ws)
        text1 = await _recv_message(ws)
        audio2 = await _recv_message(ws)
        text2 = await _recv_message(ws)
        assert audio1["type"] == "Audio"
        assert text1 == {"type": "Text", "text": "hello", "start_s": 0.0, "stop_s": 1.0}
        assert audio2["type"] == "Audio"
        assert text2["text"] == "world"


async def test_eos_flushes_trailing_audio(tts_server):
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": "hi"}))
        await _recv_message(ws)  # Audio
        await _recv_message(ws)  # Text
        await ws.send(msgpack.packb({"type": "Eos"}))
        flushed = await _recv_message(ws)
        assert flushed["type"] == "Audio"


async def test_legacy_null_byte_eos_is_accepted(tts_server):
    """Real `moshi-server`'s `py_module.rs::recv_loop` treats a raw `b"\\x00"`
    binary frame as end-of-stream in addition to the msgpack `Eos` message."""
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(b"\x00")
        flushed = await _recv_message(ws)
        assert flushed["type"] == "Audio"


async def test_voice_message_is_explicitly_rejected(tts_server):
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Voice", "embeddings": [0.1], "shape": [1]}))
        error = await _recv_message(ws)
        assert error["type"] == "Error"
        assert "not supported" in error["message"]
        # Connection stays open afterwards.
        await ws.send(msgpack.packb({"type": "Text", "text": "still works"}))
        audio = await _recv_message(ws)
        assert audio["type"] == "Audio"


async def test_second_concurrent_session_is_rejected(tts_server):
    _server, port = tts_server
    first = await _connect(port)
    await _recv_message(first)  # Ready

    second = await _connect(port)
    message = await _recv_message(second)
    assert message["type"] == "Error"

    await first.close()


async def test_unauthorized_connection_is_rejected_before_upgrade(tts_server):
    _server, port = tts_server
    with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
        await websockets.connect(f"ws://127.0.0.1:{port}{PROTOCOL_PATH}?format=PcmMessagePack")
    assert exc_info.value.response.status_code == 401
