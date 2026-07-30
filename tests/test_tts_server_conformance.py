"""End-to-end protocol conformance for `TtsServer` over a real WebSocket, using a
deterministic fake engine (see `test_stt_server_conformance.py` for the rationale
and the injection seam this relies on).
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import ClassVar

import msgpack
import pytest
import websockets
from websockets.asyncio.server import serve

from unmute_mlx_bridge.tts import server as tts_server_module
from unmute_mlx_bridge.config import TtsConfig
from unmute_mlx_bridge.observability import build_process_request
from unmute_mlx_bridge.tts.engine import TtsStepEvent
from unmute_mlx_bridge.tts.server import PROTOCOL_PATH, TtsServer


class FakeBundle:
    """Stands in for `TtsModelBundle`; carries no MLX state."""

    def resolve_voice(self, voice: str) -> str:
        return voice


@dataclass
class FakeSession:
    """Deterministic fake of `TtsSession`: one Audio + one Text event per word,
    a final flush on Eos — enough to exercise the full message sequencing without
    touching MLX.
    """

    bundle: object
    voice: str | None
    max_gen_length: int
    seed: int = 42
    temperature: float = 0.8
    top_k: int = 250
    cfg_alpha: float | None = None
    _step: int = field(default=0, init=False)
    text_inputs: list[str] = field(default_factory=list, init=False)
    instances: ClassVar[list[FakeSession]] = []

    def __post_init__(self):
        self.instances.append(self)

    def stream_text(self, text: str, _cancelled=lambda: False):
        self.text_inputs.append(text)
        for word in text.split():
            self._step += 1
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            yield TtsStepEvent(
                kind="word", text=word, start_s=self._step - 1, stop_s=self._step
            )

    def stream_eos(self, _cancelled=lambda: False):
        self._step += 1
        yield TtsStepEvent(kind="audio", pcm=[0.0, 0.0])


@asynccontextmanager
async def _running_tts_server(
    *,
    delivery_mode="streaming",
    session_cls=FakeSession,
    max_buffered_chars=4096,
    max_buffered_audio_seconds=60.0,
):
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
        delivery_mode=delivery_mode,
        max_buffered_chars=max_buffered_chars,
        max_buffered_audio_seconds=max_buffered_audio_seconds,
    )
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundle(),
        session_cls=session_cls,
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield server, port


@pytest.fixture
async def tts_server():
    FakeSession.instances.clear()
    async with _running_tts_server() as running:
        yield running


async def _connect(port: int, query: str = "format=PcmMessagePack"):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}{PROTOCOL_PATH}?{query}",
        additional_headers={"kyutai-api-key": "public_token"},
    )


async def _recv_message(ws) -> dict:
    raw = await ws.recv(decode=False)
    return msgpack.unpackb(raw)


async def _recv_until_closed(ws) -> list[dict]:
    messages: list[dict] = []
    try:
        while True:
            messages.append(await _recv_message(ws))
    except websockets.exceptions.ConnectionClosedOK:
        return messages


async def test_default_bundle_loader_forwards_configured_codebook_depth(monkeypatch):
    observed_n_q = None

    def load_bundle(_hf_repo, _voice_repo, _quantize_bits, n_q=None):
        nonlocal observed_n_q
        observed_n_q = n_q
        return FakeBundle()

    monkeypatch.setattr(tts_server_module.TtsModelBundle, "load", load_bundle)
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
        n_q=24,
    )

    server = TtsServer(config)
    await server.load_model()

    assert observed_n_q == 24


async def test_connect_receives_ready(tts_server):
    _server, port = tts_server
    async with await _connect(port) as ws:
        message = await _recv_message(ws)
        assert message == {"type": "Ready"}


async def test_ready_precedes_voice_resolution_and_disconnect_releases_slot():
    initialization_started = threading.Event()
    release_initialization = threading.Event()
    resolver_calls = 0
    resolver_calls_lock = threading.Lock()

    def blocking_voice_resolver(_bundle, voice):
        nonlocal resolver_calls
        with resolver_calls_lock:
            resolver_calls += 1
        initialization_started.set()
        if not release_initialization.wait(timeout=2):
            raise RuntimeError("test did not release voice initialization")
        return voice

    async def wait_for_cancellations(server, expected: int) -> None:
        async with asyncio.timeout(0.5):
            while server.metrics.cancellations._value.get() < expected:
                await asyncio.sleep(0)

    config = TtsConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        voice_repo="unused",
        default_voice="uncached.wav",
        quantize_bits=None,
        max_gen_length=1000,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundle(),
        session_cls=FakeSession,
        voice_resolver=blocking_voice_resolver,
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        try:
            port = ws_server.sockets[0].getsockname()[1]
            first = await _connect(port)
            assert await _recv_message(first) == {"type": "Ready"}
            assert await asyncio.to_thread(initialization_started.wait, 0.5)

            await first.close()
            await wait_for_cancellations(server, 1)

            for expected_cancellations in range(2, 8):
                connection = await _connect(port)
                assert await _recv_message(connection) == {"type": "Ready"}
                await connection.close()
                await wait_for_cancellations(server, expected_cancellations)

            with resolver_calls_lock:
                assert resolver_calls == 1
        finally:
            release_initialization.set()


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


async def test_query_settings_are_forwarded_to_the_generation_session(tts_server):
    _server, port = tts_server
    query = (
        "format=PcmMessagePack&seed=7&temperature=0.4&top_k=11"
        "&cfg_alpha=1.5&max_seq_len=321"
    )
    async with await _connect(port, query=query) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
        await _recv_message(ws)  # Audio

    session = FakeSession.instances[-1]
    assert session.seed == 7
    assert session.temperature == 0.4
    assert session.top_k == 11
    assert session.cfg_alpha == 1.5
    assert session.max_gen_length == 321


async def test_audio_is_emitted_before_text_generation_finishes():
    release_generation = threading.Event()

    @dataclass
    class BlockingSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, _cancelled=lambda: False):
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            if not release_generation.wait(timeout=2):
                raise RuntimeError("test did not release generation")
            yield TtsStepEvent(kind="word", text="hello", start_s=0.0, stop_s=1.0)

        def push_text(self, text: str) -> list[TtsStepEvent]:
            return list(self.stream_text(text))

        def stream_eos(self, _cancelled=lambda: False):
            return iter(())

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
    server = TtsServer(config, bundle_loader=lambda: FakeBundle(), session_cls=BlockingSession)
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port) as ws:
            await _recv_message(ws)  # Ready
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            try:
                first = await asyncio.wait_for(_recv_message(ws), timeout=0.5)
            finally:
                release_generation.set()
            assert first["type"] == "Audio"
            second = await _recv_message(ws)
            assert second["type"] == "Text"


@pytest.mark.parametrize(
    "eos_frame",
    [
        msgpack.packb({"type": "Eos"}),
        b"\x00",
    ],
)
async def test_buffered_turn_waits_for_eos_and_preserves_event_order(eos_frame):
    FakeSession.instances.clear()
    async with _running_tts_server(delivery_mode="buffered_turn") as (
        _server,
        port,
    ):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello "}))
            await ws.send(msgpack.packb({"type": "Text", "text": "world"}))
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(_recv_message(ws), timeout=0.05)

            await ws.send(eos_frame)
            messages = await _recv_until_closed(ws)

    assert FakeSession.instances[-1].text_inputs == ["hello world"]
    assert [message["type"] for message in messages] == [
        "Audio",
        "Text",
        "Audio",
        "Text",
        "Audio",
    ]
    assert [
        message["text"] for message in messages if message["type"] == "Text"
    ] == [
        "hello",
        "world",
    ]


async def test_buffered_turn_preserves_boundaries_between_stock_word_chunks():
    FakeSession.instances.clear()
    async with _running_tts_server(delivery_mode="buffered_turn") as (
        _server,
        port,
    ):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            await ws.send(msgpack.packb({"type": "Text", "text": "world"}))
            await ws.send(msgpack.packb({"type": "Eos"}))
            messages = await _recv_until_closed(ws)

    assert FakeSession.instances[-1].text_inputs == ["hello world"]
    assert [
        message["text"] for message in messages if message["type"] == "Text"
    ] == ["hello", "world"]


async def test_buffered_empty_turn_closes_without_generation():
    FakeSession.instances.clear()
    async with _running_tts_server(delivery_mode="buffered_turn") as (
        _server,
        port,
    ):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Eos"}))
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert FakeSession.instances[-1].text_inputs == []


async def test_buffered_turn_emits_nothing_until_generation_completes():
    generation_started = threading.Event()
    release_generation = threading.Event()

    @dataclass
    class BlockingBufferedSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, _cancelled=lambda: False):
            generation_started.set()
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            if not release_generation.wait(timeout=2):
                raise RuntimeError("test did not release generation")
            yield TtsStepEvent(
                kind="word",
                text="hello",
                start_s=0.0,
                stop_s=1.0,
            )

        def stream_eos(self, _cancelled=lambda: False):
            yield TtsStepEvent(kind="audio", pcm=[0.0, 0.0])

    async with _running_tts_server(
        delivery_mode="buffered_turn",
        session_cls=BlockingBufferedSession,
    ) as (_server, port):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            await ws.send(msgpack.packb({"type": "Eos"}))
            assert await asyncio.to_thread(generation_started.wait, 0.5)
            try:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(_recv_message(ws), timeout=0.05)
            finally:
                release_generation.set()
            messages = await _recv_until_closed(ws)

    assert [message["type"] for message in messages] == [
        "Audio",
        "Text",
        "Audio",
    ]


async def test_buffered_generation_failure_emits_only_sanitized_error():
    @dataclass
    class FailingSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, _cancelled=lambda: False):
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            raise RuntimeError("private model failure detail")

        def stream_eos(self, _cancelled=lambda: False):
            return iter(())

    async with _running_tts_server(
        delivery_mode="buffered_turn",
        session_cls=FailingSession,
    ) as (server, port):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            await ws.send(msgpack.packb({"type": "Eos"}))
            error = await _recv_message(ws)
            assert error == {
                "type": "Error",
                "message": "buffered TTS generation failed",
            }
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert server.metrics.output_audio_seconds._value.get() == 0
    assert (
        server.metrics.buffered_turn_failures.labels(
            reason="generation"
        )._value.get()
        == 1
    )


async def test_buffered_audio_limit_discards_the_complete_partial_turn():
    async with _running_tts_server(
        delivery_mode="buffered_turn",
        max_buffered_audio_seconds=2 / 24_000,
    ) as (server, port):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            await ws.send(msgpack.packb({"type": "Eos"}))
            error = await _recv_message(ws)
            assert error == {
                "type": "Error",
                "message": "buffered TTS output exceeds configured limit",
            }
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert server.metrics.output_audio_seconds._value.get() == 0
    assert (
        server.metrics.buffered_turn_failures.labels(
            reason="output_limit"
        )._value.get()
        == 1
    )


async def test_buffered_input_limit_rejects_before_retaining_extra_text():
    FakeSession.instances.clear()
    async with _running_tts_server(
        delivery_mode="buffered_turn",
        max_buffered_chars=5,
    ) as (server, port):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            await ws.send(msgpack.packb({"type": "Text", "text": "!"}))
            error = await _recv_message(ws)
            assert error == {
                "type": "Error",
                "message": "buffered TTS input exceeds configured limit",
            }
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert FakeSession.instances[-1].text_inputs == []
    assert (
        server.metrics.buffered_turn_failures.labels(
            reason="input_limit"
        )._value.get()
        == 1
    )


async def test_buffered_disconnect_before_eos_discards_text_and_frees_slot():
    FakeSession.instances.clear()
    async with _running_tts_server(delivery_mode="buffered_turn") as (
        server,
        port,
    ):
        first = await _connect(port)
        assert await _recv_message(first) == {"type": "Ready"}
        await first.send(msgpack.packb({"type": "Text", "text": "discard me"}))
        await first.close()
        async with asyncio.timeout(0.5):
            while server.health.session_active:
                await asyncio.sleep(0)
        async with await _connect(port) as second:
            assert await _recv_message(second) == {"type": "Ready"}

    assert FakeSession.instances[0].text_inputs == []


async def test_buffered_disconnect_during_generation_cancels_and_frees_slot():
    generation_started = threading.Event()
    generation_stopped = threading.Event()
    force_release = threading.Event()

    @dataclass
    class CancellableBufferedSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, cancelled=lambda: False):
            generation_started.set()
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            while not cancelled() and not force_release.is_set():
                time.sleep(0.01)
            generation_stopped.set()

        def stream_eos(self, _cancelled=lambda: False):
            return iter(())

    async with _running_tts_server(
        delivery_mode="buffered_turn",
        session_cls=CancellableBufferedSession,
    ) as (server, port):
        try:
            first = await _connect(port)
            assert await _recv_message(first) == {"type": "Ready"}
            await first.send(
                msgpack.packb({"type": "Text", "text": "hello"})
            )
            await first.send(msgpack.packb({"type": "Eos"}))
            assert await asyncio.to_thread(generation_started.wait, 0.5)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(_recv_message(first), timeout=0.05)

            await first.close()
            assert await asyncio.to_thread(generation_stopped.wait, 0.5)

            async with await _connect(port) as second:
                assert await _recv_message(second) == {"type": "Ready"}
                await second.send(msgpack.packb({"type": "Eos"}))
                with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                    await asyncio.wait_for(second.recv(), timeout=0.2)
        finally:
            force_release.set()

    assert (
        server.metrics.buffered_turn_failures.labels(
            reason="disconnect"
        )._value.get()
        == 1
    )


async def test_disconnect_cancels_generation_and_frees_the_slot():
    generation_started = threading.Event()
    generation_stopped = threading.Event()
    force_release = threading.Event()

    @dataclass
    class CancellableSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, cancelled=lambda: False):
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            generation_started.set()
            while not cancelled() and not force_release.is_set():
                time.sleep(0.01)
            generation_stopped.set()

        def stream_eos(self, _cancelled=lambda: False):
            return iter(())

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
    server = TtsServer(config, bundle_loader=lambda: FakeBundle(), session_cls=CancellableSession)
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection,
        config.host,
        config.port,
        process_request=process_request,
    ) as ws_server:
        try:
            port = ws_server.sockets[0].getsockname()[1]
            first = await _connect(port)
            await _recv_message(first)  # Ready
            await first.send(msgpack.packb({"type": "Text", "text": "hello"}))
            assert (await _recv_message(first))["type"] == "Audio"
            assert await asyncio.to_thread(generation_started.wait, 0.5)

            await first.close()
            assert await asyncio.to_thread(generation_stopped.wait, 0.5)

            async with await _connect(port) as second:
                assert await _recv_message(second) == {"type": "Ready"}
        finally:
            force_release.set()


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
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=0.2)


async def test_legacy_null_byte_eos_is_accepted(tts_server):
    """Real `moshi-server`'s `py_module.rs::recv_loop` treats a raw `b"\\x00"`
    binary frame as end-of-stream in addition to the msgpack `Eos` message."""
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(b"\x00")
        flushed = await _recv_message(ws)
        assert flushed["type"] == "Audio"
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=0.2)


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


async def test_streamed_word_burst_does_not_block_keepalive_pong(
    monkeypatch, unused_tcp_port
):
    @dataclass
    class SlowWordSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, _cancelled=lambda: False):
            time.sleep(1)
            return iter(())

        def stream_eos(self, _cancelled=lambda: False):
            return iter(())

    config = TtsConfig(
        host="127.0.0.1",
        port=unused_tcp_port,
        hf_repo="unused",
        voice_repo="unused",
        default_voice="unused.wav",
        quantize_bits=None,
        max_gen_length=1000,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundle(),
        session_cls=SlowWordSession,
    )
    monkeypatch.setattr(tts_server_module, "TtsServer", lambda _config: server)

    serve_task = asyncio.create_task(tts_server_module._serve(config))
    try:
        await asyncio.sleep(0.05)
        async with await _connect(unused_tcp_port) as websocket:
            assert await _recv_message(websocket) == {"type": "Ready"}
            for sequence in range(40):
                await websocket.send(
                    msgpack.packb({"type": "Text", "text": f"word-{sequence} "})
                )

            # Give the server time to fill its receive queue and pause socket
            # reads while the first word is still generating.
            await asyncio.sleep(0.05)
            pong = await websocket.ping(b"after-word-burst")
            await asyncio.wait_for(pong, timeout=0.5)
    finally:
        serve_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await serve_task
