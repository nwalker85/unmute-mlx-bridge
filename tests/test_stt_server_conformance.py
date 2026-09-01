"""End-to-end protocol conformance for `SttServer` over a real WebSocket, using a
deterministic fake engine so this runs on any platform with no model weights
(`SttServer.__init__(bundle_loader=..., session_cls=...)` is the injection seam
that makes this possible — see its docstring).

This is the "protocol tests start each service with an in-memory deterministic
engine" suite called for in `docs/design/architecture.md`
§Testing Strategy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import msgpack
import pytest
import websockets
from websockets.asyncio.server import serve

from unmute_mlx_bridge.config import SttConfig
from unmute_mlx_bridge.observability import build_process_request
from unmute_mlx_bridge.stt.engine import GenerationLengthLimitError, SttStepEvent
from unmute_mlx_bridge.stt.server import PROTOCOL_PATH, SttServer


class FakeBundle:
    """Stands in for `SttModelBundle`; carries no MLX state."""

    asr_delay_in_tokens = 1


@dataclass
class FakeSession:
    """Deterministic fake of `SttSession`: echoes a fixed word per Audio frame and
    tracks marker scheduling exactly like the real one (delay of `asr_delay_in_tokens`
    calls), so the marker-ordering test is meaningful.
    """

    bundle: object
    max_steps: int
    _audio_calls: int = field(default=0, init=False)
    _pending_markers: list[tuple[int, int]] = field(default_factory=list, init=False)

    def push_marker(self, marker_id: int) -> None:
        target = self._audio_calls + self.bundle.asr_delay_in_tokens
        self._pending_markers.append((target, marker_id))

    def push_audio(self, pcm: list[float]) -> list[SttStepEvent]:
        self._audio_calls += 1
        events = [
            SttStepEvent(kind="step", step_idx=self._audio_calls, prs=[0.1, 0.2, 0.3, 0.4]),
            SttStepEvent(kind="word", text=f"word{self._audio_calls}", start_time=0.0),
        ]
        still_pending = []
        for target, marker_id in self._pending_markers:
            if target <= self._audio_calls:
                events.append(SttStepEvent(kind="marker", marker_id=marker_id))
            else:
                still_pending.append((target, marker_id))
        self._pending_markers = still_pending
        return events


@pytest.fixture
async def stt_server():
    config = SttConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield server, port


async def _connect(port: int):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}{PROTOCOL_PATH}",
        additional_headers={"kyutai-api-key": "public_token"},
    )


async def _recv_message(ws) -> dict:
    raw = await ws.recv(decode=False)
    return msgpack.unpackb(raw)


async def _recv_bounded(ws, timeout: float = 5) -> dict:
    """Like `_recv_message`, but bounded (RAV-1552 F4): a regression that
    makes the server stop replying must fail this test, not hang it forever.
    """
    raw = await asyncio.wait_for(ws.recv(decode=False), timeout=timeout)
    return msgpack.unpackb(raw)


async def test_connect_receives_ready(stt_server):
    _server, port = stt_server
    async with await _connect(port) as ws:
        message = await _recv_message(ws)
        assert message == {"type": "Ready"}


async def test_audio_produces_step_then_word(stt_server):
    _server, port = stt_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
        step = await _recv_message(ws)
        word = await _recv_message(ws)
        assert step["type"] == "Step"
        assert step["prs"] == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-6)
        assert word["type"] == "Word"
        assert word["text"] == "word1"


async def test_marker_is_echoed_after_delay(stt_server):
    _server, port = stt_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Marker", "id": 99}))
        # asr_delay_in_tokens=1, so one Audio frame should surface the marker.
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
        seen_types = []
        for _ in range(3):
            seen_types.append((await _recv_message(ws))["type"])
        assert "Marker" in seen_types


#: Stands in for detail a pydantic `ValidationError`/msgpack-unpack failure
#: could otherwise embed in its own text -- mirrors `test_tts_server_
#: conformance.py`'s malformed-frame tests, which pin the same "fully
#: generic 'malformed frame' message, nothing else" contract (RAV-1552 B1/F5).
_INJECTED_DETAIL_TEXT = "private-parse-detail-9f3c1a"


async def test_malformed_frame_gets_error_and_stays_connected(stt_server):
    _server, port = stt_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(
            msgpack.packb({"type": "NotARealType", "detail": _INJECTED_DETAIL_TEXT})
        )
        error = await _recv_bounded(ws)
        assert error == {"type": "Error", "message": "malformed frame"}
        assert _INJECTED_DETAIL_TEXT not in error["message"]
        # Connection must still be usable afterwards.
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
        step = await _recv_bounded(ws)
        assert step["type"] == "Step"


async def test_generation_failure_emits_error_then_closes():
    """RAV-1552: `push_audio` runs directly on the per-message path (no
    generator boundary to cross, unlike TTS's streaming/buffered turns), so a
    generation exception used to propagate straight out of `_run_session`,
    uncaught by `handle_connection`'s `except ConnectionClosed:` -- an
    abrupt close with no protocol `Error`. It must now emit a sanitized
    `Error` (no internal exception text) and then close cleanly.
    """

    @dataclass
    class FailingSession:
        bundle: object
        max_steps: int

        def push_marker(self, marker_id: int) -> None:
            pass

        def push_audio(self, pcm: list[float]) -> list[SttStepEvent]:
            raise RuntimeError("private model failure detail")

    config = SttConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FailingSession)
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port) as ws:
            assert await _recv_bounded(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
            error = await _recv_bounded(ws)
            assert error["type"] == "Error"
            assert "private model failure detail" not in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert (
        server.metrics.stt_generation_failures.labels(reason="generation")._value.get()
        == 1
    )


async def test_length_limit_failure_gets_dedicated_error_and_metric():
    """RAV-1552 S2: a session hitting its configured `max_steps` is a
    legitimate terminal condition, not a generation failure -- it must get
    its own message and its own `reason="length_limit"` metric label,
    distinct from `reason="generation"`."""

    @dataclass
    class LengthLimitedSession:
        bundle: object
        max_steps: int

        def push_marker(self, marker_id: int) -> None:
            pass

        def push_audio(self, pcm: list[float]) -> list[SttStepEvent]:
            raise GenerationLengthLimitError(
                "reached max_steps=1000; reconnect to start a fresh session"
            )

    config = SttConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=LengthLimitedSession)
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port) as ws:
            assert await asyncio.wait_for(_recv_message(ws), timeout=5) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
            error = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert error["type"] == "Error"
            assert "length limit" in error["message"]
            assert "reconnect" in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=5)

    assert (
        server.metrics.stt_generation_failures.labels(reason="length_limit")._value.get()
        == 1
    )
    assert (
        server.metrics.stt_generation_failures.labels(reason="generation")._value.get()
        == 0
    )


async def test_push_audio_connection_closed_is_not_swallowed_as_generation_failure():
    """RAV-1552 S5: `push_audio`'s exception handling must not swallow
    `ConnectionClosed` into the generic `except Exception:` branch, unlike
    the pre-existing TTS call sites, which already exclude it. It must
    propagate to `handle_connection`'s own `except ConnectionClosed:`
    (counted as a cancellation), never miscounted as a generation failure.
    `push_audio` itself has no reason to actually raise this in production;
    this is defense in depth, not a fix for an observed crash.
    """

    @dataclass
    class ConnectionClosedSession:
        bundle: object
        max_steps: int

        def push_marker(self, marker_id: int) -> None:
            pass

        def push_audio(self, pcm: list[float]) -> list[SttStepEvent]:
            raise websockets.exceptions.ConnectionClosed(None, None)

    config = SttConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = SttServer(
        config, bundle_loader=lambda: FakeBundle(), session_cls=ConnectionClosedSession
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port) as ws:
            assert await asyncio.wait_for(_recv_message(ws), timeout=5) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
            async with asyncio.timeout(2):
                while server.metrics.cancellations._value.get() < 1:
                    await asyncio.sleep(0)

    assert (
        server.metrics.stt_generation_failures.labels(reason="generation")._value.get()
        == 0
    )
    assert (
        server.metrics.stt_generation_failures.labels(reason="length_limit")._value.get()
        == 0
    )


async def test_second_concurrent_session_is_rejected(stt_server):
    _server, port = stt_server
    first = await _connect(port)
    await _recv_message(first)  # Ready

    second = await _connect(port)
    message = await _recv_message(second)
    assert message["type"] == "Error"
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await second.recv()

    await first.close()


async def test_disconnect_frees_the_slot_for_the_next_session(stt_server):
    server, port = stt_server
    first = await _connect(port)
    await _recv_message(first)
    await first.close()
    await asyncio.sleep(0.05)  # let the server notice the close

    assert not server.health.snapshot()["session_active"]

    async with await _connect(port) as second:
        message = await _recv_message(second)
        assert message == {"type": "Ready"}


async def test_unauthorized_connection_is_rejected_before_upgrade(stt_server):
    _server, port = stt_server
    with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
        await websockets.connect(f"ws://127.0.0.1:{port}{PROTOCOL_PATH}")
    assert exc_info.value.response.status_code == 401


async def test_model_still_loading_is_a_counted_pre_ready_rejection():
    """RAV-1552: mirrors `tts/server.py`'s `tts_rejected_sessions_by_reason`
    fix -- STT's `handle_connection` has exactly one pre-Ready rejection path
    that previously emitted no metric at all (the "still loading"/"failed to
    load" gate; "no free channels" already had its own `rejected_sessions`
    counter). `load_model` is never called here, so `bundle` stays `None`."""
    config = SttConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port) as ws:
            message = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert message == {"type": "Error", "message": "model still loading"}

    assert server.metrics.stt_rejected_sessions_by_reason.labels(reason="loading")._value.get() == 1


async def test_loading_rejection_send_ignores_connection_closed():
    """RAV-1552: mirrors `tts/server.py`'s equivalent pin -- STT's "still
    loading" pre-`Ready` rejection sent its `Error` unguarded; a client that
    disconnects on its own between the upgrade and the rejection makes
    `connection.send` raise `ConnectionClosed`, which used to propagate
    straight out of `handle_connection` uncaught. `_send_rejection` must
    swallow it."""

    class _AlreadyGoneConnection:
        def __init__(self) -> None:
            self.request = SimpleNamespace(path=PROTOCOL_PATH, headers={})
            self.closed = False

        async def send(self, _data: object) -> None:
            raise websockets.exceptions.ConnectionClosed(None, None)

        async def close(self) -> None:
            self.closed = True

    config = SttConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    connection = _AlreadyGoneConnection()

    # Must not raise, despite `send` always raising `ConnectionClosed`.
    # `load_model` is never called, so `bundle` stays `None` and this hits
    # the "still loading" branch.
    await server.handle_connection(connection)

    assert connection.closed is True
    assert server.metrics.stt_rejected_sessions_by_reason.labels(reason="loading")._value.get() == 1
