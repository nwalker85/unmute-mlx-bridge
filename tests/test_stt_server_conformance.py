"""End-to-end protocol conformance for `SttServer` over a real WebSocket, using a
deterministic fake engine so this runs on any platform with no model weights
(`SttServer.__init__(bundle_loader=..., session_cls=...)` is the injection seam
that makes this possible — see its docstring).

This is the "protocol tests start each service with an in-memory deterministic
engine" suite called for in `docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md`
§Testing Strategy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import msgpack
import pytest
import websockets
from websockets.asyncio.server import serve

from unmute_mlx_bridge.config import SttConfig
from unmute_mlx_bridge.observability import build_process_request
from unmute_mlx_bridge.stt.engine import SttStepEvent
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


async def test_malformed_frame_gets_error_and_stays_connected(stt_server):
    _server, port = stt_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "NotARealType"}))
        error = await _recv_message(ws)
        assert error["type"] == "Error"
        # Connection must still be usable afterwards.
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
        step = await _recv_message(ws)
        assert step["type"] == "Step"


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
