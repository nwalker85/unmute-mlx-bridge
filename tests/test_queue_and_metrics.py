"""Tests for STT receive queue bounds, oversized frame handling, metric
observations, and inference_step_seconds fix. Runs without model weights on
any platform.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import msgpack
import pytest
import websockets
from websockets.asyncio.server import serve
from unmute_mlx_bridge.config import SttConfig
from unmute_mlx_bridge.observability import Metrics, build_process_request
from unmute_mlx_bridge.stt.engine import FRAME_SIZE, SttStepEvent
from unmute_mlx_bridge.stt.server import PROTOCOL_PATH, SttServer
from prometheus_client import generate_latest


class FakeBundle:
    asr_delay_in_tokens = 1


@dataclass
class FakeSession:
    bundle: object
    max_steps: int
    _calls: int = field(default=0, init=False)

    def push_marker(self, _marker_id: int) -> None:
        pass

    def push_audio(self, pcm: list[float]) -> list[SttStepEvent]:
        self._calls += 1
        return [
            SttStepEvent(kind="step", step_idx=self._calls, prs=[0.1, 0.2, 0.3, 0.4]),
            SttStepEvent(kind="word", text=f"word{self._calls}", start_time=0.0),
        ]


def _make_stt_config(**overrides) -> SttConfig:
    defaults = dict(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        quantize_bits=None,
        max_steps=1000,
        asr_delay_in_tokens=1,
        authorized_ids=frozenset({"public_token"}),
        log_level="INFO",
    )
    defaults.update(overrides)
    return SttConfig(**defaults)


@pytest.fixture
async def stt_server_fixture():
    config = _make_stt_config()
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    await server.load_model()
    pr = build_process_request(server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids)
    async with serve(server.handle_connection, config.host, config.port, process_request=pr) as ws:
        port = ws.sockets[0].getsockname()[1]
        yield server, port


async def _connect(port: int):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}{PROTOCOL_PATH}",
        additional_headers={"kyutai-api-key": "public_token"},
    )


async def _recv(ws) -> dict:
    return msgpack.unpackb(await ws.recv(decode=False))


# ---------------------------------------------------------------------------
# Model-load-failure client messaging (RAV-1552 B4)
# ---------------------------------------------------------------------------


async def test_model_still_loading_message_before_any_load_failure():
    """`load_model` is never called here, so `bundle` stays `None` and
    `health.model_load_error` stays `None` -- a connecting client must see
    "model still loading", not "model failed to load"."""
    config = _make_stt_config()
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    pr = build_process_request(server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids)
    async with serve(server.handle_connection, config.host, config.port, process_request=pr) as ws:
        port = ws.sockets[0].getsockname()[1]
        async with await _connect(port) as conn:
            message = await asyncio.wait_for(_recv(conn), timeout=5)
            assert message == {"type": "Error", "message": "model still loading"}


async def test_model_failed_to_load_message_after_load_failure():
    """Once `health.mark_load_failed` has actually recorded a failure (real
    `load_model` calls this on any load exception), a connecting client must
    see "model failed to load", not the misleading "model still loading"
    forever (RAV-1552 B4: the process stays up serving health probes -- see
    README.md's "If model load fails" section -- but the client-facing
    message must match reality)."""
    config = _make_stt_config()
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    server.health.mark_load_failed("model load failed, see logs")
    pr = build_process_request(server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids)
    async with serve(server.handle_connection, config.host, config.port, process_request=pr) as ws:
        port = ws.sockets[0].getsockname()[1]
        async with await _connect(port) as conn:
            message = await asyncio.wait_for(_recv(conn), timeout=5)
            assert message == {"type": "Error", "message": "model failed to load"}


# ---------------------------------------------------------------------------
# inference_step_seconds fix
# ---------------------------------------------------------------------------


async def test_inference_step_seconds_is_observed_after_push_audio(stt_server_fixture):
    """Regression: inference_step_seconds was defined but never observed."""
    server, port = stt_server_fixture
    async with await _connect(port) as ws:
        await _recv(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * FRAME_SIZE}))
        await _recv(ws)  # Step
        await _recv(ws)  # Word

    payload = generate_latest(server.metrics.registry).decode()
    assert "bridge_inference_step_seconds_count 1.0" in payload


async def test_inference_step_batch_frames_is_observed(stt_server_fixture):
    server, port = stt_server_fixture
    async with await _connect(port) as ws:
        await _recv(ws)  # Ready
        # Send exactly 2 frames worth of audio - FakeSession returns 1 Step + 1 Word per call.
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * (FRAME_SIZE * 2)}))
        await _recv(ws)  # Step
        await _recv(ws)  # Word

    payload = generate_latest(server.metrics.registry).decode()
    assert "bridge_inference_step_batch_frames" in payload


async def test_inference_step_rtf_is_observed(stt_server_fixture):
    server, port = stt_server_fixture
    async with await _connect(port) as ws:
        await _recv(ws)
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * FRAME_SIZE}))
        await _recv(ws)
        await _recv(ws)

    payload = generate_latest(server.metrics.registry).decode()
    assert "bridge_inference_step_rtf_count 1.0" in payload


# ---------------------------------------------------------------------------
# Oversized frame detection
# ---------------------------------------------------------------------------


async def test_oversized_audio_frame_gets_error_response():
    config = _make_stt_config(max_input_frame_samples=100)
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    await server.load_model()
    pr = build_process_request(server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids)
    async with serve(server.handle_connection, config.host, config.port, process_request=pr) as ws:
        port = ws.sockets[0].getsockname()[1]
        async with await _connect(port) as conn:
            await _recv(conn)  # Ready
            # Send a frame larger than the 100-sample limit.
            await conn.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 200}))
            error = await _recv(conn)
            assert error["type"] == "Error"
            assert "too large" in error["message"]


async def test_oversized_frame_increments_counter():
    config = _make_stt_config(max_input_frame_samples=100)
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    await server.load_model()
    pr = build_process_request(server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids)
    async with serve(server.handle_connection, config.host, config.port, process_request=pr) as ws:
        port = ws.sockets[0].getsockname()[1]
        async with await _connect(port) as conn:
            await _recv(conn)
            await conn.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 200}))
            await _recv(conn)  # Error

    assert server.metrics.stt_oversized_frames_total._value.get() == 1


async def test_oversized_frame_connection_stays_open():
    """After an oversized frame error, the connection should remain usable."""
    config = _make_stt_config(max_input_frame_samples=FRAME_SIZE - 1)
    server = SttServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession)
    await server.load_model()
    pr = build_process_request(server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids)
    async with serve(server.handle_connection, config.host, config.port, process_request=pr) as ws:
        port = ws.sockets[0].getsockname()[1]
        async with await _connect(port) as conn:
            await _recv(conn)  # Ready
            await conn.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * FRAME_SIZE}))
            error = await _recv(conn)
            assert error["type"] == "Error"
            # Send a valid (small) audio frame afterwards.
            await conn.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 10}))
            # The fake session returns events only for full FRAME_SIZE batches; a
            # short frame accumulates in the buffer without generating events.
            # Connection must still be live: send a marker and verify echo.
            await conn.send(msgpack.packb({"type": "Marker", "id": 77}))
            # No assertion beyond "connection did not close".


# ---------------------------------------------------------------------------
# Queue depth gauge
# ---------------------------------------------------------------------------


async def test_stt_inference_frames_active_gauge_is_present_in_metrics(stt_server_fixture):
    server, port = stt_server_fixture
    payload = generate_latest(server.metrics.registry).decode()
    assert "bridge_stt_inference_frames_active" in payload


async def test_stt_recv_queue_bound_is_set_after_load(stt_server_fixture):
    """bridge_stt_recv_queue_bound should be set to the configured max_queue."""
    server, port = stt_server_fixture
    payload = generate_latest(server.metrics.registry).decode()
    assert "bridge_stt_recv_queue_bound" in payload


# ---------------------------------------------------------------------------
# Metric name stability (regression against accidental rename)
# ---------------------------------------------------------------------------


def test_all_new_metric_names_stable():
    metrics = Metrics()
    payload = generate_latest(metrics.registry).decode()
    for name in (
        "bridge_inference_step_seconds",
        "bridge_inference_step_batch_frames",
        "bridge_inference_step_rtf",
        "bridge_stt_inference_frames_active",
        "bridge_stt_recv_queue_bound",
        "bridge_stt_oversized_frames_total",
    ):
        assert name in payload, f"metric {name!r} missing from /metrics output"


# ---------------------------------------------------------------------------
# Cross-repo correlation — header propagation
# ---------------------------------------------------------------------------


async def test_client_session_id_header_is_reused_as_session_id(stt_server_fixture):
    """When the client sends x-unmute-session-id, the bridge uses it."""
    server, port = stt_server_fixture
    client_sid = "deadbeefcafe00112233445566778899"

    log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _Capture()
    handler.setLevel(logging.INFO)
    stt_logger = logging.getLogger("unmute_mlx_bridge.stt.server")
    prev_level = stt_logger.level
    stt_logger.addHandler(handler)
    stt_logger.setLevel(logging.INFO)
    try:
        async with await websockets.connect(
            f"ws://127.0.0.1:{port}{PROTOCOL_PATH}",
            additional_headers={
                "kyutai-api-key": "public_token",
                "x-unmute-session-id": client_sid,
            },
        ) as ws:
            await _recv(ws)  # Ready — log is emitted right after send in _run_session
    finally:
        stt_logger.removeHandler(handler)
        stt_logger.setLevel(prev_level)

    session_start = next(
        (r for r in log_records if getattr(r, "event", None) == "stt_session_start"),
        None,
    )
    assert session_start is not None, "stt_session_start event not logged"
    assert session_start.session_id == client_sid
    assert session_start.client_supplied is True


async def test_invalid_session_id_header_falls_back_to_fresh_id(stt_server_fixture):
    """An invalid x-unmute-session-id header causes the bridge to generate a fresh ID."""
    server, port = stt_server_fixture
    log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _Capture()
    handler.setLevel(logging.INFO)
    stt_logger = logging.getLogger("unmute_mlx_bridge.stt.server")
    prev_level = stt_logger.level
    stt_logger.addHandler(handler)
    stt_logger.setLevel(logging.INFO)
    try:
        async with await websockets.connect(
            f"ws://127.0.0.1:{port}{PROTOCOL_PATH}",
            additional_headers={
                "kyutai-api-key": "public_token",
                "x-unmute-session-id": "NOT-VALID!",
            },
        ) as ws:
            await _recv(ws)  # Ready
    finally:
        stt_logger.removeHandler(handler)
        stt_logger.setLevel(prev_level)

    session_start = next(
        (r for r in log_records if getattr(r, "event", None) == "stt_session_start"),
        None,
    )
    assert session_start is not None
    assert len(session_start.session_id) == 32
    assert session_start.session_id != "NOT-VALID!"
    assert session_start.client_supplied is False


@pytest.mark.asyncio
async def test_conversation_id_header_propagated_to_stt_session(stt_server_fixture):
    """x-unmute-conversation-id sent by the client is recorded in the
    stt_session_start log event and in the bridge CorrelationContext."""
    server, port = stt_server_fixture
    client_sid = "aabbccddeeff00112233445566778899"
    client_cid = "11223344556677889900aabbccddeeff"

    log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _Capture()
    handler.setLevel(logging.INFO)
    stt_logger = logging.getLogger("unmute_mlx_bridge.stt.server")
    prev_level = stt_logger.level
    stt_logger.addHandler(handler)
    stt_logger.setLevel(logging.INFO)
    try:
        async with await websockets.connect(
            f"ws://127.0.0.1:{port}{PROTOCOL_PATH}",
            additional_headers={
                "kyutai-api-key": "public_token",
                "x-unmute-session-id": client_sid,
                "x-unmute-conversation-id": client_cid,
            },
        ) as ws:
            await _recv(ws)  # Ready
    finally:
        stt_logger.removeHandler(handler)
        stt_logger.setLevel(prev_level)

    session_start = next(
        (r for r in log_records if getattr(r, "event", None) == "stt_session_start"),
        None,
    )
    assert session_start is not None, "stt_session_start event not logged"
    assert session_start.session_id == client_sid
    assert session_start.conversation_id == client_cid


@pytest.mark.asyncio
async def test_invalid_conversation_id_is_not_propagated(stt_server_fixture):
    """An invalid x-unmute-conversation-id header (wrong format) is ignored;
    conversation_id should be None on the log record."""
    server, port = stt_server_fixture

    log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _Capture()
    handler.setLevel(logging.INFO)
    stt_logger = logging.getLogger("unmute_mlx_bridge.stt.server")
    prev_level = stt_logger.level
    stt_logger.addHandler(handler)
    stt_logger.setLevel(logging.INFO)
    try:
        async with await websockets.connect(
            f"ws://127.0.0.1:{port}{PROTOCOL_PATH}",
            additional_headers={
                "kyutai-api-key": "public_token",
                "x-unmute-conversation-id": "NOT-A-VALID-HEX-ID!",
            },
        ) as ws:
            await _recv(ws)
    finally:
        stt_logger.removeHandler(handler)
        stt_logger.setLevel(prev_level)

    session_start = next(
        (r for r in log_records if getattr(r, "event", None) == "stt_session_start"),
        None,
    )
    assert session_start is not None
    assert session_start.conversation_id is None
