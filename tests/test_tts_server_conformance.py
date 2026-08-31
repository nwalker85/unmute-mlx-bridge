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
from types import SimpleNamespace
from typing import ClassVar

import msgpack
import pytest
import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Request

from unmute_mlx_bridge.tts import server as tts_server_module
from unmute_mlx_bridge.config import TtsConfig
from unmute_mlx_bridge.observability import build_process_request, check_auth
from unmute_mlx_bridge.tts.engine import (
    GenerationLengthLimitError,
    TtsStepEvent,
    VoiceEmbeddingError,
)
from unmute_mlx_bridge.tts.server import PROTOCOL_PATH, TtsServer


class FakeBundle:
    """Stands in for `TtsModelBundle`; carries no MLX state."""

    def resolve_voice(self, voice: str) -> str:
        return voice


class FakeBundleWithCfg:
    """Like `FakeBundle`, but carries a `tts_model.valid_cfg_conditionings`
    (RAV-1552 B2), so `_cfg_alpha_query_error` has something to validate
    against. `FakeBundle` deliberately carries no `tts_model` at all -- most
    conformance tests use it precisely because they don't care about
    cfg_alpha's model-specific validation, and `_cfg_alpha_query_error`
    defensively no-ops when `tts_model` is absent.
    """

    def __init__(self, valid_cfg_conditionings: set[float]) -> None:
        self.tts_model = SimpleNamespace(valid_cfg_conditionings=valid_cfg_conditionings)

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
    applied_voice_embeddings: list[tuple[list[float], list[int]]] = field(
        default_factory=list, init=False
    )
    instances: ClassVar[list[FakeSession]] = []

    def __post_init__(self):
        self.instances.append(self)

    def apply_voice_embedding(self, embeddings: list[float], shape: list[int]) -> None:
        self.applied_voice_embeddings.append((embeddings, shape))

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


@dataclass
class VoiceRejectingFakeSession(FakeSession):
    """A fake engine whose `apply_voice_embedding` always fails, exercising the
    server's "never crash, never silently fall back" handling of an engine-level
    rejection (as opposed to the protocol-level shape/payload-length rejection,
    which never reaches the session at all).
    """

    def apply_voice_embedding(self, embeddings: list[float], shape: list[int]) -> None:
        raise VoiceEmbeddingError(
            "this model does not support multi-speaker conditioning; "
            "custom Voice embeddings cannot be applied"
        )


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


async def _recv_bounded(ws, timeout: float = 5) -> dict:
    """Like `_recv_message`, but bounded (RAV-1552 F4): a regression that
    makes the server stop replying must fail this test, not hang it forever.
    """
    raw = await asyncio.wait_for(ws.recv(decode=False), timeout=timeout)
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
    observed_cfg_coef = None

    def load_bundle(_hf_repo, _voice_repo, _quantize_bits, n_q=None, cfg_coef=None):
        nonlocal observed_n_q, observed_cfg_coef
        observed_n_q = n_q
        observed_cfg_coef = cfg_coef
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
        cfg_coef=2.0,
    )

    server = TtsServer(config)
    await server.load_model()

    assert observed_n_q == 24
    assert observed_cfg_coef == 2.0


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
    server, port = tts_server
    async with await _connect(port, query="format=OggOpus") as ws:
        message = await _recv_message(ws)
        assert message["type"] == "Error"
        assert "PcmMessagePack" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await ws.recv()

    # RAV-1552: every pre-Ready rejection path in `handle_connection` now
    # increments the labelled counter, not just `cfg_alpha` as before.
    assert server.metrics.tts_rejected_sessions_by_reason.labels(reason="format")._value.get() == 1


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


async def test_single_voice_query_still_resolves_to_a_plain_value(tts_server):
    """RAV-1552: adding `voices=` multi-voice blend must not change the
    single-`voice=` path's resolved value shape (a plain string/Path, never
    wrapped in a list)."""
    _server, port = tts_server
    async with await _connect(port, query="format=PcmMessagePack&voice=solo") as ws:
        await _recv_bounded(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": "hi"}))
        await _recv_bounded(ws)  # Audio

    session = FakeSession.instances[-1]
    assert session.voice == "solo"


async def test_voices_blend_resolves_each_entry_and_passes_list_to_session(tts_server):
    """RAV-1552: `voices=` (repeated query param) is a multi-voice blend --
    every entry is resolved through the same voice resolver as `voice=`, and
    the resolved list (not the raw names) is what reaches the session."""
    _server, port = tts_server
    async with await _connect(
        port, query="format=PcmMessagePack&voices=a&voices=b"
    ) as ws:
        await _recv_bounded(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": "hi"}))
        await _recv_bounded(ws)  # Audio

    session = FakeSession.instances[-1]
    assert session.voice == ["a", "b"]


async def test_voice_and_voices_together_is_a_protocol_error_before_ready(tts_server):
    """`voice` and `voices` are mutually exclusive (PROTOCOL.md); giving both
    is rejected before the channel slot is even committed to a session --
    same fail-fast shape as an unsupported `format`."""
    _server, port = tts_server
    async with await _connect(
        port, query="format=PcmMessagePack&voice=solo&voices=a&voices=b"
    ) as ws:
        message = await _recv_bounded(ws)
        assert message["type"] == "Error"
        assert "voice" in message["message"]
        assert "voices" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


async def test_more_than_five_voices_is_a_protocol_error_before_ready(tts_server):
    """`moshi_mlx.models.tts.TTSModel.make_condition_attributes` only ever
    fills 5 speaker slots (`for idx in range(5)`); a 6th entry would be
    silently dropped by the model, so this bridge rejects it explicitly
    instead."""
    server, port = tts_server
    query = "format=PcmMessagePack&" + "&".join(f"voices=v{i}" for i in range(6))
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message["type"] == "Error"
        assert "5" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []
    # RAV-1552: `_voices_query_error` rejections now count too.
    assert server.metrics.tts_rejected_sessions_by_reason.labels(reason="voices")._value.get() == 1


@pytest.mark.parametrize(
    "field_name", ["seed", "top_k", "temperature", "cfg_alpha", "max_seq_len"]
)
async def test_non_numeric_query_value_is_a_protocol_error_before_ready(
    tts_server, field_name
):
    """RAV-1552 F1: `_parse_query`'s bare `int()`/`float()` calls used to run
    outside every `try` in `handle_connection` -- an unparsable numeric query
    value (e.g. `?seed=abc`) raised uncaught, closing the socket with 1011
    and no protocol `Error`, contradicting PROTOCOL.md's documented claim
    that a `cfg_alpha` misuse (and, by the same fail-fast contract, every
    other query value) yields an `Error`. It must now be rejected the same
    way any other pre-Ready validation failure is: an explicit `Error`
    naming only the parameter -- never the raw value or the underlying
    exception text -- then a clean close, before `Ready` is ever sent."""
    server, port = tts_server
    query = f"format=PcmMessagePack&{field_name}=abc"
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message == {
            "type": "Error",
            "message": f"invalid '{field_name}' query parameter",
        }
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []
    # RAV-1552: every pre-Ready rejection now counts under a single reason.
    assert server.metrics.tts_rejected_sessions_by_reason.labels(reason="query")._value.get() == 1


@pytest.mark.parametrize(
    "field_name", ["seed", "top_k", "temperature", "cfg_alpha", "max_seq_len"]
)
async def test_repeated_numeric_query_parameter_is_a_protocol_error_before_ready(
    tts_server, field_name
):
    """RAV-1552 F6: a repeated scalar numeric query parameter (e.g.
    `?seed=1&seed=2`) used to silently take only the first value with no
    signal to the client -- the same silent-drop failure mode `voice=a&
    voice=b` had. It must now be an explicit protocol `Error`."""
    _server, port = tts_server
    query = f"format=PcmMessagePack&{field_name}=1&{field_name}=2"
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message == {
            "type": "Error",
            "message": f"invalid '{field_name}' query parameter",
        }
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


async def test_repeated_voice_query_parameter_is_a_protocol_error_before_ready(
    tts_server,
):
    """RAV-1552 F6: `?voice=a&voice=b` used to silently resolve to `voice=a`
    (`raw_voice_fields["voice"][0]`) and drop `b` entirely, with no signal to
    the client at all."""
    _server, port = tts_server
    async with await _connect(
        port, query="format=PcmMessagePack&voice=a&voice=b"
    ) as ws:
        message = await _recv_bounded(ws)
        assert message == {"type": "Error", "message": "invalid 'voice' query parameter"}
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


def test_parse_query_rejects_non_numeric_values():
    """Unit-level pin for `_parse_query` itself (RAV-1552 F1), mirroring this
    module's existing sync `_voices_query_error`/`_parse_query` unit tests."""
    for field_name in ("seed", "top_k", "temperature", "cfg_alpha", "max_seq_len"):
        with pytest.raises(tts_server_module._QueryError) as exc_info:
            tts_server_module._parse_query(
                f"/api/tts_streaming?format=PcmMessagePack&{field_name}=abc"
            )
        assert exc_info.value.param_name == field_name


def test_parse_query_rejects_repeated_scalar_values():
    """Unit-level pin for `_parse_query` itself (RAV-1552 F6)."""
    for field_name in ("seed", "top_k", "temperature", "cfg_alpha", "max_seq_len"):
        with pytest.raises(tts_server_module._QueryError) as exc_info:
            tts_server_module._parse_query(
                f"/api/tts_streaming?format=PcmMessagePack&{field_name}=1&{field_name}=2"
            )
        assert exc_info.value.param_name == field_name

    with pytest.raises(tts_server_module._QueryError) as exc_info:
        tts_server_module._parse_query(
            "/api/tts_streaming?format=PcmMessagePack&voice=a&voice=b"
        )
    assert exc_info.value.param_name == "voice"


@pytest.mark.parametrize("field_name", ["format", "auth_id"])
async def test_repeated_string_query_parameter_is_a_protocol_error_before_ready(
    tts_server, field_name
):
    """RAV-1552: `format=`/`auth_id=` used to skip the repeated-value guard
    the numeric fields and `voice=` already got -- `?format=a&format=b` (or
    `?auth_id=a&auth_id=b`) silently took `values[0]` and dropped the rest,
    the same F6 silent-drop failure mode, just missed for these two fields."""
    _server, port = tts_server
    query = f"format=PcmMessagePack&{field_name}=a&{field_name}=b"
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message == {
            "type": "Error",
            "message": f"invalid '{field_name}' query parameter",
        }
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


def test_parse_query_rejects_repeated_format_and_auth_id():
    """Unit-level pin for `_parse_query` itself (RAV-1552), mirroring
    `test_parse_query_rejects_repeated_scalar_values` above but for the two
    string fields that previously took `raw[field_name][0]` unconditionally."""
    for field_name in ("format", "auth_id"):
        with pytest.raises(tts_server_module._QueryError) as exc_info:
            tts_server_module._parse_query(
                f"/api/tts_streaming?format=PcmMessagePack&{field_name}=a&{field_name}=b"
            )
        assert exc_info.value.param_name == field_name


@pytest.mark.parametrize(
    "query_suffix,invalid_field",
    [
        ("seed=-1", "seed"),
        ("top_k=0", "top_k"),
        ("top_k=-5", "top_k"),
        ("max_seq_len=0", "max_seq_len"),
        ("max_seq_len=-1", "max_seq_len"),
        ("temperature=-0.1", "temperature"),
        ("temperature=nan", "temperature"),
        ("temperature=inf", "temperature"),
        ("temperature=-inf", "temperature"),
    ],
)
async def test_out_of_range_numeric_query_value_is_a_protocol_error_before_ready(
    tts_server, query_suffix, invalid_field
):
    """RAV-1552: a syntactically valid but out-of-range numeric query value
    (a negative `seed`, a non-positive `top_k`/`max_seq_len`, a non-finite or
    negative `temperature`) used to reach `TtsSession`/the sampler unchecked
    instead of failing fast the same way an unparsable value already does."""
    _server, port = tts_server
    query = f"format=PcmMessagePack&{query_suffix}"
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message == {
            "type": "Error",
            "message": f"invalid '{invalid_field}' query parameter",
        }
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


def test_parse_query_rejects_out_of_range_numeric_values():
    """Unit-level pin for `_parse_query` itself (RAV-1552)."""
    for query_suffix, invalid_field in (
        ("seed=-1", "seed"),
        ("top_k=0", "top_k"),
        ("max_seq_len=0", "max_seq_len"),
        ("temperature=-1", "temperature"),
        ("temperature=nan", "temperature"),
        ("temperature=inf", "temperature"),
    ):
        with pytest.raises(tts_server_module._QueryError) as exc_info:
            tts_server_module._parse_query(
                f"/api/tts_streaming?format=PcmMessagePack&{query_suffix}"
            )
        assert exc_info.value.param_name == invalid_field


def test_parse_query_allows_boundary_and_zero_temperature_values():
    """Companion to the rejection tests above: `seed=0`, `top_k=1`,
    `max_seq_len=1`, and `temperature=0` are all valid boundary values, not
    off-by-one rejections."""
    query = tts_server_module._parse_query(
        "/api/tts_streaming?format=PcmMessagePack&seed=0&top_k=1"
        "&max_seq_len=1&temperature=0"
    )
    assert query.seed == 0
    assert query.top_k == 1
    assert query.max_seq_len == 1
    assert query.temperature == 0.0


@pytest.mark.parametrize(
    "field_name", ["seed", "top_k", "temperature", "cfg_alpha", "max_seq_len", "format", "auth_id"]
)
def test_parse_query_rejects_blank_padded_repeat(field_name):
    """RAV-1552: a *blank-padded* repeat (`?field=&field=value`) used to
    bypass the repeated-value guard entirely -- `_parse_query`'s repeat-count
    checks ran on `parse_qs`'s default (blank-dropping) parse, so the blank
    entry vanished before `len(values) != 1` ever saw it, leaving exactly one
    surviving (non-blank) value. The request was silently accepted (`Ready`)
    instead of rejected -- confirmed live against the real `TtsServer` before
    this fix. Every field must now reject this the same as a plain repeat."""
    value = "PcmMessagePack" if field_name == "format" else "1" if field_name in (
        "seed", "top_k", "max_seq_len"
    ) else "0.5" if field_name in ("temperature", "cfg_alpha") else "good"
    with pytest.raises(tts_server_module._QueryError) as exc_info:
        tts_server_module._parse_query(
            f"/api/tts_streaming?format=PcmMessagePack&{field_name}=&{field_name}={value}"
        )
    assert exc_info.value.param_name == field_name

    # Ordering must not matter: the non-blank value coming first must not
    # sneak past either.
    with pytest.raises(tts_server_module._QueryError) as exc_info:
        tts_server_module._parse_query(
            f"/api/tts_streaming?format=PcmMessagePack&{field_name}={value}&{field_name}="
        )
    assert exc_info.value.param_name == field_name


@pytest.mark.parametrize("field_name", ["format", "auth_id"])
def test_parse_query_rejects_lone_blank_string_field(field_name):
    """RAV-1552: unlike the numeric fields (where `int("")`/`float("")` already
    raise `ValueError`), `format=`/`auth_id=` have no parse step to fail on an
    empty string, so a lone blank value needs its own explicit check -- the
    same treatment a blank `voice=` already gets (RAV-1552 S1)."""
    with pytest.raises(tts_server_module._QueryError) as exc_info:
        tts_server_module._parse_query(f"/api/tts_streaming?format=PcmMessagePack&{field_name}=")
    assert exc_info.value.param_name == field_name


@pytest.mark.parametrize(
    "query_suffix", ["format=&format=PcmMessagePack", "auth_id=&auth_id=good", "seed=&seed=7"]
)
async def test_blank_padded_repeat_is_a_protocol_error_before_ready(tts_server, query_suffix):
    """Wire-level pin for the blank-padded-repeat bypass (RAV-1552), confirmed
    live against the real `TtsServer` before this fix: each of these used to
    get `Ready`, not an `Error`."""
    server, port = tts_server
    query = f"format=PcmMessagePack&{query_suffix}" if not query_suffix.startswith(
        "format="
    ) else query_suffix
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message["type"] == "Error"
        assert "invalid '" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


def test_check_auth_and_parse_query_agree_on_blank_padded_repeated_auth_id():
    """RAV-1552: `check_auth`'s docstring claims it and `_parse_query` can
    never disagree about a repeated/blank `auth_id=`. This pins that for the
    blank-padded case specifically, which `check_auth` used to get wrong
    silently (confirmed live: it authorized `?auth_id=&auth_id=good` against
    `{"good"}` before this fix, the exact query `_parse_query` rejects)."""
    assert (
        check_auth(
            Request("/api/tts_streaming?auth_id=&auth_id=good", Headers()),
            frozenset({"good"}),
        )
        is False
    )
    with pytest.raises(tts_server_module._QueryError):
        tts_server_module._parse_query(
            "/api/tts_streaming?format=PcmMessagePack&auth_id=&auth_id=good"
        )


async def test_repeated_auth_id_query_param_is_rejected_before_upgrade_when_authorized():
    """Companion to the blank-padded protocol-error test above: when
    `authorized_ids` is non-empty, a malformed `auth_id=` (blank-padded
    repeat) is rejected at the pre-upgrade gate (401), never reaching
    `_parse_query` at all -- same as any other invalid `auth_id=`."""
    config = TtsConfig(
        host="127.0.0.1",
        port=0,
        hf_repo="unused",
        voice_repo="unused",
        default_voice="unused.wav",
        quantize_bits=None,
        max_gen_length=1000,
        authorized_ids=frozenset({"good"}),
        log_level="INFO",
    )
    server = TtsServer(config, bundle_loader=lambda: FakeBundle())
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
            await websockets.connect(
                f"ws://127.0.0.1:{port}{PROTOCOL_PATH}?format=PcmMessagePack"
                "&auth_id=&auth_id=good"
            )
        assert exc_info.value.response.status_code == 401


@pytest.mark.parametrize("max_seq_len", [1001, 1_000_000_000])
async def test_max_seq_len_above_operator_cap_is_a_protocol_error_before_ready(
    tts_server, max_seq_len
):
    """RAV-1552: a client-supplied `max_seq_len=` may only *lower* the
    operator's configured `TTS_MAX_GEN_LENGTH` cap, never raise it -- the
    `tts_server` fixture configures `max_gen_length=1000`, so anything above
    that must be rejected the same way any other out-of-range value is.
    Previously any positive value replaced the cap unbounded."""
    _server, port = tts_server
    query = f"format=PcmMessagePack&max_seq_len={max_seq_len}"
    async with await _connect(port, query=query) as ws:
        message = await _recv_bounded(ws)
        assert message == {
            "type": "Error",
            "message": "invalid 'max_seq_len' query parameter",
        }
        with pytest.raises(websockets.exceptions.ConnectionClosedOK):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


def test_parse_query_max_seq_len_cap_boundary():
    """Unit-level pin for the cap boundary itself: exactly the cap is
    allowed, one above it is not."""
    query = tts_server_module._parse_query(
        "/api/tts_streaming?format=PcmMessagePack&max_seq_len=1000", max_seq_len_cap=1000
    )
    assert query.max_seq_len == 1000

    with pytest.raises(tts_server_module._QueryError) as exc_info:
        tts_server_module._parse_query(
            "/api/tts_streaming?format=PcmMessagePack&max_seq_len=1001", max_seq_len_cap=1000
        )
    assert exc_info.value.param_name == "max_seq_len"

    # No cap given (unit-level callers that don't pass one) -> unbounded, as
    # before; only `handle_connection` ever supplies a cap.
    query = tts_server_module._parse_query(
        "/api/tts_streaming?format=PcmMessagePack&max_seq_len=1000000000"
    )
    assert query.max_seq_len == 1_000_000_000


async def test_query_rejection_send_ignores_connection_closed():
    """RAV-1552: every pre-`Ready` rejection branch in `handle_connection`
    sent its `Error` unguarded -- a client that disconnects on its own
    between the upgrade and the rejection makes `connection.send` raise
    `ConnectionClosed`, which used to propagate straight out of
    `handle_connection` uncaught. `_send_rejection` must swallow it; this
    pins the branch directly rather than racing a real disconnect."""

    class _AlreadyGoneConnection:
        def __init__(self, path: str) -> None:
            self.request = SimpleNamespace(path=path, headers={})
            self.closed = False

        async def send(self, _data: object) -> None:
            raise websockets.exceptions.ConnectionClosed(None, None)

        async def close(self) -> None:
            self.closed = True

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
    server = TtsServer(config, bundle_loader=lambda: FakeBundle())
    await server.load_model()
    connection = _AlreadyGoneConnection("/api/tts_streaming?format=PcmMessagePack&seed=abc")

    # Must not raise, despite `send` always raising `ConnectionClosed`.
    await server.handle_connection(connection)

    assert connection.closed is True
    assert server.metrics.tts_rejected_sessions_by_reason.labels(reason="query")._value.get() == 1


#: Stands in for the kind of operator-specific detail a real resolver
#: exception can embed -- `huggingface_hub`'s disk-space `OSError` includes
#: the local cache path, which includes the operator's username. Used by
#: both the `voices=` and single-`voice=` unresolvable-name tests below to
#: pin that this text never reaches the client (RAV-1552 B1/B3).
_INJECTED_EXCEPTION_TEXT = "no space left on device: /home/exampleoperator/.cache/huggingface/hub"


async def test_voices_blend_unresolvable_name_is_a_protocol_error():
    """An entry in `voices=` that the voice resolver cannot find must produce
    an explicit protocol `Error` naming only the client-supplied voice --
    never a silent drop, never an abrupt/crashed close, and never any part of
    the underlying resolver exception (which can embed operator-specific
    detail such as a Hugging Face cache path, per `_INJECTED_EXCEPTION_TEXT`
    above) (RAV-1552 B1)."""

    def voice_resolver(_bundle, voice):
        if voice == "bad":
            raise OSError(_INJECTED_EXCEPTION_TEXT)
        return voice

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
    FakeSession.instances.clear()
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundle(),
        session_cls=FakeSession,
        voice_resolver=voice_resolver,
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(
            port, query="format=PcmMessagePack&voices=good&voices=bad"
        ) as ws:
            # RAV-1552 S3: this test used to hang (not fail) when the fix was
            # reverted, with nothing bounding the receive loop.
            assert await asyncio.wait_for(_recv_message(ws), timeout=5) == {
                "type": "Ready"
            }
            error = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert error["type"] == "Error"
            assert "bad" in error["message"]
            assert "/" not in error["message"]
            assert _INJECTED_EXCEPTION_TEXT not in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


async def test_single_voice_unresolvable_name_is_a_protocol_error():
    """RAV-1552 B3: a single unresolvable `voice=` must get the same `Error`
    treatment as an unresolvable entry in a `voices=` blend -- this path used
    to be a bare `voice_resolution.result()` with no exception handling at
    all, crashing the socket with 1011 and no protocol `Error`."""

    def voice_resolver(_bundle, voice):
        if voice == "bad":
            raise OSError(_INJECTED_EXCEPTION_TEXT)
        return voice

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
    FakeSession.instances.clear()
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundle(),
        session_cls=FakeSession,
        voice_resolver=voice_resolver,
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port, query="format=PcmMessagePack&voice=bad") as ws:
            assert await asyncio.wait_for(_recv_message(ws), timeout=5) == {
                "type": "Ready"
            }
            error = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert error["type"] == "Error"
            assert "bad" in error["message"]
            assert "/" not in error["message"]
            assert _INJECTED_EXCEPTION_TEXT not in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


def test_voices_query_validation_rejects_empty_list():
    """An empty `voices` list can only be reached by constructing
    `TtsStreamingQuery` directly (query-string parsing drops blank repeated
    params), but the validator must still reject it defensively."""
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    query = TtsStreamingQuery(voices=[])

    error = tts_server_module._voices_query_error(query)

    assert error is not None
    assert "empty" in error


def test_voices_query_validation_rejects_more_than_five():
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    query = TtsStreamingQuery(voices=["a", "b", "c", "d", "e", "f"])

    error = tts_server_module._voices_query_error(query)

    assert error is not None
    assert "5" in error


def test_voices_query_validation_rejects_voice_and_voices_together():
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    query = TtsStreamingQuery(voice="solo", voices=["a"])

    error = tts_server_module._voices_query_error(query)

    assert error is not None


def test_voices_query_validation_allows_either_alone_or_neither():
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    assert tts_server_module._voices_query_error(TtsStreamingQuery()) is None
    assert tts_server_module._voices_query_error(TtsStreamingQuery(voice="solo")) is None
    assert (
        tts_server_module._voices_query_error(TtsStreamingQuery(voices=["a", "b"]))
        is None
    )


def test_voices_query_validation_rejects_blank_voice():
    """RAV-1552 S1: a blank `voice=` is rejected explicitly rather than
    treated as "not given"."""
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    error = tts_server_module._voices_query_error(TtsStreamingQuery(voice=""))

    assert error is not None
    assert "blank" in error


def test_voices_query_validation_rejects_blank_entry_in_voices():
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    error = tts_server_module._voices_query_error(TtsStreamingQuery(voices=["a", ""]))

    assert error is not None
    assert "blank" in error


def test_voices_query_validation_rejects_duplicate_entries():
    """RAV-1552 S1: a duplicate `voices=` entry burns a blend slot for no
    effect (`make_condition_attributes` only ever fills 5 speaker slots), so
    it is rejected rather than silently wasting one."""
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    error = tts_server_module._voices_query_error(TtsStreamingQuery(voices=["a", "a"]))

    assert error is not None
    assert "duplicate" in error


def test_voices_query_validation_blank_voice_with_voices_hits_mutual_exclusivity():
    """RAV-1552 S1: a blank `voice=` alongside a `voices=` is documented (see
    `_voices_query_error`'s docstring and PROTOCOL.md) to hit the
    mutual-exclusivity error, not the blank-`voice=` one -- a `voice=`
    present on the wire at all, blank or not, counts as "given"."""
    from unmute_mlx_bridge.protocol.tts import TtsStreamingQuery

    error = tts_server_module._voices_query_error(
        TtsStreamingQuery(voice="", voices=["a"])
    )

    assert error is not None
    assert "cannot specify both" in error


def test_parse_query_keeps_blank_voice_and_voices_values():
    """RAV-1552 S1: `_parse_query` must not silently drop a blank `voice=`/
    `voices=` value the way `parse_qs`'s default `keep_blank_values=False`
    would -- `?voices=` used to fall back to the single default voice with no
    signal to the client at all (`voices` key absent from the parsed query
    entirely)."""
    query = tts_server_module._parse_query("/api/tts_streaming?format=PcmMessagePack&voices=")

    assert query.voices == [""]

    query = tts_server_module._parse_query("/api/tts_streaming?format=PcmMessagePack&voice=")

    assert query.voice == ""

    query = tts_server_module._parse_query(
        "/api/tts_streaming?format=PcmMessagePack&voices=&voices=a"
    )

    assert query.voices == ["", "a"]


async def test_blank_voices_entry_is_a_protocol_error_before_ready(tts_server):
    _server, port = tts_server
    async with await _connect(port, query="format=PcmMessagePack&voices=") as ws:
        message = await asyncio.wait_for(_recv_message(ws), timeout=5)
        assert message["type"] == "Error"
        assert "blank" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


async def test_duplicate_voices_entries_are_a_protocol_error_before_ready(tts_server):
    _server, port = tts_server
    async with await _connect(
        port, query="format=PcmMessagePack&voices=a&voices=a"
    ) as ws:
        message = await asyncio.wait_for(_recv_message(ws), timeout=5)
        assert message["type"] == "Error"
        assert "duplicate" in message["message"]
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []


async def test_cfg_alpha_unsupported_value_is_a_protocol_error_before_ready():
    """RAV-1552 B2: an unsupported `cfg_alpha` is validated against the
    loaded model's `valid_cfg_conditionings` before a channel slot is even
    taken (same fail-fast shape as `format=`/`voices=`) -- no `Ready` is sent
    first. This used to reach `TtsSession.__post_init__`'s bare `ValueError`
    raise, uncaught by `handle_connection`, crashing the socket with 1011 and
    no protocol `Error`.
    """
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
    FakeSession.instances.clear()
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundleWithCfg({1.0, 1.5, 2.0}),
        session_cls=FakeSession,
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(
            port, query="format=PcmMessagePack&cfg_alpha=1.7"
        ) as ws:
            message = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert message["type"] == "Error"
            assert "cfg_alpha" in message["message"]
            # `connection.close()` defaults to code 1000 -- `ConnectionClosedOK`
            # is specifically "1000 (OK) or 1001, or no code" (see the brief's
            # "close 1000" requirement), not merely any `ConnectionClosed`.
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=5)

    assert FakeSession.instances == []
    # RAV-1552: moved from the unlabelled `protocol_errors` counter onto the
    # same labelled `tts_rejected_sessions_by_reason` counter every other
    # pre-Ready rejection path now uses, so this is the only pre-Ready path
    # that no longer touches `protocol_errors` at all.
    cfg_alpha_counter = server.metrics.tts_rejected_sessions_by_reason.labels(reason="cfg_alpha")
    assert cfg_alpha_counter._value.get() == 1
    assert server.metrics.protocol_errors._value.get() == 0


async def test_cfg_alpha_supported_value_still_starts_a_session():
    """Companion to the rejection test above: a *supported* `cfg_alpha`
    against a model that does have `valid_cfg_conditionings` must not be
    rejected."""
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
    FakeSession.instances.clear()
    server = TtsServer(
        config,
        bundle_loader=lambda: FakeBundleWithCfg({1.0, 1.5, 2.0}),
        session_cls=FakeSession,
    )
    await server.load_model()
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(
            port, query="format=PcmMessagePack&cfg_alpha=1.5"
        ) as ws:
            assert await asyncio.wait_for(_recv_message(ws), timeout=5) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hi"}))
            await asyncio.wait_for(_recv_message(ws), timeout=5)  # Audio

    assert len(FakeSession.instances) == 1
    assert FakeSession.instances[-1].cfg_alpha == 1.5


async def test_session_construction_failure_is_a_protocol_error_not_a_crash():
    """RAV-1552 B2 defense in depth: any construction failure `TtsSession`
    (or a test double standing in for it) raises -- not just the cfg_alpha
    mismatch the pre-lock check above already intercepts -- must still become
    a sanitized protocol `Error` and a clean close, never an uncaught
    exception leaving the socket dying with 1011 and no `Error`. The message
    is fully generic (never the exception text)."""

    @dataclass
    class ExplodingSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def __post_init__(self):
            raise ValueError("private construction failure detail")

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
    server = TtsServer(
        config, bundle_loader=lambda: FakeBundle(), session_cls=ExplodingSession
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
            error = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert error["type"] == "Error"
            assert "private construction failure detail" not in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=5)

    assert server.metrics.protocol_errors._value.get() == 1


async def test_streaming_text_length_limit_gets_dedicated_error_and_metric():
    """RAV-1552 S2: a session hitting its configured length limit is a
    legitimate terminal condition, not a generation failure -- it must get
    its own message and its own `reason="length_limit"` metric label,
    distinct from `reason="generation"`."""

    @dataclass
    class LengthLimitedSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, _cancelled=lambda: False):
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            raise GenerationLengthLimitError(
                "reached max_gen_length=1000; reconnect to start a fresh session"
            )

        def stream_eos(self, _cancelled=lambda: False):
            return iter(())

    async with _running_tts_server(session_cls=LengthLimitedSession) as (
        server,
        port,
    ):
        async with await _connect(port) as ws:
            assert await asyncio.wait_for(_recv_message(ws), timeout=5) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            audio = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert audio["type"] == "Audio"
            error = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert error["type"] == "Error"
            assert "length limit" in error["message"]
            assert "reconnect" in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=5)

    assert (
        server.metrics.streaming_failures.labels(reason="length_limit")._value.get()
        == 1
    )
    assert (
        server.metrics.streaming_failures.labels(reason="generation")._value.get()
        == 0
    )


def _make_bare_tts_server() -> tuple[TtsServer, TtsConfig]:
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
    return TtsServer(config, bundle_loader=lambda: FakeBundle(), session_cls=FakeSession), config


async def test_model_still_loading_message_before_any_load_failure():
    """`load_model` is never called here, so `bundle` stays `None` and
    `health.model_load_error` stays `None` -- a connecting client must see
    "model still loading", not "model failed to load" (RAV-1552 B4)."""
    server, config = _make_bare_tts_server()
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

    # RAV-1552: pre-Ready rejections while the model is loading/failed now
    # count too, both sharing the `loading` reason label.
    assert server.metrics.tts_rejected_sessions_by_reason.labels(reason="loading")._value.get() == 1


async def test_model_failed_to_load_message_after_load_failure():
    """Once `health.mark_load_failed` has actually recorded a failure (real
    `load_model` calls this on any load exception), a connecting client must
    see "model failed to load" -- see README.md's "If model load fails"
    section for the documented rationale for staying up and serving health
    probes rather than exiting (RAV-1552 B4)."""
    server, config = _make_bare_tts_server()
    server.health.mark_load_failed("model load failed, see logs")
    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )
    async with serve(
        server.handle_connection, config.host, config.port, process_request=process_request
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        async with await _connect(port) as ws:
            message = await asyncio.wait_for(_recv_message(ws), timeout=5)
            assert message == {"type": "Error", "message": "model failed to load"}

    assert server.metrics.tts_rejected_sessions_by_reason.labels(reason="loading")._value.get() == 1


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


async def test_streaming_text_generation_failure_emits_error_then_closes():
    """RAV-1552: in `streaming` delivery mode, a generation failure while
    processing a `Text` message used to re-raise all the way out of
    `_emit_stream` -- caught nowhere (`handle_connection` only catches
    `ConnectionClosed`/`_ClientDisconnected`) -- so the socket closed
    abruptly with no protocol `Error`. It must now emit a sanitized `Error`
    (no internal exception text) and then close cleanly, mirroring the
    buffered-turn generation-failure handling.
    """

    @dataclass
    class FailingStreamSession:
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

    async with _running_tts_server(session_cls=FailingStreamSession) as (
        server,
        port,
    ):
        async with await _connect(port) as ws:
            assert await _recv_bounded(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            audio = await _recv_bounded(ws)
            assert audio["type"] == "Audio"
            error = await _recv_bounded(ws)
            assert error["type"] == "Error"
            assert "private model failure detail" not in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert (
        server.metrics.streaming_failures.labels(reason="generation")._value.get()
        == 1
    )


async def test_streaming_eos_generation_failure_emits_error_then_closes():
    """Same failure shape as above, but during the trailing Eos flush (the
    second `_emit_stream` call site, `_finish_session`'s streaming branch)."""

    @dataclass
    class FailingEosSession:
        bundle: object
        voice: str | None
        max_gen_length: int
        seed: int = 42
        temperature: float = 0.8
        top_k: int = 250
        cfg_alpha: float | None = None

        def stream_text(self, _text: str, _cancelled=lambda: False):
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])

        def stream_eos(self, _cancelled=lambda: False):
            raise RuntimeError("private model failure detail")
            yield  # pragma: no cover -- makes this a generator

    async with _running_tts_server(session_cls=FailingEosSession) as (
        server,
        port,
    ):
        async with await _connect(port) as ws:
            assert await _recv_bounded(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Text", "text": "hello"}))
            audio = await _recv_bounded(ws)
            assert audio["type"] == "Audio"
            await ws.send(msgpack.packb({"type": "Eos"}))
            error = await _recv_bounded(ws)
            assert error["type"] == "Error"
            assert "private model failure detail" not in error["message"]
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert (
        server.metrics.streaming_failures.labels(reason="generation")._value.get()
        == 1
    )


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


async def test_voice_message_before_text_conditions_session_at_start(tts_server):
    """RAV-1504: `Voice` at session start (before any `Text`) is accepted as an
    alternative to the `voice=`/`voices=` query parameters, matching real
    `moshi-server`'s `py_module.rs::InMsg::Voice`.
    """
    _server, port = tts_server
    embeddings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    shape = [1, 2, 3]
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(
            msgpack.packb({"type": "Voice", "embeddings": embeddings, "shape": shape})
        )
        await ws.send(msgpack.packb({"type": "Text", "text": "hi"}))
        audio = await _recv_message(ws)
        assert audio["type"] == "Audio"

    session = FakeSession.instances[-1]
    assert session.applied_voice_embeddings == [(embeddings, shape)]
    assert session.text_inputs == ["hi"]


async def test_voice_message_after_text_is_rejected_once_generation_started(
    tts_server,
):
    """A `Voice` message arriving after `Text` has already started the turn is
    rejected -- a deliberate, stricter-than-upstream deviation, not a
    compatibility gap: real `moshi-server` reads a connection's pending voice
    only on the channel-init entry (`rust/moshi-server/tts.py:340-353`,
    `rust/moshi-server/src/py_module.rs:237-240`) and would silently drop this
    message instead of erroring.
    """
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": "hi"}))
        await _recv_message(ws)  # Audio
        await _recv_message(ws)  # Text (word event)

        await ws.send(
            msgpack.packb({"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 2]})
        )
        error = await _recv_message(ws)
        assert error["type"] == "Error"
        assert "generation started" in error["message"]
        assert "fixed at session start" in error["message"]

        # Connection stays open and keeps generating from the original voice.
        await ws.send(msgpack.packb({"type": "Text", "text": "still works"}))
        audio = await _recv_message(ws)
        assert audio["type"] == "Audio"

    session = FakeSession.instances[-1]
    assert session.applied_voice_embeddings == []


async def test_empty_text_message_locks_voice_conditioning_to_session_start(
    tts_server,
):
    """"Before any Text message" is read literally: even an empty `Text` message
    (which the server otherwise ignores entirely) ends the session-start window
    for `Voice`.
    """
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(msgpack.packb({"type": "Text", "text": ""}))
        await ws.send(
            msgpack.packb({"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 2]})
        )
        error = await _recv_message(ws)
        assert error["type"] == "Error"
        assert "generation started" in error["message"]

    session = FakeSession.instances[-1]
    assert session.applied_voice_embeddings == []


async def test_voice_message_shape_payload_mismatch_is_a_protocol_error(tts_server):
    """`shape` is validated against the flattened `embeddings` length at the
    protocol layer (`TtsVoiceMessage`'s pydantic model validator) -- a mismatch
    never reaches the engine, and is reported the same way any other malformed
    frame is: a fully generic message (RAV-1552 B1 -- the pydantic
    `ValidationError` raised here is caught by the same `except Exception:`
    that handles a msgpack-unpack failure, which has no safe subset to select
    from a generic exception, so the client-facing message no longer echoes
    any part of it, including this validator's own otherwise-safe "shape"/
    dimension text). The full detail is still logged server-side.
    """
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(
            msgpack.packb({"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 3]})
        )
        error = await asyncio.wait_for(_recv_message(ws), timeout=5)
        assert error == {"type": "Error", "message": "malformed frame"}

        # Never a crash, never a silent fallback: the connection stays open and
        # still works with whatever voice the session already had.
        await ws.send(msgpack.packb({"type": "Text", "text": "still works"}))
        audio = await asyncio.wait_for(_recv_message(ws), timeout=5)
        assert audio["type"] == "Audio"

    session = FakeSession.instances[-1]
    assert session.applied_voice_embeddings == []


async def test_voice_message_zero_dimension_is_a_protocol_error_never_silently_accepted(
    tts_server,
):
    """Confirmed defect (RAV-1504 blocker): `shape=[1, 512, 0]` with
    `embeddings=[]` used to be silently accepted end to end (the flattened
    length `0` matched `len([]) == 0`) -- never a `Text` message would fail,
    the engine would never surface an error, and the session's conditioning
    would just be quietly discarded. It is now a protocol `Error` (a fully
    generic "malformed frame" message -- see the shape/payload-mismatch test
    above for why, RAV-1552 B1), and never reaches the engine.
    """
    _server, port = tts_server
    async with await _connect(port) as ws:
        await _recv_message(ws)  # Ready
        await ws.send(
            msgpack.packb({"type": "Voice", "embeddings": [], "shape": [1, 512, 0]})
        )
        error = await asyncio.wait_for(_recv_message(ws), timeout=5)
        assert error == {"type": "Error", "message": "malformed frame"}

        await ws.send(msgpack.packb({"type": "Text", "text": "still works"}))
        audio = await asyncio.wait_for(_recv_message(ws), timeout=5)
        assert audio["type"] == "Audio"

    session = FakeSession.instances[-1]
    assert session.applied_voice_embeddings == []


async def test_voice_message_rejected_by_engine_is_a_protocol_error_never_a_crash():
    """A malformed/mismatched embedding that passes protocol-level shape
    validation but that the engine cannot apply (e.g. a model without
    multi-speaker support) must still return a protocol `Error`, never crash the
    connection, and never silently keep generating as if nothing happened
    without telling the client.
    """
    FakeSession.instances.clear()
    async with _running_tts_server(session_cls=VoiceRejectingFakeSession) as (
        _server,
        port,
    ):
        async with await _connect(port) as ws:
            await _recv_message(ws)  # Ready
            await ws.send(
                msgpack.packb(
                    {"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 2]}
                )
            )
            error = await _recv_message(ws)
            assert error["type"] == "Error"
            assert "invalid voice embedding" in error["message"]

            # Connection stays open and still works afterward.
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
