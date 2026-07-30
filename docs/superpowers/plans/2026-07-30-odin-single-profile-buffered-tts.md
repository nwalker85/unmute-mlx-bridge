# Odin Single-Profile Buffered TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Odin-selectable buffered-turn TTS delivery mode that uses one
q8, full-32-codebook model and emits a complete assistant turn atomically so
stock Unmute plays it continuously.

**Architecture:** `TtsConfig` gains an explicit delivery mode and bounded
input/output limits. `TtsServer` retains its existing streaming path as the
default, while `buffered_turn` preserves raw text chunks through `Eos`,
generates and collects the complete turn on the existing worker-thread path,
and emits events only after generation succeeds. Odin selects q8/q32 and
`buffered_turn` through PM2 environment variables; no second model is loaded.

**Tech Stack:** Python 3.12, asyncio, websockets, msgpack, Prometheus client,
pytest, uv, MLX 0.26.5, moshi-mlx 0.3.0, PM2 on Odin.

## Global Constraints

- Work only in
  `/Users/nate/var/worktrees/unmute-mlx-bridge/fix-tts-receive-queue`.
- Preserve the existing `streaming` mode as the default.
- Odin's candidate profile is exactly `TTS_N_Q=32`,
  `TTS_QUANTIZE_BITS=8`, and `TTS_DELIVERY_MODE=buffered_turn`.
- Default resource limits are exactly `TTS_MAX_BUFFERED_CHARS=4096` and
  `TTS_MAX_BUFFERED_AUDIO_SECONDS=60`.
- Reject unknown delivery modes and non-positive resource limits at startup.
- Preserve raw text chunks with `"".join(chunks)`; do not strip, insert, or
  normalize whitespace.
- Emit no audio or word event until the complete buffered turn succeeds.
- Preserve seed, temperature, top-k, CFG, voice, maximum sequence settings,
  event ordering, word timestamps, and `PcmMessagePack` framing.
- Keep all 32 generated codebooks. Do not restore partial codebook generation.
- Do not change MLX, moshi-mlx, or other dependency versions.
- Do not change stock Unmute, its frontend AudioWorklet, SIP, G.711, or
  telephony routing.
- Do not move TTS to Fenrir or load a second resident model.
- Use only Codex sessions or Codex agents for execution. Do not invoke Claude,
  Anthropic-backed OpenCode, or any Anthropic model.
- Portable tests must not initialize Metal or download model weights.
- Use PM2 for the Odin process; never run the TTS server as a raw background
  process.
- Do not edit implementation files directly on Odin. Deploy only a pushed
  repository commit.
- Keep Forgejo PR #5 unmerged until Nate explicitly approves PR #5.

---

### Task 1: Define The Buffered Delivery Configuration Contract

**Files:**
- Modify: `src/unmute_mlx_bridge/config.py:9-101`
- Test: `tests/test_config.py:1-27`

**Interfaces:**
- Produces: `TtsDeliveryMode =
  Literal["streaming", "buffered_turn"]`.
- Produces: `TtsConfig.delivery_mode: TtsDeliveryMode`.
- Produces: `TtsConfig.max_buffered_chars: int`.
- Produces: `TtsConfig.max_buffered_audio_seconds: float`.
- Preserves: all existing direct `TtsConfig` constructor call sites through
  defaulted fields.

- [ ] **Step 1: Write failing default and override tests**

Add `pytest` and tests that clear every new environment variable before
checking defaults:

```python
import pytest


def test_tts_buffered_delivery_defaults(monkeypatch):
    for name in (
        "TTS_DELIVERY_MODE",
        "TTS_MAX_BUFFERED_CHARS",
        "TTS_MAX_BUFFERED_AUDIO_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = TtsConfig.from_env()

    assert config.delivery_mode == "streaming"
    assert config.max_buffered_chars == 4096
    assert config.max_buffered_audio_seconds == 60.0


def test_tts_buffered_delivery_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TTS_DELIVERY_MODE", "buffered_turn")
    monkeypatch.setenv("TTS_MAX_BUFFERED_CHARS", "2048")
    monkeypatch.setenv("TTS_MAX_BUFFERED_AUDIO_SECONDS", "30.5")

    config = TtsConfig.from_env()

    assert config.delivery_mode == "buffered_turn"
    assert config.max_buffered_chars == 2048
    assert config.max_buffered_audio_seconds == 30.5
```

- [ ] **Step 2: Write failing invalid-value tests**

Add exact validation cases:

```python
def test_tts_rejects_unknown_delivery_mode(monkeypatch):
    monkeypatch.setenv("TTS_DELIVERY_MODE", "burst")

    with pytest.raises(ValueError, match="TTS_DELIVERY_MODE"):
        TtsConfig.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TTS_MAX_BUFFERED_CHARS", "0"),
        ("TTS_MAX_BUFFERED_CHARS", "-1"),
        ("TTS_MAX_BUFFERED_AUDIO_SECONDS", "0"),
        ("TTS_MAX_BUFFERED_AUDIO_SECONDS", "-0.5"),
    ],
)
def test_tts_rejects_non_positive_buffer_limits(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        TtsConfig.from_env()
```

- [ ] **Step 3: Run the focused tests and verify red**

Run:

```bash
uv run --locked pytest tests/test_config.py -q
```

Expected: the new tests fail because `TtsConfig` does not expose the delivery
mode or buffer-limit fields and does not validate their environment values.

- [ ] **Step 4: Implement the minimal validated configuration**

Add the typing import and alias:

```python
from typing import Literal, cast

TtsDeliveryMode = Literal["streaming", "buffered_turn"]
```

Append defaulted fields after `n_q`:

```python
    n_q: int = 24
    delivery_mode: TtsDeliveryMode = "streaming"
    max_buffered_chars: int = 4096
    max_buffered_audio_seconds: float = 60.0
```

At the start of `TtsConfig.from_env`, parse and validate:

```python
        authorized_ids = os.getenv("TTS_AUTHORIZED_IDS", "public_token")
        delivery_mode = os.getenv("TTS_DELIVERY_MODE", "streaming")
        if delivery_mode not in ("streaming", "buffered_turn"):
            raise ValueError(
                "TTS_DELIVERY_MODE must be 'streaming' or 'buffered_turn'"
            )
        max_buffered_chars = _env_int("TTS_MAX_BUFFERED_CHARS", 4096)
        if max_buffered_chars <= 0:
            raise ValueError("TTS_MAX_BUFFERED_CHARS must be positive")
        max_buffered_audio_seconds = _env_float(
            "TTS_MAX_BUFFERED_AUDIO_SECONDS", 60.0
        )
        if max_buffered_audio_seconds <= 0:
            raise ValueError(
                "TTS_MAX_BUFFERED_AUDIO_SECONDS must be positive"
            )
```

Pass the validated values from the constructor:

```python
            n_q=_env_int("TTS_N_Q", 24),
            delivery_mode=cast(TtsDeliveryMode, delivery_mode),
            max_buffered_chars=max_buffered_chars,
            max_buffered_audio_seconds=max_buffered_audio_seconds,
```

- [ ] **Step 5: Run focused and configuration-adjacent tests**

Run:

```bash
uv run --locked pytest tests/test_config.py tests/test_smoke.py -q
```

Expected: all selected tests pass and existing direct `TtsConfig`
constructions remain valid.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add src/unmute_mlx_bridge/config.py tests/test_config.py
git commit -m "feat: configure buffered TTS delivery" \
  -m "Co-Authored-By: OpenAI Codex <codex@openai.com>"
```

---

### Task 2: Buffer And Atomically Emit Successful Turns

**Files:**
- Modify: `src/unmute_mlx_bridge/tts/server.py:12-389`
- Test: `tests/test_tts_server_conformance.py:8-485`

**Interfaces:**
- Consumes: `TtsConfig.delivery_mode` and
  `TtsConfig.max_buffered_chars` from Task 1.
- Produces:
  `TtsServer._collect_stream(connection, events, cancelled) -> list[TtsStepEvent]`.
- Produces:
  `TtsServer._finish_session(connection, session, buffered_chunks, first_text_at, first_output_sent) -> bool`,
  shared by typed and legacy EOS handling.
- Preserves: `TtsServer._emit_stream(connection, events, first_text_at,
  first_output_sent, cancelled)` for default streaming mode.
- Preserves: `TtsServer._emit(connection, events, first_text_at,
  first_output_sent)` as the only wire-emission and output-metric path.

- [ ] **Step 1: Add a reusable real-WebSocket test server helper**

Import `asynccontextmanager`:

```python
from contextlib import asynccontextmanager
```

Add this helper below `FakeSession`:

```python
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
        server.health,
        server.metrics,
        PROTOCOL_PATH,
        config.authorized_ids,
    )
    async with serve(
        server.handle_connection,
        config.host,
        config.port,
        process_request=process_request,
    ) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield server, port
```

Change the existing `tts_server` fixture to delegate to it:

```python
@pytest.fixture
async def tts_server():
    FakeSession.instances.clear()
    async with _running_tts_server() as running:
        yield running
```

Run the existing conformance file before adding behavior tests:

```bash
uv run --locked pytest tests/test_tts_server_conformance.py -q
```

Expected: all existing tests pass; this step changes test setup only.

- [ ] **Step 2: Capture exact text passed into the fake session**

Add a per-instance list and append the unmodified value:

```python
    text_inputs: list[str] = field(default_factory=list, init=False)

    def stream_text(self, text: str, _cancelled=lambda: False):
        self.text_inputs.append(text)
        for word in text.split():
            self._step += 1
            yield TtsStepEvent(kind="audio", pcm=[0.1, 0.2, 0.3])
            yield TtsStepEvent(
                kind="word",
                text=word,
                start_s=self._step - 1,
                stop_s=self._step,
            )
```

- [ ] **Step 3: Write the failing buffered EOS and event-order tests**

Add a helper that drains messages until the normal server close:

```python
async def _recv_until_closed(ws) -> list[dict]:
    messages: list[dict] = []
    try:
        while True:
            messages.append(await _recv_message(ws))
    except websockets.exceptions.ConnectionClosedOK:
        return messages
```

Exercise typed and legacy EOS through one parameterized test:

```python
@pytest.mark.parametrize(
    "eos_frame",
    [
        msgpack.packb({"type": "Eos"}),
        b"\x00",
    ],
)
async def test_buffered_turn_waits_for_eos_and_preserves_event_order(eos_frame):
    FakeSession.instances.clear()
    async with _running_tts_server(delivery_mode="buffered_turn") as (_server, port):
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
    assert [message["text"] for message in messages if message["type"] == "Text"] == [
        "hello",
        "world",
    ]
```

Add the empty-turn contract:

```python
async def test_buffered_empty_turn_closes_without_generation():
    FakeSession.instances.clear()
    async with _running_tts_server(delivery_mode="buffered_turn") as (_server, port):
        async with await _connect(port) as ws:
            assert await _recv_message(ws) == {"type": "Ready"}
            await ws.send(msgpack.packb({"type": "Eos"}))
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await asyncio.wait_for(ws.recv(), timeout=0.2)

    assert FakeSession.instances[-1].text_inputs == []
```

- [ ] **Step 4: Write the failing atomic-success test**

Use a fake that yields one event before blocking:

```python
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
```

- [ ] **Step 5: Run the new tests and verify red**

Run:

```bash
uv run --locked pytest \
  tests/test_tts_server_conformance.py::test_buffered_turn_waits_for_eos_and_preserves_event_order \
  tests/test_tts_server_conformance.py::test_buffered_empty_turn_closes_without_generation \
  tests/test_tts_server_conformance.py::test_buffered_turn_emits_nothing_until_generation_completes \
  -q
```

Expected: failures show that text is still generated and emitted immediately
and that typed and raw EOS do not share a buffered completion path.

- [ ] **Step 6: Implement private collection without changing streaming**

Add this module-level generator beside `_next_event`; unlike
`itertools.chain`, closing it propagates to the currently active delegate:

```python
def _buffered_turn_events(
    session: TtsSession,
    text: str,
    cancelled: Callable[[], bool],
) -> Iterator[TtsStepEvent]:
    yield from session.stream_text(text, cancelled)
    yield from session.stream_eos(cancelled)
```

Add `_collect_stream` beside `_emit_stream`. It must retain the same
disconnect and worker-drain behavior while appending instead of emitting:

```python
    async def _collect_stream(
        self,
        connection: ServerConnection,
        events: Iterator[TtsStepEvent],
        cancelled: threading.Event,
    ) -> list[TtsStepEvent]:
        connection_closed = asyncio.create_task(connection.wait_closed())
        next_event: asyncio.Task[TtsStepEvent | None] | None = None
        collected: list[TtsStepEvent] = []
        try:
            while True:
                next_event = asyncio.create_task(
                    asyncio.to_thread(_next_event, events)
                )
                done, _pending = await asyncio.wait(
                    (next_event, connection_closed),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if connection_closed in done:
                    cancelled.set()
                    try:
                        await next_event
                    except Exception:
                        pass
                    raise _ClientDisconnected
                event = next_event.result()
                if event is None:
                    return collected
                collected.append(event)
        finally:
            cancelled.set()
            connection_closed.cancel()
            try:
                await connection_closed
            except asyncio.CancelledError:
                pass
            if next_event is not None and not next_event.done():
                try:
                    await asyncio.shield(next_event)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            close_events = getattr(events, "close", None)
            if close_events is not None:
                close_events()
```

- [ ] **Step 7: Implement one EOS completion method**

Add the shared method:

```python
    async def _finish_session(
        self,
        connection: ServerConnection,
        session: TtsSession,
        buffered_chunks: list[str],
        first_text_at: float | None,
        first_output_sent: bool,
    ) -> bool:
        cancelled = threading.Event()
        if self.config.delivery_mode == "streaming":
            return await self._emit_stream(
                connection,
                session.stream_eos(cancelled.is_set),
                first_text_at,
                first_output_sent,
                cancelled,
            )
        if not buffered_chunks:
            return first_output_sent
        events = _buffered_turn_events(
            session,
            "".join(buffered_chunks),
            cancelled.is_set,
        )
        collected = await self._collect_stream(
            connection,
            events,
            cancelled,
        )
        return await self._emit(
            connection,
            collected,
            first_text_at,
            first_output_sent,
        )
```

In `_run_session`, initialize:

```python
        buffered_chunks: list[str] = []
        buffered_chars = 0
```

For both raw `b"\x00"` and typed `TtsEosMessage`, call `_finish_session`,
close, and return.

For `TtsTextMessage`, preserve the current branch for `streaming`. For
`buffered_turn`, enforce the incremental character limit before appending:

```python
                if self.config.delivery_mode == "buffered_turn":
                    next_chars = buffered_chars + len(message.text)
                    if next_chars > self.config.max_buffered_chars:
                        await connection.send(
                            pack_message(
                                TtsErrorMessage(
                                    message="buffered TTS input exceeds configured limit"
                                )
                            )
                        )
                        await connection.close()
                        return
                    buffered_chunks.append(message.text)
                    buffered_chars = next_chars
                    continue
```

- [ ] **Step 8: Run successful-turn and existing streaming tests**

Run:

```bash
uv run --locked pytest tests/test_tts_server_conformance.py -q
```

Expected: all conformance tests pass. Existing streaming tests still observe
audio before generation ends; new buffered tests observe no output until EOS
and complete generation.

- [ ] **Step 9: Commit successful atomic buffering**

```bash
git add src/unmute_mlx_bridge/tts/server.py tests/test_tts_server_conformance.py
git commit -m "feat: buffer complete TTS turns" \
  -m "Co-Authored-By: OpenAI Codex <codex@openai.com>"
```

---

### Task 3: Enforce Atomic Failures, Cancellation, And Metrics

**Files:**
- Modify: `src/unmute_mlx_bridge/observability.py:108-165`
- Modify: `src/unmute_mlx_bridge/tts/server.py:46-389`
- Test: `tests/test_observability.py:1-37`
- Test: `tests/test_tts_server_conformance.py:248-485`

**Interfaces:**
- Consumes: `_collect_stream`, `_finish_session`, and buffer configuration
  from Tasks 1-2.
- Produces: `_BufferedAudioLimitExceeded`.
- Produces: these `Metrics` collectors:
  `buffered_input_characters`, `buffered_audio_seconds`,
  `buffered_synthesis_seconds`, `buffered_eos_to_first_emit_seconds`, and
  `buffered_turn_failures`.
- Produces failure-label values exactly:
  `input_limit`, `output_limit`, `generation`, and `disconnect`.

- [ ] **Step 1: Write failing metrics-surface tests**

Import `generate_latest`:

```python
from prometheus_client import generate_latest
```

Add:

```python
def test_buffered_tts_metrics_have_stable_names_and_failure_labels():
    metrics = Metrics()
    metrics.buffered_input_characters.observe(11)
    metrics.buffered_audio_seconds.observe(1.5)
    metrics.buffered_synthesis_seconds.observe(2.5)
    metrics.buffered_eos_to_first_emit_seconds.observe(2.6)
    metrics.buffered_turn_failures.labels(reason="generation").inc()

    payload = generate_latest(metrics.registry).decode()

    assert "bridge_tts_buffered_input_characters_count 1.0" in payload
    assert "bridge_tts_buffered_audio_seconds_count 1.0" in payload
    assert "bridge_tts_buffered_synthesis_seconds_count 1.0" in payload
    assert "bridge_tts_buffered_eos_to_first_emit_seconds_count 1.0" in payload
    assert (
        'bridge_tts_buffered_turn_failures_total{reason="generation"} 1.0'
        in payload
    )
```

Run:

```bash
uv run --locked pytest \
  tests/test_observability.py::test_buffered_tts_metrics_have_stable_names_and_failure_labels \
  -q
```

Expected: failure because the collectors do not exist.

- [ ] **Step 2: Add the Prometheus collectors**

Add dataclass fields:

```python
    buffered_input_characters: Histogram = field(init=False)
    buffered_audio_seconds: Histogram = field(init=False)
    buffered_synthesis_seconds: Histogram = field(init=False)
    buffered_eos_to_first_emit_seconds: Histogram = field(init=False)
    buffered_turn_failures: Counter = field(init=False)
```

Initialize them in `Metrics.__post_init__`:

```python
        self.buffered_input_characters = Histogram(
            "bridge_tts_buffered_input_characters",
            "Characters retained for one buffered TTS turn",
            registry=self.registry,
        )
        self.buffered_audio_seconds = Histogram(
            "bridge_tts_buffered_audio_seconds",
            "Audio seconds retained for one buffered TTS turn",
            registry=self.registry,
        )
        self.buffered_synthesis_seconds = Histogram(
            "bridge_tts_buffered_synthesis_seconds",
            "Wall time from EOS to complete buffered TTS synthesis",
            registry=self.registry,
        )
        self.buffered_eos_to_first_emit_seconds = Histogram(
            "bridge_tts_buffered_eos_to_first_emit_seconds",
            "Wall time from EOS to first buffered TTS emission",
            registry=self.registry,
        )
        self.buffered_turn_failures = Counter(
            "bridge_tts_buffered_turn_failures_total",
            "Buffered TTS turn failures by reason",
            labelnames=("reason",),
            registry=self.registry,
        )
```

Run:

```bash
uv run --locked pytest tests/test_observability.py -q
```

Expected: all observability tests pass.

- [ ] **Step 3: Write failing atomic-error and output-limit tests**

Add a generation failure after a private audio event:

```python
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
```

Add an output-limit test using the existing three-sample fake audio:

```python
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
```

- [ ] **Step 4: Write the failing incremental input-limit test**

```python
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
```

- [ ] **Step 5: Write failing pre-EOS and in-generation disconnect tests**

For pre-EOS discard, close after one text frame and verify a new session can
connect:

```python
async def test_buffered_disconnect_before_eos_discards_text_and_frees_slot():
    FakeSession.instances.clear()
    async with _running_tts_server(
        delivery_mode="buffered_turn"
    ) as (server, port):
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
```

Add the complete in-generation cancellation test:

```python
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
        finally:
            force_release.set()

    assert (
        server.metrics.buffered_turn_failures.labels(
            reason="disconnect"
        )._value.get()
        == 1
    )
```

- [ ] **Step 6: Run the failure tests and verify red**

Run:

```bash
uv run --locked pytest \
  tests/test_tts_server_conformance.py::test_buffered_generation_failure_emits_only_sanitized_error \
  tests/test_tts_server_conformance.py::test_buffered_audio_limit_discards_the_complete_partial_turn \
  tests/test_tts_server_conformance.py::test_buffered_input_limit_rejects_before_retaining_extra_text \
  tests/test_tts_server_conformance.py::test_buffered_disconnect_before_eos_discards_text_and_frees_slot \
  -q
```

Expected: generation errors currently escape, the audio limit is not
enforced, and failure metrics are not incremented.

- [ ] **Step 7: Enforce the output limit inside private collection**

Add the private exception:

```python
class _BufferedAudioLimitExceeded(Exception):
    """Internal signal that buffered PCM exceeded its configured bound."""
```

Extend `_collect_stream`:

```python
    async def _collect_stream(
        self,
        connection: ServerConnection,
        events: Iterator[TtsStepEvent],
        cancelled: threading.Event,
        max_audio_samples: int,
    ) -> tuple[list[TtsStepEvent], int]:
```

Initialize `audio_samples = 0`. Before appending each audio event:

```python
                if event.kind == "audio":
                    audio_samples += len(event.pcm or [])
                    if audio_samples > max_audio_samples:
                        raise _BufferedAudioLimitExceeded
                collected.append(event)
```

Return `(collected, audio_samples)`.

- [ ] **Step 8: Make buffered completion atomic for every failure**

In `_finish_session`, record `eos_started_at = time.monotonic()` and calculate:

```python
        max_audio_samples = int(
            self.config.max_buffered_audio_seconds * 24_000
        )
```

Wrap collection with exact failure behavior:

```python
        try:
            collected, audio_samples = await self._collect_stream(
                connection,
                events,
                cancelled,
                max_audio_samples,
            )
        except _BufferedAudioLimitExceeded:
            self.metrics.buffered_turn_failures.labels(
                reason="output_limit"
            ).inc()
            await connection.send(
                pack_message(
                    TtsErrorMessage(
                        message="buffered TTS output exceeds configured limit"
                    )
                )
            )
            return first_output_sent
        except _ClientDisconnected:
            self.metrics.buffered_turn_failures.labels(
                reason="disconnect"
            ).inc()
            raise
        except Exception:
            self.metrics.buffered_turn_failures.labels(
                reason="generation"
            ).inc()
            logger.exception("tts: buffered turn generation failed")
            await connection.send(
                pack_message(
                    TtsErrorMessage(
                        message="buffered TTS generation failed"
                    )
                )
            )
            return first_output_sent
```

On success, observe metrics before emission:

```python
        synthesis_seconds = time.monotonic() - eos_started_at
        self.metrics.buffered_input_characters.observe(
            len("".join(buffered_chunks))
        )
        self.metrics.buffered_audio_seconds.observe(
            audio_samples / 24_000
        )
        self.metrics.buffered_synthesis_seconds.observe(
            synthesis_seconds
        )
        self.metrics.buffered_eos_to_first_emit_seconds.observe(
            time.monotonic() - eos_started_at
        )
```

Then call `_emit`. `_run_session` closes after `_finish_session` returns, so
error paths send no buffered event and close normally.

In the incremental input-limit branch, increment:

```python
                        self.metrics.buffered_turn_failures.labels(
                            reason="input_limit"
                        ).inc()
```

Ensure a buffered connection that ends before EOS increments `disconnect`
exactly once, discards its chunk list, and releases the existing session lock.
Initialize a terminal-state flag before the current receive loop:

```python
        buffered_terminal_handled = False
```

Set it to `True` immediately before:

- handling raw `b"\x00"`;
- handling typed `TtsEosMessage`;
- sending and closing for the incremental input limit.

Wrap the existing receive loop in `try/finally` and add this exact finalizer:

```python
        finally:
            if (
                self.config.delivery_mode == "buffered_turn"
                and not buffered_terminal_handled
            ):
                self.metrics.buffered_turn_failures.labels(
                    reason="disconnect"
                ).inc()
```

This leaves the flag false only when the peer ends the connection before an
explicit terminal path. A disconnect during generation is counted inside
`_finish_session`, after the EOS branch has already set the flag, so it is not
double-counted.

- [ ] **Step 9: Run focused and full portable tests**

Run:

```bash
uv run --locked pytest \
  tests/test_observability.py \
  tests/test_tts_server_conformance.py \
  -q
```

Then:

```bash
uv run --locked pytest -q
```

Expected: every portable test passes; hardware tests remain deselected.

- [ ] **Step 10: Commit failure atomicity and observability**

```bash
git add \
  src/unmute_mlx_bridge/observability.py \
  src/unmute_mlx_bridge/tts/server.py \
  tests/test_observability.py \
  tests/test_tts_server_conformance.py
git commit -m "feat: enforce atomic buffered TTS turns" \
  -m "Co-Authored-By: OpenAI Codex <codex@openai.com>"
```

---

### Task 4: Verify And Update Forgejo PR #5

**Files:**
- Verify all files modified in Tasks 1-3.
- Do not modify `main`, the passive GitHub mirror, or Odin in this task.

**Interfaces:**
- Consumes: the three implementation commits from Tasks 1-3.
- Produces: one exact pushed candidate SHA on
  `origin/fix/tts-receive-queue`.
- Produces: current Forgejo PR #5 state and exact head confirmation.

- [ ] **Step 1: Reconcile every worktree and branch**

Run:

```bash
git worktree list --porcelain
git status --short --branch
git log -6 --oneline --decorate
```

Expected: the dedicated worktree is on `fix/tts-receive-queue`; no unrelated
worktree changed; the branch contains the approved design, plan, and
implementation commits.

- [ ] **Step 2: Run locked verification**

Run each command independently and preserve each exit code:

```bash
uv sync --locked
git diff --check origin/fix/tts-receive-queue...HEAD
uv run --locked pytest -q
```

Expected: lockfile sync succeeds without dependency changes, diff check is
clean, and all portable tests pass with hardware tests explicitly deselected.

- [ ] **Step 3: Review the exact branch delta**

Run:

```bash
git diff --stat origin/fix/tts-receive-queue...HEAD
git diff origin/fix/tts-receive-queue...HEAD -- \
  src/unmute_mlx_bridge/config.py \
  src/unmute_mlx_bridge/observability.py \
  src/unmute_mlx_bridge/tts/server.py \
  tests/test_config.py \
  tests/test_observability.py \
  tests/test_tts_server_conformance.py \
  docs/superpowers/specs/2026-07-30-odin-single-profile-buffered-tts-design.md \
  docs/superpowers/plans/2026-07-30-odin-single-profile-buffered-tts.md
```

Expected: no dependency, frontend, partial-codebook, SIP, G.711, Fenrir, or
second-model change appears.

- [ ] **Step 4: Push only the PR branch**

```bash
git push origin fix/tts-receive-queue
```

Never push `main`, `develop`, `github`, or andvari.

- [ ] **Step 5: Confirm Forgejo PR #5 exact state**

```bash
FJ_TTS_TOKEN="$(
  op item get 'Forgejo PAT (mimir-ts migration)' \
    --vault ravenmask \
    --fields credential \
    --reveal
)"
FJ_TTS_BASE="http://hrafngud.taild7ad1f.ts.net:3300/api/v1"
CANDIDATE_SHA="$(git rev-parse HEAD)"
curl -fsS \
  -H "Authorization: token ${FJ_TTS_TOKEN}" \
  "${FJ_TTS_BASE}/repos/nate/unmute-mlx-bridge/pulls/5" \
  | jq --arg sha "${CANDIDATE_SHA}" \
      '{number,state,mergeable,merged,head_sha:.head.sha,head_matches:(.head.sha==$sha)}'
```

Then enforce the gate programmatically:

```bash
curl -fsS \
  -H "Authorization: token ${FJ_TTS_TOKEN}" \
  "${FJ_TTS_BASE}/repos/nate/unmute-mlx-bridge/pulls/5" \
  | jq -e --arg sha "${CANDIDATE_SHA}" \
      '.number == 5 and .state == "open" and .mergeable == true and .merged == false and .head.sha == $sha'
```

Expected: `jq -e` prints `true` and exits zero. Do not merge.

---

### Task 5: Deploy The Exact Odin Canary And Run Physical Gates

**Files:**
- Deploy from:
  `/Users/ravenhelm/var/canaries/unmute-mlx-bridge-pr5`
- Modify local evidence tool only:
  `/Users/nate/var/ravenhelm/slices/timber.yellow.branch/tts_stream_probe.py`
- Save private WAV evidence under:
  `/Users/nate/var/ravenhelm/slices/timber.yellow.branch/`

**Interfaces:**
- Consumes: the pushed `CANDIDATE_SHA` from Task 4.
- Produces: exact deployed SHA, filtered TTS environment, `/readyz` proof,
  two deterministic controlled WAVs, and Nate's physical canary verdict.
- Rollback target:
  `9a1cbf02eb6a6e7e2af5ecfbc7de1d99824e658b`, unquantized q32 streaming.

- [ ] **Step 1: Add EOS timing to the private probe**

In the slice-local probe, capture the time immediately after sending EOS:

```python
        await ws.send(msgpack.packb({"type": "Eos"}))
        eos_at = time.monotonic()
```

After collection, calculate:

```python
    eos_to_first_audio_s = audio_times[0] - eos_at
```

Include
`eos_to_first_audio={eos_to_first_audio_s:.3f}s` in the existing summary
line. Do not commit the probe or its private voice identifier to the repo.

- [ ] **Step 2: Deploy the exact pushed commit through git**

From the local worktree:

```bash
ODIN_TTS_CANDIDATE="$(git rev-parse HEAD)"
git ls-remote origin refs/heads/fix/tts-receive-queue
```

Confirm the remote SHA equals `ODIN_TTS_CANDIDATE`, then deploy:

```bash
ssh odin-ts "zsh -lic 'cd /Users/ravenhelm/var/canaries/unmute-mlx-bridge-pr5 && git fetch origin fix/tts-receive-queue && git checkout --detach ${ODIN_TTS_CANDIDATE} && uv sync --locked'"
```

This is a repository deployment. Do not patch Python or environment files on
Odin.

- [ ] **Step 3: Restart only the TTS canary with the selected profile**

```bash
ssh odin-ts 'zsh -lic '\''
  TTS_QUANTIZE_BITS=8 \
  TTS_N_Q=32 \
  TTS_DELIVERY_MODE=buffered_turn \
  TTS_MAX_BUFFERED_CHARS=4096 \
  TTS_MAX_BUFFERED_AUDIO_SECONDS=60 \
  pm2 restart unmute-canary-tts --update-env
'\'''
```

Wait for model load in intervals shorter than 60 seconds. Do not print the
unfiltered PM2 environment.

- [ ] **Step 4: Verify exact live SHA, filtered profile, and readiness**

Run:

```bash
ssh odin-ts 'git -C /Users/ravenhelm/var/canaries/unmute-mlx-bridge-pr5 rev-parse HEAD'
ssh odin-ts 'zsh -lic '\''pm2 list; pm2 env 4 | grep "^TTS_"'\'''
ssh odin-ts 'curl -fsS http://127.0.0.1:8089/readyz'
```

Expected:

- checkout SHA equals `ODIN_TTS_CANDIDATE`;
- `unmute-canary-tts` is online;
- filtered environment shows q8, q32, `buffered_turn`, 4096 characters, and
  60 seconds;
- `/readyz` reports model loaded, no load error, and no active session.

If PM2 assigns a different numeric ID, resolve the ID from `pm2 list` and use
that numeric value only for the filtered environment check; never print the
unfiltered PM2 environment.

- [ ] **Step 5: Establish or verify the private TTS tunnel**

Check the existing control socket:

```bash
ssh -S /Users/nate/var/unmute-canary-tunnel.sock -O check odin-ts
```

If it is absent, create the tunnel:

```bash
ssh \
  -M \
  -S /Users/nate/var/unmute-canary-tunnel.sock \
  -fnNT \
  -L 18089:127.0.0.1:8089 \
  odin-ts
```

Verify:

```bash
curl -fsS http://127.0.0.1:18089/readyz
```

- [ ] **Step 6: Run the canonical controlled probe twice**

```bash
uv run --locked python \
  /Users/nate/var/ravenhelm/slices/timber.yellow.branch/tts_stream_probe.py \
  /Users/nate/var/ravenhelm/slices/timber.yellow.branch/q8-buffered-turn-run-1.wav
uv run --locked python \
  /Users/nate/var/ravenhelm/slices/timber.yellow.branch/tts_stream_probe.py \
  /Users/nate/var/ravenhelm/slices/timber.yellow.branch/q8-buffered-turn-run-2.wav
```

Both runs must report:

- 44 frames;
- 3.520 seconds of audio;
- zero clipping and no non-finite samples;
- PCM SHA-256
  `62605d3213fc7f6380659aa70bf7ac07bcadee509a6a58107415ebd94801fc01`;
- no post-start receive gap greater than 160 milliseconds;
- EOS-to-first-audio no greater than 10 seconds.

The two WAV hashes must be identical. Any PCM delta blocks the physical
canary.

- [ ] **Step 7: Verify the full stock-Unmute service chain**

Check the four existing PM2 canary processes, backend dependency health, and
the local UI route. Refresh `http://localhost:3000`, connect, and perform one
real microphone-to-speaker turn.

The physical gate requires Nate to confirm:

- intelligible words in order;
- one stable speaker identity;
- natural word speed and cadence;
- continuous playback after speech begins;
- no gibberish, mixed voices, stutter, underruns, or digital artifacts.

Record the first-text-to-audio wait and Nate's tonal assessment separately.
Do not call the path production-ready from controlled WAV evidence alone.

- [ ] **Step 8: Roll back immediately if any gate fails**

Restore the exact known-good code and profile:

```bash
ssh odin-ts 'zsh -lic '\''
  cd /Users/ravenhelm/var/canaries/unmute-mlx-bridge-pr5 &&
  git checkout --detach 9a1cbf02eb6a6e7e2af5ecfbc7de1d99824e658b &&
  uv sync --locked
'\'''
ssh odin-ts 'zsh -lic '\''
  TTS_QUANTIZE_BITS= \
  TTS_N_Q=32 \
  TTS_DELIVERY_MODE=streaming \
  TTS_MAX_BUFFERED_CHARS=4096 \
  TTS_MAX_BUFFERED_AUDIO_SECONDS=60 \
  pm2 restart unmute-canary-tts --update-env
'\'''
```

Then verify exact SHA, filtered TTS environment, PM2 online state, and
`/readyz` again.

- [ ] **Step 9: Preserve evidence and the merge boundary**

Update the PR body and slice checkpoint with:

- exact PR head and deployed SHA;
- portable test result with hardware deselection count;
- both controlled probe summaries and hashes;
- physical verdict and measured pre-speech wait;
- deployed or rolled-back profile;
- remaining runner-lane status.

Keep PR #5 open and unmerged. Ask Nate for explicit PR #5 approval only after
all gates pass.
