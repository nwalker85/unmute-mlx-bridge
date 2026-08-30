"""`unmute-mlx-tts`: a `moshi-server`-compatible `/api/tts_streaming` WebSocket server
backed by real MLX inference on Apple Silicon.

Only `format=PcmMessagePack` is implemented — the only format Unmute's own TTS
client ever requests (`unmute/tts/text_to_speech.py::TtsStreamingQuery`). A request
for any other format gets an explicit `Error` and a clean close, per the design
doc's honesty requirement, rather than silently emitting the wrong framing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from unmute_mlx_bridge.config import TtsConfig
from unmute_mlx_bridge.observability import (
    CorrelationContext,
    Metrics,
    ServiceHealth,
    build_process_request,
    extract_client_conversation_id,
    extract_client_session_id,
    extract_otel_context,
    get_tracer,
    init_otel,
    log_clock_metadata,
    setup_logging,
)
from unmute_mlx_bridge.protocol.tts import (
    SUPPORTED_FORMAT,
    TtsAudioMessage,
    TtsClientMessageAdapter,
    TtsEosMessage,
    TtsErrorMessage,
    TtsReadyMessage,
    TtsStreamingQuery,
    TtsTextEventMessage,
    TtsTextMessage,
    TtsVoiceMessage,
)
from unmute_mlx_bridge.protocol.wire import pack_message, unpack_message
from unmute_mlx_bridge.tts.engine import TtsModelBundle, TtsSession, TtsStepEvent

logger = logging.getLogger(__name__)

PROTOCOL_PATH = "/api/tts_streaming"


class _ClientDisconnected(Exception):
    """Internal signal used when a close races with a blocking generation step."""


class _BufferedAudioLimitExceeded(Exception):
    """Internal signal that buffered PCM exceeded its configured bound."""


def _next_event(events: Iterator[TtsStepEvent]) -> TtsStepEvent | None:
    return next(events, None)


def _buffered_turn_events(
    session: TtsSession,
    text: str,
    cancelled: Callable[[], bool],
) -> Iterator[TtsStepEvent]:
    yield from session.stream_text(text, cancelled)
    yield from session.stream_eos(cancelled)


def _consume_background_voice_result(task: asyncio.Task[object]) -> None:
    """Consume an abandoned resolver result so its exception is never orphaned."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.warning("tts: background voice resolution failed", exc_info=True)


def _parse_query(path: str) -> TtsStreamingQuery:
    raw = parse_qs(urlsplit(path).query)
    kwargs: dict[str, object] = {}
    for field_name in ("seed", "top_k"):
        if field_name in raw:
            kwargs[field_name] = int(raw[field_name][0])
    for field_name in ("temperature", "cfg_alpha"):
        if field_name in raw:
            kwargs[field_name] = float(raw[field_name][0])
    for field_name in ("format", "voice", "auth_id"):
        if field_name in raw:
            kwargs[field_name] = raw[field_name][0]
    if "max_seq_len" in raw:
        kwargs["max_seq_len"] = int(raw["max_seq_len"][0])
    if "voices" in raw:
        kwargs["voices"] = raw["voices"]
    return TtsStreamingQuery(**kwargs)  # type: ignore[arg-type]


def _event_to_message(event: TtsStepEvent):
    if event.kind == "audio":
        return TtsAudioMessage(pcm=event.pcm or [])
    if event.kind == "word":
        return TtsTextEventMessage(
            text=event.text or "", start_s=event.start_s or 0.0, stop_s=event.stop_s or 0.0
        )
    raise ValueError(f"unknown TTS event kind: {event.kind}")


class TtsServer:
    def __init__(
        self,
        config: TtsConfig,
        bundle_loader: Callable[[], TtsModelBundle] | None = None,
        session_cls: type = TtsSession,
        voice_resolver: Callable[[TtsModelBundle, str], str | Path | None] | None = None,
    ) -> None:
        """`bundle_loader` and `session_cls` are only ever overridden by tests; see
        `SttServer` for the rationale.
        """
        self.config = config
        self.health = ServiceHealth()
        self.metrics = Metrics()
        self.bundle: TtsModelBundle | None = None
        self._session_lock = asyncio.Lock()
        self._bundle_loader = bundle_loader or (
            lambda: TtsModelBundle.load(
                config.hf_repo,
                config.voice_repo,
                config.quantize_bits,
                n_q=config.n_q,
            )
        )
        self._session_cls = session_cls
        self._voice_resolver = voice_resolver or (
            lambda bundle, voice: bundle.resolve_voice(voice)
        )
        self._voice_resolution_lock = asyncio.Lock()
        self._background_voice_resolutions: set[asyncio.Task[object]] = set()

    async def _resolve_voice(
        self,
        bundle: TtsModelBundle,
        voice: str,
        started: asyncio.Event,
    ) -> str | Path | None:
        # Only one blocking resolver may enter the executor. Disconnected
        # clients waiting on this lock can cancel without queuing thread work.
        async with self._voice_resolution_lock:
            started.set()
            return await asyncio.to_thread(self._voice_resolver, bundle, voice)

    async def _abandon_voice_resolution(
        self,
        task: asyncio.Task[object],
        started: asyncio.Event,
    ) -> None:
        if task.done():
            _consume_background_voice_result(task)
        elif started.is_set():
            # `to_thread` can't stop a running resolver. Keep exactly this one
            # task alive until it finishes while its resolver lock prevents any
            # other thread work from starting.
            self._background_voice_resolutions.add(task)
            task.add_done_callback(_consume_background_voice_result)
            task.add_done_callback(self._background_voice_resolutions.discard)
        else:
            # The task hasn't entered the worker, so cancellation removes the
            # waiter without adding work to the executor queue.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def load_model(self) -> None:
        start = time.monotonic()
        try:
            self.bundle = await asyncio.to_thread(self._bundle_loader)
            self.health.mark_loaded()
            self.metrics.model_load_seconds.set(time.monotonic() - start)
            logger.info("tts: model ready in %.1fs", time.monotonic() - start)
        except Exception:
            logger.exception("tts: model load failed")
            self.health.mark_load_failed("model load failed, see logs")
            raise

    async def handle_connection(self, connection: ServerConnection) -> None:
        query = _parse_query(connection.request.path if connection.request else "")

        if query.format != SUPPORTED_FORMAT:
            error_message = (
                f"unsupported format {query.format!r}; only {SUPPORTED_FORMAT} is implemented"
            )
            await connection.send(pack_message(TtsErrorMessage(message=error_message)))
            await connection.close()
            return

        if self.bundle is None:
            await connection.send(pack_message(TtsErrorMessage(message="model still loading")))
            await connection.close()
            return

        if self._session_lock.locked():
            self.metrics.rejected_sessions.inc()
            await connection.send(pack_message(TtsErrorMessage(message="no free channels")))
            await connection.close()
            return

        async with self._session_lock:
            self.health.set_session_active(True)
            self.metrics.active_sessions.set(1)
            try:
                await self._run_session(connection, query)
            except (ConnectionClosed, _ClientDisconnected):
                self.metrics.cancellations.inc()
            finally:
                self.health.set_session_active(False)
                self.metrics.active_sessions.set(0)

    async def _run_session(self, connection: ServerConnection, query: TtsStreamingQuery) -> None:
        assert self.bundle is not None
        voice = query.voice or self.config.default_voice

        # Prefer the client-supplied session ID for cross-repo correlation.
        client_sid = extract_client_session_id(connection.request.headers) if connection.request else None
        client_cid = extract_client_conversation_id(connection.request.headers) if connection.request else None
        ctx = CorrelationContext(
            session_id=client_sid,
            conversation_id=client_cid,
        ) if client_sid else CorrelationContext(conversation_id=client_cid)

        tracer = get_tracer()
        # Extract W3C traceparent/tracestate explicitly from WebSocket upgrade headers.
        # The OTEL SDK does not automatically propagate context through WebSocket handshakes.
        otel_ctx = extract_otel_context(connection.request.headers) if connection.request else None
        span = tracer.start_span("tts.session", context=otel_ctx)
        span.set_attribute("session.id", ctx.session_id)
        span.set_attribute("session.voice", voice)
        if ctx.conversation_id is not None:
            span.set_attribute("conversation.id", ctx.conversation_id)
        try:
            await self._run_session_inner(connection, query, ctx)
        finally:
            span.end()

    async def _run_session_inner(
        self,
        connection: ServerConnection,
        query: TtsStreamingQuery,
        ctx: CorrelationContext,
    ) -> None:
        assert self.bundle is not None
        voice = query.voice or self.config.default_voice
        await connection.send(pack_message(TtsReadyMessage()))
        logger.info(
            "tts: session started",
            extra={
                "event": "tts_session_start",
                "session_id": ctx.session_id,
                "conversation_id": ctx.conversation_id,
                "voice": voice,
                "client_supplied": (connection.request is not None
                                    and connection.request.headers.get("x-unmute-session-id") is not None),
            },
        )
        # A first-use voice may need to be fetched from Hugging Face. Stock
        # Unmute allows only 500 ms for connect + Ready, so acknowledge the
        # admitted channel before that implementation-specific initialization
        # and keep the event loop available to flush the frame. Voice resolution
        # itself doesn't mutate shared MLX generation state, so a disconnected
        # client can abandon that task and release the sole channel immediately.
        resolution_started = asyncio.Event()
        voice_resolution = asyncio.create_task(
            self._resolve_voice(self.bundle, voice, resolution_started)
        )
        connection_closed = asyncio.create_task(connection.wait_closed())
        try:
            try:
                done, _pending = await asyncio.wait(
                    (voice_resolution, connection_closed),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                await self._abandon_voice_resolution(
                    voice_resolution, resolution_started
                )
                raise
            if connection_closed in done:
                await self._abandon_voice_resolution(
                    voice_resolution, resolution_started
                )
                raise _ClientDisconnected
            resolved_voice = voice_resolution.result()
        finally:
            connection_closed.cancel()
            try:
                await connection_closed
            except asyncio.CancelledError:
                pass

        # Construction resets shared MLX caches, so unlike voice resolution it
        # must finish while this connection still owns the single-session lock.
        session = await asyncio.to_thread(
            self._session_cls,
            bundle=self.bundle,
            voice=resolved_voice,
            max_gen_length=(
                self.config.max_gen_length
                if query.max_seq_len is None
                else query.max_seq_len
            ),
            seed=query.seed,
            temperature=query.temperature,
            top_k=query.top_k,
            cfg_alpha=query.cfg_alpha,
        )

        first_text_at: float | None = None
        first_output_sent = False
        buffered_chunks: list[str] = []
        buffered_chars = 0
        buffered_terminal_handled = False
        # Session-start boundary for the `Voice` protocol message (RAV-1504):
        # a custom voice embedding conditions the session only if it arrives
        # before any `Text` message. Real `moshi-server` only reads a
        # connection's pending voice on the channel-init entry
        # (`rust/moshi-server/tts.py:340-353`'s `if new_entry[0] == -1:`
        # branch, fed once per channel by `py_module.rs:237-240`'s
        # `if !c.sent_init { t.push(-1); ... }`); every later `Text` message
        # consumes a different, non-init entry that never reads `voice`, so
        # upstream silently forwards and then ignores any post-init `Voice`
        # message rather than raising. This bridge instead rejects it
        # explicitly (see below). Latched on the first `Text` message seen,
        # empty or not -- "before any Text message" is read literally.
        voice_conditioning_locked = False

        try:
            async for raw in connection:
                if isinstance(raw, (bytes, bytearray)) and bytes(raw) == b"\x00":
                    buffered_terminal_handled = True
                    first_output_sent = await self._finish_session(
                        connection,
                        session,
                        buffered_chunks,
                        first_text_at,
                        first_output_sent,
                    )
                    await connection.close()
                    return
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    data = unpack_message(raw)
                    message = TtsClientMessageAdapter.validate_python(data)
                except Exception as exc:
                    self.metrics.protocol_errors.inc()
                    await connection.send(
                        pack_message(
                            TtsErrorMessage(message=f"malformed frame: {exc}")
                        )
                    )
                    continue

                if isinstance(message, TtsVoiceMessage):
                    if voice_conditioning_locked:
                        # Upstream moshi-server would silently forward and
                        # then ignore this (it only reads `voice` on a
                        # channel's init entry); this bridge rejects it
                        # explicitly instead of accepting and silently
                        # discarding the request.
                        self.metrics.protocol_errors.inc()
                        await connection.send(
                            pack_message(
                                TtsErrorMessage(
                                    message=(
                                        "Voice message after generation "
                                        "started: voice conditioning is "
                                        "fixed at session start (upstream "
                                        "moshi-server reads the voice only "
                                        "on the channel init entry and "
                                        "silently ignores later Voice "
                                        "messages; this bridge rejects them "
                                        "explicitly)"
                                    )
                                )
                            )
                        )
                        continue
                    try:
                        session.apply_voice_embedding(
                            message.embeddings, message.shape
                        )
                    except Exception as exc:
                        # Never crash on a malformed/mismatched embedding, and
                        # never silently keep the prior voice while pretending
                        # to have applied the new one.
                        self.metrics.protocol_errors.inc()
                        await connection.send(
                            pack_message(
                                TtsErrorMessage(
                                    message=f"invalid voice embedding: {exc}"
                                )
                            )
                        )
                        continue
                    logger.info(
                        "tts: session conditioned from Voice message",
                        extra={
                            "event": "tts_voice_conditioned",
                            "session_id": ctx.session_id,
                        },
                    )
                    continue
                if isinstance(message, TtsTextMessage):
                    voice_conditioning_locked = True
                    if not message.text:
                        continue
                    if first_text_at is None:
                        first_text_at = time.monotonic()
                    if self.config.delivery_mode == "buffered_turn":
                        text = message.text.strip()
                        if not text:
                            continue
                        buffered_extra: dict[str, object] = {
                            "event": "tts_text_buffered",
                            "session_id": ctx.session_id,
                            "buffered_chars": buffered_chars + len(text),
                        }
                        if self.config.log_transcripts:
                            buffered_extra["text"] = text
                        logger.info("tts: text buffered", extra=buffered_extra)
                        next_chars = (
                            buffered_chars + len(text) + (1 if buffered_chunks else 0)
                        )
                        if next_chars > self.config.max_buffered_chars:
                            buffered_terminal_handled = True
                            self.metrics.buffered_turn_failures.labels(
                                reason="input_limit"
                            ).inc()
                            await connection.send(
                                pack_message(
                                    TtsErrorMessage(
                                        message=(
                                            "buffered TTS input exceeds "
                                            "configured limit"
                                        )
                                    )
                                )
                            )
                            await connection.close()
                            return
                        buffered_chunks.append(text)
                        buffered_chars = next_chars
                        continue
                    cancelled = threading.Event()
                    streaming_extra: dict[str, object] = {
                        "event": "tts_text_streaming",
                        "session_id": ctx.session_id,
                    }
                    if self.config.log_transcripts:
                        streaming_extra["text"] = message.text
                    logger.info("tts: text streaming", extra=streaming_extra)
                    first_output_sent = await self._emit_stream(
                        connection,
                        session.stream_text(message.text, cancelled.is_set),
                        first_text_at,
                        first_output_sent,
                        cancelled,
                    )
                elif isinstance(message, TtsEosMessage):
                    buffered_terminal_handled = True
                    first_output_sent = await self._finish_session(
                        connection,
                        session,
                        buffered_chunks,
                        first_text_at,
                        first_output_sent,
                    )
                    await connection.close()
                    return
        finally:
            if (
                self.config.delivery_mode == "buffered_turn"
                and not buffered_terminal_handled
            ):
                self.metrics.buffered_turn_failures.labels(
                    reason="disconnect"
                ).inc()

    async def _collect_stream(
        self,
        connection: ServerConnection,
        events: Iterator[TtsStepEvent],
        cancelled: threading.Event,
        max_audio_samples: int,
    ) -> tuple[list[TtsStepEvent], int]:
        connection_closed = asyncio.create_task(connection.wait_closed())
        next_event: asyncio.Task[TtsStepEvent | None] | None = None
        collected: list[TtsStepEvent] = []
        audio_samples = 0
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
                    return collected, audio_samples
                if event.kind == "audio":
                    audio_samples += len(event.pcm or [])
                    if audio_samples > max_audio_samples:
                        raise _BufferedAudioLimitExceeded
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
        buffered_text = " ".join(buffered_chunks)
        eos_started_at = time.monotonic()
        max_audio_samples = int(
            self.config.max_buffered_audio_seconds * 24_000
        )
        events = _buffered_turn_events(
            session,
            buffered_text,
            cancelled.is_set,
        )
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
        synthesis_seconds = time.monotonic() - eos_started_at
        self.metrics.buffered_input_characters.observe(
            len(buffered_text)
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
        return await self._emit(
            connection,
            collected,
            first_text_at,
            first_output_sent,
        )

    async def _emit_stream(
        self,
        connection: ServerConnection,
        events: Iterator[TtsStepEvent],
        first_text_at: float | None,
        first_output_sent: bool,
        cancelled: threading.Event,
    ) -> bool:
        connection_closed = asyncio.create_task(connection.wait_closed())
        next_event: asyncio.Task[TtsStepEvent | None] | None = None
        try:
            while True:
                next_event = asyncio.create_task(asyncio.to_thread(_next_event, events))
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
                    return first_output_sent
                first_output_sent = await self._emit(
                    connection, [event], first_text_at, first_output_sent
                )
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

    async def _emit(
        self,
        connection: ServerConnection,
        events: list[TtsStepEvent],
        first_text_at: float | None,
        first_output_sent: bool,
    ) -> bool:
        for event in events:
            if event.kind == "audio":
                self.metrics.output_audio_seconds.inc(len(event.pcm or []) / 24_000)
                if not first_output_sent and first_text_at is not None:
                    first_output_sent = True
                    self.metrics.time_to_first_output_seconds.observe(
                        time.monotonic() - first_text_at
                    )
            await connection.send(pack_message(_event_to_message(event)))
        return first_output_sent


async def _serve(config: TtsConfig) -> None:
    server = TtsServer(config)
    load_task = asyncio.create_task(server.load_model())

    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )

    async with serve(
        server.handle_connection,
        config.host,
        config.port,
        process_request=process_request,
        # Stock Unmute streams one WebSocket data frame per LLM word. Keep a
        # bounded burst in memory so socket reads don't pause before later Ping
        # control frames while MLX is still generating an earlier word.
        max_queue=1024,
    ):
        logger.info("tts: listening on ws://%s:%d%s", config.host, config.port, PROTOCOL_PATH)
        try:
            await load_task
        except Exception:
            logger.error("tts: continuing to serve health probes despite load failure")
        await asyncio.Future()  # run forever


def run() -> None:
    parser = argparse.ArgumentParser(description="unmute-mlx-tts")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = TtsConfig.from_env()
    if args.host:
        config = TtsConfig(**{**config.__dict__, "host": args.host})
    if args.port:
        config = TtsConfig(**{**config.__dict__, "port": args.port})

    setup_logging(config.log_level, config.log_format)
    init_otel("unmute-mlx-tts")
    log_clock_metadata()
    asyncio.run(_serve(config))


if __name__ == "__main__":
    run()
