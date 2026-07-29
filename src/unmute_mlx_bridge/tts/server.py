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
from unmute_mlx_bridge.observability import Metrics, ServiceHealth, build_process_request
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


def _next_event(events: Iterator[TtsStepEvent]) -> TtsStepEvent | None:
    return next(events, None)


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
                config.hf_repo, config.voice_repo, config.quantize_bits
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
        await connection.send(pack_message(TtsReadyMessage()))
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
            max_gen_length=self.config.max_gen_length,
        )

        first_text_at: float | None = None
        first_output_sent = False

        async for raw in connection:
            if isinstance(raw, (bytes, bytearray)) and bytes(raw) == b"\x00":
                cancelled = threading.Event()
                first_output_sent = await self._emit_stream(
                    connection,
                    session.stream_eos(cancelled.is_set),
                    first_text_at,
                    first_output_sent,
                    cancelled,
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
                    pack_message(TtsErrorMessage(message=f"malformed frame: {exc}"))
                )
                continue

            if isinstance(message, TtsVoiceMessage):
                # Custom cloned-voice embeddings are a documented non-goal for this
                # canary; reject explicitly rather than silently ignoring them.
                self.metrics.protocol_errors.inc()
                await connection.send(
                    pack_message(
                        TtsErrorMessage(
                            message="custom voice embeddings (Voice message) are not supported"
                        )
                    )
                )
                continue
            if isinstance(message, TtsTextMessage):
                if not message.text:
                    continue
                if first_text_at is None:
                    first_text_at = time.monotonic()
                cancelled = threading.Event()
                first_output_sent = await self._emit_stream(
                    connection,
                    session.stream_text(message.text, cancelled.is_set),
                    first_text_at,
                    first_output_sent,
                    cancelled,
                )
            elif isinstance(message, TtsEosMessage):
                cancelled = threading.Event()
                first_output_sent = await self._emit_stream(
                    connection,
                    session.stream_eos(cancelled.is_set),
                    first_text_at,
                    first_output_sent,
                    cancelled,
                )
                await connection.close()
                return

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

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_serve(config))


if __name__ == "__main__":
    run()
