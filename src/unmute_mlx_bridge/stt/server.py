"""`unmute-mlx-stt`: a `moshi-server`-compatible `/api/asr-streaming` WebSocket server
backed by real MLX inference on Apple Silicon.

Session lifecycle matches the design doc
(`docs/design/architecture.md` §Session
Lifecycle and Cancellation): one WebSocket owns one model session; a second
concurrent connection is rejected with an `Error` and closed rather than queued;
disconnecting cancels inference and releases the slot.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Callable

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from unmute_mlx_bridge.config import SttConfig
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
from unmute_mlx_bridge.protocol.stt import (
    SttClientMessageAdapter,
    SttErrorMessage,
    SttMarkerEchoMessage,
    SttMarkerMessage,
    SttReadyMessage,
    SttStepMessage,
)
from unmute_mlx_bridge.protocol.stt import (
    SttEndWordMessage as SttEndWordOut,
)
from unmute_mlx_bridge.protocol.stt import (
    SttWordMessage as SttWordOut,
)
from unmute_mlx_bridge.protocol.wire import pack_message, unpack_message
from unmute_mlx_bridge.stt.engine import (
    FRAME_SIZE,
    GenerationLengthLimitError,
    SttModelBundle,
    SttSession,
    SttStepEvent,
)

logger = logging.getLogger(__name__)

PROTOCOL_PATH = "/api/asr-streaming"

#: Client-facing message for `stt.engine.GenerationLengthLimitError` (RAV-1552
#: S2): same rationale as `tts/server.py`'s `_LENGTH_LIMIT_MESSAGE`.
_LENGTH_LIMIT_MESSAGE = "session length limit reached; reconnect to start a fresh session"


def _event_to_message(event: SttStepEvent):
    if event.kind == "word":
        return SttWordOut(text=event.text or "", start_time=event.start_time or 0.0)
    if event.kind == "end_word":
        return SttEndWordOut(stop_time=event.stop_time or 0.0)
    if event.kind == "marker":
        return SttMarkerEchoMessage(id=event.marker_id or 0)
    if event.kind == "step":
        return SttStepMessage(step_idx=event.step_idx or 0, prs=event.prs or [])
    raise ValueError(f"unknown STT event kind: {event.kind}")


async def _send_rejection(connection: ServerConnection, message: SttErrorMessage) -> None:
    """Send a pre-`Ready` rejection `Error`, ignoring `ConnectionClosed` (RAV-1552):
    mirrors `tts/server.py::_send_rejection` -- see its docstring. STT's only
    pre-`Ready` rejection branch of this kind is the "still loading"/"failed to
    load" gate below ("no free channels" is a separate, pre-existing path not in
    scope here).
    """
    try:
        await connection.send(pack_message(message))
    except ConnectionClosed:
        pass


class SttServer:
    """Owns the shared model bundle, the single-session admission lock, and the
    per-connection protocol loop. Split out from module-level functions so tests
    can construct one with a fake bundle factory.
    """

    def __init__(
        self,
        config: SttConfig,
        bundle_loader: Callable[[], SttModelBundle] | None = None,
        session_cls: type = SttSession,
    ) -> None:
        """`bundle_loader` and `session_cls` are only ever overridden by tests, so
        the portable conformance suite can exercise the full protocol/session
        state machine against a deterministic fake engine without downloading MLX
        weights. Production code always uses the defaults.
        """
        self.config = config
        self.health = ServiceHealth()
        self.metrics = Metrics()
        self.bundle: SttModelBundle | None = None
        self._session_lock = asyncio.Lock()
        self._bundle_loader = bundle_loader or (
            lambda: SttModelBundle.load(
                config.hf_repo, config.quantize_bits, config.asr_delay_in_tokens
            )
        )
        self._session_cls = session_cls

    async def load_model(self) -> None:
        start = time.monotonic()
        try:
            self.bundle = await asyncio.to_thread(self._bundle_loader)
            self.health.mark_loaded()
            self.metrics.model_load_seconds.set(time.monotonic() - start)
            self.metrics.stt_recv_queue_bound.set(self.config.max_recv_queue)
            logger.info("stt: model ready in %.1fs", time.monotonic() - start)
        except Exception:
            logger.exception("stt: model load failed")
            self.health.mark_load_failed("model load failed, see logs")
            raise

    async def handle_connection(self, connection: ServerConnection) -> None:
        if self.bundle is None:
            # Distinguish "still loading" from "load already failed"
            # (RAV-1552 B4): mirrors `tts/server.py::handle_connection`'s
            # same fix -- see its comment and README.md's "If model load
            # fails" section for the documented rationale for staying up and
            # serving health probes rather than exiting.
            self.metrics.stt_rejected_sessions_by_reason.labels(reason="loading").inc()
            if self.health.model_load_error is not None:
                await _send_rejection(
                    connection, SttErrorMessage(message="model failed to load")
                )
            else:
                await _send_rejection(
                    connection, SttErrorMessage(message="model still loading")
                )
            await connection.close()
            return

        if self._session_lock.locked():
            self.metrics.rejected_sessions.inc()
            await connection.send(pack_message(SttErrorMessage(message="no free channels")))
            await connection.close()
            return

        async with self._session_lock:
            self.health.set_session_active(True)
            self.metrics.active_sessions.set(1)
            try:
                await self._run_session(connection)
            except ConnectionClosed:
                self.metrics.cancellations.inc()
            finally:
                self.health.set_session_active(False)
                self.metrics.active_sessions.set(0)

    async def _run_session(self, connection: ServerConnection) -> None:
        assert self.bundle is not None
        session = self._session_cls(bundle=self.bundle, max_steps=self.config.max_steps)

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
        await connection.send(pack_message(SttReadyMessage()))
        logger.info(
            "stt: session started",
            extra={
                "event": "stt_session_start",
                "session_id": ctx.session_id,
                "conversation_id": ctx.conversation_id,
                "client_supplied": client_sid is not None,
            },
        )

        first_audio_at: float | None = None
        first_output_sent = False
        pending_audio_frames = 0

        with tracer.start_as_current_span(
            "stt.session",
            context=otel_ctx,
            attributes={
                "session.id": ctx.session_id,
                **({} if ctx.conversation_id is None else {"conversation.id": ctx.conversation_id}),
            },
        ):
            async for raw in connection:
                if not isinstance(raw, (bytes, bytearray)):
                    continue  # Text frames are not part of this protocol; ignore.
                try:
                    data = unpack_message(raw)
                    message = SttClientMessageAdapter.validate_python(data)
                except Exception:
                    # Never interpolate the exception into the client-facing
                    # message (RAV-1552 B1, same fix as `tts/server.py`'s
                    # identical malformed-frame handler): a msgpack-unpack
                    # failure or a pydantic `ValidationError` has no safe
                    # subset to select from a generic `Exception`. The full
                    # detail is always logged.
                    self.metrics.protocol_errors.inc()
                    logger.exception("stt: malformed client frame")
                    await connection.send(
                        pack_message(SttErrorMessage(message="malformed frame"))
                    )
                    continue

                if message.type == "Marker":
                    marker: SttMarkerMessage = message  # type: ignore[assignment]
                    session.push_marker(marker.id)
                    events: list[SttStepEvent] = []
                else:  # Audio
                    n_samples = len(message.pcm)
                    if n_samples > self.config.max_input_frame_samples:
                        self.metrics.stt_oversized_frames_total.inc()
                        logger.warning(
                            "stt: oversized audio frame rejected",
                            extra={
                                "event": "stt_oversized_frame",
                                "session_id": ctx.session_id,
                                "n_samples": n_samples,
                                "limit": self.config.max_input_frame_samples,
                            },
                        )
                        await connection.send(
                            pack_message(SttErrorMessage(
                                message=(
                                    f"audio frame too large: {n_samples} samples "
                                    f"(limit {self.config.max_input_frame_samples})"
                                )
                            ))
                        )
                        continue

                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                    audio_seconds = n_samples / 24_000
                    self.metrics.input_audio_seconds.inc(audio_seconds)

                    pending_audio_frames += max(1, n_samples // FRAME_SIZE)
                    self.metrics.stt_inference_frames_active.set(pending_audio_frames)

                    step_start = time.monotonic()
                    try:
                        events = await asyncio.to_thread(session.push_audio, message.pcm)
                    except ConnectionClosed:
                        # Symmetry with the TTS call sites (RAV-1552 S5): a
                        # bare `except Exception:` here would swallow
                        # `ConnectionClosed` instead of letting it propagate
                        # to `handle_connection`'s own `except
                        # ConnectionClosed:` (which counts it as a
                        # cancellation), even though `push_audio` itself has
                        # no reason to raise this -- it is defense in depth,
                        # not a fix for an observed crash.
                        raise
                    except GenerationLengthLimitError:
                        # A legitimate terminal condition, not a crash
                        # (RAV-1552 S2): reported under its own metric label
                        # and message, distinct from an unexpected generation
                        # failure.
                        self.metrics.stt_generation_failures.labels(
                            reason="length_limit"
                        ).inc()
                        await connection.send(
                            pack_message(SttErrorMessage(message=_LENGTH_LIMIT_MESSAGE))
                        )
                        await connection.close()
                        return
                    except Exception:
                        # A generation failure here has no generator/stream
                        # boundary to cross (unlike TTS) -- it would otherwise
                        # propagate straight out of `_run_session`, uncaught by
                        # `handle_connection`'s `except ConnectionClosed:`, and
                        # close the socket abruptly with no protocol Error
                        # (RAV-1552).
                        self.metrics.stt_generation_failures.labels(
                            reason="generation"
                        ).inc()
                        logger.exception("stt: generation failed")
                        await connection.send(
                            pack_message(
                                SttErrorMessage(message="speech recognition failed")
                            )
                        )
                        await connection.close()
                        return
                    step_elapsed = time.monotonic() - step_start

                    n_frames = max(1, n_samples // FRAME_SIZE)
                    pending_audio_frames = max(0, pending_audio_frames - n_frames)
                    self.metrics.stt_inference_frames_active.set(pending_audio_frames)

                    # Fix the dead inference_step_seconds metric: observe per-frame estimate.
                    per_frame_s = step_elapsed / n_frames
                    for _ in range(n_frames):
                        self.metrics.inference_step_seconds.observe(per_frame_s)
                    self.metrics.inference_step_batch_frames.observe(n_frames)
                    if audio_seconds > 0:
                        self.metrics.inference_step_rtf.observe(step_elapsed / audio_seconds)

                for event in events:
                    if not first_output_sent and event.kind in ("word", "step") and first_audio_at:
                        first_output_sent = True
                        self.metrics.time_to_first_output_seconds.observe(
                            time.monotonic() - first_audio_at
                        )
                    if event.kind == "word":
                        word_extra: dict[str, object] = {
                            "event": "stt_word",
                            "session_id": ctx.session_id,
                            "start_time": event.start_time,
                        }
                        if self.config.log_transcripts:
                            word_extra["text"] = event.text
                        logger.info("stt: word recognized", extra=word_extra)
                    await connection.send(pack_message(_event_to_message(event)))


async def _serve(config: SttConfig) -> None:
    server = SttServer(config)
    load_task = asyncio.create_task(server.load_model())

    process_request = build_process_request(
        server.health, server.metrics, PROTOCOL_PATH, config.authorized_ids
    )

    async with serve(
        server.handle_connection,
        config.host,
        config.port,
        process_request=process_request,
        max_queue=config.max_recv_queue,
    ):
        logger.info("stt: listening on ws://%s:%d%s", config.host, config.port, PROTOCOL_PATH)
        try:
            await load_task
        except Exception:
            logger.error("stt: continuing to serve health probes despite load failure")
        await asyncio.Future()  # run forever


def run() -> None:
    parser = argparse.ArgumentParser(description="unmute-mlx-stt")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = SttConfig.from_env()
    if args.host:
        config = SttConfig(**{**config.__dict__, "host": args.host})
    if args.port:
        config = SttConfig(**{**config.__dict__, "port": args.port})

    setup_logging(config.log_level, config.log_format)
    init_otel("unmute-mlx-stt")
    log_clock_metadata()
    asyncio.run(_serve(config))


if __name__ == "__main__":
    run()
