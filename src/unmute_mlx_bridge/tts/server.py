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
import math
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
from unmute_mlx_bridge.tts.engine import (
    GenerationLengthLimitError,
    TtsModelBundle,
    TtsSession,
    TtsStepEvent,
    VoiceEmbeddingError,
)

logger = logging.getLogger(__name__)

PROTOCOL_PATH = "/api/tts_streaming"


class _ClientDisconnected(Exception):
    """Internal signal used when a close races with a blocking generation step."""


class _BufferedAudioLimitExceeded(Exception):
    """Internal signal that buffered PCM exceeded its configured bound."""


class _QueryError(Exception):
    """Raised by `_parse_query` when a query parameter cannot be parsed into
    its expected type (e.g. `?seed=abc`), or is repeated where only a single
    value is accepted (e.g. `?voice=a&voice=b`) (RAV-1552 F1/F6). Carries
    only the parameter *name* -- never the raw value or the underlying
    `ValueError`, neither of which is safe to forward to a client verbatim.

    `handle_connection` catches this before `Ready` is ever sent and reports
    it as an explicit protocol `Error` naming only the parameter, then closes
    -- the same fail-fast shape as every other pre-Ready validation failure
    (`format=`, `voice=`/`voices=` structural checks, `cfg_alpha=`), instead
    of letting a bare `int()`/`float()` `ValueError` propagate out of
    `_parse_query` uncaught and kill the socket with 1011 and no `Error`.
    """

    def __init__(self, param_name: str) -> None:
        self.param_name = param_name
        super().__init__(f"invalid query parameter: {param_name}")


class _VoiceResolutionError(Exception):
    """Raised by `_resolve_voice`/`_resolve_voice_list` when the configured
    voice resolver fails for a client-supplied voice name. Carries only that
    (client-supplied, safe) name in its message -- never the underlying
    resolver exception, which may embed operator-specific details (e.g.
    `huggingface_hub`'s disk-space `OSError` embeds the local cache path,
    including the operator's username) that must never reach a client
    (RAV-1552 B1/B3). The full original exception is always logged via
    `logger.exception` at the point of failure, before this is raised.
    """

    def __init__(self, voice_name: str) -> None:
        self.voice_name = voice_name
        super().__init__(f"unresolvable voice: {voice_name}")


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


async def _send_rejection(connection: ServerConnection, message: TtsErrorMessage) -> None:
    """Send a pre-`Ready` rejection `Error`, ignoring `ConnectionClosed` (RAV-1552):
    every `handle_connection` pre-`Ready` rejection branch (`query`, `format`,
    `voices`, `loading`, `cfg_alpha`) sent its `Error` unguarded -- a client that
    disconnected on its own between the upgrade and this rejection made `send`
    raise `ConnectionClosed`, which propagated straight out of `handle_connection`
    uncaught (the same 1011-with-no-`Error` failure mode this whole rejection path
    exists to avoid, just triggered by the *client* leaving instead of a bug). The
    metric increment always happens before this call, so the rejection is still
    recorded even when the `Error` itself can no longer be delivered.
    """
    try:
        await connection.send(pack_message(message))
    except ConnectionClosed:
        pass


def _parse_query(path: str, max_seq_len_cap: int | None = None) -> TtsStreamingQuery:
    query_string = urlsplit(path).query
    # Every field is parsed with blank values kept (RAV-1552): `parse_qs`'s
    # default `keep_blank_values=False` drops a blank value entirely, as if
    # the parameter had never been given at all. Originally this was scoped
    # to just `voice=`/`voices=` (RAV-1552 S1) via a second, separate parse
    # (`raw_voice_fields`) -- every other field still used the
    # blank-dropping default. That silently defeated every "must be exactly
    # one value" guard below for a *blank-padded* repeat: `?format=&
    # format=PcmMessagePack`, `?auth_id=&auth_id=good`, `?seed=&seed=7`, and
    # `?max_seq_len=&max_seq_len=5` all had their blank entry dropped before
    # `len(values) != 1` ever saw it, leaving exactly one (the non-blank)
    # value -- so the request was silently accepted (`Ready` sent) instead
    # of rejected, the exact silent-drop failure mode this guard exists to
    # stop. Applied uniformly now; `raw_voice_fields` is gone, `raw` is the
    # single source of truth for every field.
    raw = parse_qs(query_string, keep_blank_values=True)
    kwargs: dict[str, object] = {}
    # A non-numeric value (`?seed=abc`) or a repeated occurrence of one of
    # these scalar parameters (`?seed=1&seed=2`) both raise `_QueryError`
    # here (RAV-1552 F1/F6): a bare `int()`/`float()` `ValueError` used to
    # propagate out of this function uncaught, and a repeated value used to
    # silently take only `values[0]` and drop the rest with no signal to the
    # client -- neither is caught anywhere before `handle_connection` sends
    # `Ready`, so both used to either crash the socket with 1011 and no
    # `Error`, or (for the repeat case) never surface at all. A lone blank
    # value (`?seed=`) fails the same way a non-numeric one does: `int("")`/
    # `float("")` both raise `ValueError`, so no separate check is needed
    # here for the numeric fields specifically.
    #
    # Each field also gets an explicit range check once it parses (RAV-1552):
    # a syntactically valid but out-of-range value (a negative `seed`, a
    # `top_k`/`max_seq_len` of zero or less, a non-finite or negative
    # `temperature`) used to reach `TtsSession`/the underlying sampler
    # unchecked instead of failing fast here like every other malformed
    # value. `cfg_alpha` is deliberately excluded from range checking here --
    # it is range-checked separately, against the *loaded model's* actual
    # supported set, by `_cfg_alpha_query_error` below. `max_seq_len` also
    # gets an upper-bound check against `max_seq_len_cap` (the operator's
    # configured `TTS_MAX_GEN_LENGTH`, passed in by `handle_connection`) so a
    # client-supplied override can only *lower* the operator's cap, never
    # raise it (RAV-1552) -- previously any positive value replaced the cap
    # unbounded.
    for field_name in ("seed", "top_k", "max_seq_len"):
        if field_name in raw:
            values = raw[field_name]
            if len(values) != 1:
                raise _QueryError(field_name)
            try:
                parsed_int = int(values[0])
            except ValueError:
                raise _QueryError(field_name) from None
            if field_name == "seed" and parsed_int < 0:
                raise _QueryError(field_name)
            if field_name in ("top_k", "max_seq_len") and parsed_int < 1:
                raise _QueryError(field_name)
            if (
                field_name == "max_seq_len"
                and max_seq_len_cap is not None
                and parsed_int > max_seq_len_cap
            ):
                raise _QueryError(field_name)
            kwargs[field_name] = parsed_int
    for field_name in ("temperature", "cfg_alpha"):
        if field_name in raw:
            values = raw[field_name]
            if len(values) != 1:
                raise _QueryError(field_name)
            try:
                parsed_float = float(values[0])
            except ValueError:
                raise _QueryError(field_name) from None
            if field_name == "temperature" and not (
                math.isfinite(parsed_float) and parsed_float >= 0
            ):
                raise _QueryError(field_name)
            kwargs[field_name] = parsed_float
    # `format`/`auth_id` get the same repeated-value guard as the scalar
    # numeric fields above (RAV-1552): a repeated `?format=a&format=b` or
    # `?auth_id=a&auth_id=b` used to silently take `values[0]` and drop the
    # rest, exactly the F6 silent-drop failure mode the numeric fields and
    # `voice=` were already fixed for -- these two were missed. Unlike the
    # numeric fields, a lone blank value (`?format=`) needs its own explicit
    # check -- there is no parse step here to fail on an empty string, so it
    # is rejected the same way a blank `voice=` already is (RAV-1552 S1).
    # See `observability.py::check_auth` for the matching pre-handshake
    # guard on a repeated or blank `auth_id=`, so the gate and this parser
    # can never disagree about whether one is acceptable.
    for field_name in ("format", "auth_id"):
        if field_name in raw:
            values = raw[field_name]
            if len(values) != 1:
                raise _QueryError(field_name)
            if values[0] == "":
                raise _QueryError(field_name)
            kwargs[field_name] = values[0]
    if "voice" in raw:
        voice_values = raw["voice"]
        if len(voice_values) != 1:
            # `?voice=a&voice=b` used to silently take `voice_values[0]` and
            # drop `b` with no signal to the client (RAV-1552 F6).
            raise _QueryError("voice")
        kwargs["voice"] = voice_values[0]
    if "voices" in raw:
        kwargs["voices"] = raw["voices"]
    return TtsStreamingQuery(**kwargs)  # type: ignore[arg-type]


#: `moshi_mlx.models.tts.TTSModel.make_condition_attributes` only ever fills
#: 5 speaker slots (`for idx in range(5)`); a 6th `voices=` entry would be
#: silently dropped by the model rather than rejected, so this bridge
#: enforces the limit explicitly instead.
MAX_VOICES_BLEND = 5

#: Client-facing message for `tts.engine.GenerationLengthLimitError` (RAV-1552
#: S2): a session hitting its configured `max_gen_length` is a legitimate
#: terminal condition, not a generation failure, so it gets its own message
#: (and its own `reason="length_limit"` metric label at each call site) rather
#: than the generic "TTS generation failed"/"buffered TTS generation failed".
_LENGTH_LIMIT_MESSAGE = "session length limit reached; reconnect to start a fresh session"


def _voices_query_error(query: TtsStreamingQuery) -> str | None:
    """Structural validation of `voice=`/`voices=` query parameters -- pure and
    synchronous, so it can reject a malformed request before a channel slot
    or model resolution is ever committed to it (same fail-fast shape as the
    `format=` check). Never validates whether a *name* resolves -- that
    requires the (possibly blocking, HF-fetching) voice resolver, handled
    separately once a session's channel slot is held; see
    `TtsServer._run_session_inner`.

    A blank `voice=`/`voices=` entry is rejected explicitly (RAV-1552 S1;
    see `_parse_query`, which parses these two fields with blank values kept
    so they reach here instead of being silently dropped). The mutual-
    exclusivity check runs first and deliberately treats a *blank* `voice=`
    as "given": `?voice=&voices=a` hits `"cannot specify both..."` rather
    than the blank-voice message below -- a `voice=` present on the wire at
    all, blank or not, is a value the client actually sent. This is a
    documented choice among two defensible options (see PROTOCOL.md); the
    other would have been to reject the blank `voice=` on its own regardless
    of `voices=`.
    """
    if query.voice is not None and query.voices is not None:
        return "cannot specify both 'voice' and 'voices' query parameters"
    if query.voice == "":
        return "'voice' query parameter must not be blank"
    if query.voices is not None:
        if len(query.voices) == 0:
            return "'voices' query parameter must not be empty"
        if any(voice == "" for voice in query.voices):
            return "'voices' query parameter must not contain blank entries"
        if len(set(query.voices)) != len(query.voices):
            # Duplicates burn a blend slot for no effect: `make_condition_
            # attributes` only ever fills `MAX_VOICES_BLEND` speaker slots,
            # so `voices=a&voices=a` silently wastes one of the 5 available
            # slots instead of blending a second distinct voice.
            return "'voices' query parameter must not contain duplicate entries"
        if len(query.voices) > MAX_VOICES_BLEND:
            return (
                f"'voices' supports at most {MAX_VOICES_BLEND} entries, "
                f"got {len(query.voices)}"
            )
    return None


def _cfg_alpha_query_error(bundle: object, query: TtsStreamingQuery) -> str | None:
    """Structural validation of `cfg_alpha=` against the *loaded model's*
    supported set -- run once the bundle is loaded but before a channel slot
    is committed (same fail-fast shape as `_voices_query_error`), so an
    unsupported `cfg_alpha` never reaches `TtsSession` construction
    (RAV-1552 B2; `TtsSession.__post_init__` raising `ValueError` there used
    to propagate out of `asyncio.to_thread` uncaught, closing the socket with
    1011 and no protocol `Error`).

    `bundle` is typed `object` (rather than `TtsModelBundle`) because the
    portable conformance suite's `FakeBundle` test doubles carry no
    `tts_model` at all -- `getattr` defensively no-ops for those, matching
    the pre-existing behavior of never validating `cfg_alpha` when the
    model's supported set cannot be determined.
    """
    if query.cfg_alpha is None:
        return None
    tts_model = getattr(bundle, "tts_model", None)
    valid_cfg_conditionings = getattr(tts_model, "valid_cfg_conditionings", None)
    if not valid_cfg_conditionings:
        return None
    if query.cfg_alpha not in valid_cfg_conditionings:
        return (
            f"unsupported cfg_alpha {query.cfg_alpha}; expected one of "
            f"{sorted(valid_cfg_conditionings)}"
        )
    return None


def _voice_label(config: TtsConfig, query: TtsStreamingQuery) -> str:
    """Human-readable voice identifier for logging/span attributes only --
    never used for resolution (see `_run_session_inner`, which resolves
    `voice`/`voices` through separate, dedicated code paths)."""
    if query.voices is not None:
        return ",".join(query.voices)
    return query.voice or config.default_voice


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
                cfg_coef=config.cfg_coef,
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
            try:
                return await asyncio.to_thread(self._voice_resolver, bundle, voice)
            except Exception:
                # The resolver's own exception (e.g. `huggingface_hub`'s
                # disk-space `OSError`, which embeds the local cache path
                # including the operator's username) must never reach a
                # client (RAV-1552 B1/B3). Log it in full here; the caller
                # only ever sees the client-supplied voice name.
                logger.exception("tts: voice resolution failed for %r", voice)
                raise _VoiceResolutionError(voice) from None

    async def _resolve_voice_list(
        self,
        bundle: TtsModelBundle,
        voices: list[str],
        started: asyncio.Event,
    ) -> list[str | Path | None]:
        """Resolves every entry of a `voices=` multi-voice blend through the
        same voice resolver `_resolve_voice` uses for a single `voice=`, one
        at a time, under the same single-resolver lock (so a multi-voice
        resolution still serializes against any other connection's voice
        resolution). An unresolvable entry raises `_VoiceResolutionError`
        (named for the specific entry that failed) from here and is caught
        by the caller (`TtsServer._run_session_inner`), which reports it as
        an explicit protocol `Error` -- never a silent drop.
        """
        async with self._voice_resolution_lock:
            started.set()
            resolved: list[str | Path | None] = []
            for voice in voices:
                try:
                    resolved.append(
                        await asyncio.to_thread(self._voice_resolver, bundle, voice)
                    )
                except Exception:
                    logger.exception("tts: voice resolution failed for %r", voice)
                    raise _VoiceResolutionError(voice) from None
            return resolved

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
        try:
            query = _parse_query(
                connection.request.path if connection.request else "",
                self.config.max_gen_length,
            )
        except _QueryError as exc:
            # A malformed or repeated query value (RAV-1552 F1/F6) is the
            # very first thing validated, before `format=`, before the model
            # load check -- same fail-fast shape as every other pre-Ready
            # validation failure. Only the parameter name is ever named,
            # never the raw value or the underlying `ValueError` text.
            self.metrics.tts_rejected_sessions_by_reason.labels(reason="query").inc()
            await _send_rejection(
                connection,
                TtsErrorMessage(message=f"invalid '{exc.param_name}' query parameter"),
            )
            await connection.close()
            return

        if query.format != SUPPORTED_FORMAT:
            error_message = (
                f"unsupported format {query.format!r}; only {SUPPORTED_FORMAT} is implemented"
            )
            self.metrics.tts_rejected_sessions_by_reason.labels(reason="format").inc()
            await _send_rejection(connection, TtsErrorMessage(message=error_message))
            await connection.close()
            return

        voices_error = _voices_query_error(query)
        if voices_error is not None:
            self.metrics.tts_rejected_sessions_by_reason.labels(reason="voices").inc()
            await _send_rejection(connection, TtsErrorMessage(message=voices_error))
            await connection.close()
            return

        if self.bundle is None:
            # Distinguish "still loading" from "load already failed"
            # (RAV-1552 B4): `model_load_error` is only ever set once
            # `load_model` has actually failed (see `ServiceHealth.
            # mark_load_failed`), so a client connecting after that point
            # gets a message that matches reality instead of "still loading"
            # forever. See README.md's "If model load fails" section for the
            # documented rationale for staying up and serving health probes
            # rather than exiting.
            self.metrics.tts_rejected_sessions_by_reason.labels(reason="loading").inc()
            if self.health.model_load_error is not None:
                await _send_rejection(
                    connection, TtsErrorMessage(message="model failed to load")
                )
            else:
                await _send_rejection(
                    connection, TtsErrorMessage(message="model still loading")
                )
            await connection.close()
            return

        cfg_alpha_error = _cfg_alpha_query_error(self.bundle, query)
        if cfg_alpha_error is not None:
            # Previously the only pre-Ready rejection path with a metric at
            # all (via `protocol_errors`); now uses the same labelled
            # `tts_rejected_sessions_by_reason` counter as every other path
            # above, for one consistent place to look (RAV-1552).
            self.metrics.tts_rejected_sessions_by_reason.labels(reason="cfg_alpha").inc()
            await _send_rejection(connection, TtsErrorMessage(message=cfg_alpha_error))
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
        voice = _voice_label(self.config, query)

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
        voice = _voice_label(self.config, query)
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
        is_multi_voice = query.voices is not None
        if is_multi_voice:
            assert query.voices is not None
            voice_resolution = asyncio.create_task(
                self._resolve_voice_list(self.bundle, query.voices, resolution_started)
            )
        else:
            voice_resolution = asyncio.create_task(
                self._resolve_voice(
                    self.bundle, query.voice or self.config.default_voice, resolution_started
                )
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
            # Single- and multi-voice resolution now share identical
            # exception handling (RAV-1552 B1/B3): both `_resolve_voice` and
            # `_resolve_voice_list` wrap any resolver failure in
            # `_VoiceResolutionError`, naming only the client-supplied voice
            # that failed -- never the underlying resolver exception (which
            # may embed operator-specific details, e.g. `huggingface_hub`'s
            # disk-space `OSError` embeds the local cache path including the
            # operator's username). An unresolvable voice, single or blended,
            # must be an explicit protocol Error, never a silent drop or an
            # abrupt/crashed close.
            try:
                resolved_voice: object = voice_resolution.result()
            except _VoiceResolutionError as exc:
                self.metrics.protocol_errors.inc()
                await connection.send(pack_message(TtsErrorMessage(message=str(exc))))
                await connection.close()
                return
        finally:
            connection_closed.cancel()
            try:
                await connection_closed
            except asyncio.CancelledError:
                pass

        # Construction resets shared MLX caches, so unlike voice resolution it
        # must finish while this connection still owns the single-session lock.
        # `cfg_alpha` is already validated against the loaded model's
        # supported set before the channel slot is even taken (see
        # `_cfg_alpha_query_error` in `handle_connection`), so this should
        # never fail on that account -- but any other unexpected construction
        # failure (a `ValueError` or otherwise) must still become a
        # sanitized protocol `Error` and a clean close, not an uncaught
        # exception that leaves the socket dying with 1011 and no `Error`
        # (RAV-1552 B2). The message is fully generic (never the exception
        # text) since, unlike the voice-resolution/embedding paths above,
        # there is no safe, client-supplied value to name here.
        try:
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
        except Exception:
            self.metrics.protocol_errors.inc()
            logger.exception("tts: session construction failed")
            await connection.send(
                pack_message(TtsErrorMessage(message="session initialization failed"))
            )
            await connection.close()
            return

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
                except Exception:
                    # Never interpolate the exception into the client-facing
                    # message (RAV-1552 B1): a msgpack-unpack failure or a
                    # pydantic `ValidationError` can echo back arbitrary
                    # bytes/structure from the frame itself, and there is no
                    # safe subset of that to select from a generic `Exception`.
                    # The full detail is always logged.
                    self.metrics.protocol_errors.inc()
                    logger.exception("tts: malformed client frame")
                    await connection.send(
                        pack_message(TtsErrorMessage(message="malformed frame"))
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
                    except VoiceEmbeddingError as exc:
                        # Never crash on a malformed/mismatched embedding, and
                        # never silently keep the prior voice while pretending
                        # to have applied the new one. `VoiceEmbeddingError`'s
                        # text is always ours (`tts/engine.py`) and safe to
                        # forward verbatim: the structural checks name only
                        # shapes/dimensions, and the two paths that can fail
                        # on a third-party `mlx`/`moshi_mlx` exception use a
                        # fixed, generic message instead of that exception's
                        # own text (RAV-1552 F2) -- the detail is always
                        # logged server-side, never forwarded to the client.
                        self.metrics.protocol_errors.inc()
                        await connection.send(
                            pack_message(
                                TtsErrorMessage(
                                    message=f"invalid voice embedding: {exc}"
                                )
                            )
                        )
                        continue
                    except Exception:
                        # Any *other* exception type is not ours to forward
                        # (RAV-1552 B1) -- `apply_voice_embedding` is
                        # documented to only ever raise `VoiceEmbeddingError`,
                        # but this is defense in depth against that contract
                        # ever being violated by a future change.
                        self.metrics.protocol_errors.inc()
                        logger.exception("tts: voice embedding application failed unexpectedly")
                        await connection.send(
                            pack_message(
                                TtsErrorMessage(message="invalid voice embedding")
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
                    try:
                        first_output_sent = await self._emit_stream(
                            connection,
                            session.stream_text(message.text, cancelled.is_set),
                            first_text_at,
                            first_output_sent,
                            cancelled,
                        )
                    except (ConnectionClosed, _ClientDisconnected):
                        raise
                    except GenerationLengthLimitError:
                        # A legitimate terminal condition, not a crash (RAV-1552
                        # S2): reported under its own metric label and message,
                        # distinct from an unexpected generation failure.
                        self.metrics.streaming_failures.labels(
                            reason="length_limit"
                        ).inc()
                        await connection.send(
                            pack_message(TtsErrorMessage(message=_LENGTH_LIMIT_MESSAGE))
                        )
                        await connection.close()
                        return
                    except Exception:
                        # A generation failure must not close the socket
                        # abruptly with no protocol Error (RAV-1552):
                        # `_emit_stream` re-raises whatever the session's
                        # generator raised, and `handle_connection` only
                        # catches ConnectionClosed/_ClientDisconnected.
                        self.metrics.streaming_failures.labels(
                            reason="generation"
                        ).inc()
                        logger.exception("tts: streaming generation failed")
                        await connection.send(
                            pack_message(
                                TtsErrorMessage(message="TTS generation failed")
                            )
                        )
                        await connection.close()
                        return
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
            try:
                return await self._emit_stream(
                    connection,
                    session.stream_eos(cancelled.is_set),
                    first_text_at,
                    first_output_sent,
                    cancelled,
                )
            except (ConnectionClosed, _ClientDisconnected):
                raise
            except GenerationLengthLimitError:
                # Same distinction as the Text-message call site above
                # (RAV-1552 S2): a legitimate terminal condition, not a
                # generation failure.
                self.metrics.streaming_failures.labels(reason="length_limit").inc()
                await connection.send(
                    pack_message(TtsErrorMessage(message=_LENGTH_LIMIT_MESSAGE))
                )
                return first_output_sent
            except Exception:
                # Same fix as the Text-message call site above (RAV-1552):
                # never let a generation failure during the trailing Eos
                # flush close the socket abruptly with no protocol Error.
                # Both callers of `_finish_session` (Eos message and legacy
                # null-byte Eos) close the connection themselves right after
                # this returns, so this branch only sends the Error.
                self.metrics.streaming_failures.labels(reason="generation").inc()
                logger.exception("tts: streaming generation failed")
                await connection.send(
                    pack_message(TtsErrorMessage(message="TTS generation failed"))
                )
                return first_output_sent
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
        except GenerationLengthLimitError:
            # Same distinction as the streaming-mode call sites above
            # (RAV-1552 S2): a legitimate terminal condition, not a
            # generation failure.
            self.metrics.buffered_turn_failures.labels(reason="length_limit").inc()
            await connection.send(
                pack_message(TtsErrorMessage(message=_LENGTH_LIMIT_MESSAGE))
            )
            return first_output_sent
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
