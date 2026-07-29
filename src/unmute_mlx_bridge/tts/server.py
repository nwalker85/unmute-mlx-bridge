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
import time
from collections.abc import Callable
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
            except ConnectionClosed:
                self.metrics.cancellations.inc()
            finally:
                self.health.set_session_active(False)
                self.metrics.active_sessions.set(0)

    async def _run_session(self, connection: ServerConnection, query: TtsStreamingQuery) -> None:
        assert self.bundle is not None
        voice = query.voice or self.config.default_voice
        session = self._session_cls(
            bundle=self.bundle, voice=voice, max_gen_length=self.config.max_gen_length
        )
        await connection.send(pack_message(TtsReadyMessage()))

        first_text_at: float | None = None
        first_output_sent = False

        async for raw in connection:
            if isinstance(raw, (bytes, bytearray)) and bytes(raw) == b"\x00":
                events = await asyncio.to_thread(session.push_eos)
                await self._emit(connection, events, first_text_at, first_output_sent)
                continue
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
                events = await asyncio.to_thread(session.push_text, message.text)
                first_output_sent = await self._emit(
                    connection, events, first_text_at, first_output_sent
                )
            elif isinstance(message, TtsEosMessage):
                events = await asyncio.to_thread(session.push_eos)
                first_output_sent = await self._emit(
                    connection, events, first_text_at, first_output_sent
                )

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
