"""`unmute-mlx-stt`: a `moshi-server`-compatible `/api/asr-streaming` WebSocket server
backed by real MLX inference on Apple Silicon.

Session lifecycle matches the design doc
(`docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md` §Session
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
from unmute_mlx_bridge.observability import Metrics, ServiceHealth, build_process_request
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
from unmute_mlx_bridge.stt.engine import SttModelBundle, SttSession, SttStepEvent

logger = logging.getLogger(__name__)

PROTOCOL_PATH = "/api/asr-streaming"


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
            logger.info("stt: model ready in %.1fs", time.monotonic() - start)
        except Exception:
            logger.exception("stt: model load failed")
            self.health.mark_load_failed("model load failed, see logs")
            raise

    async def handle_connection(self, connection: ServerConnection) -> None:
        if self.bundle is None:
            await connection.send(
                pack_message(SttErrorMessage(message="model still loading"))
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
        await connection.send(pack_message(SttReadyMessage()))

        first_audio_at: float | None = None
        first_output_sent = False

        async for raw in connection:
            if not isinstance(raw, (bytes, bytearray)):
                continue  # Text frames are not part of this protocol; ignore.
            try:
                data = unpack_message(raw)
                message = SttClientMessageAdapter.validate_python(data)
            except Exception as exc:
                self.metrics.protocol_errors.inc()
                await connection.send(
                    pack_message(SttErrorMessage(message=f"malformed frame: {exc}"))
                )
                continue

            if message.type == "Marker":
                marker: SttMarkerMessage = message  # type: ignore[assignment]
                session.push_marker(marker.id)
                events: list[SttStepEvent] = []
            else:  # Audio
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                self.metrics.input_audio_seconds.inc(len(message.pcm) / 24_000)
                events = await asyncio.to_thread(session.push_audio, message.pcm)

            for event in events:
                if not first_output_sent and event.kind in ("word", "step") and first_audio_at:
                    first_output_sent = True
                    self.metrics.time_to_first_output_seconds.observe(
                        time.monotonic() - first_audio_at
                    )
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

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_serve(config))


if __name__ == "__main__":
    run()
