"""Shared HTTP operations surface: health, build info, metrics, and auth.

Both servers are single `websockets` processes that also need to answer plain HTTP
health/metrics probes on the same port. `websockets.asyncio.server.serve`'s
`process_request` hook lets us intercept non-upgrade requests before the WebSocket
handshake and answer them directly; anything else falls through to the normal
protocol upgrade.

Auth mirrors real `moshi-server` (`rust/moshi-server/src/main.rs`): the
`kyutai-api-key` header is checked first, falling back to the `auth_id` query
parameter ("It's tricky to set the headers of a websocket in javascript so we pass
the token via the query too" — same comment applies to browsers/JS clients, not to
Unmute's Python client which always sends the header, but we replicate the fallback
for fidelity).
"""

from __future__ import annotations

import json
import platform
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import parse_qs, urlsplit

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

ID_HEADER = "kyutai-api-key"


def _build_info() -> dict[str, str]:
    """Return the field-compatible shape exposed by real `moshi-server`.

    Rust-specific fields are retained because stock Unmute probes this endpoint
    before admitting a session. Their values describe the Python/MLX bridge
    honestly rather than pretending this process was built by Rust or Cargo.
    """
    try:
        package_version = version("unmute-mlx-bridge")
    except PackageNotFoundError:
        package_version = "unknown"
    target = f"{platform.machine()}-{platform.system().lower()}"
    return {
        "build_timestamp": "unknown",
        "build_date": "unknown",
        "git_branch": "unknown",
        "git_timestamp": "unknown",
        "git_date": "unknown",
        "git_hash": "unknown",
        "git_describe": f"unmute-mlx-bridge {package_version}",
        "rustc_host_triple": "not-applicable",
        "rustc_version": f"python {platform.python_version()}",
        "cargo_target_triple": target,
    }


@dataclass
class ServiceHealth:
    """Mutable readiness/liveness state shared between the model-load task and the
    HTTP probe handlers. Thread-safe: `model_loop` (a worker thread) and the asyncio
    event loop both touch this.
    """

    model_loaded: bool = False
    model_load_error: str | None = None
    session_active: bool = False
    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_loaded(self) -> None:
        with self._lock:
            self.model_loaded = True
            self.model_load_error = None

    def mark_load_failed(self, error: str) -> None:
        with self._lock:
            self.model_loaded = False
            self.model_load_error = error

    def set_session_active(self, active: bool) -> None:
        with self._lock:
            self.session_active = active

    def is_ready(self) -> bool:
        with self._lock:
            return self.model_loaded and not self.session_active

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "model_loaded": self.model_loaded,
                "model_load_error": self.model_load_error,
                "session_active": self.session_active,
                "uptime_s": round(time.monotonic() - self.started_at, 3),
            }


@dataclass
class Metrics:
    """Prometheus collectors. One instance per process (own registry, so STT and
    TTS metrics never collide if ever imported into the same interpreter, e.g. in
    tests).
    """

    registry: CollectorRegistry = field(default_factory=CollectorRegistry)
    model_load_seconds: Gauge = field(init=False)
    active_sessions: Gauge = field(init=False)
    rejected_sessions: Counter = field(init=False)
    protocol_errors: Counter = field(init=False)
    cancellations: Counter = field(init=False)
    input_audio_seconds: Counter = field(init=False)
    output_audio_seconds: Counter = field(init=False)
    time_to_first_output_seconds: Histogram = field(init=False)
    inference_step_seconds: Histogram = field(init=False)

    def __post_init__(self) -> None:
        self.model_load_seconds = Gauge(
            "bridge_model_load_seconds", "Time to load model weights", registry=self.registry
        )
        self.active_sessions = Gauge(
            "bridge_active_sessions",
            "Currently admitted WebSocket sessions",
            registry=self.registry,
        )
        self.rejected_sessions = Counter(
            "bridge_rejected_sessions_total",
            "Sessions rejected because a slot was already in use",
            registry=self.registry,
        )
        self.protocol_errors = Counter(
            "bridge_protocol_errors_total",
            "Malformed or unsupported client frames",
            registry=self.registry,
        )
        self.cancellations = Counter(
            "bridge_cancellations_total",
            "Sessions cancelled by client disconnect",
            registry=self.registry,
        )
        self.input_audio_seconds = Counter(
            "bridge_input_audio_seconds_total", "Audio seconds received", registry=self.registry
        )
        self.output_audio_seconds = Counter(
            "bridge_output_audio_seconds_total", "Audio seconds synthesized", registry=self.registry
        )
        self.time_to_first_output_seconds = Histogram(
            "bridge_time_to_first_output_seconds",
            "Time from first input to first meaningful output per session",
            registry=self.registry,
        )
        self.inference_step_seconds = Histogram(
            "bridge_inference_step_seconds",
            "Wall time of a single model step",
            registry=self.registry,
        )


def check_auth(request: Request, authorized_ids: frozenset[str]) -> bool:
    """Replicates `main.rs`'s auth precedence: header first, then `auth_id` query param."""
    if not authorized_ids:
        # An explicitly empty allow-list means auth is disabled (loopback dev default).
        return True
    header_value = request.headers.get(ID_HEADER)
    if header_value is not None:
        return header_value in authorized_ids
    query = parse_qs(urlsplit(request.path).query)
    auth_id = query.get("auth_id", [None])[0]
    return auth_id is not None and auth_id in authorized_ids


def build_process_request(
    health: ServiceHealth,
    metrics: Metrics,
    protocol_path: str,
    authorized_ids: frozenset[str],
):
    """Returns a `process_request` callable for `websockets.asyncio.server.serve`.

    Handles `/healthz`, `/readyz`, `/metrics`, and moshi-server-compatible
    `/api/build_info` directly, and pre-upgrade-rejects unauthorized requests to
    `protocol_path` with HTTP 401 — matching real `moshi-server`, which checks
    `authorized_ids` before completing the WebSocket handshake
    (`main.rs::streaming_t`), not after.
    """

    async def process_request(connection, request: Request) -> Response | None:  # noqa: ARG001
        path = urlsplit(request.path).path
        if path == "/healthz":
            headers = Headers()
            headers["Content-Type"] = "text/plain"
            return Response(200, "OK", headers, b"ok\n")
        if path == "/readyz":
            body = (str(health.snapshot()) + "\n").encode()
            status = 200 if health.is_ready() else 503
            reason = "OK" if status == 200 else "Service Unavailable"
            headers = Headers()
            headers["Content-Type"] = "text/plain"
            return Response(status, reason, headers, body)
        if path == "/metrics":
            body = generate_latest(metrics.registry)
            headers = Headers()
            headers["Content-Type"] = CONTENT_TYPE_LATEST
            return Response(200, "OK", headers, body)
        if path == "/api/build_info":
            body = (json.dumps(_build_info(), separators=(",", ":")) + "\n").encode()
            headers = Headers()
            headers["Content-Type"] = "application/json"
            return Response(200, "OK", headers, body)
        if path == protocol_path and not check_auth(request, authorized_ids):
            return Response(401, "Unauthorized", Headers(), b"")
        return None  # Fall through to the normal WebSocket upgrade.

    return process_request
