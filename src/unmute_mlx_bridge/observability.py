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

---
TRANSCRIPT LOGGING POLICY (private canary boundary)
---
This module intentionally logs full STT word sequences and TTS text. That is a
deliberate, Nate-approved choice for the private canary deployment. The fields
``text`` and ``transcript`` are *explicitly excluded from sensitive-key filtering*
in ``_JsonFormatter``.

If this code is ever deployed outside the private canary, transcript logging MUST
be disabled by:
  1. Setting ``STT_LOG_TRANSCRIPTS=false`` / ``TTS_LOG_TRANSCRIPTS=false``
     (bridge servers — see ``log_transcripts`` in ``SttConfig`` / ``TtsConfig``),
     OR ``UNMUTE_LOG_TRANSCRIPTS=false`` (Unmute client), OR
  2. Removing the ``text``/``transcript`` extra fields from the structured log calls
     in stt/server.py and tts/server.py.

Never add ``text`` or ``transcript`` to Prometheus label values — labels are not
private and would leak into external monitoring systems.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import re
import socket
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Literal
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
CLIENT_SESSION_HEADER = "x-unmute-session-id"
CLIENT_CONVERSATION_HEADER = "x-unmute-conversation-id"
W3C_TRACEPARENT_HEADER = "traceparent"
W3C_TRACESTATE_HEADER = "tracestate"

# ---------------------------------------------------------------------------
# Correlation context
# ---------------------------------------------------------------------------


@dataclass
class CorrelationContext:
    """Session and turn identifiers for structured log correlation."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    conversation_id: str | None = None

    def new_turn(self) -> "CorrelationContext":
        return CorrelationContext(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )


def extract_client_session_id(headers: Headers) -> str | None:
    """Return the client-supplied session ID from WebSocket upgrade headers, or None."""
    value = headers.get(CLIENT_SESSION_HEADER)
    if value and re.fullmatch(r"[0-9a-f]{32}", value):
        return value


def extract_client_conversation_id(headers: Headers) -> str | None:
    """Return the client-supplied conversation ID from WebSocket upgrade headers, or None.

    The conversation ID is a 32-char hex string sent in the
    ``x-unmute-conversation-id`` header.  It is not a secret and safe to
    include in structured logs and span attributes for end-to-end correlation.
    """
    value = headers.get(CLIENT_CONVERSATION_HEADER)
    if value and re.fullmatch(r"[0-9a-f]{32}", value):
        return value
    return None
    return None


# ---------------------------------------------------------------------------
# URL sanitization
# ---------------------------------------------------------------------------


def _sanitize_url(url: str) -> str:
    """Strip credentials from a URL before logging.

    Removes:
    - userinfo (user:password@) from netloc
    - sensitive query parameters (token, key, secret, password, api_key, …)

    Example: ``https://user:pass@collector.example.com/path?api_key=secret``
             → ``https://***@collector.example.com/path?api_key=***``

    This function expects the *entire* string to be a URL.  For scrubbing URLs
    embedded inside a larger text string, use :func:`_scrub_urls_in_text`.
    """
    _SENSITIVE_QP = ("token", "key", "secret", "auth", "password")
    try:
        parsed = urllib.parse.urlsplit(url)
        changed = False

        netloc = parsed.netloc
        if parsed.username or parsed.password:
            netloc = re.sub(r"^[^@]+@", "***@", netloc)
            changed = True

        query = parsed.query
        if query:
            params = urllib.parse.parse_qs(query, keep_blank_values=True)
            cleaned = {
                k: (["***"] if any(s in k.lower() for s in _SENSITIVE_QP) else v)
                for k, v in params.items()
            }
            new_query = urllib.parse.urlencode(cleaned, doseq=True)
            if new_query != query:
                query = new_query
                changed = True

        if changed:
            return urllib.parse.urlunsplit(parsed._replace(netloc=netloc, query=query))
    except Exception:
        pass
    return url


# Regex that matches URLs embedded inside larger text strings.
# Captures https:// and http:// URLs up to the first whitespace, quote, or paren.
_EMBEDDED_URL_RE = re.compile(r"https?://\S+")


def _scrub_urls_in_text(text: str) -> str:
    """Scan *text* for embedded URLs and strip credentials from each.

    Replaces every ``https?://…`` substring that contains userinfo or a
    sensitive query parameter.  Non-credential URLs are left intact.
    """

    def _replace(m: re.Match) -> str:
        raw = m.group(0)
        # Strip trailing punctuation that is not part of the URL.
        trail = ""
        while raw and raw[-1] in ".,;:)\"'":
            trail = raw[-1] + trail
            raw = raw[:-1]
        return _sanitize_url(raw) + trail

    return _EMBEDDED_URL_RE.sub(_replace, text)


# Auth-material patterns in arbitrary text/exception strings.
_AUTH_PATTERN = re.compile(
    r"(?i)(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}",
)


def _scrub_text(text: str) -> str:
    """Remove obvious bearer/basic-auth credentials from arbitrary text."""
    return _AUTH_PATTERN.sub(r"\1 ***", text)


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Extra fields from ``extra={...}`` are included as top-level keys subject to
    these rules:

    1. **Sensitive-key filter** (recursive): any field whose name contains a
       known sensitive substring ('token', 'key', 'secret', 'auth', 'password')
       is dropped at every nesting level.
    2. **Value sanitization** (recursive): non-transcript string values are
       scanned for credential-bearing URLs (userinfo, sensitive query params) and
       bearer/basic-auth patterns; nested dicts and lists are sanitized
       recursively.
    3. **Transcript fidelity — top-level only**: values under the **top-level**
       ``text`` and ``transcript`` keys on the log record bypass ALL credential
       scrubbers.  This matches exactly the structured log fields emitted by the
       bridge (``stt_word``, ``tts_text_*`` events).  Nested dicts that happen
       to contain a ``text`` or ``transcript`` key are sanitized normally; only
       top-level record fields receive the bypass.
    4. **Message, exception, and extra string values**: all are scrubbed for
       credential-bearing URLs (userinfo + sensitive query params) and
       bearer/basic-auth patterns.  Ordinary URLs without credentials are never
       modified.

    See the module-level TRANSCRIPT LOGGING POLICY note before changing rule 3.
    """

    _SENSITIVE: tuple[str, ...] = ("token", "key", "secret", "auth", "password")
    # Only EXACT top-level record fields get the transcript bypass.
    # Nested dicts are NOT exempt — they go through normal sanitization.
    _ALLOWED_TRANSCRIPT_KEYS: frozenset[str] = frozenset(("text", "transcript"))
    _STDLIB: frozenset[str] = frozenset((
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message",
    ))

    def _is_sensitive(self, key: str) -> bool:
        if key in self._ALLOWED_TRANSCRIPT_KEYS:
            return False
        return any(s in key.lower() for s in self._SENSITIVE)

    def _sanitize_dict(self, d: dict) -> dict:
        """Sanitize a dict recursively.

        Sensitive keys are dropped at every nesting level.  Unlike the
        top-level ``format()`` loop, NO transcript bypass is applied here:
        nested ``text``/``transcript`` keys are sanitized the same as any
        other non-sensitive string value.
        """
        result = {}
        for k, v in d.items():
            if self._is_sensitive(k):
                continue
            result[k] = self._sanitize_value(v)
        return result

    def _sanitize_value(self, value: object) -> object:
        """Recursively sanitize a value.

        Does NOT apply the transcript bypass — that is top-level only.
        Use ``_sanitize_dict`` for key/value pairs inside nested dicts.
        """
        if isinstance(value, str):
            return _scrub_text(_scrub_urls_in_text(value))
        if isinstance(value, dict):
            return self._sanitize_dict(value)
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        return value

    def format(self, record: logging.LogRecord) -> str:
        # Scrub both bearer-auth patterns and credential-bearing embedded URLs
        # from the formatted message string.
        msg = _scrub_text(_scrub_urls_in_text(record.getMessage()))
        obj: dict[str, object] = {
            "ts": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        for k, v in record.__dict__.items():
            if k in self._STDLIB or k.startswith("_"):
                continue
            if self._is_sensitive(k):
                continue
            # Top-level transcript fields bypass ALL scrubbers.
            # Nested dicts do not receive this bypass (see _sanitize_dict).
            if k in self._ALLOWED_TRANSCRIPT_KEYS:
                obj[k] = v
            else:
                obj[k] = self._sanitize_value(v)
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            obj["exc"] = _scrub_text(_scrub_urls_in_text(exc_text))
        return json.dumps(obj, default=str)


def setup_logging(level: str = "INFO", log_format: Literal["json", "text"] = "text") -> None:
    """Configure root logger. Call once at process startup before serving."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Optional OpenTelemetry integration
# ---------------------------------------------------------------------------
#
# opentelemetry-api provides a no-op tracer by default when no provider is
# configured; importing it is always safe.  If the package is absent the shims
# below make every call a no-op.  The SDK and OTLP exporter are fully optional
# — inference continues without them.

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]

    _OTEL_API_AVAILABLE = True
except ImportError:
    _OTEL_API_AVAILABLE = False

try:
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor  # type: ignore[import-untyped]

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-untyped]
        OTLPSpanExporter as _OTLPSpanExporter,
    )

    _OTEL_OTLP_AVAILABLE = True
except ImportError:
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-untyped]
            OTLPSpanExporter as _OTLPSpanExporter,
        )

        _OTEL_OTLP_AVAILABLE = True
    except ImportError:
        _OTEL_OTLP_AVAILABLE = False

_otel_initialized = False


def init_otel(service_name: str, endpoint: str | None = None) -> bool:
    """Initialize OTEL SDK if available and endpoint is configured.

    Call this once at process startup, after ``setup_logging``.  The endpoint
    is read from the ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable when
    not supplied explicitly.  The endpoint URL is sanitized (credentials
    stripped) before logging.

    Returns True if a real TracerProvider was installed.  Returns False and
    leaves the no-op API tracer active when the endpoint is absent, the SDK
    packages are not installed, or initialisation fails.  Inference is never
    interrupted on failure.

    Endpoint topology (Alloy/Tempo vs Langfuse OTLP shim) is left to deployment
    configuration; this function only reads the standard env var and does not
    assume any particular backend.
    """
    global _otel_initialized
    if _otel_initialized:
        return False

    import os

    resolved_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    _log = logging.getLogger(__name__)
    if not resolved_endpoint:
        _log.debug("otel: OTEL_EXPORTER_OTLP_ENDPOINT not set; using no-op tracer")
        _otel_initialized = True
        return False

    safe_endpoint = _sanitize_url(resolved_endpoint)

    if not _OTEL_SDK_AVAILABLE or not _OTEL_OTLP_AVAILABLE:
        _log.warning(
            "otel: endpoint configured but opentelemetry-sdk / exporter not installed; "
            "falling back to no-op tracer",
            extra={"endpoint": safe_endpoint},
        )
        _otel_initialized = True
        return False

    if not _OTEL_API_AVAILABLE:
        _otel_initialized = True
        return False

    try:
        exporter = _OTLPSpanExporter(endpoint=resolved_endpoint)  # type: ignore[arg-type, misc]
        provider = _TracerProvider()  # type: ignore[no-untyped-call, misc]
        provider.add_span_processor(_BatchSpanProcessor(exporter))  # type: ignore[no-untyped-call, misc]
        _otel_trace.set_tracer_provider(provider)  # type: ignore[union-attr]
        _log.info(
            "otel: TracerProvider initialized",
            extra={"service_name": service_name, "endpoint": safe_endpoint},
        )
        _otel_initialized = True
        return True
    except Exception:
        _log.exception("otel: TracerProvider initialisation failed; using no-op")
        _otel_initialized = True
        return False


def get_tracer(name: str = "unmute_mlx_bridge"):
    """Return an OTEL tracer or a no-op shim if the API package is absent."""
    if _OTEL_API_AVAILABLE:
        return _otel_trace.get_tracer(name)  # type: ignore[union-attr]
    return _NoOpTracer()


class _NoOpSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def set_attribute(self, *_):
        pass

    def record_exception(self, *_):
        pass

    def set_status(self, *_):
        pass

    def end(self):
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **_kwargs):
        return _NoOpSpan()

    def start_span(self, name: str, **_kwargs):
        return _NoOpSpan()


def extract_otel_context(headers: Headers) -> object | None:
    """Extract W3C trace context from WebSocket upgrade headers.

    Reads ``traceparent`` and (if present) ``tracestate`` from the upgrade
    request headers and calls ``opentelemetry.propagate.extract`` to construct
    an OTEL context object suitable for passing as ``context=`` to
    ``tracer.start_span`` / ``tracer.start_as_current_span``.

    Returns ``None`` when:
    - ``opentelemetry-api`` is not installed
    - no ``traceparent`` header is present
    - propagation extraction raises (logged at DEBUG, never propagated)

    **This is the only supported way to pass W3C trace context to bridge
    spans.** The OTEL SDK does NOT automatically extract context from WebSocket
    upgrade headers; automatic context propagation only applies to libraries
    that explicitly instrument their HTTP/gRPC layer.
    """
    if not _OTEL_API_AVAILABLE:
        return None
    try:
        from opentelemetry import propagate as _otel_propagate  # type: ignore[import-untyped]
        traceparent = headers.get(W3C_TRACEPARENT_HEADER)
        if not traceparent:
            return None
        carrier: dict[str, str] = {W3C_TRACEPARENT_HEADER: traceparent}
        tracestate = headers.get(W3C_TRACESTATE_HEADER)
        if tracestate:
            carrier[W3C_TRACESTATE_HEADER] = tracestate
        return _otel_propagate.extract(carrier)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "otel: traceparent extraction failed: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# Clock / startup metadata
# ---------------------------------------------------------------------------


@dataclass
class ClockMetadata:
    """Startup UTC wall-clock reference and NTP source DNS check result.

    ``ntp_hostname_dns_resolved`` records whether the NTP source hostname
    resolved in DNS at startup.  This is a DNS-only reachability probe —
    it does NOT verify that the system clock is synchronized to that source.
    NTP synchronization status must be verified externally (e.g. via
    ``chronyc tracking`` or ``timedatectl show``).

    Use ``utc_startup`` for cross-service event correlation (wall-clock).
    Use ``monotonic_start_ref`` as the epoch for duration calculations within
    the same process (monotonic, no wall-clock jumps).
    """

    utc_startup: str
    monotonic_start_ref: float
    expected_ntp_source: str
    ntp_hostname_dns_resolved: bool | None
    ntp_source_addr: str | None
    ntp_source_error: str | None
    # Sync is always externally verified; this field is a reminder, not a claim.
    sync_verified_externally: None = None


def log_clock_metadata(
    logger: logging.Logger | None = None,
    ntp_source: str | None = None,
    resolve_ntp: bool = True,
) -> ClockMetadata:
    """Log startup UTC time, monotonic reference, and expected NTP source.

    Non-mutating: never modifies system NTP configuration or system time.
    When ``resolve_ntp`` is True, performs a DNS-only hostname resolution check
    (``getaddrinfo``) to verify the NTP source is reachable in DNS.

    ``ntp_source`` defaults to the ``NTP_SOURCE`` environment variable, falling
    back to the public ``pool.ntp.org`` NTP pool if unset. Deployments with a
    specific NTP source (e.g. a local stratum-1 or GPS-disciplined server) should
    set ``NTP_SOURCE`` rather than relying on the default.

    IMPORTANT: a successful DNS resolution does NOT prove the system clock is
    synchronized.  The log event includes ``ntp_sync_note`` explicitly stating
    this.  Synchronization must be verified via ``chronyc tracking`` or
    equivalent on the host.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if ntp_source is None:
        ntp_source = os.getenv("NTP_SOURCE", "pool.ntp.org")

    utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    mono_ref = time.monotonic()

    ntp_resolved: bool | None = None
    ntp_addr: str | None = None
    ntp_error: str | None = None

    if resolve_ntp:
        try:
            addrs = socket.getaddrinfo(ntp_source, 123, socket.AF_UNSPEC, socket.SOCK_DGRAM)
            ntp_resolved = True
            ntp_addr = str(addrs[0][4][0]) if addrs else None
        except OSError as exc:
            ntp_resolved = False
            ntp_error = str(exc)

    meta = ClockMetadata(
        utc_startup=utc_now,
        monotonic_start_ref=mono_ref,
        expected_ntp_source=ntp_source,
        ntp_hostname_dns_resolved=ntp_resolved,
        ntp_source_addr=ntp_addr,
        ntp_source_error=ntp_error,
    )
    logger.info(
        "startup clock metadata",
        extra={
            "event": "clock_metadata",
            "utc_startup": meta.utc_startup,
            "expected_ntp_source": meta.expected_ntp_source,
            "ntp_hostname_dns_resolved": meta.ntp_hostname_dns_resolved,
            "ntp_source_addr": meta.ntp_source_addr,
            "ntp_source_error": meta.ntp_source_error,
            "ntp_sync_note": (
                "DNS resolution only — NTP synchronization must be verified "
                "externally via chronyc or timedatectl"
            ),
        },
    )
    return meta


# ---------------------------------------------------------------------------
# Build info
# ---------------------------------------------------------------------------


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
    # --- core session counters ---
    model_load_seconds: Gauge = field(init=False)
    active_sessions: Gauge = field(init=False)
    rejected_sessions: Counter = field(init=False)
    protocol_errors: Counter = field(init=False)
    cancellations: Counter = field(init=False)
    # --- audio throughput ---
    input_audio_seconds: Counter = field(init=False)
    output_audio_seconds: Counter = field(init=False)
    # --- timing: first output ---
    time_to_first_output_seconds: Histogram = field(init=False)
    # --- inference step (STT: wall time of push_audio batches) ---
    inference_step_seconds: Histogram = field(init=False)
    inference_step_batch_frames: Histogram = field(init=False)
    inference_step_rtf: Histogram = field(init=False)
    # --- STT application-level inference tracking ---
    # ``stt_inference_frames_active``: frames in the current push_audio call.
    # Value is >0 while inference is running, 0 when the session is idle.
    # This is NOT the websockets library receive-queue depth — that internal
    # queue is bounded by ``max_queue`` (SttConfig.max_recv_queue) and is not
    # directly observable here.  An explicit ``stt_recv_queue_bound`` gauge
    # documents the configured limit so operators know what the ceiling is.
    stt_inference_frames_active: Gauge = field(init=False)
    stt_recv_queue_bound: Gauge = field(init=False)
    stt_oversized_frames_total: Counter = field(init=False)
    # --- buffered TTS metrics ---
    buffered_input_characters: Histogram = field(init=False)
    buffered_audio_seconds: Histogram = field(init=False)
    buffered_synthesis_seconds: Histogram = field(init=False)
    buffered_eos_to_first_emit_seconds: Histogram = field(init=False)
    buffered_turn_failures: Counter = field(init=False)

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
            "Estimated wall time per model step (push_audio batch duration / frame count)",
            buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 2.0),
            registry=self.registry,
        )
        self.inference_step_batch_frames = Histogram(
            "bridge_inference_step_batch_frames",
            "Number of 1920-sample frames processed per push_audio call",
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=self.registry,
        )
        self.inference_step_rtf = Histogram(
            "bridge_inference_step_rtf",
            "Real-time factor per push_audio batch (inference_s / audio_s); <1 = faster than real-time",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry,
        )
        self.stt_inference_frames_active = Gauge(
            "bridge_stt_inference_frames_active",
            "Frames in the current active push_audio call (0=idle, N=inference running). "
            "Not the websockets receive-queue depth; see bridge_stt_recv_queue_bound "
            "for the configured websockets receive-queue limit.",
            registry=self.registry,
        )
        self.stt_recv_queue_bound = Gauge(
            "bridge_stt_recv_queue_bound",
            "Configured websockets receive-queue depth limit (max_queue parameter). "
            "Set at startup; does not change at runtime.",
            registry=self.registry,
        )
        self.stt_oversized_frames_total = Counter(
            "bridge_stt_oversized_frames_total",
            "STT audio frames rejected because they exceed the configured sample limit",
            registry=self.registry,
        )
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
