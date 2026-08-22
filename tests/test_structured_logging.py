"""Tests for structured logging, correlation IDs, OTEL disabled fallback,
clock metadata, URL sanitization, and sensitive-value filtering.
These are unit/conformance tests that run on any platform without hardware.
"""
from __future__ import annotations

import json
import logging
import time

import pytest

from unmute_mlx_bridge.observability import (
    CLIENT_SESSION_HEADER,
    ClockMetadata,
    CorrelationContext,
    _JsonFormatter,
    _sanitize_url,
    _scrub_text,
    extract_client_session_id,
    extract_otel_context,
    get_tracer,
    init_otel,
    log_clock_metadata,
    setup_logging,
)


# ---------------------------------------------------------------------------
# CorrelationContext
# ---------------------------------------------------------------------------


def test_correlation_context_generates_unique_session_ids():
    a = CorrelationContext()
    b = CorrelationContext()
    assert a.session_id != b.session_id
    assert len(a.session_id) == 32  # uuid4 hex


def test_correlation_context_new_turn_shares_session_id():
    ctx = CorrelationContext()
    turn = ctx.new_turn()
    assert turn.session_id == ctx.session_id
    assert turn.turn_id != ctx.turn_id


def test_correlation_context_accepts_explicit_session_id():
    ctx = CorrelationContext(session_id="aabbccddeeff00112233445566778899")
    assert ctx.session_id == "aabbccddeeff00112233445566778899"


# ---------------------------------------------------------------------------
# URL sanitization
# ---------------------------------------------------------------------------


def test_sanitize_url_strips_credentials():
    url = "https://user:password@collector.example.com/path"
    result = _sanitize_url(url)
    assert "password" not in result
    assert "user" not in result
    assert "collector.example.com" in result
    assert "***@" in result


def test_sanitize_url_leaves_clean_url_unchanged():
    url = "http://collector.example.com:4318/v1/traces"
    assert _sanitize_url(url) == url


def test_sanitize_url_handles_non_url_gracefully():
    assert _sanitize_url("not-a-url") == "not-a-url"
    assert _sanitize_url("") == ""


# ---------------------------------------------------------------------------
# JsonFormatter — sensitive filtering
# ---------------------------------------------------------------------------


def test_json_formatter_produces_valid_json():
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["msg"] == "hello world"
    assert obj["level"] == "INFO"
    assert "ts" in obj


def test_json_formatter_includes_extra_fields():
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="event", args=(), exc_info=None,
    )
    record.session_id = "abc123"
    record.text = "hello world"
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["session_id"] == "abc123"
    assert obj["text"] == "hello world"


def test_json_formatter_strips_sensitive_fields():
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="event", args=(), exc_info=None,
    )
    record.auth_token = "secret-value"
    record.api_key = "should-not-appear"
    record.session_id = "safe-id"
    output = fmt.format(record)
    obj = json.loads(output)
    assert "auth_token" not in obj
    assert "api_key" not in obj
    assert obj["session_id"] == "safe-id"
    assert "secret-value" not in output
    assert "should-not-appear" not in output


def test_json_formatter_does_not_log_auth_tokens():
    """Auth tokens must never appear in any log output."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="check no token leak", args=(), exc_info=None,
    )
    for sensitive in ("kyutai_token", "auth_id", "password", "api_secret"):
        setattr(record, sensitive, "LEAKED_VALUE")
    output = fmt.format(record)
    assert "LEAKED_VALUE" not in output


def test_json_formatter_does_not_redact_transcript_text():
    """Transcript text is intentionally logged for the private canary."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="stt word", args=(), exc_info=None,
    )
    record.text = "hello world"
    record.transcript = "hello world goodbye"
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["text"] == "hello world"
    assert obj["transcript"] == "hello world goodbye"


def test_json_formatter_sanitizes_credential_url_in_extra_values():
    """A URL value containing credentials must be sanitized."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="otel init", args=(), exc_info=None,
    )
    record.endpoint = "https://user:s3cr3t@otlp.example.com/v1/traces"
    output = fmt.format(record)
    assert "s3cr3t" not in output
    assert "otlp.example.com" in output


def test_setup_logging_json_produces_json_lines():
    """setup_logging with json format installs _JsonFormatter."""
    import logging as _logging
    setup_logging("DEBUG", "json")
    root = _logging.getLogger()
    assert isinstance(root.handlers[0].formatter, _JsonFormatter)
    setup_logging("WARNING", "text")


# ---------------------------------------------------------------------------
# OTEL disabled fallback
# ---------------------------------------------------------------------------


def test_get_tracer_returns_tracer_or_noop():
    """get_tracer() must not raise even if opentelemetry packages are absent."""
    tracer = get_tracer("test")
    with tracer.start_as_current_span("test.span") as span:
        span.set_attribute("key", "value")


def test_otel_disabled_fallback_does_not_raise_on_inference_path():
    tracer = get_tracer("unmute_mlx_bridge.stt")
    with tracer.start_as_current_span("stt.session") as span:
        span.set_attribute("session.id", "test-session-id")
        span.set_attribute("step.frames", 1)


def test_init_otel_no_endpoint_returns_false(monkeypatch):
    """init_otel returns False when no endpoint is configured."""
    import unmute_mlx_bridge.observability as obs_module
    monkeypatch.setattr(obs_module, "_otel_initialized", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    result = obs_module.init_otel("test-service")
    assert result is False


def test_init_otel_sanitizes_endpoint_in_logs(monkeypatch, caplog):
    """init_otel must not log the raw endpoint URL when it contains credentials."""
    import unmute_mlx_bridge.observability as obs_module
    monkeypatch.setattr(obs_module, "_otel_initialized", False)
    monkeypatch.setattr(obs_module, "_OTEL_SDK_AVAILABLE", False)
    monkeypatch.setattr(obs_module, "_OTEL_OTLP_AVAILABLE", False)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://admin:hunter2@otel.example.com/ingest",
    )
    with caplog.at_level(logging.WARNING):
        obs_module.init_otel("test-service")
    full_log = " ".join(r.getMessage() for r in caplog.records) + str(caplog.records)
    assert "hunter2" not in full_log


# ---------------------------------------------------------------------------
# Clock metadata — honest wording
# ---------------------------------------------------------------------------


def test_log_clock_metadata_returns_clock_metadata():
    meta = log_clock_metadata(resolve_ntp=False)
    assert isinstance(meta, ClockMetadata)
    assert meta.utc_startup.endswith("+00:00") or meta.utc_startup.endswith("Z")
    assert meta.monotonic_start_ref > 0
    assert meta.expected_ntp_source == "pool.ntp.org"
    assert meta.ntp_hostname_dns_resolved is None  # resolve_ntp=False
    assert meta.sync_verified_externally is None


def test_log_clock_metadata_logs_event():
    import io
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.DEBUG)
    obs_logger = logging.getLogger("unmute_mlx_bridge.observability")
    obs_logger.addHandler(handler)
    obs_logger.setLevel(logging.DEBUG)
    try:
        log_clock_metadata(resolve_ntp=False)
        output = handler.stream.getvalue()
    finally:
        obs_logger.removeHandler(handler)
    assert "clock_metadata" in output or "startup clock metadata" in output


def test_log_clock_metadata_includes_sync_note():
    """The startup log must include a note that DNS resolution != NTP sync."""
    import io
    handler = logging.StreamHandler(io.StringIO())
    handler.setFormatter(_JsonFormatter())
    handler.setLevel(logging.DEBUG)
    obs_logger = logging.getLogger("unmute_mlx_bridge.observability")
    obs_logger.addHandler(handler)
    obs_logger.setLevel(logging.DEBUG)
    try:
        log_clock_metadata(resolve_ntp=False)
        output = handler.stream.getvalue()
    finally:
        obs_logger.removeHandler(handler)
    obj = json.loads(output.strip())
    assert "ntp_sync_note" in obj
    note = obj["ntp_sync_note"]
    assert "DNS" in note or "externally" in note


def test_log_clock_metadata_ntp_skipped_when_resolve_false():
    meta = log_clock_metadata(resolve_ntp=False)
    assert meta.ntp_hostname_dns_resolved is None
    assert meta.ntp_source_addr is None
    assert meta.ntp_source_error is None


def test_monotonic_reference_is_positive():
    meta = log_clock_metadata(resolve_ntp=False)
    assert meta.monotonic_start_ref > 0
    later = time.monotonic()
    assert later >= meta.monotonic_start_ref


def test_log_clock_metadata_ntp_source_from_env(monkeypatch):
    monkeypatch.setenv("NTP_SOURCE", "ntp.example.internal")
    meta = log_clock_metadata(resolve_ntp=False)
    assert meta.expected_ntp_source == "ntp.example.internal"


def test_log_clock_metadata_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("NTP_SOURCE", "ntp.example.internal")
    meta = log_clock_metadata(resolve_ntp=False, ntp_source="explicit.example.org")
    assert meta.expected_ntp_source == "explicit.example.org"


# ---------------------------------------------------------------------------
# Cross-repo correlation — header extraction
# ---------------------------------------------------------------------------


def test_extract_client_session_id_accepts_valid_hex():
    from websockets.datastructures import Headers
    valid_sid = "aabbccddeeff00112233445566778899"
    headers = Headers()
    headers[CLIENT_SESSION_HEADER] = valid_sid
    result = extract_client_session_id(headers)
    assert result == valid_sid


def test_extract_client_session_id_rejects_invalid_format():
    from websockets.datastructures import Headers
    headers = Headers()
    headers[CLIENT_SESSION_HEADER] = "not-a-hex-string"
    assert extract_client_session_id(headers) is None


def test_extract_client_session_id_returns_none_when_absent():
    from websockets.datastructures import Headers
    headers = Headers()
    assert extract_client_session_id(headers) is None


# ---------------------------------------------------------------------------
# Sanitization hardening
# ---------------------------------------------------------------------------


def test_sanitize_url_strips_query_param_credentials():
    """Sensitive query parameters must be redacted."""
    url = "https://api.example.com/v1/traces?api_key=super_secret&trace_id=abc"
    result = _sanitize_url(url)
    assert "super_secret" not in result
    # The sentinel value may be URL-encoded; verify the original secret is gone
    # and the non-sensitive param is preserved.
    assert "api_key=" in result  # param key still present
    assert "trace_id=abc" in result  # non-sensitive params preserved


def test_sanitize_url_strips_both_userinfo_and_query_creds():
    url = "https://admin:pass@host.example.com/path?token=abc123"
    result = _sanitize_url(url)
    assert "pass" not in result
    assert "abc123" not in result
    assert "host.example.com" in result


def test_scrub_text_removes_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig"
    result = _scrub_text(text)
    assert "eyJhbGciOiJSUzI1NiJ9" not in result
    assert "Bearer ***" in result


def test_scrub_text_removes_basic_auth():
    text = "got Basic dXNlcjpwYXNz in header"
    result = _scrub_text(text)
    assert "dXNlcjpwYXNz" not in result
    assert "Basic ***" in result


def test_scrub_text_leaves_normal_text_unchanged():
    text = "stt: session started with session_id=abc123"
    assert _scrub_text(text) == text


def test_json_formatter_sanitizes_nested_dict_values():
    """Credential-bearing URLs inside nested dicts are sanitized."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="nested", args=(), exc_info=None,
    )
    record.metadata = {"endpoint": "https://user:secret@otel.example.com/"}
    output = fmt.format(record)
    assert "secret" not in output
    assert "otel.example.com" in output


def test_json_formatter_drops_nested_sensitive_keys():
    """Sensitive keys inside nested dicts must be dropped at every nesting level."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="nested-sensitive", args=(), exc_info=None,
    )
    record.config = {"api_key": "should-not-appear", "host": "safe-host.example.com"}
    record.deep = {"level1": {"auth_token": "leaked-value", "ok": "visible"}}
    output = fmt.format(record)
    obj = json.loads(output)
    assert "api_key" not in output
    assert "should-not-appear" not in output
    assert "auth_token" not in output
    assert "leaked-value" not in output
    assert obj["config"]["host"] == "safe-host.example.com"
    assert obj["deep"]["level1"]["ok"] == "visible"


def test_json_formatter_transcript_values_bypass_scrubbers():
    """Transcript text containing credential-shaped strings must pass through unchanged.
    
    A spoken sentence like 'use Bearer token for auth' is NOT a credential —
    it is speech content and must not be modified by credential scrubbers.
    """
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="stt word", args=(), exc_info=None,
    )
    # A sentence a user might speak that contains credential-shaped words
    spoken = "Please use Bearer eyJhbGciOiJSUzI1NiJ9 for authentication with api_key value"
    record.text = spoken
    record.transcript = "connect to https://user:pass@example.com and use Basic dXNlcjpwYXNz"
    output = fmt.format(record)
    obj = json.loads(output)
    # Transcript fields must come through exactly as-is
    assert obj["text"] == spoken
    assert obj["transcript"] == "connect to https://user:pass@example.com and use Basic dXNlcjpwYXNz"


def test_json_formatter_nested_transcript_key_is_sanitized():
    """Nested ``text``/``transcript`` keys inside dicts are sanitized normally.

    Only TOP-LEVEL record fields named ``text``/``transcript`` get the bypass.
    A nested dict (e.g., event metadata payload) with a ``text`` key must still
    have its value scrubbed for credential-bearing strings.
    """
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="nested-transcript", args=(), exc_info=None,
    )
    credential_string = "Bearer eyJhbGciOiJSUzI1NiJ9.foobar.sig"
    record.event_data = {"text": credential_string, "confidence": 0.97}
    output = fmt.format(record)
    obj = json.loads(output)
    # The nested 'text' value must have had the bearer token scrubbed
    assert "eyJhbGciOiJSUzI1NiJ9" not in obj["event_data"]["text"]


def test_json_formatter_top_level_transcript_preserves_credential_shaped_text():
    """Top-level ``text``/``transcript`` fields preserve exact content, even if
    the string looks credential-shaped.  A user utterance like 'use Bearer token'
    must not be modified."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="stt word", args=(), exc_info=None,
    )
    spoken = "Please use Bearer eyJhbGciOiJSUzI1NiJ9 for authentication"
    record.text = spoken
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["text"] == spoken


def test_json_formatter_nested_text_key_scrubbed_while_top_level_preserved():
    """Confirm the two rules together: top-level bypass + nested sanitization."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="combined", args=(), exc_info=None,
    )
    spoken = "Bearer eyJhbGciOiJSUzI1NiJ9 spoken aloud"
    record.text = spoken  # top-level → bypass
    record.meta = {"text": spoken}  # nested → sanitize
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["text"] == spoken  # top-level: preserved
    assert "eyJhbGciOiJSUzI1NiJ9" not in obj["meta"]["text"]  # nested: scrubbed


def test_json_formatter_sanitizes_nested_list_values():
    """Credential-bearing URLs inside lists are sanitized."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="list", args=(), exc_info=None,
    )
    record.urls = ["https://user:s3cr3t@host.example.com/path", "https://safe.example.com"]
    output = fmt.format(record)
    assert "s3cr3t" not in output
    assert "safe.example.com" in output


def test_json_formatter_scrubs_bearer_in_message():
    """Bearer tokens in the formatted message string must be scrubbed."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="got Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.foobar.sig",
        args=(), exc_info=None,
    )
    output = fmt.format(record)
    assert "eyJhbGciOiJSUzI1NiJ9" not in output


def test_json_formatter_scrubs_bearer_in_exception():
    """Bearer tokens in exception text must be scrubbed."""
    fmt = _JsonFormatter()
    try:
        raise ValueError("token rejected: Bearer eyJhbGciOiJSUzI1NiJ9.abc.def")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="", lineno=0,
        msg="error", args=(), exc_info=exc_info,
    )
    output = fmt.format(record)
    assert "eyJhbGciOiJSUzI1NiJ9" not in output


def test_json_formatter_scrubs_url_userinfo_in_message():
    """Credential-bearing URLs in the formatted message string must have
    userinfo stripped.  Ordinary host-only URLs must be left intact."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="connecting to https://admin:s3cr3t@db.example.com/path",
        args=(), exc_info=None,
    )
    output = fmt.format(record)
    assert "s3cr3t" not in output
    assert "db.example.com" in output


def test_json_formatter_scrubs_url_query_creds_in_message():
    """Sensitive query-param credentials in the message string must be removed."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="url=https://api.example.com/v1?api_key=super_secret&ok=yes",
        args=(), exc_info=None,
    )
    output = fmt.format(record)
    assert "super_secret" not in output
    assert "api.example.com" in output
    assert "ok=yes" in output


def test_json_formatter_message_url_no_creds_unchanged():
    """Message strings containing ordinary URLs without credentials must
    not be modified."""
    fmt = _JsonFormatter()
    msg = "started server at https://inference.example.com/v1/stt"
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["msg"] == msg


def test_json_formatter_transcript_field_unchanged_despite_url():
    """Top-level transcript fields with credential-shaped URLs must NOT be
    sanitized even though the message path now applies _sanitize_url."""
    fmt = _JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="stt word", args=(), exc_info=None,
    )
    spoken = "visit https://user:pass@example.com to log in"
    record.text = spoken
    output = fmt.format(record)
    obj = json.loads(output)
    assert obj["text"] == spoken


# ---------------------------------------------------------------------------
# extract_otel_context
# ---------------------------------------------------------------------------


def test_extract_otel_context_returns_none_when_no_traceparent():
    from websockets.datastructures import Headers
    headers = Headers()
    result = extract_otel_context(headers)
    assert result is None


def test_extract_otel_context_does_not_raise_without_otel():
    """Must not raise even when opentelemetry packages are absent."""
    from websockets.datastructures import Headers
    import unmute_mlx_bridge.observability as obs
    headers = Headers()
    headers["traceparent"] = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    # Either returns a context object (if otel installed) or None (if not).
    result = extract_otel_context(headers)
    assert result is None or hasattr(result, "__class__")
