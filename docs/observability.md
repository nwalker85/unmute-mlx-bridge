# Observability Reference

This document covers the observability configuration for `unmute-mlx-bridge` (bridge
servers) and the Unmute client instrumentation layer.

---

## Transcript Logging (Private Canary Only)

Full STT word sequences and TTS text are logged in structured events as the `text`
and `transcript` fields. This is an intentional choice for the **private canary**
deployment where Nate explicitly requested logs+Langfuse for transcript capture.

**Transcript logging is disabled by default** (safe for general use). It is
controlled by environment variables that must be explicitly set to enable it.

| Component | Env var | Default | Enable for canary |
|-----------|---------|---------|-------------------|
| Bridge STT server | `STT_LOG_TRANSCRIPTS` | `false` | `STT_LOG_TRANSCRIPTS=true` |
| Bridge TTS server | `TTS_LOG_TRANSCRIPTS` | `false` | `TTS_LOG_TRANSCRIPTS=true` |
| Unmute client (STT + TTS) | `UNMUTE_LOG_TRANSCRIPTS` | `false` | `UNMUTE_LOG_TRANSCRIPTS=true` |

Accepted truthy values: `1`, `true`, `yes` (case-insensitive).

**Never add `text` or `transcript` values to Prometheus labels.** Labels are not
private and leak into any external monitoring system that scrapes `/metrics`.

Gates applied when `log_transcripts` is disabled:
- STT bridge: `stt_word` event omits the `text` field
- TTS bridge: `tts_text_buffered` and `tts_text_streaming` events omit the `text` field
- Unmute STT client: `stt_client_word` omits `text`; `stt_session_end` omits `transcript`
- Unmute TTS client: `tts_client_word` omits `text`; `tts_session_end` omits `transcript`

### Transcript gate consistency: bridge (startup-sampled) vs Unmute client (runtime)

The bridge `STT_LOG_TRANSCRIPTS` and `TTS_LOG_TRANSCRIPTS` flags are **sampled
once at process startup** — `SttConfig.from_env()` and `TtsConfig.from_env()` are
called when the process boots and the result is frozen for the process lifetime.
Changing the environment variable after boot has no effect; a process restart is
required to toggle transcript logging on the bridge.

The Unmute client `UNMUTE_LOG_TRANSCRIPTS` flag is **evaluated at each session
start** via a runtime helper (`_should_log_transcripts()`). Changing the variable
while the Unmute process is running takes effect on the next session without a
restart.

This inconsistency is intentional: the bridge is a long-running server process
where frozen configuration is simpler to reason about; the Unmute client is a
shorter-lived interactive process where runtime reconfiguration is useful for
experimentation.

### Structured log sanitization and transcript fidelity

The bridge JSON log formatter (`_JsonFormatter`) applies credential scrubbing to
all fields **except top-level `text` and `transcript` fields**, which bypass
all scrubbers:

- **Top-level `text`/`transcript`** (the exact fields emitted by `stt_word`,
  `tts_text_buffered`, and `tts_text_streaming` events): values pass through
  unchanged, even if they contain credential-shaped strings. This is the
  transcript fidelity guarantee.
- **Nested dicts** that happen to have a `text` or `transcript` key are sanitized
  normally; the bypass applies only to the top-level record fields.
- **Formatted log messages** (`msg` field): both credential-bearing URLs
  (userinfo and sensitive query parameters) and bearer/basic-auth patterns are
  scrubbed. Ordinary URLs without credentials are left intact.
- **Exception text**: same sanitization as messages.

---

## OpenTelemetry (OTEL) Endpoint

Both bridge processes (`unmute-mlx-stt` and `unmute-mlx-tts`) call `init_otel()`
at startup. No traces are exported unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy.example.internal:4318
```

**Endpoint topology is left to deployment configuration.** This bridge does not
assume Alloy/Tempo vs Langfuse — both accept OTLP. The endpoint URL is
sanitized (userinfo/credentials stripped) before appearing in any log line.

Safe no-op defaults:
- If `OTEL_EXPORTER_OTLP_ENDPOINT` is unset → no-op tracer, no export, no failure.
- If `opentelemetry-sdk` / OTLP exporter packages are absent → same no-op behavior.
- If the endpoint is unreachable → `init_otel()` logs the error and continues.

For Langfuse transcript capture: Langfuse accepts OTLP traces via its `/api/public/otel`
endpoint. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to the Langfuse OTLP URL and the span
attributes (`session.id`, `session.voice`, etc.) appear in Langfuse as trace
metadata. Full transcript text is in structured logs; use a log-forwarding pipeline
(e.g. Alloy → Loki) to ingest those into Langfuse or a private log store.

---

## Cross-Repo Correlation Headers

The Unmute client sends its `session_id` as the `x-unmute-session-id` WebSocket
upgrade header and its `conversation_id` as the `x-unmute-conversation-id` header
when connecting to both the STT and TTS bridge servers.

The bridge servers extract both headers and include them in all structured log
events and OTEL span attributes for that session:

```
Unmute client (session_id + conversation_id generated in __init__)
  → WebSocket headers:
      x-unmute-session-id: <hex32>
      x-unmute-conversation-id: <hex32>   (only when conversation_id is set)
  → Bridge: session_id + conversation_id in log events, span attributes
      session.id, conversation.id
```

Header format: 32-character lowercase hex UUID4 (`[0-9a-f]{32}`). Any other format
is rejected (the field is treated as absent); the bridge generates a fresh session ID
if `x-unmute-session-id` is invalid, and `conversation_id` is `None` if the
`x-unmute-conversation-id` header is absent or invalid.

The `kyutai-api-key` auth header is never logged. Both `x-unmute-session-id` and
`x-unmute-conversation-id` headers contain no credentials and are safe to log.

### W3C Trace Context Propagation

W3C `traceparent` and `tracestate` headers are extracted **explicitly** from the
WebSocket upgrade request headers on the bridge handshake. The bridge calls
`extract_otel_context(connection.request.headers)` and passes the resulting context
when starting each session span.

**The OTEL SDK does NOT automatically propagate context from WebSocket upgrade
headers.** Automatic context propagation only applies to HTTP/gRPC libraries that
have been explicitly instrumented. For WebSocket connections, extraction must be
done manually — which is what `extract_otel_context` does.

When `opentelemetry-api` is not installed or no `traceparent` header is present,
`extract_otel_context` returns `None` and spans are started without a parent context.

### Conversation-Level Correlation

**`UnmuteHandler` generates a shared `conversation_id` at construction time** and
wires it to all `SpeechToText` and `TextToSpeech` instances it creates. This means
all STT sessions and TTS sessions belonging to the same conversation share the same
`conversation_id` in their log events.

```
UnmuteHandler.__init__
  → self._conversation_id = uuid.uuid4().hex
  → start_up_stt → SpeechToText(conversation_id=handler._conversation_id)
  → start_up_tts (per turn) → TextToSpeech(conversation_id=handler._conversation_id)
```

Each turn creates a new TTS session with a new `session_id`, but the same
`conversation_id`. The STT session lives for the connection duration and also
carries the same `conversation_id`.

**Scope:** `conversation_id` correlates all sessions within one `UnmuteHandler`
instance (one FastRTC connection). Across separate connections, handlers generate
independent `conversation_id` values.

---

## STT Queue Metrics

| Metric | Description |
|--------|-------------|
| `bridge_stt_inference_frames_active` | Frames in the current active `push_audio` call. `0` = idle, `N` = inference running. **Not** the websockets receive-queue depth. |
| `bridge_stt_recv_queue_bound` | Configured `max_queue` for the websockets receive queue (set at startup from `STT_MAX_RECV_QUEUE`, default 1024). This is the ceiling for the internal websockets backlog; not directly observable as a live depth. |
| `bridge_stt_oversized_frames_total` | Audio frames rejected for exceeding `STT_MAX_INPUT_FRAME_SAMPLES`. |
| `bridge_inference_step_seconds` | Per-frame inference wall time (push_audio batch / frame count). |
| `bridge_inference_step_rtf` | Real-time factor per batch (inference_s / audio_s). |

The websockets library's internal receive queue is bounded by `STT_MAX_RECV_QUEUE`
(passed as `max_queue` to `websockets.serve`). Its live depth is not exposed to
application code. `bridge_stt_inference_frames_active` accurately describes what IS
observable: the number of frames being processed in the current call.

---

## TTS Audio Generation Lateness Events (Unmute Client)

The Unmute client detects when generated audio is released from the `RealtimeQueue`
significantly later than its scheduled playout time. This measures **generation
pipeline lateness** — specifically, the lag between when an audio item was scheduled
to be released and when it actually becomes available. It does **not** confirm
hardware or OS playback underflow.

| Event | Condition | Level |
|-------|-----------|-------|
| `tts_queue_high_water` | `queue_depth >= 30` — generation accumulating faster than real-time | WARNING |
| `tts_queue_recovered` | Returned below 5 after high-water | INFO |
| `tts_generation_late` | Released audio item is `> _TTS_UNDERFLOW_LATE_S` (1.0 s) past its scheduled time | WARNING |
| `tts_generation_caught_up` | Subsequent items are within threshold after lateness | INFO |

**Lateness is schedule-relative, not depth-based.** After `get_nowait()` releases
an audio item `(scheduled_t, message)`, lateness is computed as:

```
lateness = (queue.get_time() - queue.start_time) - scheduled_t
```

A lateness > `_TTS_UNDERFLOW_LATE_S` (1.0 s) means the generation pipeline
delivered audio more than 1 second past its intended playout time. This threshold
is >> `AUDIO_BUFFER_SEC` (~0.32 s) to avoid false positives from normal jitter.

Thresholds (`_TTS_QUEUE_HIGH_WATER = 30`, `_TTS_QUEUE_LOW_WATER = 5`,
`_TTS_UNDERFLOW_LATE_S = 1.0`) are module-level constants in
`unmute/tts/text_to_speech.py`.

---

## NTP Clock Verification

Both bridge servers call `log_clock_metadata()` at startup. This records:

- `utc_startup`: UTC wall-clock time (ISO 8601) for cross-service event correlation.
- `monotonic_start_ref`: monotonic epoch for within-process duration calculations.
- `ntp_hostname_dns_resolved`: whether the configured NTP source resolved in DNS
  at startup.
- `ntp_sync_note`: explicit reminder that DNS resolution ≠ NTP synchronization.

**DNS resolution is not NTP synchronization.** A `true` value for
`ntp_hostname_dns_resolved` only means the hostname was reachable in DNS; it does
not mean the system clock is disciplined to that source. Synchronization status
must be verified externally:

```bash
# On the host running the bridge container:
chronyc tracking
# or
timedatectl show
```

The expected NTP source defaults to the public `pool.ntp.org` pool and is
configurable via the `NTP_SOURCE` environment variable — set it to your own
stratum-1 or GPS-disciplined server if you have one. This is recorded for
operator reference; the bridge never modifies system NTP configuration.
