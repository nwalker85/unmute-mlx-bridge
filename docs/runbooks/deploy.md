# Runbook: Running the bridge as a long-lived service

Operational reference for keeping `unmute-mlx-stt` and `unmute-mlx-tts` running
continuously on an Apple Silicon Mac, instead of in a foreground terminal. This
is generic guidance — it names no specific host. See `README.md`'s
[Configuration](../../README.md#configuration) section for the authoritative
environment-variable list; this runbook only calls out the subset that matters
for a service deployment.

## Service shape

Both processes are single long-running Python processes with no daemonization
of their own — they log to stdout/stderr and expect a process supervisor to
keep them running, restart them on crash, and manage their logs. Pick one:

### Option A — launchd (macOS-native)

A per-user `LaunchAgent` plist per process. Minimal template
(`~/Library/LaunchAgents/net.example.unmute-mlx-tts.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>net.example.unmute-mlx-tts</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/uv</string>
    <string>run</string>
    <string>--locked</string>
    <string>unmute-mlx-tts</string>
  </array>
  <key>WorkingDirectory</key><string>/path/to/unmute-mlx-bridge</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>TTS_HOST</key><string>127.0.0.1</string>
    <key>TTS_PORT</key><string>8089</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/path/to/logs/unmute-mlx-tts.log</string>
  <key>StandardErrorPath</key><string>/path/to/logs/unmute-mlx-tts.err.log</string>
</dict>
</plist>
```

Load/reload with `launchctl bootstrap gui/$(id -u) <plist>` (or `load -w` on
older macOS), check status with `launchctl list | grep unmute-mlx`, and tail
`StandardOutPath`/`StandardErrorPath` for logs. `KeepAlive` restarts the
process on crash; it does not distinguish a crash from a deliberate exit, so
stop the agent (`launchctl bootout`) before intentionally killing the process.

### Option B — PM2 (cross-platform, if already managing other services)

PM2 doesn't require Node — it can supervise any command via
`interpreter: "none"`. Example `ecosystem.config.js`:

```js
module.exports = {
  apps: [
    {
      name: "unmute-mlx-stt",
      cwd: "/path/to/unmute-mlx-bridge",
      script: "uv",
      args: "run --locked unmute-mlx-stt",
      interpreter: "none",
      env: { STT_HOST: "127.0.0.1", STT_PORT: "8090" },
    },
    {
      name: "unmute-mlx-tts",
      cwd: "/path/to/unmute-mlx-bridge",
      script: "uv",
      args: "run --locked unmute-mlx-tts",
      interpreter: "none",
      env: { TTS_HOST: "127.0.0.1", TTS_PORT: "8089" },
    },
  ],
};
```

Start with `pm2 start ecosystem.config.js`, inspect with `pm2 list` /
`pm2 env <id>`, tail logs with `pm2 logs <name>`, and persist across reboots
with `pm2 save` + `pm2 startup` (macOS: a launchd agent under the hood).

Either option is one process per service — there is no multi-worker or
cluster mode. A restart drops the in-flight session (see [Single active
session per process] below); there is nothing to drain.

## Environment

Every setting is an environment variable — no config file, no CLI flags. See
`README.md`'s [Configuration](../../README.md#configuration) table for the
full list, defaults, and validation behavior; the ones that matter most for a
service deployment:

| Concern | Variables |
|---|---|
| Bind address | `STT_HOST`/`STT_PORT` (default `127.0.0.1:8090`), `TTS_HOST`/`TTS_PORT` (default `127.0.0.1:8089`) |
| Auth (required once off loopback) | `STT_AUTHORIZED_IDS`, `TTS_AUTHORIZED_IDS` — comma-separated tokens; empty accepts the loopback-dev default `public_token` |
| Model source | `STT_HF_REPO`, `TTS_HF_REPO`, `TTS_VOICE_REPO` |
| Performance profile | `TTS_QUANTIZE_BITS` (`8` is the validated profile), `TTS_N_Q` (do not lower), `TTS_DELIVERY_MODE` (`streaming` or `buffered_turn`) |
| Transcript logging (off by default; only enable with consent) | `STT_LOG_TRANSCRIPTS`, `TTS_LOG_TRANSCRIPTS` |

Do not bind either service to a non-loopback interface without setting
`*_AUTHORIZED_IDS` to something other than the default.

## Ports

- STT: `/api/asr-streaming` WebSocket, plus `/healthz`, `/readyz`, `/metrics`,
  `/api/build_info` HTTP, all on `STT_PORT` (default `8090`).
- TTS: `/api/tts_streaming` WebSocket, plus the same four HTTP paths, on
  `TTS_PORT` (default `8089`).

These are the exact ports stock Unmute's backend looks for
(`KYUTAI_STT_URL`/`KYUTAI_TTS_URL`) with zero configuration on its side.

## Health, readiness, and metrics

- `GET /healthz` — liveness only. `200 ok` once the process is up; does not
  reflect model-load state.
- `GET /readyz` — `200` once the model is loaded **and** no session currently
  holds the single admission slot; `503` otherwise. The body is a snapshot
  (`model_loaded`, `model_load_error`, `session_active`, `uptime_s`) —
  useful for scripted polling, not just a status code check. Poll this after
  starting the service instead of guessing when weights have finished
  downloading (first start pulls several GB from Hugging Face).
- `GET /metrics` — Prometheus text format: request/session counters,
  rejection counters by reason, generation-failure counters by reason. See
  `docs/observability.md` for the full metric catalogue and structured
  logging fields.

## Load-failure behavior

If model load fails (bad `TTS_CFG_COEF`, a Hugging Face fetch failure, corrupt
weights, etc.), **the process does not exit.** It keeps serving `/healthz`,
`/readyz`, and `/metrics` so a supervisor or operator can observe the failure
through `/readyz`'s `model_load_error` field rather than the process
disappearing — this project assumes no orchestrator will restart it based on
liveness alone. A client that connects while load has failed gets a protocol
`Error` (`"model failed to load"`) and a clean close, distinct from the
`"model still loading"` message sent while the load attempt is still in
flight. See README.md's "If model load fails" section for the full detail;
this is why `/readyz`'s body, not just its status code, is worth checking in
an alert.

## Single active session per process

Each process admits exactly one WebSocket session at a time, matching stock
`moshi-server`. A second concurrent connection is rejected with an `Error`
(`"no free channels"`) and closed, not queued. There is no multi-tenant
sharing within one process — run a second instance on different ports if you
need concurrent sessions.

## Logs

Structured logging (`TTS_LOG_FORMAT`/equivalent, see
`docs/observability.md`) goes to stdout/stderr; there is no file-logging
built in, so the supervisor (launchd's `StandardOutPath`, PM2's log files)
owns log rotation and retention. Transcript content is redacted by default
(`*_LOG_TRANSCRIPTS=false`) — do not flip that on for a log sink you don't
control or without the speaker's consent.
