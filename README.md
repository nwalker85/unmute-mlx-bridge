# unmute-mlx-bridge

Apple Silicon MLX streaming STT/TTS servers compatible with the Kyutai Unmute
model protocol.

> **Status:** Private incubation — real MLX inference, protocol-conformance
> tested, and verified against the pinned stock Unmute orchestrator. Not yet
> run in a full microphone-to-speaker canary; see **What's proven / What's
> not**, below.
> Repository visibility change requires explicit publication approval.
>
> **Unofficial community project.** Not an official Kyutai release. Model weights
> retain their CC-BY-4.0 license. Source code is Apache-2.0.

## What this is

Two long-lived processes:

- `unmute-mlx-stt` — STT model server on port 8090, implementing the Kyutai
  Unmute `/api/asr-streaming` WebSocket contract.
- `unmute-mlx-tts` — TTS model server on port 8089, implementing the Kyutai
  Unmute `/api/tts_streaming` WebSocket contract.

Both use `moshi-mlx` for real MLX inference on Apple Silicon. See
[`PROTOCOL.md`](PROTOCOL.md) for the annotated wire-format writeup (message
schema, framing, query params, word-timing derivation) verified directly
against `kyutai-labs/unmute` and `kyutai-labs/moshi` source, and
[`docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md`](docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md)
for the full architecture and delivery plan.

## What's proven / what's not

This is honest canary-stage status, not a claim of production readiness.

**Proven, with evidence (this pass):**

- The wire protocol implementation (`src/unmute_mlx_bridge/protocol/`) matches
  real `moshi-server`'s message shapes field-for-field, verified by reading its
  Rust source directly (not documentation) — see `PROTOCOL.md`.
- Both servers implement the full session lifecycle against that protocol over
  a real WebSocket connection: `Ready` on connect, single-session admission
  (second connection rejected with `Error`+close), malformed-frame handling
  (`Error`, connection stays open), pre-upgrade `401` for bad auth, and
  disconnect releasing the session slot. Proven by
  `tests/test_stt_server_conformance.py` / `tests/test_tts_server_conformance.py`
  running the real server classes against a deterministic fake engine over a
  real `websockets` connection (portable, no model weights, runs on any
  platform).
- Real MLX STT inference: real `kyutai/stt-1b-en_fr-candle` weights loaded into
  `moshi-mlx`, a real
  macOS-`say`-synthesized WAV fixture with a known transcript, transcribed
  end-to-end through `SttSession` — see "Hardware evidence", below, for the
  actual run's numbers.
- Real MLX TTS inference: real `kyutai/tts-1.6b-en_fr` weights, real synthesis
  of a known sentence, non-silent audio out with word-level timing events
  derived from the model's own state machine — see "Hardware evidence".
- The pinned stock Unmute backend connects to both servers without source
  changes. A four-response stock load-test conversation completed with three
  user STT/semantic-VAD turns, streamed TTS on every response, no errors, and
  `OK fraction: 1.0`.

**Not yet proven — the honest gaps:**

- No end-to-end microphone → STT → LLM → TTS → speaker canary. This project is
  scoped to the two model servers only (see design doc §Non-Goals);
  integrating with Freyr/Gjallarhorn is explicitly out of scope for this pass.
- No sustained/long-duration real-time-factor measurement, no concurrent-load
  testing, no memory-growth-over-many-sessions measurement.
- No barge-in / cancellation-under-load testing beyond a clean WebSocket close.
- Voice quality (MOS) is unmeasured.

## Quick start

```bash
# Requires uv and Python 3.12; uv.lock is committed
uv sync --locked

# Portable tests: protocol shapes + full server conformance against a fake
# engine. No model weights downloaded, runs on any platform.
uv run --locked pytest -q

# Hardware tests: real MLX inference, real ~2-4GB model downloads from
# Hugging Face on first run, Apple Silicon only.
uv run --locked pytest -m hardware -q -s
```

### Running a server for real

```bash
# Terminal 1 — STT on ws://127.0.0.1:8090/api/asr-streaming
uv run unmute-mlx-stt

# Terminal 2 — TTS on ws://127.0.0.1:8089/api/tts_streaming
uv run unmute-mlx-tts
```

Both processes:

- Download weights from Hugging Face on first start (standard `huggingface_hub`
  cache; no weights are vendored in this repo).
- Expose `GET /healthz` (process liveness), `GET /readyz` (model loaded and a
  session slot free — `503` while loading or busy), and `GET /metrics`
  (Prometheus) on the same port as the WebSocket endpoint.
- Accept the `kyutai-api-key: public_token` header (or `?auth_id=public_token`
  query param) by default — override with `STT_AUTHORIZED_IDS` /
  `TTS_AUTHORIZED_IDS` (comma-separated; set empty to disable auth entirely,
  loopback-dev default).

TTS sends `Ready` as soon as the loaded model admits the channel. A requested
voice that is not already cached is fetched afterward, so the first audio for
that voice can take longer even though stock Unmute's 500 ms startup handshake
is satisfied.

Key environment variables (see `src/unmute_mlx_bridge/config.py` for the full
list and defaults):

| Var | Server | Meaning |
|---|---|---|
| `STT_HOST`, `STT_PORT` | STT | Bind address, default `127.0.0.1:8090` |
| `STT_HF_REPO` | STT | Default `kyutai/stt-1b-en_fr-candle` |
| `STT_QUANTIZE_BITS` | STT | Post-load quantization, e.g. `4` or `8` |
| `STT_MAX_STEPS` | STT | Per-session step budget before a reconnect is required |
| `TTS_HOST`, `TTS_PORT` | TTS | Bind address, default `127.0.0.1:8089` |
| `TTS_HF_REPO` | TTS | Default `kyutai/tts-1.6b-en_fr` |
| `TTS_DEFAULT_VOICE` | TTS | Voice file used when the client doesn't pass `?voice=` |
| `TTS_N_Q` | TTS | Generated audio codebooks, default `24` to match stock Unmute; lower values trade fidelity for speed |

### Talking to a running server manually

```bash
# STT: send one msgpack Audio frame, print whatever comes back
uv run python - <<'EOF'
import asyncio, msgpack, websockets

async def main():
    async with websockets.connect(
        "ws://127.0.0.1:8090/api/asr-streaming",
        additional_headers={"kyutai-api-key": "public_token"},
    ) as ws:
        print(msgpack.unpackb(await ws.recv(decode=False)))  # {'type': 'Ready'}
        await ws.send(msgpack.packb({"type": "Audio", "pcm": [0.0] * 1920}))
        print(msgpack.unpackb(await ws.recv(decode=False)))

asyncio.run(main())
EOF
```

## Hardware evidence

Real, opt-in tests that download real weights and run real inference on Apple
Silicon:

- `tests/hardware/test_stt_hardware.py` — synthesizes a known sentence with
  macOS `say`, transcribes it with real `kyutai/stt-1b-en_fr-candle` weights
  through `moshi-mlx`, and asserts the expected words appear with monotonically
  increasing timestamps. It also proves the semantic-VAD head crosses stock
  Unmute's pause threshold on trailing silence and runs two sequential sessions
  on the same process to confirm cache/state does not leak between connections.
- `tests/hardware/test_tts_hardware.py` — synthesizes a known sentence with
  real `kyutai/tts-1.6b-en_fr` inference via `TtsSession`, and asserts the
  output audio is non-silent (RMS threshold) with plausible duration and
  monotonically ordered word-timing events. Also writes the synthesized WAV to
  a temp path for manual listening, and repeats the state-leak check.

Run them yourself and see real numbers:

```bash
uv run --locked pytest -m hardware -q -s
```

Both print `real_time_factor` (audio seconds ÷ wall-clock seconds) so you can
see whether MLX inference on your machine is keeping up with real time — the
first hard question ADR-0007 asks before Phase 2 is trusted. See the PR
description for the numbers observed on the reference host used to build this.

## Platform

- macOS 14 or newer on Apple Silicon (`arm64`).
- Python `>=3.12,<3.13`, package manager `uv`.
- Portable CI targets the repo-owned Forgejo K3s label
  `unmute-mlx-bridge` without model weights; it exercises the protocol layer
  and both servers' full session state machine against a fake engine, not real
  MLX.
- Hardware tests (real models, Apple Silicon) are opt-in: `pytest -m hardware`.

## Repository Map

- `PROTOCOL.md` — the wire-format compatibility writeup; start here before
  touching `src/unmute_mlx_bridge/protocol/`.
- `src/unmute_mlx_bridge/protocol/` — msgpack message models (`stt.py`,
  `tts.py`) and framing (`wire.py`).
- `src/unmute_mlx_bridge/stt/`, `src/unmute_mlx_bridge/tts/` — `engine.py` (MLX
  inference + protocol state machine) and `server.py` (WebSocket server, health
  probes, auth, session admission) for each service.
- `src/unmute_mlx_bridge/observability.py` — shared `/healthz`, `/readyz`,
  `/metrics`, and auth handling.
- `src/unmute_mlx_bridge/config.py` — environment-driven server configuration.
- `tests/` — portable protocol + conformance suite (default `pytest` run);
  `tests/hardware/` — opt-in real-MLX suite (`pytest -m hardware`).
- `AGENTS.md` — repo authority, guardrails, and privacy requirements.
- `docs/repo-intake.md` — lifecycle decisions (namespace, SDK/API, CI, observability).
- `.agents/context/repo-map.md` — entry points and build commands.
- `.agents/checklists/pr.md` — before opening a PR.
- `.agents/checklists/release.md` — before claiming a release is live.
- `CHANGELOG.md` — before tagging a release.
- `.forgejo/workflows/ci.yml` — active private-incubation portable CI.
- `.github/workflows/ci.yml` — dormant publication-cutover CI.
- `docs/architecture/decisions/` — before making compatibility decisions.
- `docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md` — full design spec.

## Release And Package Posture

The `package-surface.json` records the release gates. Before any release:

- `registry_target` — `none` until explicit publication approval
- `publication_status` — `private_incubation`
- `semver_policy` — `semver-v2` starting at `0.1.0`
- `ci_lane` — `forgejo-actions`
- `runner_label` — `unmute-mlx-bridge`
- `nix_flake` — `flake-check-required-before-publish`

## Attribution

This project is compatible with and depends on:

- [Kyutai Unmute](https://github.com/kyutai-labs/unmute)
- [Kyutai Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)

Model weights (`kyutai/stt-1b-en_fr-candle`, `kyutai/tts-1.6b-en_fr`) are not
redistributed. They retain their CC-BY-4.0 license. Refer to the upstream Kyutai
repositories for model licensing terms.

## License

Apache-2.0. See [LICENSE](LICENSE).
