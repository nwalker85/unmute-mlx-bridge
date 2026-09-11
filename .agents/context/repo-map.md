# Repo Map — unmute-mlx-bridge

## Purpose

Public, Apache-2.0 Apple Silicon MLX servers implementing Kyutai Unmute's STT
and TTS WebSocket contracts. The bridge does not fork Unmute, own conversation
state, or embed consumer-specific logic.

## Entry Points

- `unmute-mlx-stt` — STT server on port 8090
- `unmute-mlx-tts` — TTS server on port 8089
- `/api/asr-streaming` and `/api/tts_streaming` — protocol endpoints
- `/healthz`, `/readyz`, `/metrics` — operational endpoints on both processes
- `tests/` — portable protocol and lifecycle conformance
- `tests/hardware/` — opt-in real-model Apple Silicon proof

## Important Paths

```text
AGENTS.md                              repository authority and guardrails
PROTOCOL.md                            wire-compatibility oracle
package-surface.json                   machine-readable release posture
pyproject.toml / uv.lock               Python package and locked dependencies
src/unmute_mlx_bridge/protocol/        message models and framing
src/unmute_mlx_bridge/{stt,tts}/       MLX engines and WebSocket servers
src/unmute_mlx_bridge/observability.py health, readiness, metrics, auth
src/unmute_mlx_bridge/config.py        environment-driven configuration
tests/                                 portable conformance suite
tests/hardware/                        real weights and MLX, explicit opt-in
docs/design/architecture.md            design, boundaries, and proof criteria
docs/architecture/decisions/           ADRs
docs/runbooks/                         operating procedures
.github/workflows/ci.yml               public CI and manual hardware proof
.github/agents/                         GitHub Copilot custom agents
```

## Validation

```bash
uv sync --locked
git diff --check
uv run --locked pytest -q

# Explicit Apple Silicon proof only:
# uv run --locked pytest -m hardware -q -s
```

## Known Footguns

- Portable CI must not import MLX or download model weights.
- The upstream compatibility SHA, protocol fixtures, tests, and documentation
  change together.
- Each server admits one session at a time, matching the pinned upstream
  behavior.
- Non-loopback binding requires a configured authentication token.
- Health and portable conformance do not prove real-model or downstream
  end-to-end behavior.
- Public comments, fixtures, CI logs, issues, and discussions share the same
  privacy boundary as committed source.
