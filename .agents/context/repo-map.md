# Repo Map — unmute-mlx-bridge

## Purpose

`unmute-mlx-bridge` provides persistent Apple Silicon MLX STT and TTS services
that implement the Kyutai Unmute model-server WebSocket contract. It is a
community Python package (Apache-2.0), currently in private incubation. It does
not fork Unmute, own conversation state, or embed consumer-specific logic.

## Entry Points

- Application: `unmute-mlx-stt` (port 8090) and `unmute-mlx-tts` (port 8089) —
  implemented, real MLX inference.
- API: `/api/asr-streaming` (STT WebSocket), `/api/tts_streaming` (TTS
  WebSocket), `/healthz`, `/readyz`, `/metrics` on each process — implemented.
- Worker: none.
- CLI: `unmute-mlx-bridge` entry point prints usage; use `unmute-mlx-stt` /
  `unmute-mlx-tts` to run a server.
- Tests: `tests/` — portable protocol + full-session conformance suite (default
  `pytest` run, no model weights); `tests/hardware/` — real-MLX suite behind
  `pytest.mark.hardware`.

## Important Directories

```text
.
├── AGENTS.md                     repo authority and guardrails
├── PROTOCOL.md                   wire-format compatibility writeup
├── package-surface.json          machine-readable release contract
├── pyproject.toml                Python package definition (hatchling, uv)
├── src/
│   └── unmute_mlx_bridge/
│       ├── protocol/              msgpack message models + framing (stt.py, tts.py, wire.py)
│       ├── stt/                   engine.py (MLX inference), server.py (WS server)
│       ├── tts/                   engine.py (MLX inference), server.py (WS server)
│       ├── observability.py       /healthz, /readyz, /metrics, auth
│       └── config.py              environment-driven server configuration
├── tests/
│   ├── test_protocol_{stt,tts}.py         wire-shape tests
│   ├── test_{stt,tts}_server_conformance.py  full session state machine, fake engine
│   ├── test_smoke.py                      portable smoke test
│   └── hardware/                          real MLX inference, opt-in
├── docs/
│   ├── repo-intake.md            lifecycle decisions
│   ├── architecture/decisions/   ADRs
│   ├── runbooks/                 operational procedures
│   └── superpowers/specs/        design spec (do not modify)
├── .github/
│   └── workflows/ci.yml          GitHub Actions portable CI
└── flake.nix                     Nix dev shell (python312 + uv)
```

## Build And Test Commands

```bash
# Setup (uv.lock is committed)
uv sync --locked

# Portable tests (runs on Linux CI and local)
uv run --locked pytest -q

# Hardware tests — Apple Silicon only, opt-in, never on portable CI
# uv run --locked pytest -m hardware -q

# Whitespace check (matches CI)
git diff --check
```

## Deploy And Runtime

- CI: GitHub Actions (`.github/workflows/ci.yml`), `ubuntu-latest`
- Artifact: none until publication approved
- Runtime: local Apple Silicon for canary runs
- Logs: none (canary artifacts are stored outside CI)
- Health check: `/healthz` and `/readyz` per service process (planned)

## Sensitive Data Boundaries

Do not commit secrets, raw evidence, customer data, private exports, generated
evidence bundles, `.env` files, cookie values, private hostnames, infrastructure
topology, or model weights.

Hardware test results and benchmark reports are artifacts only. They are stored
in the private repository and require sanitization before any publication.

## Known Footguns

- Model weights are not in the repo and must not be downloaded during portable
  CI. The `pytest -m hardware` gate enforces this.
- The pinned upstream Unmute compatibility baseline (commit SHAs in the design
  spec) must be updated together with contract test fixtures when upstream changes.
- Binding STT/TTS ports to a non-loopback interface requires explicit static
  token configuration. Loopback-only is the safe default.
