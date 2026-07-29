# Repo Map — unmute-mlx-bridge

## Purpose

`unmute-mlx-bridge` provides persistent Apple Silicon MLX STT and TTS services
that implement the Kyutai Unmute model-server WebSocket contract. It is a
community Python package (Apache-2.0), currently in private incubation. It does
not fork Unmute, own conversation state, or embed consumer-specific logic.

## Entry Points

- Application: `unmute-mlx-stt` (port 8090, planned) and `unmute-mlx-tts`
  (port 8089, planned) — not yet implemented in scaffold.
- API: `/api/asr-streaming` (STT WebSocket), `/api/tts_streaming` (TTS
  WebSocket), `/healthz`, `/readyz`, `/metrics` — planned, not yet implemented.
- Worker: none in scaffold.
- CLI: `unmute-mlx-bridge` entry point (scaffold stub only).
- Tests: `tests/` — portable smoke test; hardware tests behind `pytest.mark.hardware`.

## Important Directories

```text
.
├── AGENTS.md                     repo authority and guardrails
├── package-surface.json          machine-readable release contract
├── pyproject.toml                Python package definition (hatchling, uv)
├── src/
│   └── unmute_mlx_bridge/        Python package (scaffold stub)
├── tests/
│   └── test_smoke.py             portable smoke test
├── docs/
│   ├── repo-intake.md            lifecycle decisions
│   ├── architecture/decisions/   ADRs
│   ├── runbooks/                 operational procedures
│   └── superpowers/specs/        design spec (do not modify)
├── .forgejo/
│   └── workflows/ci.yml          active private-incubation portable CI
├── .github/
│   └── workflows/ci.yml          dormant publication-cutover CI
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

- CI: Forgejo Actions (`.forgejo/workflows/ci.yml`), repo-owned K3s label
  `unmute-mlx-bridge`
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
