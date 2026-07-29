# Changelog

All notable changes to this project are documented here.

This project follows Semantic Versioning once it exposes a public contract,
deployable artifact, API, SDK, schema, or specification.

Pre-1.0 releases use `0.y.z` under SemVer v2. Breaking changes are documented
even under `0.y.z`. See `docs/architecture/decisions/` for decisions affecting
compatibility or version labels.

## [Unreleased]

### Added

- Initial repository scaffold: governance files, package seed, and portable CI.
- `docs/repo-intake.md` with lifecycle decisions.
- `package-surface.json` recording GitHub-primary exception posture.
- `.github/workflows/ci.yml` using GitHub Actions for portable CI.
- `PROTOCOL.md`: annotated wire-format writeup for `/api/asr-streaming` and
  `/api/tts_streaming`, verified against `kyutai-labs/unmute` and
  `kyutai-labs/moshi` source at the commits pinned in the design spec.
- `src/unmute_mlx_bridge/protocol/`: pydantic msgpack message models for both
  endpoints, matching real `moshi-server`'s `OutMsg`/`InMsg` shapes field-for-field.
- `src/unmute_mlx_bridge/stt/`: real MLX STT inference (`engine.py`, backed by
  `moshi-mlx` + `kyutai/stt-1b-en_fr-mlx`) and a `moshi-server`-compatible
  `/api/asr-streaming` WebSocket server (`server.py`), including the
  Word/EndWord segmentation state machine ported from `moshi-core/src/asr.rs`.
- `src/unmute_mlx_bridge/tts/`: real MLX TTS inference (`engine.py`, backed by
  `moshi-mlx` + `kyutai/tts-1.6b-en_fr`) and a `moshi-server`-compatible
  `/api/tts_streaming` WebSocket server (`server.py`), including word-timing
  derivation from the TTS state machine's transcript.
- `src/unmute_mlx_bridge/observability.py`: shared `/healthz`, `/readyz`,
  `/metrics`, and pre-upgrade auth (`kyutai-api-key` header / `auth_id` query
  param, matching real `moshi-server`).
- `src/unmute_mlx_bridge/config.py`: environment-driven server configuration.
- Portable protocol + full-session conformance test suite
  (`tests/test_protocol_{stt,tts}.py`, `tests/test_{stt,tts}_server_conformance.py`)
  exercising the real server classes against a deterministic fake engine over a
  real WebSocket connection — no model weights required.
- Opt-in hardware test suite (`tests/hardware/`, `pytest -m hardware`): real
  MLX inference on Apple Silicon, real audio fixtures synthesized with macOS
  `say`, transcription/synthesis correctness and state-leak checks.

### Fixed

- `moshi-mlx`/`mlx` are marker-gated to `sys_platform == 'darwin' and
  platform_machine == 'arm64'` in `pyproject.toml`, and every runtime import of
  them in `stt/engine.py` / `tts/engine.py` is deferred into the function that
  actually needs it (module-level imports removed). Without this, adding real
  MLX inference would have broken `uv sync --locked` on the existing portable
  `ubuntu-latest` CI lane entirely — `mlx`'s Linux wheel still requires the
  macOS-only `mlx-metal` backend package, so it cannot resolve on Linux at all.
  This matches the design spec's own requirement that MLX imports stay behind
  explicit adapter construction so the portable suite runs on Linux CI.

### Known limitations

- Single active session per process (documented, see `PROTOCOL.md`).
- Ogg/Opus framing and TTS custom voice-embedding cloning are not implemented
  (documented non-goals; Unmute's own backend client never uses either).
- Not yet run against the actual Unmute orchestrator process or a full
  microphone-to-speaker canary — see README §What's proven / what's not.

## [0.1.0] — planned

Initial scaffold posture. Not yet released.

### Added

- Python package seed under `src/unmute_mlx_bridge/`.
- Smoke test confirming `main` is callable.
- Development shell via `flake.nix`.
