# unmute-mlx-bridge

Apple Silicon MLX streaming STT/TTS servers compatible with the Kyutai Unmute
model protocol.

> **Status:** Private incubation — scaffold only. Runtime implementation is not
> yet present. Repository visibility change requires explicit publication approval.
>
> **Unofficial community project.** Not an official Kyutai release. Model weights
> retain their CC-BY-4.0 license. Source code is Apache-2.0.

## What this is

`unmute-mlx-bridge` will provide two long-lived processes:

- `unmute-mlx-stt` — STT model server on port 8090, implementing the Kyutai
  Unmute `/api/asr-streaming` WebSocket contract.
- `unmute-mlx-tts` — TTS model server on port 8089, implementing the Kyutai
  Unmute `/api/tts_streaming` WebSocket contract.

Both services use MLX for inference on Apple Silicon. The stock pinned Unmute
backend connects without source patches. See the design spec for the full
protocol, architecture, and delivery plan.

## Quick Start

> Runtime implementation not yet available. This is a governance and scaffold
> seed only.

```bash
# Requires uv and Python 3.12; uv.lock is committed
uv sync --locked

# Run portable smoke tests
uv run --locked pytest -q
```

## Platform

- macOS 14 or newer on Apple Silicon (`arm64`).
- Python `>=3.12,<3.13`, package manager `uv`.
- Portable CI targets the repo-owned Forgejo K3s label
  `unmute-mlx-bridge` without model weights.
- Hardware tests (real models, Apple Silicon) are opt-in: `pytest -m hardware`.

## Repository Map

Start with:

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

Model weights (`kyutai/stt-1b-en_fr-mlx`, `kyutai/tts-1.6b-en_fr`) are not
redistributed. They retain their CC-BY-4.0 license. Refer to the upstream Kyutai
repositories for model licensing terms.

## License

Apache-2.0. See [LICENSE](LICENSE).
