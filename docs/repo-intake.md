# Repository Lifecycle Record — unmute-mlx-bridge

Durable repository and release posture. Update this file with `AGENTS.md` and
`package-surface.json` whenever authority or lifecycle state changes.

## Identity

| Field | Value |
|---|---|
| Repository | `nwalker85/unmute-mlx-bridge` |
| Canonical forge | GitHub (public) |
| Disaster-recovery mirror | Forgejo `nate/unmute-mlx-bridge` |
| Owner | Nate Walker |
| Platform area | Runestack; community voice infrastructure |
| License | Apache-2.0 |
| Package status | Public source; no PyPI or container publication |

## Purpose

`unmute-mlx-bridge` provides persistent Apple Silicon MLX speech-to-text and
text-to-speech services implementing Kyutai Unmute's model-server WebSocket
contract. It lets stock Unmute use MLX inference without forking Unmute or
embedding consumer-specific orchestration.

This is an unofficial community project. Kyutai code, model, and voice
attribution and license boundaries are documented in `README.md` and `NOTICE`.

## API And Runtime Surface

- STT WebSocket: `/api/asr-streaming`
- TTS WebSocket: `/api/tts_streaming`
- Operations: `/healthz`, `/readyz`, and `/metrics` on each process
- Runtime entry points: `unmute-mlx-stt` and `unmute-mlx-tts`
- SDK: none
- Package registry: none
- Production service: none; local Apple Silicon and downstream-managed
  deployments only

The bridge accepts an optional static `kyutai-api-key`. Non-loopback exposure
requires a configured token. It does not own user identity, conversation state,
prompts, tools, or agent policy.

## CI And Evidence

- GitHub Actions is canonical CI.
- The portable Python 3.12 suite runs automatically on pull requests and pushes
  to `main` using `ubuntu-latest`.
- Real-model tests use a manual `workflow_dispatch` input and an Apple Silicon
  `macos-14` runner. They are never executed for untrusted pull requests.
- Local gate: `uv sync --locked`, `git diff --check`, and
  `uv run --locked pytest -q`.
- Hardware and end-to-end evidence remain distinct from portable conformance
  tests. Report the exact commit and evidence surface for every claim.

## Versioning And Release

- SemVer v2, currently `0.1.x`.
- Breaking changes are documented even before 1.0.
- A GitHub Release requires a changelog entry, exact-commit validation,
  attribution/license review, and explicit release approval.
- Source visibility does not authorize PyPI, container, or production
  publication.

## Lifecycle State

| Gate | State | Evidence or next action |
|---|---|---|
| Public source | Complete | GitHub repository is public |
| Canonical authority | Complete | GitHub Issues, PRs, Actions, Discussions |
| Community files | Complete | README, contributing, conduct, security, support, templates |
| Portable CI | Active | GitHub Actions `Portable tests` job |
| Hardware CI | Manual | Trusted `workflow_dispatch` only |
| Secret scanning | Active | GitHub secret scanning and push protection |
| Private vulnerability reporting | Active | GitHub Security advisory intake |
| Code scanning | Active | GitHub CodeQL default setup |
| Package registry | Not planned | Source distribution only |
| GitHub Release | Pending approval | No tag or release has been published |
| Production cutover | Not authorized | Requires its own ADR and approval |

## Public-History Variance

[ADR-0001](architecture/decisions/0001-public-release.md) specified a fresh,
sanitized history export. The repository was instead made public in place with
existing history and old development branches. The current tree can be cleaned
normally, but branch deletion or history rewriting is destructive and requires
a separate plan and explicit approval. Do not treat this record as that
approval.
