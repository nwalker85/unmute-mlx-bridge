# Repository Intake Record — unmute-mlx-bridge

Durable lifecycle intake record. Update this file when any decision changes.
Cross-reference: `AGENTS.md`, `package-surface.json`.

## Identity

| Field | Value |
|---|---|
| codename | unmute-mlx-bridge |
| repo | nwalker85/unmute-mlx-bridge |
| owner | Nate Walker |
| filesystem_path | `~/src/platforms/runestack/unmute-mlx-bridge` |
| platform_area | Runestack; community voice infrastructure |

## Purpose

`unmute-mlx-bridge` provides persistent Apple Silicon MLX speech-to-text and
text-to-speech services that implement the model-server WebSocket contract used
by Kyutai Unmute. It fills the gap between the Kyutai stock Unmute backend
(which expects Linux/CUDA model servers) and Apple Silicon MLX inference, without
forking Unmute or embedding consumer-specific orchestration.

The source is Apache-2.0. Documentation attributes Kyutai, links the upstream
repositories and paper, distinguishes this community project from an official
Kyutai release, and tells users that model weights retain their CC-BY-4.0 license.

## Namespace

Not applicable. This package exposes local model-server endpoints only, with
configurable loopback defaults. It does not define a canonical tenant path.
If deployed as a network-accessible service the caller is
responsible for its own routing and namespace conventions.

## SDK/API Surface

- `sdk_surface`: none for v0.1. No SDK package is published.
- `api_surface`: MessagePack WebSocket model protocol (STT at
  `/api/asr-streaming`, TTS at `/api/tts_streaming`) plus HTTP health
  (`/healthz`), readiness (`/readyz`), and metrics (`/metrics`) endpoints.
  Protocol shapes are pinned to the upstream Unmute compatibility baseline.

## Identity and Authority

The bridge does not integrate with organization identity systems. It accepts an
optional static `kyutai-api-key` header for loopback-to-non-loopback promotion.
Tokens come from environment variables and are never logged. Full identity
authority is not required for v0.1.

## Runner and CI

- `runner_labels`: GitHub-hosted `ubuntu-latest` for portable CI.
- GitHub is canonical, and GitHub Actions is the only CI surface. There is no
  alternate CI enrollment.
- Registry and artifact publication are deferred until explicit approval.

## Observability

- `observability`: Prometheus endpoint (`/metrics`) included in the package with
  standard process and inference metrics. Dashboards and downstream metrics
  storage integrations are owned by downstream deployments — not by this
  repository. Hardware test measurements are canary artifacts, not CI
  requirements.

## Versioning

- `semver_policy`: `0.y.z` under SemVer v2, starting at `0.1.0`.
  Breaking changes are documented in `CHANGELOG.md` even under `0.y.z`.
  The first `1.0.0` release signals a stable public contract.

## Publication Status

`private_incubation`. Visibility change from private to public is a separate
explicit decision. See the publication gate in `AGENTS.md`.

## Lifecycle Gates

| Gate | Status | Notes |
|---|---|---|
| Intake record | ✓ present | This document |
| AGENTS.md | ✓ present | GitHub canonical, privacy boundaries, guardrails |
| package-surface.json | ✓ present | GitHub CI posture, private_incubation |
| Scaffold lane | ✓ python-uv | GitHub Actions on ubuntu-latest |
| SemVer policy | ✓ 0.y.z | semver-v2 from 0.1.0 |
| Namespace decision | ✓ not applicable | Loopback model endpoints only |
| SDK/API decision | ✓ recorded | MessagePack WS + HTTP ops |
| Identity decision | ✓ not applicable | Optional static header only |
| Runner decision | ✓ recorded | GitHub-hosted, no alternate CI enrollment |
| Observability decision | ✓ recorded | Prometheus in package; dashboards downstream |
| CI | ✓ .github/workflows/ci.yml | GitHub Actions |
| Publication gate | pending | Private incubation; explicit approval required |
| ADR: production cutover | pending | Required before any downstream integration |
| ADR: publication | pending | Required before visibility change |
