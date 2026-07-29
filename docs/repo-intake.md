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

- `runner_labels`: repo-scoped Norns ARC label `norns-unmute-mlx-bridge` for
  portable Linux/amd64 CI.
- GitHub is canonical. The intended `ci_lane` is `github-actions` on
  `norns-unmute-mlx-bridge`.
- Verified 2026-07-26: `.github/workflows/ci.yml` passes local `actionlint` and
  `yamllint`. The repository Actions API reports Actions enabled with
  `allowed_actions=all`, but workflow-triggered runs stop before job allocation
  as nameless `BuildFailed` / `startup_failure` runs with zero jobs. They
  provide no remote test evidence.
- Verified 2026-07-29: `norns-unmute-mlx-bridge` registered successfully and
  its listener authenticated to GitHub. Fresh PR events still stopped as
  `BuildFailed` / `startup_failure` with zero jobs, while the listener received
  zero assigned jobs. The remaining blocker precedes runner scheduling.
- Until GitHub restores workflow-start capability for this private repository,
  locked local validation (`uv sync --locked`, `git diff --check`, and
  `uv run --locked pytest -q`) is the active private-incubation PR evidence
  gate. Workflow changes also require local `actionlint` and `yamllint`
  evidence.
- GitHub's `CLEAN` merge state is mergeability metadata, not CI evidence, and
  must not be reported as a passing check.
- The ARC release is deployed and ready but has no successful execution
  evidence because GitHub never creates a job. Remote CI execution and branch
  protection/ruleset enforcement remain pending external GitHub account
  capability; current branch protection and ruleset API requests return `403`.
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
| Scaffold lane | ✓ python-uv | GitHub Actions on repo-scoped Norns ARC |
| SemVer policy | ✓ 0.y.z | semver-v2 from 0.1.0 |
| Namespace decision | ✓ not applicable | Loopback model endpoints only |
| SDK/API decision | ✓ recorded | MessagePack WS + HTTP ops |
| Identity decision | ✓ not applicable | Optional static header only |
| Runner decision | ✓ recorded | Repo-scoped `norns-unmute-mlx-bridge`, scale 0..1 |
| Observability decision | ✓ recorded | Prometheus in package; dashboards downstream |
| CI workflow definition | ✓ locally validated | `.github/workflows/ci.yml` passes actionlint and yamllint |
| Private-incubation PR evidence | ✓ active local gate | Locked sync, diff check, and portable tests |
| Remote CI execution | blocked upstream | ARC live; GitHub creates zero jobs |
| Branch protection/rulesets | pending external capability | GitHub API returns 403 |
| Publication gate | pending | Private incubation; explicit approval required |
| ADR: production cutover | pending | Required before any downstream integration |
| ADR: publication | pending | Required before visibility change |
