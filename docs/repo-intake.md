# Repository Intake Record — unmute-mlx-bridge

Durable lifecycle intake record. Update this file when any decision changes.
Cross-reference: `AGENTS.md`, `package-surface.json`.

## Identity

| Field | Value |
|---|---|
| codename | unmute-mlx-bridge |
| repo | `nate/unmute-mlx-bridge` (Forgejo, canonical private forge) |
| mirror_repo | `nwalker85/unmute-mlx-bridge` (GitHub, private passive mirror) |
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

- `runner_labels`: repo-owned Norns K3s label `unmute-mlx-bridge` for portable
  Linux/amd64 CI.
- Forgejo is canonical. The intended `ci_lane` is `forgejo-actions` on
  `unmute-mlx-bridge`.
- GitHub is a private passive mirror; `.github/workflows/ci.yml` is dormant
  until explicit public-publication cutover approval.
- Locked local validation (`uv sync --locked`, `git diff --check`, and
  `uv run --locked pytest -q`) is the active private-incubation PR evidence
  gate until the K3s lane runs the exact commit. Workflow changes also require
  local `actionlint` and `yamllint` evidence.
- Advisory review uses Snotra with Fenrir only; no cloud fallback is allowed.
- GitHub's `CLEAN` merge state is mergeability metadata, not CI evidence, and
  must not be reported as a passing check.
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
| AGENTS.md | ✓ present | Forgejo canonical, privacy boundaries, guardrails |
| package-surface.json | ✓ present | Forgejo CI posture, private_incubation |
| Scaffold lane | ✓ python-uv | Forgejo Actions on repo-owned Norns K3s label |
| SemVer policy | ✓ 0.y.z | semver-v2 from 0.1.0 |
| Namespace decision | ✓ not applicable | Loopback model endpoints only |
| SDK/API decision | ✓ recorded | MessagePack WS + HTTP ops |
| Identity decision | ✓ not applicable | Optional static header only |
| Runner decision | ✓ recorded | Repo-owned `unmute-mlx-bridge`, scale 0..1 |
| Observability decision | ✓ recorded | Prometheus in package; dashboards downstream |
| CI workflow definition | pending validation | `.forgejo/workflows/ci.yml` must pass local lint and a Forgejo canary |
| Private-incubation PR evidence | ✓ active local gate | Locked sync, diff check, and portable tests |
| Remote CI execution | pending runner PR | K3s lane must deploy and run the exact commit |
| Branch protection | pending CI canary | Configure after the required Forgejo contexts are proven |
| Publication gate | pending | Private incubation; explicit approval required |
| ADR: production cutover | pending | Required before any downstream integration |
| ADR: publication | pending | Required before visibility change |

## Publication Export Plan

Decided 2026-08-22 (Nate). Records how the private Forgejo history becomes the
public GitHub-canonical repo at cutover — supersedes any prior assumption that
`main` is pushed as-is.

- **History mechanism: fresh squashed history.** At cutover, export the
  sanitized tree as a new (single- or few-commit) history for the
  GitHub-canonical repo. The full private development history — including
  agent working notes, canary iteration, and anything caught by the private-
  reference sweep below — stays on Forgejo only and is never pushed to the
  public remote. Do not `git push --force` the existing Forgejo `main` history
  to GitHub as a rewrite-in-place; generate a new history instead.
- **Excluded from the public export** (Forgejo-private only — internal ops
  runbooks, not user-facing documentation):
  - `docs/superpowers/plans/2026-07-29-tts-query-fidelity.md`
  - `docs/superpowers/plans/2026-07-30-odin-single-profile-buffered-tts.md`
  - `docs/superpowers/specs/2026-07-29-tts-query-fidelity-design.md`
  - `docs/superpowers/specs/2026-07-29-odin-tts-codebook-depth-design.md`
  - `docs/superpowers/specs/2026-07-30-odin-single-profile-buffered-tts-design.md`
  - `docs/superpowers/specs/2026-07-30-tts-fidelity-lessons.md`

  These contain internal infrastructure identifiers, SSH aliases, and canary
  filesystem paths. `docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md`
  is the core architecture doc (linked from `README.md`) and IS included in
  the public export — it was verified clean of such references
  (`gitleaks detect --log-opts="--all"`, 2026-08-22: 0 leaks; plus a pattern
  sweep for internal infrastructure identifiers).
- **The unmerged `forge-ci-bootstrap` branch** (internal CI bootstrap
  scaffolding referencing an internal build host, a private package registry,
  and a private token path) is excluded by construction — the export only ever
  walks `main`. Left as a private branch pending a separate prune decision.
- Before the actual visibility flip, re-run the full sweep against the exact
  commit being exported, not just this snapshot: `gitleaks detect --source .
  --log-opts="--all"`, `git secrets --scan-history`, and the internal-identifier
  pattern grep. **The concrete pattern list is deliberately not written here** —
  this file is itself part of the public export, and a list of internal
  hostnames is exactly the kind of content this gate exists to keep out. Keep
  the patterns in the private ops checklist instead.
