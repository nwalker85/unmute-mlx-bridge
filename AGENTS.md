# Agent Guide — unmute-mlx-bridge

This file is authoritative for this repository. Update it when repository
authority, CI, release surfaces, or safety boundaries change.

## Source Of Truth

- Canonical repository: `https://github.com/nwalker85/unmute-mlx-bridge`
  (public, GitHub-primary).
- Disaster-recovery mirror: Forgejo `nate/unmute-mlx-bridge`.
- Human documentation: `README.md`, `PROTOCOL.md`, and `docs/`.
- Project work: GitHub Issues, Pull Requests, and Discussions.
- Production runtime: none. This is a source distribution for local Apple
  Silicon use; downstream deployment is a separate decision.

Some older local clones name the Forgejo remote `origin` and GitHub remote
`github`. Resolve the actual remote URL before pushing; contributors cloning
from GitHub will normally have GitHub as `origin`.

## Repository Posture

- Work on a branch or dedicated worktree and open a GitHub pull request.
- Never push directly to `main`; each merge requires Nate's explicit approval
  for that pull request.
- GitHub Actions is the public CI surface. Portable tests run automatically on
  pushes to `main` and pull requests targeting `main`.
- Real-model Apple Silicon tests are manual and opt-in. They download several
  gigabytes of model weights and must not run for untrusted pull requests.
- The repository publishes source, not a PyPI package or container image.
  Creating a GitHub Release or publishing an artifact requires separate,
  explicit approval.
- Advisory AI review may be used, but it never replaces exact-commit tests or
  human merge approval.

## Privacy And Data Boundaries

- No private consumer names, hostnames, topology, logs, recordings, or secrets
  may enter any commit, issue, discussion, workflow log, or release artifact.
- No production cutover without an evidence-backed ADR and explicit cutover
  approval.
- No model weights in the repository. Downloads use the standard Hugging Face
  cache outside the checkout.
- Hardware measurements and voice samples must be sanitized before they are
  shared publicly. Do not publish identifiable recordings without consent.

## Protocol And Runtime Guardrails

- `PROTOCOL.md` is the compatibility oracle. A wire-contract change must update
  the pinned upstream revisions, fixtures, tests, and compatibility notes
  together.
- Portable CI must not import MLX, download weights, or require Apple hardware.
- Full hardware tests use `pytest -m hardware` and must remain explicitly
  selected.
- Preserve the single-session admission, cancellation, health/readiness, and
  authentication behavior documented by the conformance tests.
- Binding to a non-loopback interface requires an explicit token.
- Do not claim production readiness from component health alone.

## Build And Test

```bash
# Install the locked Python 3.12 environment.
uv sync --locked

# Run the portable protocol and lifecycle suite.
uv run --locked pytest -q

# Check whitespace before committing.
git diff --check

# Apple Silicon only; downloads and runs real model weights.
# uv run --locked pytest -m hardware -q -s
```

When workflow files change, also validate the YAML and run `actionlint` when it
is available. Treat a green shell exit as a claim to verify, not proof by
itself; report the exact commit and command output.

## Toolchain

- Python: `>=3.12,<3.13`
- Package manager: `uv` with committed `uv.lock`
- Build backend: Hatchling
- Tests: pytest
- CI: GitHub Actions on `ubuntu-latest`; manual hardware proof on `macos-14`

## Release And Package Surface

`package-surface.json` is the machine-readable release contract:

- `registry_target`: `none`
- `publication_status`: `public_source`
- `semver_policy`: `semver-v2`, currently `0.1.x`
- `ci_lane`: `github-actions`
- `runner_label`: `github-hosted:ubuntu-latest`

Before a GitHub Release, update `CHANGELOG.md`, validate the exact proposed tag
commit, confirm attribution and model-license guidance, and obtain explicit
release approval. Merged, released, deployed, and live-verified are distinct
states.

## Agent Workspace

- `.agents/context/repo-map.md` — code, tests, and runtime entry points
- `.agents/checklists/pr.md` — PR evidence checklist
- `.agents/checklists/release.md` — release checklist
- `.agents/plans/implementation/` — active implementation plans
- `.agents/archive/` — completed or superseded plans
- `.github/agents/` — public GitHub Copilot custom agents
