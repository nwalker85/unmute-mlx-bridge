# Agent Guide — unmute-mlx-bridge

This file is authoritative for this repository. Update it when repo rules,
remotes, deploy surfaces, or safety boundaries change.

## Source Of Truth

- Primary git: https://github.com/nwalker85/unmute-mlx-bridge (GitHub, canonical)
- Project tracker: `docs/repo-intake.md` and GitHub PRs are the durable tracking surfaces for this repo
- Human docs: `docs/` in this repo
- Production URL or runtime: none — private incubation, no production cutover

## Repository Posture

- **GitHub is canonical** for this repo. It is a publication candidate targeting
  community release under Apache-2.0.
- The repo remains **private** throughout incubation. Visibility change requires
  explicit publication approval from Nate (see `docs/repo-intake.md`).
- **GitHub Actions is the intended remote CI surface**, using the repo-scoped
  Norns ARC scale set `norns-unmute-mlx-bridge` for portable Linux/amd64 CI.
- Registry and artifact publication are deferred until explicit approval.

### Verified Private-Incubation Enforcement State

Verified through 2026-07-29:

- `.github/workflows/ci.yml` passes local `actionlint` and `yamllint`
  validation.
- The repository Actions API reports Actions enabled with
  `allowed_actions=all`, but workflow-triggered runs currently stop before job
  allocation as nameless `BuildFailed` / `startup_failure` runs with zero jobs.
  Those runs provide no remote test evidence.
- On 2026-07-29, the `norns-unmute-mlx-bridge` scale set registered
  successfully and its listener authenticated to GitHub. Fresh PR events still
  stopped as `BuildFailed` / `startup_failure` with zero jobs; the listener
  received zero assigned jobs. The remaining blocker precedes runner
  scheduling.
- Until GitHub restores workflow-start capability for this private repository,
  locked local validation is the active private-incubation PR evidence gate:
  `uv sync --locked`, `git diff --check`, and
  `uv run --locked pytest -q`. Include `actionlint` and `yamllint` evidence
  when workflow files change.
- GitHub's `CLEAN` merge state is mergeability metadata, not CI evidence. Do
  not report it as a passing check or substitute it for test results.
- The ARC release is deployed and ready but has no successful execution
  evidence because GitHub never creates a job. Remote CI execution and branch
  protection/ruleset enforcement remain pending external GitHub account
  capability; current API attempts to inspect or configure branch policy
  return `403`.

## Privacy and Data Boundaries

- No private consumer names, hostnames, topology, logs, recordings, or secrets
  may enter any commit.
- No private infrastructure references in any file.
- No production cutover without an evidence-backed ADR and explicit cutover approval.
- No model weights in the repository. Model downloads use the standard Hugging Face
  cache and remain outside the repository.
- Hardware test results and canary benchmark reports are artifacts stored in the
  private repository only; they are not published without sanitization and explicit
  approval.

## Guardrails

- Never push directly to `main`.
- PR approval required for each merge.
- Do not hardcode secrets or tokens of any kind.
- Do not commit raw data, private exports, customer data, generated evidence,
  `.env` files, or cookie values.
- Do not claim production readiness from component health checks alone.
- Separate merged, deployed, and live-verified states when reporting status.
- Full hardware tests (Apple Silicon, real model weights) are opt-in and must
  not run on portable CI. Use `pytest -m hardware` or an equivalent explicit
  marker to gate them.
- Portable CI must not download model weights.

## Build And Test

```bash
# Install dependencies (requires uv and Python 3.12; uv.lock is committed)
uv sync --locked

# Run portable test suite
uv run --locked pytest -q

# Run hardware tests (Apple Silicon only, opt-in)
# uv run --locked pytest -m hardware -q
```

## Toolchain

- Python: `>=3.12,<3.13`
- Package manager and lockfile: `uv` — `uv.lock` is committed and must stay committed
- Test runner: `pytest`
- Type checker: Pyright (not yet configured in scaffold)
- Formatter/linter: Ruff (not yet configured in scaffold)

## Deploy Model

- Intended remote CI: GitHub Actions (`.github/workflows/ci.yml`) on
  `norns-unmute-mlx-bridge`; the scale set is live, but locked local validation
  remains the active private-incubation PR evidence gate while GitHub fails
  before creating a job
- Artifact: none until publication approved
- Runtime host: local Apple Silicon (canary only)
- Migration command: not applicable
- Rollback: not applicable (no production deployment)

## Release And Package Surface

Use `package-surface.json` as the machine-readable release contract. Before
claiming a release, verify:

- `registry_target` — currently `none`; must be set before any publication
- `publication_status` — currently `private_incubation`
- `semver_policy` — `semver-v2`, starting at `0.1.0`
- `ci_lane` — `github-actions`
- `runner_label` — `norns-unmute-mlx-bridge`
- `nix_flake` — `flake-check-required-before-publish`

If any field is still a placeholder, planned value, or deferral, report the
remaining evidence gap instead of marking the issue Done.

## Publication Gate

Changing repository visibility from private to public requires:

1. Green protocol and portable CI suites on the exact proposed public commit.
2. Clean secret scan across the entire Git history.
3. No private hostnames, credentials, topology, issue references, non-public logs,
   or identifiable voice recordings anywhere in history.
4. Complete Apache-2.0 notices, Kyutai attribution, model-license guidance, and
   unofficial-community-project disclaimer.
5. Reproducible installation and canary instructions that do not depend on
   private infrastructure.
6. Reviewed README, SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, issue
   templates, and release notes.
7. Final visibility diff and explicit approval from Nate.

## Agent Workspace

Use `.agents/` for operational handoffs:

- `.agents/context/repo-map.md` — entry points, tests, deploy surfaces
- `.agents/checklists/pr.md` — before opening a PR
- `.agents/checklists/release.md` — before claiming a release is live
- `.agents/plans/implementation/` — active implementation plans
- `.agents/archive/` — completed or superseded agent plans
