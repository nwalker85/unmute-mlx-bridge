# Agent Guide — unmute-mlx-bridge

This file is authoritative for this repository. Update it when repo rules,
remotes, deploy surfaces, or safety boundaries change.

## Source Of Truth

This section, and "Repository Posture" below, describe **two states**: the
current phase, and what changes at the public-release flip. Both are decided
(see [ADR-0001](docs/architecture/decisions/0001-public-release.md)) but only
the current-phase column is live today. Do not read the "at flip" column as
already true, and do not treat ADR-0001 itself as authorization to make it
true — that still requires Nate's explicit approval per the Publication Gate
below.

| | Current phase (private incubation) | At the flip (ADR-0001) |
|---|---|---|
| Canonical git | Forgejo `nate/unmute-mlx-bridge` (local remote `origin`) | GitHub `nwalker85/unmute-mlx-bridge` |
| Mirror | GitHub `nwalker85/unmute-mlx-bridge`, private, passive (local remote `github`) | Forgejo `nate/unmute-mlx-bridge`, pull/DR mirror |
| History | Full private development history | Fresh squashed history exported to GitHub (see `docs/repo-intake.md` → "Publication Export Plan"); old private-mirror history on GitHub is discarded, not preserved |
| CI surface | `.forgejo/workflows/ci.yml` (repo-owned Norns K3s runner) | `.github/workflows/ci.yml` (its triggers widen past `workflow_dispatch`) |
| Visibility | Private | Public |

- Project tracker: `docs/repo-intake.md` and Forgejo PRs are the durable
  tracking surfaces for this repo (today; GitHub Issues/PRs at the flip).
- Human docs: `docs/` in this repo.
- Production URL or runtime: none — private incubation, no production cutover.
  The public release described here is a **repository** visibility change,
  not a production cutover; those remain separate, both requiring their own
  explicit approval.

## Repository Posture

**Today:**

- **Forgejo is canonical** for development, PRs, CI, and review. GitHub is a
  private passive mirror.
- The repo is **private**. Visibility change requires explicit publication
  approval from Nate (see `docs/repo-intake.md` and the Publication Gate
  below) — ADR-0001 records the decision's shape, not its approval.
- **Forgejo Actions is the active remote CI surface**, using the repo-owned
  Norns K3s runner label `unmute-mlx-bridge` for portable Linux/amd64 CI.
- GitHub Actions is dormant (`on: workflow_dispatch` only in
  `.github/workflows/ci.yml`) while GitHub is a passive mirror. This includes
  its `hardware` job (macOS/Apple Silicon, added under RAV-1613) — dormant
  along with everything else in that workflow, not live CI today.
- Registry and artifact publication are deferred until explicit approval.

**At the flip** (per ADR-0001, once Nate gives explicit publication
approval): the authority boundary reverses — GitHub becomes canonical,
Forgejo becomes a pull/DR mirror, `.github/workflows/ci.yml`'s triggers widen
past manual dispatch, and history is exported per `docs/repo-intake.md` →
"Publication Export Plan" rather than pushed in place. Update this file's
"Today" bullets to describe the new reality once that happens — do not leave
this section describing the pre-flip state after the flip has occurred.

### Verified Private-Incubation Enforcement State

Verified through 2026-07-29:

- The private repository and its branch history are present on Forgejo, and
  Forgejo bootstrap CI has executed successfully.
- The repo-owned K3s runner lane is proposed separately. Until that lane is
  deployed and runs this repository's exact commit, locked local validation is
  the active private-incubation PR evidence gate: `uv sync --locked`,
  `git diff --check`, and `uv run --locked pytest -q`.
- Include `actionlint` and `yamllint` evidence when workflow files change.
- GitHub's `CLEAN` merge state is mergeability metadata, not CI evidence. Do
  not report it as a passing check or substitute it for test results.

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
- Advisory review must use Snotra with the Fenrir backend. Do not configure
  Anthropic, OpenAI, or an automatic cloud fallback.

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

- Intended remote CI: Forgejo Actions (`.forgejo/workflows/ci.yml`) on the
  repo-owned `unmute-mlx-bridge` K3s runner label
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
- `ci_lane` — `forgejo-actions`
- `runner_label` — `unmute-mlx-bridge`
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

See `docs/repo-intake.md` → "Publication Export Plan" for the decided export
mechanism (fresh squashed history for the public repo; full private history
stays Forgejo-only), `.agents/checklists/publication-export.md` for the
current by-name exclusion list, and
[ADR-0001](docs/architecture/decisions/0001-public-release.md) for the full
set of sub-decisions (history mechanism, authority flip, agent-surface
publication, voice licensing, `cfg_coef` conformance) this gate's checklist
items depend on. None of that is itself the approval this gate requires.

## Agent Workspace

Use `.agents/` for operational handoffs:

- `.agents/context/repo-map.md` — entry points, tests, deploy surfaces
- `.agents/checklists/pr.md` — before opening a PR
- `.agents/checklists/release.md` — before claiming a release is live
- `.agents/checklists/publication-export.md` — private-ops detail for the
  publication export: by-name exclusion list, pre-flip sweep procedure
- `.agents/plans/implementation/` — active implementation plans
- `.agents/archive/` — completed or superseded agent plans
