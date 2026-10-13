---
name: nate-repo-lifecycle-scaffold-pattern
description: New Ravenhelm/private repos often already have a repo-lifecycle scaffold branch/PR open (AGENTS.md, pyproject.toml, CI, design spec) before any real implementation starts — check for it and stack on it
metadata:
  type: project
---

When picking up a "build X from scratch" task in a repo that looks empty
(just README + LICENSE at `main`), check `git branch -a` / `git log --all`
and `gh pr list --state all` before assuming there's nothing to build on.
In `nwalker85/unmute-mlx-bridge`, `main` was a one-commit stub, but a
`docs/mlx-bridge-design` branch (open PR #1, unmerged) already had a full
`ravenhelm-repo-lifecycle`-skill scaffold: `AGENTS.md`, `pyproject.toml` with
`uv`/`hatchling`, `.github/workflows/ci.yml`, `package-surface.json`,
`.agents/` operational-handoff directory, and — critically — a detailed
design spec at `docs/superpowers/specs/<date>-<name>-design.md` that had
already pinned upstream compatibility commits, chosen the wire-format
approach, and specified test strategy in more implementation-ready detail
than the task prompt itself.

**How to apply:** branch new implementation work off that scaffold branch,
not off `main` — `main` doesn't have the scaffold until PR #1 merges, and
duplicating it would create merge conflicts / a redundant diff. Open the new
PR with `--base <scaffold-branch>` (a stacked PR), and say so explicitly in
the PR body so the reviewer isn't confused about why the diff includes files
from PR #1. Treat the existing design spec as authoritative unless it
conflicts with source-verified reality (in this case it didn't — it held up
well against direct Rust/Python source reading).

See [[moshi-server-protocol-verification]] for what was actually built here.
