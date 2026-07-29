# Contributing

## Status

This repository is private during incubation. External contributions are not
yet accepted. This file documents the current development workflow and will be
expanded before publication.

## Workflow

- Work on a branch or dedicated worktree.
- Open a pull request for review.
- Do not push directly to `main`. PR approval is required for every merge.
- List validation commands in the PR body.
- Complete `.agents/checklists/pr.md` before requesting review.

## Commit Style

Use clear, scoped commit messages. Conventional Commit prefixes are preferred:

- `feat:` — new behavior
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — test only
- `chore:` — scaffolding, tooling, dependencies

## Validation

Before opening a PR:

```bash
uv sync --locked
git diff --check
uv run --locked pytest -q
```

## Privacy Requirements

- No private consumer names, hostnames, topology, logs, recordings, or secrets
  in any commit — even in comments or test fixtures.
- No model weights in the repository.
- Hardware test results are artifacts, not committed source.

## Hardware Tests

Hardware tests (Apple Silicon, real model weights) are opt-in. They must not
run on portable CI. Mark them with `pytest.mark.hardware` and run explicitly:

```bash
uv run --locked pytest -m hardware -q
```

## Publication Gate

Before requesting publication approval, complete every item in the publication
gate described in `AGENTS.md`. Passing the private canary does not authorize
publication.

## Upstream Attribution

This project depends on and is compatible with upstream Kyutai projects
(Unmute, Moshi, delayed-streams-modeling). It is an unofficial community
project and must not present itself as an official Kyutai component. All
contributions must preserve correct attribution and the unofficial-project
disclaimer.
