# Contributing

Thanks for helping make Unmute work well on Apple Silicon. Protocol fixtures,
compatibility fixes, performance measurements, documentation improvements, and
focused MLX runtime changes are all welcome.

## Start In The Right Place

- Ask usage and troubleshooting questions in [GitHub Discussions](https://github.com/nwalker85/unmute-mlx-bridge/discussions).
- Report reproducible bugs with the issue templates.
- Propose larger behavior or protocol changes in an Idea discussion before
  writing a large patch.
- Report vulnerabilities through GitHub's private vulnerability reporting;
  see [SECURITY.md](SECURITY.md).

## Development Workflow

1. Fork or branch from current `main`.
2. Use Python 3.12 and install the locked environment with `uv sync --locked`.
3. Make the smallest coherent change and add or update tests.
4. Run the portable validation below.
5. Open a pull request and explain the behavior, compatibility impact, and
   exact commands you ran.

Do not push directly to `main`. Maintainers merge reviewed pull requests only.

## Validation

```bash
uv sync --locked
git diff --check
uv run --locked pytest -q
```

Portable tests must not download weights, import MLX, or require Apple
hardware. Put real-model tests under `tests/hardware/`, mark them with
`pytest.mark.hardware`, and run them only when explicitly requested:

```bash
uv run --locked pytest -m hardware -q -s
```

When reporting a performance result, include the Mac model, memory, macOS
version, bridge commit, upstream Unmute revision, model identifiers,
quantization, delivery mode, and printed `real_time_factor`.

## Protocol Changes

Read [PROTOCOL.md](PROTOCOL.md) before changing message models, framing, or
session behavior. A compatibility change must update all of these together:

- the pinned upstream source revision and citation;
- protocol fixtures and conformance tests;
- `PROTOCOL.md` and any affected architecture notes;
- `CHANGELOG.md`, including a breaking-change callout when applicable.

## Commit And Pull Request Style

Use clear, scoped commits. Conventional Commit prefixes are encouraged:

- `feat:` new behavior
- `fix:` bug fix
- `docs:` documentation only
- `test:` test-only change
- `chore:` tooling or maintenance

Keep generated artifacts, model weights, `.env` files, recordings, and private
infrastructure details out of every commit. This is a public repository, so
comments, fixtures, workflow logs, and discussion posts are public too.

## Attribution

This is an unofficial community project, not an official Kyutai component.
Contributions must preserve the upstream attribution and per-model/per-voice
license guidance in [NOTICE](NOTICE) and the README.
