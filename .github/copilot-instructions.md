# GitHub Copilot instructions

Follow the repository authority and safety contract in `AGENTS.md`.

- Treat `PROTOCOL.md` as the wire-compatibility oracle.
- Preserve the pinned upstream revisions, protocol fixtures, and conformance
  tests as one change unit.
- Keep the default test suite portable: no MLX import, model download, or Apple
  hardware dependency outside explicitly marked hardware tests.
- Never add secrets, recordings, private infrastructure identifiers, model
  weights, or generated evidence to the repository.
- Prefer focused changes with tests. Validate with `uv sync --locked`,
  `git diff --check`, and `uv run --locked pytest -q`.
- Do not claim release, deployment, hardware, or end-to-end proof from portable
  tests alone.
