# PR Checklist

- [ ] Work is on a branch or dedicated worktree, not direct `main`.
- [ ] Scope is described clearly in the PR body.
- [ ] Tests run are listed with exact commands (`uv sync --locked && git diff --check && uv run --locked pytest -q`).
- [ ] Screenshots or evidence are attached for UI/user-visible changes.
- [ ] Backward compatibility and migration notes are included when relevant.
- [ ] Sensitive data scan completed: no secrets, raw data, private exports,
      private hostnames, topology, recordings, or generated evidence.
- [ ] No model weights or Hugging Face cache artifacts committed.
- [ ] Hardware-gated tests (`pytest -m hardware`) are not in the portable CI job.
- [ ] `package-surface.json` is consistent with any CI or registry changes.
- [ ] Known limitations and follow-ups are stated.
- [ ] Reviewer can reproduce validation from the PR body.
