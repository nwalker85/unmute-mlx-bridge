# PR Checklist

- [ ] Work is on a branch or dedicated worktree, not direct `main`.
- [ ] Scope is described clearly in the PR body.
- [ ] Tests run are listed with exact commands (`uv sync --locked && git diff --check && uv run --locked pytest -q`).
- [ ] Remote GitHub Actions startup state is reported for the PR commit,
      including run name, conclusion/failure class, and job count. A nameless
      `BuildFailed` / `startup_failure` run with zero jobs is reported as no
      remote CI evidence, not as a passing check.
- [ ] If no remote job starts, the PR body includes locked local evidence for
      `uv sync --locked`, `git diff --check`, and
      `uv run --locked pytest -q`, with exact commands and results.
- [ ] GitHub `CLEAN` merge state is treated only as mergeability metadata, not
      as CI evidence.
- [ ] Screenshots or evidence are attached for UI/user-visible changes.
- [ ] Backward compatibility and migration notes are included when relevant.
- [ ] Sensitive data scan completed: no secrets, raw data, private exports,
      private hostnames, topology, recordings, or generated evidence.
- [ ] No model weights or Hugging Face cache artifacts committed.
- [ ] Hardware-gated tests (`pytest -m hardware`) are not in the portable CI job.
- [ ] `package-surface.json` is consistent with any CI or registry changes.
- [ ] Known limitations and follow-ups are stated.
- [ ] Reviewer can reproduce validation from the PR body.
