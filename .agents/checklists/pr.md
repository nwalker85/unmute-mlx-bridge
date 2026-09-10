# Pull Request Checklist

- [ ] Work is on a branch or dedicated worktree, never directly on `main`.
- [ ] The PR explains scope, compatibility impact, and proof boundaries.
- [ ] `uv sync --locked` succeeds.
- [ ] `git diff --check` succeeds.
- [ ] `uv run --locked pytest -q` succeeds on the exact commit.
- [ ] Workflow changes pass YAML validation and `actionlint` when available.
- [ ] GitHub Actions reports `Portable tests` for the exact PR commit.
- [ ] Hardware tests remain explicitly marked and absent from untrusted PR CI.
- [ ] Protocol changes update upstream pins, citations, fixtures, tests, and
      `PROTOCOL.md` together.
- [ ] No secrets, recordings, model weights, private identifiers, generated
      evidence, or internal infrastructure details are present.
- [ ] README, changelog, support, security, and architecture docs reflect any
      public behavior change.
- [ ] The report distinguishes local validation, CI, hardware proof, release,
      deployment, and downstream live verification.
- [ ] Merge waits for Nate's explicit approval of this specific PR.
