## Summary

## Compatibility and proof boundary

Describe any protocol or lifecycle impact, and state what this PR does not
prove (for example, real-model Apple Silicon or downstream end-to-end behavior).

## Validation

Exact commands run and their results:

```
uv sync --locked
git diff --check
uv run --locked pytest -q
```

## Checklist

- [ ] No secrets, tokens, private hostnames, or infrastructure topology in
      the diff.
- [ ] No model weights or Hugging Face cache artifacts committed.
- [ ] Hardware-gated tests (`pytest -m hardware`) are not in the portable CI
      job.
- [ ] Protocol changes update upstream pins, citations, fixtures, tests, and
      `PROTOCOL.md` together.
- [ ] `CHANGELOG.md` updated if this changes user-facing behavior.
- [ ] Known limitations and follow-ups are stated below.

## Notes
