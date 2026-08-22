## Summary

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
- [ ] `CHANGELOG.md` updated if this changes user-facing behavior.
- [ ] Known limitations and follow-ups are stated below.

## Notes
