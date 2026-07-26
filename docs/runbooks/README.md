# Runbooks

Use this folder for human-facing operational runbooks.

Agent-specific copy-paste command notes can live in `.agents/runbooks/`, but
durable operational procedures should graduate here.

## Planned runbooks

- `canary-benchmark.md` — steps to run and record the Apple Silicon canary
  benchmark on the reference host.
- `model-download.md` — how to pre-download STT and TTS model weights using
  the Hugging Face cache.
- `hardware-test.md` — running opt-in hardware tests with `pytest -m hardware`.
