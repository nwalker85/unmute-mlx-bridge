# TTS Query Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply stock-Unmute's parsed TTS seed, sampler, CFG, and session-length
settings so repeated sessions are deterministic and voice-stable.

**Architecture:** `TtsServer` forwards the validated query into `TtsSession`.
The session seeds MLX, validates and applies CFG conditioning, and constructs
both samplers with the requested temperature and top-k. No model, protocol, or
frontend changes are included.

**Tech Stack:** Python 3.12, MLX 0.26.5, moshi-mlx 0.3.0, pytest, uv.

## Global Constraints

- Retain full 32-codebook generation and q4 quantization.
- Do not change MLX dependencies in this plan.
- Keep PR #5 unmerged until a physical canary passes.
- Portable tests must not initialize Metal or download weights.

---

### Task 1: Prove Query Forwarding

**Files:**
- Modify: `tests/test_tts_server_conformance.py`
- Modify: `src/unmute_mlx_bridge/tts/server.py`

- [ ] Add seed, temperature, top-k, and CFG fields to `FakeSession`, plus an
  instance capture list.
- [ ] Add a websocket test using
  `seed=7&temperature=0.4&top_k=11&cfg_alpha=1.5&max_seq_len=321`.
- [ ] Assert the constructed fake session received those five literal values.
- [ ] Run the focused test and observe failure because the server does not pass
  the values.
- [ ] Pass parsed values to `_session_cls`; use configured max length only when
  `query.max_seq_len is None`.
- [ ] Rerun the focused test and observe a pass.

### Task 2: Prove Session Sampling Fidelity

**Files:**
- Modify: `tests/test_tts_engine.py`
- Modify: `src/unmute_mlx_bridge/tts/engine.py`

- [ ] Build portable fake MLX, `Sampler`, `LmGen`, model, and bundle objects.
- [ ] Instantiate `TtsSession` with seed 7, temperature 0.4, top-k 11, and CFG
  1.5.
- [ ] Assert MLX was seeded with 7, both samplers received `(0.4, 11)`, and
  `make_condition_attributes` received CFG 1.5.
- [ ] Add a test that CFG 1.25 raises before generation when the model supports
  only 1.0 and 1.5.
- [ ] Run both tests and observe failure because `TtsSession` does not accept
  the query settings.
- [ ] Add defaulted `TtsSession` fields matching stock defaults: seed 42,
  temperature 0.8, top-k 250, and no CFG override.
- [ ] Seed MLX after cache resets, validate the requested CFG value, and use the
  requested settings for both samplers.
- [ ] Rerun `tests/test_tts_engine.py` and observe all tests pass.

### Task 3: Verify And Ship The PR Update

**Files:**
- Verify all files modified in Tasks 1 and 2.

- [ ] Run `uv sync --locked`.
- [ ] Run `git diff --check`.
- [ ] Run `uv run --locked pytest -q`.
- [ ] Review `git diff` and confirm only query-fidelity files changed.
- [ ] Commit with `fix: honor TTS session query settings`.
- [ ] Push `fix/tts-receive-queue`.
- [ ] Confirm Forgejo PR #5 points to the exact commit and remains unmerged.

### Task 4: Deterministic Odin Canary

**Files:**
- Deploy worktree:
  `/Users/ravenhelm/var/canaries/unmute-mlx-bridge-pr5`

- [ ] Deploy the exact PR SHA with `TTS_N_Q=32` and
  `TTS_QUANTIZE_BITS=4`.
- [ ] Generate the same text twice through the websocket with voice
  `unmute-prod-website/p329_022.wav` and CFG 1.5.
- [ ] Confirm both PCM hashes are identical.
- [ ] Save one private WAV and run the physical intelligibility/voice-stability
  canary.
- [ ] Do not begin MLX 0.32 benchmarking unless this canary passes.
