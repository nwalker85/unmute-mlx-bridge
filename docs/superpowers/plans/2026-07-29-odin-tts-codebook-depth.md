# Odin TTS Codebook Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `TTS_N_Q` reduce depformer execution and prove the resulting
Odin TTS canary can sustain smooth realtime audio.

**Architecture:** The bridge will limit the loaded LM's generated-codebook
configuration and depformer slice list after strict checkpoint loading and
before quantization. Existing `LmGen` then treats omitted codebooks as
zero-valued external streams and passes only retained generated codebooks to
Mimi. No upstream package files or stock-Unmute services are modified.

**Tech Stack:** Python 3.12, MLX 0.26.5, moshi-mlx 0.3.0, pytest, uv, PM2,
Forgejo.

## Global Constraints

- Fenrir remains outside the TTS path.
- Keep Forgejo PR #5 unmerged until Nate approves that specific PR after the
  physical canary.
- Load every checkpoint weight strictly before changing the executable
  depformer depth.
- Portable tests must not import MLX or download model weights.
- Hardware tests remain opt-in and run only on Apple Silicon.
- The winning Odin setting must sustain at least 1.10x realtime, emit first
  audio within 1.0 second, and keep post-start inter-frame gaps at or below
  160 milliseconds in the controlled benchmark.

---

### Task 1: Enforce Generated-Codebook Depth

**Files:**
- Modify: `tests/test_tts_engine.py`
- Modify: `src/unmute_mlx_bridge/tts/engine.py`

**Interfaces:**
- Consumes: a loaded `moshi_mlx.models.Lm` and requested integer `n_q`.
- Produces: `_limit_generated_codebooks(lm: object, n_q: int) -> None`.
- Produces: `TtsModelBundle.load(...)` whose `tts_model.lm.depformer.slices`
  and `tts_model.lm.cfg.depformer.num_slices` both match `n_q`.

- [ ] **Step 1: Strengthen the existing bundle test**

Change the fake LM configuration and model to expose the same mutable
boundaries as the real model:

```python
class FakeLmConfig:
    transformer = SimpleNamespace(context=2048, max_seq_len=0)
    depformer = SimpleNamespace(num_slices=32)

    @property
    def generated_codebooks(self):
        return self.depformer.num_slices

class FakeLm:
    def __init__(self, config):
        self.cfg = config
        self.depformer = SimpleNamespace(slices=list(range(32)))
        self.transformer = SimpleNamespace(layers=[])
```

Make `FakeTtsModel` retain `self.lm = _lm`, then add these assertions to
`test_model_bundle_applies_requested_audio_codebook_depth`:

```python
assert bundle.tts_model.lm.cfg.depformer.num_slices == 24
assert len(bundle.tts_model.lm.depformer.slices) == 24
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run --locked pytest -q \
  tests/test_tts_engine.py::test_model_bundle_applies_requested_audio_codebook_depth
```

Expected: FAIL because the configured count remains `32`.

- [ ] **Step 3: Add boundary tests**

Add a minimal LM factory and a parametrized test proving `0`, `-1`, and `33`
raise before mutation:

```python
@pytest.mark.parametrize("n_q", [0, -1, 33])
def test_generated_audio_codebook_depth_rejects_invalid_values(n_q):
    lm = SimpleNamespace(
        cfg=SimpleNamespace(depformer=SimpleNamespace(num_slices=32)),
        depformer=SimpleNamespace(slices=list(range(32))),
    )

    with pytest.raises(ValueError, match="generated audio codebooks"):
        tts_engine._limit_generated_codebooks(lm, n_q)

    assert lm.cfg.depformer.num_slices == 32
    assert len(lm.depformer.slices) == 32
```

- [ ] **Step 4: Implement the minimal helper**

Add this bridge-owned helper above `TtsModelBundle`:

```python
def _limit_generated_codebooks(lm: object, n_q: int) -> None:
    available = lm.cfg.depformer.num_slices
    if n_q < 1 or n_q > available:
        raise ValueError(
            f"generated audio codebooks must be between 1 and {available}, got {n_q}"
        )
    lm.depformer.slices = lm.depformer.slices[:n_q]
    lm.cfg.depformer.num_slices = n_q
```

In `TtsModelBundle.load`, preserve the original Mimi codebook count, load both
LM and Mimi weights strictly, call `_limit_generated_codebooks(lm, n_q)`, and
only then quantize the retained LM slices.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run --locked pytest -q tests/test_tts_engine.py
```

Expected: every test passes.

### Task 2: Record Hardware Timing And PCM Quality

**Files:**
- Modify: `tests/hardware/test_tts_hardware.py`

**Interfaces:**
- Consumes: `TtsConfig.from_env()` so each isolated run honors
  `TTS_QUANTIZE_BITS` and `TTS_N_Q`.
- Produces: hardware-test output containing codebook depth, audio duration,
  wall time, realtime factor, first-audio latency, maximum inter-frame gap,
  RMS, clipping count, and WAV path.

- [ ] **Step 1: Make the hardware fixture honor runtime configuration**

Load the bundle with:

```python
config = TtsConfig.from_env()
return TtsModelBundle.load(
    config.hf_repo,
    config.voice_repo,
    config.quantize_bits,
    config.n_q,
)
```

- [ ] **Step 2: Capture timing at each audio event**

Record the monotonic timestamp of the first and each subsequent audio event.
Derive:

```python
first_audio_s = audio_event_times[0] - wall_start
max_frame_gap_s = max(
    (right - left for left, right in zip(audio_event_times, audio_event_times[1:])),
    default=0.0,
)
```

Print both values with the existing duration and realtime-factor line.

- [ ] **Step 3: Add PCM safety assertions**

Assert every sample is finite, count samples with absolute magnitude at least
`0.999`, and retain the existing non-silence and word-timing assertions. Do
not add machine-specific performance assertions to the portable hardware
test.

- [ ] **Step 4: Run portable collection**

Run:

```bash
uv run --locked pytest -q
```

Expected: portable tests pass and hardware tests remain deselected.

### Task 3: Verify, Commit, And Update PR #5

**Files:**
- Verify all files changed in Tasks 1 and 2.

**Interfaces:**
- Produces: one exact Forgejo branch commit for deployment and review.

- [ ] **Step 1: Run complete local verification**

Run:

```bash
uv sync --locked
git diff --check
uv run --locked pytest -q
```

Expected: all commands exit zero.

- [ ] **Step 2: Review the diff**

Run:

```bash
git diff --stat
git diff
git status --short --branch
```

Confirm only the approved bridge, tests, spec, and plan changed.

- [ ] **Step 3: Commit implementation**

Run:

```bash
git add src/unmute_mlx_bridge/tts/engine.py \
  tests/test_tts_engine.py \
  tests/hardware/test_tts_hardware.py \
  docs/superpowers/plans/2026-07-29-odin-tts-codebook-depth.md
git commit -m "perf: honor TTS generated codebook depth"
```

- [ ] **Step 4: Push the PR branch**

Run:

```bash
git push origin fix/tts-receive-queue
```

Record the exact pushed commit and confirm Forgejo PR #5 points to it.

### Task 4: Benchmark The Exact PR Commit On Odin

**Files:**
- Deploy worktree:
  `/Users/ravenhelm/var/canaries/unmute-mlx-bridge-pr5`
- PM2 process: `unmute-canary-tts`

**Interfaces:**
- Consumes: exact pushed PR commit.
- Produces: controlled q4 results at `n_q` 24, 20, and 16 plus private WAVs.

- [ ] **Step 1: Stop only the TTS canary**

Run on Odin:

```bash
/opt/homebrew/bin/pm2 stop unmute-canary-tts
```

Confirm the backend, frontend, and STT canaries remain online.

- [ ] **Step 2: Deploy the exact PR commit**

In the deploy worktree, fetch `origin/fix/tts-receive-queue`, detach at the
recorded commit, run `uv sync --locked`, and verify `git rev-parse HEAD`
matches the pushed SHA.

- [ ] **Step 3: Run the controlled depth ladder**

Run each command separately:

```bash
TTS_QUANTIZE_BITS=4 TTS_N_Q=24 \
  uv run --locked pytest -m hardware -q -s \
  tests/hardware/test_tts_hardware.py::test_synthesizes_non_silent_audio_with_word_timing
TTS_QUANTIZE_BITS=4 TTS_N_Q=20 \
  uv run --locked pytest -m hardware -q -s \
  tests/hardware/test_tts_hardware.py::test_synthesizes_non_silent_audio_with_word_timing
TTS_QUANTIZE_BITS=4 TTS_N_Q=16 \
  uv run --locked pytest -m hardware -q -s \
  tests/hardware/test_tts_hardware.py::test_synthesizes_non_silent_audio_with_word_timing
```

Save each printed result and WAV under the private slice state directory.

- [ ] **Step 4: Select and restore the canary**

Choose the highest depth meeting every benchmark gate. Start
`unmute-canary-tts` from the exact PR worktree with
`TTS_QUANTIZE_BITS=4` and the chosen literal `TTS_N_Q`, then confirm its
health endpoint and logs.

If no depth reaches 1.10x, restore the exact PR commit at `TTS_N_Q=24` and
report the measured shortfall before attempting another optimization class.

### Task 5: Physical Canary And Merge Gate

**Files:**
- No repository files.

**Interfaces:**
- Consumes: stock-Unmute at `http://localhost:3000`.
- Produces: Nate's physical microphone/speaker verdict.

- [ ] **Step 1: Confirm the full canary path**

Verify frontend, backend, STT, and TTS process health and exact deployed SHAs.

- [ ] **Step 2: Ask Nate to run one short voice turn**

Collect the backend TTS metrics and TTS logs for the same time window.

- [ ] **Step 3: Report the merge gate honestly**

Report portable tests, exact deployed SHA, controlled benchmark, live metrics,
and Nate's audible verdict separately. Do not merge PR #5 without Nate's
explicit approval for PR #5.
