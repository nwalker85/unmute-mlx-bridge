---
name: moshi-server-protocol-verification
description: Where the real ground truth lives for Kyutai Unmute/moshi-server's undocumented STT/TTS WebSocket protocol, for building a compatible MLX bridge
metadata:
  type: reference
---

Built `nwalker85/unmute-mlx-bridge` (private, Apple Silicon MLX STT/TTS
servers implementing `moshi-server`'s WebSocket contract so Unmute's
orchestrator can point at them). DeepWiki-style summaries of this protocol
are incomplete/wrong in places; the real ground truth is source, at these
locations (clone shallow, checkout the pinned commit, grep):

- **Wire message shapes (`OutMsg`/`InMsg` enums)**: `kyutai-labs/moshi`,
  `rust/moshi-server/src/{asr,batched_asr,py_module,tts}.rs`. These are
  `rmp_serde` (msgpack) maps with a `"type"` tag — NOT the length-prefixed
  binary framing described in `rust/protocol.md` (that doc is for the
  unrelated `/api/chat` full-Moshi endpoint's Ogg/Opus framing).
- **Query params + defaults + auth**: `rust/moshi-server/src/main.rs` (structs
  like `TtsStreamingQuery`, `AsrStreamingQuery`; `ID_HEADER =
  "kyutai-api-key"`; header-then-query-param auth precedence, checked
  pre-upgrade with HTTP 401, not post-upgrade).
- **Word/EndWord segmentation state machine (STT)**: NOT in `moshi-mlx`'s
  Python package at all (its reference scripts just print raw tokens). Ground
  truth is `moshi-core/src/asr.rs::State::step_tokens`. Two load-bearing
  token-id constants (confirmed independently by that Rust file AND the
  official `stt_from_file_mlx.py` demo's `if text_token not in (0, 3)`):
  `PAD_TOKEN=0`, `WORD_BOUNDARY_TOKEN=3`.
- **TTS word-timing derivation**: real production glue is
  `rust/moshi-server/tts.py` (PyTorch, run via PyO3 from `py_module.rs`) —
  `MaskFlags.WORD_FINISHED` fires at the *start* of a new word, not the end;
  a word's `Text{start_s,stop_s}` event is only emitted once the *next* word
  starts. `moshi_mlx.models.tts.TTSModel`'s `StateMachine`/`State.transcript`
  (list of `(word, step)` pairs) is the one place in the MLX package that
  already tracks this, but nothing in the OSS MLX estate exposes it as an
  event stream — had to derive it (see `tts/engine.py::TtsSession._step`).
- **Only real incremental (word-by-word) MLX TTS driving code**:
  `kyutai-labs/delayed-streams-modeling`, `scripts/tts_mlx_streaming.py`'s
  `TTSGen` dataclass. Not part of the published `moshi_mlx` package API —
  had to vendor/adapt it directly.
- **`_on_text_hook` subtlety**: the hook receives `text_tokens` of shape
  `(batch, 1)`, so a naive `for token in text_tokens.tolist()` gives a
  1-element list, not a scalar int — `StateMachine.process`'s `token not in
  [new_word, pad]` clamp therefore always fires, and `new_word` ends up
  driven by the padding countdown rather than the model spontaneously
  sampling it. Confirmed this is NOT an adaptation bug: identical pattern (no
  `token[0]` unpack) appears in both the official streaming reference script
  AND real moshi-server's own production `tts.py`. Only the *batch*
  `TTSModel.generate()` path unpacks `token[0]` — a different call site.

Full annotated writeup with all of this cross-referenced:
`PROTOCOL.md` in the `unmute-mlx-bridge` repo root.

Pinned compatibility commits used (from the repo's own design spec,
`docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md`):
`kyutai-labs/unmute@c49982e`, `kyutai-labs/moshi@e6a55d2`,
`kyutai-labs/delayed-streams-modeling@4c4f65e`.
