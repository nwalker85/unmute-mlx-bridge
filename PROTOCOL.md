# PROTOCOL — `moshi-server` STT/TTS wire compatibility

This document is the annotated writeup of the wire protocol this bridge
implements: the WebSocket message schema/framing, the TOML config shape real
`moshi-server` expects, and the version/model-path assumptions it makes. It was
produced by reading upstream source directly (not documentation summaries) at
the commits pinned below, and it is the compatibility oracle for
`src/unmute_mlx_bridge/protocol/` and the two servers.

Everything here traces to a specific file and line range in one of these three
repositories, pinned exactly as in
`docs/design/architecture.md`:

| Repo | Commit | Role |
|---|---|---|
| `kyutai-labs/unmute` | `c49982eb3aeaf76633dfe4155fa3b8dcb5b3d962` | The client whose expectations we must satisfy |
| `kyutai-labs/moshi` | `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` | `moshi-server` (Rust) — ground truth for the wire format; `moshi-mlx` — the inference library we actually run |
| `kyutai-labs/delayed-streams-modeling` | `4c4f65e147df056adf3346290d64c7b9649b18c9` | Reference scripts driving `moshi-mlx` for STT/TTS (`scripts/stt_from_file_mlx.py`, `scripts/tts_mlx_streaming.py`) |

`moshi-mlx==0.3.0` (installed; `pyproject.toml` pins `>=0.2.6`) was verified
directly against its installed source under `site-packages/moshi_mlx/`, not just
against the reference scripts, since the scripts demo simple driving code and
don't themselves implement the full protocol.

## Transport

Both endpoints are plain WebSocket (`ws://`, or `wss://` behind a TLS
terminator — this project doesn't terminate TLS itself). All application
messages are **binary** WebSocket frames containing a single **MessagePack map**
with a string `"type"` tag field — *not* the length-prefixed binary framing
described in `moshi/rust/protocol.md` (that document describes the older
`/api/chat` full-Moshi endpoint's Ogg/Opus framing, a different, unrelated
protocol not used by Unmute's STT/TTS backend WebSockets).

Real `moshi-server` serializes with `rmp_serde`, configured
`.with_human_readable().with_struct_map()` (`rust/moshi-server/src/{asr,tts,py_module}.rs`)
— msgpack maps, not array-packed structs. Unmute's Python client mirrors this
exactly with `msgpack.packb(data, use_bin_type=True, use_single_float=True)` on
the way out (`unmute/stt/speech_to_text.py::_send`,
`unmute/tts/text_to_speech.py::send`) and plain `msgpack.unpackb(...)` on the
way in. This bridge does the same (`src/unmute_mlx_bridge/protocol/wire.py`):
outgoing floats are packed single-precision, which is lossless relative to the
model's own float32 native precision and indistinguishable to any conformant
msgpack client (wire width does not affect the decoded Python `float`).

Auth: the `kyutai-api-key` header (`ID_HEADER` in `main.rs`) is checked first;
if absent, the `auth_id` query parameter is used instead — real moshi-server's
own comment explains why: *"It's tricky to set the headers of a websocket in
javascript so we pass the token via the query too"* (`main.rs::streaming_t`,
`asr_router::streaming_t`). An invalid or missing id gets **HTTP 401 before the
WebSocket upgrade completes** — not a post-upgrade `Error` message. This bridge
replicates that via `websockets`' `process_request` hook
(`src/unmute_mlx_bridge/observability.py::build_process_request`,
`check_auth`). A repeated `?auth_id=a&auth_id=b`, a blank `?auth_id=`, or a
*blank-padded* repeat (`?auth_id=&auth_id=good`) is rejected fail-closed
(401) here rather than silently taking the first value (RAV-1552) — this
matters for TTS specifically, since `tts/server.py::_parse_query` also parses
`auth_id` independently, post-upgrade; both checks treat a repeated or blank
value the same way (invalid) so they can never disagree about a given query
string. The blank-padded case requires parsing with `keep_blank_values=True`
here too — the default `parse_qs` drops the blank half of the pair entirely,
which used to make `?auth_id=&auth_id=good` look like the single value
`good` to this gate.

## `GET /api/asr-streaming` — Speech-to-Text

Endpoint constant: `unmute/kyutai_constants.py::SPEECH_TO_TEXT_PATH`.
Default port: `8090` (`unmute/kyutai_constants.py::STT_SERVER`).

Query params (`main.rs::AsrStreamingQuery`): only `auth_id: Option<String>`.

On connection, before any client message, the server sends:

```json
{"type": "Ready"}
```

In real moshi-server this is not literally instantaneous: the batched server
(`rust/moshi-server/src/batched_asr.rs::handle_socket`) allocates a channel and
internally injects `InMsg::Init` into its own model loop — the *client never
sends `Init` itself* (confirmed by reading `unmute/stt/speech_to_text.py::start_up`,
which just connects and waits for the first message). `Ready` arrives once that
internal `Init` has been processed by one iteration of the model's inference
loop. This bridge reproduces the observable behavior (fresh session, `Ready` sent
once the session's model slot is granted) without needing to accept a
client-sent `Init`.

### Client → server messages

| `type` | Fields | Notes |
|---|---|---|
| `Audio` | `pcm: [f32]` | Mono float PCM @ 24kHz. Unmute always sends exactly `SAMPLES_PER_FRAME = 1920` samples per message (`unmute/kyutai_constants.py`), one Mimi frame at `frame_rate=12.5`Hz (`FRAME_TIME_SEC = 1920/24000 = 0.08s`). The real server still buffers arbitrary-length input into complete 1920-sample frames (`batched_asr.rs::Channel::extend_data`); this bridge does the same (`SttSession.push_audio`) for robustness beyond the happy path. |
| `Marker` | `id: i64` | Round-trip correlation id. Echoed back only once all audio submitted *before* the marker has crossed the ASR delay — see **Marker scheduling** below. |

Real moshi-server's `InMsg` enum (`rust/moshi-core`-adjacent `asr.rs::InMsg`)
also has `Init` and `OggOpus { data: Vec<u8> }` variants. Neither is modeled by
this bridge: `Init` is never client-sent (see above), and `OggOpus` input
(browser-facing Ogg/Opus transport) is out of scope — Unmute's backend STT
client only ever sends raw `Audio` float frames. A client that sends `OggOpus`
gets an explicit `Error`, not silent data loss.

### Server → client messages

| `type` | Fields | Notes |
|---|---|---|
| `Ready` | — | Sent once, on connect. |
| `Word` | `text: str`, `start_time: f64` | One completed word. `start_time` is the *ASR-delay-corrected* time the word started, in seconds. |
| `EndWord` | `stop_time: f64` | Marks the end of the most recently emitted word (a silence/pad boundary was reached). |
| `Marker` | `id: i64` | Echo of a client `Marker`, once due. |
| `Step` | `step_idx: usize`, `prs: [f32]`, `buffered_pcm: usize` | One per processed audio frame. `prs` are the model's "extra head" softmax outputs (`asr.toml`'s `[modules.asr.model.extra_heads]`, `num_heads=4, dim=6`); Unmute uses `prs[2]` as its semantic-VAD pause-probability signal (`unmute/stt/speech_to_text.py::__aiter__`, `self.pause_prediction.update(dt=..., new_value=message.prs[2])`). `buffered_pcm` is a real-server implementation detail Unmute's client ignores (pydantic silently drops unknown fields); this bridge emits it as `0` for wire-shape fidelity. |
| `Error` | `message: str` | Protocol/capacity/model errors. Never includes raw exception text — including a third-party (`mlx`/`moshi_mlx`/`huggingface_hub`) library's own exception text, e.g. an array-broadcast message or a disk-space `OSError` that embeds a local cache path — file/cache paths, hostnames, or other internal detail (RAV-1552 B1/F2). A client-supplied value that was successfully parsed but failed a subsequent validation (a voice name, a numeric `cfg_alpha=`) may be echoed back since the client already has it; a query value that could not even be *parsed* (e.g. `?seed=abc`) instead names only the parameter, never the unparsable raw text (RAV-1552 F1/F6). Anything from the underlying failure itself is logged server-side instead and replaced with a fixed, generic message. |

### Word/EndWord segmentation — the actual state machine

This is **not** exposed by `moshi-mlx`'s Python API (its reference scripts just
print raw decoded tokens). It is a direct port of the real server's Rust
implementation, `moshi-core/src/asr.rs::State::step_tokens` — the ground truth
for how `Word`/`EndWord` boundaries and `start_time`/`stop_time` are computed.
Ported into `SttSession._step` (`src/unmute_mlx_bridge/stt/engine.py`):

- Two token ids are load-bearing protocol constants (confirmed by *both* the
  Rust state machine and the official `stt_from_file_mlx.py` demo script's
  `if text_token not in (0, 3): ...`):
  - `PAD_TOKEN = 0` — silence/pad. Also closes a pending word.
  - `WORD_BOUNDARY_TOKEN = 3` — `existing_text_padding_id` in the downloaded
    model's `config.json`. Also closes a pending word, but only `PAD_TOKEN`
    additionally *ends* it (emits `EndWord`).
- A per-session step counter only starts counting once
  `asr_delay_in_tokens` frames have been processed (`item.step_idx >=
  asr_delay_in_tokens` in Rust); before that, tokens are discarded.
- Accumulate non-boundary tokens into `word_tokens`. On `PAD_TOKEN` or
  `WORD_BOUNDARY_TOKEN`: if `word_tokens` is non-empty, decode it
  (`text_tokenizer.decode(word_tokens)` — matches Rust's
  `decode_piece_ids(&tokens)`, called once **per word**, not accumulated across
  the whole session — this is why word text does not include the sentencepiece
  leading-space marker `▁` and why word boundaries reset segmentation state
  per-word) and emit `Word{text, start_time=last_stop_time}`.
- On `PAD_TOKEN` specifically: compute
  `stop_time = (item_step_idx - asr_delay_in_tokens) / frame_rate`; if a word is
  pending, emit `EndWord{stop_time}`.
- `asr_delay_in_tokens`: real moshi-server's `stt.toml` for
  `kyutai/stt-1b-en_fr-candle` hardcodes `6`. This bridge derives it from the
  downloaded model's `config.json::stt_config.audio_delay_seconds` (`0.5`) times
  `frame_rate` (`12.5`) = `6`, confirming the TOML's literal value, and falls
  back to `6` if that key is absent. Overridable via `STT_ASR_DELAY_IN_TOKENS`.

### Marker scheduling

`batched_asr.rs::pre_process` computes, at the moment a `Marker` is received,
`target_step = current_step_idx + asr_delay_in_tokens + buffered_frame_count`,
then echoes the marker once the model's step counter reaches that target
(`post_process`'s marker `BinaryHeap`). This bridge's `SttSession.push_marker`
schedules `target_item_step = item_step_idx + asr_delay_in_tokens` and the echo
is emitted from `_step()` once reached (`SttSession._pending_markers`).

## `GET /api/tts_streaming` — Text-to-Speech

Endpoint constant: `unmute/kyutai_constants.py::TEXT_TO_SPEECH_PATH`.
Default port: `8089` (`unmute/kyutai_constants.py::TTS_SERVER`).

Query params (`main.rs::TtsStreamingQuery`, `PyStreamingQuery`), with real
server defaults:

| Param | Type | Default | Notes |
|---|---|---|---|
| `seed` | `u64` | `42` | Rejected (`Error`) if negative — this bridge parses it as a signed Python `int`, unlike Rust's unsigned `u64`, so a negative value must be checked explicitly rather than relying on the parse itself to fail (RAV-1552). |
| `temperature` | `f64` | `0.8` | Rejected (`Error`) if non-finite (`nan`/`inf`/`-inf`) or negative (RAV-1552) — `float()` parses all three without raising, so these need their own explicit range check, unlike a plain unparsable value. |
| `top_k` | `usize` | `250` | Rejected (`Error`) if less than `1` (RAV-1552), for the same signed-`int`-vs-unsigned-`usize` reason as `seed`. |
| `format` | enum | `OggOpus` | `Pcm \| PcmMessagePack \| OggOpus \| OggOpusMessagePack`. **This bridge implements `PcmMessagePack` only** — the only format Unmute's own client ever requests (`unmute/tts/text_to_speech.py::TtsStreamingQuery.format = "PcmMessagePack"`). Any other value gets an explicit `Error` and clean close, not silently-wrong framing. A repeated `?format=a&format=b`, a blank `?format=`, or a *blank-padded* repeat (`?format=&format=PcmMessagePack`) are all rejected as an explicit `Error` (RAV-1552) — this field previously took `values[0]` unconditionally, missing the repeated-value guard `seed=`/`top_k=`/`temperature=`/`cfg_alpha=`/`max_seq_len=`/`voice=` already had; the blank-padded case specifically survived a first fix for the plain-repeat case, because the repeat check ran on a parse that drops blank values entirely (see the `voices=` row's note on `keep_blank_values`), so the blank half of the pair silently vanished and left exactly one (non-blank) value behind. |
| `voice` | `str?` | — | Voice file name, e.g. `expresso/ex03-ex01_happy_001_channel1_334s.wav`, resolved against the voice repo (`kyutai/tts-voices` by default). A blank value (`?voice=`) is rejected as an explicit `Error`, not treated as "not given" (RAV-1552 S1). |
| `voices` | `[str]?` | — | Multi-voice blend (up to 5 entries; `moshi_mlx.models.tts.TTSModel.make_condition_attributes` only ever fills 5 speaker slots); mutually exclusive with `voice`. Giving both, an empty list, a blank entry (e.g. `?voices=`, or `?voices=&voices=a`), a duplicate entry (e.g. `?voices=a&voices=a` — burns a blend slot for no effect), more than 5 entries, or a name that fails to resolve is an explicit `Error`, never a silent drop (RAV-1552 S1: query-string parsing keeps blank values so a blank entry reaches this validation instead of being silently dropped, which used to fall back to the single default voice with no signal to the client — originally scoped to just `voice=`/`voices=`, now applied to every query field, RAV-1552). `?voice=&voices=a` (a blank `voice=` alongside a `voices=`) hits the *mutual-exclusivity* error, not the blank-`voice=` one — a `voice=` present on the wire at all, blank or not, counts as "given". **This bridge implements the blend even though the real `Py`/`py_module.rs` module referenced by stock Unmute's `tts.toml` does not** — see the known-deviations list below. |
| `max_seq_len` | `usize?` | — | Rejected (`Error`) if less than `1` when given (RAV-1552), same reasoning as `top_k`. Also rejected (`Error`) if it exceeds the operator's configured `TTS_MAX_GEN_LENGTH` (RAV-1552): a client-supplied `max_seq_len=` may only *lower* the operator's cap for a session, never raise it — previously any positive value replaced the cap unbounded, so `?max_seq_len=1000000000` silently overrode operator capacity planning. |
| `cfg_alpha` | `f64?` | — | Per-request classifier-free-guidance conditioning strength override. When omitted, the session falls back to this bridge's `TTS_CFG_COEF` config default (`2.0`, matching upstream `moshi-server`'s `tts.toml`; see the TOML shape below). Validated against the loaded model's `valid_cfg_conditionings` (`{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}` for `kyutai/tts-1.6b-en_fr`) *before* a channel slot is taken — same fail-fast shape as `format=` — so an unsupported value gets an `Error` and clean close with no `Ready` sent first (RAV-1552 B2; this used to reach `TtsSession` construction and crash the socket with 1011 and no `Error`). No separate positive/finite range check here (unlike `temperature`) — the model's own supported set is the authority. |
| `auth_id` | `str?` | — | See **Auth**, above. A repeated `?auth_id=a&auth_id=b`, a blank `?auth_id=`, or a *blank-padded* repeat (`?auth_id=&auth_id=good`) is rejected both here (post-upgrade, as an explicit `Error`) and at the pre-upgrade auth gate (as HTTP 401) — the two checks are kept in lockstep so which one runs first can never change the outcome (RAV-1552; see **Auth**, above). Previously this field took `values[0]` unconditionally at both sites, and the blank-padded case survived even after that first fix, for the same `keep_blank_values` reason as `format=` above — confirmed live: `?auth_id=&auth_id=good` used to authorize against `{"good"}` at the gate, the exact query the post-upgrade parser already rejected. |

Validation order (each stage fails fast with an `Error` and a clean close, no `Ready` sent first): unparsable, repeated (including blank-padded), blank (`format=`/`auth_id=` only), or out-of-range query values, including `max_seq_len=` above the operator's cap (RAV-1552 F1/F6; RAV-1552) → `format=` → `voice=`/`voices=` structural checks (pure and synchronous, so they run before the model is even loaded) → the model-loading gate (`"model still loading"`/`"model failed to load"`) → `cfg_alpha=` against the *loaded* model's supported set (RAV-1552 B2), which is why `cfg_alpha=` is validated last among these — it needs the model, unlike the others. Every one of these pre-`Ready` rejection paths in `handle_connection` increments `bridge_tts_rejected_sessions_total{reason=...}` (`reason` one of `query`, `format`, `voices`, `loading`, `cfg_alpha`) — see `docs/observability.md`. Each such rejection's `Error` send ignores `ConnectionClosed` (RAV-1552): a client that has already disconnected must not make `handle_connection` raise uncaught — the counter above still records the rejection either way.

On connection, before any client message, the server sends:

```json
{"type": "Ready"}
```

Unlike STT, this is sent as soon as a channel slot is granted
(`py_module.rs::M::channels`, `encoder.encode_msg(OutMsg::Ready)`), with no
model-loop round-trip required.

### Client → server messages

| `type` | Fields | Notes |
|---|---|---|
| `Text` | `text: str` | Appends text to be spoken. |
| `Voice` | `embeddings: [f32]`, `shape: [usize]` | Custom cloned-voice embedding. **Accepted at session start** (before any `Text` message), as an alternative to the `voice` query param — matching real `moshi-server`'s `py_module.rs::InMsg::Voice`. Real `moshi-server` reads a connection's pending voice only on the channel-init entry (`rust/moshi-server/tts.py:340-353`'s `if new_entry[0] == -1:` branch, fed once per channel by `rust/moshi-server/src/py_module.rs:237-240`'s `if !c.sent_init { t.push(-1); ... }`, both pinned at `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`); every later `Text` message never reads `voice` again, so upstream silently drops any `Voice` message received after the first `Text` (RAV-1504). This bridge instead returns an explicit `Error` for a `Voice` message arriving after generation has started — a deliberate, stricter-than-upstream deviation (see below), unobservable by the pinned Unmute client, which only ever uses the `voice=` query parameter and never sends `Voice`. `shape` must account for exactly the flattened `embeddings` length and every dimension must be strictly positive (a non-positive dimension, e.g. `shape=[1, 512, 0]` with `embeddings=[]`, can otherwise make the flattened length accidentally match and reach `reshape`, which infers the missing dimension instead of erroring), or the message is rejected as malformed. |
| `Eos` | — | Signals no more text is coming; flush and finish. |

Real server (`py_module.rs::recv_loop`) also treats a raw **binary `\x00`**
frame as a legacy end-of-stream signal, in addition to the msgpack `Eos`
message. This bridge accepts both for the same reason the real server does.

### Server → client messages

| `type` | Fields | Notes |
|---|---|---|
| `Ready` | — | Sent once, on connect. |
| `Audio` | `pcm: [f32]` | One Mimi frame (1920 samples @ 24kHz) of synthesized audio. |
| `Text` | `text: str`, `start_s: f64`, `stop_s: f64` | Word-level timing. **Finalized one word late**: the real server's `MASK_WORD_FINISHED` "currently indicates the beginning of a new word rather than the end of one" (`rust/moshi-server/tts.py`'s own comment) — a word's `Text` event is only emitted once the *next* word starts being consumed, using that next word's step as `stop_s`. The very first `Text` message a real server emits is always the degenerate `{"type":"Text","text":"","start_s":0,"stop_s":0}` (an artifact of the state machine's initial empty entry); Unmute's client explicitly filters it out and so does not depend on it, and this bridge does not emit it. |
| `Error` | `message: str` | Protocol/capacity/model/unsupported-format errors. Same sanitization policy as the STT `Error` message above (RAV-1552 B1). |

### Word timing derivation

`moshi_mlx.models.tts.TTSModel`'s `StateMachine` is the one place in the OSS MLX
stack that already tracks word consumption: `State.transcript: list[tuple[str,
int]]` grows by one `(word, step)` pair every time `StateMachine.process()`
consumes a new `Entry` from the queue — exactly the "beginning of a new word"
event the real server's `MASK_WORD_FINISHED` flag signals
(`rust/moshi-server/tts.py::TTSService._on_text_hook`,
`self.tts_model.machine.process(...)` → `consumed_new_word`). This bridge
(`TtsSession._step`, `src/unmute_mlx_bridge/tts/engine.py`) watches
`state.transcript` growing after every generation step: when a new entry
appears, the *previous* pending word is finalized as
`Text{text=prev_word, start_s=prev_step/frame_rate, stop_s=new_step/frame_rate}`,
matching `rust/moshi-server/src/tts.rs::Channel::on_end_of_word`'s
`start_s = last_epad_index/12.5; stop_s = step_idx/12.5` exactly (`frame_rate =
12.5`). The trailing pending word is flushed at `Eos`/generation end, using the
final step offset as its `stop_s` — matching the real server calling
`on_end_of_word()` once more right before closing the channel on
`MASK_IS_EOS`.

### Streaming generation loop

`moshi_mlx`'s only reference implementation of incremental (word-by-word, not
batch) TTS generation is `delayed-streams-modeling/scripts/tts_mlx_streaming.py`'s
`TTSGen` dataclass — it is not part of the published `moshi_mlx` package API.
`TtsSession` in this bridge is a direct adaptation of that class: same
`on_text_hook`/`on_audio_hook` wiring into `moshi_mlx.models.generate.LmGen`,
same `process()`/`process_last()` step-budgeting logic, extended with the
word-timing derivation above and with per-connection reset of
`lm.transformer_cache`, `lm.depformer_cache`, and `mimi.reset_all()` (this
bridge admits only one session at a time, so it is safe to reuse the shared
model weights' mutable cache state across sessions as long as it is reset at
session start — mirrors `TTSModel.generate()`'s own cache-reset preamble).

## TOML config shape (real `moshi-server`, for reference)

This bridge does **not** parse or require moshi-server's TOML — Unmute talks to
it purely over the WebSocket URL (`KYUTAI_STT_URL`/`KYUTAI_TTS_URL` env vars),
never by reading its config file. The TOML shape is documented here only
because "does this look like a drop-in replacement for `moshi-server`" requires
knowing what config space it collapses into two `hf_repo` values in this
project.

`services/moshi-server/configs/stt.toml` (from the pinned `unmute` commit):

```toml
static_dir = "./static/"
log_dir = "/tmp/unmute_logs"
instance_name = "tts"
authorized_ids = ["public_token"]

[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"
lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model.safetensors"
text_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"
asr_delay_in_tokens = 6
batch_size = 1
conditioning_learnt_padding = true
temperature = 0.25
# [modules.asr.model], [modules.asr.model.transformer], [modules.asr.model.extra_heads]
# are the full architecture hyperparameters — encoded in the model's own
# config.json, not re-specified by this bridge.
```

`services/moshi-server/configs/tts.toml`:

```toml
[modules.tts_py]
type = "Py"
path = "/api/tts_streaming"
text_tokenizer_file = "hf://kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model"
batch_size = 2
text_bos_token = 1

[modules.tts_py.py]
voice_folder = "hf-snapshot://kyutai/tts-voices/**/*.safetensors"
default_voice = "unmute-prod-website/default_voice.wav"
cfg_coef = 2.0
cfg_is_no_text = true
padding_between = 1
n_q = 24
```

This bridge deliberately uses the same `kyutai/stt-1b-en_fr-candle` checkpoint
as stock `stt.toml`, loading its PyTorch-shaped safetensors through
`moshi_mlx.load_pytorch_weights`. The similarly named
`kyutai/stt-1b-en_fr-mlx` checkpoint can transcribe audio but omits the
`extra_heads` weights that produce `Step.prs`; without those heads stock Unmute
cannot observe the `prs[2]` semantic pause signal and cannot end a user turn.
The TTS default remains `kyutai/tts-1.6b-en_fr`; its
`moshi_name`/`mimi_name`/`tokenizer_name` fields drive
`TtsModelBundle.load` directly.

## Known, documented deviations from real `moshi-server`

1. **Ogg/Opus is not implemented**, for either STT input or TTS output framing.
   Only PCM float32 (`Audio` messages) is supported end to end. Unmute's
   backend-facing client never uses Ogg/Opus for these two endpoints (it's used
   for the *browser*-facing transport, a separate protocol) so this does not
   affect drop-in compatibility with Unmute specifically.
2. **`Voice` (TTS custom cloned-voice embeddings) is supported at session
   start only** (RAV-1504). A client may condition a session on custom
   embeddings before sending any `Text`, matching real `moshi-server`'s
   `py_module.rs::InMsg::Voice`. A `Voice` message received after generation
   has started gets an explicit `Error` — a deliberate, stricter-than-upstream
   deviation: real `moshi-server` only reads the voice on a channel's init
   entry (`rust/moshi-server/tts.py:340-353`, `rust/moshi-server/src/py_module.rs:237-240`,
   pinned at `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`) and
   silently drops any later `Voice` message rather than erroring. This
   difference is unobservable by the pinned Unmute client, which only ever
   sends `voice=`/`voices=` query parameters and never sends a `Voice`
   message at all.
3. **Single session per process.** Real `moshi-server`'s `BatchedAsr`/`Py`
   modules admit multiple concurrent channels (`batch_size` in the TOML). This
   bridge's first release admits exactly one; a second connection gets an
   `Error` and is closed. See design doc §Process Architecture.
4. **`asr_delay_in_tokens` is derived, not hardcoded**, from the downloaded
   model's `config.json` rather than copied from the pinned TOML — see the
   Word/EndWord section above. They agree (`6`) for the current model.
5. **`voices` (TTS multi-voice blend) is a documented superset over the real
   `Py` module stock Unmute actually configures** (RAV-1552). At the pinned
   `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`,
   `rust/moshi-server/src/main.rs::TtsStreamingQuery` declares `voices:
   Option<Vec<String>>` and it round-trips through `PyStreamingQuery`, but
   `py_module.rs`'s own two `channels()` call sites
   (`rust/moshi-server/src/py_module.rs:486`, `:547`) pass only
   `query.voice.clone()` — `query.voices` is parsed and then silently
   discarded, never reaching generation, in the module that stock Unmute's
   `tts.toml` (`[modules.tts_py] type = "Py"`) actually wires up. A *different*
   native module (`rust/moshi-server/src/tts.rs`, `type = "Tts"`, not
   configured by stock Unmute) does fully implement `voices` via
   `voice_ca_src(voice, voices)`, including the same mutual-exclusivity
   validation this bridge now enforces. This bridge implements the blend
   deliberately: `moshi_mlx.models.tts.TTSModel.make_condition_attributes`
   already supports up to 5 voices natively, and this document's `voices` row
   promised the behavior before this deviation was confirmed. Unobservable by
   the pinned Unmute client, which never sends `voices=` itself.
