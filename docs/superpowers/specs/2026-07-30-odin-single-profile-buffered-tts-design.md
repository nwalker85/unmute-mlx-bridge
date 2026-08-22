# Odin Single-Profile Buffered TTS Design

## Decision

Run one TTS model process on Odin with one reliability-first deployment
profile:

```text
TTS_N_Q=32
TTS_QUANTIZE_BITS=8
TTS_DELIVERY_MODE=buffered_turn
```

The bridge will preserve every raw text chunk until the client sends `Eos`,
synthesize the complete assistant turn, and release the resulting audio and
word events only after synthesis succeeds.

This is a deliberate latency-for-continuity trade. It does not make Odin
generate faster than real time. It makes a sub-real-time generator usable by
turning intermittent playback underruns into one pre-speech wait followed by
continuous speech.

## Goal

Make the stock-Unmute microphone-to-speaker path produce coherent, stable
speech on Odin now, without a second resident model, channel router, reduced
codebook depth, Fenrir dependency, or frontend fork.

The first release optimizes for:

1. intelligibility;
2. one stable voice identity;
3. natural within-sentence cadence;
4. continuous playback without stutter or digital artifacts;
5. bounded resource use and clean cancellation.

Tonal richness remains a scored quality factor, not a universal hard gate.
That distinction does not preclude later q8 use on a constrained channel such
as G.711 telephony, where the codec may mask the difference, while the audible
low- and mid-frequency trade-off remains recorded for wideband listening.
Telephony quality still requires its own codec-path canary.

## Verified Baseline

The current safe code baseline is commit
`9a1cbf02eb6a6e7e2af5ecfbc7de1d99824e658b`, deployed on Odin with
unquantized weights and all 32 generated codebooks. The live server's
`/readyz` endpoint reports the model loaded and no active session.

The current Forgejo PR #5 head is
`14ed8d64ae96b3e1e69c876a2044f7f2c2923b46`. It is open, mergeable,
unmerged, and differs from the live code only by documentation.

The final unquantized stock-style word stream produced:

- 44 audio frames;
- 3.52 seconds of audio;
- zero clipping;
- deterministic PCM SHA-256
  `f8297f05812152d74143be59fd060ef093146caee83aa7078ec89f0ef83e3f83`;
- physically verified intelligibility, voice identity, cadence, and absence
  of artifacts.

Its observed output rate was only 0.374 seconds of audio per second of wall
time, so immediate playback cannot remain continuous.

Full-model q8 with the same 32 codebooks produced two identical runs:

- 44 audio frames;
- 3.52 seconds of audio;
- zero clipping;
- deterministic PCM SHA-256
  `62605d3213fc7f6380659aa70bf7ac07bcadee509a6a58107415ebd94801fc01`;
- warmed output rate of 0.594 seconds of audio per second of wall time.

Nate verified q8 intelligibility, cadence, identity, and artifact-free output.
He also heard reduced low- and mid-frequency richness. q8 is therefore
acceptable for this reliability-first profile but is not considered
wideband-tonally equivalent to the unquantized model.

## Architecture

### One resident model

Odin will load one q8 model bundle with all 32 generated codebooks. There is
no simultaneous wideband/telephony model pair and no runtime model switching.

`TTS_DELIVERY_MODE` controls scheduling, not model residency:

- `streaming` retains the existing per-chunk behavior for compatibility and
  rollback;
- `buffered_turn` selects the new reliability-first behavior.

Odin will run `buffered_turn`. The default remains `streaming` so the bridge
does not silently change behavior for other installations.

Configuration parsing will reject an unknown delivery mode or a non-positive
buffer limit at process startup.

### Turn input buffer

For `buffered_turn`, `TtsServer` will append each non-empty
`TtsTextMessage.text` value to an ordered list without stripping, inserting,
or normalizing whitespace. Stock Unmute's chunks already include their
required whitespace, so the complete turn is exactly `"".join(chunks)`.

The server will continue reading the WebSocket until it receives:

- `TtsEosMessage`;
- the legacy raw `b"\x00"` end marker;
- client disconnect;
- a protocol or resource-limit error.

Receiving text does not invoke MLX generation in this mode.

### Atomic turn synthesis

At `Eos`, the server will pass the concatenated turn into the existing
`TtsSession` as one speaker turn, then flush with `stream_eos`.

Audio and word events will be collected in generation order in a private
per-connection buffer. No generated event is sent to the client until the
complete turn succeeds. The server then sends the buffered events in their
original order and closes the WebSocket normally.

This preserves:

- the existing speaker-turn marker contract;
- the existing 32-codebook autoregressive path;
- word timestamps and audio/text event ordering;
- seed, temperature, top-k, CFG, voice, and maximum sequence settings;
- the stock `PcmMessagePack` wire format.

### Resource bounds

Buffered delivery needs explicit limits because it retains PCM in memory.
Add two environment-driven limits:

```text
TTS_MAX_BUFFERED_CHARS=4096
TTS_MAX_BUFFERED_AUDIO_SECONDS=60
```

The server maintains a running character count and checks the character limit
before appending each incoming chunk. Exceeding it sends one protocol `Error`
and closes without retaining the rejected chunk or generating audio.

The audio-duration limit is enforced while collecting generated events by
counting PCM samples at the fixed 24 kHz output rate. Exceeding it cancels
generation, discards the partial turn, sends one `Error`, and closes without
audio.

These are per-session buffers. The existing single-session lock remains the
concurrency boundary.

## Data Flow

1. The client connects with its normal query settings.
2. The server sends `Ready` immediately, resolves the voice, and constructs
   the session as today.
3. The client streams raw `Text` messages. The server preserves them without
   starting generation.
4. The client sends `Eos`.
5. The server concatenates the already bounded raw chunks exactly.
6. An empty completed turn closes normally without generating or emitting
   audio.
7. A non-empty turn is synthesized and flushed on the MLX worker thread while
   the asyncio event loop remains available for WebSocket ping/pong and
   disconnect handling.
8. Generated events remain private until the full turn succeeds.
9. The server emits all events in their original order and closes normally.
10. Stock Unmute's existing realtime queue schedules the received audio and
    word events for continuous playback.

## Cancellation And Error Handling

- A disconnect before `Eos` discards buffered text and releases the session.
- A disconnect during generation sets the existing cancellation signal,
  drains the worker safely, discards generated events, and releases the
  session.
- Malformed frames retain the current explicit `Error` behavior.
- Unsupported voice embeddings retain the current explicit rejection.
- Input or output limit violations emit one `Error`, send no partial audio,
  and close.
- A generation exception emits one sanitized `Error`, sends no partial audio,
  records the failure in server logs and metrics, and closes.
- A client cannot receive half a synthesized turn in `buffered_turn` mode.

The existing `streaming` mode is not changed by this work.

## Observability

Keep the existing model-load, active-session, cancellation, protocol-error,
and first-output metrics. Add:

- buffered input characters;
- buffered audio seconds;
- seconds from `Eos` to completed synthesis;
- seconds from `Eos` to first emitted audio;
- buffered-turn failures by reason: input limit, output limit, generation,
  or disconnect.

Time to first audio must continue to measure from the first text message so
the latency cost remains visible rather than being redefined away.

## Validation

### Portable tests

Tests using the fake engine will prove:

- the default delivery mode remains `streaming`;
- unknown delivery modes and non-positive buffer limits fail configuration
  parsing;
- `buffered_turn` preserves raw chunks exactly and invokes generation only
  after `Eos`;
- legacy `b"\x00"` and typed `Eos` take the same path;
- all buffered events are emitted in original order after success;
- no event is emitted when generation raises;
- empty turns close without generation;
- input and output limits fail atomically with one `Error`;
- disconnect before `Eos` discards input;
- disconnect during generation cancels and discards output;
- existing streaming conformance tests remain unchanged and green.

The complete locked portable suite and `git diff --check` must pass.

### Odin controlled canary

Deploy the exact PR commit with q8, 32 codebooks, and `buffered_turn`. Send the
canonical stock-style word stream twice using the production voice and query
settings.

Both runs must:

- return 44 frames and 3.52 seconds of audio;
- match each other byte-for-byte;
- match the established q8 PCM SHA-256
  `62605d3213fc7f6380659aa70bf7ac07bcadee509a6a58107415ebd94801fc01`;
- contain no non-finite or clipped samples;
- emit no partial audio before synthesis completes;
- have no post-start receive gap greater than 160 milliseconds;
- complete from `Eos` to first emitted audio within 10 seconds on a warmed
  Odin process.

Any PCM change blocks the physical canary. Changing the model or sampling
contract requires a separate design rather than weakening this gate.

### Physical stock-Unmute canary

The browser microphone-to-speaker turn must have:

- intelligible words in the expected order;
- one stable speaker identity;
- natural word speed and sentence cadence;
- no gibberish, mixed voices, stutter, underruns, or digital artifacts;
- continuous playback after speech begins.

Record the pre-speech wait and Nate's tonal assessment. The wait is accepted
for this first working profile but must not be omitted from the result.

## Rollout And Rollback

1. Commit and push through Forgejo PR #5.
2. Keep PR #5 open and unmerged.
3. Deploy the exact candidate commit to the isolated Odin canary.
4. Set q8, 32 codebooks, and `buffered_turn`.
5. Run portable, controlled Odin, and physical stock-Unmute gates.
6. If any gate fails, restore exact code commit
   `9a1cbf02eb6a6e7e2af5ecfbc7de1d99824e658b` with unquantized q32.
7. Merge PR #5 only after Nate explicitly approves that PR.

This is a canary configuration, not a production architecture cutover.

## Follow-On Performance Work

Once buffered q8 passes the physical canary, a separate design and plan may
reduce the pre-speech wait within the same single-profile contract. Candidate
work includes avoiding discarded sampler log-probability reductions and
carefully compiling the stateful decode step while capturing cache and random
state.

Those optimizations are not part of this first implementation. Each must
preserve the selected model, all 32 codebooks, deterministic PCM, and the
physical quality gate.

## Out Of Scope

- A second resident TTS model or simultaneous quality tiers.
- Dynamic channel detection or endpoint routing.
- SIP signaling, G.711 encoding, or telephony integration.
- Partial codebook generation.
- q4 or lower-bit model quantization.
- MLX or `moshi-mlx` dependency upgrades.
- Frontend AudioWorklet changes.
- Moving TTS to Fenrir.
- Merging PR #5 without explicit PR-specific approval.
