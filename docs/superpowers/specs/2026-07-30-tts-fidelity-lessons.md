# Odin TTS fidelity lessons

Date: 2026-07-30  
Status: final word-stream fidelity physically verified; performance experiments ongoing
Scope: `unmute-mlx-bridge` on Odin, Kyutai `tts-1.6b-en_fr`, stock Unmute

## What “slow” meant

The physically observed speech did not stretch or distort individual words.
Each word had natural speed, tone, and voice identity. The generated waveform
contained long pauses between words, making the speaker sound deliberately
halting.

This distinction matters:

- A completed raw WAV removes WebSocket arrival-time gaps.
- Pauses heard in that WAV are synthesized by the model path, not caused by
  network jitter, browser playback, or a slow consumer.
- Measure waveform duration and silence intervals before describing a cadence
  defect as throughput or stuttering.

The broken unquantized bridge WAV was 7.92 seconds with ten detected silence
spans. Kyutai's clean reference WAV was 3.36 seconds with four short, natural
silence spans.

## Proven failure modes

### 1. Four-bit MLX quantization corrupts this TTS model

Kyutai's unchanged pinned MLX reference generator produced gibberish and mixed
voices with 4-bit quantization. The same reference generator, checkpoint,
voice, seed, text, and host produced perfectly intelligible speech
unquantized.

Therefore:

- q4 is not a valid optimization for this model on Odin.
- Correct word metadata and plausible PCM statistics do not prove speech
  quality.
- Test future quantization levels against an untouched upstream generator and
  a physical listening canary before deploying them.

Full-model 8-bit quantization passed intelligibility, cadence, identity,
determinism, and clipping checks. Two identical word-stream runs produced 44
frames / 3.52 seconds, zero clipping, identical PCM, and warmed throughput of
0.594x versus the observed 0.374x unquantized run. It nevertheless failed tonal
parity: Nate heard an EQ-like reduction in low- and mid-frequency richness.

Therefore full-model q8 is also rejected as the deployed setting. Fidelity
includes timbre and spectral body, not just intelligibility. Odin was restored
to unquantized q32 after this canary.

### 2. Reducing generated codebooks violates the autoregressive contract

The earlier q16 experiment improved measured realtime factor but produced
unintelligible speech. Omitted codebook tokens also condition future temporal
steps; zero-filling them changes model state rather than merely lowering audio
fidelity.

Do not treat `n_q` truncation as a safe speed-quality knob unless the upstream
model implementation explicitly supports that generation path.

### 3. `moshi-mlx` 0.3.0 changed text-token shape

The pinned reference script targets `moshi-mlx==0.2.12`, where the streaming
text hook receives scalar batch items. Version 0.3.0 added batching and presents
each item with shape `(1,)`.

Passing that list directly to `StateMachine.process` makes every sampled token
fail the scalar `NEW_WORD`/`PAD` checks. The machine clamps predictions to
padding and advances only after the maximum padding countdown, producing
unnaturally long inter-word pauses.

The 0.3.0 contract is to pass `token[0]` and restore the result with shape
`(batch, 1)`. Dependency upgrades require re-validating tensor shapes at every
adaptation boundary; a comment copied from an older reference is not proof.

### 4. Incomplete delayed frames must not be decoded

Kyutai's reference generator and batch API discard any audio frame containing
the `-1` zero/padding token. The bridge decoded those incomplete frames into
PCM.

The bridge must preserve any word event from that generation step while
skipping audio decode and emission until every codebook contains a real token.

### 5. Streaming chunks continue one speaker turn

Stock Unmute sends many text frames during one assistant response. Each frame
is not a new script turn.

Calling the multi-speaker script tokenizer afresh for every word reinserts the
main-speaker control token on every chunk. Speaker injection belongs only on
the first non-empty text chunk in a TTS session; later chunks continue the same
turn.

After scalar-token handling and incomplete-frame filtering were corrected:

- word-by-word input with repeated speaker markers: 82 frames / 6.56 seconds;
- one-frame input with one speaker marker: 44 frames / 3.52 seconds;
- untouched Kyutai reference: 42 frames / 3.36 seconds.

Nate physically verified the 3.52-second sample as perfect.

## Required validation order

1. Establish an unquantized, full-codebook upstream-reference baseline.
2. Compare the bridge against the same checkpoint, voice, seed, sampler, CFG,
   and text.
3. Save the completed raw PCM WAV and measure duration and silence intervals.
4. Listen to the raw WAV, bypassing browser and realtime playback.
5. Test the real stock-Unmute word stream.
6. Only after fidelity passes, measure generation throughput and test one
   optimization variable at a time.
7. Treat repository tests, deployed SHA, raw-waveform behavior, realtime
   browser behavior, and physical mic/speaker results as separate proof gates.
8. Include tonal richness in the physical gate; clean but spectrally thinned
   speech is a quality regression.

Fenrir is not part of this TTS optimization lane. It remains the dedicated
shared Qwen reasoning/tool-calling inference node.

## Regression contracts

The bridge test suite must prove:

- batched text tokens are unpacked to scalars before entering the state machine;
- delayed frames containing zero/padding codebooks are never decoded;
- only the first text chunk in a session injects a speaker-turn marker;
- query seed, temperature, top-k, CFG conditioning, and maximum sequence length
  reach the session explicitly.
