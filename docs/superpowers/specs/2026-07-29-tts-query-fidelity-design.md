# TTS Query Fidelity Design

## Goal

Restore the deterministic, voice-stable stock-Unmute TTS baseline before any
further Odin performance optimization.

## Verified Gap

The stock client connects with a fixed voice and `cfg_alpha=1.5`. The bridge
parses that value plus the moshi-server defaults `seed=42`,
`temperature=0.8`, and `top_k=250`, but only uses the voice name.

Generation currently uses:

- process-global RNG state without a per-session seed;
- hard-coded temperature `0.6`;
- the sampler's default top-p filtering;
- model-load-time CFG conditioning `1.0`.

The deployed model explicitly supports CFG conditioning values from `1.0`
through `4.0` in increments of `0.5`, including the requested `1.5`.

## Design

Extend `TtsSession` with the parsed per-connection values:

```python
seed: int
temperature: float
top_k: int
cfg_alpha: float | None
```

`TtsServer` will pass `query.seed`, `query.temperature`, `query.top_k`, and
`query.cfg_alpha` into each session. It will also honor `query.max_seq_len`
when present, otherwise retaining the configured maximum.

During session construction:

1. Reset all transformer, depformer, and Mimi streaming state as today.
2. Call `mx.random.seed(seed)` while the connection owns the single-session
   lock.
3. Use the requested CFG conditioning when present; otherwise use the model's
   existing default conditioning.
4. Reject unsupported CFG values before generation.
5. Construct both text and audio samplers with the requested temperature and
   top-k value.

The model weights, q4 quantization, 32 generated codebooks, voice embedding,
websocket framing, and frontend remain unchanged.

## Validation

Portable tests will prove that:

- the server forwards all parsed query values to the session;
- omitted query values resolve to seed 42, temperature 0.8, and top-k 250;
- `max_seq_len` overrides only the per-session generation limit;
- session construction seeds MLX and creates text/audio samplers with the
  requested settings;
- CFG 1.5 reaches `make_condition_attributes`;
- an unsupported CFG value raises before generation.

After the locked portable suite passes, deploy the exact PR commit at full
q32/q4. Generate the same sentence twice with the same voice and query. The
PCM hashes must match, the raw WAV must remain intelligible, and only then may
the browser microphone/speaker canary proceed.

## Rollback

Redeploy PR head `086e35e137cc6d68c4eb038c793e1ebc0a587df1` with
`TTS_N_Q=32` and `TTS_QUANTIZE_BITS=4`.

## Out Of Scope

- MLX dependency changes.
- Lower-bit quantization.
- Partial codebook generation.
- Frontend buffering changes.
- Fenrir.
