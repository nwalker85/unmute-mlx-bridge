# Odin TTS Codebook Depth Design

## Outcome

Rejected by the physical canary on 2026-07-29. Limiting generation to 16
codebooks improved controlled throughput from 0.77x at 24 codebooks to
1.09-1.13x, but the resulting speech was unintelligible. Nate described it as
"speaking in tongues."

The omitted codebook tokens also condition later temporal-transformer steps.
Supplying zeros for those streams therefore changes future generation rather
than merely removing residual codec detail. The implementation was reverted
and must not be restored without a model-supported partial-generation
contract. Full 32-codebook generation remains the safe runtime behavior.

## Goal

Make `TTS_N_Q` reduce the amount of delayed-stream TTS work performed on
Apple Silicon, then find the lowest codebook depth that produces acceptable
speech while sustaining at least 1.10 seconds of audio per second of wall
time on Odin.

Fenrir is outside this design. It remains a shared inference node for Qwen3.6
and other services.

## Verified Problem

The physical stock-Unmute canary on Odin produced intelligible but severely
broken audio. Backend metrics measured 28.24 seconds of generated audio over
38.239 seconds of generation time, or 0.738x realtime.

Odin is already using High Power mode. It reported no thermal or performance
warnings and had 58 percent free memory during diagnosis.

The bridge passes `TTS_N_Q` into `moshi_mlx.models.tts.TTSModel`, but
`moshi-mlx 0.3.0` only stores that value. `LmGen` still executes every
configured depformer slice, `last_audio_tokens()` returns every generated
codebook, and Mimi receives the full frame. The earlier 24- and 16-codebook
benchmarks therefore did not reduce generation work.

## Design

Add one bridge-owned helper that applies the requested generated-codebook
limit after all checkpoint weights have loaded successfully and before
quantization:

```python
def _limit_generated_codebooks(lm: object, n_q: int) -> None:
    ...
```

The helper will:

1. Read the checkpoint's original generated-codebook count from
   `lm.cfg.depformer.num_slices`.
2. Reject `n_q < 1` or `n_q` greater than the checkpoint count with
   `ValueError`.
3. Truncate `lm.depformer.slices` to the first `n_q` slices.
4. Set `lm.cfg.depformer.num_slices` to `n_q`.

The checkpoint must be instantiated and loaded at its original shape before
truncation so strict weight loading continues to validate every checkpoint
weight. Applying the limit before quantization avoids quantizing slices that
will not be executed.

No `moshi-mlx` package files will be patched. The workaround remains in the
bridge and can be removed when an upstream release implements the advertised
`TTSModel.n_q` behavior.

## Runtime Data Flow

After the limit is applied, existing `LmGen` behavior supplies the rest of the
pipeline:

1. `lm.cfg.depformer.num_slices` makes `LmGen.main_codebooks` equal `n_q`.
2. The bridge creates zero-valued input tokens for the omitted codebooks.
3. The temporal transformer consumes the retained generated streams plus the
   zero-valued omitted streams.
4. The depformer executes only the retained slices.
5. `last_audio_tokens()` returns only those retained codebooks.
6. Mimi decodes that smaller codebook set into PCM.

Protocol framing, word timing, cancellation, ping responsiveness, and the
stock-Unmute backend/frontend remain unchanged.

## Validation And Error Handling

Portable regression tests will prove:

- Requesting 16 codebooks changes both the depformer configuration and the
  executable slice list from 32 to 16.
- Requesting the checkpoint's full depth leaves all slices available.
- Zero, negative, and greater-than-checkpoint values raise `ValueError` before
  mutating the model.

The regression test for the 16-codebook case must be run against the current
implementation first and fail because the helper does not exist.

The complete locked portable suite and `git diff --check` must pass before the
branch is pushed.

## Odin Benchmark Ladder

Deploy the exact updated PR commit to the existing isolated canary and test
q4 quantization at generated-codebook depths 24, 20, and 16. Only the TTS
process may be restarted between runs; the stock-Unmute backend, frontend, and
STT process remain in place.

Each run records:

- generated audio duration;
- wall-clock generation duration and realtime factor;
- time to first audio;
- maximum inter-frame gap after generation begins;
- WAV output for private comparison;
- non-finite, clipping, silence, and duration checks.

The preferred setting is the highest codebook depth that meets all of these
gates:

- sustained realtime factor of at least 1.10x;
- first audio within 1.0 second after the first text input;
- no post-start inter-frame gap greater than 160 milliseconds in the
  controlled benchmark;
- no non-finite samples or sustained clipping;
- acceptable physical microphone/speaker canary to Nate.

If none of the three depths reaches the throughput gate, restore the exact
updated PR commit at 24 codebooks and proceed to a separate optimization
design for MLX 0.32, lower-bit quantization, sampler overhead, and compilation.

## Rollback

Setting `TTS_N_Q=32` restores full checkpoint-depth generation on the updated
bridge. Redeploying commit `d37b6da` restores the pre-change bridge exactly.

PR #5 remains unmerged until portable validation, the Odin benchmark, and the
physical microphone/speaker canary all pass.

## Out Of Scope

- Moving TTS or any other Unmute workload to Fenrir.
- Replacing the stock-Unmute backend, frontend, or websocket protocol.
- Upgrading MLX or `moshi-mlx`.
- Changing sampling temperature or probability filtering.
- Applying `mx.compile`.
- Fixing the separate frontend AudioWorklet underrun bug.
