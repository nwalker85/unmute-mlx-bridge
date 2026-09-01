# Examples

Two small, dependency-light WebSocket clients that talk directly to a running
`unmute-mlx-tts` / `unmute-mlx-stt` server, without going through Unmute
itself. Useful for a quick manual check that a server is actually working, or
as a starting point for writing your own client. Both use only this project's
existing dependencies (`websockets`, `msgpack`, `numpy`) plus the standard
library, so `uv run` from a checkout needs no extra installation.

Start a server first (see the main [README](../README.md#quick-start)):

```bash
uv run unmute-mlx-tts   # ws://127.0.0.1:8089/api/tts_streaming
uv run unmute-mlx-stt   # ws://127.0.0.1:8090/api/asr-streaming
```

## `tts_client.py`

Connects, streams text word-by-word (matching how Unmute's own backend feeds
text), saves the synthesized audio to a WAV file, and prints each word's
timing:

```bash
uv run python examples/tts_client.py --text "Hello there, this is a test." --out hello.wav
```

Run `--help` for the full option list (`--url`, `--voice`, `--cfg-alpha`,
`--auth-id`, `--timeout`).

## `stt_client.py`

Streams a WAV file to the STT server in fixed-size frames, then keeps
streaming silence past a `Marker` message until it's echoed back (the
signal that the model has finished processing everything sent before it —
see [`PROTOCOL.md`](../PROTOCOL.md)), and prints the recognized transcript:

```bash
uv run python examples/stt_client.py --wav hello.wav
```

The input WAV must be mono, 16-bit PCM, 24 kHz (`tts_client.py`'s output
already matches this). Convert anything else first, e.g.:

```bash
ffmpeg -i in.wav -ac 1 -ar 24000 -sample_fmt s16 out.wav
```

Run `--help` for the full option list.
