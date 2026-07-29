"""unmute-mlx-bridge — Apple Silicon MLX STT/TTS bridge for Kyutai Unmute.

Two long-lived processes implement `moshi-server`'s WebSocket contract on top of
real MLX inference:

- `unmute-mlx-stt` — `/api/asr-streaming`, see `unmute_mlx_bridge.stt.server`.
- `unmute-mlx-tts` — `/api/tts_streaming`, see `unmute_mlx_bridge.tts.server`.

See `PROTOCOL.md` at the repository root for the wire-format writeup.
"""


def main() -> None:
    """Generic entry point; prints usage. Use `unmute-mlx-stt` / `unmute-mlx-tts`
    to actually run a server.
    """
    print(
        "unmute-mlx-bridge: run `unmute-mlx-stt` or `unmute-mlx-tts` to start a server"
    )
