"""Wire-protocol models for the moshi-server-compatible STT/TTS WebSocket contracts.

These models are the compatibility oracle for this project: they encode the exact
message shapes emitted and consumed by `kyutai-labs/unmute`'s speech-services client
(`unmute/stt/speech_to_text.py`, `unmute/tts/text_to_speech.py`) and by the real
`moshi-server` binary (`kyutai-labs/moshi`, `rust/moshi-server/src/{asr,batched_asr,py_module}.rs`).

See `PROTOCOL.md` at the repository root for the annotated wire-format writeup and
the exact upstream commits this was verified against.
"""

from unmute_mlx_bridge.protocol.wire import pack_message, unpack_message

__all__ = ["pack_message", "unpack_message"]
