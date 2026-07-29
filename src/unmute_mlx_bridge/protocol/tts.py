"""TTS (`/api/tts_streaming`) wire messages.

Verified against `kyutai-labs/unmute@c49982eb3aeaf76633dfe4155fa3b8dcb5b3d962`
(`unmute/tts/text_to_speech.py`) and `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`
(`rust/moshi-server/src/py_module.rs`, `rust/moshi-server/src/tts.rs`,
`rust/moshi-server/src/main.rs::TtsStreamingQuery`).

Notes on fidelity to the real server:

- `Voice` (custom cloned-voice embeddings) is accepted for shape-parsing but
  rejected with an explicit `Error`, matching the design decision in
  `docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md`: this is a
  documented non-goal, not a silently-ignored field.
- The real server's `py_module.rs::recv_loop` also treats a raw binary `\\x00`
  frame as a legacy end-of-stream signal, in addition to the msgpack
  `{"type": "Eos"}` message. We accept both for the same reason the real server
  does: cheap, and some client generations may still send it.
- `TtsStreamingQuery` mirrors `main.rs::TtsStreamingQuery` field-for-field,
  including its exact defaults (`seed=42`, `temperature=0.8`, `top_k=250`,
  `format=OggOpus`). Unmute always overrides `format=PcmMessagePack` explicitly, and
  that is the only format this bridge implements — anything else gets an `Error`
  and a clean close rather than pretending to support Ogg/Opus framing.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

# ---------------------------------------------------------------------------
# Query parameters (GET /api/tts_streaming?...)
# ---------------------------------------------------------------------------

SUPPORTED_FORMAT = "PcmMessagePack"


class TtsStreamingQuery(BaseModel):
    seed: int = 42
    temperature: float = 0.8
    top_k: int = 250
    format: str = "OggOpus"
    voice: str | None = None
    voices: list[str] | None = None
    max_seq_len: int | None = None
    cfg_alpha: float | None = None
    auth_id: str | None = None


# ---------------------------------------------------------------------------
# Client -> server
# ---------------------------------------------------------------------------


class TtsTextMessage(BaseModel):
    type: Literal["Text"] = "Text"
    text: str


class TtsVoiceMessage(BaseModel):
    type: Literal["Voice"] = "Voice"
    embeddings: list[float]
    shape: list[int]


class TtsEosMessage(BaseModel):
    type: Literal["Eos"] = "Eos"


TtsClientMessage = Annotated[
    TtsTextMessage | TtsVoiceMessage | TtsEosMessage,
    Field(discriminator="type"),
]
TtsClientMessageAdapter: TypeAdapter[TtsClientMessage] = TypeAdapter(TtsClientMessage)

# ---------------------------------------------------------------------------
# Server -> client
# ---------------------------------------------------------------------------


class TtsReadyMessage(BaseModel):
    type: Literal["Ready"] = "Ready"


class TtsAudioMessage(BaseModel):
    type: Literal["Audio"] = "Audio"
    pcm: list[float]


class TtsTextEventMessage(BaseModel):
    type: Literal["Text"] = "Text"
    text: str
    start_s: float
    stop_s: float


class TtsErrorMessage(BaseModel):
    type: Literal["Error"] = "Error"
    message: str


TtsServerMessage = Annotated[
    TtsReadyMessage | TtsAudioMessage | TtsTextEventMessage | TtsErrorMessage,
    Field(discriminator="type"),
]
TtsServerMessageAdapter: TypeAdapter[TtsServerMessage] = TypeAdapter(TtsServerMessage)
