"""STT (`/api/asr-streaming`) wire messages.

Verified against `kyutai-labs/unmute@c49982eb3aeaf76633dfe4155fa3b8dcb5b3d962`
(`unmute/stt/speech_to_text.py`) and `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`
(`rust/moshi-server/src/asr.rs`, `rust/moshi-server/src/batched_asr.rs`).

Notes on fidelity to the real server:

- Real `moshi-server` also accepts a client `Init` message (used to reset a batch
  slot) and an `OggOpus` audio variant. Unmute's own STT client never sends either
  one — audio always arrives as `Audio{pcm}` float frames, and `Init` is injected
  server-side right after a channel is allocated (see `batched_asr.rs::handle_socket`,
  `in_tx.send(InMsg::Init)`). We reproduce that server-side behavior (a session is
  always fresh and sends `Ready` once its model slot is granted) without needing to
  accept a client-sent `Init`, and we do not implement Ogg/Opus decoding — both are
  out of scope for a project whose only consumer is Unmute's PCM-only client. A
  client that sends `OggOpus` gets an explicit `Error`, not silent data loss.
- `OutMsg::Step` on the real server carries a `buffered_pcm: usize` field the
  Unmute client ignores (pydantic silently drops unknown fields on the way in,
  so this cannot break compatibility either direction). We emit it for byte-shape
  fidelity.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

# ---------------------------------------------------------------------------
# Client -> server
# ---------------------------------------------------------------------------


class SttAudioMessage(BaseModel):
    type: Literal["Audio"] = "Audio"
    pcm: list[float]


class SttMarkerMessage(BaseModel):
    type: Literal["Marker"] = "Marker"
    id: int


SttClientMessage = Annotated[
    SttAudioMessage | SttMarkerMessage,
    Field(discriminator="type"),
]
SttClientMessageAdapter: TypeAdapter[SttClientMessage] = TypeAdapter(SttClientMessage)

# ---------------------------------------------------------------------------
# Server -> client
# ---------------------------------------------------------------------------


class SttReadyMessage(BaseModel):
    type: Literal["Ready"] = "Ready"


class SttWordMessage(BaseModel):
    type: Literal["Word"] = "Word"
    text: str
    start_time: float


class SttEndWordMessage(BaseModel):
    type: Literal["EndWord"] = "EndWord"
    stop_time: float


class SttMarkerEchoMessage(BaseModel):
    type: Literal["Marker"] = "Marker"
    id: int


class SttStepMessage(BaseModel):
    type: Literal["Step"] = "Step"
    step_idx: int
    prs: list[float]
    buffered_pcm: int = 0


class SttErrorMessage(BaseModel):
    type: Literal["Error"] = "Error"
    message: str


SttServerMessage = Annotated[
    SttReadyMessage
    | SttWordMessage
    | SttEndWordMessage
    | SttMarkerEchoMessage
    | SttStepMessage
    | SttErrorMessage,
    Field(discriminator="type"),
]
SttServerMessageAdapter: TypeAdapter[SttServerMessage] = TypeAdapter(SttServerMessage)
