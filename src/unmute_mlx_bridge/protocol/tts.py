"""TTS (`/api/tts_streaming`) wire messages.

Verified against `kyutai-labs/unmute@c49982eb3aeaf76633dfe4155fa3b8dcb5b3d962`
(`unmute/tts/text_to_speech.py`) and `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`
(`rust/moshi-server/src/py_module.rs`, `rust/moshi-server/src/tts.rs`,
`rust/moshi-server/src/main.rs::TtsStreamingQuery`).

Notes on fidelity to the real server:

- `Voice` (custom cloned-voice embeddings) is accepted **at session start**,
  before any `Text` message, as an alternative to the `voice=`/`voices=` query
  parameters — matching real `moshi-server`'s actual `py_module.rs::InMsg::Voice`
  handling. Real `moshi-server` reads a connection's pending voice only on the
  channel-init entry (`rust/moshi-server/tts.py:340-353`'s
  `if new_entry[0] == -1:` branch, fed once per channel by
  `rust/moshi-server/src/py_module.rs:237-240`'s
  `if !c.sent_init { t.push(-1); ... }`, both pinned at
  `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`); every later
  `Text` message consumes a different, non-init entry that never reads
  `voice`, so a `Voice` message received after the first `Text` is forwarded
  by upstream but silently ignored, never applied. This bridge instead
  rejects a post-session-start `Voice` message with an explicit protocol
  `Error` (a deliberate, stricter-than-upstream deviation — see
  `PROTOCOL.md`'s known-deviations list). An earlier pass of RAV-1504
  misread this as per-chunk, one-shot conditioning applied at every `Text`
  message; that reading was incorrect (upstream never re-reads `voice` after
  the init entry, and `moshi_mlx` caches cross-attention K/V per session, so a
  later per-chunk swap would have been a no-op anyway) and has been corrected
  back to session-start-only; see
  `docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md` for the
  full history.
  `shape` is validated against the flattened `embeddings` payload length here
  (a `ValueError` from this model, surfaced as a protocol `Error` by the
  server), and every dimension must be strictly positive — a shape/payload
  pair like `shape=[1, 512, 0]` with `embeddings=[]` used to pass this length
  check (`0 == len([]) == 0`) and silently reach `reshape`, which infers the
  zero-length dimension instead of erroring, discarding valid conditioning
  without any signal to the client. The model-specific tensor layout
  (dimensionality, multi-speaker support) is validated by
  `tts/engine.py::TtsSession.apply_voice_embedding`.
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

from pydantic import BaseModel, Field, TypeAdapter, model_validator

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

    @model_validator(mode="after")
    def _shape_matches_payload_length(self) -> TtsVoiceMessage:
        """Protocol-level structural checks only: every dimension in `shape`
        must be strictly positive, and `shape` must account for exactly the
        number of values in `embeddings`. Model-specific constraints (tensor
        dimensionality, multi-speaker support) are the engine's concern — see
        `tts/engine.py::TtsSession.apply_voice_embedding`.

        The positivity check runs first and independently of the length
        check: a non-positive dimension (e.g. `shape=[1, 512, 0]` with
        `embeddings=[]`) can make the product of `shape` accidentally match
        `len(embeddings)` (`0 == len([]) == 0`), which would otherwise pass
        silently and let a malformed/empty embedding through to `reshape`,
        which infers the missing dimension instead of erroring.
        """
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(
                f"voice embedding shape {self.shape} must have strictly "
                "positive dimensions"
            )
        expected_values = 1
        for dim in self.shape:
            expected_values *= dim
        if expected_values != len(self.embeddings):
            raise ValueError(
                f"voice embedding shape {self.shape} implies "
                f"{expected_values} values, but {len(self.embeddings)} "
                "were provided"
            )
        return self


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
