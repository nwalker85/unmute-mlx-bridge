"""Protocol tests for `/api/tts_streaming` messages.

Pinned against `kyutai-labs/unmute`'s TTS client (`unmute/tts/text_to_speech.py`)
and real `moshi-server`'s Python TTS module (`rust/moshi-server/src/py_module.rs`,
`rust/moshi-server/src/main.rs::TtsStreamingQuery`).
"""

from __future__ import annotations

import msgpack
import pytest
from pydantic import ValidationError

from unmute_mlx_bridge.protocol.tts import (
    SUPPORTED_FORMAT,
    TtsAudioMessage,
    TtsClientMessageAdapter,
    TtsErrorMessage,
    TtsReadyMessage,
    TtsServerMessageAdapter,
    TtsStreamingQuery,
    TtsTextEventMessage,
)
from unmute_mlx_bridge.protocol.wire import pack_message, unpack_message


def _client_send(data: dict) -> bytes:
    return msgpack.packb(data)


def test_client_text_message_parses():
    raw = _client_send({"type": "Text", "text": "hello there"})
    message = TtsClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.type == "Text"
    assert message.text == "hello there"


def test_client_eos_message_parses():
    raw = _client_send({"type": "Eos"})
    message = TtsClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.type == "Eos"


def test_client_voice_message_parses_but_acceptance_is_a_server_session_state_decision():
    """The message itself is valid protocol — whether the server *applies* it
    (RAV-1504: accepted at session start, rejected once generation has
    started) is a server-side session-state decision, not a parsing concern.
    This model only enforces the structural shape/payload invariant below.
    """
    raw = _client_send({"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 2]})
    message = TtsClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.type == "Voice"
    assert message.shape == [1, 2]


def test_client_voice_message_rejects_shape_payload_length_mismatch():
    """`shape` must account for exactly the number of flattened `embeddings`
    values — this is a pure structural check, independent of any model-specific
    constraint (dimensionality, multi-speaker support), which is the engine's
    concern instead (see `tests/test_tts_engine.py`).
    """
    raw = _client_send({"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 3]})
    with pytest.raises(ValidationError, match="implies 3 values"):
        TtsClientMessageAdapter.validate_python(unpack_message(raw))


def test_client_voice_message_rejects_zero_dimension_with_matching_product():
    """Confirmed defect (RAV-1504 blocker): `shape=[1, 512, 0]` with
    `embeddings=[]` used to pass this check unmodified (the flattened length
    `0` matched `len([]) == 0`) and reach the engine, where `reshape` would
    silently infer the zero-length dimension instead of erroring, discarding
    valid conditioning without any signal to the client. Any non-positive
    dimension is now rejected here regardless of whether the product happens
    to match.
    """
    raw = _client_send({"type": "Voice", "embeddings": [], "shape": [1, 512, 0]})
    with pytest.raises(ValidationError, match="strictly positive"):
        TtsClientMessageAdapter.validate_python(unpack_message(raw))


def test_client_voice_message_rejects_negative_dimension():
    raw = _client_send({"type": "Voice", "embeddings": [0.1] * 6, "shape": [1, -2, 3]})
    with pytest.raises(ValidationError, match="strictly positive"):
        TtsClientMessageAdapter.validate_python(unpack_message(raw))


def test_client_voice_message_accepts_matching_shape_payload():
    raw = _client_send(
        {"type": "Voice", "embeddings": [0.1] * 64_000, "shape": [1, 512, 125]}
    )
    message = TtsClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.shape == [1, 512, 125]
    assert len(message.embeddings) == 64_000


def test_client_unknown_type_rejected():
    with pytest.raises(ValidationError):
        TtsClientMessageAdapter.validate_python({"type": "Bogus"})


@pytest.mark.parametrize(
    "message",
    [
        TtsReadyMessage(),
        TtsAudioMessage(pcm=[0.1, -0.1, 0.2]),
        TtsTextEventMessage(text="hi", start_s=0.0, stop_s=0.5),
        TtsErrorMessage(message="nope"),
    ],
)
def test_server_message_roundtrip(message):
    # See the equivalent STT test for why this compares field-by-field.
    packed = pack_message(message)
    decoded = unpack_message(packed)
    assert decoded["type"] == message.type
    back = TtsServerMessageAdapter.validate_python(decoded)
    for field_name, expected in message.model_dump().items():
        actual = getattr(back, field_name)
        if isinstance(expected, (list, float)):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected


def test_streaming_query_defaults_match_real_moshi_server():
    """Pinned against `main.rs`: `default_seed=42`, `default_temperature=0.8`,
    `default_top_k=250`, `default_format=OggOpus`.
    """
    query = TtsStreamingQuery()
    assert query.seed == 42
    assert query.temperature == 0.8
    assert query.top_k == 250
    assert query.format == "OggOpus"
    assert query.voice is None
    assert query.voices is None


def test_only_pcm_message_pack_format_is_the_supported_target():
    assert SUPPORTED_FORMAT == "PcmMessagePack"
    query = TtsStreamingQuery(format="PcmMessagePack")
    assert query.format == SUPPORTED_FORMAT
