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


def test_client_voice_message_parses_but_is_a_server_policy_decision():
    """The message itself is valid protocol — whether the server accepts custom
    voice embeddings is a server-side policy choice (this bridge rejects it), not
    a parsing concern.
    """
    raw = _client_send({"type": "Voice", "embeddings": [0.1, 0.2], "shape": [1, 2]})
    message = TtsClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.type == "Voice"
    assert message.shape == [1, 2]


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
