"""Protocol tests for `/api/asr-streaming` messages.

These pin the exact wire shapes captured from `kyutai-labs/unmute`'s STT client
(`unmute/stt/speech_to_text.py`) and real `moshi-server`'s Rust `OutMsg`/`InMsg`
enums (`rust/moshi-server/src/asr.rs`). A field rename, type change, or missing
tag here means silent drift from what the real client actually sends/expects.
"""

from __future__ import annotations

import msgpack
import pytest
from pydantic import ValidationError

from unmute_mlx_bridge.protocol.stt import (
    SttClientMessageAdapter,
    SttEndWordMessage,
    SttErrorMessage,
    SttMarkerEchoMessage,
    SttReadyMessage,
    SttServerMessageAdapter,
    SttStepMessage,
    SttWordMessage,
)
from unmute_mlx_bridge.protocol.wire import pack_message, unpack_message


def _client_send(data: dict) -> bytes:
    """Exactly what `unmute`'s `SpeechToText._send` does."""
    return msgpack.packb(data, use_bin_type=True, use_single_float=True)


def test_client_audio_message_parses():
    raw = _client_send({"type": "Audio", "pcm": [0.0, 0.5, -0.5]})
    message = SttClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.type == "Audio"
    assert message.pcm == pytest.approx([0.0, 0.5, -0.5])


def test_client_marker_message_parses():
    raw = _client_send({"type": "Marker", "id": 7})
    message = SttClientMessageAdapter.validate_python(unpack_message(raw))
    assert message.type == "Marker"
    assert message.id == 7


def test_client_unknown_type_rejected():
    raw = _client_send({"type": "OggOpus", "data": b"\x00\x01"})
    with pytest.raises(ValidationError):
        SttClientMessageAdapter.validate_python(unpack_message(raw))


def test_client_missing_type_rejected():
    with pytest.raises((ValidationError, KeyError, TypeError)):
        SttClientMessageAdapter.validate_python({"pcm": [0.1]})


@pytest.mark.parametrize(
    "message",
    [
        SttReadyMessage(),
        SttWordMessage(text="hello", start_time=1.5),
        SttEndWordMessage(stop_time=2.0),
        SttMarkerEchoMessage(id=3),
        SttStepMessage(step_idx=10, prs=[0.1, 0.2, 0.3, 0.4]),
        SttErrorMessage(message="boom"),
    ],
)
def test_server_message_roundtrip(message):
    # Outgoing floats are packed single-precision (see `wire.py`), so round-tripped
    # values are approximately, not exactly, equal to the float64 originals. Compare
    # field-by-field since `pytest.approx` does not recurse into list-valued dict
    # entries (e.g. `Step.prs`) in this pytest version.
    packed = pack_message(message)
    decoded = unpack_message(packed)
    assert decoded["type"] == message.type
    back = SttServerMessageAdapter.validate_python(decoded)
    for field_name, expected in message.model_dump().items():
        actual = getattr(back, field_name)
        if isinstance(expected, list):
            assert actual == pytest.approx(expected)
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected


def test_word_message_matches_real_moshi_server_shape():
    """Real server: `OutMsg::Word { text: String, start_time: f64 }`
    (`moshi-core/src/asr.rs`). No other fields.
    """
    packed = pack_message(SttWordMessage(text="bonjour", start_time=0.42))
    decoded = unpack_message(packed)
    assert set(decoded) == {"type", "text", "start_time"}


def test_step_message_includes_buffered_pcm_for_wire_fidelity():
    """Real server's `OutMsg::Step` carries `buffered_pcm: usize`
    (`asr.rs::InMsg`/`OutMsg`) even though Unmute's client ignores it."""
    packed = pack_message(SttStepMessage(step_idx=1, prs=[0.1]))
    decoded = unpack_message(packed)
    assert decoded["buffered_pcm"] == 0
    assert set(decoded) == {"type", "step_idx", "prs", "buffered_pcm"}
