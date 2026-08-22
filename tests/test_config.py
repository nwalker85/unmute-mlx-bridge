from __future__ import annotations

import pytest

from unmute_mlx_bridge.config import SttConfig, TtsConfig


def test_stt_default_model_includes_semantic_vad_heads(monkeypatch):
    monkeypatch.delenv("STT_HF_REPO", raising=False)

    assert SttConfig.from_env().hf_repo == "kyutai/stt-1b-en_fr-candle"


def test_stt_model_can_still_be_overridden(monkeypatch):
    monkeypatch.setenv("STT_HF_REPO", "example/custom-stt")

    assert SttConfig.from_env().hf_repo == "example/custom-stt"


def test_tts_defaults_to_stock_unmute_codebook_depth(monkeypatch):
    monkeypatch.delenv("TTS_N_Q", raising=False)

    assert TtsConfig.from_env().n_q == 24


def test_tts_codebook_depth_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TTS_N_Q", "16")

    assert TtsConfig.from_env().n_q == 16


def test_tts_buffered_delivery_defaults(monkeypatch):
    for name in (
        "TTS_DELIVERY_MODE",
        "TTS_MAX_BUFFERED_CHARS",
        "TTS_MAX_BUFFERED_AUDIO_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = TtsConfig.from_env()

    assert config.delivery_mode == "streaming"
    assert config.max_buffered_chars == 4096
    assert config.max_buffered_audio_seconds == 60.0


def test_tts_buffered_delivery_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TTS_DELIVERY_MODE", "buffered_turn")
    monkeypatch.setenv("TTS_MAX_BUFFERED_CHARS", "2048")
    monkeypatch.setenv("TTS_MAX_BUFFERED_AUDIO_SECONDS", "30.5")

    config = TtsConfig.from_env()

    assert config.delivery_mode == "buffered_turn"
    assert config.max_buffered_chars == 2048
    assert config.max_buffered_audio_seconds == 30.5


def test_tts_rejects_unknown_delivery_mode(monkeypatch):
    monkeypatch.setenv("TTS_DELIVERY_MODE", "burst")

    with pytest.raises(ValueError, match="TTS_DELIVERY_MODE"):
        TtsConfig.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TTS_MAX_BUFFERED_CHARS", "0"),
        ("TTS_MAX_BUFFERED_CHARS", "-1"),
        ("TTS_MAX_BUFFERED_AUDIO_SECONDS", "0"),
        ("TTS_MAX_BUFFERED_AUDIO_SECONDS", "-0.5"),
    ],
)
def test_tts_rejects_non_positive_buffer_limits(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        TtsConfig.from_env()
