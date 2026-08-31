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


def test_tts_cfg_coef_defaults_to_upstream_toml_value(monkeypatch):
    """Upstream `moshi-server`'s `tts.toml` ships `cfg_coef = 2.0`
    (`[modules.tts_py.py]`); this bridge's engine used to hardcode `1.0`
    instead, which renders voices nearly flat (RAV-1552)."""
    monkeypatch.delenv("TTS_CFG_COEF", raising=False)

    assert TtsConfig.from_env().cfg_coef == 2.0


def test_tts_cfg_coef_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TTS_CFG_COEF", "1.5")

    assert TtsConfig.from_env().cfg_coef == 1.5


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "-0.5", "nan", "inf", "-inf"],
)
def test_tts_cfg_coef_rejects_non_positive_or_non_finite(monkeypatch, value):
    """RAV-1552 B4: `TTS_CFG_COEF` must be a positive, finite number --
    checked at config-parse time, before any Hugging Face download."""
    monkeypatch.setenv("TTS_CFG_COEF", value)

    with pytest.raises(ValueError, match="TTS_CFG_COEF"):
        TtsConfig.from_env()


def test_tts_cfg_coef_rejects_unparseable_value_and_names_the_variable(monkeypatch):
    """RAV-1552 B4: `float("abc")` on its own never names the offending
    environment variable; `_env_float` must attach it."""
    monkeypatch.setenv("TTS_CFG_COEF", "not-a-number")

    with pytest.raises(ValueError, match="TTS_CFG_COEF"):
        TtsConfig.from_env()


def test_tts_max_buffered_audio_seconds_unparseable_value_names_the_variable(monkeypatch):
    """Same `_env_float` fix, exercised through the other float-typed env var
    that uses it."""
    monkeypatch.setenv("TTS_MAX_BUFFERED_AUDIO_SECONDS", "not-a-number")

    with pytest.raises(ValueError, match="TTS_MAX_BUFFERED_AUDIO_SECONDS"):
        TtsConfig.from_env()
