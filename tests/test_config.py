from __future__ import annotations

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
