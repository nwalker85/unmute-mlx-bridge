from __future__ import annotations

from unmute_mlx_bridge.config import SttConfig


def test_stt_default_model_includes_semantic_vad_heads(monkeypatch):
    monkeypatch.delenv("STT_HF_REPO", raising=False)

    assert SttConfig.from_env().hf_repo == "kyutai/stt-1b-en_fr-candle"


def test_stt_model_can_still_be_overridden(monkeypatch):
    monkeypatch.setenv("STT_HF_REPO", "example/custom-stt")

    assert SttConfig.from_env().hf_repo == "example/custom-stt"
