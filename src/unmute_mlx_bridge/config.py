"""Environment-driven configuration for the STT and TTS servers.

Follows the same `@dataclass(frozen=True)` + `from_env()` convention used by
`puter/services/stt-shim` (`puter_stt_shim/config.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


@dataclass(frozen=True)
class SttConfig:
    """Runtime configuration for `unmute-mlx-stt`.

    Mirrors the shape of `moshi-server`'s `stt.toml` (`[modules.asr]`), but the
    only fields that matter to us are the ones that change model behavior; MLX
    weight loading takes care of the rest via `hf_repo`.
    """

    host: str
    port: int
    hf_repo: str
    quantize_bits: int | None
    max_steps: int
    asr_delay_in_tokens: int | None
    """If unset, derived from the downloaded model's `stt_config.audio_delay_seconds`
    (falls back to 6, matching the real `kyutai/stt-1b-en_fr-candle` `stt.toml`)."""
    authorized_ids: frozenset[str]
    log_level: str

    @classmethod
    def from_env(cls) -> SttConfig:
        authorized_ids = os.getenv("STT_AUTHORIZED_IDS", "public_token")
        return cls(
            host=os.getenv("STT_HOST", "127.0.0.1"),
            port=_env_int("STT_PORT", 8090),
            hf_repo=os.getenv("STT_HF_REPO", "kyutai/stt-1b-en_fr-candle"),
            quantize_bits=_env_optional_int("STT_QUANTIZE_BITS"),
            max_steps=_env_int("STT_MAX_STEPS", 32_768),
            asr_delay_in_tokens=_env_optional_int("STT_ASR_DELAY_IN_TOKENS"),
            authorized_ids=frozenset(
                x.strip() for x in authorized_ids.split(",") if x.strip()
            ),
            log_level=os.getenv("STT_LOG_LEVEL", "INFO"),
        )


@dataclass(frozen=True)
class TtsConfig:
    """Runtime configuration for `unmute-mlx-tts`.

    Mirrors `moshi-server`'s `tts.toml` (`[modules.tts_py]`).
    """

    host: str
    port: int
    hf_repo: str
    voice_repo: str
    default_voice: str
    quantize_bits: int | None
    max_gen_length: int
    authorized_ids: frozenset[str]
    log_level: str

    @classmethod
    def from_env(cls) -> TtsConfig:
        authorized_ids = os.getenv("TTS_AUTHORIZED_IDS", "public_token")
        return cls(
            host=os.getenv("TTS_HOST", "127.0.0.1"),
            port=_env_int("TTS_PORT", 8089),
            hf_repo=os.getenv("TTS_HF_REPO", "kyutai/tts-1.6b-en_fr"),
            voice_repo=os.getenv("TTS_VOICE_REPO", "kyutai/tts-voices"),
            default_voice=os.getenv(
                "TTS_DEFAULT_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav"
            ),
            quantize_bits=_env_optional_int("TTS_QUANTIZE_BITS"),
            max_gen_length=_env_int("TTS_MAX_GEN_LENGTH", 30_000),
            authorized_ids=frozenset(
                x.strip() for x in authorized_ids.split(",") if x.strip()
            ),
            log_level=os.getenv("TTS_LOG_LEVEL", "INFO"),
        )
