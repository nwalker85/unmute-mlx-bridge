"""Real MLX TTS inference on Apple Silicon: real synthesis, real (non-silent)
audio out, real word-timing events. Evidence for ADR-0007's hardware verification
checklist — see the PR description for the actual numbers observed.
"""

from __future__ import annotations

import math
import time
import wave
from pathlib import Path

import pytest

from unmute_mlx_bridge.tts.engine import TtsModelBundle, TtsSession

pytestmark = pytest.mark.hardware

SAMPLE_RATE = 24_000


@pytest.fixture(scope="module")
def tts_bundle():
    return TtsModelBundle.load("kyutai/tts-1.6b-en_fr")


def _rms(pcm: list[float]) -> float:
    if not pcm:
        return 0.0
    return math.sqrt(sum(x * x for x in pcm) / len(pcm))


def test_synthesizes_non_silent_audio_with_word_timing(tts_bundle, tmp_path: Path):
    session = TtsSession(bundle=tts_bundle, voice=None, max_gen_length=4096)

    text = "The quick brown fox jumps over the lazy dog."
    words: list[tuple[str, float, float]] = []
    pcm_chunks: list[float] = []

    wall_start = time.monotonic()
    for event in session.push_text(text):
        if event.kind == "audio":
            pcm_chunks.extend(event.pcm or [])
        elif event.kind == "word":
            words.append((event.text or "", event.start_s or 0.0, event.stop_s or 0.0))
    for event in session.push_eos():
        if event.kind == "audio":
            pcm_chunks.extend(event.pcm or [])
        elif event.kind == "word":
            words.append((event.text or "", event.start_s or 0.0, event.stop_s or 0.0))
    wall_elapsed = time.monotonic() - wall_start

    audio_duration_s = len(pcm_chunks) / SAMPLE_RATE
    real_time_factor = audio_duration_s / wall_elapsed if wall_elapsed else float("inf")
    print(
        f"\n[tts-hardware] audio={audio_duration_s:.2f}s wall={wall_elapsed:.2f}s "
        f"real_time_factor={real_time_factor:.2f}x words={[w[0] for w in words]}"
    )

    assert pcm_chunks, "expected at least one audio frame"
    assert audio_duration_s > 0.5, "synthesized audio suspiciously short"
    rms = _rms(pcm_chunks)
    print(f"[tts-hardware] rms={rms:.4f}")
    assert rms > 0.001, f"synthesized audio looks silent (rms={rms:.6f})"

    assert words, "expected at least one Text/word timing event"
    got_words = " ".join(w for w, _, _ in words).lower()
    for expected_word in ("quick", "brown", "fox", "dog"):
        assert expected_word in got_words, f"expected {expected_word!r} in {got_words!r}"

    # Word boundaries must be non-decreasing and each stop_s >= its own start_s.
    for _, start_s, stop_s in words:
        assert stop_s >= start_s
    starts = [s for _, s, _ in words]
    assert starts == sorted(starts)

    out_path = tmp_path / "tts_hardware_output.wav"
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        clipped = [max(-1.0, min(1.0, x)) for x in pcm_chunks]
        frames = b"".join(
            int(x * 32767).to_bytes(2, "little", signed=True) for x in clipped
        )
        wav_file.writeframes(frames)
    print(f"[tts-hardware] wrote {out_path}")


def test_second_session_on_same_bundle_does_not_leak_state(tts_bundle):
    """Two sequential sessions on the same process must not carry over transformer
    cache / Mimi streaming state (see `TtsSession.__post_init__`).
    """

    def run_once() -> float:
        session = TtsSession(bundle=tts_bundle, voice=None, max_gen_length=4096)
        pcm: list[float] = []
        for event in session.push_text("Hello there."):
            if event.kind == "audio":
                pcm.extend(event.pcm or [])
        for event in session.push_eos():
            if event.kind == "audio":
                pcm.extend(event.pcm or [])
        return _rms(pcm)

    first_rms = run_once()
    second_rms = run_once()
    assert first_rms > 0.001
    assert second_rms > 0.001
