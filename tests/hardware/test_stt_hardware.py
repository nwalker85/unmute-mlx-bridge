"""Real MLX STT inference on Apple Silicon: real weights, real audio, real
transcription. This is the evidence for ADR-0007's verification-checklist items
"Kyutai MLX builds run in real time" and hardware confirmation — see the PR
description for the actual numbers observed on the reference host.
"""

from __future__ import annotations

import array
import time
import wave
from pathlib import Path

import pytest

from unmute_mlx_bridge.stt.engine import FRAME_SIZE, SAMPLE_RATE, SttModelBundle, SttSession

pytestmark = pytest.mark.hardware


def _read_wav_mono_float32(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)
        sample_width = wav_file.getsampwidth()
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM, got {sample_width * 8}-bit")
    samples = array.array("h")
    samples.frombytes(raw)
    return [s / 32768.0 for s in samples], sample_rate


@pytest.fixture(scope="module")
def stt_bundle():
    return SttModelBundle.load("kyutai/stt-1b-en_fr-candle")


def test_transcribes_known_sentence(stt_bundle, spoken_fixture):
    wav_path, expected_text = spoken_fixture
    pcm, sample_rate = _read_wav_mono_float32(wav_path)
    assert sample_rate == SAMPLE_RATE
    speech_sample_count = len(pcm)

    session = SttSession(bundle=stt_bundle, max_steps=4096)

    # Pad with 1s of trailing silence so the delayed ASR head flushes the final
    # word, matching the reference script's own padding convention.
    pcm = pcm + [0.0] * SAMPLE_RATE

    words: list[str] = []
    start_times: list[float] = []
    silence_pause_probs: list[float] = []
    step_count = 0
    wall_start = time.monotonic()
    for offset in range(0, len(pcm) - FRAME_SIZE + 1, FRAME_SIZE):
        frame = pcm[offset : offset + FRAME_SIZE]
        for event in session.push_audio(frame):
            step_count += 1
            if (
                event.kind == "step"
                and event.prs is not None
                and len(event.prs) > 2
                and offset >= speech_sample_count
            ):
                silence_pause_probs.append(event.prs[2])
            if event.kind == "word":
                words.append((event.text or "").strip())
                start_times.append(event.start_time or 0.0)
    wall_elapsed = time.monotonic() - wall_start

    audio_duration_s = len(pcm) / SAMPLE_RATE
    real_time_factor = audio_duration_s / wall_elapsed if wall_elapsed else float("inf")
    print(
        f"\n[stt-hardware] audio={audio_duration_s:.2f}s wall={wall_elapsed:.2f}s "
        f"real_time_factor={real_time_factor:.2f}x steps={step_count}"
    )

    transcript = " ".join(w for w in words if w)
    print(f"[stt-hardware] expected={expected_text!r} got={transcript!r}")

    assert transcript, "expected at least one transcribed word"
    got_lower = transcript.lower()
    for expected_word in ("quick", "brown", "fox", "lazy", "dog"):
        assert expected_word in got_lower, (
            f"expected {expected_word!r} in transcript, got {transcript!r}"
        )

    # Stock Unmute uses prs[2] as its semantic pause prediction and waits for
    # it to cross 0.6 before ending a user turn. The MLX-only checkpoint omits
    # the extra heads entirely; the Candle checkpoint includes them and is
    # still executed by this bridge through moshi-mlx.
    assert silence_pause_probs, "expected the semantic VAD head on silence frames"
    assert max(silence_pause_probs) > 0.6

    # start_time must be non-decreasing across words (monotonic ASR timeline).
    assert start_times == sorted(start_times)


def test_second_session_on_same_bundle_does_not_leak_state(stt_bundle, spoken_fixture):
    """Two sequential sessions on the same process must not carry over transformer
    cache / Mimi streaming state (see `SttSession.__post_init__`).
    """
    wav_path, _expected_text = spoken_fixture
    pcm, _sample_rate = _read_wav_mono_float32(wav_path)
    pcm = pcm + [0.0] * SAMPLE_RATE

    def run_once() -> str:
        session = SttSession(bundle=stt_bundle, max_steps=4096)
        words = []
        for offset in range(0, len(pcm) - FRAME_SIZE + 1, FRAME_SIZE):
            for event in session.push_audio(pcm[offset : offset + FRAME_SIZE]):
                if event.kind == "word":
                    words.append((event.text or "").strip())
        return " ".join(w for w in words if w).lower()

    first = run_once()
    second = run_once()
    assert "fox" in first
    assert "fox" in second
