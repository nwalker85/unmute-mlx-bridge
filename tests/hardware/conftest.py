"""Shared fixtures for the opt-in Apple Silicon hardware suite.

Everything here downloads real model weights from Hugging Face and runs real MLX
inference — never run on portable CI (see `AGENTS.md` §Guardrails and
`pyproject.toml`'s `hardware` marker). Invoke explicitly with:

    uv run --locked pytest -m hardware -q
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import wave
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("hardware suite requires Apple Silicon (Darwin/arm64)")


@pytest.fixture(scope="session")
def spoken_fixture(tmp_path_factory) -> tuple[Path, str]:
    """Synthesizes a short WAV with macOS `say`, at a known sample rate, with a
    known transcript — a real, reproducible audio fixture that does not depend on
    checking binary audio into the repository.
    """
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        pytest.skip("macOS `say` and `ffmpeg` are required to synthesize the fixture")

    text = "The quick brown fox jumps over the lazy dog"
    out_dir = tmp_path_factory.mktemp("stt_fixture")
    aiff_path = out_dir / "speech.aiff"
    wav_path = out_dir / "speech.wav"

    subprocess.run(["say", "-o", str(aiff_path), text], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(aiff_path),
            "-ar",
            "24000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path, text


def read_wav_mono_float32(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)
        sample_width = wav_file.getsampwidth()
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM, got {sample_width * 8}-bit")
    import array

    samples = array.array("h")
    samples.frombytes(raw)
    pcm = [s / 32768.0 for s in samples]
    return pcm, sample_rate
