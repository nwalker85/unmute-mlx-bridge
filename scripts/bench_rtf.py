#!/usr/bin/env python3
"""Real-time-factor (RTF) benchmark for `unmute-mlx-bridge`'s TTS engine.

# requires Apple Silicon + weights

Loads the TTS model bundle exactly as `unmute-mlx-tts` does (same env vars,
same `TtsModelBundle.load` / `TtsSession` engine classes), synthesizes a
fixed ~60-second paragraph, and prints the numbers that belong in the
README's "Performance envelope" table: host, commit, quantization, `n_q`,
`cfg_coef`, audio seconds produced, wall-clock seconds taken, the resulting
RTF, and time-to-first-audio.

This intentionally reuses `TtsConfig.from_env()` -- run it with the same
environment variables you'd use to start the real server (`TTS_HF_REPO`,
`TTS_QUANTIZE_BITS`, `TTS_N_Q`, `TTS_CFG_COEF`, ...) to benchmark the exact
profile you plan to deploy.

Usage:
    uv run --locked python scripts/bench_rtf.py
    TTS_QUANTIZE_BITS=8 uv run --locked python scripts/bench_rtf.py

Downloads real weights from Hugging Face on first run (a few GB) if not
already cached. Does not run on portable/Linux CI -- see `pyproject.toml`'s
platform markers on `mlx`/`moshi-mlx` and this repo's `pytest -m hardware`
convention, which this script mirrors in spirit without being a pytest test
itself (it's meant to be run directly and read by a human).
"""

from __future__ import annotations

import subprocess
import sys
import time

BENCH_TEXT = (
    "The history of technology is, in large part, a history of people "
    "learning to trust machines with tasks they once did by hand. Every "
    "generation faces the same question: how much of the work can be "
    "handed over, and how much must a person still watch closely. Early "
    "calculators freed clerks from tedious arithmetic, but nobody expected "
    "them to make judgment calls. Modern systems are different. They are "
    "asked not just to compute, but to converse, to synthesize speech, to "
    "listen and respond in something close to real time. That shift changes "
    "what reliability even means. A calculator that is slow is merely "
    "inconvenient. A voice system that stutters mid-sentence breaks the "
    "illusion of a conversation entirely, and the listener notices "
    "immediately, without needing to understand anything about the "
    "underlying model. That is why raw throughput and perceived "
    "responsiveness are not the same measurement, and why any serious "
    "benchmark has to report both."
)


def _host_string() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown (sysctl unavailable -- not macOS?)"


def _commit_string() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown (not a git checkout?)"


def main() -> int:
    from unmute_mlx_bridge.config import TtsConfig
    from unmute_mlx_bridge.tts.engine import TtsModelBundle, TtsSession

    config = TtsConfig.from_env()

    print(f"host:        {_host_string()}")
    print(f"commit:      {_commit_string()}")
    print(f"hf_repo:     {config.hf_repo}")
    print(f"quantize:    {config.quantize_bits!r} bits")
    print(f"n_q:         {config.n_q}")
    print(f"cfg_coef:    {config.cfg_coef}")
    print(f"text:        {len(BENCH_TEXT.split())} words, {len(BENCH_TEXT)} chars")
    print("loading model bundle (downloads weights on first run)...", file=sys.stderr)

    load_start = time.monotonic()
    bundle = TtsModelBundle.load(
        config.hf_repo,
        voice_repo=config.voice_repo,
        quantize_bits=config.quantize_bits,
        n_q=config.n_q,
        cfg_coef=config.cfg_coef,
    )
    load_elapsed = time.monotonic() - load_start
    print(f"model load:  {load_elapsed:.1f}s", file=sys.stderr)

    session = TtsSession(bundle=bundle, voice=None, max_gen_length=config.max_gen_length)

    audio_samples = 0
    first_audio_s: float | None = None
    wall_start = time.monotonic()

    for event in session.push_text(BENCH_TEXT):
        if event.kind == "audio" and event.pcm:
            if first_audio_s is None:
                first_audio_s = time.monotonic() - wall_start
            audio_samples += len(event.pcm)
    for event in session.push_eos():
        if event.kind == "audio" and event.pcm:
            if first_audio_s is None:
                first_audio_s = time.monotonic() - wall_start
            audio_samples += len(event.pcm)

    wall_elapsed = time.monotonic() - wall_start
    audio_s = audio_samples / 24_000
    rtf = audio_s / wall_elapsed if wall_elapsed else float("inf")

    print(f"audio_s:     {audio_s:.2f}")
    print(f"wall_s:      {wall_elapsed:.2f}")
    print(f"rtf:         {rtf:.3f}x real time")
    print(
        "time_to_first_audio_s: "
        f"{first_audio_s:.2f}" if first_audio_s is not None else "time_to_first_audio_s: n/a"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
