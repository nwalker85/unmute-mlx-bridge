#!/usr/bin/env python3
"""Minimal STT client: connect to a running `unmute-mlx-stt` server, stream a
WAV file in fixed-size frames, and print recognized words and the Marker
echo.

Dependencies: `websockets`, `msgpack`, `numpy` -- all already project
dependencies of `unmute-mlx-bridge` (see `pyproject.toml`), so `uv run` from a
checkout needs nothing extra. WAV reading uses only the standard-library
`wave` module.

Usage:
    uv run python examples/stt_client.py --wav /path/to/mono16bit24k.wav

The input WAV must be mono, 16-bit PCM, 24 kHz -- the sample rate this
bridge's STT model expects. Convert with e.g.
`ffmpeg -i in.wav -ac 1 -ar 24000 -sample_fmt s16 out.wav` if yours isn't.

A `Marker` sent by the client is only echoed back once audio submitted
*after* it has crossed the model's internal ASR delay -- see PROTOCOL.md --
so this client keeps streaming silence after the real audio and after the
Marker until the echo arrives (or a bound is hit).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave

import msgpack
import numpy as np
import websockets

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1920  # 80 ms at 24 kHz, matching the server's expected chunk size
MARKER_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8090/api/asr-streaming",
        help="STT WebSocket URL (default: %(default)s)",
    )
    parser.add_argument(
        "--wav",
        required=True,
        help="Path to a mono, 16-bit PCM, 24 kHz WAV file to transcribe",
    )
    parser.add_argument(
        "--auth-id",
        default="public_token",
        help="Value for ?auth_id= (default: %(default)s, matches the server's "
        "loopback-dev default)",
    )
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=20,
        help="Silence frames sent after real audio to flush the ASR delay "
        "(default: %(default)s, ~1.6s at the default frame size)",
    )
    parser.add_argument(
        "--marker-wait-frames",
        type=int,
        default=60,
        help="Maximum additional silence frames sent while waiting for the "
        "Marker echo (default: %(default)s, ~4.8s)",
    )
    return parser.parse_args()


def load_pcm(path: str) -> list[float]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2 or framerate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected mono 16-bit PCM at {SAMPLE_RATE} Hz, got "
            f"{channels}ch {sample_width * 8}-bit {framerate} Hz -- convert "
            "first, e.g. `ffmpeg -i in.wav -ac 1 -ar 24000 -sample_fmt s16 out.wav`"
        )

    ints = np.frombuffer(frames, dtype="<i2")
    return (ints.astype(np.float32) / 32767.0).tolist()


async def run(args: argparse.Namespace) -> int:
    pcm = load_pcm(args.wav)
    url = f"{args.url}?auth_id={args.auth_id}"

    words: list[str] = []
    steps = 0
    marker_seen: int | None = None
    errors: list[str] = []
    ready = asyncio.Event()
    done = asyncio.Event()

    print(f"connecting to {url}", file=sys.stderr)

    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:

        async def receive() -> None:
            nonlocal steps, marker_seen
            try:
                async for raw in ws:
                    message = msgpack.unpackb(raw, raw=False)
                    kind = message.get("type")
                    if kind == "Ready":
                        ready.set()
                    elif kind == "Word":
                        text = message.get("text")
                        if text:
                            words.append(text)
                    elif kind == "Step":
                        steps += 1
                    elif kind == "Marker":
                        marker_seen = message.get("id")
                        done.set()
                    elif kind == "Error":
                        errors.append(message.get("message", ""))
                        done.set()
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                done.set()

        receiver = asyncio.create_task(receive())

        try:
            await asyncio.wait_for(ready.wait(), timeout=30)
        except TimeoutError:
            if errors:
                print(f"server rejected the session: {errors[0]}", file=sys.stderr)
            else:
                print("timed out waiting for Ready", file=sys.stderr)
            receiver.cancel()
            return 1

        print(
            f"Ready -- streaming {len(pcm) / SAMPLE_RATE:.2f}s of audio in "
            f"{FRAME_SAMPLES}-sample frames",
            file=sys.stderr,
        )
        num_frames = len(pcm) // FRAME_SAMPLES
        for i in range(num_frames):
            frame = pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            await ws.send(msgpack.packb({"type": "Audio", "pcm": frame}))
            await asyncio.sleep(0.02)

        silence = [0.0] * FRAME_SAMPLES
        for _ in range(args.flush_frames):
            await ws.send(msgpack.packb({"type": "Audio", "pcm": silence}))
            await asyncio.sleep(0.02)

        await ws.send(msgpack.packb({"type": "Marker", "id": MARKER_ID}))

        # Keep feeding silence after the Marker until it's echoed -- a Marker
        # is only echoed once audio submitted *after* it has crossed the
        # model's internal ASR delay (see this file's module docstring).
        for _ in range(args.marker_wait_frames):
            if done.is_set():
                break
            await ws.send(msgpack.packb({"type": "Audio", "pcm": silence}))
            await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(done.wait(), timeout=30)
        except TimeoutError:
            print("timed out waiting for the Marker echo", file=sys.stderr)
        await ws.close()
        await receiver

    if errors:
        print(f"error: {errors[0]}", file=sys.stderr)
        return 1

    transcript = " ".join(words)
    print(f"transcript: {transcript!r}")
    print(f"Step messages received: {steps}")
    if marker_seen == MARKER_ID:
        print(f"Marker {MARKER_ID} echoed back correctly")
        return 0
    print(f"Marker {MARKER_ID} was not echoed (got {marker_seen!r})", file=sys.stderr)
    return 1


def main() -> None:
    args = parse_args()
    try:
        sys.exit(asyncio.run(run(args)))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
