#!/usr/bin/env python3
"""Minimal TTS client: connect to a running `unmute-mlx-tts` server, stream
text word-by-word (matching how Unmute's own backend feeds text), save the
synthesized audio to a WAV file, and print each word's timing.

Dependencies: `websockets`, `msgpack`, `numpy` -- all already project
dependencies of `unmute-mlx-bridge` (see `pyproject.toml`), so `uv run` from a
checkout needs nothing extra.

Usage:
    uv run python examples/tts_client.py --text "Hello there, this is a test."
    uv run python examples/tts_client.py --url ws://127.0.0.1:8089/api/tts_streaming \\
        --voice unmute-prod-website/default_voice.wav --out out.wav

See PROTOCOL.md for the full wire-format reference this client implements a
small slice of.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8089/api/tts_streaming",
        help="TTS WebSocket URL (default: %(default)s)",
    )
    parser.add_argument(
        "--text",
        default="The weather is lovely today. I think we should go for a walk later.",
        help="Text to synthesize (default: a short two-sentence sample)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice path, e.g. unmute-prod-website/default_voice.wav "
        "(default: server's TTS_DEFAULT_VOICE)",
    )
    parser.add_argument(
        "--auth-id",
        default="public_token",
        help="Value for ?auth_id= (default: %(default)s, matches the server's "
        "loopback-dev default)",
    )
    parser.add_argument(
        "--cfg-alpha",
        type=float,
        default=None,
        help="Per-session classifier-free-guidance override (default: server's "
        "TTS_CFG_COEF)",
    )
    parser.add_argument(
        "--out",
        default="tts_output.wav",
        help="Output WAV path (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Overall timeout in seconds waiting for synthesis to finish "
        "(default: %(default)s)",
    )
    return parser.parse_args()


def build_url(args: argparse.Namespace) -> str:
    params = ["format=PcmMessagePack", f"auth_id={args.auth_id}"]
    if args.voice:
        params.append(f"voice={args.voice}")
    if args.cfg_alpha is not None:
        params.append(f"cfg_alpha={args.cfg_alpha}")
    separator = "&" if "?" in args.url else "?"
    return f"{args.url}{separator}{'&'.join(params)}"


def save_wav(path: str, pcm: list[float]) -> None:
    samples = np.clip(np.array(pcm, dtype=np.float32), -1.0, 1.0)
    ints = (samples * 32767).astype("<i2")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(ints.tobytes())


async def run(args: argparse.Namespace) -> int:
    url = build_url(args)
    pcm: list[float] = []
    words: list[tuple[str, float, float]] = []
    errors: list[str] = []
    ready = asyncio.Event()
    done = asyncio.Event()

    print(f"connecting to {url}", file=sys.stderr)

    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:

        async def receive() -> None:
            try:
                async for raw in ws:
                    message = msgpack.unpackb(raw, raw=False)
                    kind = message.get("type")
                    if kind == "Ready":
                        ready.set()
                    elif kind == "Audio":
                        pcm.extend(message.get("pcm") or [])
                    elif kind == "Text":
                        words.append(
                            (message["text"], message["start_s"], message["stop_s"])
                        )
                    elif kind == "Error":
                        errors.append(message.get("message", ""))
                        done.set()
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                done.set()

        receiver = asyncio.create_task(receive())

        try:
            await asyncio.wait_for(ready.wait(), timeout=args.timeout)
        except TimeoutError:
            if errors:
                print(f"server rejected the session: {errors[0]}", file=sys.stderr)
            else:
                print("timed out waiting for Ready", file=sys.stderr)
            receiver.cancel()
            return 1

        print(f"Ready -- streaming {len(args.text.split())} words", file=sys.stderr)
        for word in args.text.split():
            await ws.send(msgpack.packb({"type": "Text", "text": word + " "}))
            await asyncio.sleep(0.01)
        await ws.send(msgpack.packb({"type": "Eos"}))

        try:
            await asyncio.wait_for(done.wait(), timeout=args.timeout)
        except TimeoutError:
            print("timed out waiting for synthesis to finish", file=sys.stderr)
        await receiver

    if errors:
        print(f"error: {errors[0]}", file=sys.stderr)
        return 1
    if not pcm:
        print("no audio received", file=sys.stderr)
        return 1

    save_wav(args.out, pcm)
    duration_s = len(pcm) / SAMPLE_RATE
    print(f"wrote {args.out} ({duration_s:.2f}s audio, {len(words)} word timings)")
    for text, start_s, stop_s in words:
        print(f"  {start_s:6.2f}s - {stop_s:6.2f}s  {text!r}")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
