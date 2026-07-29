"""Real MLX inference for streaming speech-to-text.

Model loading mirrors `kyutai-labs/delayed-streams-modeling`'s reference
`scripts/stt_from_file_mlx.py` (pinned at
`4c4f65e147df056adf3346290d64c7b9649b18c9`): download weights via
`huggingface_hub`, build `moshi_mlx.models.Lm` + `Mimi` audio tokenizer, and step
them frame-by-frame with `models.LmGen.step_with_extra_heads`.

The word/end-of-word segmentation state machine below is *not* present in
`moshi_mlx` (that reference script just prints raw tokens). It is a direct port of
the real `moshi-server`'s Rust implementation in `moshi-core/src/asr.rs`
(`State::step_tokens`), which is the ground truth for how `Word`/`EndWord`
boundaries, `start_time`/`stop_time`, and the `asr_delay_in_tokens` gate are
computed. Two token ids are load-bearing constants from that state machine:

- `PAD_TOKEN = 0`: silence / pad. Also closes a pending word (`EndWord`).
- `WORD_BOUNDARY_TOKEN = 3`: `existing_text_padding_id` in the downloaded model's
  `config.json`; also closes a pending word but does not, by itself, end it.

Both are confirmed independently by `moshi-core/src/asr.rs` (`token == 3 ||
token == 0`) and by the official demo script
(`if text_token not in (0, 3): ...`), so they are treated as fixed protocol
constants rather than something to read out of a config file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sentencepiece
from huggingface_hub import hf_hub_download

# `mlx`/`moshi-mlx` only ship wheels for Apple Silicon (plus a partial, older-pinned
# Linux x86_64 build that still requires the macOS-only `mlx-metal` backend package).
# Importing them at module level would break `import unmute_mlx_bridge.stt.engine`
# (and therefore `unmute_mlx_bridge.stt.server`, and therefore the portable
# protocol/conformance test suite) on GitHub-hosted `ubuntu-latest` CI. Per the
# design doc's own requirement ("MLX imports and model downloads remain behind
# explicit adapter construction so the normal suite runs on GitHub-hosted Linux
# CI"), the real imports are deferred into the functions that actually need them —
# only reachable from `SttModelBundle.load()` and `SttSession`'s generation methods,
# never from portable test collection or from constructing a fake bundle/session.
if TYPE_CHECKING:
    from moshi_mlx import models

logger = logging.getLogger(__name__)

FRAME_SIZE = 1920  # samples per Mimi frame @ 24kHz (matches moshi-server's FRAME_SIZE const).
SAMPLE_RATE = 24_000
DEFAULT_ASR_DELAY_IN_TOKENS = 6  # matches the real kyutai/stt-1b-en_fr-candle stt.toml.

PAD_TOKEN = 0
WORD_BOUNDARY_TOKEN = 3


@dataclass
class SttStepEvent:
    """One protocol-relevant event produced by a single 1920-sample frame step."""

    kind: str  # "word" | "end_word" | "marker" | "step"
    text: str | None = None
    start_time: float | None = None
    stop_time: float | None = None
    marker_id: int | None = None
    step_idx: int | None = None
    prs: list[float] | None = None


@dataclass
class SttModelBundle:
    """Weights shared across sessions; loaded once per process."""

    lm: models.Lm
    mimi: models.mimi.Mimi
    text_tokenizer: sentencepiece.SentencePieceProcessor
    frame_rate: float
    asr_delay_in_tokens: int
    hf_repo: str

    @classmethod
    def load(
        cls,
        hf_repo: str,
        quantize_bits: int | None = None,
        asr_delay_in_tokens: int | None = None,
    ) -> SttModelBundle:
        import mlx.core as mx
        import mlx.nn as nn
        from moshi_mlx import models

        logger.info("stt: downloading config for %s", hf_repo)
        config_path = hf_hub_download(hf_repo, "config.json")
        with open(config_path) as fobj:
            raw_config = json.load(fobj)

        mimi_weights = hf_hub_download(hf_repo, raw_config["mimi_name"])
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(hf_repo, moshi_name)
        text_tokenizer_path = hf_hub_download(hf_repo, raw_config["tokenizer_name"])

        lm_config = models.LmConfig.from_config_dict(raw_config)
        lm = models.Lm(lm_config)
        lm.set_dtype(mx.bfloat16)

        logger.info("stt: loading LM weights from %s", moshi_weights)
        # `-candle` repos (including the default, kyutai/stt-1b-en_fr-candle)
        # ship genuine PyTorch state-dict-shaped safetensors (keys like
        # `out_norm.alpha`) and need `load_pytorch_weights`'s key
        # remapping/reshaping. `-mlx` repos ship checkpoints that are already
        # MLX-native (verified directly: `mx.load(model.safetensors)` on
        # kyutai/stt-1b-en_fr-mlx shows `out_norm.weight`, not
        # `out_norm.alpha`, and 131 keys matching `Lm`'s own parameter tree)
        # and must go through the plain `nn.Module.load_weights`, exactly as
        # `delayed-streams-modeling/scripts/stt_from_file_mlx.py` branches on
        # `args.hf_repo.endswith("-candle")` to decide between the two loaders.
        if hf_repo.endswith("-candle"):
            lm.load_pytorch_weights(moshi_weights, lm_config, strict=True)
        else:
            lm.load_weights(moshi_weights, strict=True)
        if quantize_bits is not None:
            logger.info("stt: quantizing LM to %d bits", quantize_bits)
            group_size = 32 if quantize_bits == 4 else 64
            nn.quantize(lm, bits=quantize_bits, group_size=group_size)

        text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer_path)  # type: ignore[call-arg]

        logger.info("stt: loading Mimi audio tokenizer from %s", mimi_weights)
        mimi = models.mimi.Mimi(models.mimi_202407(32))
        mimi.load_pytorch_weights(str(mimi_weights), strict=True)

        logger.info("stt: warming up")
        lm.warmup()
        mimi.warmup()

        stt_config = raw_config.get("stt_config", {})
        delay_seconds = stt_config.get("audio_delay_seconds")
        if asr_delay_in_tokens is not None:
            resolved_delay = asr_delay_in_tokens
        elif delay_seconds is not None:
            resolved_delay = round(delay_seconds * mimi.frame_rate)
        else:
            resolved_delay = DEFAULT_ASR_DELAY_IN_TOKENS
        logger.info(
            "stt: asr_delay_in_tokens=%d (frame_rate=%.2f)", resolved_delay, mimi.frame_rate
        )

        return cls(
            lm=lm,
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            frame_rate=mimi.frame_rate,
            asr_delay_in_tokens=resolved_delay,
            hf_repo=hf_repo,
        )


@dataclass
class SttSession:
    """Per-connection generation state. Cheap to create; the expensive weights live
    in the shared `SttModelBundle`. Only one session is ever active at a time in
    this project's first release (see design doc §Process Architecture), so it is
    safe to reset the shared `Mimi` tokenizer's streaming state here.
    """

    bundle: SttModelBundle
    max_steps: int
    gen: models.LmGen = field(init=False)
    _buffer: list[float] = field(default_factory=list, init=False)
    _word_tokens: list[int] = field(default_factory=list, init=False)
    _unended_word: bool = field(default=False, init=False)
    _last_stop_time: float = field(default=0.0, init=False)
    _item_step_idx: int = field(default=0, init=False)
    _model_step_idx: int = field(default=0, init=False)
    _pending_markers: list[tuple[int, int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        from moshi_mlx import models
        from moshi_mlx.utils.sampling import Sampler

        self.bundle.mimi.reset_all()
        for cache_entry in self.bundle.lm.transformer_cache:
            cache_entry.reset()
        for cache_entry in getattr(self.bundle.lm, "depformer_cache", []):
            cache_entry.reset()
        self.gen = models.LmGen(
            model=self.bundle.lm,
            max_steps=self.max_steps,
            text_sampler=Sampler(top_k=25, temp=0.0),
            audio_sampler=Sampler(top_k=250, temp=0.8),
            check=False,
        )

    def push_marker(self, marker_id: int) -> None:
        """Schedule a `Marker` echo once all audio submitted before it has crossed
        the ASR delay, matching `batched_asr.rs`'s marker scheduling.
        """
        target_item_step = self._item_step_idx + self.bundle.asr_delay_in_tokens
        self._pending_markers.append((target_item_step, marker_id))

    def push_audio(self, pcm: list[float]) -> list[SttStepEvent]:
        """Buffer arbitrary-length PCM and run one model step per complete
        `FRAME_SIZE` frame, mirroring `batched_asr.rs::Channel::extend_data`.
        """
        self._buffer.extend(pcm)
        events: list[SttStepEvent] = []
        while len(self._buffer) >= FRAME_SIZE:
            frame, self._buffer = self._buffer[:FRAME_SIZE], self._buffer[FRAME_SIZE:]
            events.extend(self._step(frame))
        return events

    def _step(self, frame: list[float]) -> list[SttStepEvent]:
        import mlx.core as mx

        if self.gen.step_idx >= self.gen.max_steps:
            raise RuntimeError(
                f"reached max_steps={self.gen.max_steps}; reconnect to start a fresh session"
            )

        block = mx.array(frame, dtype=mx.float32).reshape(1, 1, FRAME_SIZE)
        audio_tokens = self.bundle.mimi.encode_step(block).transpose(0, 2, 1)
        text_token_arr, vad_heads = self.gen.step_with_extra_heads(audio_tokens[0])
        text_token = int(text_token_arr[0].item())
        self._model_step_idx += 1

        events: list[SttStepEvent] = []
        if vad_heads:
            prs = [float(head[0, 0, 0].item()) for head in vad_heads]
            events.append(SttStepEvent(kind="step", step_idx=self._model_step_idx, prs=prs))

        self._item_step_idx += 1
        if self._item_step_idx >= self.bundle.asr_delay_in_tokens:
            if text_token in (PAD_TOKEN, WORD_BOUNDARY_TOKEN):
                if self._word_tokens:
                    text = self.bundle.text_tokenizer.decode(self._word_tokens)
                    events.append(
                        SttStepEvent(kind="word", text=text, start_time=self._last_stop_time)
                    )
                    self._word_tokens = []
                    self._unended_word = True
            else:
                self._word_tokens.append(text_token)

            if text_token == PAD_TOKEN:
                stop_time = (
                    self._item_step_idx - self.bundle.asr_delay_in_tokens
                ) / self.bundle.frame_rate
                if self._unended_word:
                    self._unended_word = False
                    events.append(SttStepEvent(kind="end_word", stop_time=stop_time))
                self._last_stop_time = stop_time

        still_pending: list[tuple[int, int]] = []
        for target_item_step, marker_id in self._pending_markers:
            if target_item_step <= self._item_step_idx:
                events.append(SttStepEvent(kind="marker", marker_id=marker_id))
            else:
                still_pending.append((target_item_step, marker_id))
        self._pending_markers = still_pending

        return events
