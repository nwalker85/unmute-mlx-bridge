"""Real MLX inference for streaming text-to-speech.

Model loading and the per-step generation loop are a direct adaptation of
`kyutai-labs/delayed-streams-modeling`'s reference `scripts/tts_mlx_streaming.py`
(pinned at `4c4f65e147df056adf3346290d64c7b9649b18c9`) — specifically its `TTSGen`
dataclass, which is the only place in the Kyutai OSS estate that drives
`moshi_mlx`'s `TTSModel`/`StateMachine` incrementally (word-by-word) rather than
as a single batch `generate()` call. That script writes audio straight to a
soundcard/file and never reports word timing; the timing logic below (deriving
`Text{text, start_s, stop_s}` events) is new, ported from the real `moshi-server`'s
`rust/moshi-server/tts.py` (`TTSService.step` / `MaskFlags.WORD_FINISHED`) and
`rust/moshi-server/src/tts.rs` (`Channel::on_end_of_word`): a word's boundaries are
finalized the moment the *next* word starts being consumed, using
`state.transcript`'s `(word, step)` pairs divided by `mimi.frame_rate` (12.5 Hz).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import sentencepiece

# See the equivalent comment in `stt/engine.py`: `mlx`/`moshi-mlx` imports are
# deferred out of module scope so this module (and therefore `tts/server.py` and
# the portable conformance suite) stays importable on platforms without MLX.
if TYPE_CHECKING:
    import mlx.core as mx
    from moshi_mlx.models.generate import LmGen
    from moshi_mlx.models.tts import TTSModel
    from moshi_mlx.modules.conditioner import ConditionTensor

# `moshi_mlx.models.tts.DEFAULT_DSM_TTS_VOICE_REPO` — inlined as a literal because
# it is used as a function *default argument value*, which Python evaluates at
# module-import time (unlike a type annotation, which `from __future__ import
# annotations` makes lazy). Its value: "kyutai/tts-voices".
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"

logger = logging.getLogger(__name__)


@dataclass
class TtsStepEvent:
    kind: str  # "audio" | "word"
    pcm: list[float] | None = None
    text: str | None = None
    start_s: float | None = None
    stop_s: float | None = None


@dataclass
class TtsModelBundle:
    """Weights shared across sessions; loaded once per process."""

    tts_model: TTSModel
    cfg_coef_conditioning: float | None
    hf_repo: str
    voice_repo: str

    def resolve_voice(self, voice: str) -> Path | None:
        """Resolve a requested voice without mutating shared MLX generation state.

        The Hugging Face fetch may block on a first-use cache miss, so the server
        runs this phase independently from `TtsSession` construction.
        """
        if not self.tts_model.multi_speaker:
            return None
        return self.tts_model.get_voice_path(voice)

    @classmethod
    def load(
        cls,
        hf_repo: str,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        quantize_bits: int | None = None,
    ) -> TtsModelBundle:
        import mlx.core as mx
        import mlx.nn as nn
        from moshi_mlx import models
        from moshi_mlx.models.tts import TTSModel
        from moshi_mlx.utils.loaders import hf_get

        logger.info("tts: downloading config for %s", hf_repo)
        raw_config_path = hf_get("config.json", hf_repo)
        with open(raw_config_path) as fobj:
            raw_config = json.load(fobj)

        mimi_weights = hf_get(raw_config["mimi_name"], hf_repo)
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, hf_repo)
        tokenizer_path = hf_get(raw_config["tokenizer_name"], hf_repo)

        lm_config = models.LmConfig.from_config_dict(raw_config)
        # Works around a bug in moshi_mlx<=0.3.0's ring KV-cache handling for the
        # TTS depformer, per the upstream reference script's own comment.
        lm_config.transformer.max_seq_len = lm_config.transformer.context
        lm = models.Lm(lm_config)
        lm.set_dtype(mx.bfloat16)

        logger.info("tts: loading LM weights from %s", moshi_weights)
        lm.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)
        if quantize_bits is not None:
            logger.info("tts: quantizing depformer + attention/gating to %d bits", quantize_bits)
            nn.quantize(lm.depformer, bits=quantize_bits)
            for layer in lm.transformer.layers:
                nn.quantize(layer.self_attn, bits=quantize_bits)
                nn.quantize(layer.gating, bits=quantize_bits)

        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))  # type: ignore[call-arg]

        logger.info("tts: loading Mimi audio tokenizer from %s", mimi_weights)
        generated_codebooks = lm_config.generated_codebooks
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        tts_model = TTSModel(
            lm,
            audio_tokenizer,
            text_tokenizer,
            voice_repo=voice_repo,
            temp=0.6,
            cfg_coef=1.0,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0.0,
            raw_config=raw_config,
        )

        cfg_coef_conditioning: float | None = None
        if tts_model.valid_cfg_conditionings:
            # Model was trained with CFG distillation: pass cfg via conditioning,
            # not via a live classifier-free-guidance pass (matches tts_mlx.py).
            cfg_coef_conditioning = tts_model.cfg_coef
            tts_model.cfg_coef = 1.0

        logger.info("tts: warming up")
        _warm_state = tts_model.machine.new_state([])
        del _warm_state

        return cls(
            tts_model=tts_model,
            cfg_coef_conditioning=cfg_coef_conditioning,
            hf_repo=hf_repo,
            voice_repo=voice_repo,
        )


@dataclass
class TtsSession:
    """Per-connection generation state, adapted from `tts_mlx_streaming.py`'s
    `TTSGen`. Only one session is active at a time (see design doc), so it is safe
    to reset the shared model's caches here.
    """

    bundle: TtsModelBundle
    voice: str | Path | None
    max_gen_length: int

    state: object = field(init=False)
    lm_gen: LmGen = field(init=False)
    offset: int = field(default=0, init=False)
    _ct: ConditionTensor | None = field(default=None, init=False)
    _cross_attention_src: object = field(default=None, init=False)
    _pending_frames: list[mx.array] = field(default_factory=list, init=False)
    _pending_word: tuple[str, int] | None = field(default=None, init=False)
    _emitted_transcript_len: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import mlx.core as mx
        from moshi_mlx.models.generate import LmGen
        from moshi_mlx.modules.conditioner import ConditionTensor
        from moshi_mlx.utils.sampling import Sampler

        tts_model = self.bundle.tts_model
        for cache_entry in tts_model.lm.transformer_cache:
            cache_entry.reset()
        for cache_entry in tts_model.lm.depformer_cache:
            cache_entry.reset()
        tts_model.mimi.reset_all()

        if tts_model.multi_speaker:
            if isinstance(self.voice, Path):
                voice_path = self.voice
            else:
                voice_path = tts_model.get_voice_path(
                    self.voice or "expresso/ex03-ex01_happy_001_channel1_334s.wav"
                )
            voices = [voice_path]
        else:
            voices = []
        attributes = tts_model.make_condition_attributes(
            voices, self.bundle.cfg_coef_conditioning
        )

        self.state = tts_model.machine.new_state([])
        self.offset = 0

        ct: ConditionTensor | None = None
        cross_attention_src = None
        assert tts_model.lm.condition_provider is not None
        for key, value in attributes.text.items():
            attr_ct = tts_model.lm.condition_provider.condition_tensor(key, value)
            ct = attr_ct if ct is None else ConditionTensor(ct.tensor + attr_ct.tensor)
        for key, value in attributes.tensor.items():
            conditioner = tts_model.lm.condition_provider.conditioners[key]
            ca_src = conditioner.condition(value)
            if cross_attention_src is None:
                cross_attention_src = ca_src
            else:
                raise ValueError("multiple cross-attention conditioners")
        self._ct = ct
        self._cross_attention_src = cross_attention_src

        def _on_text_hook(text_tokens: mx.array) -> None:
            # NOTE: `text_tokens` has shape (batch, 1), so `token` here is a
            # single-element list, not a scalar. `StateMachine.process`'s `token
            # not in [new_word, pad]` clamp therefore always fires, and
            # `new_word` is driven by the padding countdown rather than the
            # model spontaneously sampling `token_ids.new_word`. Verified this
            # is not an adaptation bug: the *exact same* pattern (no `token[0]`
            # unpacking) is used both by the upstream reference streaming script
            # (`tts_mlx_streaming.py::TTSGen._on_text_hook`) and by real
            # `moshi-server`'s own production Python TTS glue
            # (`rust/moshi-server/tts.py::TTSService._on_text_hook`) — only the
            # *batch* `TTSModel.generate()` path unpacks `token[0]`, a different
            # call site with a different shape convention.
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _consumed = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        def _on_audio_hook(audio_tokens: mx.array) -> None:
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=self.max_gen_length,
            text_sampler=Sampler(temp=tts_model.temp),
            audio_sampler=Sampler(temp=tts_model.temp),
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
        )

    def push_text(self, text: str) -> list[TtsStepEvent]:
        return list(self.stream_text(text))

    def stream_text(
        self, text: str, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[TtsStepEvent]:
        """Tokenize `text` into word `Entry`s and run generation steps until the
        state machine needs another word (mirrors `TTSGen.process()`).
        """
        tts_model = self.bundle.tts_model
        entries = tts_model.prepare_script([text], padding_between=1)
        self.state.entries.extend(entries)
        while len(self.state.entries) > tts_model.machine.second_stream_ahead:
            if cancelled is not None and cancelled():
                return
            yield from self._step()

    def push_eos(self) -> list[TtsStepEvent]:
        return list(self.stream_eos())

    def stream_eos(
        self, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[TtsStepEvent]:
        """Drain remaining entries and run the trailing padding steps, mirroring
        `TTSGen.process_last()`.
        """
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            if cancelled is not None and cancelled():
                return
            yield from self._step()
        tts_model = self.bundle.tts_model
        additional_steps = tts_model.delay_steps + max(tts_model.lm.delays) + 8
        for _ in range(additional_steps):
            if cancelled is not None and cancelled():
                return
            yield from self._step()
        if self._pending_word is not None:
            word, start_step = self._pending_word
            yield TtsStepEvent(
                kind="word",
                text=word,
                start_s=start_step / tts_model.mimi.frame_rate,
                stop_s=self.offset / tts_model.mimi.frame_rate,
            )
            self._pending_word = None

    def _step(self) -> list[TtsStepEvent]:
        import mlx.core as mx

        if self.offset >= self.max_gen_length:
            raise RuntimeError(
                f"reached max_gen_length={self.max_gen_length}; reconnect to start a fresh session"
            )
        tts_model = self.bundle.tts_model
        missing = tts_model.lm.n_q - tts_model.lm.dep_q
        input_tokens = mx.ones((1, missing), dtype=mx.int64) * tts_model.machine.token_ids.zero
        self.lm_gen.step(input_tokens, ct=self._ct, cross_attention_src=self._cross_attention_src)
        frame = self.lm_gen.last_audio_tokens()
        self.offset += 1

        events: list[TtsStepEvent] = []
        transcript = self.state.transcript
        while self._emitted_transcript_len < len(transcript):
            word, step = transcript[self._emitted_transcript_len]
            self._emitted_transcript_len += 1
            if self._pending_word is not None:
                prev_word, prev_start = self._pending_word
                events.append(
                    TtsStepEvent(
                        kind="word",
                        text=prev_word,
                        start_s=prev_start / tts_model.mimi.frame_rate,
                        stop_s=step / tts_model.mimi.frame_rate,
                    )
                )
            self._pending_word = (word, step)

        if frame is not None:
            pcm = tts_model.mimi.decode_step(frame[:, :, None])
            pcm = mx.clip(pcm[0, 0], -1, 1)
            events.append(TtsStepEvent(kind="audio", pcm=pcm.tolist()))
        return events
