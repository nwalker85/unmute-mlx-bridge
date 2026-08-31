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
        n_q: int = 24,
        cfg_coef: float = 2.0,
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
            cfg_coef=cfg_coef,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            n_q=n_q,
            padding_bonus=0.0,
            raw_config=raw_config,
        )

        cfg_coef_conditioning: float | None = None
        if tts_model.valid_cfg_conditionings:
            if cfg_coef not in tts_model.valid_cfg_conditionings:
                raise ValueError(
                    f"unsupported cfg_coef {cfg_coef}; expected one of "
                    f"{sorted(tts_model.valid_cfg_conditionings)}"
                )
            # Model was trained with CFG distillation: pass cfg via conditioning,
            # not via a live classifier-free-guidance pass (matches tts_mlx.py).
            cfg_coef_conditioning = tts_model.cfg_coef
        # Always reset the live CFG pass to unconditional (1.0), regardless of
        # whether this model is CFG-distilled (RAV-1552 fix). `TTSModel`'s own
        # constructor sets `self.cfg_coef = cfg_coef` (the *live* classifier-
        # free-guidance strength `LmGen.step`/`_make_null` reads every step to
        # decide whether to double the batch); this used to only get reset to
        # `1.0` inside the `if valid_cfg_conditionings:` branch above, so a
        # non-distilled model kept whatever `cfg_coef` the constructor
        # received (e.g. the config-default `2.0`) and silently doubled batch
        # size/compute on every generation step instead of ever applying cfg
        # (which, for a non-distilled model, this bridge does not implement a
        # live CFG pass for at all -- see `cfg_coef_conditioning is None`
        # above).
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


class GenerationLengthLimitError(RuntimeError):
    """Raised by `TtsSession._step` when a session reaches its configured
    `max_gen_length`. This is a legitimate terminal condition, not a crash --
    the client is expected to reconnect and start a fresh session -- so
    `tts/server.py` reports it under its own `reason="length_limit"` metric
    label and a dedicated, client-safe message, distinct from an unexpected
    generation failure (`reason="generation"`). A dedicated exception class
    (rather than matching the message text of a generic `RuntimeError`) keeps
    that distinction from drifting if the message wording ever changes.
    """


class VoiceEmbeddingError(ValueError):
    """Raised when a client-supplied `Voice` message (custom cloned-voice
    embeddings) cannot be applied as session conditioning: a malformed shape, a
    shape/payload mismatch that slipped past protocol-level validation, or a
    model that does not support multi-speaker conditioning. Always caught by
    `tts/server.py` and reported as a protocol `Error` — never allowed to
    propagate as a crash or a silent fallback to the session's existing voice.
    """


def _condition_tensors_from_attributes(
    tts_model: TTSModel, attributes: object
) -> tuple[ConditionTensor | None, object]:
    """Build the `(ct, cross_attention_src)` pair `LmGen.step` expects from a
    `ConditionAttributes`. Shared by `TtsSession.__post_init__` (initial
    conditioning from the resolved query-param/default voice) and
    `TtsSession.apply_voice_embedding` (session-start re-conditioning from a
    client-supplied `Voice` message), so the two conditioning paths can never
    drift apart.
    """
    from moshi_mlx.modules.conditioner import ConditionTensor

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
    return ct, cross_attention_src


@dataclass
class TtsSession:
    """Per-connection generation state, adapted from `tts_mlx_streaming.py`'s
    `TTSGen`. Only one session is active at a time (see design doc), so it is safe
    to reset the shared model's caches here.
    """

    bundle: TtsModelBundle
    voice: str | Path | list[str | Path] | None
    """A single voice name/path (`voice=` query param), or a list of them for
    a `voices=` multi-voice blend (each entry already resolved to a `Path` by
    `tts/server.py`, or a raw name -- see `__post_init__`, which resolves any
    raw name via `tts_model.get_voice_path` exactly like the single-voice
    path does). The single-voice path (`str | Path | None`) is unchanged and
    byte-identical to before this field grew a list variant (RAV-1552)."""
    max_gen_length: int
    seed: int = 42
    temperature: float = 0.8
    top_k: int = 250
    cfg_alpha: float | None = None

    state: object = field(init=False)
    lm_gen: LmGen = field(init=False)
    offset: int = field(default=0, init=False)
    _ct: ConditionTensor | None = field(default=None, init=False)
    _cross_attention_src: object = field(default=None, init=False)
    _pending_frames: list[mx.array] = field(default_factory=list, init=False)
    _pending_word: tuple[str, int] | None = field(default=None, init=False)
    _emitted_transcript_len: int = field(default=0, init=False)
    _first_text_chunk: bool = field(default=True, init=False)
    # Set the moment `stream_text` is first called (even for a chunk that
    # produces no generation steps yet). Real `moshi-server` reads a
    # connection's pending `Voice` only on the channel-init entry
    # (`rust/moshi-server/tts.py:340-353`'s `if new_entry[0] == -1:` branch,
    # fed once per channel by `py_module.rs:237-240`'s
    # `if !c.sent_init { t.push(-1); ... }`) -- every later `Text` chunk takes
    # a different code path that never reads `voice`. `apply_voice_embedding`
    # uses this flag to reject a `Voice` message once generation has started,
    # enforcing the same session-start-only contract at the engine level that
    # `tts/server.py` also enforces via its own latch.
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        import mlx.core as mx
        from moshi_mlx.models.generate import LmGen
        from moshi_mlx.utils.sampling import Sampler

        tts_model = self.bundle.tts_model
        for cache_entry in tts_model.lm.transformer_cache:
            cache_entry.reset()
        for cache_entry in tts_model.lm.depformer_cache:
            cache_entry.reset()
        tts_model.mimi.reset_all()
        mx.random.seed(self.seed)

        if tts_model.multi_speaker:
            if isinstance(self.voice, list):
                # `voices=` multi-voice blend (RAV-1552): each entry is either
                # already a resolved `Path` (the normal `tts/server.py` path)
                # or a raw name to resolve here, exactly like the
                # single-voice branch below.
                voices = [
                    entry if isinstance(entry, Path) else tts_model.get_voice_path(entry)
                    for entry in self.voice
                ]
            elif isinstance(self.voice, Path):
                voices = [self.voice]
            else:
                voices = [
                    tts_model.get_voice_path(
                        self.voice or "expresso/ex03-ex01_happy_001_channel1_334s.wav"
                    )
                ]
        else:
            voices = []
        cfg_coef_conditioning = self.bundle.cfg_coef_conditioning
        if self.cfg_alpha is not None:
            if self.cfg_alpha not in tts_model.valid_cfg_conditionings:
                raise ValueError(
                    f"unsupported cfg_alpha {self.cfg_alpha}; expected one of "
                    f"{sorted(tts_model.valid_cfg_conditionings)}"
                )
            cfg_coef_conditioning = self.cfg_alpha
        attributes = tts_model.make_condition_attributes(
            voices, cfg_coef_conditioning
        )

        self.state = tts_model.machine.new_state([])
        self.offset = 0
        self._ct, self._cross_attention_src = _condition_tensors_from_attributes(
            tts_model, attributes
        )

        def _on_text_hook(text_tokens: mx.array) -> None:
            # moshi-mlx 0.3.0 added batching, so each item has shape (1,) here.
            # StateMachine.process expects the scalar token, matching the
            # package's batch TTS implementation.
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _consumed = tts_model.machine.process(
                    self.offset, self.state, token[0]
                )
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)[:, None]

        def _on_audio_hook(audio_tokens: mx.array) -> None:
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=self.max_gen_length,
            text_sampler=Sampler(temp=self.temperature, top_k=self.top_k),
            audio_sampler=Sampler(temp=self.temperature, top_k=self.top_k),
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
        )

    def apply_voice_embedding(self, embeddings: list[float], shape: list[int]) -> None:
        """Recondition this session on a client-supplied voice embedding (`Voice`
        protocol message) instead of the named voice resolved from `voice=`/
        `voices=` query parameters or config default that `__post_init__` already
        conditioned this session on.

        Session-start only, enforced here at the engine level (not only by
        `tts/server.py`'s own latch) so a direct user of `TtsSession` gets the
        same contract: raises `VoiceEmbeddingError` once `stream_text` has been
        called at least once (see `_started`). Real `moshi-server` mirrors this
        because it reads a connection's pending `Voice` only on the
        channel-init entry (`rust/moshi-server/tts.py:340-353`'s
        `if new_entry[0] == -1:` branch, fed once per channel by
        `py_module.rs:237-240`'s `if !c.sent_init { t.push(-1); ... }`) --
        every later `Text` chunk takes a different code path that never reads
        `voice`, so upstream silently forwards and then ignores any `Voice`
        message received after the first `Text`. This bridge instead rejects
        it explicitly here (and in `tts/server.py`, before generation is even
        attempted), rather than accepting it and silently discarding the
        request the way upstream does.

        `moshi_mlx.models.tts.TTSModel.make_condition_attributes` builds the same
        single-voice tensor layout but only accepts voice *paths* to load from a
        `.safetensors` file on disk; this bridge's `Voice` message carries the
        `speaker_wavs` tensor in-memory instead, so that ~20-line tensor
        construction is mirrored here rather than round-tripping embeddings
        through a temp file just to reuse a function built for paths.

        Raises `VoiceEmbeddingError` on any malformed or mismatched input, a
        model that does not support multi-speaker conditioning, or a session
        where generation has already started — never raises a bare/unexpected
        exception, and never silently keeps the prior voice while pretending
        to have applied the new one.
        """
        import mlx.core as mx

        if self._started:
            raise VoiceEmbeddingError(
                "voice conditioning is fixed once generation starts; upstream "
                "moshi-server only reads the voice on a channel's init entry "
                "(rust/moshi-server/tts.py:340-353, "
                "rust/moshi-server/src/py_module.rs:237-240) and silently "
                "ignores a Voice message received afterward — this bridge "
                "rejects it explicitly instead"
            )

        tts_model = self.bundle.tts_model
        if not tts_model.multi_speaker:
            raise VoiceEmbeddingError(
                "this model does not support multi-speaker conditioning; "
                "custom Voice embeddings cannot be applied"
            )
        if len(shape) != 3:
            raise VoiceEmbeddingError(
                f"voice embedding shape must have exactly 3 dimensions, got {shape!r}"
            )
        if any(dim <= 0 for dim in shape):
            # Defense in depth: `protocol/tts.py::TtsVoiceMessage` already
            # rejects this structurally, but a non-positive dimension (e.g.
            # `shape=[1, 512, 0]` with `embeddings=[]`) can make the product
            # of `shape` accidentally match `len(embeddings)` below, which
            # would otherwise reach `reshape` and have it silently infer the
            # missing dimension instead of erroring.
            raise VoiceEmbeddingError(
                f"voice embedding shape {shape!r} must have strictly positive dimensions"
            )
        expected_values = 1
        for dim in shape:
            expected_values *= dim
        if expected_values != len(embeddings):
            raise VoiceEmbeddingError(
                f"voice embedding shape {shape} implies {expected_values} values, "
                f"but {len(embeddings)} were provided"
            )

        try:
            emb = mx.array(embeddings, dtype=mx.float32).reshape(shape)
            max_speakers = tts_model.max_speakers
            voice_tensor = mx.zeros((1, max_speakers, emb.shape[2], emb.shape[1]))
            mask = mx.zeros((1, max_speakers, emb.shape[2]), dtype=mx.uint8)
            voice_tensor[:, 0, :, :] = emb.swapaxes(1, 2)
            mask[:, 0, :] = True
            voice_tensor = voice_tensor.reshape(1, -1, voice_tensor.shape[-1])
            mask = mask.reshape(1, -1)
        except Exception as exc:
            # `exc` is a third-party (`mlx`/`moshi_mlx`) exception -- e.g. an
            # array-broadcast `ValueError` like "Cannot broadcast array of
            # shape (2,1,3,2) into shape (1,1,3,2)" -- and must never reach a
            # client verbatim (RAV-1552 F2; this used to interpolate `exc`
            # directly into the message forwarded by `tts/server.py`). A
            # fixed, generic message is used instead; the full detail is
            # always logged server-side.
            logger.exception("tts: voice embedding tensor construction failed")
            raise VoiceEmbeddingError(
                "could not build voice embedding tensor"
            ) from exc

        from moshi_mlx.modules.conditioner import ConditionAttributes, TensorCondition

        cfg_coef_conditioning = self.bundle.cfg_coef_conditioning
        if self.cfg_alpha is not None:
            cfg_coef_conditioning = self.cfg_alpha
        text: dict[str, str | None] = {"control": "ok"}
        text["cfg"] = (
            None
            if cfg_coef_conditioning is None
            else format(cfg_coef_conditioning, ".1f")
        )

        # Note: only the shared `(ct, cross_attention_src)` construction below
        # is guaranteed not to drift from `__post_init__`'s path
        # (`_condition_tensors_from_attributes`). This `ConditionAttributes`
        # construction itself is duplicated, not shared, and already diverges:
        # upstream's `make_condition_attributes` raises `ValueError` when
        # `cfg_coef_conditioning not in valid_cfg_conditionings`, while this
        # path only formats the value with no such validation. Not currently
        # reachable (cfg_alpha is already validated against
        # `valid_cfg_conditionings` in `__post_init__`, and the config-default
        # `cfg_coef_conditioning` is trusted bundle state), but a real
        # divergence if either assumption ever changes.
        attributes = ConditionAttributes(
            text=text,
            tensor={"speaker_wavs": TensorCondition(voice_tensor, mask)},
        )
        try:
            self._ct, self._cross_attention_src = _condition_tensors_from_attributes(
                tts_model, attributes
            )
        except Exception as exc:
            # Same rationale as the tensor-construction wrap above (RAV-1552
            # F2): `exc` is a third-party exception and must never reach a
            # client verbatim. A fixed, generic message is used instead; the
            # full detail is always logged server-side.
            logger.exception("tts: voice embedding conditioning failed")
            raise VoiceEmbeddingError(
                "could not apply voice embedding conditioning"
            ) from exc

    def push_text(self, text: str) -> list[TtsStepEvent]:
        return list(self.stream_text(text))

    def stream_text(
        self, text: str, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[TtsStepEvent]:
        """Tokenize `text` into word `Entry`s and run generation steps until the
        state machine needs another word (mirrors `TTSGen.process()`).

        Marks the session as started (see `_started`) before doing anything
        else: this is the signal `apply_voice_embedding` uses to reject a
        `Voice` message once generation is underway (RAV-1504), matching real
        `moshi-server`'s session-start-only conditioning.
        """
        self._started = True

        from moshi_mlx.models.tts import script_to_entries

        tts_model = self.bundle.tts_model
        entries = script_to_entries(
            tts_model.tokenizer,
            tts_model.machine.token_ids,
            tts_model.mimi.frame_rate,
            [text],
            multi_speaker=self._first_text_chunk and tts_model.multi_speaker,
            padding_between=1,
        )
        if entries:
            self._first_text_chunk = False
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
            raise GenerationLengthLimitError(
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

        if frame is not None and not (
            frame == tts_model.machine.token_ids.zero
        ).any():
            pcm = tts_model.mimi.decode_step(frame[:, :, None])
            pcm = mx.clip(pcm[0, 0], -1, 1)
            events.append(TtsStepEvent(kind="audio", pcm=pcm.tolist()))
        return events
