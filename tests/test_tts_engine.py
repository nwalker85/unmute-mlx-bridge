from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unmute_mlx_bridge.tts import engine as tts_engine


def _fake_session_bundle(cfg_calls):
    tts_model = SimpleNamespace(
        lm=SimpleNamespace(
            transformer_cache=[],
            depformer_cache=[],
            condition_provider=object(),
        ),
        mimi=SimpleNamespace(reset_all=lambda: None),
        multi_speaker=False,
        valid_cfg_conditionings={1.0, 1.5},
        machine=SimpleNamespace(
            new_state=lambda _entries: SimpleNamespace(
                entries=[], end_step=None, transcript=[]
            )
        ),
        make_condition_attributes=lambda voices, cfg: (
            cfg_calls.append((voices, cfg))
            or SimpleNamespace(text={}, tensor={})
        ),
        temp=0.6,
    )
    return SimpleNamespace(tts_model=tts_model, cfg_coef_conditioning=1.0)


def _install_fake_generation_modules(monkeypatch):
    seeds: list[int] = []
    sampler_args: list[tuple[float, int | None]] = []
    lm_gen_kwargs: list[dict[str, object]] = []

    mlx_core = types.ModuleType("mlx.core")
    mlx_core.random = SimpleNamespace(seed=seeds.append)
    mlx_core.array = np.array
    mlx_core.int64 = np.int64

    class FakeSampler:
        def __init__(self, *, temp=0.8, top_k=None, **_kwargs):
            sampler_args.append((temp, top_k))

    class FakeLmGen:
        def __init__(self, *_args, **kwargs):
            lm_gen_kwargs.append(kwargs)

    generate_module = types.ModuleType("moshi_mlx.models.generate")
    generate_module.LmGen = FakeLmGen
    conditioner_module = types.ModuleType("moshi_mlx.modules.conditioner")
    conditioner_module.ConditionTensor = object
    sampling_module = types.ModuleType("moshi_mlx.utils.sampling")
    sampling_module.Sampler = FakeSampler

    for name, module in {
        "mlx.core": mlx_core,
        "moshi_mlx.models.generate": generate_module,
        "moshi_mlx.modules.conditioner": conditioner_module,
        "moshi_mlx.utils.sampling": sampling_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return seeds, sampler_args, lm_gen_kwargs


def test_session_applies_seed_sampler_and_cfg_query_settings(monkeypatch):
    seeds, sampler_args, _lm_gen_kwargs = _install_fake_generation_modules(monkeypatch)
    cfg_calls = []

    tts_engine.TtsSession(
        bundle=_fake_session_bundle(cfg_calls),
        voice=None,
        max_gen_length=1000,
        seed=7,
        temperature=0.4,
        top_k=11,
        cfg_alpha=1.5,
    )

    assert seeds == [7]
    assert sampler_args == [(0.4, 11), (0.4, 11)]
    assert cfg_calls == [([], 1.5)]


def test_session_rejects_unsupported_cfg_before_generation(monkeypatch):
    _install_fake_generation_modules(monkeypatch)
    cfg_calls = []

    with pytest.raises(ValueError, match="unsupported cfg_alpha"):
        tts_engine.TtsSession(
            bundle=_fake_session_bundle(cfg_calls),
            voice=None,
            max_gen_length=1000,
            cfg_alpha=1.25,
        )

    assert cfg_calls == []


def test_text_hook_unpacks_batched_tokens_before_state_machine(monkeypatch):
    _seeds, _sampler_args, lm_gen_kwargs = _install_fake_generation_modules(
        monkeypatch
    )
    cfg_calls = []
    bundle = _fake_session_bundle(cfg_calls)
    processed_tokens = []

    def process(_offset, _state, token):
        processed_tokens.append(token)
        return 3, False

    bundle.tts_model.machine.process = process
    tts_engine.TtsSession(
        bundle=bundle,
        voice=None,
        max_gen_length=1000,
    )
    text_tokens = np.array([[0]], dtype=np.int64)

    lm_gen_kwargs[0]["on_text_hook"](text_tokens)

    assert processed_tokens == [0]
    assert text_tokens.tolist() == [[3]]


def test_stream_text_injects_speaker_marker_only_for_first_chunk(monkeypatch):
    tts_module = types.ModuleType("moshi_mlx.models.tts")

    def script_to_entries(
        _tokenizer,
        _token_ids,
        _frame_rate,
        script,
        *,
        multi_speaker,
        padding_between,
    ):
        assert padding_between == 1
        tokens = [1, len(script[0])] if multi_speaker else [len(script[0])]
        return [SimpleNamespace(tokens=tokens, text=script[0])]

    tts_module.script_to_entries = script_to_entries
    monkeypatch.setitem(sys.modules, "moshi_mlx.models.tts", tts_module)

    tts_model = SimpleNamespace(
        tokenizer=object(),
        machine=SimpleNamespace(
            token_ids=object(),
            second_stream_ahead=100,
        ),
        mimi=SimpleNamespace(frame_rate=12.5),
        multi_speaker=True,
        prepare_script=lambda script, padding_between: [
            SimpleNamespace(tokens=[1, len(script[0])], text=script[0])
        ],
    )
    session = object.__new__(tts_engine.TtsSession)
    session.bundle = SimpleNamespace(tts_model=tts_model)
    session.state = SimpleNamespace(entries=[])
    session._first_text_chunk = True
    session._started = False

    list(session.stream_text("one"))
    list(session.stream_text("two"))

    assert [entry.tokens for entry in session.state.entries] == [[1, 3], [3]]


def test_step_does_not_decode_frames_containing_zero_tokens(monkeypatch):
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.int64 = np.int64
    mlx_core.ones = np.ones
    mlx_core.clip = np.clip
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    decode_calls = []
    tts_model = SimpleNamespace(
        lm=SimpleNamespace(n_q=2, dep_q=2),
        machine=SimpleNamespace(token_ids=SimpleNamespace(zero=-1)),
        mimi=SimpleNamespace(
            decode_step=lambda frame: (
                decode_calls.append(frame) or np.zeros((1, 1, 1920))
            )
        ),
    )
    session = object.__new__(tts_engine.TtsSession)
    session.bundle = SimpleNamespace(tts_model=tts_model)
    session.max_gen_length = 100
    session.state = SimpleNamespace(transcript=[])
    session.lm_gen = SimpleNamespace(
        step=lambda *_args, **_kwargs: None,
        last_audio_tokens=lambda: np.array([[-1, 7]], dtype=np.int64),
    )
    session.offset = 0
    session._emitted_transcript_len = 0
    session._pending_word = None

    events = session._step()

    assert events == []
    assert decode_calls == []


def _install_fake_apply_voice_modules(monkeypatch):
    """Minimal `mlx.core` (backed by numpy, matching this file's existing
    mocking style) and `moshi_mlx.modules.conditioner` stand-ins, just enough
    to exercise `TtsSession.apply_voice_embedding`'s tensor construction without
    real MLX or model weights.
    """
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = np.array
    mlx_core.zeros = np.zeros
    mlx_core.float32 = np.float32
    mlx_core.uint8 = np.uint8
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    class FakeConditionTensor:
        def __init__(self, tensor):
            self.tensor = tensor

        def __add__(self, other):
            return FakeConditionTensor(self.tensor + other)

    class FakeConditionAttributes:
        def __init__(self, text, tensor):
            self.text = text
            self.tensor = tensor

    class FakeTensorCondition:
        def __init__(self, tensor, mask):
            self.tensor = tensor
            self.mask = mask

    conditioner_module = types.ModuleType("moshi_mlx.modules.conditioner")
    conditioner_module.ConditionTensor = FakeConditionTensor
    conditioner_module.ConditionAttributes = FakeConditionAttributes
    conditioner_module.TensorCondition = FakeTensorCondition
    monkeypatch.setitem(sys.modules, "moshi_mlx.modules.conditioner", conditioner_module)


def _fake_apply_voice_session(*, multi_speaker=True, max_speakers=2, cfg_alpha=None):
    captured_tensor_conditions: list[object] = []

    conditioners = {
        "speaker_wavs": SimpleNamespace(
            condition=lambda value: captured_tensor_conditions.append(value)
            or SimpleNamespace(marker="cross_attention_src")
        )
    }
    tts_model = SimpleNamespace(
        multi_speaker=multi_speaker,
        max_speakers=max_speakers,
        lm=SimpleNamespace(
            condition_provider=SimpleNamespace(
                condition_tensor=lambda _key, _value: SimpleNamespace(tensor=0),
                conditioners=conditioners,
            )
        ),
    )
    session = object.__new__(tts_engine.TtsSession)
    session.bundle = SimpleNamespace(tts_model=tts_model, cfg_coef_conditioning=None)
    session.cfg_alpha = cfg_alpha
    session._ct = "unset"
    session._cross_attention_src = "unset"
    session._started = False
    return session, captured_tensor_conditions


def test_apply_voice_embedding_rejects_non_multi_speaker_model(monkeypatch):
    _install_fake_apply_voice_modules(monkeypatch)
    session, _captured = _fake_apply_voice_session(multi_speaker=False)

    with pytest.raises(tts_engine.VoiceEmbeddingError, match="multi-speaker"):
        session.apply_voice_embedding([0.1, 0.2, 0.3], [1, 3, 1])

    assert session._ct == "unset"
    assert session._cross_attention_src == "unset"


def test_apply_voice_embedding_rejects_wrong_number_of_dimensions(monkeypatch):
    _install_fake_apply_voice_modules(monkeypatch)
    session, _captured = _fake_apply_voice_session()

    with pytest.raises(tts_engine.VoiceEmbeddingError, match="3 dimensions"):
        session.apply_voice_embedding([0.1, 0.2], [1, 2])

    assert session._ct == "unset"
    assert session._cross_attention_src == "unset"


def test_apply_voice_embedding_rejects_zero_dimension_with_matching_product(monkeypatch):
    """Confirmed defect (RAV-1504 blocker): `shape=[1, 512, 0]` with
    `embeddings=[]` used to pass the length-product check unmodified
    (`0 == len([]) == 0`) and reach `reshape`, which infers the zero-length
    dimension instead of erroring -- silently discarding valid conditioning.
    A non-positive dimension is now rejected before that check runs, no
    matter what `embeddings` accompanies it.
    """
    _install_fake_apply_voice_modules(monkeypatch)
    session, _captured = _fake_apply_voice_session()

    with pytest.raises(tts_engine.VoiceEmbeddingError, match="strictly positive"):
        session.apply_voice_embedding([], [1, 512, 0])

    assert session._ct == "unset"
    assert session._cross_attention_src == "unset"


def test_apply_voice_embedding_rejects_negative_dimension(monkeypatch):
    """A negative dimension is rejected by the positivity check itself, before
    the length-product comparison even runs -- so this is caught the same way
    regardless of whether some other combination of magnitudes would also
    happen to fail (or accidentally pass) that later comparison.
    """
    _install_fake_apply_voice_modules(monkeypatch)
    session, _captured = _fake_apply_voice_session()

    with pytest.raises(tts_engine.VoiceEmbeddingError, match="strictly positive"):
        session.apply_voice_embedding([0.1] * 6, [1, -2, 3])

    assert session._ct == "unset"
    assert session._cross_attention_src == "unset"


def test_apply_voice_embedding_rejects_shape_payload_mismatch(monkeypatch):
    _install_fake_apply_voice_modules(monkeypatch)
    session, _captured = _fake_apply_voice_session()

    with pytest.raises(tts_engine.VoiceEmbeddingError, match="implies 6 values"):
        session.apply_voice_embedding([0.1, 0.2, 0.3], [1, 3, 2])

    assert session._ct == "unset"
    assert session._cross_attention_src == "unset"


def test_apply_voice_embedding_rejects_once_generation_has_started(monkeypatch):
    """Engine-level enforcement (RAV-1504) of the same session-start-only
    contract `tts/server.py` enforces via its own latch: real `moshi-server`
    reads a connection's pending voice only on the channel-init entry
    (`rust/moshi-server/tts.py:340-353`, `rust/moshi-server/src/py_module.rs:237-240`)
    and never re-reads it once generation is underway, so this must raise
    regardless of whether the embedding itself would otherwise be valid --
    and it must raise before any other validation, since the model-specific
    checks below are irrelevant once the session has already started.
    """
    _install_fake_apply_voice_modules(monkeypatch)
    session, captured = _fake_apply_voice_session()
    session._started = True

    with pytest.raises(tts_engine.VoiceEmbeddingError, match="generation starts"):
        session.apply_voice_embedding([0.1, 0.2, 0.3], [1, 3, 1])

    assert session._ct == "unset"
    assert session._cross_attention_src == "unset"
    assert captured == []


def test_apply_voice_embedding_builds_single_voice_tensor_at_slot_zero(monkeypatch):
    """Mirrors `moshi_mlx.models.tts.TTSModel.make_condition_attributes`'s
    single-voice tensor layout: only speaker slot 0 carries the embedding (as
    `[D, T] -> [T, D]`), every other one of `max_speakers` slots is zero and
    unmasked.
    """
    _install_fake_apply_voice_modules(monkeypatch)
    session, captured = _fake_apply_voice_session(max_speakers=2)

    # D=3, T=2: emb[0] = [[1, 2], [3, 4], [5, 6]] (row per D, column per T).
    session.apply_voice_embedding([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 3, 2])

    assert len(captured) == 1
    tensor_condition = captured[0]
    # (1, max_speakers * T, D) = (1, 4, 3).
    assert tensor_condition.tensor.shape == (1, 4, 3)
    assert tensor_condition.mask.shape == (1, 4)
    # Slot 0 (rows 0-1) holds the swapaxes(1, 2) of the embedding; slot 1
    # (rows 2-3) is zero and unmasked.
    np.testing.assert_array_equal(
        tensor_condition.tensor[0, :2, :], [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    )
    np.testing.assert_array_equal(tensor_condition.tensor[0, 2:, :], np.zeros((2, 3)))
    assert tensor_condition.mask[0].tolist() == [1, 1, 0, 0]

    # Applied directly, not staged: the active `_ct`/`_cross_attention_src`
    # used by generation are updated immediately (RAV-1504 session-start
    # conditioning -- see `TtsSession.__post_init__` and `.stream_text`).
    assert session._cross_attention_src == SimpleNamespace(marker="cross_attention_src")


def test_apply_voice_embedding_wraps_unexpected_tensor_errors(monkeypatch):
    """A tensor-construction failure downstream of the shape/length checks must
    still surface as `VoiceEmbeddingError` -- never a bare exception escaping
    to the caller.
    """
    _install_fake_apply_voice_modules(monkeypatch)
    session, _captured = _fake_apply_voice_session()

    # Shape [1, 2, 2] passes both the dimensionality and length checks
    # (product 4 == len(embeddings) 4): the failure this test exercises is
    # downstream, in the tensor construction itself. Setting `max_speakers=0`
    # makes the voice tensor's speaker axis zero-length, so writing to
    # speaker slot 0 (`voice_tensor[:, 0, :, :] = ...`) raises an IndexError
    # that must still be wrapped as `VoiceEmbeddingError`.
    session.bundle.tts_model.max_speakers = 0

    with pytest.raises(
        tts_engine.VoiceEmbeddingError,
        match="could not build voice embedding tensor",
    ):
        session.apply_voice_embedding([1.0, 2.0, 3.0, 4.0], [1, 2, 2])


def test_model_bundle_applies_requested_audio_codebook_depth(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "mimi_name": "mimi.safetensors",
                "moshi_name": "model.safetensors",
                "tokenizer_name": "tokenizer.model",
            }
        )
    )

    class FakeLmConfig:
        transformer = SimpleNamespace(context=2048, max_seq_len=0)
        generated_codebooks = 32

    class FakeLm:
        def __init__(self, config):
            self.config = config
            self.depformer = object()
            self.transformer = SimpleNamespace(layers=[])

        def set_dtype(self, _dtype):
            pass

        def load_pytorch_weights(self, _path, _config, strict):
            assert strict is True

    class FakeMimi:
        def __init__(self, _config):
            pass

        def load_pytorch_weights(self, _path, strict):
            assert strict is True

    class FakeTtsModel:
        def __init__(self, _lm, _mimi, _tokenizer, *, n_q=32, **_kwargs):
            self.n_q = n_q
            self.valid_cfg_conditionings = set()
            self.cfg_coef = 1.0
            self.machine = SimpleNamespace(new_state=lambda _entries: object())

    mlx_package = types.ModuleType("mlx")
    mlx_package.__path__ = []
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.bfloat16 = object()
    mlx_nn = types.ModuleType("mlx.nn")
    mlx_nn.quantize = lambda *_args, **_kwargs: None
    mlx_package.core = mlx_core
    mlx_package.nn = mlx_nn

    moshi_package = types.ModuleType("moshi_mlx")
    moshi_package.__path__ = []
    models_module = types.ModuleType("moshi_mlx.models")
    models_module.__path__ = []
    models_module.LmConfig = SimpleNamespace(
        from_config_dict=lambda _raw_config: FakeLmConfig()
    )
    models_module.Lm = FakeLm
    models_module.mimi = SimpleNamespace(Mimi=FakeMimi)
    models_module.mimi_202407 = lambda _generated_codebooks: object()
    moshi_package.models = models_module
    tts_module = types.ModuleType("moshi_mlx.models.tts")
    tts_module.TTSModel = FakeTtsModel
    utils_module = types.ModuleType("moshi_mlx.utils")
    utils_module.__path__ = []
    loaders_module = types.ModuleType("moshi_mlx.utils.loaders")

    def hf_get(name, _repo):
        return config_path if name == "config.json" else tmp_path / name

    loaders_module.hf_get = hf_get

    for name, module in {
        "mlx": mlx_package,
        "mlx.core": mlx_core,
        "mlx.nn": mlx_nn,
        "moshi_mlx": moshi_package,
        "moshi_mlx.models": models_module,
        "moshi_mlx.models.tts": tts_module,
        "moshi_mlx.utils": utils_module,
        "moshi_mlx.utils.loaders": loaders_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        tts_engine.sentencepiece,
        "SentencePieceProcessor",
        lambda _path: object(),
    )

    bundle = tts_engine.TtsModelBundle.load(
        "kyutai/tts-1.6b-en_fr",
        quantize_bits=None,
        n_q=24,
    )

    assert bundle.tts_model.n_q == 24
