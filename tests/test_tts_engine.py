from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from unmute_mlx_bridge.tts import engine as tts_engine


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
