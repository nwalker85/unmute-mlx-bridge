# Changelog

All notable changes to this project are documented here.

This project follows Semantic Versioning once it exposes a public contract,
deployable artifact, API, SDK, schema, or specification.

Pre-1.0 releases use `0.y.z` under SemVer v2. Breaking changes are documented
even under `0.y.z`. See `docs/architecture/decisions/` for decisions affecting
compatibility or version labels.

## [Unreleased]

### Added

- Initial repository scaffold: governance files, package seed, and portable CI.
- `docs/repo-intake.md` with lifecycle decisions.
- `package-surface.json` recording GitHub-primary exception posture.
- `.github/workflows/ci.yml` using GitHub Actions for portable CI.
- `PROTOCOL.md`: annotated wire-format writeup for `/api/asr-streaming` and
  `/api/tts_streaming`, verified against `kyutai-labs/unmute` and
  `kyutai-labs/moshi` source at the commits pinned in the design spec.
- `src/unmute_mlx_bridge/protocol/`: pydantic msgpack message models for both
  endpoints, matching real `moshi-server`'s `OutMsg`/`InMsg` shapes field-for-field.
- `src/unmute_mlx_bridge/stt/`: real MLX STT inference (`engine.py`, backed by
  `moshi-mlx` + `kyutai/stt-1b-en_fr-candle`) and a `moshi-server`-compatible
  `/api/asr-streaming` WebSocket server (`server.py`), including the
  Word/EndWord segmentation state machine ported from `moshi-core/src/asr.rs`.
- `src/unmute_mlx_bridge/tts/`: real MLX TTS inference (`engine.py`, backed by
  `moshi-mlx` + `kyutai/tts-1.6b-en_fr`) and a `moshi-server`-compatible
  `/api/tts_streaming` WebSocket server (`server.py`), including word-timing
  derivation from the TTS state machine's transcript.
- `src/unmute_mlx_bridge/observability.py`: shared `/healthz`, `/readyz`,
  `/metrics`, and pre-upgrade auth (`kyutai-api-key` header / `auth_id` query
  param, matching real `moshi-server`).
- `src/unmute_mlx_bridge/config.py`: environment-driven server configuration.
- Portable protocol + full-session conformance test suite
  (`tests/test_protocol_{stt,tts}.py`, `tests/test_{stt,tts}_server_conformance.py`)
  exercising the real server classes against a deterministic fake engine over a
  real WebSocket connection — no model weights required.
- Opt-in hardware test suite (`tests/hardware/`, `pytest -m hardware`): real
  MLX inference on Apple Silicon, real audio fixtures synthesized with macOS
  `say`, transcription/synthesis correctness and state-leak checks.
- Repo-scoped self-hosted ARC runner configuration for portable Linux/amd64 CI;
  Apple Silicon hardware tests remain opt-in. `.github/workflows/ci.yml` now
  targets a GitHub-hosted runner (`ubuntu-latest`) by default; see
  `docs/repo-intake.md` for the current CI lane decision.
- TTS `Voice` protocol message (RAV-1504): a custom cloned-voice embedding may
  now condition a session at session start, before any `Text` message, as an
  alternative to the `voice=`/`voices=` query parameters — matching real
  `moshi-server`'s `py_module.rs::InMsg::Voice`. A `Voice` message arriving
  after generation has started is rejected with an explicit protocol `Error`
  (a deliberate, stricter-than-upstream deviation; see PROTOCOL.md).
- TTS `voices=` multi-voice blend (RAV-1552): up to 5 repeated `voices=` query
  parameters resolve and blend distinct voices, mirroring the real `Py`
  module's `voice_ca_src` (which the pinned Unmute client never itself
  exercises). Mutually exclusive with `voice=`.
- `bridge_tts_streaming_failures_total{reason=...}` and
  `bridge_stt_generation_failures_total{reason=...}` metrics (RAV-1552),
  labeled `generation` (unexpected model failure) or `length_limit` (the
  session reached its configured `max_gen_length`/`max_steps` — a legitimate
  terminal condition, distinct from a crash).
- `bridge_tts_rejected_sessions_total{reason=...}` and
  `bridge_stt_rejected_sessions_total{reason=...}` metrics (RAV-1552):
  every pre-`Ready` rejection path in each server's `handle_connection` now
  increments a labelled counter (TTS: `query`, `format`, `voices`, `loading`,
  `cfg_alpha`; STT: `loading`), not just `cfg_alpha` as before. See
  `docs/observability.md`.

### Fixed

- The default STT checkpoint is now `kyutai/stt-1b-en_fr-candle`. Unlike the
  MLX-only checkpoint, it includes the extra heads stock Unmute requires for
  the `Step.prs[2]` semantic pause signal; the weights still execute through
  `moshi-mlx`.
- TTS now emits `Ready` immediately after channel admission and initializes an
  uncached voice off the event loop. This keeps first-use voice downloads from
  exceeding stock Unmute's 500 ms startup budget.
- `moshi-mlx`/`mlx` are marker-gated to `sys_platform == 'darwin' and
  platform_machine == 'arm64'` in `pyproject.toml`, and every runtime import of
  them in `stt/engine.py` / `tts/engine.py` is deferred into the function that
  actually needs it (module-level imports removed). Without this, adding real
  MLX inference would have broken `uv sync --locked` on the portable Linux CI
  lane entirely — `mlx`'s Linux wheel still requires the
  macOS-only `mlx-metal` backend package, so it cannot resolve on Linux at all.
  This matches the design spec's own requirement that MLX imports stay behind
  explicit adapter construction so the portable suite runs on Linux CI.
- `TTS_CFG_COEF` now defaults to `2.0` (matching upstream `moshi-server`'s
  `tts.toml`), not the engine's previous hardcoded `1.0`, which rendered
  synthesized voices nearly flat (RAV-1552). Validated as a positive, finite
  number at config-parse time (naming the variable in the error, unlike a
  bare `float()` parse failure); the loaded model's specific supported set is
  still checked at model load / per-request `cfg_alpha=`.
- Streaming-mode (TTS) and per-frame (STT) generation failures during an
  active session now emit a protocol `Error` and close cleanly instead of an
  abrupt, uncaught-exception close with no `Error` (RAV-1552). This covers
  both TTS delivery modes' trailing-flush paths and STT's per-Audio-frame
  path.
- Adversarial-review hardening (RAV-1552), ahead of this repo going public as
  a drop-in `moshi-server` replacement:
  - No client-visible `Error` message ever includes raw exception text, file/
    cache paths, or hostnames again. An unresolvable `voice=`/`voices=` entry
    now names only the client-supplied voice; a malformed frame, an
    unexpected `Voice`-embedding failure, and unexpected `TtsSession`
    construction failures all get a fully generic message instead. The full
    detail is always logged server-side.
  - `TtsSession.apply_voice_embedding`'s two `mlx`/`moshi_mlx`-dependent
    steps (tensor construction and conditioning) no longer interpolate the
    underlying third-party exception's own text into `VoiceEmbeddingError`
    (RAV-1552 F2) — a tensor-build failure could previously reach the client
    verbatim, e.g. `mlx`'s own "Cannot broadcast array of shape (2,1,3,2)
    into shape (1,1,3,2)". Both now raise a fixed, generic message; the
    detail is still logged server-side.
  - A non-numeric or repeated TTS query value (`?seed=abc`, `?cfg_alpha=1&
    cfg_alpha=2`, `?voice=a&voice=b`, …) now gets a protocol `Error` naming
    only the parameter, and a clean close, before `Ready` is ever sent
    (RAV-1552 F1/F6) — `_parse_query`'s bare `int()`/`float()` calls used to
    run outside every `try` in `handle_connection`, so an unparsable numeric
    value crashed the socket with 1011 and no `Error`; a repeated value
    silently took only the first occurrence with no signal to the client at
    all.
  - An unsupported `cfg_alpha=` now gets a protocol `Error` (naming the valid
    set) and a clean close, validated before a channel slot is even taken —
    it previously reached `TtsSession` construction uncaught, killing the
    socket with 1011 and no `Error`.
  - A single unresolvable `voice=` now gets the same `Error` treatment as an
    unresolvable `voices=` blend entry — it previously reached a bare
    `voice_resolution.result()` uncaught, with the same 1011 failure mode.
  - `voice=`/`voices=` query-string parsing now keeps blank values instead of
    silently dropping them: a blank `?voices=` used to fall back to the
    single default voice with no signal to the client at all. Blank entries
    and duplicate entries in `voices=` are now explicit protocol errors.
  - `TtsModelBundle.load` now resets the live classifier-free-guidance pass
    (`tts_model.cfg_coef`) to `1.0` unconditionally, not only for a
    CFG-distilled model — a non-distilled model previously kept whatever
    `cfg_coef` the constructor received (e.g. the config-default `2.0`),
    silently doubling batch size/compute on every generation step.
  - `stt/server.py`'s per-frame exception handling now excludes
    `ConnectionClosed` from its generic failure branch, matching the TTS call
    sites' existing symmetry (defense in depth; not a fix for an observed
    crash).
  - A client connecting after model load has actually failed now gets
    `"model failed to load"` instead of `"model still loading"` forever; the
    process still stays up serving health probes (`/readyz`'s
    `model_load_error` field) rather than exiting — see README.md's "If model
    load fails" section for the documented rationale.
  - Both CI workflows (`.forgejo/workflows/ci.yml`, `.github/workflows/
    ci.yml`) now set `timeout-minutes` so a hung test (this PR's own
    regression-tested `voices=` unresolvable-name conformance test hung,
    rather than failed, when its fix was reverted) bounds the job instead of
    queuing forever.
  - A repeated `?format=a&format=b` or `?auth_id=a&auth_id=b` TTS query value
    is now rejected the same way `seed=`/`top_k=`/`temperature=`/`cfg_alpha=`/
    `max_seq_len=`/`voice=` already were (RAV-1552): `_parse_query` previously
    took `raw[field_name][0]` unconditionally for these two fields, silently
    dropping any repeat with no signal to the client — the same F6
    silent-drop failure mode, just missed for these two. `observability.py::
    check_auth`'s pre-upgrade `auth_id=` gate gets the matching fix (fail
    closed on a repeat, instead of `values[0]`), so the gate and the
    post-upgrade parser can never disagree about whether a given `auth_id=`
    query string is acceptable.
  - A syntactically valid but out-of-range TTS numeric query value — a
    negative `seed`, a `top_k`/`max_seq_len` of zero or less, or a
    non-finite (`nan`/`inf`) or negative `temperature` — is now rejected with
    an explicit `Error` naming only the parameter (RAV-1552), the same
    fail-fast shape as an unparsable value. These previously parsed
    successfully and reached `TtsSession`/the underlying sampler unchecked.
    `cfg_alpha=` is unaffected — it is range-checked separately, against the
    loaded model's actual supported set.
  - A *blank-padded* repeat (`?format=&format=PcmMessagePack`, `?auth_id=&
    auth_id=good`, `?seed=&seed=7`, `?max_seq_len=&max_seq_len=5`, …) used to
    bypass the repeated-value guard added above (RAV-1552), confirmed live
    against the real `TtsServer`: `_parse_query`'s repeat-count checks ran on
    `parse_qs`'s default (blank-dropping) parse, so the blank entry vanished
    before `len(values) != 1` ever saw it, leaving exactly one surviving
    value and silently sending `Ready` instead of rejecting. Every query
    field is now parsed with blank values kept (previously scoped to just
    `voice=`/`voices=`, RAV-1552 S1), and a lone blank `format=`/`auth_id=`
    is rejected explicitly (there is no parse step for these two to fail on
    an empty string, unlike the numeric fields). `observability.py::
    check_auth`'s pre-upgrade gate gets the same `keep_blank_values=True`
    fix — it used to authorize `?auth_id=&auth_id=good` against an allowed
    id, the exact query `_parse_query` already rejected post-upgrade.
  - A client-supplied `max_seq_len=` can no longer raise the operator's
    configured `TTS_MAX_GEN_LENGTH` cap, only lower it (RAV-1552):
    `?max_seq_len=1000000000` previously replaced the cap unbounded. Rejected
    with an explicit `Error` the same way any other out-of-range value is.
  - Every pre-`Ready` rejection branch in `handle_connection` (both servers)
    now ignores `ConnectionClosed` when sending its `Error` (RAV-1552): a
    client that disconnects on its own between the upgrade and the rejection
    used to make that `send` raise uncaught, propagating out of
    `handle_connection` with no clean close — the metric increment (see
    above) still records the rejection either way.

### Known limitations

- Single active session per process (documented, see `PROTOCOL.md`).
- Ogg/Opus framing and TTS custom voice-embedding cloning are not implemented
  (documented non-goals; Unmute's own backend client never uses either).
- Not yet run as a full microphone-to-speaker canary — see README §What's
  proven / what's not. Stock Unmute process compatibility is proven.

## [0.1.0] — planned

Initial scaffold posture. Not yet released.

### Added

- Python package seed under `src/unmute_mlx_bridge/`.
- Smoke test confirming `main` is callable.
- Development shell via `flake.nix`.
