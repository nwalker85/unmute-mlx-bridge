# Changelog

All notable changes to this project are documented here.

This project follows Semantic Versioning once it exposes a public contract,
deployable artifact, API, SDK, schema, or specification.

Pre-1.0 releases use `0.y.z` under SemVer v2. Breaking changes are documented
even under `0.y.z`. See `docs/architecture/decisions/` for decisions affecting
compatibility or version labels.

## [Unreleased]

## [0.1.0] - 2026-08-31

### Performance

<!-- RTF numbers pending benchmark: FILLED 2026-08-31, see below -->

Re-measured 2026-08-31 at commit `3249e9e` on an Apple M4 Pro (Mac Mini),
~40s sustained continuous speech, in-process, `kyutai/tts-1.6b-en_fr`,
`n_q=32`, default voice: **q8 generates faster than real time** — 1.844× at
this bridge's default `cfg_coef=2.0` (time-to-first-audio 1.54s), 1.886× at
production Unmute's `cfg_alpha=1.5` (1.05s); unquantized at `cfg_coef=2.0` is
0.589× (1.58s), the profile `buffered_turn` is actually for. This inverts the
previous framing (TTS categorically "below real time") and supersedes —
rather than merely refines — the old "8-bit 0.594×" figure, which was
measured at `cfg_coef=1.0` and does not reproduce; today's unquantized number
lands almost exactly on that old figure instead, the likely real explanation.
See README.md's "Performance envelope" section for the full table, the
correction note, and `scripts/bench_rtf.py` to reproduce on your own
hardware.

### Changed

- **Breaking (behavior):** `TTS_DEFAULT_VOICE` now defaults to
  `unmute-prod-website/p329_022.wav` (RAV-1613) — Nate's blind-audition pick
  among candidates. The previous default,
  `expresso/ex03-ex01_happy_001_channel1_334s.wav`, is licensed CC BY-NC 4.0
  (non-commercial only). The new default is a **commercially-safe default
  (CC BY 4.0, attribution provided — see NOTICE)**: despite living under
  `unmute-prod-website/`, this specific file is VCTK speaker p329, not one of
  that directory's own CC0 recordings. This is a deliberate deviation from
  upstream `moshi-server`'s own default (`unmute-prod-website/
  default_voice.wav`, CC0) made for how it sounds, not a licensing
  necessity — CC0 alternatives remain available
  (`unmute-prod-website/default_voice.wav`, `voice-donations/*`) via
  `TTS_DEFAULT_VOICE`, and the CC BY-NC 4.0 `expresso/`/`ears/` voices remain
  opt-in only. A deployment that relied on the old default's specific voice
  identity and does not set `TTS_DEFAULT_VOICE` explicitly will hear a
  different voice after upgrading. See README.md's "Voice licensing" section
  for the full per-directory licensing table — voices are not uniformly CC
  BY 4.0 as an earlier version of this document implied.

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
- **Public release readiness (RAV-1613):**
  - `pyproject.toml` gains `license = "Apache-2.0"` (+ `license-files`),
    `readme`, `authors`, `keywords`, `classifiers`, and `[project.urls]`
    (Homepage/Repository/Issues/Changelog) pointing at the GitHub-canonical
    repo.
  - `examples/tts_client.py` and `examples/stt_client.py`: small,
    dependency-light WebSocket clients (`examples/README.md`), linked from
    the main README, generalized from this project's ad-hoc e2e test
    harness — argparse for URL/voice/file/auth, no absolute or
    host-specific paths.
  - `scripts/bench_rtf.py`: real-time-factor benchmark against the actual
    `TtsModelBundle`/`TtsSession` engine, printing host, commit,
    quantization, `n_q`, `cfg_coef`, audio/wall seconds, RTF, and
    time-to-first-audio. Marked "requires Apple Silicon + weights"; README's
    Performance envelope section now points users at it.
  - `.github/workflows/ci.yml` gains a `hardware` job on `macos-14` (Apple
    Silicon), running `pytest -m hardware -q -s` with the Hugging Face
    cache preserved via `actions/cache`. Stays dormant with the rest of
    that workflow (`on: workflow_dispatch`) until publication cutover; not
    added to `.forgejo/workflows/ci.yml` (no macOS runner capacity there).
  - `docs/architecture/decisions/0001-public-release.md`
    (status: **Proposed**): records the history-export mechanism, the
    Forgejo/GitHub authority flip, agent-surfaces-ship-publicly rationale,
    voice licensing posture, and the `cfg_coef=2.0` default as a
    conformance decision — not itself an approval to execute any of it.
  - `AGENTS.md` rewritten to describe both the current phase and the flip
    (per ADR-0001) instead of only the current direction as an absolute;
    guardrails unchanged.
  - The by-name publication-export exclusion list moved from
    `docs/repo-intake.md` into `.agents/checklists/publication-export.md`
    (private-ops detail level); three private-ops docs whose filenames
    previously named an internal reference host renamed to drop that
    hostname (content unchanged) since the exclusion list referencing them
    now ships publicly via `.agents/`.
  - The core design spec moved from
    `docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md` to
    `docs/design/architecture.md` (it is the one `superpowers/` doc that
    ships publicly); every cross-reference updated.
  - `docs/runbooks/deploy.md`: the first real runbook — launchd/PM2 service
    shape, environment table, ports, `/healthz`/`/readyz`/`/metrics`,
    load-failure behavior.

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

<!--
  A prior scaffold-only "[0.1.0] — planned" entry lived here, listing just
  the package seed, smoke test, and flake.nix. Removed as a duplicate
  version header once 0.1.0 became a real, dated release above — its
  content is already covered by this section's "Initial repository
  scaffold" bullet.
-->
