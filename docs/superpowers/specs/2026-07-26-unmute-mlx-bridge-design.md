# Unmute MLX Bridge Design

**Status:** Approved for a canary spike

**Date:** 2026-07-26

**Repository:** `nwalker85/unmute-mlx-bridge`

**License:** Apache-2.0

**Visibility:** Private during incubation; publication requires explicit approval

## Purpose

`unmute-mlx-bridge` will provide persistent Apple Silicon MLX speech-to-text
and text-to-speech services that implement the model-server WebSocket contract
used by Kyutai Unmute.

Kyutai publishes production WebSocket servers for Linux/CUDA and MLX inference
implementations for Mac and iPhone. It does not currently publish an
Unmute-compatible production MLX server. This project fills that
interoperability gap without forking Unmute or embedding consumer-specific
orchestration.

The first consumer is a private Apple Silicon canary. That consumer does not
define the planned public API, and no private credentials, hostnames, topology,
logs, or recordings belong in this repository. The repository remains private
throughout incubation.

## Upstream Compatibility Baseline

Compatibility is pinned to these upstream revisions:

- `kyutai-labs/unmute@c49982eb3aeaf76633dfe4155fa3b8dcb5b3d962`
- `kyutai-labs/delayed-streams-modeling@4c4f65e147df056adf3346290d64c7b9649b18c9`
- `kyutai-labs/moshi@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`

Contract tests will encode the message shapes consumed by the pinned Unmute
revision. Upstream compatibility changes must update the pinned revision,
fixtures, and compatibility notes together.

## Goals

1. Run Kyutai STT and TTS persistently on Apple Silicon through MLX.
2. Allow the pinned stock Unmute backend to connect without source changes.
3. Preserve streaming transcripts, timestamps, semantic VAD signals, streaming
   audio, text timing, and cancellation behavior.
4. Expose operational health, readiness, capacity, and latency measurements.
5. Provide deterministic protocol tests that do not download model weights.
6. Provide opt-in hardware tests and a repeatable canary benchmark.
7. Publish a focused community project with clear attribution and contribution
   boundaries.

## Non-Goals

- Reimplementing or forking the Unmute browser or backend.
- Acting as an OpenAI-compatible LLM gateway.
- Owning tool calling, conversation state, prompts, or agent policy.
- Replacing downstream voice orchestration, LLM gateways, or tool runtimes.
- Redistributing Kyutai model weights.
- Claiming production readiness from component health checks alone.
- Supporting CUDA, Linux, Intel Macs, iOS, or multiple concurrent sessions in
  the first canary.
- Performing a downstream production cutover before an evidence-backed
  architecture decision and explicit cutover approval.

## System Boundary

The publication candidate owns only the two model services and their shared
protocol library:

```text
microphone
    |
    v
stock Unmute backend ------> external OpenAI-compatible LLM
    |                                      |
    |                                      +--> optional downstream tools
    |
    +--> MLX STT service  /api/asr-streaming
    |
    +--> MLX TTS service  /api/tts_streaming
                         |
                         v
                       speaker
```

Unmute continues to own browser audio transport, OpenAI Realtime-style events,
turn coordination, and interruption policy. The bridge owns only the
backend-to-model WebSocket contracts and MLX inference lifecycle.

## Process Architecture

STT and TTS run as separate long-lived processes:

- `unmute-mlx-stt` loads one STT model and serves port `8090` by default.
- `unmute-mlx-tts` loads one TTS model and serves port `8089` by default.
- `unmute_mlx_bridge.protocol` contains MessagePack models and validation shared
  by both services.
- `unmute_mlx_bridge.stt` adapts streaming audio frames to the MLX STT engine.
- `unmute_mlx_bridge.tts` adapts streaming text to the MLX TTS engine.
- `unmute_mlx_bridge.observability` provides health state and Prometheus metrics.

Separate processes provide model isolation, independent restarts, and honest
memory accounting. Each process loads its model once before reporting ready.
The first release admits one active WebSocket session per process. A second
connection receives a protocol `Error` message and closes cleanly rather than
waiting in an unbounded queue.

The default STT model is `kyutai/stt-1b-en_fr-candle`, loaded through
`moshi-mlx`. Live stock-Unmute verification showed that the similarly named
`kyutai/stt-1b-en_fr-mlx` checkpoint transcribes audio but omits the extra
heads required for the `Step.prs[2]` semantic-VAD pause signal. The default TTS
model is `kyutai/tts-1.6b-en_fr`. Model identifiers and TTS quantization are
configuration values. The reference canary starts TTS at 8-bit quantization
and records whether it is faster than real time before considering higher
fidelity.

## Wire Contracts

All model WebSocket messages are binary MessagePack maps. Audio samples are
24 kHz mono `float32` values represented as MessagePack arrays, matching the
pinned Unmute client.

### Speech-to-Text

Endpoint: `GET /api/asr-streaming`

On connection the server emits:

```json
{"type": "Ready"}
```

Accepted client messages:

- `{"type": "Audio", "pcm": [float, ...]}` streams mono samples.
- `{"type": "Marker", "id": integer}` requests marker round-trip correlation.

Emitted server messages:

- `{"type": "Word", "text": string, "start_time": float}`
- `{"type": "EndWord", "stop_time": float}`
- `{"type": "Marker", "id": integer}`
- `{"type": "Step", "step_idx": integer, "prs": [float, ...]}`
- `{"type": "Error", "message": string}`

The service validates finite samples, one-dimensional frame shape, and a
bounded frame size before inference. A marker is emitted only after all audio
submitted before that marker has crossed the inference boundary.

### Text-to-Speech

Endpoint: `GET /api/tts_streaming`

The endpoint accepts the pinned Unmute query parameters, including `seed`,
`temperature`, `top_k`, `format`, `voice`, `voices`, `max_seq_len`,
`cfg_alpha`, and `auth_id`. The canary supports `format=PcmMessagePack`; any
other format receives an `Error` and a clean close.

On connection the server emits:

```json
{"type": "Ready"}
```

Accepted client messages:

- `{"type": "Text", "text": string}` appends text.
- `{"type": "Eos"}` closes the utterance input.

`Voice` messages are parsed and rejected with an explicit unsupported-feature
`Error` in the first canary. This makes the contract honest without silently
ignoring custom voice embeddings.

Emitted server messages:

- `{"type": "Audio", "pcm": [float, ...]}`
- `{"type": "Text", "text": string, "start_s": float, "stop_s": float}`
- `{"type": "Error", "message": string}`

TTS begins generation after sufficient text is available; it does not wait for
the entire response or the `Eos` message. Text chunking is punctuation-aware
with a bounded maximum wait so a streaming LLM cannot starve first audio.

### Authentication Header

The bridge accepts the `kyutai-api-key` header because the stock Unmute client
sends it. Authentication is optional on loopback by default. Binding to a
non-loopback interface requires a configured constant-time token check. Tokens
come from environment variables and are never logged.

## HTTP Operations Endpoints

Each process exposes:

- `GET /healthz`: process is running.
- `GET /readyz`: model is loaded and the process can admit a session.
- `GET /metrics`: Prometheus text exposition.

Readiness returns a non-success status while a model is loading or the sole
session slot is unavailable. Metrics include model load duration, active
sessions, rejected sessions, input/output audio duration, time to first STT
word, time to first TTS audio, inference real-time factor, cancellation count,
protocol errors, and process resident memory.

No transcript, prompt text, audio, token, or model cache path appears in logs or
metric labels.

## Session Lifecycle and Cancellation

Each WebSocket owns one model session and one bounded inference queue.
Disconnecting the WebSocket cancels inference, drains queued frames, releases
the capacity slot, and records the cancellation. Shutdown stops admission,
closes active sockets, cancels inference, and exits after a bounded grace
period.

STT and TTS failures use three categories:

1. Protocol errors emit `Error` and close only the offending connection.
2. Capacity errors emit `Error` and leave the process ready for the current
   session.
3. Model failures emit `Error`, mark readiness false, and require a process
   restart rather than pretending the model remains usable.

## Packaging and Platform Contract

- Python version: `>=3.12,<3.13` for the first release.
- Platform: macOS 14 or newer on Apple Silicon (`arm64`).
- Package manager and lockfile: `uv`.
- Runtime: `moshi-mlx>=0.2.6`, MLX, NumPy, MessagePack, Pydantic, WebSockets,
  and Prometheus client.
- Test runner: `pytest` with `pytest-asyncio`.
- Formatting and linting: Ruff.
- Type checking: Pyright.

Installation fails early with a clear platform message on unsupported systems.
Model downloads use the standard Hugging Face cache and remain outside the
repository.

The source is Apache-2.0. Documentation attributes Kyutai, links the upstream
repositories and paper, distinguishes this community project from an official
Kyutai release, and tells users that model weights retain their CC-BY-4.0
license.

## Testing Strategy

### Protocol tests

Protocol tests start each service with an in-memory deterministic engine. They
verify exact MessagePack shapes, startup `Ready`, message ordering, marker
round trips, text streaming, `Eos`, unsupported voice errors, malformed frames,
capacity rejection, disconnect cancellation, and readiness transitions.

Captured client fixtures from the pinned Unmute revision provide the
compatibility oracle. Tests fail if a field name, type, endpoint, or lifecycle
order drifts.

### Engine adapter tests

Adapter tests exercise real chunking, queues, timestamp conversion, and
cancellation against deterministic engine fakes. MLX imports and model
downloads remain behind explicit adapter construction so the normal suite runs
on GitHub-hosted Linux CI.

### Apple Silicon hardware tests

Opt-in tests on a reference Apple Silicon host load the real models and verify:

- STT accepts a fixed 24 kHz fixture and emits the expected phrase with
  monotonically increasing timestamps.
- TTS accepts streamed text and emits non-silent audio before `Eos`.
- Both services complete three sequential sessions without increasing resident
  memory by more than 10 percent after the first warmed session.
- Cancellation releases capacity and the next session succeeds.

Hardware measurements are artifacts, not required GitHub checks.

## Canary Proof Boundary

The bridge is proven viable when all of the following are captured from one
pinned build on the reference Apple Silicon host:

1. The stock pinned Unmute backend connects to both model services without a
   source patch.
2. One real microphone turn completes microphone to STT to an external
   OpenAI-compatible LLM to streamed TTS to speaker.
3. One private-consumer canary reaches an external LLM, completes a safe tool
   call, and returns audible speech with correlated turn identifiers.
4. User speech during output cancels the active TTS stream and the next turn
   succeeds.
5. Twenty consecutive turns complete with no process restart, session leak, or
   unbounded memory growth.
6. The report includes component-level p50 and p95 latency, real-time factor,
   peak memory, exact source commits, installed package versions, model
   identifiers, and model-weight licenses.

The latency report is observational for the first canary. The proof does not
invent a pass threshold before measuring the reference hardware. Protocol
compatibility, end-to-end completion, interruption, repeatability, and bounded
resources are the hard pass criteria.

Internal service health alone is not canary proof. The evidence must cross the
authenticated browser or microphone boundary and return audible output.

## Delivery and Decision Sequence

1. Keep the GitHub repository private throughout development and canary testing.
2. Merge only reviewed project PRs with green portable CI.
3. Run the pinned bridge build locally on the reference Apple Silicon host
   under a supervised process.
4. Run the stock Unmute and standalone microphone canaries.
5. Run the private-consumer canary without changing its production default
   path.
6. Record a sanitized benchmark report inside the private repository.
7. If the hard proof criteria pass, write the downstream architecture decision
   using the measurements.
8. Seek explicit approval for the ADR, integration PRs, and any production
   cutover.
9. Complete the publication gate and seek separate explicit approval before
   changing repository visibility.

Failure remains a useful outcome. The report will identify whether the blocker
is model speed, memory pressure, MLX adapter behavior, protocol mismatch, or
integration latency, without retroactively weakening the proof criteria.

## Publication Gate

Changing the repository from private to public is an independent release
decision. Before asking for publication approval, the project must have:

- Green protocol and portable CI suites on the exact proposed public commit.
- A clean secret scan across the entire Git history.
- No private hostnames, credentials, topology, issue references, internal logs,
  or identifiable voice recordings.
- Complete Apache-2.0 notices, Kyutai attribution, model-license guidance, and
  an unofficial-community-project disclaimer.
- Reproducible installation and canary instructions that do not depend on
  private infrastructure.
- A reviewed README, security policy, contribution guide, code of conduct,
  issue templates, and release notes.
- A final visibility diff and explicit approval from Nate for publication.

Passing the private-consumer canary does not itself authorize publication.

## Community Contribution Model

After publication, the repository will welcome protocol fixtures, Apple
Silicon measurements, MLX adapter improvements, and compatibility reports.
Issues must include the hardware model, memory, macOS version, model identifier,
quantization, bridge commit, and upstream Unmute revision.

Project documentation will explicitly invite upstream coordination. The
project will not present itself as an official Kyutai component or use Kyutai
branding beyond factual compatibility and attribution.

## References

- [Kyutai Unmute](https://github.com/kyutai-labs/unmute)
- [Kyutai Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)
- [Kyutai STT 1B English/French Candle checkpoint](https://huggingface.co/kyutai/stt-1b-en_fr-candle)
- [Kyutai TTS 1.6B English/French model](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- [Delayed Streams Modeling paper](https://arxiv.org/abs/2509.08753)
