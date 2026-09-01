# ADR-0001: Public release posture

## Status

**Proposed.** This ADR records the decisions this repository's public-release
work (RAV-1613) was implemented against. It does not itself authorize the
visibility flip, the history rewrite, or any cutover action — those require
Nate's explicit approval per `AGENTS.md`'s Publication Gate and
`.agents/checklists/release.md`'s pre-publication gate, separately from
accepting this ADR's text.

## Context

`unmute-mlx-bridge` has been developed as a private Forgejo-canonical
repository (`nate/unmute-mlx-bridge`) with a private GitHub passive mirror
(`nwalker85/unmute-mlx-bridge`). Nate has decided to take this repository
public. That decision has several sub-decisions that need to be on the
record before any cutover, because they affect what the codebase, docs, and
CI should look like *right now* (so the eventual flip is a visibility change
and a history export, not a scramble to retrofit governance) as well as after.

## Decisions

### 1. History mechanism: fresh squashed history, not a rewrite of `main`

At cutover, the sanitized tree is exported as a new (single- or few-commit)
history into the existing GitHub repository (`nwalker85/unmute-mlx-bridge`),
replacing whatever private-mirror history that repository currently holds.
The full private Forgejo development history — agent working notes, canary
iteration, and the private-ops files tracked in
`.agents/checklists/publication-export.md` — stays on Forgejo only and is
never pushed to the public remote. This is **not** `git push --force` of the
existing `main` history to GitHub; it is a new history generated from the
sanitized tree. See `docs/repo-intake.md` → "Publication Export Plan" and
`.agents/checklists/publication-export.md` for the exclusion list and
pre-flip sweep procedure this decision depends on.

### 2. Authority flips: GitHub becomes canonical, Forgejo becomes mirror

Today, Forgejo is canonical for development, PRs, CI, and review; GitHub is a
private passive mirror with dormant Actions (see `.github/workflows/ci.yml`'s
`on: workflow_dispatch` — automatic triggers are intentionally not enabled).
At the flip:

- GitHub (`nwalker85/unmute-mlx-bridge`) becomes canonical — the place issues,
  PRs, and reviews happen, and where `.github/workflows/ci.yml`'s triggers are
  widened past manual dispatch.
- Forgejo becomes a pull/DR mirror, receiving history from GitHub rather than
  the other way around.
- `.forgejo/workflows/ci.yml` (the repo-owned K3s-runner portable CI) is
  retired or repurposed for internal-only validation; it is not the public
  project's CI surface after the flip.

This is the reverse of the boundary this repo has operated under so far, and
`AGENTS.md` is rewritten (this same change) to describe both the current
phase and this flip explicitly, instead of stating only the current
direction as an absolute.

### 3. Agent surfaces ship publicly

`AGENTS.md`, `.agents/` (including
`.agents/checklists/publication-export.md`, itself kept at private-ops detail
level rather than excluded outright), and `package-surface.json` are included
in the public export, not stripped out as internal-only scaffolding.
Rationale: this project is meant to demonstrate an agent-native repository —
one where the operational contract for coding agents (entry points, test
commands, guardrails, release checklist) is itself part of the public
artifact, not a private appendix. The private-ops detail that must not ship
(internal hostnames, the by-name export-exclusion list's own targets, the
concrete internal-identifier grep patterns) is scoped narrowly to specific
files/sections rather than excluding the whole `.agents/` surface.

### 4. Voice licensing posture

The bridge's built-in default voice (`TTS_DEFAULT_VOICE`) is
`unmute-prod-website/p329_022.wav` — Nate's blind-audition pick. Despite
living under `unmute-prod-website/`, this specific file is VCTK speaker p329,
so it is licensed **CC BY 4.0, not CC0**: commercially safe, with the
attribution it requires ("Uses a voice from the VCTK corpus (CSTR, University
of Edinburgh), CC BY 4.0") recorded in `NOTICE` and README's "Voice
licensing" section. The accurate summary of this project's voice posture is:
**a commercially-safe default (CC BY 4.0, attribution provided), CC0
alternatives available (`unmute-prod-website/default_voice.wav`,
`voice-donations/*`), and the CC BY-NC 4.0 `expresso/`/`ears/` voices as an
explicit, non-commercial-only opt-in** — not a blanket CC0 or blanket CC BY
4.0 claim either way.

This is a **deliberate deviation from upstream `moshi-server`'s own default**
(`unmute-prod-website/default_voice.wav`, CC0) — a preference choice made
from Nate's own blind listening comparison, not a correctness fix or a
licensing-driven substitution. The previously-shipped default before this
ADR's work began, an `expresso/` sample, was CC BY-NC 4.0 (non-commercial
only) and was not appropriate as an out-of-the-box default for a public
release aimed at general use; that problem is what prompted the review that
led here, but the specific voice landed on (`p329_022.wav`, attribution
required) was chosen for how it sounds, not because it was the only
commercially-safe option — CC0 options existed and remain available.
`expresso/`/`ears/` voices remain available and fully supported — they are
opt-in via `TTS_DEFAULT_VOICE` or `?voice=`/`?voices=`, with the
non-commercial restriction documented at the point of use. See the RAV-1613
commits that made this change for the full per-directory licensing audit
against `kyutai/tts-voices`'s own README.

### 5. `cfg_coef=2.0` default is a conformance decision, not a tuning choice

`TTS_CFG_COEF` defaults to `2.0`, matching upstream `moshi-server`'s own
default and production Unmute's effective operating point (`cfg_alpha=1.5`
sent by `unmute/tts/voices.py`, both nonzero). This was fixed under RAV-1552,
prior to this ADR, but is recorded here as part of the public-release
posture because it is a compatibility conformance decision this project
holds itself to, not a free parameter: shipping any other default would mean
this bridge no longer sounds like the systems it claims wire-compatibility
with. A per-request `?cfg_alpha=` override remains available and always
takes precedence for a given session.

## Consequences

- Everything above is implemented in this repository's public-release work
  now, while the repo is still private, so that the flip itself is a
  visibility change plus a history export — not a scramble to retrofit
  governance, licensing correctness, or CI after the fact.
- `AGENTS.md` describes two states (current phase, and the flip) rather than
  a single absolute, which is unusual for that file's normal style; this is
  deliberate and should be preserved until the flip actually happens, at
  which point the "current phase" language is updated to match reality and
  the "at flip" language is retired.
- Nothing in this ADR authorizes the flip itself. The publication gate in
  `AGENTS.md` and the pre-publication checklist in
  `.agents/checklists/release.md` remain the actual approval path.
