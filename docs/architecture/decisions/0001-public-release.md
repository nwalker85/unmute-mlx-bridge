# ADR-0001: Public release posture

## Status

**Accepted, with implementation variance recorded 2026-09-10.** Nate approved
the public repository and GitHub-primary authority direction, and the
repository is now public. The visibility change was made in place with its
existing Git history rather than through the fresh-history export specified
below. That history decision remains part of the record; changing published
history now would be a separate destructive action requiring explicit approval.

## Context

`unmute-mlx-bridge` was developed as a private Forgejo-canonical repository
(`nate/unmute-mlx-bridge`) with a private GitHub passive mirror
(`nwalker85/unmute-mlx-bridge`). This ADR recorded the decisions required to
make the source public without weakening attribution, protocol conformance,
privacy, or release governance.

## Decisions

### 1. History mechanism: fresh squashed history, not a rewrite of `main`

At cutover, the sanitized tree is exported as a new (single- or few-commit)
history into the existing GitHub repository (`nwalker85/unmute-mlx-bridge`),
replacing whatever private-mirror history that repository currently holds.
The full private Forgejo development history — agent working notes, canary
iteration, and the then-current private publication checklist — stays on
Forgejo only and is never pushed to the public remote. This is **not**
`git push --force` of the existing `main` history to GitHub; it is a new
history generated from the sanitized tree. The original checklist and export
procedure remain recoverable from repository history as evidence of the
decision that was made at the time.

**Implementation variance:** the GitHub repository was made public in place
with its existing history, and old development branches also remained public.
The current public tree has been cleaned of private-incubation working plans,
but deletion does not erase them from Git history. This ADR does not authorize
a force-push, history rewrite, or public-branch deletion; each would require a
separate plan and explicit approval.

### 2. Authority flips: GitHub becomes canonical, Forgejo becomes mirror

The authority boundary after publication is:

- GitHub (`nwalker85/unmute-mlx-bridge`) is canonical — the place issues, PRs,
  and reviews happen, with automatic portable GitHub Actions checks.
- Forgejo is a pull/DR mirror, receiving history from GitHub rather than
  the other way around.
- `.forgejo/workflows/ci.yml` is not the public project's CI surface.

`AGENTS.md`, `package-surface.json`, and the lifecycle documentation now state
this current boundary directly.

### 3. Agent surfaces ship publicly

`AGENTS.md`, the reusable portions of `.agents/`, and
`package-surface.json` are included in the public repository rather than
stripped out as internal-only scaffolding.
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

- GitHub is the public source of truth for issues, pull requests, CI, and
  releases; Forgejo is a disaster-recovery mirror.
- Repository-wide agent instructions and purpose-built GitHub custom agents
  are public, reviewable project interfaces.
- Portable CI is automatic. Real-model Apple Silicon proof is manual and
  trusted-only because it executes contributed code while using large cached
  model artifacts.
- Publication of source does not imply a PyPI release, production deployment,
  or downstream cutover. Those remain separately approved events.
- The published-history variance is visible and unresolved. Normal cleanup may
  improve the current tree, but destructive history remediation is outside
  this ADR's authority.
