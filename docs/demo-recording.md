# Recording the demo

The README leads with the claim that Unmute's speech-to-speech stack runs on a
Mac. A demo recording is the single highest-value thing that can back that
claim — for this project, showing it is worth more than any amount of prose.

This document is the spec for that recording. It exists so the demo proves the
claim honestly rather than flattering it.

## What the demo must show

The claim is *"stock Unmute, unmodified, running against MLX servers on a Mac."*
A recording that doesn't establish all four of these proves something weaker:

1. **It's a Mac.** Show the machine, or the menu bar and About This Mac, or a
   terminal running `sysctl -n machdep.cpu.brand_string`. Don't just assert it.
2. **The servers are the bridge.** Show both processes starting —
   `uv run unmute-mlx-stt` / `uv run unmute-mlx-tts` — and reaching `/readyz`.
3. **Unmute is unmodified.** Show `git status` clean on an upstream Unmute
   checkout, ideally with `git log -1` showing an upstream commit. This is the
   whole differentiator; a viewer can't take it on faith.
4. **A real conversation.** Uncut. Speak, get a response, speak again.

## What the demo must not do

- **Do not cut the latency out.** TTS on this hardware generates below real
  time (see the README's Performance envelope). If there's a pause before the
  assistant speaks, the pause stays in the recording. Trimming it turns an
  honest demo into a misleading one, and the README's credibility rests
  entirely on not doing that.
- **Do not speed up the video** — same reason. If length is a problem, have a
  shorter conversation, not a faster one.
- **Do not stage a scripted "perfect" exchange** and present it as typical. If
  it took several attempts, say so in the caption.
- **Do not show a canary or deployment that isn't reproducible** from this
  repo's public instructions. A demo of private infrastructure proves nothing a
  reader can repeat.

## Privacy checklist

A voice demo is personal data. Before recording, and again before publishing:

- [ ] The voice in the recording belongs to someone who has agreed to it being
      published publicly and permanently.
- [ ] No private hostnames, IPs, or internal service names visible in any
      terminal, browser tab, URL bar, or window title.
- [ ] No tokens, API keys, or `.env` contents on screen.
- [ ] Nothing identifying in the spoken content — no names, addresses, or
      details about real people.
- [ ] Browser tabs, notifications, Slack/Messages popups, and desktop contents
      reviewed frame by frame.
- [ ] Terminal scrollback doesn't reveal earlier private work.

Record in a clean user session or a fresh browser profile — it's far easier
than auditing a messy one.

## Format

**Preferred: MP4 with audio.** This is a speech project; a silent GIF cannot
demonstrate speech. Audio is the point.

- Under ~60 seconds. Attention is short and the claim is simple.
- 720p is enough and helps you fit the size cap.
- **Size caps** (GitHub attachments): video is **10 MB on free plans**, 100 MB
  on paid — keyed to the repo owner's plan. Images and GIFs are 10 MB. A 10 MB
  budget is roughly 30–60s at 720p with sane encoding. If it doesn't fit,
  shorten the demo; don't degrade it into unwatchability.
- Accepted video formats: MP4, MOV, WebM.

**Do not commit the video to the repository.** Video in git history is
permanent weight on every future clone. Use a GitHub attachment (below).

## Embedding it

1. Open a new issue or PR comment in this repository. **You do not have to
   submit it.**
2. Drag the `.mp4` into the comment box; wait for the upload to finish.
3. GitHub replaces it with a `https://github.com/user-attachments/assets/<uuid>`
   URL.
4. Paste that URL **on its own line** in `README.md` where the demo placeholder
   comment sits. A bare attachment URL on its own line renders as a native
   video player.
5. Discard the draft comment — the asset URL stays valid.

There is **no API for this**; minting an attachment URL is manual by design, so
it cannot be automated in CI.

### What actually renders (verified against GitHub's renderer, 2026-08-22)

| Syntax | Result |
|---|---|
| Bare `user-attachments` URL alone on its line | ✅ native player |
| `<video src="…user-attachments…"></video>` | ✅ native player |
| `[text](…user-attachments…)` | ⚠️ becomes a player; **your link text is discarded** |
| `<video src="https://raw.githubusercontent.com/…">` | ❌ **silently stripped** — renders as nothing |
| `<video><source src="…"></video>` | ❌ silently stripped — the nested `<source>` form never works |
| `![demo](…user-attachments…)` | ❌ broken-image icon |
| `<audio src="…">` | ❌ stripped |

Two traps worth stating plainly:

- The common claim that *"GitHub strips `<video>` from READMEs"* is **false**.
  `<video>` works — but **only** with a `user-attachments` src. Point it at a
  repo-committed file and it vanishes with **no error and no fallback**, which
  is why the folklore contradicts itself.
- `poster`, `width`, `loop`, `autoplay`, and `playsinline` are **all stripped**.
  You get click-to-play, muted, max-height 640px, and no styling control. If
  you need autoplay/looping, that's a GIF, not a video.

### Other caveats

- At render time GitHub rewrites the src to a short-lived signed
  `private-user-images.githubusercontent.com` URL that expires in ~5 minutes.
  **Never hardcode that** — always store the canonical `user-attachments` form.
- The asset lives outside the repo. It is a **dead link on PyPI, in a clone, in
  a tarball, and on any non-GitHub renderer.** Anything load-bearing should not
  be video-only.
- Attachment visibility follows the repository's. This repository is public,
  so a committed attachment URL is publicly viewable; verify the rendered
  README from a logged-out browser before treating the embed as proof.
- Caption it with the hardware, the delivery mode, and whether the pause you
  can hear is representative.

## Consider a chart instead — or as well

For infrastructure and ML projects, **a static chart with numbers on it is
often stronger proof than a screen recording**, because it is falsifiable and
it renders everywhere — mirrors, package registries, clones. `ruff` and `uv`
both lead with an SVG benchmark chart rather than a demo video.

This project has a natural candidate: the real-time-factor table in the
README's Performance envelope, drawn as a bar chart with the 1.0× real-time
line marked. It would show honestly that TTS sits below that line — which is
more credible than a video that quietly hides it.

Best combination: **static figure above the fold, video below it.**

## A still fallback

If a video isn't feasible, a terminal screenshot showing both servers ready,
`/readyz` returning 200, and a clean `git status` on an upstream Unmute
checkout still proves more than prose. Commit stills to `docs/images/` — small
enough that history weight isn't a concern, and they survive mirroring.
