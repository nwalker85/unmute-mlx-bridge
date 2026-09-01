# Publication Export Checklist — unmute-mlx-bridge

Private-ops detail level. This checklist itself carries by-name references to
Forgejo-private-only documents; it is not part of the public export. See
`docs/repo-intake.md` → "Publication Export Plan" for the durable, exportable
summary of the decision this checklist supports.

## By-name exclusion list

Excluded from the public export (Forgejo-private only — internal ops
runbooks, not user-facing documentation):

- `docs/superpowers/plans/2026-07-30-single-profile-buffered-tts.md`
- `docs/superpowers/specs/2026-07-29-tts-codebook-depth-design.md`
- `docs/superpowers/specs/2026-07-30-single-profile-buffered-tts-design.md`
- `docs/superpowers/specs/2026-07-29-tts-query-fidelity-design.md`
- `docs/superpowers/specs/2026-07-30-tts-fidelity-lessons.md`
- `docs/superpowers/plans/2026-07-29-tts-query-fidelity.md`

These contain internal infrastructure identifiers, SSH aliases, and canary
filesystem paths. `docs/design/architecture.md` (formerly
`docs/superpowers/specs/2026-07-26-unmute-mlx-bridge-design.md`) is the core
architecture doc (linked from `README.md`) and IS included in the public
export — it was verified clean of such references
(`gitleaks detect --log-opts="--all"`, 2026-08-22: 0 leaks; plus a pattern
sweep for internal infrastructure identifiers).

Three of the six files above (the first three in the list) previously had
filenames that named an internal reference host. They were renamed under
RAV-1613 to their current, neutral names because this checklist — which
does ship in the public export via `.agents/` — would otherwise put that
hostname in front of every public reader, even though the files themselves
stay excluded. Content is unchanged; only the filenames moved.

## The unmerged bootstrap branch

The unmerged `forge-ci-bootstrap` branch (internal CI bootstrap scaffolding
referencing an internal build host, a private package registry, and a private
token path) is excluded by construction — the export only ever walks `main`.
Left as a private branch pending a separate prune decision.

## Concrete internal-identifier pattern list

Deliberately not written here in full (this file ships publicly via
`.agents/`, and a list of internal hostnames would itself be exactly the kind
of leak this checklist exists to prevent). Before the actual visibility flip,
re-run against the exact commit being exported:

- `gitleaks detect --source . --log-opts="--all"`
- `git secrets --scan-history`
- A grep sweep for this estate's private hostname/IP conventions (ask Nate,
  or pull the pattern list from the private Doom/memory continuity surface —
  not from any file in this repository).

## Pre-flip checklist

- [ ] Fresh squashed history generated from the sanitized `main` tree (not a
      force-push rewrite of existing history).
- [ ] All six by-name-excluded files above confirmed absent from the exported
      tree and its history.
- [ ] `forge-ci-bootstrap` branch confirmed not walked by the export.
- [ ] `gitleaks`, `git secrets --scan-history`, and the internal-identifier
      grep sweep all clean on the exact exported commit.
- [ ] Final visibility diff reviewed and explicit approval obtained from Nate
      (see `AGENTS.md`'s Publication Gate and `.agents/checklists/release.md`).
