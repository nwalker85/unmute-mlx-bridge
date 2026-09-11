# Architecture Decision Records

Use this folder for ADRs.

Create an ADR when a decision affects:

- Public APIs or SDKs.
- Schema or event compatibility.
- Version labels.
- Deployment model.
- Security posture.
- Data custody or privacy boundaries.

ADR filenames should use:

```text
NNN-short-title.md
```

## Decisions and pending work

The following decisions are tracked and will require ADRs before their
respective milestones:

- Downstream production cutover (requires evidence-backed ADR and explicit
  approval before any integration with a production voice pipeline).
- Repository public-release posture: accepted in
  [ADR-0001](0001-public-release.md). GitHub is canonical and public; the ADR
  records an implementation variance because the repository was published
  with existing history rather than the planned fresh-history export.
- Any change to the MessagePack wire contract that breaks the pinned Unmute
  compatibility baseline.
