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

## Pending decisions

The following decisions are tracked and will require ADRs before their
respective milestones:

- Downstream production cutover (requires evidence-backed ADR and explicit
  approval before any integration with a production voice pipeline).
- Repository visibility change from private to public (requires publication
  gate completion and explicit approval).
- Any change to the MessagePack wire contract that breaks the pinned Unmute
  compatibility baseline.
