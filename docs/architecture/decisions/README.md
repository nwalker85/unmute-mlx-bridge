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
- Repository visibility change from private to public: drafted as
  [ADR-0001](0001-public-release.md), status **Proposed**. That ADR records
  the sub-decisions this work depends on (history mechanism, authority flip,
  agent-surface publication, voice licensing posture, `cfg_coef` conformance)
  but does not itself authorize the flip — the publication gate in
  `AGENTS.md` and `.agents/checklists/release.md`'s pre-publication checklist
  are still the actual approval path, and require explicit approval from
  Nate.
- Any change to the MessagePack wire contract that breaks the pinned Unmute
  compatibility baseline.
