# Release Checklist

## Source Release

- [ ] Release version follows SemVer v2 and is reflected in project metadata.
- [ ] `CHANGELOG.md` has a dated section with breaking changes called out.
- [ ] The proposed tag resolves to the exact reviewed and tested commit.
- [ ] Portable CI is green for that commit.
- [ ] Hardware or end-to-end claims name their separate evidence and do not
      borrow authority from portable CI.
- [ ] README installation, compatibility, attribution, and license guidance is
      current.
- [ ] `package-surface.json` still truthfully describes registry and artifact
      publication.
- [ ] Generated release notes have been reviewed rather than accepted blindly.
- [ ] Nate has explicitly approved this GitHub Release.
- [ ] Final reporting separates merged, released, deployed, and live-verified.

## Additional Gates

- Publishing to PyPI, a container registry, or another package channel requires
  an explicit target, credentials plan, provenance, rollback plan, and separate
  approval.
- A downstream production integration or cutover requires an evidence-backed
  ADR and explicit cutover approval. A source release does not authorize it.
- Rewriting published history or deleting public branches is destructive and
  requires a separate remediation plan and explicit approval.
