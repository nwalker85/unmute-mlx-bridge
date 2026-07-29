# Release Checklist

- [ ] PR merged to the release branch.
- [ ] CI passed for the exact merge SHA on GitHub Actions.
- [ ] `package-surface.json` was reviewed for:
  - `registry_target` — still `none` until publication approval
  - `publication_status` — `private_incubation` until explicit approval
  - `semver_policy` — `semver-v2`
  - `ci_lane` — `github-actions`
  - `runner_label` — `github-hosted:ubuntu-latest`
  - `nix_flake` — `flake-check-required-before-publish`
- [ ] `CHANGELOG.md` has a dated entry for the release version.
- [ ] Breaking changes are labeled as MAJOR (or MINOR for `0.y.z`).
- [ ] Migration notes exist for incompatible changes.
- [ ] ADR exists for major compatibility or conformance decisions.
- [ ] Artifact built for the exact merge SHA (if applicable).
- [ ] For any publication: complete the publication gate in `AGENTS.md` and
      obtain explicit approval from Nate.
- [ ] Final status separates merged, deployed, and live-verified states.

## Pre-publication gate (additional, before visibility change)

- [ ] Green protocol and portable CI on the exact proposed public commit.
- [ ] Clean secret scan across entire Git history.
- [ ] No private hostnames, credentials, topology, issue references, internal
      logs, or identifiable voice recordings anywhere in history.
- [ ] Apache-2.0 notices, Kyutai attribution, model-license guidance, and
      unofficial-community-project disclaimer complete.
- [ ] Reproducible install and canary instructions require no private infra.
- [ ] README, SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md reviewed.
- [ ] Final visibility diff reviewed and explicit approval from Nate obtained.
