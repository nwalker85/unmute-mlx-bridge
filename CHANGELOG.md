# Changelog

All notable changes to this project are documented here.

This project follows Semantic Versioning once it exposes a public contract,
deployable artifact, API, SDK, schema, or specification.

Pre-1.0 releases use `0.y.z` under SemVer v2. Breaking changes are documented
even under `0.y.z`. See `docs/architecture/decisions/` for decisions affecting
compatibility or version labels.

## [Unreleased]

### Added

- Initial repository scaffold: governance files, package seed, and portable CI.
- `docs/repo-intake.md` with lifecycle decisions.
- `package-surface.json` recording GitHub-primary exception posture.
- `.github/workflows/ci.yml` using GitHub Actions for portable CI.
- Repo-scoped `norns-unmute-mlx-bridge` ARC runner configuration for portable
  Linux/amd64 CI; Apple Silicon hardware tests remain opt-in.

## [0.1.0] — planned

Initial scaffold posture. Not yet released.

### Added

- Python package seed under `src/unmute_mlx_bridge/`.
- Smoke test confirming `main` is callable.
- Development shell via `flake.nix`.
