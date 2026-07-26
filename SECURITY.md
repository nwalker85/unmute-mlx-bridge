# Security

## Reporting Vulnerabilities

This repository is in **private incubation**. GitHub private vulnerability
reporting must be enabled before publication; it is not yet active.

During private incubation, security reports should be sent through an
established private channel directly to the repository owner. Do not open
public issues for security vulnerabilities.

## Sensitive Data

Do not commit:

- Secrets, tokens, API keys, or credentials of any kind.
- `.env` files or any file containing environment secrets.
- Raw forensic evidence or recordings.
- Cookie values or session tokens.
- Customer data or private exports.
- Generated evidence bundles.
- Private hostnames, internal network topology, or infrastructure identifiers.
- Model weights or any artifact derived from private model runs.

## Supported Versions

This project is in private incubation at `0.y.z` under SemVer v2. Security
fixes will be applied to the current development head. Supported version policy
will be documented at first public release.

## Authentication Notes

The bridge accepts an optional `kyutai-api-key` header on loopback by default.
Binding to a non-loopback interface requires a configured constant-time token
check. Tokens come from environment variables and must never be logged, traced,
or committed.

## Model Weight License

This project does not redistribute Kyutai model weights. Model weights retain
their original CC-BY-4.0 license. Refer to the upstream Kyutai repositories
for model licensing terms.
