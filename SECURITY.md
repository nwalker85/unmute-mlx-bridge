# Security

## Reporting Vulnerabilities

**Please report security issues privately, not as a public issue.**

Use GitHub's private vulnerability reporting: go to the **Security** tab →
**Report a vulnerability**. That opens a private advisory visible only to
maintainers.

Please include: what you found, how to reproduce it, the version or commit, and
what an attacker could do with it. You'll get an acknowledgement; this is a
small project maintained in spare time, so please allow reasonable time for a
fix before public disclosure.

> **Maintainer note — must be done before or at publication:** private
> vulnerability reporting is **opt-in** and is not enabled by default. Enable it
> at **Settings → Advanced Security → Private vulnerability reporting**. Until
> that toggle is on, the Security tab offers reporters no private channel and
> the instructions above do not work.

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
