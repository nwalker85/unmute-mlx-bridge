# Security Policy

## Reporting A Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting: open the repository's
**Security** tab and choose **Report a vulnerability**. The report creates a
private advisory visible only to repository maintainers.

Include the affected version or commit, reproduction steps, expected impact,
and any suggested mitigation. This is a small community project maintained in
spare time; reports will be acknowledged as promptly as possible, and reporters
are asked to allow reasonable remediation time before disclosure.

## Supported Versions

| Version | Supported |
|---|---|
| Current `main` | Yes |
| Latest `0.1.x` release | Yes |
| Older snapshots | Best effort |

The project is pre-1.0. Security fixes target current `main` and the latest
published `0.1.x` release when one exists.

## Sensitive Data

Do not commit or publish:

- secrets, tokens, API keys, credentials, or `.env` files;
- raw recordings, customer data, private exports, or generated evidence;
- cookies, session tokens, private hostnames, or infrastructure topology;
- model weights or artifacts derived from private model runs.

## Authentication Notes

The bridge accepts an optional `kyutai-api-key` header on loopback by default.
Binding to a non-loopback interface requires a configured constant-time token
check. Tokens come from environment variables and must never be logged, traced,
or committed.

## Model And Voice Licenses

This repository does not redistribute Kyutai model weights or voices. They
retain their upstream licenses. See [NOTICE](NOTICE) and the README's voice
licensing section before redistributing generated or source material.
