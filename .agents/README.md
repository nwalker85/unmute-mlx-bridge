# Agent Workspace

This directory holds repo-specific agent working context. Durable product
documentation belongs in `docs/`; authoritative repo rules belong in
`AGENTS.md`.

## Layout

- `context/` — repo map, entry points, test commands, deploy surfaces.
- `checklists/` — PR, release, privacy, smoke, and review gates.
- `plans/implementation/` — active implementation plans and multi-agent
  handoffs.
- `archive/` — completed or superseded agent plans.

Recommended when useful:

- `contracts/` — frozen API, route, schema, event, or evidence contracts.
- `prompts/` — reusable subagent prompts.
- `runbooks/` — copy-paste-safe operational commands.
- `fixtures/` — sanitized payload examples only.

## Boundaries

- Do not store secrets, `.env` files, raw evidence, private exports, customer
  data, generated evidence bundles, or cookie values.
- No private consumer names, hostnames, topology, logs, or recordings.
- Keep files short enough for subagents to read before working.
- Prefer explicit ownership, commands, and stop conditions.
