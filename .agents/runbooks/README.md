# Agent Runbooks

Copy-paste-safe operational commands for agent use.

Durable operational procedures should graduate to `docs/runbooks/`. Use this
directory for short-lived or agent-specific command notes.

## Useful one-liners

```bash
# Install and run portable tests
uv sync --locked && uv run --locked pytest -q

# Check for whitespace issues
git diff --check

# Check for whitespace issues against HEAD (staged)
git diff --cached --check

# List files changed relative to main
git diff --name-only main

# Verify package-surface.json is valid JSON
python3 -c "import json; json.load(open('package-surface.json'))" && echo ok
```
