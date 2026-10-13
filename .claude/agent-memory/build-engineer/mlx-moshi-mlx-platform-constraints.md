---
name: mlx-moshi-mlx-platform-constraints
description: mlx/moshi-mlx have no usable Linux wheels in practice; must marker-gate deps and defer imports or portable Linux CI breaks
metadata:
  type: project
---

`mlx` (Apple's ML framework, used by `moshi-mlx`) publishes an `mlx` wheel for
`manylinux_2_35_x86_64` for some versions, but `mlx` unconditionally depends on
`mlx-metal` (the actual native backend), which **only ships macOS wheels**.
`moshi-mlx` additionally pins `mlx<0.27` (as of `moshi-mlx==0.3.0`), and that
version range has *no* Linux `mlx-metal` wheel at all. Net effect: `mlx`
cannot actually be installed on Linux regardless of what `mlx`'s own wheel
tags claim.

**Why it matters:** if a repo's portable/default CI lane runs on
`ubuntu-latest` (common for Ravenhelm Python services — ADR/AGENTS.md often
say "portable CI, no model weights, runs on Linux"), adding `moshi-mlx`/`mlx`
as unconditional `[project.dependencies]` breaks `uv sync --locked` outright
on that lane, not just at runtime.

**How to apply — do both, not just one:**

1. **pyproject.toml**: marker-gate the packages so uv doesn't even try to
   resolve/install them off-platform:
   ```toml
   "moshi-mlx>=0.2.6; sys_platform == 'darwin' and platform_machine == 'arm64'",
   "mlx>=0.26; sys_platform == 'darwin' and platform_machine == 'arm64'",
   ```
   Verify with `uv lock` then `grep` the generated `uv.lock`'s
   `[package.metadata].requires-dist` for the `marker = "..."` field — that's
   the actual proof the gate took effect, not just eyeballing the toml.
2. **Source code**: any module that does `import mlx.core` / `from moshi_mlx
   import ...` at module scope becomes unimportable on a platform without
   those wheels — which breaks even *portable* tests that only import the
   module to reach unrelated code (e.g. a WebSocket server class that happens
   to live in the same file as the MLX inference engine). Fix: move those
   imports into the function bodies that actually touch MLX (model-load
   classmethods, per-step inference methods), and use `if TYPE_CHECKING:` +
   `from __future__ import annotations` for type-hint-only references so
   dataclass field annotations etc. don't need the real import either.

**Verification without a live Linux box:** `uv sync --locked
--python-platform linux --python <a-throwaway-venv>` simulates the resolution
+ install on Linux. In one session this hung indefinitely because of
degraded sandbox network (see [[network-degradation-in-sandbox]] if that
memory exists) — the *lockfile marker* check (`uv lock` + grep) is the
reliable fallback verification when a live cross-platform install can't
complete in reasonable time.

See [[moshi-server-protocol-verification]] for the broader Kyutai/Unmute
project this came up in (`nwalker85/unmute-mlx-bridge`).
