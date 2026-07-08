---
name: Python deps & production deploy quirks
description: Why Python deps must stay undeclared in pyproject here, and how production finds .pythonlibs packages
---

## Rule 1: keep `dependencies = []` in pyproject — declaring deps BREAKS publish

The publish pipeline's "Installing packages" step runs `uv lock` + `uv sync`. In this
workspace `.pythonlibs` is a PYTHONUSERBASE-style directory (no `pyvenv.cfg`), so uv
falls back to the read-only nix-store site-packages and `uv sync` fails with
`Permission denied (os error 13)` — failing the entire build.

**Why:** declared runtime deps in pyproject as "insurance" (2026-07-08); the very next
publish build failed in `uv sync` with exactly this error. Reverting to
`dependencies = []` (with a comment listing the real imports) restored publishability —
`uv sync` with an empty dep list is a no-op.

**How to apply:** never populate `[project] dependencies` (and never run bare
`uv add`/`uv sync`) while `.pythonlibs` is not a real venv. Document runtime imports in
a pyproject comment instead. Packages ship pre-installed inside `.pythonlibs` in the
deploy snapshot.

## Rule 2: production run env must set PYTHONUSERBASE

The deployment container does NOT inherit the workspace's `PYTHONUSERBASE`, so without
it `python3` cannot import anything from `.pythonlibs` (verified:
`env -i python3 -c "import flask"` fails; adding
`PYTHONUSERBASE=/home/runner/workspace/.pythonlibs` makes all runtime imports work).

**How to apply:** in the artifact's `[services.production.run.env]`, set
`PYTHONUSERBASE = "/home/runner/workspace/.pythonlibs"` alongside `PORT`.

## Rule 3: scaffolded artifact.toml has a broken production section for Python apps

The react-vite scaffold used as a registration vehicle (see python-artifact-registration.md)
leaves `[services.production]` with `serve = "proxy"` and NO run command — the published
app never starts and the URL 500s.

**How to apply:** before first publish, set via `verifyAndReplaceArtifactToml`:
`[services.production.run] args = ["python3", "artifacts/<dir>/app.py"]`, the env vars
from Rule 2, and `[services.production.health.startup] path` pointing at an
UNAUTHENTICATED route (add `/healthz`; `/` behind login 302s and fails the probe).
Production cwd is the repo root; anchor app file paths to `Path(__file__).parent`.
