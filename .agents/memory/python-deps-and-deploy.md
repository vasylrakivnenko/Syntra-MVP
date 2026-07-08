---
name: Python deps & production deploy quirks
description: How to declare Python deps and configure production run for the Flask artifact in this workspace
---

## Rule 1: `uv add` cannot install here — declare deps via pyproject edit + `uv lock`

`.pythonlibs` is a PYTHONUSERBASE-style directory (no `pyvenv.cfg`), so uv does not
treat it as a project venv. `uv add` (and the platform package installer) falls back
to the read-only nix-store system site-packages and fails with `Permission denied (os error 13)`.

**Why:** hit this twice when trying to declare runtime deps after finding
`dependencies = []` in pyproject with a fully-populated `.pythonlibs`.

**How to apply:** to declare deps, edit `pyproject.toml` `dependencies` by hand and run
`uv lock` (resolves without installing — succeeds). Never run a bare `uv sync` here — it
would try the same broken install path. Lazy/dev-only imports (e.g. vendored lib's
`anthropic`, `pandas` in build tooling) do NOT need declaring.

## Rule 2: scaffolded artifact.toml has a broken production section for Python apps

The react-vite scaffold used as a registration vehicle (see python-artifact-registration.md)
leaves `[services.production]` as `serve = "proxy"` with a no-op build and NO run command.
Result: the published deployment starts only other artifacts; the Flask app never runs and
the production URL returns "Internal Server Error".

**Why:** first publish of the Flask app 500'd for exactly this reason.

**How to apply:** before first publish of a Python artifact, set via
`verifyAndReplaceArtifactToml`:
`[services.production.run] args = ["python3", "artifacts/<dir>/app.py"]`,
`[services.production.run.env] PORT = "<port>"`, and a
`[services.production.health.startup] path` pointing at an UNAUTHENTICATED route
(add a `/healthz` returning 200 — `/` behind login 302s and can fail health checks).
Production cwd is the repo root; keep all app file paths anchored to `Path(__file__).parent`.
