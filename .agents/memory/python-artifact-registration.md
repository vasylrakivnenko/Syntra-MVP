---
name: Python artifact registration
description: How to register a pure-Python Flask app as a routed artifact in the Replit pnpm monorepo
---

## Rule

You cannot write `artifact.toml` directly (blocked). To register a new Python artifact:

1. Call `createArtifact("react-vite", "<slug>", "<previewPath>", "<title>")` to scaffold a placeholder and get a real `artifact.toml` created by the platform.
2. Write `artifacts/<slug>/.replit-artifact/artifact.edit.toml` with the correct `localPort` (Flask port) and `run = "python3 artifacts/<python-dir>/app.py"` under `[services.development]`.
3. **Keep the `[[integratedSkills]]` block identical to the original** — `verifyAndReplaceArtifactToml` rejects changes to `integratedSkills`.
4. Call `verifyAndReplaceArtifactToml(edit path, real path)` — both must be absolute paths.
5. Restart the Python workflow (not the scaffolded react-vite one).

**Why:** `verifyAndReplaceArtifactToml` requires the real file to exist first; direct writes to `artifact.toml` are sandboxed out. The react-vite scaffold is just a vehicle to create the real `artifact.toml` entry in the platform registry.

**How to apply:** Any time a Python-only artifact needs proxy routing at a path (e.g. `/`).
