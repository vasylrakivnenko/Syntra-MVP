---
name: Python package installs in this workspace
description: How to install Python packages here — uv/installLanguagePackages fail; use pip --target .pythonlibs
---

**Rule:** Install Python packages with
`python3 -m pip install --target /home/runner/workspace/.pythonlibs/lib/python3.11/site-packages <pkg>`.

**Why:** `.pythonlibs` is NOT a venv (no pyvenv.cfg). `installLanguagePackages` and any
`uv add`/`uv pip install` resolve to the read-only nix-store interpreter and die with
"Permission denied … /nix/store/...". The root `pyproject.toml` deliberately declares
`dependencies = []` because the publish pipeline runs `uv sync`, which cannot install
into `.pythonlibs` — packages must live pre-installed there.

**How to apply:** Any time a new Python dependency is needed for the Flask app, use the
pip --target command above and, if it's a runtime import, keep the pyproject NOTE
comment's import list current. Never "fix" pyproject by declaring the deps there.
