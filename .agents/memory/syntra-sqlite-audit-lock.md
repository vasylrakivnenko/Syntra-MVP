---
name: SQLite audit writes must happen outside get_db transactions
description: Syntra's AuditLog opens its own connection; calling it inside a `with get_db()` block that has pending writes deadlocks ("database is locked").
---

**Rule:** In the Syntra Flask app, never call `AuditLog().append_simple(...)` (or any second-connection write) inside a `with get_db()` block that has performed an INSERT/UPDATE. Commit first (exit the block), then audit.

**Why:** SQLite allows one writer at a time even in WAL mode. `get_db()` commits only on block exit, so the route's connection holds the write lock while `AuditLog` opens a fresh connection and times out with `sqlite3.OperationalError: database is locked`. This bit the attorney review route in production-like testing.

**How to apply:** Structure routes as: (1) `with get_db():` do all row writes; (2) after the block, call AuditLog; (3) then flash/redirect. A quick indentation-scan for `append_simple` inside `with get_db()` blocks catches regressions.

Related Jinja gotcha hit the same day: a context-processor dict exposed under key `items` — `inbox.items` in Jinja resolves to `dict.items` (the method), not the key. Use a key name like `entries`.
