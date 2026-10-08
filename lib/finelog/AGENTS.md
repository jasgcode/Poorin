# Finelog Agent Notes

Finelog's Python client. Poorin keeps it because Levanter's telemetry tracker writes training
metrics to Finelog tables. The server is the published `marin-finelog-server` wheel, pinned in
the root `pyproject.toml`; its Rust source, deployments, dashboard, and benchmarks were deleted in
P6 and remain at the `upstream-baseline` tag.

Start with the shared instructions in `/AGENTS.md`.

## Source Layout

- `src/finelog/client/` — `LogClient` (logs, stats tables, and object queries) and its helpers
- `src/finelog/embedded.py` — starts an in-process server from the installed wheel
- `src/finelog/rpc/` — generated `_pb2`/`_connect` modules for `src/finelog/proto/`
- `src/finelog/schema.py`, `table_spec.py`, `types.py`, `policy.py`, `errors.py` — table schemas,
  shared types, and errors
- `tests/` — client and embedded-server tests

## Conventions

- Finelog imports nothing from the rest of the workspace except `rigging`.
- Regenerating `rpc/` from `proto/` needs the generator at the `upstream-baseline` tag.
