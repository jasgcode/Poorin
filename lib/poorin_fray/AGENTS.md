# Fray Agent Notes

Distributed execution abstraction layer. Start with the shared instructions in `/AGENTS.md`; only Fray-specific conventions are below.

## Source Layout

- `src/poorin_fray/__init__.py` — docstring only. Import from the defining submodule
  (`from poorin_fray.client import Client`, `from poorin_fray.types import ResourceConfig`); a
  re-export hub here over-selects CI tests, so do not add re-exports to it.
- `src/poorin_fray/client.py` — `Client` protocol
- `src/poorin_fray/current_client.py` — `current_client()` returns the client set with `set_current_client()`, otherwise a `LocalClient`
- `src/poorin_fray/types.py` — `JobRequest`, `ResourceConfig`, `DeviceConfig` (CPU/GPU/TPU)
- `src/poorin_fray/actor.py` — `ActorHandle`, `ActorGroup`, actor hosting
- `src/poorin_fray/local_backend.py` — Local/thread backend, the only backend until sub-project 2
- `src/poorin_fray/device_flops.py` — TPU/GPU flops calculation
- `src/poorin_fray/cluster/` — type re-exports (back-compat shim for `from poorin_fray.cluster import ResourceConfig`)

## Conventions

- Always use the `Client` protocol, not concrete backend implementations.
- Testing: use `LocalClient`.
