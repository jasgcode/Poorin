# Fray Agent Notes

Distributed execution abstraction layer. Start with the shared instructions in `/AGENTS.md`; only Fray-specific conventions are below.

## Source Layout

- `src/fray/__init__.py` — docstring only. Import from the defining submodule
  (`from fray.client import Client`, `from fray.types import ResourceConfig`); a
  re-export hub here over-selects CI tests, so do not add re-exports to it.
- `src/fray/client.py` — `Client` protocol
- `src/fray/current_client.py` — `current_client()` returns the client set with `set_current_client()`, otherwise a `LocalClient`
- `src/fray/types.py` — `JobRequest`, `ResourceConfig`, `DeviceConfig` (CPU/GPU/TPU)
- `src/fray/actor.py` — `ActorHandle`, `ActorGroup`, actor hosting
- `src/fray/local_backend.py` — Local/thread backend, the only backend until sub-project 2
- `src/fray/device_flops.py` — TPU/GPU flops calculation

## Conventions

- Always use the `Client` protocol, not concrete backend implementations.
- Testing: use `LocalClient`.
