# Fray Agent Notes

Distributed execution abstraction layer. Start with the shared instructions in `/AGENTS.md`; only Fray-specific conventions are below.

## Source Layout

- `src/fray/__init__.py` — docstring only. Import from the defining submodule
  (`from fray.client import Client`, `from fray.types import ResourceConfig`); a
  re-export hub here over-selects CI tests, so do not add re-exports to it.
- `src/fray/client.py` — `Client` protocol; `current_client()` returns the set client or a `LocalClient`
- `src/fray/types.py` — `JobRequest`, `ResourceConfig`, `DeviceConfig` (CPU/GPU/TPU)
- `src/fray/actor.py` — `ActorHandle`, `ActorGroup`, actor hosting
- `src/fray/local_backend.py` — Local/thread backend, the only backend until sub-project 2
- `src/fray/device_flops.py` — TPU/GPU flops calculation
- `src/fray/cluster/` — type re-exports (back-compat shim for `from fray.cluster import ResourceConfig`)

## Conventions

- Always use the `Client` protocol, not concrete backend implementations.
- Actor resources: set `num_cpus=0` on actors to avoid head-node resource contention.
- Testing: use `LocalClient`.
- Task setup: `EnvironmentConfig(setup_scripts=...)` controls how the worker prepares
  the env — `None` for the default uv-sync, `[]` to skip (bring-your-own image), or a
  verbatim list.
