# poorin_fray

Job-dispatch interfaces for Poorin: resource and job types (`poorin_fray.types`), the client and
actor protocols, the per-device FLOPs table, and `LocalClient`, which runs jobs in the current
process. `poorin_fray.current_client()` returns the client set for the current context, or a new
`LocalClient`.

## Poorin's lighter fork of Fray

This library is Marin's Fray, renamed (D85) because Poorin cut it down to its local backend.
Against the `upstream-baseline` tag it has 1,925 lines of source instead of 2,768.

Removed:

- The Iris backend (`iris_backend.py`, 820 lines), with Iris itself (D5).
- `cloudpickle`, which serialized callables for remote backends, and the TPU test dependency
  group.

Kept but unused locally: `ResourceConfig` fields that only a remote backend reads, such as
`target_cluster` and `processes_per_task`; `LocalClient` ignores them. Sub-project 2 (compute
dispatch) decides their future.
