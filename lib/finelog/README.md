# finelog

Python client for Finelog, the log and metrics store that Levanter's telemetry tracker writes to:
`LogClient` (typed stats tables, SQL queries, process logs), the generated RPC stubs, table
schemas, and `finelog.embedded`, which starts an in-process server for tests and for
`tools.prune_check`. The server itself is upstream's published `marin-finelog-server` wheel, pinned
in the root `pyproject.toml`; its module is `finelog_server`, and its protobuf packages keep the
names `finelog.stats` and `finelog.logging`, which the wire protocol carries.

## Poorin's lighter fork of Finelog

This library is Marin's Finelog, distributed as `poorin-finelog` (D85, D86) because Poorin kept only its
Python client; the import name stays `finelog`. Against
the `upstream-baseline` tag it has 2,938 lines of Python source instead of 10,023, and none of the
other code.

Removed (D14, P6):

- The Rust server: 74,921 lines.
- The web dashboard: 6,672 lines.
- The deploy CLI, its cluster configs, and the benchmarks.
- The `click`, `fsspec`, `gcsfs`, and `pyyaml` dependencies the deploy CLI used.

Removed later (D89): `ObjectQueryClient`, which ran SQL over catalog snapshots in object storage,
and `RemoteLogHandler`, which shipped Python log records through a `LogClient`. Nothing outside
Finelog used them.

Regenerating the RPC stubs from `proto/`, or patching the server, needs the generator and the Rust
source at `upstream-baseline`.
