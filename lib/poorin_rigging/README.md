# poorin_rigging

The base layer every other Poorin library builds on:

- `poorin_rigging.filesystem`: fsspec access to GCS, S3-compatible stores, and local paths, with
  cross-region transfer guards and the `mirror://` scheme.
- `poorin_rigging.fsutil`: the `fsutil` command-line tool for listing, copying, inspecting, and
  deleting data in buckets.
- `poorin_rigging.cache`: content hashes that key compiled-kernel and autotune caches.
- Logging setup, config discovery, provenance, timing, and the telemetry probes Levanter's
  telemetry tracker starts.

## Poorin's lighter fork of Rigging

This library is Marin's Rigging, renamed (D85) because Poorin removed the parts that served a
shared fleet. Against the `upstream-baseline` tag it has 10,352 lines of source instead of 14,465.

Removed:

- Authentication and secrets: the token authority, server auth, JWT and IAP login, credential
  stores, GCP Secret Manager references, and log redaction (`auth`, `auth_config`,
  `server_auth`, `token_authority`, `credentials`, `credential_store`, `secrets`, `redaction`).
- Cluster connectivity: `connect`, `tunnel`, `rpc`, `network`, and the cluster manifest.
- The Prometheus bridge and the NVIDIA telemetry probe.
- The `pyjwt` dependency and the `iap`, `prometheus`, and `secrets` extras.

Still here but likely vestigial on one GPU (see `docs/poorin/architecture.md`): the NCCL stall
probes under `telemetry/` and the `mirror://` filesystem.
