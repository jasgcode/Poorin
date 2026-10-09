# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The realized artifact, its on-disk record, and the drift check.

An :class:`Artifact` is the produced, persisted output of a step; an ``ArtifactStep`` is the
inert handle that builds one. This module owns the
framework — one base type and the record around it — while concrete artifact types live with
their producers (``LevanterCheckpoint`` in ``poorin.training.training``, ``TokenizedCache`` in
``poorin.processing.tokenize.tokenize``):

- :class:`Artifact` — a directory with a record (provenance + an optional JSON payload) and a
  ``raw_load`` that reads it back. The default ``raw_load`` returns a handle into the path; a subclass
  that declares value fields round-trips them through the record's ``result`` automatically.
- :class:`ArtifactRecord` — the single descriptor written next to a step's output: its config,
  fingerprint, provenance, and (for a value artifact) its ``result``.
- ``read_record``/``write_record`` (the full record) and ``read_artifact``/``write_artifact``
  (the manual typed-payload API), two entry points over one serialization scheme.
- :func:`check_drift` — the recipe-drift guard the runner applies before serving or resuming an
  output.
"""

import functools
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Self, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.cluster_config import poorin_prefix
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance, launch_provenance

from poorin.execution.fingerprint import describe_drift
from poorin.execution.step_spec import StepSpec, _is_relative_path

M = TypeVar("M", bound=BaseModel)

# JSON-shaped value, used for the human-readable config and the value payload.
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

# The record file written next to every output, and the only record name read back. Dot-prefixed so
# the datakit ``normalize._discover_files`` walk and the tokenizer file filter — both skip dotfiles —
# never mistake it for training data and parse it as JSONL.
RECORD_FILENAME = ".artifact.json"

# The old Ray executor wrote a per-step ``.executor_info`` sidecar (the ``ExecutorStepInfo``
# schema) instead of an ``.artifact.json``. Caches built that way — e.g. the pinned llama3
# Nemotron-CC caches — carry their materialized ``config`` (tokenizer, format, tags) only there,
# so we read it as a last resort to recover the record. Never written.
_LEGACY_EXECUTOR_INFO_FILENAME = ".executor_info"

# The lazy ``StepRunner`` also writes a ``.executor_info`` — but at *schedule* time, before the
# step runs — tagged with this ``executor_version``. Unlike a genuine legacy Ray sidecar, its
# ``config`` is the identity ``hash_attrs`` (deps/fingerprint/version), not the materialized
# config. ``read_record`` ignores it so an incomplete step's stub can never be served as a
# record — which would present, e.g., a tokenizer-less tokenized cache (#6836).
STEP_RUNNER_EXECUTOR_VERSION = "step_runner"

# Keys under ``StepSpec.hash_attrs`` carrying the artifact's identity, so the runner can
# apply the drift check without knowing about the lazy layer.
FINGERPRINT_KEY = "fingerprint"
VERSION_KEY = "version"
RESULT_TYPE_KEY = "result_type"
# Set only on a pinned (``override_path``) step, which points at existing data: the drift check skips it.
PINNED_KEY = "pinned"
ARTIFACT_LOAD_CONTEXT_KEY = "artifact_load"


class FingerprintMismatchError(Exception):
    """An output's recorded recipe differs from the recipe now being built, or two handles in one
    graph build the same ``name@version`` from different recipes."""


class DriftAcceptanceError(ValueError):
    """``accept_drift`` named an artifact whose recipe did not change: a typo, or a stale flag."""


class ArtifactTypeMismatchError(Exception):
    """A served record's ``result_type`` differs from the requested handle's ``result_type``."""


def result_type_name(artifact_type: "type[Artifact]") -> str:
    """The canonical ``module.Qualname`` recorded as a value artifact's ``result_type``.

    The single source of truth for both sides of the round-trip: written into the record when a
    step succeeds and compared against on :meth:`Artifact.raw_load`.
    """
    return f"{artifact_type.__module__}.{artifact_type.__qualname__}"


class Artifact(BaseModel):
    """A produced, persisted artifact: a directory with a record and a ``raw_load``.

    The default ``raw_load`` is a data ref — it returns a handle into ``path`` whose ``.record``
    carries provenance and the run's config, pulling no weights/caches into the launcher. A
    subclass that declares value fields persists and reloads them through ``record.result`` with
    no override (see :meth:`result_payload`). Not frozen: ``raw_load`` sets ``path``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    path: str = ""

    @functools.cached_property
    def record(self) -> "ArtifactRecord | None":
        """The record sidecar at ``path`` (read once), or ``None`` if absent."""
        return read_record(self.path)

    def result_payload(self) -> dict | None:
        """What the record stores as ``result``: this artifact's *declared* value fields
        (every field but ``path``), or ``None`` for a pure data ref that declares none.

        Uses ``type(self).model_fields`` rather than ``model_dump()`` so ``extra="allow"`` extras
        (e.g. the cached ``record``) never leak into the payload. Override to persist something
        else.
        """
        keys = {name for name in type(self).model_fields if name != "path"}
        if not keys:
            return None
        return self.model_dump(mode="json", include=keys) or None

    @classmethod
    def raw_load(cls, source: str) -> Self:
        """The raw, unguarded loader: a handle into ``source`` whose value fields a subclass
        repopulates from ``record.result``. Prefer :func:`run`/:func:`resolve` (driver) or
        :meth:`StepContext.resolved` (inside a step) — reach for ``raw_load`` only to read an
        artifact you already know is built.

        Tolerant of a missing record by design: an adopted or pinned artifact resolves to its
        pre-existing data location (:meth:`ArtifactStep.path`) — real data, but no marin record
        there — and its resolved value is intentionally path-only, so a missing record yields a
        path handle rather than an error. A normal computed handle is never loaded before it is
        built (``run``/``resolve`` build first; ``StepContext.resolved`` reads only
        runner-materialized deps), so a missing record cannot arise through the supported paths.

        When a record *is* present it must agree: raises :class:`ArtifactTypeMismatchError` if it
        was written by a different artifact class (a value type that changed under a reused
        version).
        """
        rec = read_record(source)
        if rec is not None and rec.result_type and rec.result_type != result_type_name(cls):
            raise ArtifactTypeMismatchError(
                f"{source}: recorded result_type is {rec.result_type}, but loading as "
                f"{result_type_name(cls)}. The value type changed under a reused version — bump the version."
            )
        return cls(path=source, **((rec.result if rec is not None else None) or {}))


class ArtifactRecord(BaseModel):
    """The single descriptor written next to a step's output.

    All fields carry a default, so a minimal manual record (:func:`write_artifact`) and a
    pre-existing legacy file both load without error; the lazy runner fills them all.
    """

    name: str = ""
    version: str = ""
    fingerprint: str = ""
    result_type: str = ""
    output_path: str = ""
    deps: list[str] = Field(default_factory=list)
    """Dependency identities as ``name@version`` strings."""
    dep_paths: list[str] = Field(default_factory=list)
    """Resolved output paths of the dependencies, aligned index-wise with ``deps``. ``deps`` is
    the portable identity; this is where each dep's record actually lives, which differs from a
    reconstruction off the identity when the dep overrode its output path."""
    config: dict[str, JSONValue] | None = None
    """The materialized config that ran (canonical-encoded), for humans and consumer metadata."""
    source: str | None = None
    """For an adopted artifact, the pre-existing data location this ``name@version`` aliases."""
    result: dict[str, JSONValue] | None = None
    """A value artifact's declared fields; ``None`` for a data artifact."""
    fingerprint_payload: str | None = None
    """The canonical config JSON the ``fingerprint`` hashes, kept for the drift diff."""
    accepted_from: list[str] = Field(default_factory=list)
    """Earlier fingerprints of this output whose changed recipe the owner accepted (``--accept-drift``)."""
    provenance: Provenance | None = None
    """Who/when/which-commit/which-argv produced this — ``None`` for a minimal manual write."""


# The one canonical CalVer form (``YYYY.MM.DD`` with an optional ``.N`` for two immutable
# revisions on the same day). Kept here, beside ``validate_version``, so callers agree on version
# identity without importing ``lazy``.
CALVER_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}(\.\d+)?$")


def is_mutable_version(version: str) -> bool:
    """A ``dev`` version is mutable: the drift check is skipped and it always rebuilds."""
    return version == "dev" or version.endswith("-dev")


def validate_path_segment(label: str, value: str) -> None:
    """A ``name``/``version`` is a single path segment: non-empty, no URL scheme, no ``..``, no
    leading/trailing slash. A malformed one is a caller bug, not a silent malformed path."""
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if "://" in value or urlparse(value).scheme:
        raise ValueError(f"{label} {value!r} must not contain a URL scheme")
    if ".." in value:
        raise ValueError(f"{label} {value!r} must not contain '..'")
    if value.startswith("/") or value.endswith("/"):
        raise ValueError(f"{label} {value!r} must not start or end with '/'")


def validate_version(version: str) -> None:
    """Validate an artifact version string, raising :class:`ValueError` if malformed.

    A version is a path segment that is either a calendar version ``YYYY.MM.DD`` (optionally
    ``YYYY.MM.DD.N``) or a mutable ``dev``/``<label>-dev``. ``v1``-style tags are rejected: an
    artifact's version is the author's explicit statement of "when this recipe was frozen", not an
    opaque label.
    """
    validate_path_segment("version", version)
    if is_mutable_version(version):
        return
    if not CALVER_RE.match(version):
        raise ValueError(
            f"version {version!r} must be a calendar version YYYY.MM.DD (optionally YYYY.MM.DD.N) "
            "or a mutable 'dev'/'<label>-dev'"
        )


def _resolved(output_path: str) -> str:
    """A relative output path is rooted at ``poorin_prefix()``; an absolute/URL path is used as-is.

    Mirrors the launcher's path resolution so a manual ``read_artifact``/``read_record`` of a
    relative step name reads the same location the runner wrote.
    """
    return prefix_join(poorin_prefix(), output_path) if _is_relative_path(output_path) else output_path


def _read_text(output_path: str, filename: str) -> str | None:
    path = prefix_join(output_path, filename)
    fs = url_to_fs(path, use_listings_cache=False)[0]
    if not fs.exists(path):
        return None
    return StoragePath(path).read_text()


def _record_from_executor_info(text: str) -> ArtifactRecord | None:
    """Map an old Ray executor ``.executor_info`` sidecar into an :class:`ArtifactRecord`.

    Only the fields a consumer reads back are carried across: ``name``, the materialized
    ``config`` (where a tokenized cache keeps its tokenizer/format/tags), ``output_path``, and
    ``dependencies`` (recorded as output paths, not ``name@version``). The legacy ``version`` is a
    per-dependency dict rather than a string, so it is dropped rather than coerced.

    Returns ``None`` for a schedule-time ``StepRunner`` stub (``executor_version`` is
    ``STEP_RUNNER_EXECUTOR_VERSION``): that file predates the step's own record and carries only
    the identity ``hash_attrs`` under ``config``, so serving it would present a config-less (e.g.
    tokenizer-less) record for a cache that may never have finished (#6836).
    """
    info = json.loads(text)
    if info.get("executor_version") == STEP_RUNNER_EXECUTOR_VERSION:
        return None
    return ArtifactRecord(
        name=info.get("name") or "",
        config=info.get("config"),
        output_path=info.get("output_path") or "",
        deps=list(info.get("dependencies") or []),
    )


# Record-native identity fields a genuine ArtifactRecord sets; a pre-#6649 bare payload has none.
_RECORD_IDENTITY_FIELDS = ("name", "fingerprint", "result_type", "result")
_RECORD_FIELDS = frozenset(ArtifactRecord.model_fields)


def _parse_record(text: str) -> ArtifactRecord:
    """Parse an ``.artifact.json``, adopting a pre-#6649 bare payload as the record's ``result``.

    Before #6649 the value model was serialized straight into ``.artifact.json`` (e.g.
    ``{"version": "v1", "output_dir": ..., "counters": ...}``), so it parses as an
    :class:`ArtifactRecord` with every native field defaulted and the real payload dropped as
    ignored extras — leaving ``read_artifact`` with no ``result``. Detect that shape (no record
    identity, but keys foreign to the record schema) and take the whole document as ``result`` so
    those caches load. A genuine record never carries foreign keys, so it is untouched.
    """
    record = ArtifactRecord.model_validate_json(text)
    if any(getattr(record, field) for field in _RECORD_IDENTITY_FIELDS):
        return record
    document = json.loads(text)
    if isinstance(document, dict) and document.keys() - _RECORD_FIELDS:
        return ArtifactRecord(result=document)
    return record


def read_record(output_path: str) -> ArtifactRecord | None:
    """The record at ``{output_path}/.artifact.json``, else ``None``.

    Falls back to a genuine legacy Ray executor ``.executor_info`` sidecar when no record file is
    present, so caches built before the record file existed still resolve their config. A
    schedule-time ``StepRunner`` stub (see :data:`STEP_RUNNER_EXECUTOR_VERSION`) is *not* honored:
    it carries no materialized config, so such an output reads back as having no record. A
    corrupt/partial file raises :class:`pydantic.ValidationError`.
    """
    output_path = _resolved(output_path)
    text = _read_text(output_path, RECORD_FILENAME)
    if text is not None:
        return _parse_record(text)
    executor_info = _read_text(output_path, _LEGACY_EXECUTOR_INFO_FILENAME)
    if executor_info is not None:
        return _record_from_executor_info(executor_info)
    return None


@dataclass(frozen=True)
class RecordedRecipe:
    """The recipe an existing output was built, or started, from."""

    fingerprint: str
    payload: str | None
    """The canonical config JSON the fingerprint hashes, for the drift diff; ``None`` when not recorded."""
    filename: str
    """The file that records it: the finished build's record or the runner's start stub."""


def recorded_recipe(output_path: str) -> RecordedRecipe | None:
    """The recipe recorded at ``output_path``: the record's when the build finished, else the stub's
    the runner writes when a build starts. ``None`` when neither records a fingerprint (nothing
    built, a manual :func:`write_artifact`, or a legacy Ray sidecar)."""
    output_path = _resolved(output_path)
    text = _read_text(output_path, RECORD_FILENAME)
    if text is not None:
        record = _parse_record(text)
        if not record.fingerprint:
            return None
        return RecordedRecipe(record.fingerprint, record.fingerprint_payload, RECORD_FILENAME)
    stub_text = _read_text(output_path, _LEGACY_EXECUTOR_INFO_FILENAME)
    if stub_text is None:
        return None
    stub = json.loads(stub_text)
    if stub.get("executor_version") != STEP_RUNNER_EXECUTOR_VERSION:
        return None
    fingerprint = (stub.get("config") or {}).get(FINGERPRINT_KEY)
    if not fingerprint:
        return None
    return RecordedRecipe(fingerprint, stub.get("fingerprint_payload"), _LEGACY_EXECUTOR_INFO_FILENAME)


def accept_recipe(output_path: str, fingerprint: str, payload: str) -> None:
    """Record ``fingerprint`` as the recipe of the output at ``output_path``: the owner accepted a
    changed recipe. A finished build's record keeps the old fingerprint in ``accepted_from``; a
    started build's stub is re-stamped, so the build resumes under the new recipe."""
    output_path = _resolved(output_path)
    recorded = recorded_recipe(output_path)
    if recorded is None:
        raise ValueError(f"no recorded recipe at {output_path} to accept a change to")
    if recorded.filename == RECORD_FILENAME:
        record = read_record(output_path)
        assert record is not None
        write_record(
            record.model_copy(
                update={
                    "fingerprint": fingerprint,
                    "fingerprint_payload": payload,
                    "accepted_from": [*record.accepted_from, record.fingerprint],
                }
            )
        )
        return
    stub_path = StoragePath(prefix_join(output_path, _LEGACY_EXECUTOR_INFO_FILENAME))
    stub = json.loads(stub_path.read_text())
    stub["config"][FINGERPRINT_KEY] = fingerprint
    stub["fingerprint_payload"] = payload
    stub_path.write_text(json.dumps(stub, indent=2))


def drift_message(label: str, recorded: str, now: str, change: str) -> str:
    """Why ``label`` cannot reuse its output, and the two ways forward."""
    name = label.split("@", 1)[0]
    return (
        f"{label}: the output records recipe {recorded}, but the recipe is now {now}.{change}\n"
        f"Change the version to build the new recipe, or pass --accept-drift {name} to keep the output."
    )


def write_record(record: ArtifactRecord) -> None:
    """Write ``record`` to ``{record.output_path}/.artifact.json``."""
    StoragePath(prefix_join(record.output_path, RECORD_FILENAME)).write_text(record.model_dump_json(indent=2))


def payload_json(value: object) -> JSONValue:
    """Canonical JSON encoding for a record payload: a ``BaseModel`` or dataclass to its dict,
    anything already JSON-shaped straight through."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value  # ty: ignore[invalid-return-type]


def read_artifact(output_path: str, schema: type[M]) -> M:
    """Load a typed payload: ``read_record(output_path).result`` validated as ``schema``.

    Raises :class:`FileNotFoundError` if no record carrying a ``result`` is present.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(f"schema must be a pydantic BaseModel subclass, got {schema!r}")
    output_path = _resolved(output_path)
    record = read_record(output_path)
    if record is not None and record.result is not None:
        return schema.model_validate(record.result, context={ARTIFACT_LOAD_CONTEXT_KEY: True})
    raise FileNotFoundError(f"no artifact payload at {output_path}")


def write_artifact(value: object, output_path: str) -> None:
    """Write a minimal record carrying ``value`` as its ``result`` — the manual save API."""
    write_record(ArtifactRecord(output_path=output_path, result=payload_json(value)))


@dataclass(frozen=True)
class StepRecordIdentity:
    """Identity + lineage of a ``StepRunner`` step, as plain data (no callable) so a remote
    write site can serialize it into a worker closure."""

    name: str
    deps: list[str]
    """Dependency identities, each a ``name_with_hash``."""
    dep_paths: list[str]
    """Resolved dependency locations, aligned index-wise with ``deps``."""
    config: dict[str, JSONValue] | None
    """The step's ``hash_attrs`` -- its materialized identity params."""
    fingerprint_payload: str | None = None


def write_step_record(identity: StepRecordIdentity, *, output_path: str, result: object) -> None:
    """Persist a ``StepRunner`` step's full record: identity + lineage + payload + provenance.

    Unlike :func:`write_artifact` (which records only ``output_path`` + ``result``), this carries
    the ``name``, the dependency identities (``deps`` -- each a ``name_with_hash``) alongside their
    resolved locations (``dep_paths`` -- where each dep's record actually lives, even when the dep
    overrode its output path), the ``config`` that determined the output, and best-effort
    provenance -- so a produced directory answers "what made me, from what" on its own, walkable
    recursively through ``dep_paths``.
    """
    write_record(
        ArtifactRecord(
            name=identity.name,
            output_path=output_path,
            deps=identity.deps,
            dep_paths=identity.dep_paths,
            config=identity.config,
            result=payload_json(result) if result is not None else None,
            fingerprint_payload=identity.fingerprint_payload,
            provenance=launch_provenance(),
        )
    )


def check_drift(step: StepSpec) -> bool:
    """The drift guard the runner applies before serving or resuming an output.

    Returns ``False`` for a non-lazy step (no fingerprint). Raises :class:`FingerprintMismatchError`
    when the output at the step's path records a different recipe, in a finished build's record or
    a started build's stub; a pinned step points at existing data and is not compared. Otherwise
    returns whether the version is mutable (``dev``), which always rebuilds.
    :func:`poorin.execution.lazy.run` checks the whole graph before the runner starts; this covers
    code that calls the runner directly.
    """
    fingerprint = step.hash_attrs.get(FINGERPRINT_KEY)
    if fingerprint is None:
        return False
    version = step.hash_attrs.get(VERSION_KEY, "")
    if step.hash_attrs.get(PINNED_KEY):
        return is_mutable_version(version)
    recorded = recorded_recipe(step.output_path)
    if recorded is not None and recorded.fingerprint != fingerprint:
        change = describe_drift(recorded.payload, step.fingerprint_payload)
        raise FingerprintMismatchError(
            drift_message(f"{step.name}@{version}", recorded.fingerprint, fingerprint, change)
        )
    return is_mutable_version(version)
