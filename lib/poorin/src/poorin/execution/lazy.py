# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact steps addressed by an explicit ``name@version``.

The model:

- An :class:`ArtifactStep` is a content-free handle to a versioned artifact:
  ``(name, version, artifact_type)`` plus how to build it — ``run``, ``build_config``, ``deps``.
  Constructing one runs nothing; :func:`run` / :func:`resolve` execute it.
- ``build_config(ctx)`` is a pure function of a :class:`StepContext`, which draws the line between
  identity and execution: values written as literals (model, hyperparameters, dep *versions*)
  bear identity, while values *pulled* from the context — ``ctx.output_path``,
  ``ctx.artifact_path(dep)``, ``ctx.prefix``, ``ctx.region``, ``ctx.runtime_arg(key)`` — are
  where/how the step runs and never do.
- ``run(config)`` is the step function: a plain callable. It returns the produced
  :class:`~poorin.execution.artifact.Artifact` (a data ref or a value), or ``None`` for a data
  step that writes its bytes to ``ctx.output_path``. To run a step on Fray, wrap your own function
  with :func:`~poorin.execution.remote.remote` and pass that as ``run`` — the step layer does no
  remote dispatch and knows nothing about resources.
- Identity is the explicit ``{prefix}/{name}/{version}`` path — no content hash. The recipe
  *fingerprint* (the config built with dep versions in place of paths and placeholders for
  everything pulled from ``ctx``) is recorded beside the output, never in the path, and a changed
  recipe under an existing output is refused.

Pre-existing data is brought in with :meth:`ArtifactStep.adopt`. :func:`run` runs handles for their side
effects; :func:`resolve` runs one and loads its typed artifact. A custom step is
just a function that constructs and returns an ``ArtifactStep`` (see ``poorin.experiment.data`` /
``poorin.experiment.train`` for the dataset and training builders) — there is no generic wrapper to
learn.
"""

import inspect
import json
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Generic, TypeVar, cast

from rigging.filesystem.cluster_config import poorin_prefix, poorin_region
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import Provenance

from poorin.execution.artifact import (
    FINGERPRINT_KEY,
    PINNED_KEY,
    RESULT_TYPE_KEY,
    VERSION_KEY,
    Artifact,
    ArtifactRecord,
    FingerprintMismatchError,
    accept_recipe,
    drift_message,
    is_mutable_version,
    result_type_name,
    validate_path_segment,
    validate_version,
    write_record,
)
from poorin.execution.build_context import resolve_version
from poorin.execution.cache_state import CacheState, HandleState, check_accept_drift, handle_state, record_path
from poorin.execution.fingerprint import canonical_json, fingerprint_hash
from poorin.execution.step_runner import StepRunner
from poorin.execution.step_spec import StepSpec, _is_relative_path

T = TypeVar("T", bound=Artifact)


@dataclass(frozen=True)
class StepContext:
    """What a step resolves its config against — and the dividing line for identity.

    Values written as literals in ``build_config`` (the model, hyperparameters, dep *versions*)
    bear identity and enter the fingerprint; values pulled from the context never do.
    :meth:`for_run` binds the live environment; :meth:`for_fingerprint` substitutes placeholders,
    which is what realizes the exclusion.

    Pull points: ``ctx.output_path`` (this step's output dir), ``ctx.artifact_path(dep)`` (a dep's
    resolved path), ``ctx.resolved(dep)`` (a dep's loaded, typed artifact — its value/metrics),
    ``ctx.prefix`` (the live storage prefix), ``ctx.region`` (the live region), and
    ``ctx.runtime_arg(key)`` (a step-declared runtime arg, e.g. the TPU a dispatched job runs on).
    ``artifact_path`` and ``resolved`` accept only declared ``deps``: a handle the step did not
    list is a bug (the runner never materialized it), so both raise on one. ``is_fingerprint`` is
    ``True`` only under :meth:`for_fingerprint`, for a ``build_config`` that must emit an identity
    summary (rather than read records that do not exist yet).
    """

    output_path: str
    prefix: str
    region: str | None
    is_fingerprint: bool
    _dep_ref: Callable[["ArtifactStep"], str]
    _runtime_args: Mapping[str, Any]
    _deps: tuple["ArtifactStep", ...] = ()
    _loaded: dict[int, Artifact] = field(default_factory=dict)

    def artifact_path(self, dep: "ArtifactStep") -> str:
        """The resolved output path of a dependency this step builds on."""
        self._require_declared(dep)
        return self._dep_ref(dep)

    def resolved(self, dep: "ArtifactStep[T]") -> T:
        """The loaded, typed artifact of a declared dependency — its value/metrics, not just a path.

        Reads the dep's record once and caches it, so resolving the same dep N times costs one
        store read. Loads the record sidecar (a data ref into the path), not the heavy payload, and
        never runs anything: the runner has materialized every declared dep before this step's
        function executes. Use ``.path`` on the result for the output dir.

        Raises :class:`ValueError` for an undeclared dep, and at fingerprint time (no records exist
        yet — branch on ``is_fingerprint`` and emit an identity summary instead).
        """
        if self.is_fingerprint:
            raise ValueError("ctx.resolved is unavailable at fingerprint time; branch on ctx.is_fingerprint")
        self._require_declared(dep)
        cached = self._loaded.get(id(dep))
        if cached is None:
            cached = dep.artifact_type.raw_load(self._dep_ref(dep))
            self._loaded[id(dep)] = cached
        return cast(T, cached)

    def _require_declared(self, dep: "ArtifactStep") -> None:
        if dep not in self._deps:
            raise ValueError(
                f"{dep.name}@{dep.version} is not a declared dependency of this step; "
                "add it to deps=(...) so the runner materializes it before this step runs"
            )

    def runtime_arg(self, key: str) -> Any:
        """A step-declared runtime arg: its real value at run time, a ``<key>`` placeholder at
        fingerprint time. Pull execution choices through here so they reach the step but never
        bear on its identity."""
        try:
            return self._runtime_args[key]
        except KeyError:
            raise KeyError(
                f"runtime arg {key!r} is not declared in the step's runtime_args {sorted(self._runtime_args)}"
            ) from None

    def is_context_value(self, value: object) -> bool:
        """Whether ``value`` is what this fingerprint context put in place of a pulled value or a
        dependency reference, or a string built from a pulled value (``<output_path>/hf``): the
        parts of a config an override must not replace. Always ``False`` at run time, where the
        fingerprint pass has already refused them."""
        if not self.is_fingerprint or not isinstance(value, str):
            return False
        placeholders = (self.output_path, self.prefix, self.region, *self._runtime_args.values())
        if any(placeholder in value for placeholder in placeholders if placeholder):
            return True
        return value in {self._dep_ref(dep) for dep in self._deps}

    @staticmethod
    def for_run(
        output_path: str,
        prefix: str,
        *,
        region: str | None = None,
        runtime_args: Mapping[str, Any] | None = None,
        deps: Iterable["ArtifactStep"] = (),
    ) -> "StepContext":
        return StepContext(
            output_path=output_path,
            prefix=prefix,
            region=region,
            is_fingerprint=False,
            _dep_ref=lambda d: d.path(prefix),
            _runtime_args=runtime_args or {},
            _deps=tuple(deps),
        )

    @staticmethod
    def for_fingerprint(runtime_arg_keys: Iterable[str] = (), deps: Iterable["ArtifactStep"] = ()) -> "StepContext":
        return StepContext(
            output_path="<output_path>",
            prefix="<prefix>",
            region="<region>",
            is_fingerprint=True,
            _dep_ref=lambda d: f"{d.name}@{d.version}",
            _runtime_args={key: f"<{key}>" for key in runtime_arg_keys},
            _deps=tuple(deps),
        )


# eq=False gives handles object identity (hashable despite unhashable values in their config),
# so a handle can key a mixture dict. Value-equality lives in name@version and the fingerprint.
@dataclass(frozen=True, eq=False)
class ArtifactStep(Generic[T]):
    """A lazy, content-free handle to a versioned artifact and how to build it.

    ``run(build_config(ctx))`` produces the artifact: ``build_config`` assembles the config from
    the :class:`StepContext` (literals bear identity; pulled values do not), and ``run`` is the
    plain callable that does the work, returning the produced ``artifact_type`` (or ``None`` for a
    data step that writes to ``ctx.output_path``). ``deps`` are the handles this step builds on;
    pass the same handles ``build_config`` reads via ``ctx.artifact_path`` so they materialize
    first.
    """

    name: str
    version: str
    artifact_type: type[T]
    run: Callable[[Any], "Artifact | None"]
    build_config: Callable[[StepContext], Any]
    deps: tuple["ArtifactStep", ...] = ()
    runtime_args: Mapping[str, Any] = field(default_factory=dict)
    """Execution choices the config pulls via ``ctx.runtime_arg(key)``. Excluded from the
    fingerprint, so changing one never forks identity."""
    override_path: str | None = None
    """Pin the artifact to an existing location instead of ``{name}/{version}``. A relative
    pin is resolved against the prefix; an absolute one is used as-is. Writes no record.
    Mutually exclusive with ``adopt_source``."""
    adopt_source: str | None = None
    """Adopt pre-existing data at this location as a managed ``name@version`` (set via
    :meth:`ArtifactStep.adopt`). Consumers resolve to ``adopt_source``; running it writes a provenance
    record at the canonical address. Mutually exclusive with ``override_path``."""
    adopt_config: dict | None = None
    """For an adopted artifact, the synthetic ``config`` to record (e.g. a tokenized cache's
    tokenizer/format), so consumers read metadata the same way as for a produced artifact."""

    def __post_init__(self) -> None:
        validate_path_segment("name", self.name)
        validate_version(self.version)
        if self.adopt_source is not None and self.override_path is not None:
            raise ValueError(f"{self.name}@{self.version}: an artifact cannot be both adopted and pinned")

    def fingerprint_payload(self) -> str:
        """The canonical config bytes this handle's fingerprint hashes. For a computed
        artifact, the config built with dep versions for paths and placeholders for pulled
        values; for an adopted one, its source location and recorded config."""
        if self.adopt_source is not None:
            return json.dumps({"adopt_source": self.adopt_source, "config": self.adopt_config}, sort_keys=True)
        config = self.build_config(StepContext.for_fingerprint(self.runtime_args.keys(), self.deps))
        return canonical_json(config)

    def fingerprint(self) -> str:
        """Stable id of *how* this artifact is built. Changes iff a config value or a dep
        version changes; independent of the storage prefix/region."""
        return fingerprint_hash(self.fingerprint_payload())

    def path(self, prefix: str | None = None) -> str:
        resolved_prefix = prefix or poorin_prefix()
        # An adopted artifact resolves to its source; a pin to its location; else the
        # canonical name@version address.
        location = self.adopt_source if self.adopt_source is not None else self.override_path
        if location is not None:
            if _is_relative_path(location):
                return prefix_join(resolved_prefix, location)
            return location
        # The explicit, hash-free ``{prefix}/{name}/{version}`` address.
        return prefix_join(resolved_prefix, f"{self.name}/{self.version}")

    def lower(self) -> StepSpec:
        """Lower this handle graph into a ``StepSpec`` graph the ``StepRunner`` can run.

        A pure structural transform (no execution): captures provenance once and threads it
        through the graph. See :func:`_lower`.
        """
        return _lower(self, Provenance.capture())

    @classmethod
    def adopt(
        cls,
        name: str,
        version: str,
        source: str,
        *,
        kind: type[T] = Artifact,  # ty: ignore[invalid-parameter-default]
        config: dict | None = None,
    ) -> "ArtifactStep[T]":
        """Register pre-existing data at ``source`` as a managed ``name@version`` handle.

        The data is neither moved nor recomputed: a consumer that depends on the returned handle
        resolves to ``source``. Running it writes a provenance record at the canonical
        ``{prefix}/{name}/{version}`` (with ``source`` recorded). ``kind`` selects the handle's
        ``artifact_type`` for consumer routing (:class:`Artifact` by default), so a raw checkpoint
        path becomes a typed handle: ``ArtifactStep.adopt(name, version, "gs://…", kind=LevanterCheckpoint)``.
        ``config`` records synthetic metadata (e.g. a tokenized cache's tokenizer/format) so
        consumers read it the same way as for a produced artifact.
        """
        return cls(
            name=name,
            version=version,
            artifact_type=kind,
            run=_adopt_noop,
            build_config=lambda _ctx: None,
            adopt_source=source,
            adopt_config=config,
        )


def artifact_identity(step: ArtifactStep) -> str:
    """Return the immutable recipe identity carried across artifact boundaries."""
    return f"{step.name}@{step.version}:{step.fingerprint()}"


def _adopt_noop(_config: Any) -> None:
    raise AssertionError("adopted artifacts are registered, not computed")


def materialized_config(handle: "ArtifactStep", prefix: str) -> Any:
    """The concrete config a run would receive, for inspection/golden tests.

    Resolves ``ctx.output_path`` / ``ctx.artifact_path(dep)`` against ``prefix`` and passes the
    handle's real ``runtime_args`` (so a ``build_config`` that reads ``ctx.runtime_arg(...)`` does
    not ``KeyError``) without running anything. Region is ``None``, so this is not identical to the
    run-time config, which also binds a real region.
    """
    return handle.build_config(
        StepContext.for_run(
            output_path=handle.path(prefix), prefix=prefix, runtime_args=handle.runtime_args, deps=handle.deps
        )
    )


def _output_path_spec(handle: "ArtifactStep") -> str:
    """The relative output path StepSpec should use: the pin if set, else ``{name}/{version}``."""
    return handle.override_path or f"{handle.name}/{handle.version}"


def _lower(handle: "ArtifactStep", provenance: Provenance, memo: dict[int, StepSpec] | None = None) -> StepSpec:
    """Lower a handle graph into a ``StepSpec`` graph, recording ``provenance`` on every step.

    Each handle becomes a step addressed by its explicit ``{name}/{version}`` (or its pin); the
    fingerprint, version, artifact type, dep identities, and a pin's mark travel in ``hash_attrs``
    so the runner applies the drift check before serving a cached output (skipping a pin), and the
    step fn writes a full :class:`ArtifactRecord` on success (unless pinned). Pure transform of
    *structure* — it never inspects ``handle.run``.

    ``memo`` caches the spec of each handle by identity, so a handle several consumers share is
    lowered once. Without it a diamond re-walks the shared subgraph — and re-runs its
    ``build_config`` — once per path through it. Pass one ``memo`` across a whole ``run`` to share
    it between terminals.

    Raises :class:`ValueError` if a fixed (non-``dev``) handle has a ``dev`` dep.
    """
    memo = {} if memo is None else memo
    cached = memo.get(id(handle))
    if cached is not None:
        return cached
    dep_specs = [_lower(dep, provenance, memo) for dep in handle.deps]
    payload = handle.fingerprint_payload()
    fingerprint = fingerprint_hash(payload)

    if not is_mutable_version(handle.version):
        for dep in handle.deps:
            if is_mutable_version(dep.version):
                raise ValueError(
                    f"{handle.name}@{handle.version} is a fixed version but depends on mutable "
                    f"{dep.name}@{dep.version}; a mutable dep would rebuild while the fixed parent stays "
                    "cached. Make the parent dev, or pin the dep."
                )

    dep_refs = [f"{d.name}@{d.version}" for d in handle.deps]
    rt_name = result_type_name(handle.artifact_type)
    # A pin references existing data; it writes no record. Everything else records provenance.
    record_provenance = handle.override_path is None

    def fn(output_path: str, _handle: "ArtifactStep" = handle) -> Any:
        if _handle.adopt_source is not None:
            # Adoption registers pre-existing data: no compute, the record points at it.
            source = _handle.path(poorin_prefix())
            fs = url_to_fs(source, use_listings_cache=False)[0]
            if not fs.exists(source):
                raise FileNotFoundError(f"cannot adopt {_handle.name}@{_handle.version}: no data at {source}")
            result = None
            config_json: dict[str, Any] | None = _handle.adopt_config
            result_json: dict[str, Any] | None = None
        else:
            ctx = StepContext.for_run(
                output_path=output_path,
                prefix=poorin_prefix(),
                region=poorin_region(),
                runtime_args=_handle.runtime_args,
                deps=_handle.deps,
            )
            config = _handle.build_config(ctx)
            result = _handle.run(config)
            source = None
            config_json = json.loads(canonical_json(config))
            result_json = result.result_payload() if isinstance(result, Artifact) else None
        if record_provenance:
            write_record(
                ArtifactRecord(
                    name=_handle.name,
                    version=_handle.version,
                    fingerprint=fingerprint,
                    result_type=rt_name,
                    output_path=output_path,
                    deps=dep_refs,
                    config=config_json,
                    source=source,
                    result=result_json,
                    fingerprint_payload=payload,
                    provenance=provenance,
                )
            )
        return result

    hash_attrs: dict[str, Any] = {
        FINGERPRINT_KEY: fingerprint,
        VERSION_KEY: handle.version,
        RESULT_TYPE_KEY: rt_name,
        "deps": dep_refs,
    }
    if handle.override_path is not None:
        hash_attrs[PINNED_KEY] = True

    spec = StepSpec(
        name=handle.name,
        override_output_path=_output_path_spec(handle),
        deps=dep_specs,
        hash_attrs=hash_attrs,
        fingerprint_payload=payload,
        fn=fn,
        writes_record=True,
    )
    memo[id(handle)] = spec
    return spec


def _refuse_drift(handles: list["ArtifactStep"], accept_drift: Collection[str]) -> None:
    """Refuse a run if any artifact in the graph records a different recipe, unless accepted.

    Checks every handle, dependencies the runner would prune behind a built consumer included. An
    accepted artifact's recorded recipe is re-stamped, so the runner serves or resumes it. Several
    handles can build one output (a sweep's trials each construct the shared dataset): handles that
    share a ``name@version`` and a fingerprint are one artifact, reported and re-stamped once, and
    handles that share a ``name@version`` but not a fingerprint are two recipes claiming one output,
    which is refused.
    """
    states: dict[str, HandleState] = {}
    for state in map(handle_state, handles):
        first = states.setdefault(state.label, state)
        if first.fingerprint != state.fingerprint:
            raise FingerprintMismatchError(
                f"{state.label} is built from two recipes in this graph, {first.fingerprint} and "
                f"{state.fingerprint}. Give each recipe its own name or version."
            )
    check_accept_drift(states.values(), accept_drift)
    drifted = [s for s in states.values() if s.state is CacheState.DRIFTED]
    refused = [s for s in drifted if s.handle.name not in accept_drift]
    if refused:
        raise FingerprintMismatchError(
            "\n\n".join(drift_message(s.label, s.recorded_fingerprint or "", s.fingerprint, s.change) for s in refused)
        )
    for state in drifted:
        accept_recipe(record_path(state.handle), state.fingerprint, state.handle.fingerprint_payload())


def run(
    *handles: "ArtifactStep[T]",
    max_concurrent: int = 8,
    force_run_failed: bool = True,
    accept_drift: Collection[str] = (),
) -> list[T]:
    """Build ``handles`` and return their loaded, typed artifacts, in argument order.

    The entry point for one or several handles: lowers the graph, builds every step (independent
    steps in parallel up to ``max_concurrent``), then loads each top-level handle's typed artifact
    from the record the runner just wrote — so the caller gets the resolved values directly, no
    separate read. Provenance is captured once, so every artifact built records the same launch.
    ``force_run_failed`` reruns a previously-FAILED step instead of raising.

    Before anything runs, every artifact in the graph is checked against the recipe its output
    records; ``accept_drift`` names artifacts whose changed recipe is accepted. Lowering validates
    the graph first, so a graph it rejects re-stamps no record.
    """
    provenance = Provenance.capture()
    memo: dict[int, StepSpec] = {}
    specs = [_lower(h, provenance, memo) for h in handles]
    _refuse_drift(graph_handles(list(handles)), accept_drift)
    StepRunner().run(specs, force_run_failed=force_run_failed, max_concurrent=max_concurrent)
    return [h.artifact_type.raw_load(h.path()) for h in handles]


def lower(handle: "ArtifactStep") -> StepSpec:
    """Lower ``handle``'s graph into a ``StepSpec`` the ``StepRunner`` can run.

    Captures provenance once and threads it through the graph. Lowering validates the graph:
    :mod:`poorin.experiment.cli` lowers the plan it prints, to reject a graph it could not run. To
    build handles, call :func:`run`. ``StepRunner().run`` on lowered specs skips ``run``'s
    whole-graph drift check, which covers dependencies the runner prunes behind a built consumer
    and two recipes for one ``name@version``.
    """
    return _lower(handle, Provenance.capture())


def resolve(handle: "ArtifactStep[T]", *, max_concurrent: int = 8) -> T:
    """Build one ``handle`` and return its loaded, typed artifact — :func:`run` for a single handle.

    The driver-side "build this and give me the value" verb. To resolve several handles, prefer
    one :func:`run` call (which builds them in parallel and returns all their artifacts) over
    ``resolve``-ing in a loop.
    """
    return run(handle, max_concurrent=max_concurrent)[0]


def graph_handles(handles: list[ArtifactStep]) -> list[ArtifactStep]:
    """Every handle reachable from ``handles`` (deps included), deduped by identity, deps first."""
    seen: set[int] = set()
    order: list[ArtifactStep] = []

    def walk(handle: ArtifactStep) -> None:
        if id(handle) in seen:
            return
        seen.add(id(handle))
        for dep in handle.deps:
            walk(dep)
        order.append(handle)

    for handle in handles:
        walk(handle)
    return order


class _OutSentinel:
    """The :data:`OUT` sentinel for :func:`apply`: resolves to ``ctx.output_path`` at run time."""

    def __repr__(self) -> str:
        return "OUT"


OUT: Final = _OutSentinel()


def _collect_deps(inputs: Mapping[str, Any]) -> list["ArtifactStep"]:
    """The ``ArtifactStep`` handles in ``inputs`` (recursing list/tuple/dict), deduped by
    identity, in first-seen order."""
    deps: list[ArtifactStep] = []
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, ArtifactStep):
            if id(value) not in seen:
                seen.add(id(value))
                deps.append(value)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for value in inputs.values():
        walk(value)
    return deps


def _resolve_input(value: Any, ctx: StepContext) -> Any:
    """Resolve one ``apply`` input against ``ctx``: an ``ArtifactStep`` to its path, ``OUT`` to
    ``ctx.output_path``, recursing builtin containers; anything else is a literal."""
    if isinstance(value, ArtifactStep):
        return ctx.artifact_path(value)
    if value is OUT:
        return ctx.output_path
    if isinstance(value, list):
        return [_resolve_input(item, ctx) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_input(item, ctx) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_input(item, ctx) for key, item in value.items()}
    return value


def apply(
    name: str,
    fn: Callable[..., Any],
    *,
    version: str | None = None,
    artifact_type: type[T] = Artifact,  # ty: ignore[invalid-parameter-default]
    pin: str | None = None,
    **inputs: Any,
) -> "ArtifactStep[T]":
    """A single-step builder that calls ``fn(**inputs)`` directly — for a step whose function
    takes keyword arguments rather than a config object.

    Each value in ``inputs`` is classified, recursing into ``list``/``tuple``/``dict``: an
    :class:`ArtifactStep` handle becomes a dep and resolves to ``ctx.artifact_path(handle)`` at run
    time (its ``name@version`` enters identity); the :data:`OUT` sentinel resolves to
    ``ctx.output_path``; anything else is a literal that bears identity. To dispatch on Fray, pass
    an already-wrapped ``remote(fn, resources=…)`` as ``fn``. ``artifact_type`` selects the
    produced :class:`Artifact` type; ``pin`` references existing data.

    ``version`` defaults to the ambient :class:`~poorin.execution.build_context.BuildContext`'s
    version for ``name`` when omitted (build inside :func:`~poorin.execution.build_context.build_context`);
    an explicit value always wins.

    Raises :class:`TypeError` if ``fn``'s signature cannot bind the inputs.
    """
    version = resolve_version(name, version)
    try:
        inspect.signature(fn).bind(**inputs)
    except TypeError as e:
        fn_name = getattr(fn, "__name__", repr(fn))
        raise TypeError(f"apply({name!r}): {fn_name} cannot bind the given inputs: {e}") from e

    deps = _collect_deps(inputs)

    def build_config(ctx: StepContext) -> dict[str, Any]:
        return {key: _resolve_input(value, ctx) for key, value in inputs.items()}

    def step_fn(config: dict[str, Any], _fn: Callable[..., Any] = fn) -> Any:
        return _fn(**config)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=artifact_type,
        run=step_fn,
        build_config=build_config,
        deps=tuple(deps),
        override_path=pin,
    )
