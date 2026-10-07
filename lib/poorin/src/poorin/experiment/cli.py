# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click bootstrap for experiment ``main``s that defer their artifact versions.

An experiment's driver constructs a handle graph and then either prints its plan or builds it.
This module makes that a one-liner while wiring up the deferred-version machinery
(:mod:`poorin.execution.build_context`):

- :func:`build_options` is a Click decorator that adds ``--version`` (the run-wide default),
  ``--override NAME=VERSION`` (per-artifact, repeatable), ``--run`` (build vs. print the plan), and
  ``--max-concurrent``. It builds the wrapped function's handle graph **inside** a
  :class:`~poorin.execution.build_context.BuildContext` assembled from those options, so any builder
  that deferred its version (omitted ``version=``) resolves against them.
- :func:`experiment_main` wraps a nullary ``build()`` into a ready-to-run command.

``--version`` is required — there is no silent ``dev`` default, because a deferred *dataset* at a
mutable version rebuilds its (often multi-TB) cache on every run and, unlike a training checkpoint,
is not namespaced per user. To iterate, pass ``--version dev``: each invocation gets a fresh,
suffixed version, and ``--resume <version>`` continues an earlier one. The printed plan flags
every artifact that resolved to a mutable version and will rebuild.

An experiment with its own options composes with :func:`build_options` by stacking it *below* the
experiment's own ``@click.option``s and ``@click.command()`` on top::

    @click.command()
    @click.option("--device", ...)
    @build_options
    def main(device):
        return build(device=device)   # version now deferred to --version / --override

    if __name__ == "__main__":
        main()

Construct the whole handle graph inside ``build()``; a builder invoked lazily after it returns has
no active context and raises. Handles are frozen at construction, so reusing one across two
invocations keeps its first resolved version.
"""

import functools
import secrets
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import click

from poorin.execution.artifact import is_mutable_version, validate_version
from poorin.execution.build_context import BuildContext, VersionCodex, build_context
from poorin.execution.lazy import ArtifactStep, lower, run

BuildResult = ArtifactStep | Mapping[str, ArtifactStep] | list[ArtifactStep] | tuple[ArtifactStep, ...]
BuildRunner = Callable[[list[ArtifactStep], int], None]


def _version_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Reject a malformed ``--version`` at parse time, as a Click error rather than a traceback."""
    if value is None:
        return None
    try:
        validate_version(value)
    except ValueError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e
    return value


def _resume_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Accept only a mutable version: calendar versions are stable and need no resume."""
    if value is None:
        return None
    try:
        validate_version(value)
    except ValueError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e
    if not is_mutable_version(value):
        raise click.BadParameter(
            f"{value!r} is a calendar version; --resume continues a '-dev' run", ctx=ctx, param=param
        )
    return value


def _override_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``NAME=VERSION`` into a validated ``{name: version}`` map, as Click errors."""
    overrides: dict[str, str] = {}
    for item in value:
        name, sep, version = item.partition("=")
        if not sep or not name:
            raise click.BadParameter(f"{item!r} is not NAME=VERSION", ctx=ctx, param=param)
        if name in overrides:
            raise click.BadParameter(f"duplicate override for {name!r}", ctx=ctx, param=param)
        try:
            validate_version(version)
        except ValueError as e:
            raise click.BadParameter(f"override {name!r}: {e}", ctx=ctx, param=param) from e
        overrides[name] = version
    return overrides


def run_version(version: str, *, now: datetime, token: str) -> str:
    """The version a run uses for ``--version``.

    A mutable version (``dev`` or ``<label>-dev``) gets a run suffix so each invocation writes to
    a fresh output path: ``dev`` becomes ``20261005-091500-a1b2-dev`` and ``probe-dev`` becomes
    ``probe-20261005-091500-a1b2-dev``. Without the suffix, a rerun lands on the previous run's
    ``{name}/dev`` path and Levanter's checkpointer resumes from its last checkpoint instead of
    training from scratch. A calendar version is returned unchanged. ``--resume`` reuses a
    suffixed version to continue that run.
    """
    if not is_mutable_version(version):
        return version
    label = version.removesuffix("dev").rstrip("-")
    stamped = f"{now:%Y%m%d-%H%M%S}-{token}"
    return f"{label}-{stamped}-dev" if label else f"{stamped}-dev"


def _as_handles(result: BuildResult) -> list[ArtifactStep]:
    """Normalize a ``build()`` return — one handle, a list/tuple, or a name→handle mapping."""
    if isinstance(result, ArtifactStep):
        return [result]
    if isinstance(result, Mapping):
        return list(result.values())
    if isinstance(result, list | tuple):
        return list(result)
    raise TypeError(
        f"build() must return an ArtifactStep, a list/tuple, or a mapping of them, not {type(result).__name__}"
    )


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


def _print_plan(handles: list[ArtifactStep]) -> None:
    """Print the lowered plan, then a summary of every artifact's resolved ``name@version``.

    The summary flags each artifact that resolved to a mutable version inline, since a mutable
    version rebuilds on every run — the thing a driver most wants to catch before ``--run``.
    """
    for handle in handles:
        click.echo(lower(handle))
    click.echo("\nResolved versions:")
    for handle in graph_handles(handles):
        flag = "  (mutable — rebuilds every run)" if is_mutable_version(handle.version) else ""
        click.echo(f"  {handle.name}@{handle.version}{flag}")


def _run_locally(handles: list[ArtifactStep], max_concurrent: int) -> None:
    run(*handles, max_concurrent=max_concurrent)


def build_options_with_runner(fn: Callable[..., BuildResult], runner: BuildRunner) -> Callable[..., None]:
    """Add shared build options and delegate ``--run`` to a domain-specific runner."""

    @click.option(
        "--version",
        "default_version",
        default=None,
        callback=_version_callback,
        help="Run-wide default version for artifacts that defer (omit version=). A calendar "
        "version YYYY.MM.DD, or 'dev' / '<label>-dev' for a fresh run with a unique suffix.",
    )
    @click.option(
        "--resume",
        "resume_version",
        default=None,
        callback=_resume_callback,
        metavar="VERSION",
        help="Continue an earlier dev run by reusing its exact version, as printed when it started.",
    )
    @click.option(
        "--override",
        "overrides",
        multiple=True,
        metavar="NAME=VERSION",
        callback=_override_callback,
        help="Per-artifact version override, keyed by the artifact's name. Repeatable.",
    )
    @click.option("--run", "do_run", is_flag=True, help="Build the handles (default: print the plan).")
    @click.option("--max-concurrent", type=int, default=8, show_default=True, help="Max steps built concurrently.")
    @functools.wraps(fn)
    def wrapper(
        *args: Any,
        default_version: str | None,
        resume_version: str | None,
        overrides: dict[str, str],
        do_run: bool,
        max_concurrent: int,
        **kwargs: Any,
    ) -> None:
        if (default_version is None) == (resume_version is None):
            raise click.UsageError("pass exactly one of --version or --resume")
        if resume_version is not None:
            version = resume_version
        else:
            assert default_version is not None
            version = run_version(default_version, now=datetime.now(), token=secrets.token_hex(2))
        if is_mutable_version(version):
            click.echo(f"Run version: {version} (continue this run with --resume {version})")
        codex = VersionCodex(default=version, overrides=overrides)
        with build_context(BuildContext(versions=codex)):
            handles = _as_handles(fn(*args, **kwargs))
        unused = codex.unused_overrides()
        if unused:
            raise click.BadParameter(
                f"never applied to any artifact: {sorted(unused)}. The name is a typo, or that "
                "artifact hardcodes its version (an explicit version wins over an override).",
                param_hint="--override",
            )
        if not do_run:
            _print_plan(handles)
            return
        runner(handles, max_concurrent)

    return wrapper


def build_options(fn: Callable[..., BuildResult]) -> Callable[..., None]:
    """Add the shared version/run options to a command that returns the experiment's handle(s).

    ``fn`` returns one :class:`~poorin.execution.lazy.ArtifactStep`, a list/tuple of them, or a
    name→handle mapping; it is called inside a :class:`~poorin.execution.build_context.BuildContext`
    built from ``--version`` and ``--override``. Without ``--run`` the lowered plan is printed;
    with it the handles are built. An ``--override`` naming an artifact that never deferred (a typo,
    or one that hardcodes its version) is rejected, so a silently-ignored override cannot slip a run
    under the wrong version.
    """

    return build_options_with_runner(fn, _run_locally)


def experiment_main(build: Callable[[], BuildResult]) -> Callable[..., None]:
    """Turn a nullary ``build()`` into a ready-to-run experiment command with the shared options.

    ``experiment_main(build)()`` is the whole ``if __name__ == "__main__":`` body for an experiment
    that takes no options of its own. For one that does, use :func:`build_options` directly (see the
    module docstring for the stacking order).
    """
    return click.command()(build_options(build))
