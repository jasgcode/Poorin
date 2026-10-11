# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List the runs under the prefix and show the code behind one (D107).

    uv run --no-sync python -m poorin.experiment.runs list [FILTER] [--limit N]
    uv run --no-sync python -m poorin.experiment.runs show NAME@VERSION | PATH

Both read what the runner writes beside each output, the record (``.artifact.json``) and the
status file, so they need neither W&B nor the experiment's code. Git, when the commit is in this
repository, adds what a record does not hold: how far its branch has moved since, a snapshot's line
counts, and the subject of a commit recorded before subjects were.
"""

import dataclasses
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath

import click
from rigging.filesystem.cluster_config import poorin_prefix
from rigging.filesystem.factory import url_to_fs
from rigging.provenance import Provenance, read_git

from poorin.execution.artifact import ArtifactRecord, read_record
from poorin.execution.step_status import STATUS_DEP_FAILED, STATUS_FAILED, STATUS_SUCCESS, StatusFile

# The file beside every output the runner scheduled; it marks an artifact's directory.
STATUS_FILENAME = ".executor_status"
# Prefix directories that hold scratch space, not artifacts.
SCRATCH_DIRS = frozenset({"tmp"})
DEFAULT_LIMIT = 30
# Hash characters `show` prints in a command: unambiguous in a repository of any size.
SHOWN_HASH_LENGTH = 12
# Continuation lines of `show` start under its values.
SHOW_INDENT = " " * 15
NO_CODE = "-"


class RunState(StrEnum):
    BUILT = "built"
    STARTED = "started"
    FAILED = "failed"


# What `show` says in its Code line for an output whose record is missing, by the output's state.
NO_RECORD_REASONS = {
    RunState.STARTED: "not recorded yet: the record is written when the build finishes",
    RunState.FAILED: "not recorded: the build failed before writing its record",
    RunState.BUILT: "not recorded: the output has no record",
}
# ... and for a record written before provenance existed, or a legacy sidecar.
NO_PROVENANCE_REASON = "not recorded: the output predates code provenance"


@dataclass(frozen=True)
class Run:
    """One output under the prefix."""

    label: str
    """``name@version``."""
    path: str
    state: RunState
    record: ArtifactRecord | None
    """``None`` until the build finishes."""
    when: datetime


def _state(output_path: str) -> RunState:
    status = StatusFile(output_path, worker_id="runs").status
    if status == STATUS_SUCCESS:
        return RunState.BUILT
    if status in (STATUS_FAILED, STATUS_DEP_FAILED):
        return RunState.FAILED
    return RunState.STARTED


def _naive_local(moment: datetime) -> datetime:
    return moment.astimezone().replace(tzinfo=None) if moment.tzinfo is not None else moment


def _output_dirs(fs, directory: str, top: bool) -> Iterator[str]:
    """Directories under ``directory`` that hold an output; an output holds no other output."""
    entries = fs.ls(directory, detail=True)
    if any(PurePosixPath(entry["name"]).name == STATUS_FILENAME for entry in entries):
        yield directory
        return
    for entry in entries:
        name = PurePosixPath(entry["name"]).name
        if entry["type"] == "directory" and not (top and name in SCRATCH_DIRS):
            yield from _output_dirs(fs, entry["name"], top=False)


def collect_runs(prefix: str) -> list[Run]:
    """Every output under ``prefix``, newest first."""
    fs, root = url_to_fs(prefix)
    if not fs.exists(root):
        return []
    runs = []
    for directory in _output_dirs(fs, root, top=True):
        relative = PurePosixPath(directory).relative_to(PurePosixPath(root))
        path = f"{prefix.rstrip('/')}/{relative}"
        record = read_record(path)
        created = record.provenance.created_at if record is not None and record.provenance is not None else ""
        when = (
            datetime.fromisoformat(created) if created else _naive_local(fs.modified(f"{directory}/{STATUS_FILENAME}"))
        )
        runs.append(Run(f"{relative.parent}@{relative.name}", path, _state(path), record, when))
    return sorted(runs, key=lambda r: r.when, reverse=True)


def _commit(rev: str) -> str:
    """The full hash of the commit ``rev`` names when it is in this repository, else "".

    A record's commits come from a file, so every later Git call takes this hash, never the recorded
    value; ``--end-of-options`` stops a value that starts with "-" from being read as an option here.
    """
    return read_git(["rev-parse", "--verify", "--quiet", "--end-of-options", f"{rev}^{{commit}}"]).strip()


def _with_subject(provenance: Provenance) -> Provenance:
    """``provenance`` with its commit's subject looked up when the record predates subjects."""
    if provenance.commit_subject or not provenance.base_commit:
        return provenance
    full = _commit(provenance.base_commit)
    subject = read_git(["log", "-1", "--no-show-signature", "--format=%s", full]).strip() if full else ""
    return dataclasses.replace(provenance, commit_subject=subject)


def _code_label(run: Run) -> str:
    if run.record is None or run.record.provenance is None:
        return NO_CODE
    return _with_subject(run.record.provenance).label


def _time(iso: str) -> str:
    return f"{datetime.fromisoformat(iso):%Y-%m-%d %H:%M}" if iso else "unknown"


def _commit_lines(provenance: Provenance, full: str) -> list[str]:
    when = f", {_time(provenance.commit_time)}" if provenance.commit_time else ""
    branch = provenance.branch or "a detached HEAD"
    lines = [f"  Commit       {provenance.base_commit} on {branch}{when}"]
    if provenance.commit_subject:
        lines.append(f"{SHOW_INDENT}{provenance.commit_subject}")
    if not full:
        lines.append(f"{SHOW_INDENT}not in this repository")
        return lines
    if provenance.branch:
        moved = read_git(["rev-list", "--count", f"{full}..{provenance.branch}"]).strip()
        if moved == "0":
            lines.append(f"{SHOW_INDENT}{provenance.branch} has not moved since")
        elif moved:
            noun = "commit" if moved == "1" else "commits"
            lines.append(f"{SHOW_INDENT}{provenance.branch} has moved {moved} {noun} past it")
    return lines


def _line_counts(base: str, snapshot: str) -> dict[str, str]:
    """Per-path line counts of the snapshot against its base, both full hashes."""
    counts = {}
    for row in read_git(["diff", "--numstat", base, snapshot]).splitlines():
        added, removed, path = row.split("\t", 2)
        counts[path] = "binary" if added == "-" else f"+{added} -{removed}" if removed != "0" else f"+{added}"
    return counts


def _uncommitted_lines(provenance: Provenance, full: str) -> list[str]:
    snapshot = provenance.snapshot_commit
    kept = _commit(snapshot) if snapshot else ""
    count = len(provenance.uncommitted)
    files = f"{count} file{'' if count == 1 else 's'}" if count else "changes"
    if not snapshot:
        saved = "not saved"
    else:
        saved = f"saved as snapshot {snapshot[:SHOWN_HASH_LENGTH]}" + ("" if kept else " (not in this repository)")
    lines = [f"  Uncommitted  {files}, {saved}"]
    counts = _line_counts(full, kept) if full and kept else {}
    width = max((len(entry) - 3 for entry in provenance.uncommitted), default=0)
    for entry in provenance.uncommitted:
        status, path = entry[:2].strip(), entry[3:]
        lines.append(f"{SHOW_INDENT}  {status:<3} {path:<{width}}  {counts.get(path, '')}".rstrip())
    # The commands print the hashes Git verified, never the record's text.
    short = kept[:SHOWN_HASH_LENGTH]
    if full and kept:
        lines.append(f"  See changes  git diff {full[:SHOWN_HASH_LENGTH]} {short}")
    if kept:
        lines.append(f"  Restore      git worktree add ../run-{short} {short}")
    return lines


def describe(run: Run) -> list[str]:
    """Everything the record and Git say about one run's code."""
    lines = [f"{run.label}   {run.state}"]
    if run.record is None or run.record.provenance is None:
        reason = NO_RECORD_REASONS[run.state] if run.record is None else NO_PROVENANCE_REASON
        lines.append(f"  Code         {reason}")
        return lines
    provenance = _with_subject(run.record.provenance)
    lines.append(f"  Built        {_time(provenance.created_at)} by {provenance.built_by or 'unknown'}")
    if provenance.command_line:
        lines.append(f"  Command      {shlex.join(provenance.command_line)}")
    if provenance.note:
        lines.append(f"  Note         {provenance.note}")
    if not provenance.base_commit:
        lines.append("  Commit       none: not built from a Git checkout")
    else:
        full = _commit(provenance.base_commit)
        lines.extend(_commit_lines(provenance, full))
        if provenance.dirty:
            lines.extend(_uncommitted_lines(provenance, full))
    if run.record.deps:
        lines.append(f"  Inputs       {', '.join(run.record.deps)}")
    return lines


@click.group()
def main() -> None:
    """List the runs under the prefix, and show the code behind one."""


@main.command("list")
@click.argument("name_filter", required=False, default="")
@click.option("--limit", type=click.IntRange(min=1), default=DEFAULT_LIMIT, show_default=True)
@click.option("--prefix", default=None, help="Read this prefix instead of POORIN_PREFIX.")
def list_runs(name_filter: str, limit: int, prefix: str | None) -> None:
    """One line per run, newest first; FILTER keeps runs whose name contains it."""
    runs = [r for r in collect_runs(prefix or poorin_prefix()) if name_filter in r.label][:limit]
    width = max((len(r.label) for r in runs), default=0)
    for r in runs:
        click.echo(f"{r.when:%Y-%m-%d %H:%M}  {r.state:<7}  {r.label:<{width}}  {_code_label(r)}")


@main.command()
@click.argument("target")
@click.option("--prefix", default=None, help="Read this prefix instead of POORIN_PREFIX.")
def show(target: str, prefix: str | None) -> None:
    """Everything about one run's code. TARGET is NAME@VERSION or the output's path."""
    if "@" in target:
        name, _, version = target.rpartition("@")
        path = f"{(prefix or poorin_prefix()).rstrip('/')}/{name}/{version}"
    else:
        path = target.rstrip("/")
    record = read_record(path)
    fs, stripped = url_to_fs(path)
    if record is None and not fs.exists(f"{stripped}/{STATUS_FILENAME}"):
        raise click.ClickException(f"no run at {path}")
    relative = PurePosixPath(path)
    label = target if "@" in target else f"{relative.parent.name}@{relative.name}"
    click.echo("\n".join(describe(Run(label, path, _state(path), record, datetime.now()))))


if __name__ == "__main__":
    main()
