# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provenance of a built artifact: a content-addressed tree hash plus the human
and launch context (branch, base commit, builder, origin, time, argv) needed to read it.

The ``tree_hash`` is content-addressed and deterministic — the same working-tree
content always produces the same hash regardless of when or by whom it is built —
so it is the natural deduplication key for image tags and version comparisons. The
remaining fields render that hash in a form a human can recognize and record the
launch that produced an artifact.

Two constructors capture it. :meth:`Provenance.from_git` is strict — it raises if git
is unavailable — for an image build, where the content-addressed ``tree_hash`` must
exist. :meth:`Provenance.capture` is best-effort — every field degrades to empty
outside a checkout — for stamping an artifact built by an arbitrary launch.

A dirty tree's changes can be saved with :func:`save_snapshot`, a commit kept under
``refs/poorin/snapshots/`` that leaves the branch and index alone.
"""

import contextlib
import dataclasses
import functools
import getpass
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from rigging.timing import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

# The longest note or commit subject a label shows before cutting it with "…".
LABEL_TEXT_WIDTH = 60
# The most uncommitted files a label names; more end in ", …".
LABEL_FILES = 3
# Snapshot commits are kept under this ref namespace, one ref per commit, so `git gc` keeps them.
SNAPSHOT_REF_PREFIX = "refs/poorin/snapshots/"
# How long a snapshot waits for a concurrent launch of the same state to finish writing its ref.
SNAPSHOT_REF_WAIT = 5.0
# The author and committer of every snapshot: a snapshot needs no Git identity on the machine, and
# one state gives one commit on any machine. The record's ``built_by`` names who ran it.
SNAPSHOT_AUTHOR_NAME = "Poorin snapshot"
SNAPSHOT_AUTHOR_EMAIL = "snapshot@poorin.invalid"


@dataclass(frozen=True)
class Provenance:
    """Identity, human context, and launch context of a built artifact.

    Attributes:
        tree_hash: Short git tree hash of the tracked working-tree content. The
            deduplication key — content-addressed and stable across builds.
        base_commit: Short hash of the HEAD commit the tree was built from.
        dirty: Whether the working tree had uncommitted or untracked changes.
        branch: Branch name at build time, or ``None`` for a detached HEAD.
        built_by: OS user that ran the build, or ``None`` if unknown.
        git_remote: ``origin`` remote URL, or ``None`` if unknown.
        created_at: ISO-8601 time the launch was captured, or ``""`` if uncaptured.
        command_line: ``sys.argv`` of the launching process; empty if uncaptured.
        commit_subject: First line of the HEAD commit's message, or "" if uncaptured.
        commit_time: HEAD's committer date, ISO 8601, or "" if uncaptured.
        uncommitted: The working tree's changes as ``"XY path"`` entries (``git status
            --porcelain`` codes), untracked files included; a rename is listed under its new path.
        snapshot_commit: The commit that saved the uncommitted changes (``save_snapshot``), if any.
        note: The launch's own description of what the run tries, if given.
    """

    tree_hash: str
    base_commit: str
    dirty: bool
    branch: str | None
    built_by: str | None
    git_remote: str | None = None
    created_at: str = ""
    command_line: tuple[str, ...] = ()
    commit_subject: str = ""
    commit_time: str = ""
    uncommitted: tuple[str, ...] = ()
    snapshot_commit: str | None = None
    note: str | None = None

    @classmethod
    def from_git(cls, repo_dir: Path | None = None) -> "Provenance":
        """Capture build provenance from the git repository at ``repo_dir`` (or cwd).

        Strict: raises if git is unavailable, since the content-addressed
        ``tree_hash`` is the reason to build at all.
        """
        cwd = str(repo_dir) if repo_dir is not None else None
        # `git stash create` captures any dirty state as a transient commit whose
        # tree we deref below — dropping the per-call timestamp a commit hash
        # would carry, so tree_hash stays purely content-addressed. Empty (no
        # output) means a clean tree, so fall back to HEAD.
        stash = _git(["stash", "create"], cwd)
        ref = stash or "HEAD"
        return cls(
            tree_hash=_git(["rev-parse", "--short", f"{ref}^{{tree}}"], cwd),
            base_commit=_git(["rev-parse", "--short", "HEAD"], cwd),
            dirty=bool(_git(["status", "--porcelain", "--untracked-files=all"], cwd)),
            branch=_git(["symbolic-ref", "--short", "-q", "HEAD"], cwd, check=False) or None,
            built_by=_getuser(),
        )

    @classmethod
    def capture(cls, repo_dir: Path | None = None) -> "Provenance":
        """Best-effort provenance of the current launch: git context plus the origin
        remote, capture time, and argv. It also records HEAD's subject and date and the
        uncommitted paths.

        Tolerant git capture: every git field degrades to empty/``None`` outside a checkout or
        when the ``git`` binary is absent, so it never raises. Use to stamp an artifact built by
        an arbitrary run.
        """
        cwd = str(repo_dir) if repo_dir is not None else None

        def g(*args: str) -> str:
            return read_git(args, cwd).strip()

        stash = g("stash", "create")
        ref = stash or "HEAD"
        uncommitted = _status_entries(read_git(["status", "--porcelain", "-z", "--untracked-files=all"], cwd))
        commit_time, _, commit_subject = g("log", "-1", "--no-show-signature", "--format=%cI%n%s").partition("\n")
        return cls(
            tree_hash=g("rev-parse", "--short", f"{ref}^{{tree}}"),
            base_commit=g("rev-parse", "--short", "HEAD"),
            dirty=bool(uncommitted),
            branch=g("symbolic-ref", "--short", "-q", "HEAD") or None,
            built_by=_getuser(),
            git_remote=g("remote", "get-url", "origin") or None,
            created_at=datetime.now().isoformat(),
            command_line=tuple(sys.argv),
            commit_subject=commit_subject,
            commit_time=commit_time,
            uncommitted=uncommitted,
        )

    @classmethod
    def from_json(cls, raw: str) -> "Provenance":
        d = json.loads(raw)
        built_by = d.get("built_by")
        if built_by is not None and not isinstance(built_by, str):
            # built_by feeds per-user namespacing; a non-string is a malformed payload,
            # not an identity.
            raise TypeError(f"built_by must be a string or null, got {type(built_by).__name__}")
        return cls(
            tree_hash=d["tree_hash"],
            base_commit=d["base_commit"],
            dirty=bool(d["dirty"]),
            branch=d.get("branch"),
            built_by=built_by,
            git_remote=d.get("git_remote"),
            created_at=d.get("created_at", ""),
            command_line=tuple(d.get("command_line", ())),
            commit_subject=d.get("commit_subject", ""),
            commit_time=d.get("commit_time", ""),
            uncommitted=tuple(d.get("uncommitted", ())),
            snapshot_commit=d.get("snapshot_commit"),
            note=d.get("note"),
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @property
    def dedup_key(self) -> str:
        """The value to key image tags and version comparisons on."""
        return self.tree_hash

    @property
    def change_summary(self) -> str:
        """The uncommitted files in a few words, such as ``3 uncommitted: llama.py, model.py, notes.md``."""
        if not self.uncommitted:
            return "uncommitted changes"
        names = [PurePosixPath(entry[3:]).name for entry in self.uncommitted]
        shown = ", ".join(names[:LABEL_FILES]) + (", …" if len(names) > LABEL_FILES else "")
        return f"{len(names)} uncommitted: {shown}"

    @property
    def label(self) -> str:
        """The code in one line: branch, short commit, and the subject, the note, or the changes."""
        if not self.base_commit:
            return "no git checkout"
        head = f"{self.branch or 'detached'}@{self.base_commit[:7]}"
        if not self.dirty:
            return f"{head}  {_one_line(self.note or self.commit_subject)}".rstrip()
        if self.note:
            changes = f"+ {len(self.uncommitted)} uncommitted  {_one_line(self.note)}"
        else:
            changes = f"+ {self.change_summary}"
        snapshot = f"  (snapshot {self.snapshot_commit[:7]})" if self.snapshot_commit else ""
        return f"{head} {changes}{snapshot}"

    def __str__(self) -> str:
        return self.label


def read_git(args: Sequence[str], repo_dir: Path | str | None = None) -> str:
    """Raw stdout of ``git args`` run in ``repo_dir``; "" when Git fails or is not installed.

    Bytes that are not UTF-8, such as in a file name, decode to U+FFFD instead of raising.
    """
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, errors="replace", cwd=repo_dir)
    except FileNotFoundError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def _status_entries(porcelain_z: str) -> tuple[str, ...]:
    """``git status --porcelain -z`` output as ``"XY path"`` entries.

    ``-z`` leaves paths unquoted. A rename or copy is followed by its source path, which is skipped,
    so it is listed once under its new path.
    """
    entries: list[str] = []
    fields = iter(porcelain_z.split("\0"))
    for field in fields:
        if not field:
            continue
        entries.append(field)
        if "R" in field[:2] or "C" in field[:2]:
            next(fields, None)
    return tuple(entries)


def _one_line(text: str) -> str:
    """``text`` on one line, cut with "…" at the label's width."""
    flat = " ".join(text.split())
    return flat if len(flat) <= LABEL_TEXT_WIDTH else flat[: LABEL_TEXT_WIDTH - 1] + "…"


@functools.cache
def _capture_once() -> Provenance:
    return Provenance.capture()


# functools.cache does not serialize concurrent first calls, and two concurrent
# `git stash create` runs race on the repo's index.lock — the loser silently stamps
# a dirty tree as clean HEAD. The lock makes the cache-filling capture exclusive.
_capture_lock = threading.Lock()


# The provenance an experiment command installed for its launch, innermost last.
_launches: list[Provenance] = []


@contextlib.contextmanager
def launched_as(provenance: Provenance) -> Iterator[Provenance]:
    """Make ``provenance`` this process's launch provenance until the block exits.

    The experiment command line installs the provenance it captured, snapshot and note included, so
    the artifact records and the W&B tracker of the jobs it runs, which are threads of this
    process, record the same launch.
    """
    with _capture_lock:
        _launches.append(provenance)
    try:
        yield provenance
    finally:
        with _capture_lock:
            _launches.remove(provenance)


def current_launch() -> Provenance | None:
    """The provenance installed with :func:`launched_as`, or ``None`` outside one."""
    with _capture_lock:
        return _launches[-1] if _launches else None


def launch_provenance() -> Provenance:
    """The current launch's provenance: the installed one, else captured once per process (thread-safe)."""
    with _capture_lock:
        return _launches[-1] if _launches else _capture_once()


def save_snapshot(message: str, repo_dir: Path | None = None) -> str:
    """Commit the working tree, untracked files included, without touching the branch, index, or
    files, and return the commit's full hash.

    The tree is built in a temporary index, so ``.gitignore`` applies and the real index is left
    alone. The commit's parent is ``HEAD``, its dates are ``HEAD``'s committer date, and its author
    and committer are the fixed snapshot identity, so one tree, parent, and message always give one
    commit, and repeated runs of one dirty state share it. A ref under :data:`SNAPSHOT_REF_PREFIX`
    keeps it from ``git gc``.

    Raises:
        RuntimeError: Outside a checkout, or in a repository with no commits.
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    head = _git(["rev-parse", "HEAD"], cwd)
    head_date = _git(["log", "-1", "--no-show-signature", "--format=%cI", "HEAD"], cwd)
    with tempfile.TemporaryDirectory(prefix="poorin-snapshot-") as tmp:
        env = {
            **os.environ,
            "GIT_INDEX_FILE": os.path.join(tmp, "index"),
            "GIT_AUTHOR_DATE": head_date,
            "GIT_COMMITTER_DATE": head_date,
            "GIT_AUTHOR_NAME": SNAPSHOT_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": SNAPSHOT_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": SNAPSHOT_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": SNAPSHOT_AUTHOR_EMAIL,
        }
        _git(["read-tree", "HEAD"], cwd, env=env)
        _git(["add", "--all"], cwd, env=env)
        tree = _git(["write-tree"], cwd, env=env)
        commit = _git(["commit-tree", tree, "-p", head, "-m", message], cwd, env=env)
    ref = f"{SNAPSHOT_REF_PREFIX}{commit}"
    try:
        _git(["update-ref", ref, commit], cwd)
    except RuntimeError:
        # Another launch of the same state holds the ref's lock and is writing this same commit.
        written = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
            lambda: read_git(["rev-parse", "--verify", "--quiet", ref], cwd).strip() == commit,
            timeout=Duration.from_seconds(SNAPSHOT_REF_WAIT),
        )
        if not written:
            raise
    return commit


def _git(args: list[str], cwd: str | None, check: bool = True, env: Mapping[str, str] | None = None) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _getuser() -> str | None:
    # getpass.getuser() raises when no username resolves from the environment or the
    # password database; provenance is best-effort, so treat that as unknown.
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return None


# A path segment is lowercase alphanumerics plus '_' and '-'; collapse anything else.
_USER_SEGMENT_RE = re.compile(r"[^a-z0-9_-]+")

# Machine/service logins that should not own a per-user namespace: resolving to one of
# these means the run was launched from a shared or service account.
_MACHINE_LOGINS = frozenset({"root", "runner", "ubuntu", "exedev"})


def username_segment() -> str:
    """The current user as a path-safe segment, for per-user artifact namespacing.

    Resolves the same user that stamps provenance ``built_by``: the local OS login. The raw
    name is reduced to a clean segment: an email-like login drops its domain but keeps the
    whole local name (``russell.power@host`` → ``russell-power``), and the result is
    lowercased with any remaining character collapsed to ``-``. The full local name is kept so distinct users stay
    distinct. Raises ``RuntimeError`` if no username resolves — per-user namespacing must never
    silently fall back to a shared bucket.
    """
    raw = _getuser()
    if not raw:
        raise RuntimeError("cannot resolve a username for per-user namespacing: getpass.getuser found none")
    name = raw.strip()
    if "@" in name:
        name = name.split("@", 1)[0]
    segment = _USER_SEGMENT_RE.sub("-", name.lower()).strip("-")
    if not segment:
        raise RuntimeError(f"username {raw!r} did not sanitize to a usable path segment")
    if segment in _MACHINE_LOGINS:
        logger.warning(
            "per-user namespace resolved to machine login %r; launch from your own account",
            segment,
        )
    return segment
