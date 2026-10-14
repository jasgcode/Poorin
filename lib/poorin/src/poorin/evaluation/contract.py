# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a template implements so its runs can be evaluated (D114).

An evaluation is an artifact step (:mod:`poorin.experiment.evaluate`). Its inputs are a finished run
and one of its permanent checkpoint steps, its recipe is a :class:`Suite`, and its output is an
:class:`EvalResult`. A template's training run records a checkpoint type that implements
:class:`EvaluableCheckpoint` and names the suites its runs offer. ``poorin`` never imports a
template: a suite decodes its own recipe from :class:`FinishedRun` (:func:`recorded`) and loads its
own checkpoint.
"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar, NamedTuple, Protocol, TypeVar, runtime_checkable

from draccus.utils import DecodingError, ParsingError
from rigging.filesystem.factory import url_to_fs

from poorin.execution.artifact import Artifact, ArtifactRecord, JSONValue, read_record, result_type_name
from poorin.execution.recorded_config import decode_recorded
from poorin.training.training import DEFAULT_CHECKPOINTS_PATH

T = TypeVar("T")

MIGRATE_COMMAND = "python -m poorin.experiment.runs migrate {label} --to {checkpoint_type}"
CHECKPOINT_STEP_PREFIX = "step-"


class FinishedRun(NamedTuple):
    """A finished training run, as a suite reads it."""

    label: str
    """Its full ``name@version``."""
    path: str
    """Its output directory."""
    record: ArtifactRecord


class EvalResult(Artifact):
    """One checkpoint of one run, scored by one suite."""

    run: str
    """The evaluated run's full ``name@version``."""
    step: int
    suite: str
    metrics: dict[str, float]
    """Keyed as the run logs them in training: ``eval/loss``, ``eval/video_loss``, ..."""
    examples: int
    """Clips or sequences scored."""
    samples_path: str | None = None
    """A per-example file, for a suite that keeps one."""


@dataclass(frozen=True)
class EvalRuntime:
    """Execution choices a suite receives outside its recipe."""

    batch_size: int | None
    """Device batch size, for a suite whose result does not depend on it; ``None`` lets it choose."""


class Suite(Protocol):
    """A frozen dataclass whose fields are an evaluation's recipe: changing one is a new evaluation."""

    name: ClassVar[str]

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult: ...


@runtime_checkable
class EvaluableCheckpoint(Protocol):
    """The artifact type a template's training run records: it names the suites its runs offer."""

    @classmethod
    def suites(cls) -> Mapping[str, type[Suite]]: ...


def finished_run_at(path: str) -> FinishedRun:
    """The finished run whose output is ``path``."""
    record = read_record(path)
    if record is None:
        raise FileNotFoundError(f"{path} has no record, so it is not a finished run")
    return FinishedRun(f"{record.name}@{record.version}", path, record)


def artifact_class(result_type: str) -> type[Artifact]:
    """The class a record's ``result_type`` names."""
    module, _, name = result_type.rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, Artifact)):
        raise TypeError(f"{result_type} is not an Artifact class")
    return cls


def offered_suites(label: str, artifact_type: type[Artifact]) -> Mapping[str, type[Suite]]:
    """The suites the run ``label``'s checkpoint type offers."""
    if not issubclass(artifact_type, EvaluableCheckpoint):
        command = MIGRATE_COMMAND.format(label=label, checkpoint_type="MODULE.CLASS")
        raise ValueError(
            f"{label} records {result_type_name(artifact_type)}, which offers no evaluation suites. A run "
            f"trained before evaluation artifacts (D114) records LevanterCheckpoint; name its template's "
            f"checkpoint type once with: {command}"
        )
    return artifact_type.suites()


def checkpoint_path(run: FinishedRun, step: int) -> str:
    """The run's permanent checkpoint at ``step``."""
    return f"{run.path}/{DEFAULT_CHECKPOINTS_PATH}/{CHECKPOINT_STEP_PREFIX}{step}"


def checkpoint_steps(run_path: str) -> list[int]:
    """The steps of the run's permanent checkpoints, ascending."""
    fs, root = url_to_fs(f"{run_path}/{DEFAULT_CHECKPOINTS_PATH}")
    if not fs.exists(root):
        return []
    steps = []
    for entry in fs.ls(root, detail=False):
        number = PurePosixPath(entry).name.removeprefix(CHECKPOINT_STEP_PREFIX)
        if PurePosixPath(entry).name.startswith(CHECKPOINT_STEP_PREFIX) and number.isdigit():
            steps.append(int(number))
    return sorted(steps)


def recorded_value(run: FinishedRun, path: str) -> JSONValue:
    """The raw value at the dotted ``path`` of the run's recorded config."""
    value: Any = run.record.config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"{run.label} records no config field {path!r}")
        value = value[key]
    return value


def recorded(run: FinishedRun, path: str, cls: type[T]) -> T:
    """The value at the dotted ``path`` of the run's recorded config, decoded as ``cls``."""
    try:
        return decode_recorded(cls, recorded_value(run, path))
    except (DecodingError, ParsingError) as e:
        command = MIGRATE_COMMAND.format(label=run.label, checkpoint_type=run.record.result_type)
        raise ValueError(
            f"{run.label}: its recorded {path} does not decode as {cls.__name__}: {e}. A field the current "
            f"code moved out of the recipe is dropped once with: {command} --drop {path}.FIELD"
        ) from e
