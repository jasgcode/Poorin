# Evaluation Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluations become artifact steps over finished runs, with a contract every template can implement, short run references, and a post-hoc held-out loss suite for the episode templates and grug base that reproduces the held-out loss each run logged in training.

**Architecture:** `poorin.evaluation.contract` defines the protocols and the `EvalResult` artifact; `poorin.experiment.evaluate` builds evaluation handles over pinned finished runs and runs a run-by-step matrix through the experiment CLI; `poorin.experiment.run_refs` resolves `NAME@60b9` / `NAME@accept` / `NAME@latest`. Each template declares a checkpoint type (`EpisodeCheckpoint`, `GrugCheckpoint`) that names its suites, decodes its own recipe from the run's record with `poorin.execution.recorded_config`, and scores a checkpoint with the same function its training loop uses.

**Tech Stack:** The lazy artifact layer (`ArtifactStep`, pins, records), Click and the experiment CLI (`build_options`), draccus decoding, JAX and Levanter's checkpoint loader, Levanter's `TaggedEvaluator` for grug.

**Spec:** `docs/poorin/specs/2026-10-13-evaluation-artifacts-design.md`, decision D114.

## Global Constraints

- Run every command with `uv run --no-sync`. Unit tests run on CPU (`JAX_PLATFORMS=cpu`); run one file in-process with `-n 0`.
- Python >= 3.13; all imports at the top of the file; no `TYPE_CHECKING`; no `*_utils.py`; no `__all__` added.
- `poorin` never imports a template (`experiments.*`). Dependency direction is unchanged.
- An evaluation's name is `eval/{run name without users/<user>/}/{run version}/{suite segment}/step-{N}`, user-namespaced for a dev evaluation version.
- An evaluation's fingerprint is the run's `name@version`, the step, and the suite's class and fields; the device batch size is a runtime arg.
- The episode suite's defaults are training's: `clips = 128`, `batch_size = 32`, `flow_times = 8`, `seed = 0` (`EVAL_SEED`).
- Grug's suite: `max_sequences = None`, `weights = ema` (falls back to current weights for a run without EMA).
- Metric keys are the ones training logs, without timing keys.
- Both suites raise `FloatingPointError` on a non-finite metric.
- The process lock is renamed `exclusive_accelerator`; every evaluation step holds it.
- Every new module gets a row in `docs/poorin/architecture.md` (`tests/tools/test_architecture.py` checks the list).
- Commits: `[poorin]` (or `[grug]`) prefix, imperative, at most 72 characters, no attribution trailer. Each decision or ruling goes in `docs/poorin/decisions.md` (D114) in the same commit as the change.
- A run's logged evaluation at tracker step `N - 1` is the checkpoint `step-N`: both loops log at the index of the step just taken and checkpoint at the count of steps taken.

## Review Focus

- A dev run that is evaluated and then resumed: evaluating it must never touch its output (status file, record, checkpoints). The step runner rebuilds every dev version, a pinned one included, so without the `StepSpec.rebuilds` fix the runner would treat the pinned run as a build.
- A run recorded before the current code dropped a config field (D113's acceptance runs hold `data.num_workers`): the suite refuses with the `runs migrate ... --drop` command, never a bare draccus traceback, and never silently ignores the field.
- A token two runs share, or a label with several runs: refused, listing the candidates; a label resolves to the latest by the timestamp in the version, not by file time.
- `--steps 750` on a run with checkpoints at 500, 1000, ...: refused before anything runs, listing the steps that exist.
- Grug's held-out loss with a different device batch size: the same result, since the batch size is outside the fingerprint.

Tests pinning each line: Task 5 (`test_evaluating_a_pinned_dev_run_leaves_it_untouched`), Task 8 (`test_a_record_with_a_field_the_code_dropped_names_the_migration`), Task 4 (`test_a_token_two_runs_share_is_refused_listing_both`, `test_a_label_resolves_to_its_latest_run_by_version_time`), Task 7 (`test_a_step_without_a_checkpoint_is_refused_listing_the_steps`), Task 9 (`test_the_held_out_loss_does_not_depend_on_the_device_batch_size`).

---

### Task 1: Spec, plan, and decision

**Files:**
- Create: `docs/poorin/specs/2026-10-13-evaluation-artifacts-design.md` (written and amended)
- Create: `docs/poorin/plans/2026-10-13-evaluation-artifacts.md` (this file)
- Modify: `docs/poorin/decisions.md` (D114, written)

- [ ] **Step 1: Commit**

```bash
git add docs/poorin/specs/2026-10-13-evaluation-artifacts-design.md docs/poorin/plans/2026-10-13-evaluation-artifacts.md docs/poorin/decisions.md
git commit -m "[poorin] Add the evaluation artifacts spec and plan (D114)"
```

### Task 2: Decode a recorded config value

**Files:**
- Create: `lib/poorin/src/poorin/execution/recorded_config.py`
- Modify: `lib/poorin/src/poorin/execution/fingerprint.py` (rename `_class_label` to `class_label`)
- Modify: `lib/poorin/src/poorin/experiment/overrides.py` (import `field_types` instead of defining `_field_types`)
- Modify: `docs/poorin/architecture.md` (row for `poorin.execution.recorded_config`)
- Test: `tests/execution/test_recorded_config.py`

**Interfaces:**
- Produces: `decode_recorded(cls: type[T], value: Any) -> T`; raises `draccus.utils.DecodingError` for a field `cls` lacks and `ValueError` for a recorded class that is neither the declared type nor one of its choices. `field_types(cls: type) -> dict[str, Any]` (moved from `overrides._field_types`). `class_label(cls: type) -> str` (renamed from `fingerprint._class_label`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/execution/test_recorded_config.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading a recorded config value back into its dataclass (poorin.execution.recorded_config)."""

import json
from dataclasses import dataclass, field

import pytest
from draccus.utils import DecodingError
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig, OptimizerConfig
from poorin.execution.fingerprint import canonical_json
from poorin.execution.recorded_config import decode_recorded


@dataclass(frozen=True)
class Clip:
    history: int = 2
    horizon: int = 2


@dataclass(frozen=True)
class Recipe:
    clip: Clip
    optimizer: OptimizerConfig
    adam: AdamConfig
    label: str | None
    sizes: tuple[int, ...] = (1, 2)
    weights: dict[str, float] = field(default_factory=dict)


def _recorded(value: object) -> object:
    """``value`` as a record stores it."""
    return json.loads(canonical_json(value))


def test_a_recorded_config_decodes_to_the_dataclass_that_wrote_it():
    recipe = Recipe(
        clip=Clip(history=4),
        optimizer=AdamConfig(learning_rate=3e-3),
        adam=AdamConfig(),
        label=None,
        weights={"a": 0.5},
    )
    assert decode_recorded(Recipe, _recorded(recipe)) == recipe


def test_a_recorded_mixture_decodes_with_its_component_choices():
    text = TextLmDatasetFormat()
    data = LmDataConfig(
        components={
            "wikitext": DatasetComponent(
                source=UrlDatasetSourceConfig(cache_dir="/c/wikitext", format=text),
                cache_dir="/c/wikitext",
                format=text,
            )
        },
        train_weights={"wikitext": 0.0},
        tokenizer="marin-community/marin-tokenizer",
    )
    assert _recorded(decode_recorded(LmDataConfig, _recorded(data))) == _recorded(data)


def test_a_field_the_class_no_longer_has_is_refused():
    recorded = _recorded(Clip())
    recorded["num_workers"] = 16
    with pytest.raises(DecodingError, match="num_workers"):
        decode_recorded(Clip, recorded)


def test_a_recorded_class_that_is_not_the_declared_one_is_refused():
    with pytest.raises(ValueError, match="adam"):
        decode_recorded(Clip, _recorded(AdamConfig()))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_recorded_config.py -n 0 -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'poorin.execution.recorded_config'`.

- [ ] **Step 3: Rename `_class_label` to `class_label` in `fingerprint.py`**

Rename the function and its one call in `_FingerprintEncoder.default`. Its docstring is unchanged.

- [ ] **Step 4: Write the module**

```python
# lib/poorin/src/poorin/execution/recorded_config.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a config value back from the canonical form a record stores (D114).

A record's ``config`` is the fingerprint's canonical JSON (:mod:`poorin.execution.fingerprint`):
each dataclass carries its class under ``__class__``, as its draccus choice name when it is a
registered choice (``adam``, ``cached``) and its class name otherwise. :func:`decode_recorded`
walks the target type's fields, so a choice gets draccus's ``type`` key only where the declared
type is a choice base, and hands the result to ``draccus.decode``. Evaluating a run reads the run's
own recipe this way.
"""

import inspect
import types
import typing
from typing import Any, TypeVar

import draccus
from draccus.choice_types import CHOICE_TYPE_KEY, ChoiceRegistryBase

from poorin.execution.fingerprint import CLASS_KEY, class_label

T = TypeVar("T")


def field_types(cls: type) -> dict[str, Any]:
    """The evaluated annotation of each field of ``cls`` and its bases.

    ``typing.get_type_hints`` returns nothing for a class marked ``@no_type_check``, as grug's launch
    config is (so the tests' beartype hook accepts the placeholders a fingerprint puts in its fields).
    Decoding still needs the annotations, so this reads each class's own. Only an annotation written
    whole as a string is evaluated: ``Annotated`` is not unwrapped, and a string forward reference
    nested inside a generic (``list["Foo"]``) is not resolved.
    """
    found: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        found.update(inspect.get_annotations(base, eval_str=True))
    return found


def decode_recorded(cls: type[T], value: Any) -> T:
    """``value``, a config value as a record stores it, decoded as ``cls``.

    Raises ``draccus.utils.DecodingError`` when the value holds a field ``cls`` no longer has, and
    ``ValueError`` when a recorded class is neither the declared type nor one of its choices.
    """
    return draccus.decode(cls, _draccus_form(cls, value))


def _names(tp: Any, label: str) -> bool:
    """Whether the recorded class ``label`` is ``tp`` or one of its registered choices."""
    if not isinstance(tp, type):
        return False
    if class_label(tp) == label:
        return True
    return issubclass(tp, ChoiceRegistryBase) and label in tp.get_known_choices()


def _dataclass_form(tp: Any, value: dict[str, Any]) -> dict[str, Any]:
    label = value[CLASS_KEY]
    if not _names(tp, label):
        raise ValueError(f"the record holds a {label}, which is neither {tp} nor one of its choices")
    choice = class_label(tp) != label
    concrete = tp.get_choice_class(label) if choice else tp
    hints = field_types(concrete)
    # A key the class lacks passes through untouched, so draccus refuses it by name.
    form = {key: _draccus_form(hints.get(key, Any), item) for key, item in value.items() if key != CLASS_KEY}
    if choice:
        form[CHOICE_TYPE_KEY] = label
    return form


def _draccus_form(tp: Any, value: Any) -> Any:
    """``value`` with each ``__class__`` key replaced by what draccus reads at type ``tp``."""
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        if value is None:
            return None
        arms = [arm for arm in typing.get_args(tp) if arm is not type(None)]
        if isinstance(value, dict) and CLASS_KEY in value:
            matching = [arm for arm in arms if _names(arm, value[CLASS_KEY])]
            if not matching:
                raise ValueError(f"the record holds a {value[CLASS_KEY]}, which no arm of {tp} names")
            return _dataclass_form(matching[0], value)
        return _draccus_form(arms[0], value) if len(arms) == 1 else value
    if isinstance(value, dict) and CLASS_KEY in value:
        return _dataclass_form(tp, value)
    args = typing.get_args(tp)
    if isinstance(value, dict):
        return {key: _draccus_form(args[1] if len(args) == 2 else Any, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_draccus_form(args[0] if args else Any, item) for item in value]
    return value
```

- [ ] **Step 5: Point `overrides.py` at the moved helper**

Delete `_field_types` from `overrides.py`, add `from poorin.execution.recorded_config import field_types`, and change its one call in `_replace_at` to `field_types(type(obj))[head]`.

- [ ] **Step 6: Add the architecture row**

In `docs/poorin/architecture.md`, under `### poorin`, beside `poorin.execution.lazy`:

```markdown
| `poorin.execution.recorded_config` | Reads a recorded config value back into its dataclass, choices included (D114) | R1 |
```

- [ ] **Step 7: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_recorded_config.py tests/experiment/test_overrides.py tests/execution/test_fingerprint.py -n 0 -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add lib/poorin/src/poorin/execution/recorded_config.py lib/poorin/src/poorin/execution/fingerprint.py lib/poorin/src/poorin/experiment/overrides.py docs/poorin/architecture.md tests/execution/test_recorded_config.py
git commit -m "[poorin] Decode a recorded config value back into its dataclass"
```

### Task 3: The evaluation contract

**Files:**
- Create: `lib/poorin/src/poorin/evaluation/contract.py`
- Modify: `docs/poorin/architecture.md` (row)
- Test: `tests/evaluation/test_contract.py`

**Interfaces:**
- Consumes: `decode_recorded` (Task 2).
- Produces:
  - `FinishedRun(NamedTuple)`: `label: str`, `path: str`, `record: ArtifactRecord`.
  - `EvalResult(Artifact)`: `run: str`, `step: int`, `suite: str`, `metrics: dict[str, float]`, `examples: int`, `samples_path: str | None = None`.
  - `EvalRuntime` (frozen dataclass): `batch_size: int | None`.
  - `Suite(Protocol)`: `name: ClassVar[str]`; `evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult`.
  - `EvaluableCheckpoint(Protocol)`, runtime-checkable: classmethod `suites(cls) -> Mapping[str, type[Suite]]`.
  - `finished_run_at(path: str) -> FinishedRun`.
  - `artifact_class(result_type: str) -> type[Artifact]`.
  - `offered_suites(label: str, artifact_type: type[Artifact]) -> Mapping[str, type[Suite]]`, which raises `ValueError` naming `runs migrate`.
  - `checkpoint_path(run: FinishedRun, step: int) -> str`.
  - `checkpoint_steps(run_path: str) -> list[int]`.
  - `recorded_value(run: FinishedRun, path: str) -> JSONValue`.
  - `recorded(run: FinishedRun, path: str, cls: type[T]) -> T`, which raises `ValueError` naming `runs migrate ... --drop`.
  - `MIGRATE_COMMAND` (a format string with `{label}` and `{checkpoint_type}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_contract.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaluation contract (poorin.evaluation.contract)."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from poorin.evaluation.contract import (
    EvalResult,
    FinishedRun,
    checkpoint_steps,
    offered_suites,
    recorded,
)
from poorin.execution.artifact import Artifact, ArtifactRecord
from poorin.execution.lazy import apply, run


@dataclass(frozen=True)
class Clip:
    history: int = 2


@dataclass(frozen=True)
class Score:
    name: ClassVar[str] = "score"

    def evaluate(self, run, step, output_path, runtime):
        raise AssertionError("not called")


class ScoredCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {Score.name: Score}


def _score() -> EvalResult:
    return EvalResult(run="toy/run@2026.10.13", step=2, suite="score", metrics={"eval/loss": 0.5}, examples=4)


def test_an_eval_result_round_trips_through_its_record(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    (loaded,) = run(apply("eval/toy", _score, version="2026.10.13", artifact_type=EvalResult))
    assert (loaded.run, loaded.step, loaded.suite) == ("toy/run@2026.10.13", 2, "score")
    assert loaded.metrics == {"eval/loss": 0.5}
    assert (loaded.examples, loaded.samples_path) == (4, None)


def test_a_record_field_the_code_dropped_names_the_migration():
    record = ArtifactRecord(
        name="toy/run",
        version="2026.10.13",
        result_type="toy.ScoredCheckpoint",
        config={"data": {"__class__": "Clip", "history": 2, "num_workers": 16}},
    )
    finished = FinishedRun("toy/run@2026.10.13", "/runs/toy", record)
    with pytest.raises(ValueError, match=r"runs migrate toy/run@2026\.10\.13 .*--drop data\.FIELD"):
        recorded(finished, "data", Clip)


def test_a_checkpoint_type_without_suites_is_refused_naming_the_migration():
    assert offered_suites("toy/run@2026.10.13", ScoredCheckpoint) == {"score": Score}
    with pytest.raises(ValueError, match="runs migrate"):
        offered_suites("toy/run@2026.10.13", Artifact)


def test_checkpoint_steps_lists_the_permanent_checkpoints_in_step_order(tmp_path):
    for name in ("step-1000", "step-500", "step-2000", "tmp"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
    assert checkpoint_steps(str(tmp_path)) == [500, 1000, 2000]
    assert checkpoint_steps(str(tmp_path / "nothing")) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/evaluation/test_contract.py -n 0 -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'poorin.evaluation.contract'`.

- [ ] **Step 3: Write the module**

```python
# lib/poorin/src/poorin/evaluation/contract.py
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
```

- [ ] **Step 4: Add the architecture row**

```markdown
| `poorin.evaluation.contract` | The evaluation contract: suites, evaluable checkpoint types, finished runs, the `EvalResult` artifact (D114) | `poorin.experiment.evaluate` |
```

- [ ] **Step 5: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/evaluation/test_contract.py -n 0 -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add lib/poorin/src/poorin/evaluation/contract.py docs/poorin/architecture.md tests/evaluation/test_contract.py
git commit -m "[poorin] Define the evaluation contract and its result artifact"
```

### Task 4: Short run references

**Files:**
- Create: `lib/poorin/src/poorin/experiment/run_refs.py` (with `Run`, `RunState`, and `collect_runs`, moved from `runs.py`)
- Modify: `lib/poorin/src/poorin/experiment/runs.py` (import the moved names; `show` resolves `NAME@REF`)
- Modify: `lib/poorin/src/poorin/experiment/cli.py` (`_resume_callback` resolves a token)
- Modify: `tools/prune_check.py:48` (import `collect_runs` from `run_refs`)
- Modify: `docs/poorin/architecture.md` (row)
- Test: `tests/experiment/test_run_refs.py`, `tests/experiment/test_runs.py`, `tests/experiment/test_cli.py`

**Interfaces:**
- Produces:
  - `RunRefError(ValueError)`.
  - `resolve_run(prefix: str, ref: str, *, finished_only: bool) -> Run`.
  - `resume_version(prefix: str, value: str) -> str`.
  - `stamped(version: str) -> StampedVersion | None`, where `StampedVersion` has `label: str | None`, `stamp: str`, `token: str`.
  - `Run`, `RunState`, and `collect_runs(prefix) -> list[Run]`, unchanged but now in `run_refs`.
- Behavior:
  - An exact version match wins.
  - A 4-hex `REF` is a token.
  - `latest` is the latest stamped dev version.
  - Anything else is a label, resolved to the latest `label-<stamp>-<token>-dev`.
  - "Latest" orders by the `YYYYMMDD-HHMMSS` stamp in the version.
  - `NAME` matches the run's name with or without `users/<you>/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/experiment/test_run_refs.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short references to runs (poorin.experiment.run_refs)."""

from pathlib import Path

import pytest
from poorin.execution.artifact import ArtifactRecord, write_record
from poorin.execution.step_status import STATUS_RUNNING, STATUS_SUCCESS, StatusFile
from poorin.experiment.run_refs import RunRefError, resolve_run, resume_version
from rigging.provenance import username_segment

ACCEPT_OLD = "accept-20261011-210614-7f16-dev"
ACCEPT_NEW = "accept-20261012-201822-60b9-dev"
PROBE = "probe-20261012-090000-a1b2-dev"


@pytest.fixture
def prefix(tmp_path) -> Path:
    return tmp_path / "prefix"


def _run(prefix: Path, name: str, version: str, status: str = STATUS_SUCCESS) -> None:
    path = prefix / name / version
    path.mkdir(parents=True)
    write_record(ArtifactRecord(name=name, version=version, output_path=str(path)))
    StatusFile(str(path), worker_id="test").write_status(status)


def _dev(name: str) -> str:
    return f"users/{username_segment()}/{name}"


def _version(prefix: Path, ref: str, *, finished_only: bool = True) -> str:
    return resolve_run(str(prefix), ref, finished_only=finished_only).label.rpartition("@")[2]


def test_a_full_version_resolves_to_itself(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    _run(prefix, "episode/wam", "2026.10.13")
    assert _version(prefix, f"episode/wam@{ACCEPT_NEW}") == ACCEPT_NEW
    assert _version(prefix, "episode/wam@2026.10.13") == "2026.10.13"


def test_a_token_resolves_to_the_run_it_suffixes(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    assert _version(prefix, "episode/wam@60b9") == ACCEPT_NEW


def test_a_token_two_runs_share_is_refused_listing_both(prefix):
    twin = "probe-20261013-080000-60b9-dev"
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    _run(prefix, _dev("episode/wam"), twin)
    with pytest.raises(RunRefError) as refused:
        resolve_run(str(prefix), "episode/wam@60b9", finished_only=True)
    assert ACCEPT_NEW in str(refused.value) and twin in str(refused.value)


def test_a_label_resolves_to_its_latest_run_by_version_time(prefix):
    # Written newest first, so file times disagree with the versions' timestamps.
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), PROBE)
    assert _version(prefix, "episode/wam@accept") == ACCEPT_NEW


def test_latest_resolves_to_the_latest_dev_run(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), PROBE)
    _run(prefix, "episode/wam", "2026.10.13")
    assert _version(prefix, "episode/wam@latest") == PROBE


def test_an_unfinished_run_is_skipped_only_when_finished_runs_are_asked_for(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW, status=STATUS_RUNNING)
    assert _version(prefix, "episode/wam@accept") == ACCEPT_OLD
    assert _version(prefix, "episode/wam@accept", finished_only=False) == ACCEPT_NEW


def test_an_unknown_reference_lists_the_versions_there_are(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    with pytest.raises(RunRefError, match=ACCEPT_NEW):
        resolve_run(str(prefix), "episode/wam@ffff", finished_only=True)


def test_resume_takes_a_token_or_a_full_version(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW, status=STATUS_RUNNING)
    _run(prefix, _dev("episode/vla"), ACCEPT_NEW, status=STATUS_RUNNING)  # one sweep, one version
    assert resume_version(str(prefix), "60b9") == ACCEPT_NEW
    assert resume_version(str(prefix), PROBE) == PROBE
    with pytest.raises(RunRefError, match="no dev run"):
        resume_version(str(prefix), "ffff")
```

Add to `tests/experiment/test_runs.py`, which already has the `checkout` and `prefix` fixtures and the `_build` helper:

```python
def test_show_takes_a_short_reference(checkout, prefix):
    _build("data/thing", Provenance.capture(), version="probe-20261012-090000-a1b2-dev")
    result = CliRunner().invoke(main, ["show", "data/thing@a1b2"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("data/thing@probe-20261012-090000-a1b2-dev")
```

Change `test_show_rejects_an_unknown_run` to assert `"no run data/nothing@2026.07.16"` in the output. Add to `tests/experiment/test_cli.py`, next to `test_resume_continues_the_version_an_earlier_run_printed`, using that test's builders:

```python
def test_resume_takes_the_token_of_an_earlier_run(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    started = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--run"])
    (version,) = re.findall(r"--resume (\S+)\)", started.output)
    token = version.removesuffix("-dev").rpartition("-")[2]
    resumed = runner.invoke(experiment_main(_build_two), ["--resume", token])
    assert resumed.exit_code == 0, resumed.output
    assert _resolved_model_version(resumed.output) == version
```

Read `test_resume_continues_the_version_an_earlier_run_printed` first and start the earlier run the same way it does; if it starts the run without `--run`, do the same here.

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_run_refs.py tests/experiment/test_runs.py tests/experiment/test_cli.py -n 0 -q`
Expected: FAIL. `test_run_refs.py` fails at collection with `ModuleNotFoundError`. The new `show` and `--resume` tests fail: `show` reads `data/thing@a1b2` as a full version, and `--resume a1b2` is rejected as a malformed version.

- [ ] **Step 3: Move run collection into `run_refs.py` and add resolution**

Move `SCRATCH_DIRS`, `RunState`, `Run`, `_state`, `_naive_local`, `_output_dirs`, and `collect_runs` verbatim from `runs.py` to `run_refs.py`, with the imports they use. Then add:

```python
# lib/poorin/src/poorin/experiment/run_refs.py (module docstring and the additions)
"""The runs under the prefix, and short references to one of them (D114).

``NAME@REF`` names a run wherever one is typed (``evaluate``, ``runs show``):

- a full version names itself;
- a 4-hex token (``60b9``) names the dev run whose version it suffixes;
- ``latest`` names the newest dev run of ``NAME``;
- anything else is a label: ``accept`` names the newest ``accept-<stamp>-<token>-dev`` run.

"Newest" orders by the timestamp a dev version carries (:func:`poorin.experiment.cli.run_version`).
``NAME`` matches a run's name with or without the user's ``users/<you>/`` namespace. References
resolve once, at the command line; names, fingerprints, and records hold only full versions.
"""

# A suffixed dev version: [label-]YYYYMMDD-HHMMSS-<token>-dev (poorin.experiment.cli.run_version).
STAMPED_VERSION = re.compile(r"^(?:(?P<label>.+)-)?(?P<stamp>\d{8}-\d{6})-(?P<token>[0-9a-f]{4})-dev$")
TOKEN = re.compile(r"^[0-9a-f]{4}$")
LATEST = "latest"
# How many of a name's versions an unknown reference lists.
LISTED_VERSIONS = 10


class RunRefError(ValueError):
    """A run reference that names no run, or several."""


@dataclass(frozen=True)
class StampedVersion:
    label: str | None
    stamp: str
    token: str


def stamped(version: str) -> StampedVersion | None:
    """The parts of a suffixed dev version, or ``None`` for any other version."""
    match = STAMPED_VERSION.match(version)
    return None if match is None else StampedVersion(match["label"], match["stamp"], match["token"])


def _split(label: str) -> tuple[str, str]:
    name, _, version = label.rpartition("@")
    return name, version


def _newest(runs: list[tuple[Run, StampedVersion]]) -> list[Run]:
    if not runs:
        return []
    return [max(runs, key=lambda pair: pair[1].stamp)[0]]


def resolve_run(prefix: str, ref: str, *, finished_only: bool) -> Run:
    """The run ``ref`` (``NAME@REF``) names under ``prefix``; only built runs when ``finished_only``."""
    name, sep, wanted = ref.rpartition("@")
    if not sep or not name or not wanted:
        raise RunRefError(f"{ref!r} is not NAME@REF")
    names = {name, f"users/{username_segment()}/{name}"}
    runs = [r for r in collect_runs(prefix) if _split(r.label)[0] in names]
    if finished_only:
        runs = [r for r in runs if r.state is RunState.BUILT]
    exact = [r for r in runs if _split(r.label)[1] == wanted]
    if exact:
        return exact[0]
    dev = [(r, s) for r in runs if (s := stamped(_split(r.label)[1])) is not None]
    if TOKEN.match(wanted):
        matches = [r for r, s in dev if s.token == wanted]
    elif wanted == LATEST:
        matches = _newest(dev)
    else:
        matches = _newest([(r, s) for r, s in dev if s.label == wanted])
    kind = "finished run" if finished_only else "run"
    if not matches:
        versions = sorted((_split(r.label)[1] for r in runs), reverse=True)[:LISTED_VERSIONS]
        raise RunRefError(f"no {kind} {ref}; {name} has the {kind}s {versions}")
    if len(matches) > 1:
        raise RunRefError(f"{ref} names several {kind}s, so pass the full version: {sorted(r.label for r in matches)}")
    return matches[0]


def resume_version(prefix: str, value: str) -> str:
    """The version ``--resume VALUE`` continues: ``VALUE`` itself, or the dev version whose token it is.

    ``--resume`` takes no name, so a token is matched against every dev run under ``prefix``; a
    sweep's trials share one version, so several runs may carry it.
    """
    if not TOKEN.match(value):
        return value
    versions = sorted(
        {version for r in collect_runs(prefix) if (s := stamped(version := _split(r.label)[1])) and s.token == value}
    )
    if not versions:
        raise RunRefError(f"no dev run under {prefix} has the token {value}")
    if len(versions) > 1:
        raise RunRefError(f"the token {value} suffixes several versions, so pass one: {versions}")
    return versions[0]
```

Add `import re`, `from dataclasses import dataclass`, and `from rigging.provenance import username_segment`. If the walrus in `resume_version` reads poorly, write it as a loop.

- [ ] **Step 4: Use the moved names and resolution in `runs.py`**

Import `Run`, `RunRefError`, `RunState`, `collect_runs`, `resolve_run`, and `_state` from `poorin.experiment.run_refs`. Rename `_state` to the public `run_state`, since two modules now use it, and update its callers. Rewrite the `NAME@...` branch of `show`:

```python
    if "@" in target:
        try:
            found = resolve_run(prefix or poorin_prefix(), target, finished_only=False)
        except RunRefError as e:
            raise click.ClickException(str(e)) from e
        click.echo("\n".join(describe(found)))
        return
```

Keep the path branch as it is. Update the `show` docstring: "TARGET is NAME@REF (a version, a token, a label, or latest) or the output's path."

- [ ] **Step 5: Resolve a token in `--resume`**

In `cli.py`:

```python
def _resume_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Accept a dev version, or the token of one (``60b9``): calendar versions need no resume."""
    if value is None:
        return None
    try:
        value = resume_version(poorin_prefix(), value)
    except RunRefError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e
    value = _version_callback(ctx, param, value)
    ...  # the existing calendar-version refusal, unchanged
```

Change the option's help to: "Continue an earlier dev run by its version, as printed when it started, or the 4-hex token in it."

- [ ] **Step 6: Update `tools/prune_check.py` and add the architecture row**

Change `from poorin.experiment.runs import collect_runs` to `from poorin.experiment.run_refs import collect_runs`. Add the row:

```markdown
| `poorin.experiment.run_refs` | The runs under the prefix, and short references (`NAME@60b9`, `@accept`, `@latest`) to one (D114) | `poorin.experiment.runs` |
```

- [ ] **Step 7: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_run_refs.py tests/experiment/test_runs.py tests/experiment/test_cli.py -n 0 -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add lib/poorin/src/poorin/experiment/run_refs.py lib/poorin/src/poorin/experiment/runs.py lib/poorin/src/poorin/experiment/cli.py tools/prune_check.py docs/poorin/architecture.md tests/experiment/test_run_refs.py tests/experiment/test_runs.py tests/experiment/test_cli.py
git commit -m "[poorin] Name runs by short references: a token, a label, latest"
```

### Task 5: A calendar step over a pinned dev run

**Files:**
- Modify: `lib/poorin/src/poorin/execution/lazy.py` (`ArtifactStep.__post_init__`)
- Modify: `lib/poorin/src/poorin/execution/step_spec.py` (`StepSpec.rebuilds`)
- Test: `tests/execution/test_lazy.py`

**Interfaces:**
- Produces two rules:
  - A fixed-version handle may depend on a pinned mutable handle.
  - A pinned step never rebuilds, so a built pin is served from its status.

- [ ] **Step 1: Write the failing tests**

Append to `tests/execution/test_lazy.py`, and add `OUT` and `read_status`/`STATUS_SUCCESS` to its imports if they are missing:

```python
DEV_RUN = "accept-20261012-201822-60b9-dev"


def _train(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "weights"), "w") as f:
        f.write("trained")


def _never_built() -> None:
    raise AssertionError("a pinned run is never built")


def _score(run_dir: str) -> Artifact:
    with open(os.path.join(run_dir, "weights")) as f:
        assert f.read() == "trained"
    return Artifact()


def test_a_calendar_step_may_depend_on_a_pinned_dev_run(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    trained = apply("toy/run", _train, version=DEV_RUN, out=OUT)
    run(trained)
    pinned = apply("toy/run", _never_built, version=DEV_RUN, pin=trained.path())
    run(apply("eval/toy", _score, version="2026.10.13", run_dir=pinned))


def test_a_calendar_step_refuses_an_unpinned_dev_dependency():
    trained = apply("toy/run", _train, version=DEV_RUN, out=OUT)
    with pytest.raises(ValueError, match="mutable"):
        apply("eval/toy", _score, version="2026.10.13", run_dir=trained)


def test_evaluating_a_pinned_dev_run_leaves_it_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    trained = apply("toy/run", _train, version=DEV_RUN, out=OUT)
    run(trained)
    status_file = os.path.join(trained.path(), ".executor_status")
    before = (read_record(trained.path()), open(status_file).read(), os.stat(status_file).st_mtime_ns)
    pinned = apply("toy/run", _never_built, version=DEV_RUN, pin=trained.path())
    run(apply("eval/toy", _score, version="2026.10.13", run_dir=pinned))
    assert (read_record(trained.path()), open(status_file).read(), os.stat(status_file).st_mtime_ns) == before
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_lazy.py -n 0 -q -k "pinned or unpinned"`
Expected: `test_a_calendar_step_may_depend_on_a_pinned_dev_run` and `test_evaluating_a_pinned_dev_run_leaves_it_untouched` FAIL with `ValueError: eval/toy@2026.10.13 is a fixed version but depends on mutable toy/run@...`. The unpinned test passes, since it pins the existing rule.

- [ ] **Step 3: Skip pinned deps in the mutable-dep rule**

In `ArtifactStep.__post_init__`:

```python
        if not is_mutable_version(self.version):
            for dep in self.deps:
                # A pin references existing data and never rebuilds, so a fixed parent may cite one.
                if is_mutable_version(dep.version) and dep.override_path is None:
                    raise ValueError(...)  # message unchanged
```

- [ ] **Step 4: Run the tests and watch the pinned-untouched test**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_lazy.py -n 0 -q -k "pinned or unpinned"`
Expected: the first test passes. `test_evaluating_a_pinned_dev_run_leaves_it_untouched` fails: either `AssertionError: a pinned run is never built` or a changed status file, because `StepSpec.rebuilds` is true for the pinned dev step.

- [ ] **Step 5: A pinned step never rebuilds**

In `step_spec.py`:

```python
    @property
    def rebuilds(self) -> bool:
        """A ``dev`` version always rebuilds, over a finished output too; a pin never does."""
        return is_mutable_version(self.version) and not self.pinned
```

- [ ] **Step 6: Run the execution tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution -n 0 -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/poorin/src/poorin/execution/lazy.py lib/poorin/src/poorin/execution/step_spec.py tests/execution/test_lazy.py
git commit -m "[poorin] Let a calendar step cite a pinned dev run, never rebuilt"
```

### Task 6: Evaluation handles and the accelerator lock

**Files:**
- Create: `lib/poorin/src/poorin/experiment/evaluate.py` (handles; the command comes in Task 7)
- Modify: `lib/poorin/src/poorin/training/training.py` (rename `exclusive_training` to `exclusive_accelerator`, `_TRAINING_LOCK` to `_ACCELERATOR_LOCK`)
- Modify: `experiments/grug/dispatch.py:13,38` (the new name)
- Modify: `docs/explanations/lazy-artifacts.md:274` (the new name, and that evaluations hold it too)
- Modify: `docs/poorin/architecture.md` (row)
- Test: `tests/experiment/test_evaluate.py`

**Interfaces:**
- Consumes: Task 3 (`EvalResult`, `EvalRuntime`, `Suite`, `artifact_class`, `finished_run_at`, `offered_suites`), Task 4 (`resolve_run`, `RunRefError`, `collect_runs`), Task 5 (pinned dev deps).
- Produces:
  - `Checkpoint(run: ArtifactStep, step: int)`.
  - `finished_run(ref: str) -> ArtifactStep`: pinned; `artifact_type` is the record's.
  - `evaluate(checkpoint: Checkpoint, suite: Suite, *, version: str | None = None, overrides: Overrides = Overrides(), batch_size: int | None = None) -> ArtifactStep[EvalResult]`.
  - `eval_results(prefix: str, *, run: str | None = None, suite: str | None = None) -> list[EvalResult]`.
  - `EvalRecipe(suite)`, which is what `--set` reaches.
  - `exclusive_accelerator(label: str)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/experiment/test_evaluate.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation handles over finished runs (poorin.experiment.evaluate)."""

import os
from dataclasses import dataclass
from typing import ClassVar

import pytest
from poorin.evaluation.contract import EvalResult, EvalRuntime, FinishedRun
from poorin.execution.artifact import Artifact
from poorin.execution.lazy import OUT, apply, run
from poorin.execution.step_status import STATUS_RUNNING, StatusFile
from poorin.experiment.evaluate import Checkpoint, eval_results, evaluate, finished_run
from poorin.experiment.run_refs import RunRefError
from rigging.provenance import username_segment

DEV = "accept-20261012-201822-60b9-dev"
OTHER_DEV = "accept-20261013-080000-c3d4-dev"


@dataclass(frozen=True)
class Score:
    """A toy suite: the loss is the step times ``scale``."""

    name: ClassVar[str] = "score"
    scale: float = 1.0

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        return EvalResult(run=run.label, step=step, suite=self.name, metrics={"eval/loss": self.scale * step}, examples=1)


@dataclass(frozen=True)
class Unoffered:
    name: ClassVar[str] = "unoffered"

    def evaluate(self, run, step, output_path, runtime):
        raise AssertionError("not offered")


class ToyCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {Score.name: Score}


def _train(out: str) -> None:
    for step in (1, 2):
        os.makedirs(os.path.join(out, "checkpoints", f"step-{step}"))


def train_toy(version: str = DEV) -> str:
    """A finished dev run of the toy template in the user's namespace; its full label."""
    handle = apply(f"users/{username_segment()}/toy/run", _train, version=version, artifact_type=ToyCheckpoint, out=OUT)
    run(handle)
    return handle.label


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    return tmp_path


def test_an_evaluation_is_named_after_the_run_suite_and_step(prefix):
    train_toy()
    cell = evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Score(), version="2026.10.13")
    assert cell.label == f"eval/toy/run/{DEV}/score/step-2@2026.10.13"


def test_a_dev_evaluation_is_namespaced_and_two_run_versions_get_two_names(prefix):
    train_toy()
    train_toy(OTHER_DEV)
    version = "20261013-090000-e5f6-dev"
    first = evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Score(), version=version)
    second = evaluate(Checkpoint(finished_run("toy/run@c3d4"), 2), Score(), version=version)
    assert first.name == f"users/{username_segment()}/eval/toy/run/{DEV}/score/step-2"
    assert first.name != second.name


def test_the_suite_and_step_bear_identity_and_the_batch_size_does_not(prefix):
    train_toy()
    pinned = finished_run("toy/run@60b9")

    def fingerprint(step: int, suite: Score, batch_size: int | None = None) -> str:
        return evaluate(Checkpoint(pinned, step), suite, version="2026.10.13", batch_size=batch_size).fingerprint()

    base = fingerprint(2, Score())
    assert fingerprint(2, Score(scale=2.0)) != base
    assert fingerprint(1, Score()) != base
    assert fingerprint(2, Score(), batch_size=64) == base


def test_an_evaluation_builds_its_result_and_eval_results_finds_it(prefix):
    label = train_toy()
    (result,) = run(evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Score(scale=0.5), version="2026.10.13"))
    assert (result.run, result.step, result.metrics) == (label, 2, {"eval/loss": 1.0})
    assert [r.metrics for r in eval_results(str(prefix), run=label)] == [{"eval/loss": 1.0}]
    assert eval_results(str(prefix), suite="other") == []


def test_an_unfinished_run_is_not_a_finished_run(prefix):
    label = train_toy()
    StatusFile(str(prefix / label.replace("@", "/")), worker_id="test").write_status(STATUS_RUNNING)
    with pytest.raises(RunRefError, match="no finished run"):
        finished_run("toy/run@60b9")


def test_a_suite_the_checkpoint_type_does_not_offer_is_refused(prefix):
    train_toy()
    with pytest.raises(ValueError, match=r"offers the suites \['score'\]"):
        evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Unoffered(), version="2026.10.13")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_evaluate.py -n 0 -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'poorin.experiment.evaluate'`.

- [ ] **Step 3: Rename the lock**

In `training.py`:

```python
# Held by the one training or evaluation using the accelerator in this process; see ``exclusive_accelerator``.
_ACCELERATOR_LOCK = threading.Lock()


@contextlib.contextmanager
def exclusive_accelerator(label: str) -> Iterator[None]:
    """Use the accelerator only while no other training or evaluation does in this process; wait for one that does.

    (keep the existing paragraph on Fray's LocalClient threads, the tracker, and tracker_metrics.jsonl,
    then add:) Two evaluations, or an evaluation and a training, would also share one GPU's memory.
    Memory held by other programs is outside the lock and fails loudly (D113).
    """
    if _ACCELERATOR_LOCK.locked():
        logger.info("%s waits for the accelerator, which another step in this process is using", label)
    with _ACCELERATOR_LOCK:
        yield
```

Update its callers in `training.py` (lines 531 and 539) and `experiments/grug/dispatch.py`. Run `rg "exclusive_training|_TRAINING_LOCK"` afterwards and expect nothing outside `docs/poorin/` (D98's text is history and stays).

- [ ] **Step 4: Write the handles**

```python
# lib/poorin/src/poorin/experiment/evaluate.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluations as artifacts (D114): handles over finished runs, and the command that runs a matrix.

    uv run --no-sync python -m poorin.experiment.evaluate RUN@REF [RUN@REF ...] \\
        --suite NAME --steps 500,1000|all --version VERSION [--set suite.FIELD=VALUE] [--run]

An evaluation is an artifact step whose input is a finished run, pinned so it is never rebuilt, and
one of its permanent checkpoint steps; its recipe is a suite (:mod:`poorin.evaluation.contract`).
Its name is ``eval/{run name}/{run version}/{suite}/step-{N}``, with the run's ``users/<you>/``
namespace stripped, and a dev evaluation is namespaced like a dev run. Its fingerprint is the run's
``name@version``, the step, and the suite's class and fields; the device batch size is a runtime
arg. Every evaluation step holds the accelerator lock.
"""

import re
from dataclasses import dataclass
from typing import Any, no_type_check

from rigging.filesystem.cluster_config import poorin_prefix

from poorin.evaluation.contract import (
    EvalResult,
    EvalRuntime,
    Suite,
    artifact_class,
    finished_run_at,
    offered_suites,
)
from poorin.execution.artifact import result_type_name
from poorin.execution.build_context import resolve_version
from poorin.execution.lazy import ArtifactStep, StepContext
from poorin.experiment.namespacing import user_namespaced_name
from poorin.experiment.overrides import Overrides
from poorin.experiment.run_refs import RunRefError, collect_runs, resolve_run
from poorin.training.training import exclusive_accelerator

EVAL_PREFIX = "eval"
USER_NAMESPACE = re.compile(r"^users/[^/]+/")


@dataclass(frozen=True)
class Checkpoint:
    """One permanent checkpoint of a run: the run's handle and the step."""

    run: ArtifactStep
    step: int


@dataclass(frozen=True)
class EvalRecipe:
    """What ``--set`` reaches: ``suite.FIELD``. The run and the step are not overridable."""

    suite: Any


# The batch size holds a placeholder string while the fingerprint is computed, which the tests'
# beartype hook would reject.
@no_type_check
@dataclass(frozen=True)
class EvaluationConfig:
    run: str
    """The run's output directory."""
    step: int
    suite: Any
    output_path: str
    batch_size: int | None


def _never_built(config: Any) -> None:
    raise RuntimeError("a finished run is pinned and never built")


def finished_run(ref: str) -> ArtifactStep:
    """A pinned handle to the finished run ``ref`` (``NAME@REF``) names, typed by its record."""
    found = resolve_run(poorin_prefix(), ref, finished_only=True)
    record = found.record
    if record is None:
        raise RunRefError(f"{found.label} is built but has no record, so its checkpoint type is unknown")
    return ArtifactStep(
        name=record.name,
        version=record.version,
        artifact_type=artifact_class(record.result_type),
        run=_never_built,
        build_config=lambda ctx: record.config or {},
        override_path=found.path,
    )


def run_evaluation(config: EvaluationConfig) -> EvalResult:
    """Score one checkpoint with one suite, holding the accelerator."""
    finished = finished_run_at(config.run)
    with exclusive_accelerator(f"Evaluation of {finished.label} at step {config.step}"):
        return config.suite.evaluate(finished, config.step, config.output_path, EvalRuntime(config.batch_size))


def evaluate(
    checkpoint: Checkpoint,
    suite: Suite,
    *,
    version: str | None = None,
    overrides: Overrides = Overrides(),
    batch_size: int | None = None,
) -> ArtifactStep[EvalResult]:
    """The evaluation of ``checkpoint`` by ``suite``.

    ``version`` defaults to the ambient build context's. ``overrides`` reach the suite's fields
    (``suite.clips``) and join the suite's segment of the name. ``batch_size`` is the device batch
    size for a suite whose result does not depend on it; it stays out of the fingerprint.
    """
    run = checkpoint.run
    suites = offered_suites(run.label, run.artifact_type)
    if not isinstance(suite, suites.get(suite.name, ())):
        raise ValueError(f"{run.label}'s checkpoint type offers the suites {sorted(suites)}, not {suite.name!r}")
    base = f"{EVAL_PREFIX}/{USER_NAMESPACE.sub('', run.name)}/{run.version}"
    step = f"step-{checkpoint.step}"
    version = resolve_version(f"{base}/{suite.name}/{step}", version)

    def build_config(ctx: StepContext) -> EvaluationConfig:
        recipe = overrides.apply(ctx, EvalRecipe(suite=suite))
        return EvaluationConfig(
            run=ctx.artifact_path(run),
            step=checkpoint.step,
            suite=recipe.suite,
            output_path=ctx.output_path,
            batch_size=ctx.runtime_arg("batch_size"),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{base}/{overrides.named(suite.name)}/{step}", version),
        version=version,
        artifact_type=EvalResult,
        run=run_evaluation,
        build_config=build_config,
        deps=(run,),
        runtime_args={"batch_size": batch_size},
    )


def eval_results(prefix: str, *, run: str | None = None, suite: str | None = None) -> list[EvalResult]:
    """Every evaluation recorded under ``prefix``, for one run (its full ``name@version``) or one suite."""
    wanted = result_type_name(EvalResult)
    results = [
        EvalResult.raw_load(r.path)
        for r in collect_runs(prefix)
        if r.record is not None and r.record.result_type == wanted
    ]
    return [r for r in results if (run is None or r.run == run) and (suite is None or r.suite == suite)]
```

The `isinstance(suite, suites.get(suite.name, ()))` check refuses both an unknown name and a suite of another class with the same name. If `ty` rejects the empty-tuple default, write the two conditions out. If `overrides: Overrides = Overrides()` trips the linter's mutable-default rule, pass `Overrides()` at the call sites and drop the default; record that ruling.

- [ ] **Step 5: Add the architecture row**

```markdown
| `poorin.experiment.evaluate` | Evaluations as artifacts: handles over pinned finished runs, `eval_results`, and the run-by-step command (D114) | `poorin.experiment.evaluate` |
```

Add `"poorin.experiment.evaluate"` to `[kept] modules` in `tools/roots.toml`, under the `# Workbench: evaluation` comment.

- [ ] **Step 6: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_evaluate.py tests/test_grug_dispatch.py -n 0 -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/poorin/src/poorin/experiment/evaluate.py lib/poorin/src/poorin/training/training.py experiments/grug/dispatch.py docs/explanations/lazy-artifacts.md docs/poorin/architecture.md tools/roots.toml tests/experiment/test_evaluate.py
git commit -m "[poorin] Build evaluations as artifacts over pinned finished runs"
```

### Task 7: The evaluate command and run migration

**Files:**
- Modify: `lib/poorin/src/poorin/experiment/evaluate.py` (the command)
- Modify: `lib/poorin/src/poorin/experiment/runs.py` (`migrate`)
- Test: `tests/experiment/test_evaluate.py`

**Interfaces:**
- Consumes: Task 6 (`finished_run`, `evaluate`, `Checkpoint`), Task 3 (`checkpoint_steps`, `offered_suites`, `artifact_class`), `build_options` from `cli.py`.
- Produces:
  - `evaluation_matrix(refs, *, suite_name: str, steps: str, batch_size: int | None, overrides: Overrides) -> list[ArtifactStep[EvalResult]]`.
  - The Click command `main`.
  - `runs migrate TARGET --to MODULE.CLASS [--drop PATH ...]`.
  - `migrated(record: ArtifactRecord, checkpoint_type: str, dropped: Sequence[str]) -> ArtifactRecord`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiment/test_evaluate.py`, and add `CliRunner`, `main`, the runs `main`, and `LevanterCheckpoint` to the imports:

```python
from click.testing import CliRunner
from poorin.execution.artifact import read_record
from poorin.experiment.evaluate import main as evaluate_main
from poorin.experiment.runs import main as runs_main
from poorin.training.training import LevanterCheckpoint

CALENDAR = ["--version", "2026.10.13"]


def _invoke(*args: str):
    return CliRunner().invoke(evaluate_main, list(args), catch_exceptions=False)


def test_the_plan_has_one_evaluation_per_permanent_checkpoint(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "all", *CALENDAR)
    assert result.exit_code == 0, result.output
    for step in (1, 2):
        assert f"eval/toy/run/{DEV}/score/step-{step}@2026.10.13" in result.output


def test_a_set_reaches_the_suite_and_names_the_evaluation(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "2", "--set", "suite.scale=3", *CALENDAR)
    assert result.exit_code == 0, result.output
    assert f"eval/toy/run/{DEV}/score/suite.scale=3/step-2@2026.10.13" in result.output


def test_a_set_outside_the_suite_is_refused(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "2", "--set", "step=1", *CALENDAR)
    assert result.exit_code != 0
    assert "has no field 'step'" in result.output


def test_a_step_without_a_checkpoint_is_refused_listing_the_steps(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "1,3", *CALENDAR)
    assert result.exit_code != 0
    assert "step [3]" in result.output and "[1, 2]" in result.output


def test_a_suite_the_run_does_not_offer_is_refused_listing_its_suites(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "lm-eval", "--steps", "2", *CALENDAR)
    assert result.exit_code != 0
    assert "['score']" in result.output


def test_an_unknown_or_unfinished_run_is_refused(prefix):
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "2", *CALENDAR)
    assert result.exit_code != 0
    assert "no finished run toy/run@60b9" in result.output


def test_a_run_trained_before_evaluation_artifacts_is_refused_naming_the_migration(prefix):
    old = apply(f"users/{username_segment()}/toy/old", _train, version=DEV, artifact_type=LevanterCheckpoint, out=OUT)
    run(old)
    result = _invoke("toy/old@60b9", "--suite", "score", "--steps", "2", *CALENDAR)
    assert result.exit_code != 0
    assert "runs migrate" in result.output


def test_migrate_names_the_checkpoint_type_and_drops_a_moved_field(prefix):
    old = apply(f"users/{username_segment()}/toy/old", _train, version=DEV, artifact_type=LevanterCheckpoint, out=OUT)
    run(old)
    toy_type = f"{ToyCheckpoint.__module__}.ToyCheckpoint"
    result = CliRunner().invoke(runs_main, ["migrate", "toy/old@60b9", "--to", toy_type, "--drop", "out"])
    assert result.exit_code == 0, result.output
    record = read_record(old.path())
    assert record.result_type == toy_type
    assert "out" not in record.config
    planned = _invoke("toy/old@60b9", "--suite", "score", "--steps", "2", *CALENDAR)
    assert planned.exit_code == 0, planned.output


def test_migrate_refuses_a_run_that_records_another_type(prefix):
    train_toy()
    result = CliRunner().invoke(runs_main, ["migrate", "toy/run@60b9", "--to", "poorin.evaluation.contract.EvalResult"])
    assert result.exit_code != 0
```

Add a build test that runs in a clean Git checkout, so the calendar version's code check passes. Use the `checkout` fixture's body from `tests/experiment/test_runs.py`, copied into this file as a fixture named `checkout`:

```python
def test_a_rerun_of_a_built_matrix_builds_nothing(checkout, prefix):
    train_toy()
    args = ("toy/run@60b9", "--suite", "score", "--steps", "all", *CALENDAR)
    built = _invoke(*args, "--run")
    assert built.exit_code == 0, built.output
    plan = _invoke(*args)
    lines = [line for line in plan.output.splitlines() if "eval/toy/run" in line]
    assert len(lines) == 2 and all(line.split()[0] == "built" for line in lines)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_evaluate.py -n 0 -q`
Expected: FAIL with `ImportError: cannot import name 'main' from 'poorin.experiment.evaluate'`.

- [ ] **Step 3: Write the command**

Append to `evaluate.py`, and add `click`, `read_record`, `build_options`, and `checkpoint_steps` to its imports:

```python
STEPS_ALL = "all"


def _steps(text: str, label: str, available: list[int]) -> list[int]:
    if text == STEPS_ALL:
        if not available:
            raise click.UsageError(f"{label} has no permanent checkpoints")
        return available
    try:
        wanted = [int(part) for part in text.split(",")]
    except ValueError as e:
        raise click.BadParameter(f"{text!r} is not comma-separated steps or 'all'", param_hint="--steps") from e
    missing = [step for step in wanted if step not in available]
    if missing:
        raise click.UsageError(
            f"{label} has no permanent checkpoint at step {missing}; its checkpoints are at steps {available}"
        )
    return wanted


def evaluation_matrix(
    refs: tuple[str, ...], *, suite_name: str, steps: str, batch_size: int | None, overrides: Overrides
) -> list[ArtifactStep[EvalResult]]:
    """One evaluation per run, step, and the trial's suite settings; refuses before anything runs."""
    cells = []
    for ref in refs:
        try:
            run = finished_run(ref)
            suites = offered_suites(run.label, run.artifact_type)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        if suite_name not in suites:
            raise click.UsageError(f"{run.label}'s checkpoint type offers the suites {sorted(suites)}, not {suite_name!r}")
        record = read_record(run.path())
        trained_at = record.provenance.label if record is not None and record.provenance is not None else "unrecorded code"
        click.echo(f"{run.label} trained at {trained_at}")
        for step in _steps(steps, run.label, checkpoint_steps(run.path())):
            cells.append(
                evaluate(Checkpoint(run, step), suites[suite_name](), overrides=overrides, batch_size=batch_size)
            )
    return cells


@click.command()
@click.argument("refs", nargs=-1, required=True, metavar="RUN@REF...")
@click.option("--suite", "suite_name", required=True, help="The suite, by the name the run's checkpoint type gives it.")
@click.option("--steps", required=True, help="Checkpoint steps, comma-separated, or 'all' for every permanent checkpoint.")
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Device batch size, for a suite whose result does not depend on it.",
)
@build_options
def main(
    refs: tuple[str, ...], suite_name: str, steps: str, batch_size: int | None, overrides: Overrides
) -> list[ArtifactStep[EvalResult]]:
    """Evaluate finished runs' checkpoints with one suite: print the plan, or build it with --run."""
    return evaluation_matrix(refs, suite_name=suite_name, steps=steps, batch_size=batch_size, overrides=overrides)


if __name__ == "__main__":
    main()
```

`RunRefError` is a `ValueError`, so the one `except` covers an unknown reference, an unfinished run, and a checkpoint type without suites.

- [ ] **Step 4: Write `runs migrate`**

In `runs.py`, add `import copy` and `from collections.abc import Sequence`, and import `artifact_class` and `offered_suites` from `poorin.evaluation.contract` and `write_record` from the artifact module:

```python
# What a training run recorded before its template named a checkpoint type (D114).
LEVANTER_CHECKPOINT = "poorin.training.training.LevanterCheckpoint"


def migrated(record: ArtifactRecord, checkpoint_type: str, dropped: Sequence[str]) -> ArtifactRecord:
    """``record`` naming ``checkpoint_type``, without the config fields at the dotted paths ``dropped``.

    The fingerprint and its payload stay as recorded: they describe what ran.
    """
    label = f"{record.name}@{record.version}"
    if record.result_type not in (LEVANTER_CHECKPOINT, checkpoint_type):
        raise ValueError(f"{label} records {record.result_type}; only a run recording {LEVANTER_CHECKPOINT} migrates")
    config = copy.deepcopy(record.config or {})
    for path in dropped:
        *parents, leaf = path.split(".")
        node = config
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"{label} records no config field {path!r}")
        del node[leaf]
    return record.model_copy(update={"result_type": checkpoint_type, "config": config})


@main.command()
@click.argument("target")
@click.option("--to", "checkpoint_type", required=True, metavar="MODULE.CLASS", help="The template's checkpoint type.")
@click.option("--drop", "dropped", multiple=True, metavar="PATH", help="A recorded config field the code moved out of the recipe. Repeatable.")
@click.option("--prefix", default=None, help="Read this prefix instead of POORIN_PREFIX.")
def migrate(target: str, checkpoint_type: str, dropped: tuple[str, ...], prefix: str | None) -> None:
    """Name the checkpoint type of a run trained before evaluation artifacts (D114), once."""
    try:
        found = resolve_run(prefix or poorin_prefix(), target, finished_only=True)
        offered_suites(found.label, artifact_class(checkpoint_type))
        if found.record is None:
            raise ValueError(f"{found.label} has no record")
        write_record(migrated(found.record, checkpoint_type, dropped))
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{found.label} now records {checkpoint_type}" + (f", without {', '.join(dropped)}" if dropped else ""))
```

`write_record` writes to `record.output_path`; a record written by the runner holds the output's path, which `migrated` keeps.

- [ ] **Step 5: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment -n 0 -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lib/poorin/src/poorin/experiment/evaluate.py lib/poorin/src/poorin/experiment/runs.py tests/experiment/test_evaluate.py
git commit -m "[poorin] Add the evaluate command and run migration"
```

### Task 8: The episode templates' held-out loss suite

**Files:**
- Modify: `experiments/episode_models/flow.py` (`eval_times(count)`)
- Modify: `experiments/episode_models/train.py` (`load_examples`, `held_out_loss`)
- Create: `experiments/episode_models/evaluation.py`
- Modify: `experiments/episode_models/launch.py` (`artifact_type=EpisodeCheckpoint`), and the family `launch.py` files' return annotations
- Modify: `experiments/episode_models/tests/test_flow.py:32` (`eval_times(8)`)
- Modify: `docs/poorin/architecture.md` (row)
- Test: `experiments/episode_models/tests/test_evaluation.py`

**Interfaces:**
- Consumes: Task 3 (`EvalResult`, `EvalRuntime`, `FinishedRun`, `Suite`, `checkpoint_path`, `recorded`, `recorded_value`).
- Produces:
  - `EpisodeCheckpoint(Artifact)`, whose `suites()` is `{"held-out": HeldOutLoss}`.
  - `HeldOutLoss(clips=128, batch_size=32, flow_times=8, seed=0)`.
  - `RecordedEpisodeRun(model, data, mp, dataset_path, action_stats_path)`.
  - `recorded_episode_run(run: FinishedRun) -> RecordedEpisodeRun`.
  - `held_out_metrics(run: RecordedEpisodeRun, checkpoint: str, suite: HeldOutLoss) -> dict[str, float]`.
  - In `train.py`: `EpisodeExamples(dims, train, held_out)`.
  - `load_examples(dataset_path: str, action_stats_path: str | None, data: EpisodeDataConfig) -> EpisodeExamples`.
  - `held_out_loss(mp: jmp.Policy, streams, weights, *, flow_times: int, seed: int)`, which returns the `evaluate(params, batches) -> dict[str, float]` closure.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/episode_models/tests/test_evaluation.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The episode templates' held-out loss suite reproduces the held-out loss training logged."""

import dataclasses
import json
from pathlib import Path

import pytest
from fray.types import ResourceConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from poorin.evaluation.contract import EvalRuntime, FinishedRun
from poorin.execution.artifact import ArtifactRecord, result_type_name
from poorin.execution.fingerprint import canonical_json

from experiments.episode_models.evaluation import (
    EpisodeCheckpoint,
    HeldOutLoss,
    RecordedEpisodeRun,
    held_out_metrics,
)
from experiments.episode_models.launch import EpisodeLaunchConfig, EpisodeRunConfig
from experiments.episode_models.tests.test_train import FAMILIES, json_records, tiny, tiny_run  # noqa: F401
from experiments.episode_models.train import EpisodeTrainingRun, run_episode_training

STEPS = 2
# The tiny runs evaluate one batch of two clips (tiny_run: eval_batches=1, eval_batch_size=2).
TINY_SUITE = HeldOutLoss(clips=2, batch_size=2)


def _train(run_dir: Path, family: str, tiny: tuple[str, str]) -> tuple[EpisodeTrainingRun, dict[str, float]]:
    """Train ``family`` for ``STEPS`` with its checkpoints at ``run_dir/checkpoints/step-N``; the run and
    the held-out loss it logged last."""
    with json_records() as (name, stream):
        run = tiny_run(run_dir, family, STEPS, tiny, name)
        checkpointer = dataclasses.replace(run.trainer.checkpointer, append_run_id_to_base_path=False)
        run = dataclasses.replace(run, trainer=dataclasses.replace(run.trainer, checkpointer=checkpointer))
        run_episode_training(run)
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    (*_, logged) = [r["metrics"] for r in records if r["event"] == "log" and "eval/loss" in r["metrics"]]
    return run, logged


def _recorded(run: EpisodeTrainingRun) -> RecordedEpisodeRun:
    return RecordedEpisodeRun(
        model=run.model,
        data=run.data,
        mp=run.train.mp,
        dataset_path=run.dataset_path,
        action_stats_path=run.action_stats_path,
    )


def _finished(run: EpisodeTrainingRun, output: Path, *, stale_data_field: bool = False) -> FinishedRun:
    launch = EpisodeLaunchConfig(
        model=run.model,
        data=run.data,
        train=run.train,
        run=EpisodeRunConfig(
            output_path=str(output),
            run_id="test",
            resources=ResourceConfig.with_cpu(),
            tracker=JsonLoggerConfig(),
            finelog_url=None,
            dataset_path=run.dataset_path,
            action_stats_path=run.action_stats_path,
            num_workers=0,
            log_every=1,
            checkpoint_every=STEPS,
        ),
    )
    config = json.loads(canonical_json(launch))
    if stale_data_field:
        config["data"]["num_workers"] = 16  # as D113's acceptance runs recorded it
    record = ArtifactRecord(
        name="episode/tiny",
        version="2026.10.13",
        result_type=result_type_name(EpisodeCheckpoint),
        output_path=str(output),
        config=config,
    )
    return FinishedRun("episode/tiny@2026.10.13", str(output), record)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_held_out_loss_of_a_checkpoint_reproduces_the_logged_one(tmp_path, tiny, family):
    run, logged = _train(tmp_path, family, tiny)
    metrics = held_out_metrics(_recorded(run), str(tmp_path / "checkpoints" / f"step-{STEPS}"), TINY_SUITE)
    assert metrics == pytest.approx(logged, rel=1e-5)


def test_the_suite_reads_the_run_from_its_record(tmp_path, tiny):
    run, logged = _train(tmp_path, "wam", tiny)
    result = TINY_SUITE.evaluate(_finished(run, tmp_path), STEPS, str(tmp_path / "eval"), EvalRuntime(None))
    assert result.metrics == pytest.approx(logged, rel=1e-5)
    assert (result.run, result.step, result.suite, result.examples) == ("episode/tiny@2026.10.13", STEPS, "held-out", 2)


def test_a_record_with_a_field_the_code_dropped_names_the_migration(tmp_path, tiny):
    run = tiny_run(tmp_path, "wam", STEPS, tiny, "unused")
    finished = _finished(run, tmp_path, stale_data_field=True)
    with pytest.raises(ValueError, match=r"--drop data\.FIELD"):
        TINY_SUITE.evaluate(finished, STEPS, str(tmp_path / "eval"), EvalRuntime(None))


def test_the_suite_refuses_a_device_batch_size(tmp_path, tiny):
    run = tiny_run(tmp_path, "wam", STEPS, tiny, "unused")
    with pytest.raises(ValueError, match="suite.batch_size"):
        TINY_SUITE.evaluate(_finished(run, tmp_path), STEPS, str(tmp_path / "eval"), EvalRuntime(8))
```

The logged dict holds only the `eval/*` keys: the loop logs the held-out loss in its own `tracker.log` call. If the record also carries other keys, compare `{k: v for k, v in logged.items() if k.startswith("eval/")}`.

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest experiments/episode_models/tests/test_evaluation.py -n 0 -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'experiments.episode_models.evaluation'`.

- [ ] **Step 3: Give `eval_times` its count**

```python
def eval_times(count: int) -> np.ndarray:
    """The grid a held-out loss averages over: the midpoints of ``count`` equal intervals of [0, 1]."""
    assert count >= 1, count
    return (np.arange(count) + 0.5) / count
```

Change `test_flow.py:32` to `eval_times(EVAL_TIME_COUNT)`, importing `EVAL_TIME_COUNT`.

- [ ] **Step 4: Expose the examples and the held-out loss from `train.py`**

Add `import grain` and import `EVAL_TIME_COUNT` from `flow`. Then:

```python
@dataclass(frozen=True)
class EpisodeExamples:
    """A dataset's sizes and its training and held-out clips, as model examples."""

    dims: EpisodeDims
    train: grain.MapDataset
    held_out: grain.MapDataset


def load_examples(dataset_path: str, action_stats_path: str | None, data: EpisodeDataConfig) -> EpisodeExamples:
    dataset = EpisodeDataset.raw_load(dataset_path)
    stats = None
    if data.action_field is not None:
        assert action_stats_path is not None
        stats = read_action_stats(action_stats_path, data.action_field)
    tasks = task_vocabulary(dataset.index, data.task_key) if data.task_key is not None else None
    train_rows, eval_rows = split_rows(dataset.index, data.eval_percent)
    return EpisodeExamples(
        dims=dataset_dims(dataset, data),
        train=example_dataset(dataset, data, train_rows, stats, tasks),
        held_out=example_dataset(dataset, data, eval_rows, stats, tasks),
    )
```

Rename `_make_evaluate` to `held_out_loss(mp, streams, weights, *, flow_times: int, seed: int)`. Use `eval_times(flow_times)` for the grid and `jax.random.key(seed)` for the noise key, and change its docstring's "fixed key" to "the key ``seed``". In `_train_locally`, replace the dataset/stats/tasks/split/example lines with `examples = load_examples(run.dataset_path, run.action_stats_path, run.data)`. Use `examples.dims`, `examples.train`, and `examples.held_out`, and build the evaluator as `held_out_loss(mp, streams, weights, flow_times=EVAL_TIME_COUNT, seed=EVAL_SEED)`.

- [ ] **Step 5: Run the training tests to confirm the refactor changed nothing**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest experiments/episode_models/tests/test_train.py experiments/episode_models/tests/test_flow.py -n 0 -q`
Expected: PASS.

- [ ] **Step 6: Write the suite**

```python
# experiments/episode_models/evaluation.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluating episode-model runs (D114): their checkpoint type and the held-out loss suite.

An episode trial records :class:`EpisodeCheckpoint`, so ``python -m poorin.experiment.evaluate``
finds the suites below. :class:`HeldOutLoss` recomputes the training loop's held-out loss from a
permanent checkpoint with the loop's own function. Its defaults are the training defaults, so with
them it reproduces the ``eval/*`` values the run logged at that step (logged at step ``N - 1`` for
the checkpoint ``step-N``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import jax
import jmp
from haliax.partitioning import set_mesh
from poorin.evaluation.contract import (
    EvalResult,
    EvalRuntime,
    FinishedRun,
    Suite,
    checkpoint_path,
    recorded,
    recorded_value,
)
from poorin.execution.artifact import Artifact
from poorin.execution.fingerprint import CLASS_KEY

from experiments.episode_models.data import EpisodeDataConfig, data_mesh, eval_batches, to_batch
from experiments.episode_models.flow import EVAL_TIME_COUNT
from experiments.episode_models.train import EVAL_SEED, EpisodeModelConfig, held_out_loss, load_examples
from experiments.episode_models.vla.model import VlaConfig
from experiments.episode_models.wam.model import WamConfig
from experiments.episode_models.world_model.model import WorldModelConfig
from experiments.grug.checkpointing import load_grug_checkpoint

# A family's model config, by the class name its record carries.
MODEL_CONFIGS: dict[str, type] = {cls.__name__: cls for cls in (WamConfig, WorldModelConfig, VlaConfig)}


@dataclass(frozen=True)
class RecordedEpisodeRun:
    """What the held-out loss reads from a run's recipe."""

    model: EpisodeModelConfig
    data: EpisodeDataConfig
    mp: str
    dataset_path: str
    action_stats_path: str | None


def recorded_episode_run(run: FinishedRun) -> RecordedEpisodeRun:
    family = cast(str, recorded_value(run, f"model.{CLASS_KEY}"))
    if family not in MODEL_CONFIGS:
        raise ValueError(f"{run.label} records a {family} model; the episode families are {sorted(MODEL_CONFIGS)}")
    return RecordedEpisodeRun(
        model=recorded(run, "model", MODEL_CONFIGS[family]),
        data=recorded(run, "data", EpisodeDataConfig),
        mp=recorded(run, "train.mp", str),
        dataset_path=recorded(run, "run.dataset_path", str),
        action_stats_path=cast(str | None, recorded_value(run, "run.action_stats_path")),
    )


def held_out_metrics(run: RecordedEpisodeRun, checkpoint: str, suite: "HeldOutLoss") -> dict[str, float]:
    """The held-out loss of the parameters at ``checkpoint``, by the training loop's own function."""
    examples = load_examples(run.dataset_path, run.action_stats_path, run.data)
    mp = jmp.get_policy(run.mp)
    mesh = data_mesh()
    with set_mesh(mesh):
        exemplar = jax.jit(lambda key: mp.cast_to_param(run.model.init(examples.dims, run.data.clip, key=key)))(
            jax.random.key(0)
        )
        loaded = load_grug_checkpoint(state={"params": exemplar}, candidate=checkpoint, mesh=mesh, allow_partial=False)
        batches = [
            to_batch(batch, mesh)
            for batch in eval_batches(examples.held_out, batch_size=suite.batch_size, count=suite.clips // suite.batch_size)
        ]
        evaluate = held_out_loss(
            mp, run.model.streams, run.model.stream_weights, flow_times=suite.flow_times, seed=suite.seed
        )
        return evaluate(loaded["params"], batches)


@dataclass(frozen=True)
class HeldOutLoss:
    """The training loop's held-out loss at one checkpoint.

    ``clips`` held-out clips spread evenly over the split, in batches of ``batch_size``, each stream's
    loss averaged over a grid of ``flow_times`` flow times. Batch ``i`` draws its noise from
    ``fold_in(key(seed), i)``, so the batch size is part of the recipe.
    """

    name: ClassVar[str] = "held-out"
    clips: int = 128
    batch_size: int = 32
    flow_times: int = EVAL_TIME_COUNT
    seed: int = EVAL_SEED

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.clips < self.batch_size or self.clips % self.batch_size:
            raise ValueError(f"clips must be a positive multiple of batch_size, not {self.clips} and {self.batch_size}")
        if self.flow_times < 1:
            raise ValueError(f"flow_times must be at least 1, not {self.flow_times}")

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        if runtime.batch_size is not None:
            raise ValueError(
                "the episode held-out loss draws its noise per batch, so its batch size is part of the "
                "recipe: pass --set suite.batch_size=N, not --batch-size"
            )
        metrics = held_out_metrics(recorded_episode_run(run), checkpoint_path(run, step), self)
        return EvalResult(run=run.label, step=step, suite=self.name, metrics=metrics, examples=self.clips)


class EpisodeCheckpoint(Artifact):
    """A finished episode-model training run: a WAM, a world model, or a VLA."""

    @classmethod
    def suites(cls) -> Mapping[str, type[Suite]]:
        return {HeldOutLoss.name: HeldOutLoss}
```

- [ ] **Step 7: Record the checkpoint type**

In `launch.py`, import `EpisodeCheckpoint` from `experiments.episode_models.evaluation`, set `artifact_type=EpisodeCheckpoint` in `episode_trial`, and change its return annotation and the family launchers' (`wam/launch.py`, `world_model/launch.py`, `vla/launch.py`) to `ArtifactStep[EpisodeCheckpoint]`. Run `rg LevanterCheckpoint experiments/episode_models` and update each test or annotation it finds.

- [ ] **Step 8: Add the architecture row**

```markdown
| `experiments.episode_models.evaluation` | The episode runs' checkpoint type and the held-out loss suite, which reproduces the loss training logged (D114) | `experiments.episode_models.evaluation` |
```

- [ ] **Step 9: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest experiments/episode_models/tests -n 0 -q`
Expected: PASS, including the three-family reproduction.

- [ ] **Step 10: Commit**

```bash
git add experiments/episode_models docs/poorin/architecture.md
git commit -m "[poorin] Evaluate episode runs' held-out loss from a checkpoint"
```

### Task 9: Grug base's held-out set and suite

**Files:**
- Create: `experiments/datasets/wikitext.py`
- Modify: `experiments/tutorials/train_tiny_model.py` (use `wikitext2_dataset`)
- Modify: `experiments/grug/base/train.py` (`tagged_evaluator`, under `build_tagged_evaluator`)
- Create: `experiments/grug/base/evaluation.py`
- Modify: `experiments/grug/base/launch.py` (validation set, eval at the final step, `artifact_type=GrugCheckpoint`)
- Modify: `tests/test_grug_variant_contracts.py` (make `tiny_grug_base_run_config`, `json_tracker_logger`, `json_tracker_records` public)
- Modify: `docs/poorin/architecture.md` (rows)
- Test: `tests/test_grug_base_evaluation.py`, `tests/test_grug_base_launch.py`

**Interfaces:**
- Consumes: Task 3 (`EvalResult`, `EvalRuntime`, `FinishedRun`, `Suite`, `checkpoint_path`, `recorded`, `recorded_value`).
- Produces:
  - `wikitext2_dataset() -> ArtifactStep[TokenizedCache]`.
  - `tagged_evaluator(*, data_config, max_seq_len, mesh, batch_size, batch_pspec, max_examples_per_dataset, compute_bpb)`.
  - `Weights(StrEnum)`: `EMA = "ema"`, `CURRENT = "current"`.
  - `HeldOutLoss(max_sequences: int | None = None, weights: Weights = Weights.EMA)`.
  - `RecordedGrugRun(model, data, mp, ema, eval_batch_size)`.
  - `recorded_grug_run(run: FinishedRun) -> RecordedGrugRun`.
  - `held_out_metrics(run: RecordedGrugRun, checkpoint: str, suite: HeldOutLoss, *, batch_size: int, compute_bpb: bool) -> tuple[dict[str, float], int]`.
  - `GrugCheckpoint(Artifact)`, whose `suites()` is `{"held-out": HeldOutLoss}`.

- [ ] **Step 1: Make the tiny-run helpers public**

In `tests/test_grug_variant_contracts.py`, rename `_tiny_grug_base_run_config` to `tiny_grug_base_run_config`, `_json_tracker_logger` to `json_tracker_logger`, and `_json_tracker_records` to `json_tracker_records`, with their call sites.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_grug_base_evaluation.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug base's held-out loss suite reproduces the evaluation training logged."""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import pytest
from fray.types import ResourceConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.examples import GrugLmExample
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from poorin.evaluation.contract import FinishedRun
from poorin.execution.artifact import ArtifactRecord
from poorin.execution.fingerprint import canonical_json

from experiments.grug.base import train as train_module
from experiments.grug.base.evaluation import HeldOutLoss, RecordedGrugRun, Weights, held_out_metrics, recorded_grug_run
from experiments.grug.base.launch import GrugBaseLaunchConfig
from tests.test_grug_variant_contracts import json_tracker_logger, json_tracker_records, tiny_grug_base_run_config

RUN_ID = "test-grug-eval"
F32 = "params=float32,compute=float32,output=float32"


def _train(tmp_path: Path) -> tuple[train_module.GrugRunConfig, dict[str, float], str]:
    """A one-step tiny grug run; its config, the evaluation it logged, and its last checkpoint."""
    with json_tracker_logger() as (logger_name, stream):
        config = tiny_grug_base_run_config(tmp_path / "run", run_id=RUN_ID, logger_name=logger_name)
        train_module.run_grug(config)
    (*_, logged) = [r["metrics"] for r in json_tracker_records(stream) if r["event"] == "log" and "eval/loss" in r["metrics"]]
    return config, logged, str(tmp_path / "run" / "checkpoints" / RUN_ID / "step-1")


def test_the_held_out_loss_of_a_checkpoint_reproduces_the_logged_one(tmp_path):
    config, logged, checkpoint = _train(tmp_path)
    recorded = RecordedGrugRun(model=config.model, data=config.data, mp=F32, ema=False, eval_batch_size=None)
    metrics, examples = held_out_metrics(
        recorded, checkpoint, HeldOutLoss(max_sequences=1, weights=Weights.CURRENT), batch_size=1, compute_bpb=False
    )
    expected = {k: v for k, v in logged.items() if not k.endswith(("loading_time", "total_time"))}
    assert metrics == pytest.approx(expected, rel=1e-5)
    assert examples == 1


def test_the_held_out_loss_does_not_depend_on_the_device_batch_size(tmp_path):
    config, _, checkpoint = _train(tmp_path)
    held_out = [GrugLmExample.causal((jnp.arange(32, dtype=jnp.int32) + i) % 128) for i in range(4)]
    data = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(held_out)})},
        vocab_size=128,
        tokenizer="passthrough",
    )
    recorded = RecordedGrugRun(model=config.model, data=data, mp=F32, ema=False, eval_batch_size=None)
    suite = HeldOutLoss(weights=Weights.CURRENT)
    losses = [
        held_out_metrics(recorded, checkpoint, suite, batch_size=size, compute_bpb=False)[0]["eval/loss"]
        for size in (1, 2, 4)
    ]
    assert losses == pytest.approx([losses[0]] * 3, rel=1e-6)


def test_the_suite_reads_the_run_from_its_record():
    text = TextLmDatasetFormat()
    data = LmDataConfig(
        components={
            "wikitext": DatasetComponent(
                source=UrlDatasetSourceConfig(cache_dir="/c/wikitext", format=text), cache_dir="/c/wikitext", format=text
            )
        },
        train_weights={"wikitext": 0.0},
        tokenizer="marin-community/marin-tokenizer",
    )
    model = tiny_grug_base_run_config(Path("/unused"), run_id="r", logger_name="unused").model
    launch = GrugBaseLaunchConfig(
        model=model, data=data, output_path="/runs/grug", run_id="r", resources=ResourceConfig.with_cpu(), steps=1,
        batch_size=1, seed=0, mp=F32, tracker=JsonLoggerConfig(), finelog_url=None, optimizer=AdamConfig(),
        watch_interval=0, backward_flow_interval=0, ema_beta=0.999, eval_batch_size=16,
    )
    record = ArtifactRecord(name="grug/base-trial", version="2026.10.13", config=json.loads(canonical_json(launch)))
    recorded = recorded_grug_run(FinishedRun("grug/base-trial@2026.10.13", "/runs/grug", record))
    assert (recorded.model, recorded.mp, recorded.ema, recorded.eval_batch_size) == (model, F32, True, 16)
    assert json.loads(canonical_json(recorded.data)) == json.loads(canonical_json(data))
```

`tiny_grug_base_run_config` makes `run_dir`, so build the model in the last test without it if that matters: copy its `_small_model_config(...)` call instead, and record that ruling. Read the tiny trainer's `mp` before relying on `F32`. If `TrainerConfig`'s default policy is not float32 parameters, pass the policy string matching it.

Add to `tests/test_grug_base_launch.py`:

```python
def test_the_trial_holds_out_wikitext_and_evaluates_at_its_last_step():
    trial = grug_base_trial(overrides=Overrides())
    assert "wikitext" in {dep.name for dep in trial.deps}
    config = json.loads(trial.fingerprint_payload())
    assert config["data"]["train_weights"]["wikitext"] == 0.0
    assert config["eval_batch_size"] == 16
    assert trial.artifact_type.suites().keys() == {"held-out"}
```

The fingerprint form of a mixture's weights is `{name@version: weight}` (`mixture` at fingerprint time). If `train_weights` is keyed by name there, assert on the key the payload actually holds.

- [ ] **Step 3: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_grug_base_evaluation.py tests/test_grug_base_launch.py -n 0 -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'experiments.grug.base.evaluation'`.

- [ ] **Step 4: The WikiText-2 handle**

```python
# experiments/datasets/wikitext.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WikiText-2 (detokenized) from the Hugging Face Hub, a 1,000-document sample per shard tokenized
with the Marin tokenizer. The tutorial trains on it; grug base holds out its validation split."""

from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import tokenized
from poorin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def wikitext2_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "wikitext",
        tokenizer=marin_tokenizer,
        source="dlwh/wikitext_2_detokenized",
        sample_count=1000,
        version="2026.06.28",
    )
```

In `train_tiny_model.py`, drop `"wikitext"` from `RAW_SOURCES`. In `dataset()`, return `wikitext2_dataset()` for `"wikitext"` and update the docstring. The handle's name, version, and arguments are the tutorial's, so its fingerprint and output path are unchanged.

- [ ] **Step 5: Split the evaluator builder**

In `experiments/grug/base/train.py`, move the body of `build_tagged_evaluator` into:

```python
def tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    batch_size: int,
    batch_pspec: P,
    max_examples_per_dataset: int | None,
    compute_bpb: bool,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    """The evaluator over every validation set of ``data_config``; training and post-hoc evaluation share it."""
    ...  # the existing body, reading these arguments instead of eval_cfg's fields


def build_tagged_evaluator(
    *, data_config: LmDataConfig, max_seq_len: int, mesh: Mesh, eval_cfg: GrugEvalConfig
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    max_examples = None if eval_cfg.max_eval_batches is None else eval_cfg.max_eval_batches * eval_cfg.eval_batch_size
    return tagged_evaluator(
        data_config=data_config,
        max_seq_len=max_seq_len,
        mesh=mesh,
        batch_size=eval_cfg.eval_batch_size,
        batch_pspec=eval_cfg.eval_batch_pspec,
        max_examples_per_dataset=max_examples,
        compute_bpb=eval_cfg.compute_bpb,
    )
```

- [ ] **Step 6: Write the suite**

```python
# experiments/grug/base/evaluation.py
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluating grug base runs (D114): their checkpoint type and the held-out loss suite.

A grug base trial records :class:`GrugCheckpoint`. :class:`HeldOutLoss` scores a permanent
checkpoint on each of the run's validation sets with the evaluator training uses, and reports the
keys training logs without its timing keys: ``eval/loss``, ``eval/bpb``, ``eval/<set>/loss``, and
``eval/<set>/bpb``, with ``eval/macro_*`` when there are several sets, under ``eval/ema/`` for EMA
weights.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import jax
import jmp
from haliax.partitioning import set_mesh
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import eval_model
from levanter.grug.sharding import compact_grug_mesh
from poorin.evaluation.contract import (
    EvalResult,
    EvalRuntime,
    FinishedRun,
    Suite,
    checkpoint_path,
    recorded,
    recorded_value,
)
from poorin.execution.artifact import Artifact

from experiments.grug.base.model import GrugModelConfig, Transformer
from experiments.grug.base.train import GrugEvalConfig, tagged_evaluator
from experiments.grug.checkpointing import load_grug_checkpoint

# Training's evaluation settings: the prefix its keys log under, the batch sharding, and bpb.
TRAINING_EVAL = GrugEvalConfig()
TIMING_KEYS = ("loading_time", "total_time")
# The device batch size for a run that recorded none.
DEFAULT_BATCH_SIZE = 16


class Weights(StrEnum):
    EMA = "ema"
    CURRENT = "current"


@dataclass(frozen=True)
class RecordedGrugRun:
    """What the held-out loss reads from a run's recipe."""

    model: GrugModelConfig
    data: LmDataConfig
    mp: str
    ema: bool
    """Whether the run kept EMA weights."""
    eval_batch_size: int | None


def recorded_grug_run(run: FinishedRun) -> RecordedGrugRun:
    return RecordedGrugRun(
        model=recorded(run, "model", GrugModelConfig),
        data=recorded(run, "data", LmDataConfig),
        mp=recorded(run, "mp", str),
        ema=recorded_value(run, "ema_beta") is not None,
        eval_batch_size=cast(int | None, recorded_value(run, "eval_batch_size")),
    )


def held_out_metrics(
    run: RecordedGrugRun, checkpoint: str, suite: "HeldOutLoss", *, batch_size: int, compute_bpb: bool
) -> tuple[dict[str, float], int]:
    """The loss and bpb of the weights at ``checkpoint`` on each validation set, and the sequences scored."""
    ema = suite.weights is Weights.EMA and run.ema
    prefix = f"{TRAINING_EVAL.prefix}/ema" if ema else TRAINING_EVAL.prefix
    weights_field = "ema_params" if ema else "params"
    mp = jmp.get_policy(run.mp)
    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        evaluator = tagged_evaluator(
            data_config=run.data,
            max_seq_len=run.model.max_seq_len,
            mesh=mesh,
            batch_size=batch_size,
            batch_pspec=TRAINING_EVAL.eval_batch_pspec,
            max_examples_per_dataset=suite.max_sequences,
            compute_bpb=compute_bpb,
        )
        if evaluator is None:
            raise ValueError(f"{checkpoint}: the run's data has no validation set")
        exemplar = jax.jit(lambda key: mp.cast_to_param(Transformer.init(run.model, key=key)))(jax.random.key(0))
        loaded = load_grug_checkpoint(state={weights_field: exemplar}, candidate=checkpoint, mesh=mesh, allow_partial=False)
        log = eval_model(evaluator, loaded[weights_field], prefix=prefix)
    metrics = {key: float(value) for key, value in log.items() if not key.endswith(TIMING_KEYS)}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"the held-out loss at {checkpoint} is not finite: {metrics}")
    return metrics, len(evaluator.dataset.as_sync_dataset())


@dataclass(frozen=True)
class HeldOutLoss:
    """Loss and bits per byte on each of the run's validation sets at one checkpoint."""

    name: ClassVar[str] = "held-out"
    max_sequences: int | None = None
    """Sequences scored per validation set; ``None`` scores each whole set. Training caps each set at
    ``max_eval_batches * eval_batch_size``, so reproducing a run passes that cap."""
    weights: Weights = Weights.EMA
    """``ema`` falls back to the current weights for a run that kept no EMA."""

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        recorded_run = recorded_grug_run(run)
        batch_size = runtime.batch_size or recorded_run.eval_batch_size or DEFAULT_BATCH_SIZE
        metrics, examples = held_out_metrics(
            recorded_run, checkpoint_path(run, step), self, batch_size=batch_size, compute_bpb=TRAINING_EVAL.compute_bpb
        )
        return EvalResult(run=run.label, step=step, suite=self.name, metrics=metrics, examples=examples)


class GrugCheckpoint(Artifact):
    """A finished grug base training run."""

    @classmethod
    def suites(cls) -> Mapping[str, type[Suite]]:
        return {HeldOutLoss.name: HeldOutLoss}
```

If `evaluator.dataset.as_sync_dataset()` is not the way to count the capped sequences, count them from the evaluator's `DomainTaggedDataset` offsets, and record the ruling.

- [ ] **Step 7: Give the trial its held-out set**

In `experiments/grug/base/launch.py`, import `wikitext2_dataset` and `GrugCheckpoint`, and add a module constant `_HELD_OUT_DATA = wikitext2_dataset()` beside `_TRAIN_DATA`. In `grug_base_trial`:

```python
                data=mixture(ctx, train, validation=(_HELD_OUT_DATA,)),
                ...
                # One evaluation, at the final step: the loop forces its callbacks there, and steps_per_eval
                # (1000) is past the trial's 50 steps. 8 batches of 16 score 128 held-out sequences.
                eval_batch_size=16,
```

Set `deps=(*train, _HELD_OUT_DATA)` and `artifact_type=GrugCheckpoint`, and change the return annotations of `grug_base_trial` and `main` to `ArtifactStep[GrugCheckpoint]`. Update the module docstring's "typed `Checkpoint` handle" wording if it names the type.

- [ ] **Step 8: Add the architecture rows**

```markdown
| `experiments.datasets.wikitext` | WikiText-2 tokenized with the Marin tokenizer: the tutorial's data and grug base's held-out set | R1 |
| `experiments.grug.base.evaluation` | Grug base runs' checkpoint type and the held-out loss suite, which reproduces the evaluation training logged (D114) | R2 |
```

- [ ] **Step 9: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_grug_base_evaluation.py tests/test_grug_base_launch.py tests/test_grug_variant_contracts.py -n 0 -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add experiments/datasets/wikitext.py experiments/tutorials/train_tiny_model.py experiments/grug/base tests/test_grug_base_evaluation.py tests/test_grug_base_launch.py tests/test_grug_variant_contracts.py docs/poorin/architecture.md
git commit -m "[grug] Hold out WikiText-2 and evaluate grug base runs post hoc"
```

### Task 10: Documentation and the whole suite

**Files:**
- Modify: `docs/explanations/evaluation.md` (a section on evaluation artifacts)
- Modify: `README.md` ("Run something": one example)
- Modify: `lib/poorin/README.md` (`poorin.evaluation` line)

- [ ] **Step 1: Write the documentation**

Add a section `## Evaluations as artifacts` to `docs/explanations/evaluation.md`. Follow the `writing-style` skill's reference-docs guide. Cover:

- What an evaluation is, and its name.
- The three kinds of short references.
- One command for a plan and one with `--run`.
- `--steps all`, `--set suite.FIELD`, and `--batch-size`.
- The two `held-out` suites and their defaults.
- Reproducing a training value: the episode suite does it by default; grug needs `--set suite.max_sequences=128` for the base trial.
- `runs migrate`.
- Reading results with `eval_results`.

Link the spec and D114. Add to the README's "Run something" block:

```bash
# Score every checkpoint of a finished run on its held-out loss
uv run --no-sync python -m poorin.experiment.evaluate episode/wam-libero-spatial@latest \
    --suite held-out --steps all --version dev --run
```

Change the `poorin.evaluation` line of `lib/poorin/README.md` to: "`poorin.evaluation`: the evaluation contract (suites, checkpoint types, `EvalResult`), the lm-eval harness integration, evaluation records, and statistics."

- [ ] **Step 2: Lint, type-check, and run the affected tests**

Run: `uv run --no-sync python -m tools.lint --fix`
Expected: clean; it also runs `ty`.

Run: `uv run --no-sync python -m tools.affected_tests`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/explanations/evaluation.md README.md lib/poorin/README.md
git commit -m "[poorin] Document evaluation artifacts and the evaluate command"
```

### Task 11: GPU acceptance

Run on the RTX 5090 with nothing else on the GPU (`nvidia-smi` shows no other process). Calendar versions refuse uncommitted code, so every earlier task is committed first.

**Files:**
- Modify: `docs/poorin/decisions.md` (D114: an acceptance bullet with the measured values)
- Modify: `docs/poorin/prune-metrics.csv` (through `prune_check`)

- [ ] **Step 1: Migrate the three D113 acceptance runs**

```bash
uv run --no-sync python -m poorin.experiment.runs migrate episode/wam-libero-spatial@60b9 --to experiments.episode_models.evaluation.EpisodeCheckpoint --drop data.num_workers
```

```bash
uv run --no-sync python -m poorin.experiment.runs migrate episode/world-model-ssv2-500@f777 --to experiments.episode_models.evaluation.EpisodeCheckpoint --drop data.num_workers
```

```bash
uv run --no-sync python -m poorin.experiment.runs migrate episode/vla-libero-spatial@3ddf --to experiments.episode_models.evaluation.EpisodeCheckpoint --drop data.num_workers
```

Expected: each prints `... now records experiments.episode_models.evaluation.EpisodeCheckpoint, without data.num_workers`.

- [ ] **Step 2: Reproduce their held-out losses at step 2000**

```bash
uv run --no-sync python -m poorin.experiment.evaluate episode/wam-libero-spatial@60b9 episode/world-model-ssv2-500@f777 episode/vla-libero-spatial@3ddf --suite held-out --steps 2000 --version 2026.10.13 --run
```

Compare each result with the run's logged summary. The `tracker_metrics.jsonl` summary holds the last logged values, which are step 2000's:

```bash
uv run --no-sync python -c "
import json
from poorin.experiment.evaluate import eval_results
from rigging.filesystem.cluster_config import poorin_prefix
for r in eval_results(poorin_prefix(), suite='held-out'):
    run_dir = poorin_prefix() + '/' + r.run.replace('@', '/')
    logged = json.loads(open(run_dir + '/tracker_metrics.jsonl').readline())['summary']
    print(r.run, r.step, {k: (round(v, 6), round(logged[k], 6)) for k, v in r.metrics.items()})
"
```

Expected: each pair agrees to about 1e-5 relative. The `eval/loss` values are WAM 0.7413, world model about 0.580, and VLA about 0.222. bf16 compute on the GPU can differ from training in the last digits if XLA fuses differently; a difference above 1e-3 relative is a finding to debug, not a tolerance to widen.

- [ ] **Step 3: Train a short grug run and reproduce its logged evaluation**

```bash
uv run --no-sync python experiments/grug/base/launch.py --version accept-dev --run
```

Note the printed version's token, then:

```bash
uv run --no-sync python -m poorin.experiment.evaluate grug/base-trial@TOKEN --suite held-out --steps all --version dev --set suite.max_sequences=128 --set suite.weights=current --run
```

Expected: the result's `eval/loss` and `eval/bpb` match the run's `tracker_metrics.jsonl` summary values to about 1e-5 relative, and the WikiText-2 cache has a `validation/` directory.

- [ ] **Step 4: Run one matrix twice**

```bash
uv run --no-sync python -m poorin.experiment.evaluate episode/wam-libero-spatial@60b9 episode/vla-libero-spatial@3ddf --suite held-out --steps all --version 2026.10.13 --run
```

Run the same command again without `--run`. Expected: the second plan shows every evaluation `built`, including the step-2000 cells from Step 2, and the first run built only the six new cells.

- [ ] **Step 5: Record the metrics row**

```bash
uv run --no-sync python -m tools.prune_check --phase D114 --step "evaluation artifacts"
```

Expected: every check passes and both training roots train (R2 now evaluates at its last step).

- [ ] **Step 6: Record acceptance in D114 and commit**

Add an acceptance bullet to D114 with the reproduced values (the run, the logged value, the post-hoc value), the grug run's values, and the matrix's second-run result.

```bash
git add docs/poorin/decisions.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Record evaluation artifacts acceptance on the RTX 5090"
```
