# Experiment Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Poorin's experiments typed `--set`/`--sweep` overrides, a readable plan with
`--show-config`, and a drift check that refuses to serve a stale output, all on draccus.

**Architecture:** A new `poorin.execution.cache_state` module computes each artifact's state (built,
started, drifted) from what its output path records; the CLI plan and a whole-graph preflight in
`lazy.run` both use it. The fingerprint gains each dataclass's class and becomes strict. A new
`poorin.experiment.overrides` module decodes `PATH=VALUE` with draccus and is applied by each
experiment inside `build_config`, so overrides enter the fingerprint and the artifact name.

**Tech Stack:** Python 3.12, draccus 0.11.6 (`draccus.decode`, `draccus.cfgparsing.parse_string`,
`draccus.choice_types.ChoiceRegistryBase`), click, PyYAML, pytest.

**Spec:** `docs/poorin/specs/2026-10-08-experiment-overrides-design.md` (decision D92).

## Global Constraints

- Run every command with `uv run --no-sync`. Unit tests run on CPU: `JAX_PLATFORMS=cpu uv run --no-sync pytest <file> -n 0`.
- Do not pass a partial marker expression such as `-m "not slow"`.
- Each commit runs `uv run --no-sync python -m tools.prune_check --phase Overrides --step "<label>"` in full and stages its new `docs/poorin/prune-metrics.csv` row.
- Commit titles: imperative, at most 72 characters, `[poorin]` prefix; bodies follow `.agents/skills/writing-style/pull-requests.md`. No `Co-Authored-By` or tool attribution. Each commit step gives the body's content in `<angle brackets>`; write it as plain prose before committing.
- Record each ruling that departs from the spec in `docs/poorin/decisions.md` (D93, "Rulings while building D92") in the commit that makes it.
- No backward compatibility: `--override` is renamed `--version-of` with no alias; `expected_fingerprint`, `set_strict`, and `POORIN_FINGERPRINT_STRICT` are deleted with every caller updated.
- All imports at the top of the file. No `TYPE_CHECKING` guards; break the `lazy` ↔ `cache_state` cycle with a `Protocol`.
- Every new module gets a row in `docs/poorin/architecture.md` (`tests/tools/test_architecture.py` enforces it).
- Tests follow `TESTING.md`: behavior, not guard-mirroring; no relaxed assertions.

## Review Focus

1. A `--sweep` whose values are mappings (`--sweep 'optimizer=[{type: adam}, {type: muon}]'`): each trial must get a distinct `set-<digest>` name and decode to the right class. Test in Task 6.
2. A `--set` value containing `=` or `,` (`--set mp=params=float32,compute=bfloat16`): the path is everything before the first `=`, and the trial takes the digest name, never a readable name another override set could share. Test in Task 5.
3. Optional fields: `--set ema_beta=0.999` on a `float | None` field that is `None`, and `--set eval_batch_size=null`, must decode through the union. Test in Task 5.
4. `--accept-drift` naming a *dependency* whose recipe changed under a built consumer: the dependency is re-stamped and the consumer served. Test in Task 3.
5. A `--set` path that walks into a non-dataclass value (a list or a string field): a clear error naming the path, not an `AttributeError`. Test in Task 5.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `lib/poorin/src/poorin/execution/cache_state.py` | Create | An artifact's state at its output path: `CacheState`, `HandleState`, `handle_state`, `record_path` |
| `lib/poorin/src/poorin/execution/artifact.py` | Modify | `RecordedRecipe`, `recorded_recipe`, `accept_recipe`, `ArtifactRecord.accepted_from`, fail-closed `check_drift`, `drift_message`; drop `EXPECTED_FINGERPRINT_KEY` |
| `lib/poorin/src/poorin/execution/fingerprint.py` | Modify | `__class__` key per dataclass; strict only |
| `lib/poorin/src/poorin/execution/lazy.py` | Modify | `graph_handles` (moved from cli), `StepContext.is_context_value`, preflight in `run(accept_drift=...)`; drop `expected_fingerprint` |
| `lib/poorin/src/poorin/execution/step_runner.py` | Modify | The start stub records `fingerprint_payload` |
| `lib/poorin/src/poorin/execution/build_context.py` | Modify | `VersionCodex.overrides` → `by_name` |
| `lib/poorin/src/poorin/experiment/overrides.py` | Create | `Override`, `Overrides`, `OverrideError`: parse, apply with draccus, name, run ID |
| `lib/poorin/src/poorin/experiment/cli.py` | Modify | Plan printer, `--show-config`, `--accept-drift`, `--set`, `--sweep`, `--version-of`, `--max-concurrent` rule |
| `lib/poorin/src/poorin/experiment/train.py` | Modify | Tracker settings as runtime args; `overrides` parameter |
| `lib/poorin/src/poorin/experiment/sweep.py` | Modify | Docstring points at `--sweep` |
| `experiments/grug/base/launch.py` | Modify | Tracker settings as runtime args; `@no_type_check`; `overrides` |
| `experiments/tutorials/train_tiny_model.py` | Modify | Passes `overrides` to `train_lm` |
| Tests | Create/modify | `tests/execution/test_cache_state.py`, `test_artifact.py`, `test_adopt.py`, `test_fingerprint.py`, `test_lazy.py`, `test_build_context.py`; `tests/experiment/test_cli.py`, `test_overrides.py`, `test_train_lm.py`; `tests/test_grug_base_launch.py` |
| Docs | Modify | `docs/explanations/lazy-artifacts.md`, `.agents/skills/write-pipeline/references/artifact-binding.md`, `experiments/grug/README.md`, `lib/poorin/README.md`, `docs/poorin/architecture.md`, `docs/poorin/decisions.md`, the spec |

---

### Task 1: Cache state and a readable plan (commit 1)

**Files:**
- Create: `lib/poorin/src/poorin/execution/cache_state.py`
- Modify: `lib/poorin/src/poorin/execution/artifact.py` (add after `read_record`, around line 305)
- Modify: `lib/poorin/src/poorin/execution/lazy.py` (add `graph_handles` after `resolve`, around line 451)
- Modify: `lib/poorin/src/poorin/experiment/cli.py:129-158` (remove `graph_handles`, rewrite `_print_plan`), `:165-228` (add `--show-config`)
- Test: `tests/execution/test_cache_state.py` (create), `tests/experiment/test_cli.py`
- Docs: `docs/poorin/architecture.md` (new row after `poorin.execution.build_context`)

**Interfaces:**
- Produces: `RecordedRecipe(fingerprint: str, payload: str | None, filename: str)` and `recorded_recipe(output_path: str) -> RecordedRecipe | None` in `artifact.py`; `CacheState`, `HandleState(handle, fingerprint, state, recorded_fingerprint, change)` with property `label -> str`, `handle_state(handle) -> HandleState`, `record_path(handle) -> str` in `cache_state.py`; `graph_handles(handles: list[ArtifactStep]) -> list[ArtifactStep]` in `lazy.py`.

- [ ] **Step 1: Write the failing state tests**

Create `tests/execution/test_cache_state.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What the plan and the drift check read at an artifact's output path."""

import pytest
from poorin.execution.artifact import Artifact
from poorin.execution.cache_state import CacheState, handle_state
from poorin.execution.lazy import ArtifactStep, run

V = "2026.06.28"


class Toy(Artifact):
    payload: str


def _toy(version: str, payload: str, *, fail: bool = False) -> ArtifactStep[Toy]:
    def build(config):
        if fail:
            raise RuntimeError("the build crashed")
        return Toy(payload=config["payload"])

    return ArtifactStep(
        name="datasets/toy",
        version=version,
        artifact_type=Toy,
        run=build,
        build_config=lambda ctx: {"out": ctx.output_path, "payload": payload},
    )


@pytest.fixture(autouse=True)
def _prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))


def test_a_calendar_artifact_moves_from_not_built_to_built():
    assert handle_state(_toy(V, "a")).state is CacheState.NOT_BUILT
    run(_toy(V, "a"))
    assert handle_state(_toy(V, "a")).state is CacheState.BUILT


def test_a_changed_recipe_reads_as_drifted_with_the_field_diff():
    run(_toy(V, "a"))
    state = handle_state(_toy(V, "b"))
    assert state.state is CacheState.DRIFTED
    assert state.recorded_fingerprint == _toy(V, "a").fingerprint()
    assert "payload: 'a' -> 'b'" in state.change


def test_a_build_that_crashed_reads_as_started():
    with pytest.raises(RuntimeError):
        run(_toy(V, "a", fail=True))
    assert handle_state(_toy(V, "a")).state is CacheState.STARTED


def test_a_dev_version_reads_as_mutable():
    assert handle_state(_toy("dev", "a")).state is CacheState.MUTABLE


def test_a_pinned_artifact_is_not_checked(tmp_path):
    pinned = ArtifactStep(
        name="datasets/pinned",
        version=V,
        artifact_type=Toy,
        run=lambda config: None,
        build_config=lambda ctx: {},
        override_path=str(tmp_path / "elsewhere"),
    )
    assert handle_state(pinned).state is CacheState.PINNED
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_cache_state.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poorin.execution.cache_state'`.

- [ ] **Step 3: Add `recorded_recipe` to `artifact.py`**

Add after `read_record` (it uses `_resolved`, `_read_text`, `_parse_record`, `FINGERPRINT_KEY`, `STEP_RUNNER_EXECUTOR_VERSION`, all in this module):

```python
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
```

- [ ] **Step 4: Create `cache_state.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What exists at an artifact's output path: nothing, a started build, a finished one, or one
built from another recipe.

The plan that ``poorin.experiment.cli`` prints and the drift check in
:func:`poorin.execution.lazy.run` both read an artifact's state here, so they agree on what a run
will do. Handles are typed by a protocol so that ``lazy`` can import this module.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rigging.filesystem.cluster_config import poorin_prefix
from rigging.filesystem.storage_path import prefix_join

from poorin.execution.artifact import is_mutable_version, recorded_recipe
from poorin.execution.fingerprint import describe_drift, fingerprint_hash
from poorin.execution.step_status import STATUS_SUCCESS, StatusFile


class Handle(Protocol):
    """The parts of an ``ArtifactStep`` the state check reads."""

    name: str
    version: str
    override_path: str | None

    def fingerprint_payload(self) -> str: ...


class CacheState(StrEnum):
    PINNED = "pinned"
    """Points at existing data through ``override_path``; not checked."""
    MUTABLE = "mutable"
    """A dev version: always builds, and a resumed one continues its checkpoints."""
    NOT_BUILT = "not built"
    STARTED = "started"
    """A calendar version whose build started but did not succeed: the build restarts."""
    BUILT = "built"
    DRIFTED = "built from another recipe"


@dataclass(frozen=True)
class HandleState:
    handle: Handle
    fingerprint: str
    state: CacheState
    recorded_fingerprint: str | None
    change: str
    """The field-level diff from :func:`describe_drift`; empty unless ``DRIFTED``."""

    @property
    def label(self) -> str:
        return f"{self.handle.name}@{self.handle.version}"


def record_path(handle: Handle) -> str:
    """Where ``handle``'s status and record live: ``{prefix}/{name}/{version}``, an adopted handle
    included (its data stays at the source; its record does not)."""
    return prefix_join(poorin_prefix(), f"{handle.name}/{handle.version}")


def handle_state(handle: Handle) -> HandleState:
    """``handle``'s state, from the recipe and status recorded at :func:`record_path`."""
    payload = handle.fingerprint_payload()
    fingerprint = fingerprint_hash(payload)
    if handle.override_path is not None:
        return HandleState(handle, fingerprint, CacheState.PINNED, None, "")
    path = record_path(handle)
    recorded = recorded_recipe(path)
    if recorded is not None and recorded.fingerprint != fingerprint:
        change = describe_drift(recorded.payload, payload)
        return HandleState(handle, fingerprint, CacheState.DRIFTED, recorded.fingerprint, change)
    recorded_fingerprint = recorded.fingerprint if recorded is not None else None
    if is_mutable_version(handle.version):
        return HandleState(handle, fingerprint, CacheState.MUTABLE, recorded_fingerprint, "")
    if StatusFile(path, worker_id="plan").status == STATUS_SUCCESS:
        return HandleState(handle, fingerprint, CacheState.BUILT, recorded_fingerprint, "")
    if recorded is not None:
        return HandleState(handle, fingerprint, CacheState.STARTED, recorded_fingerprint, "")
    return HandleState(handle, fingerprint, CacheState.NOT_BUILT, None, "")
```

- [ ] **Step 5: Run the state tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_cache_state.py -n 0 -q`
Expected: 5 passed. If `test_a_build_that_crashed_reads_as_started` raises something other than `RuntimeError`, read the runner's wrapping in `step_runner.py` and match the type it raises (the adopt test uses `RuntimeError`).

- [ ] **Step 6: Move `graph_handles` into `lazy.py`**

Cut `graph_handles` from `cli.py:129-144` and paste it unchanged after `resolve` in `lazy.py`. In `cli.py`, import it: `from poorin.execution.lazy import ArtifactStep, graph_handles, run`.

- [ ] **Step 7: Write the failing plan tests**

In `tests/experiment/test_cli.py`, replace `_resolved_model_version` and `test_plan_flags_artifacts_that_resolved_to_a_mutable_version`, and add plan tests:

```python
def _resolved_model_version(output: str) -> str:
    (line,) = [line for line in output.splitlines() if "models/toy@" in line]
    return line.split("models/toy@", 1)[1].split()[0]


def test_plan_reports_each_artifact_state(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))

    def build(payload):
        return lambda **_: apply("scratch/marker", _write_marker_with, out=OUT, payload=payload)

    first = runner.invoke(experiment_main(build("a")), ["--version", "2026.07.16"])
    assert "not built" in first.output
    runner.invoke(experiment_main(build("a")), ["--version", "2026.07.16", "--run"])
    built = runner.invoke(experiment_main(build("a")), ["--version", "2026.07.16"])
    assert re.search(r"^\s+built\s", built.output, re.MULTILINE)  # "not built" must not match
    changed = runner.invoke(experiment_main(build("b")), ["--version", "2026.07.16"])
    assert "built from another recipe" in changed.output
    assert "payload: 'a' -> 'b'" in changed.output
    dev = runner.invoke(experiment_main(build("a")), ["--version", "dev"])
    assert "mutable" in dev.output


def test_show_config_prints_the_identity_config_and_refuses_run(runner):
    shown = runner.invoke(experiment_main(_build_two), ["--version", "2026.07.16", "--show-config"])
    assert shown.exit_code == 0, shown.output
    assert "# models/toy@2026.07.16" in shown.output
    assert "data: data/toy@2026.07.16" in shown.output  # a dependency renders as name@version

    refused = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--show-config", "--run"])
    assert refused.exit_code != 0


def _write_marker_with(out, payload):
    _write_marker(out)
```

Add `import re`. Update `_build_two` and every other build function in this file to accept and ignore keyword arguments (`def _build_two(**_):`), because Task 6 passes `overrides=` to them; doing it now keeps Task 6's diff small.

- [ ] **Step 8: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_cli.py -n 0 -q`
Expected: FAIL: `"not built"` is not in the output, and `--show-config` is an unknown option.

- [ ] **Step 9: Rewrite the plan printer and add `--show-config`**

In `cli.py`, add imports `import json`, `import textwrap`, `import yaml`, `from rigging.filesystem.cluster_config import poorin_prefix`, `from poorin.execution.cache_state import HandleState, handle_state`. Replace `_print_plan`:

```python
def _print_plan(states: list[HandleState], shown: list[HandleState]) -> None:
    """One line per artifact (state, fingerprint, ``name@version``) with the diff under a drifted
    one, then the identity config of each artifact in ``shown`` as YAML."""
    click.echo(f"Prefix: {poorin_prefix()}")
    seen: set[str] = set()
    for state in states:
        if state.label in seen:
            continue
        seen.add(state.label)
        click.echo(f"  {state.state.value:<26} {state.fingerprint}  {state.label}")
        if state.change:
            click.echo(textwrap.indent(state.change.strip(), "      "))
    for state in shown:
        config = json.loads(state.handle.fingerprint_payload())
        click.echo(f"\n# {state.label}\n{yaml.safe_dump(config, sort_keys=True).rstrip()}")
```

In `build_options_with_runner`, add the option and use the states:

```python
    @click.option(
        "--show-config",
        is_flag=True,
        help="Print the plan and each returned artifact's identity config as YAML; run nothing.",
    )
```

with `show_config: bool` in `wrapper`'s keyword parameters, and replace the tail of `wrapper` from `if not do_run:`:

```python
        if show_config and do_run:
            raise click.UsageError("--show-config prints the plan; it cannot be combined with --run")
        states = [handle_state(h) for h in graph_handles(handles)]
        if not do_run:
            returned = {id(h) for h in handles}
            _print_plan(states, [s for s in states if id(s.handle) in returned] if show_config else [])
            return
        runner(handles, max_concurrent)
```

- [ ] **Step 10: Run the CLI and execution tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_cli.py tests/execution -n 0 -q`
Expected: all pass.

- [ ] **Step 11: Add the architecture row and lint**

Add to `docs/poorin/architecture.md` after the `poorin.execution.build_context` row:

```markdown
| `poorin.execution.cache_state` | An artifact's state at its output path (built, started, drifted) for the plan and the drift check | R1 |
```

Update the `poorin.execution.lazy` row's purpose to end with "graph walk". Run `uv run --no-sync python -m tools.lint --fix`, then `uv run --no-sync python -m tools.lint` (expect `OK`).

- [ ] **Step 12: Full check and commit**

Run: `uv run --no-sync python -m tools.prune_check --phase Overrides --step "Print a plan with each artifact's cache state"`
Expected: exit 0.

```bash
git add lib/poorin/src/poorin/execution/cache_state.py lib/poorin/src/poorin/execution/artifact.py lib/poorin/src/poorin/execution/lazy.py lib/poorin/src/poorin/experiment/cli.py tests/execution/test_cache_state.py tests/experiment/test_cli.py docs/poorin/architecture.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Print a plan with each artifact's cache state" -m "<body: the plan was one lowered StepSpec repr per handle; it now prints one line per artifact with its state (not built, started, built, mutable, pinned, built from another recipe), its fingerprint, and the field diff under a drifted one. --show-config adds each returned artifact's identity config as YAML. Running is unchanged. Decision D92.>"
```

---

### Task 2: Class names in the fingerprint, strict only, no `expected_fingerprint` (part of commit 2)

**Files:**
- Modify: `lib/poorin/src/poorin/execution/fingerprint.py` (docstring, `_FingerprintEncoder.default:99-101`, delete `_STRICT_OVERRIDE`, `set_strict`, `_is_strict`, `_fallback`, `_SEEN_UNFINGERPRINTABLE`)
- Modify: `lib/poorin/src/poorin/execution/lazy.py` (drop `expected_fingerprint` field `:202-205` and its check `:331-336`, `:399-400`; drop `EXPECTED_FINGERPRINT_KEY` import)
- Modify: `lib/poorin/src/poorin/execution/artifact.py` (drop `EXPECTED_FINGERPRINT_KEY` and its use in `check_drift`)
- Test: `tests/execution/test_fingerprint.py`, `tests/execution/test_artifact.py:101-120`

**Interfaces:**
- Produces: `CLASS_KEY = "__class__"` in `fingerprint.py`; every dataclass encodes as `{"__class__": <label>, **fields}`, where the label is the draccus choice name for a registered `ChoiceRegistryBase` subclass, else `__qualname__`. An unknown type raises `TypeError`.

- [ ] **Step 1: Write the failing fingerprint tests**

In `tests/execution/test_fingerprint.py`: delete the `_reset_strict` fixture, `test_unknown_type_falls_back_to_a_stable_form`, and `test_strict_mode_rejects_callable_via_env`; remove `set_strict` and `fingerprint_module` from the imports; in `test_strict_mode_rejects_default_repr_object` delete the `set_strict(True)` line and rename it `test_a_default_repr_object_is_rejected`. Add:

```python
from levanter.optim.config import AdamConfig
from levanter.optim.grugmuon import GrugMuonConfig
from levanter.optim.muon import MuonConfig


def test_a_callable_is_rejected():
    with pytest.raises(TypeError, match="callable"):
        canonical_json({"fn": lambda x: x})


def test_an_object_with_no_canonical_form_is_rejected():
    class Widget:
        def __init__(self, n: int) -> None:
            self.n = n

    with pytest.raises(TypeError, match="cannot fingerprint"):
        canonical_json({"w": Widget(1)})


@dataclasses.dataclass(frozen=True)
class _A:
    x: int = 1


@dataclasses.dataclass(frozen=True)
class _B:
    x: int = 1


def test_two_dataclasses_with_equal_fields_fingerprint_differently():
    assert canonical_json(_A()) != canonical_json(_B())


def test_a_registered_config_is_keyed_by_its_choice_name():
    assert '"__class__": "adam"' in canonical_json(AdamConfig())
    assert canonical_json(MuonConfig()) != canonical_json(GrugMuonConfig())


def test_nested_dataclasses_carry_their_class_at_every_level():
    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: _A

    assert canonical_json(Outer(_A())).count("__class__") == 2
```

In `tests/execution/test_artifact.py`, delete `test_expected_fingerprint_pin_raises_at_lower` and `test_expected_fingerprint_pin_hard_fails_drift`.

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_fingerprint.py -n 0 -q`
Expected: FAIL: the two-dataclass and choice-name tests fail, and `Widget` does not raise.

- [ ] **Step 3: Change the encoder**

In `fingerprint.py`: replace the module docstring's second paragraph with: "The fingerprint decides whether an existing output may be reused, so the encoder is total over the types it knows and raises on any other: a value with no canonical form would make the same config fingerprint differently between processes. :func:`register_fingerprint` teaches it a canonical form for an identity-bearing custom type. A dataclass encodes as its fields plus a ``__class__`` key, so two configs that differ only in class fingerprint differently." Delete `os` from the imports if unused. Add `from draccus.choice_types import ChoiceRegistryBase`, then:

```python
# The key a dataclass's class is recorded under in its canonical form.
CLASS_KEY = "__class__"


def _class_label(cls: type) -> str:
    """A dataclass's class in a fingerprint: its draccus choice name when it is a registered choice
    (``adam``, ``grug_muon``), else its class name. The module is left out, so moving a class does
    not change fingerprints."""
    if issubclass(cls, ChoiceRegistryBase):
        for name, choice in cls.get_known_choices().items():
            if choice is cls:
                return name
    return cls.__qualname__
```

In `_FingerprintEncoder.default`, replace the dataclass branch:

```python
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            # Shallow, so each nested dataclass reaches this method and records its own class.
            return {CLASS_KEY: _class_label(type(o)), **{f.name: getattr(o, f.name) for f in dataclasses.fields(o)}}
```

Replace the tail of `default` after the registered-encoder lookup with the strict branch, unconditionally:

```python
        if inspect.isroutine(o) or isinstance(o, functools.partial):
            raise TypeError(_unfingerprintable(o, "a callable has no stable identity"))
        if type(o).__repr__ is object.__repr__:
            raise TypeError(_unfingerprintable(o, "it uses the default object repr (a memory address)"))
        raise TypeError(_unfingerprintable(o, "no deterministic serialization is known"))
```

Delete `_STRICT_OVERRIDE`, `set_strict`, `_is_strict`, `_fallback`, and `_SEEN_UNFINGERPRINTABLE`. In `_unfingerprintable`, drop ", or disable strict mode" from the message. Update the class docstring: "Canonical handlers run first, then registered converters; any other type raises."

- [ ] **Step 4: Delete `expected_fingerprint`**

In `lazy.py`: delete the `expected_fingerprint` field and its docstring, the `expected_fingerprint` check in `_lower` (and its "Raises :class:`FingerprintMismatchError` if ``expected_fingerprint`` is set" docstring sentence), the `EXPECTED_FINGERPRINT_KEY` lines in `hash_attrs`, and the `EXPECTED_FINGERPRINT_KEY` and `FingerprintMismatchError` imports if unused. In `artifact.py`: delete `EXPECTED_FINGERPRINT_KEY` and the pinned branch of `check_drift` (Task 3 rewrites `check_drift`; leave its warning branch for now).

- [ ] **Step 5: Run the fingerprint, artifact, and full poorin suites**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_fingerprint.py tests/execution/test_artifact.py -n 0 -q`
Expected: pass.

Run: `uv run --no-sync python -m tools.affected_tests`
Expected: pass. A test that now raises `TypeError: cannot fingerprint` had a config with a value that has no canonical form (often a lambda); give it a canonical value or `register_fingerprint` a form. Do not reintroduce a fallback.

- [ ] **Step 6: Check both training roots fingerprint strictly**

```bash
JAX_PLATFORMS=cpu uv run --no-sync python experiments/grug/base/launch.py --version 2026.10.09
JAX_PLATFORMS=cpu uv run --no-sync python experiments/tutorials/train_tiny_model.py --dataset wikitext --version 2026.10.09
```

Expected: each prints a plan and exits 0. No commit yet; Task 3 completes commit 2.

---

### Task 3: Fail-closed drift and `--accept-drift` (completes commit 2)

**Files:**
- Modify: `lib/poorin/src/poorin/execution/artifact.py` (`ArtifactRecord.accepted_from`, `accept_recipe`, `drift_message`, `check_drift`)
- Modify: `lib/poorin/src/poorin/execution/step_runner.py:80-90` (stub records `fingerprint_payload`)
- Modify: `lib/poorin/src/poorin/execution/lazy.py` (`run(accept_drift=...)`, `_refuse_drift`; module docstring)
- Modify: `lib/poorin/src/poorin/experiment/cli.py` (`--accept-drift`, the `--resume` check, `BuildRunner`)
- Test: `tests/execution/test_artifact.py`, `tests/execution/test_adopt.py:42-51`, `tests/experiment/test_cli.py`
- Docs: `docs/explanations/lazy-artifacts.md` ("Advisory drift" and the lines that say "advisory"), `.agents/skills/write-pipeline/references/artifact-binding.md` (the "warns and serves cached output" sentence), the spec's "Pinned and adopted" sentence, `docs/poorin/decisions.md` (D93)

**Interfaces:**
- Consumes: `recorded_recipe`, `RecordedRecipe.filename`, `handle_state`, `record_path`, `CacheState.DRIFTED`, `graph_handles` (Task 1).
- Produces: `accept_recipe(output_path: str, fingerprint: str, payload: str) -> None`; `drift_message(label: str, recorded: str, now: str, change: str) -> str`; `run(*handles, max_concurrent: int = 8, force_run_failed: bool = True, accept_drift: Collection[str] = ()) -> list[T]`; `BuildRunner = Callable[[list[ArtifactStep], int, frozenset[str]], None]`.

- [ ] **Step 1: Write the failing drift tests**

In `tests/execution/test_artifact.py`, replace `test_changed_recipe_warns_and_serves_cached` and add (import `DriftAcceptanceError`, `FingerprintMismatchError`, `read_record`; reuse `_toy` and `V`):

```python
def _consumer(dep: ArtifactStep) -> ArtifactStep[Toy]:
    return ArtifactStep(
        name="checkpoints/consumer",
        version=V,
        artifact_type=Toy,
        run=lambda config: Toy(payload="consumer"),
        build_config=lambda ctx: {"out": ctx.output_path, "data": ctx.artifact_path(dep)},
        deps=(dep,),
    )


def test_a_changed_recipe_is_refused_and_names_the_field(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))
    with pytest.raises(FingerprintMismatchError, match="payload: 'a' -> 'b'"):
        run(_toy(V, "b"))
    assert read_record(f"{tmp_path}/datasets/toy/{V}").result == {"payload": "a"}


def test_a_bumped_dependency_version_is_refused_at_the_built_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_consumer(_toy(V, "a")))
    with pytest.raises(FingerprintMismatchError, match="checkpoints/consumer"):
        run(_consumer(_toy("2026.06.29", "a")))


def test_a_dependency_changed_under_its_version_is_refused_behind_a_built_consumer(tmp_path, monkeypatch):
    # Regression: the built consumer pruned its dependency, so nothing checked it and nothing was logged.
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_consumer(_toy(V, "a")))
    with pytest.raises(FingerprintMismatchError, match="datasets/toy"):
        run(_consumer(_toy(V, "b")))


def test_accepting_drift_serves_the_output_and_records_the_old_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    old = _toy(V, "a")
    run(old)
    new = _toy(V, "b")
    (served,) = run(new, accept_drift={"datasets/toy"})
    assert served.payload == "a"
    record = read_record(f"{tmp_path}/datasets/toy/{V}")
    assert (record.fingerprint, record.accepted_from) == (new.fingerprint(), [old.fingerprint()])
    run(new)  # accepted once, on the record: later runs are clean


def test_accepting_a_drifted_dependency_serves_its_built_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_consumer(_toy(V, "a")))
    run(_consumer(_toy(V, "b")), accept_drift={"datasets/toy"})
    assert read_record(f"{tmp_path}/datasets/toy/{V}").fingerprint == _toy(V, "b").fingerprint()


def test_accepting_an_artifact_that_did_not_drift_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    with pytest.raises(DriftAcceptanceError, match="datasets/typo"):
        run(_toy(V, "a"), accept_drift={"datasets/typo"})


def _crash(config):
    raise RuntimeError("the build crashed")


def test_a_restarted_build_with_a_changed_recipe_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    with pytest.raises(RuntimeError):
        run(dataclasses.replace(_toy(V, "a"), run=_crash))
    with pytest.raises(FingerprintMismatchError, match="payload: 'a' -> 'b'"):
        run(_toy(V, "b"))


def test_a_resumed_dev_run_with_a_changed_recipe_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    version = "20261005-091500-a1b2-dev"
    run(_toy(version, "a"))
    with pytest.raises(FingerprintMismatchError):
        run(_toy(version, "b"))
```

In `tests/execution/test_adopt.py`, replace `test_readopt_different_source_warns` (drop the `logging` import if unused):

```python
def test_readopting_a_different_source_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="a"))
    with pytest.raises(FingerprintMismatchError, match="adopt_source"):
        run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="b"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_artifact.py tests/execution/test_adopt.py -n 0 -q`
Expected: FAIL: drift only warns, and `run` has no `accept_drift`.

- [ ] **Step 3: Record the payload in the start stub**

In `step_runner.py` `_write_executor_info`, add to `info`: `"fingerprint_payload": step.fingerprint_payload,`.

- [ ] **Step 4: Add `accepted_from`, `accept_recipe`, `drift_message`; make `check_drift` fail closed**

In `artifact.py`, add next to `FingerprintMismatchError`:

```python
class DriftAcceptanceError(ValueError):
    """``accept_drift`` named an artifact whose recipe did not change: a typo, or a stale flag."""
```

Add to `ArtifactRecord` after `fingerprint_payload`:

```python
    accepted_from: list[str] = Field(default_factory=list)
    """Earlier fingerprints of this output whose changed recipe the owner accepted (``--accept-drift``)."""
```

Add after `recorded_recipe`:

```python
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
```

Replace `check_drift`:

```python
def check_drift(step: StepSpec) -> bool:
    """The drift guard the runner applies before serving or resuming an output.

    Returns ``False`` for a non-lazy step (no fingerprint). Raises :class:`FingerprintMismatchError`
    when the output at the step's path records a different recipe, in a finished build's record or
    a started build's stub. Otherwise returns whether the version is mutable (``dev``), which always
    rebuilds. :func:`poorin.execution.lazy.run` checks the whole graph before the runner starts; this
    covers code that calls the runner directly.
    """
    fingerprint = step.hash_attrs.get(FINGERPRINT_KEY)
    if fingerprint is None:
        return False
    version = step.hash_attrs.get(VERSION_KEY, "")
    recorded = recorded_recipe(step.output_path)
    if recorded is not None and recorded.fingerprint != fingerprint:
        change = describe_drift(recorded.payload, step.fingerprint_payload)
        raise FingerprintMismatchError(drift_message(f"{step.name}@{version}", recorded.fingerprint, fingerprint, change))
    return is_mutable_version(version)
```

Delete the `logger.warning` drift branch. Update the module docstring bullet: ":func:`check_drift` — the recipe-drift guard the runner applies before serving or resuming an output."

- [ ] **Step 5: Check the whole graph in `lazy.run`**

In `lazy.py`, import `from collections.abc import Callable, Collection, Iterable, Mapping`, `from poorin.execution.cache_state import CacheState, handle_state, record_path`, and `accept_recipe`, `drift_message`, `DriftAcceptanceError`, `FingerprintMismatchError` from `artifact`. Add before `run`:

```python
def _refuse_drift(handles: list["ArtifactStep"], accept_drift: Collection[str]) -> None:
    """Refuse a run if any artifact in the graph records a different recipe, unless accepted.

    Checks every handle, dependencies the runner would prune behind a built consumer included. An
    accepted artifact's recorded recipe is re-stamped, so the runner serves or resumes it.
    """
    drifted = [s for s in map(handle_state, handles) if s.state is CacheState.DRIFTED]
    unknown = sorted(set(accept_drift) - {s.handle.name for s in drifted})
    if unknown:
        raise DriftAcceptanceError(f"accept_drift names artifacts whose recipe did not change: {unknown}")
    refused = [s for s in drifted if s.handle.name not in accept_drift]
    if refused:
        raise FingerprintMismatchError(
            "\n\n".join(drift_message(s.label, s.recorded_fingerprint or "", s.fingerprint, s.change) for s in refused)
        )
    for state in drifted:
        accept_recipe(record_path(state.handle), state.fingerprint, state.handle.fingerprint_payload())
```

Change `run`:

```python
def run(
    *handles: "ArtifactStep[T]",
    max_concurrent: int = 8,
    force_run_failed: bool = True,
    accept_drift: Collection[str] = (),
) -> list[T]:
```

with docstring sentence "Before anything runs, every artifact in the graph is checked against the recipe its output records; ``accept_drift`` names artifacts whose changed recipe is accepted." and first line of the body `_refuse_drift(graph_handles(list(handles)), accept_drift)`. `graph_handles` is defined after `run` in the module; that is fine at call time. In the module docstring, replace "recorded for an advisory drift check" with "recorded, and a changed recipe under an existing output is refused".

- [ ] **Step 6: Run the drift tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution -n 0 -q`
Expected: pass. `test_a_restarted_build_with_a_changed_recipe_is_refused` needs Step 3's stub payload for its field diff.

- [ ] **Step 7: Add `--accept-drift` and the `--resume` check to the CLI**

Write the failing CLI tests first, in `tests/experiment/test_cli.py`:

```python
def test_accept_drift_naming_an_unchanged_artifact_is_rejected(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    result = runner.invoke(
        experiment_main(_build_two), ["--version", "2026.07.16", "--accept-drift", "data/typo", "--run"]
    )
    assert result.exit_code != 0
    assert "data/typo" in result.output


def test_resume_with_nothing_to_resume_is_rejected(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    result = runner.invoke(experiment_main(_build_two), ["--resume", "20261005-091500-a1b2-dev", "--run"])
    assert result.exit_code != 0
    assert "nothing to resume" in result.output
```

The existing `test_resume_reuses_the_given_version_verbatim` prints a plan without `--run`; keep it passing by checking `--resume` only with `--run`.

Then in `cli.py`: change `BuildRunner = Callable[[list[ArtifactStep], int, frozenset[str]], None]` and `_run_locally(handles, max_concurrent, accept_drift)` to call `run(*handles, max_concurrent=max_concurrent, accept_drift=accept_drift)`. Add the option:

```python
    @click.option(
        "--accept-drift",
        "accept_drift",
        multiple=True,
        metavar="NAME",
        help="Keep the output of the artifact NAME although its recipe changed, and record the change. Repeatable.",
    )
```

and in `wrapper` (parameter `accept_drift: tuple[str, ...]`), before `runner(...)`:

```python
        if resume_version is not None:
            returned = {id(h) for h in handles}
            empty = [s.label for s in states if id(s.handle) in returned and s.recorded_fingerprint is None]
            if empty:
                raise click.UsageError(
                    f"nothing to resume at {empty}: no run started there. Check that the --set flags match the original run's."
                )
        try:
            runner(handles, max_concurrent, frozenset(accept_drift))
        except (FingerprintMismatchError, DriftAcceptanceError) as e:
            raise click.ClickException(str(e)) from e
```

(Import both from `poorin.execution.artifact`.)

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_cli.py tests/execution -n 0 -q`
Expected: pass.

- [ ] **Step 8: Update the docs and record the ruling**

- `docs/explanations/lazy-artifacts.md`: rename "## Advisory drift" to "## Recipe drift" and rewrite its body: each computed artifact records a fingerprint; when an existing output (a finished build, or a started one) records a different fingerprint, the run is refused before anything runs, listing each changed field; every artifact in the graph is checked, including dependencies behind a built consumer; to build the new recipe, change the version; to keep the output, pass `--accept-drift NAME` (or `run(..., accept_drift={...})`), which records the new fingerprint and keeps the old one in `accepted_from`. Fix the other "advisory" mentions (around lines 35, 192, 233) to match, and the cache-check bullet: "if the output records a different recipe, the run is refused".
- `.agents/skills/write-pipeline/references/artifact-binding.md`: replace "A fixed `name@version` with changed config warns and serves cached output; bump the version to build changed content." with "A fixed `name@version` whose config changed is refused before anything runs; change the version to build the new content, or pass `--accept-drift NAME` to keep the existing output."
- The spec: replace "Pinned and adopted artifacts aren't checked, because they point at existing data." with "Pinned artifacts (`override_path`) are not checked, because they point at existing data. An adopted artifact is checked: its record at the canonical address names its source, so re-adopting from another source is refused."
- `docs/poorin/decisions.md`: add after D92:

```markdown
### D93. Rulings while building D92

- **When:** 2026-10-08, building D92. **Status:** Adopted.
- **Adopted artifacts are drift-checked.** The spec exempted them with pins, but an adopted
  artifact writes a record at its canonical address, and re-adopting the same `name@version` from
  another source must be refused, as it was warned about before.
```

- [ ] **Step 9: Absorb the fingerprint change in prune_check's cache**

Task 2 changed every fingerprint, so the two calendar-versioned datasets in prune_check's prefix now drift. Accept them once, through the new flag:

```bash
POORIN_PREFIX=~/.cache/poorin/prune-check WANDB_MODE=offline uv run --no-sync python experiments/tutorials/train_tiny_model.py --device rtx5090x1 --dataset wikitext --version dev
```

Expected: the plan lists each calendar-versioned artifact under that prefix (at least `wikitext@2026.06.28`) as `built from another recipe`, with diffs that only add `__class__` keys. Do the same for the grug root (`experiments/grug/base/launch.py --version dev`, at least `fineweb-edu-10M@2026.06.28`). Then rerun each command with one `--accept-drift NAME` per drifted artifact and `--run`. If any diff shows a change other than added `__class__` keys, stop and report it.

- [ ] **Step 10: Lint, full check, and commit**

Run `uv run --no-sync python -m tools.lint --fix`, then `uv run --no-sync python -m tools.prune_check --phase Overrides --step "Refuse a changed recipe under an existing output"`. Expected: exit 0.

```bash
git add -A lib/poorin/src/poorin/execution tests/execution tests/experiment/test_cli.py lib/poorin/src/poorin/experiment/cli.py docs/explanations/lazy-artifacts.md .agents/skills/write-pipeline/references/artifact-binding.md docs/poorin/specs/2026-10-08-experiment-overrides-design.md docs/poorin/decisions.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Refuse a changed recipe under an existing output" -m "<body: the four stale cases (config change, dependency bump, dependency recipe changed under its version, class swap), checked over the whole graph before a run; the class enters the fingerprint, fingerprinting is strict, expected_fingerprint is gone; --accept-drift re-stamps a record; prune_check's two cached datasets were accepted once. Decisions D92 and D93.>"
```

---

### Task 4: Tracker settings leave identity (commit 3)

**Files:**
- Modify: `experiments/grug/base/launch.py:76-123` (`@no_type_check`, `_trackers`), `:193-246` (`grug_base_trial`)
- Modify: `lib/poorin/src/poorin/experiment/train.py:160-247`
- Test: `tests/test_grug_base_launch.py`, `tests/experiment/test_train_lm.py`
- Docs: `docs/poorin/decisions.md` (D93 bullet)

**Interfaces:**
- Produces: grug's runtime args gain `"run_id"`, `"tracker"`, `"log_every"`; `train_lm`'s gain `"run_id"`, `"tracker"`. A tracker's `replicate_path` is set from `ctx.output_path` at run time.

- [ ] **Step 1: Write the failing tests**

In `tests/test_grug_base_launch.py` (it runs under the root suite's beartype hook, so it also proves `GrugBaseLaunchConfig` accepts placeholders):

```python
import dataclasses

from experiments.grug.base.launch import grug_base_trial
from poorin.execution.artifact import ArtifactRecord, write_record
from poorin.execution.lazy import materialized_config
from poorin.processing.tokenize.tokenize import TokenizedCache


def test_tracker_settings_do_not_change_the_trials_fingerprint():
    trial = grug_base_trial(version="2026.10.09")
    retagged = dataclasses.replace(
        trial,
        runtime_args={**trial.runtime_args, "tracker": WandbConfig(project="other", tags=["x"]), "log_every": 10},
    )
    assert retagged.fingerprint() == trial.fingerprint()


def test_the_trials_tracker_mirrors_its_metrics_next_to_the_output(tmp_path):
    trial = grug_base_trial(version="2026.10.09")
    # The minimal record a built dataset leaves, so mixture() can read its tokenizer offline.
    for dep in trial.deps:
        write_record(
            ArtifactRecord(
                name=dep.name,
                version=dep.version,
                output_path=dep.path(str(tmp_path)),
                result_type=f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}",
                config={"tokenizer": "marin-community/marin-tokenizer", "format": {"text_key": "text"}},
            )
        )
    config = materialized_config(trial, str(tmp_path))
    assert config.tracker.replicate_path == trial.path(str(tmp_path))
```

In `tests/experiment/test_train_lm.py`:

```python
def test_tracker_settings_stay_out_of_the_fingerprint(tmp_path):
    corpus = _corpus()

    def tagged(tags):
        return train_lm(
            name="checkpoints/unit", model=_MODEL, optimizer=_OPTIMIZER, datasets={corpus: 1.0},
            batch_size=8, seq_len=128, num_train_steps=50, z_loss_weight=1e-4, evals=None,
            resources=_RESOURCES, run_id="unit", tags=tags, version=_V,
        )

    assert tagged(["a"]).fingerprint() == tagged(["b"]).fingerprint()
    tracker = _assemble(tagged(["a"]), str(tmp_path)).train_config.trainer.tracker
    assert (tracker.tags, tracker.replicate_path) == (["a"], f"{tmp_path}/checkpoints/unit/{_V}")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_grug_base_launch.py tests/experiment/test_train_lm.py -n 0 -q`
Expected: FAIL: the grug fingerprint test raises a beartype violation on `resources="<train_resources>"`, and the tag test sees different fingerprints.

- [ ] **Step 3: Move grug's tracker settings into runtime args**

In `launch.py`: import `from typing import no_type_check`. Decorate `GrugBaseLaunchConfig` with `@no_type_check` above `@dataclass(frozen=True)`, with the comment "# Fields pulled from the step context hold placeholder strings such as "<train_resources>" while the fingerprint is computed, which the tests' beartype hook would reject." In `build_config`, replace the `run_id`, `tracker`, and `log_every` arguments:

```python
            run_id=ctx.runtime_arg("run_id"),
            tracker=_run_tracker(ctx),
            log_every=ctx.runtime_arg("log_every"),
```

and add above `grug_base_trial`:

```python
def _run_tracker(ctx: StepContext) -> TrackerConfig:
    """The W&B tracker from the runtime args, mirroring its metrics next to the run's output."""
    tracker = ctx.runtime_arg("tracker")
    return tracker if ctx.is_fingerprint else dataclasses.replace(tracker, replicate_path=ctx.output_path)
```

In the `ArtifactStep(...)` call, extend `runtime_args`:

```python
        runtime_args={
            "train_resources": _TRAIN_RESOURCES,
            "finelog_url": finelog_url,
            "run_id": run_id,
            "tracker": WandbConfig(project="poorin", tags=["grug", "template"], group="grug-base-trial", name=None),
            "log_every": 1,
        },
```

Keep `run_id = f"grug-base-{version}"` where it is.

- [ ] **Step 4: Move `train_lm`'s tracker settings into runtime args**

In `train.py` `build_config`, replace the `id=tracker_id,` and `tracker=WandbConfig(...)` arguments of `TrainerConfig` with `id=ctx.runtime_arg("run_id"),` and `tracker=_run_tracker(ctx),`, add the same `_run_tracker` helper above `train_lm` (with `import dataclasses` and `TrackerConfig` imports), and extend `runtime_args`:

```python
        runtime_args={
            "train_resources": resources,
            "run_id": tracker_id,
            "tracker": WandbConfig(project=wandb_project, name=tracker_id, tags=[*tags], group=wandb_group),
        },
```

Update the docstring's execution-choices paragraph: "tracker metadata (project, tags, group, run ID) and ``resources`` are runtime args, so they never enter the checkpoint's fingerprint".

- [ ] **Step 5: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_grug_base_launch.py tests/experiment/test_train_lm.py -n 0 -q`
Expected: pass, including the existing `test_mutable_versions_give_each_run_its_own_tracker_id`.

- [ ] **Step 6: Record the ruling, check, commit**

Append to D93:

```markdown
- **`GrugBaseLaunchConfig` is exempt from the tests' beartype hook** (`@no_type_check`, as
  `GrugTrainState` is): while the fingerprint is computed, its runtime fields hold placeholder
  strings. Computing grug's fingerprint under the hook failed on `resources` before this change.
```

Run `uv run --no-sync python -m tools.lint --fix` and `uv run --no-sync python -m tools.prune_check --phase Overrides --step "Keep tracker settings out of a run's identity"`. Expected: exit 0.

```bash
git add experiments/grug/base/launch.py lib/poorin/src/poorin/experiment/train.py tests/test_grug_base_launch.py tests/experiment/test_train_lm.py docs/poorin/decisions.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Keep tracker settings out of a run's identity" -m "<body: W&B project, tags, group, run ID, and grug's log cadence become runtime args, so editing a tag no longer changes the fingerprint or trips the drift check; replicate_path is set at run time. GrugBaseLaunchConfig is exempt from the tests' beartype hook. Decisions D92 and D93.>"
```

---

### Task 5: The overrides module (part of commit 4)

**Files:**
- Create: `lib/poorin/src/poorin/experiment/overrides.py`
- Modify: `lib/poorin/src/poorin/execution/lazy.py` (`StepContext.is_context_value`)
- Test: `tests/experiment/test_overrides.py` (create)

**Interfaces:**
- Consumes: `StepContext` (`for_fingerprint`, `for_run`), `canonical_json`, `fingerprint_hash`.
- Produces: `OverrideError(ValueError)`; `Override(path: tuple[str, ...], text: str)` with `Override.parse(item: str) -> Override` and `label -> str`; `Overrides(items: tuple[Override, ...] = ())` with `apply(ctx: StepContext, config: T) -> T`, `named(name: str) -> str`, `run_id(base: str) -> str`, `unapplied() -> list[str]`, `digest -> str`; `StepContext.is_context_value(value: object) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment/test_overrides.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed --set overrides: decoding, refusals, and the names they give a trial."""

import dataclasses

import pytest
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.optim.muon import MuonConfig
from poorin.execution.lazy import StepContext
from poorin.experiment.overrides import Override, OverrideError, Overrides


@dataclasses.dataclass(frozen=True)
class Trainer:
    steps: int
    tags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Config:
    trainer: Trainer
    optimizer: OptimizerConfig
    output_path: str
    ema_beta: float | None = None
    eval_batch_size: int | None = 512
    mp: str = "p=f32,c=bfloat16"


FINGERPRINT = StepContext.for_fingerprint()


def _config(ctx: StepContext = FINGERPRINT) -> Config:
    return Config(trainer=Trainer(steps=50), optimizer=AdamConfig(learning_rate=3e-3), output_path=ctx.output_path)


def _set(*items: str) -> Overrides:
    return Overrides(tuple(Override.parse(item) for item in items))


def test_a_nested_field_is_decoded_to_its_type():
    config = _set("trainer.steps=200", "optimizer.learning_rate=1e-3").apply(FINGERPRINT, _config())
    assert (config.trainer.steps, config.optimizer.learning_rate) == (200, 0.001)


def test_a_registry_field_swaps_its_class():
    config = _set("optimizer={type: muon, learning_rate: 0.02}").apply(FINGERPRINT, _config())
    assert isinstance(config.optimizer, MuonConfig)
    assert config.optimizer.learning_rate == 0.02


def test_optional_fields_decode_through_their_union():
    config = _set("ema_beta=0.999", "eval_batch_size=null").apply(FINGERPRINT, _config())
    assert (config.ema_beta, config.eval_batch_size) == (0.999, None)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("trainer.stepz=1", "did you mean steps"),
        ("trainer.steps=2.5", "trainer.steps"),
        ("output_path=/tmp/x", "step context"),
        ("trainer.tags.first=x", "not a dataclass"),
    ],
)
def test_a_bad_override_is_refused_with_its_path(item, message):
    with pytest.raises(OverrideError, match=message):
        _set(item).apply(FINGERPRINT, _config())


def test_a_path_set_twice_is_refused():
    with pytest.raises(OverrideError, match="trainer.steps"):
        _set("trainer.steps=1", "trainer.steps=2")


def test_unapplied_overrides_are_reported():
    overrides = _set("trainer.steps=1")
    assert overrides.unapplied() == ["trainer.steps"]
    overrides.apply(FINGERPRINT, _config())
    assert overrides.unapplied() == []


def test_a_trial_is_named_by_its_overrides_with_numbers_in_one_spelling():
    assert _set("trainer.steps=200", "optimizer.learning_rate=1e-3").named("grug/base") == (
        "grug/base/optimizer.learning_rate=0.001,trainer.steps=200"
    )
    assert _set("optimizer.learning_rate=0.001").named("x") == _set("optimizer.learning_rate=1e-3").named("x")
    assert Overrides().named("grug/base") == "grug/base"


def test_a_value_with_a_comma_or_equals_takes_the_digest_name():
    # "mp=a=b,c=d" could otherwise read like two overrides another set might spell the same way.
    name = _set("mp=params=float32,compute=bfloat16").named("grug/base")
    assert name.startswith("grug/base/set-")
    assert _set("mp=params=float32,compute=bfloat16").items[0].text == "params=float32,compute=bfloat16"


def test_each_trial_gets_its_own_run_id():
    assert _set("trainer.steps=1").run_id("grug") != _set("trainer.steps=2").run_id("grug")
    assert Overrides().run_id("grug") == "grug"


def test_a_path_must_be_dotted_identifiers():
    with pytest.raises(OverrideError):
        Override.parse("trainer/steps=1")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_overrides.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poorin.experiment.overrides'`.

- [ ] **Step 3: Add `StepContext.is_context_value`**

In `lazy.py`, add to `StepContext` after `runtime_arg`:

```python
    def is_context_value(self, value: object) -> bool:
        """Whether ``value`` is what this fingerprint context put in place of a pulled value or a
        dependency reference: the parts of a config an override must not replace. Always ``False``
        at run time, where the fingerprint pass has already refused them."""
        if not self.is_fingerprint or not isinstance(value, str):
            return False
        placeholders = {self.output_path, self.prefix, self.region, *self._runtime_args.values()}
        return value in placeholders or value in {self._dep_ref(dep) for dep in self._deps}
```

- [ ] **Step 4: Create `overrides.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed dotted-path overrides of an experiment's config, from ``--set`` and ``--sweep``.

``poorin.experiment.cli`` parses the command line into one :class:`Overrides` per trial and passes
it to the experiment, which applies it inside ``build_config`` (so the fingerprint sees it), names
its artifact with :meth:`Overrides.named`, and derives its tracker run ID with
:meth:`Overrides.run_id`. A value is YAML decoded by draccus against the field's annotated type, so
union fields and draccus registries (``optimizer={type: muon}``) decode as they do in Levanter's
configs.
"""

import dataclasses
import difflib
import re
import typing
from dataclasses import dataclass, field
from typing import Any, TypeVar

import draccus
import levanter.config  # noqa: F401  # registers Levanter's draccus codecs (jmp policies, timedeltas, dtypes)
from draccus.cfgparsing import parse_string
from draccus.utils import DecodingError

from poorin.execution.fingerprint import canonical_json, fingerprint_hash
from poorin.execution.lazy import StepContext

T = TypeVar("T")

# A value that may appear in an artifact's name as written; any other value names the trial by digest.
_PLAIN_VALUE = re.compile(r"^[A-Za-z0-9_.+-]{1,40}$")
_MAX_READABLE_SUFFIX = 100


class OverrideError(ValueError):
    """A ``--set`` or ``--sweep`` value that cannot apply to the experiment's config."""


def _normalized(text: str) -> str:
    """``text`` in one spelling when it is a number, so ``1e-3`` and ``0.001`` name the same artifact."""
    for parse in (int, float):
        try:
            return repr(parse(text))
        except ValueError:
            continue
    return text.strip()


@dataclass(frozen=True)
class Override:
    """One ``PATH=VALUE``: a dotted path into the config and its YAML value as typed."""

    path: tuple[str, ...]
    text: str

    @classmethod
    def parse(cls, item: str) -> "Override":
        """Split ``PATH=VALUE`` at the first ``=``; the path is dotted identifiers."""
        key, sep, text = item.partition("=")
        path = tuple(key.split("."))
        if not sep or not all(segment.isidentifier() for segment in path):
            raise OverrideError(f"{item!r} is not PATH=VALUE with a dotted path of field names")
        return cls(path, text)

    @property
    def label(self) -> str:
        return f"{'.'.join(self.path)}={_normalized(self.text)}"


@dataclass(frozen=True)
class Overrides:
    """One trial's field overrides. ``apply`` runs inside ``build_config``, so the fingerprint sees them."""

    items: tuple[Override, ...] = ()
    # Paths apply() consumed, for the unapplied-override check. Instrumentation only: mutating the
    # set on a frozen instance is allowed (compare=False, init=False).
    _applied: set[tuple[str, ...]] = field(default_factory=set, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        paths = [item.path for item in self.items]
        twice = sorted({".".join(path) for path in paths if paths.count(path) > 1})
        if twice:
            raise OverrideError(f"set twice in one trial: {twice}")

    def apply(self, ctx: StepContext, config: T) -> T:
        """``config`` with every override decoded into place."""
        for item in self.items:
            config = _replace_at(ctx, config, item, item.path)
            self._applied.add(item.path)
        return config

    def unapplied(self) -> list[str]:
        """Paths ``apply`` never consumed: the experiment ignored its overrides."""
        return sorted(".".join(item.path) for item in self.items if item.path not in self._applied)

    @property
    def digest(self) -> str:
        return fingerprint_hash(canonical_json(sorted(item.label for item in self.items)))

    def named(self, name: str) -> str:
        """``name``, or ``name/<overrides>``: readable when every value is plain, else ``set-<digest>``."""
        if not self.items:
            return name
        readable = ",".join(sorted(item.label for item in self.items))
        plain = all(_PLAIN_VALUE.match(_normalized(item.text)) for item in self.items)
        if plain and len(readable) <= _MAX_READABLE_SUFFIX:
            return f"{name}/{readable}"
        return f"{name}/set-{self.digest}"

    def run_id(self, base: str) -> str:
        """``base``, or ``base-<digest>``, so each trial logs to its own tracker run."""
        return base if not self.items else f"{base}-{self.digest}"


def _replace_at(ctx: StepContext, obj: Any, item: Override, path: tuple[str, ...]) -> Any:
    """``obj`` with ``item``'s value at ``path`` (the rest of ``item.path`` below ``obj``)."""
    full = ".".join(item.path)
    head, *rest = path
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise OverrideError(f"{full}: reaches a {type(obj).__name__}, which is not a dataclass with fields")
    names = [f.name for f in dataclasses.fields(obj)]
    if head not in names:
        close = difflib.get_close_matches(head, names, n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise OverrideError(f"{full}: {type(obj).__name__} has no field {head!r}{hint}")
    current = getattr(obj, head)
    if ctx.is_context_value(current):
        raise OverrideError(f"{full}: its value comes from the step context ({current}), so it is not part of the recipe")
    if rest:
        return dataclasses.replace(obj, **{head: _replace_at(ctx, current, item, tuple(rest))})
    field_type = typing.get_type_hints(type(obj))[head]
    try:
        value = draccus.decode(field_type, parse_string(item.text))
    except DecodingError as e:
        raise OverrideError(f"{full}={item.text}: {e}") from e
    return dataclasses.replace(obj, **{head: value})
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_overrides.py -n 0 -q`
Expected: pass. If `trainer.tags.first=x` raises before reaching `_replace_at`'s dataclass check, the message must still name the path; adjust the check order, not the test.

No commit: Task 6 completes commit 4.

---

### Task 6: `--set`, `--sweep`, `--version-of`, and adoption (completes commit 4)

**Files:**
- Modify: `lib/poorin/src/poorin/experiment/cli.py`
- Modify: `lib/poorin/src/poorin/execution/build_context.py` (`overrides` → `by_name`, docstrings)
- Modify: `lib/poorin/src/poorin/experiment/train.py` (`overrides` parameter)
- Modify: `experiments/grug/base/launch.py` (`grug_base_trial`, `main`)
- Modify: `experiments/tutorials/train_tiny_model.py` (`build`, `main`, docstring)
- Modify: `lib/poorin/src/poorin/experiment/sweep.py` (docstring)
- Test: `tests/experiment/test_cli.py`, `tests/execution/test_build_context.py`, `tests/experiment/test_train_lm.py`
- Docs: `experiments/grug/README.md`, `lib/poorin/README.md`, `docs/explanations/lazy-artifacts.md`, `docs/poorin/architecture.md`, `docs/poorin/decisions.md`

**Interfaces:**
- Consumes: `Override`, `Overrides`, `OverrideError` (Task 5); `handle_state`, `graph_handles` (Task 1); `run(accept_drift=...)` (Task 3); `poorin.experiment.sweep.grid`.
- Produces: every function wrapped by `build_options` is called with `overrides: Overrides`; `experiment_main(build)` calls `build(overrides=...)`; `train_lm(..., overrides: Overrides = Overrides())`; `grug_base_trial(*, overrides: Overrides, version: str | None = None, finelog_url: str | None = None)`; `VersionCodex(default, by_name)`.

- [ ] **Step 1: Write the failing CLI tests**

In `tests/experiment/test_cli.py`, rename `--override` to `--version-of` in the existing tests (and the test names that say "override"), and add a toy experiment that applies its overrides:

```python
@dataclasses.dataclass(frozen=True)
class _Knobs:
    out: str
    lr: float
    steps: int
    optimizer: OptimizerConfig


def _knob_trial(*, overrides: Overrides) -> ArtifactStep:
    """A toy experiment that applies its overrides, as grug and train_lm do."""
    return ArtifactStep(
        name=overrides.named("models/knobs"),
        version=resolve_version("models/knobs", None),
        artifact_type=Artifact,
        run=lambda config: _write_marker(config.out),
        build_config=lambda ctx: overrides.apply(
            ctx, _Knobs(out=ctx.output_path, lr=3e-3, steps=50, optimizer=AdamConfig())
        ),
    )
```

Then:

```python
def test_set_lands_at_the_override_named_path_and_records_the_value(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    args = ["--version", "2026.07.16", "--set", "lr=1e-3", "--set", "steps=200", "--run"]
    result = runner.invoke(experiment_main(_knob_trial), args)
    assert result.exit_code == 0, result.output
    record = read_record(f"{tmp_path}/models/knobs/lr=0.001,steps=200/2026.07.16")
    assert (record.config["lr"], record.config["steps"]) == (0.001, 200)


def test_a_rerun_sweep_runs_only_its_missing_trials(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))

    def built():
        return sorted(p.parent.parent.name for p in tmp_path.glob("models/knobs/*/2026.07.16/done"))

    base = ["--version", "2026.07.16", "--max-concurrent", "1", "--run"]
    runner.invoke(experiment_main(_knob_trial), [*base, "--sweep", "lr=[1e-3, 3e-3]"])
    assert built() == ["lr=0.001", "lr=0.003"]
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.glob("models/knobs/*/2026.07.16/done")}
    runner.invoke(experiment_main(_knob_trial), [*base, "--sweep", "lr=[1e-3, 3e-3, 1e-2]"])
    assert built() == ["lr=0.001", "lr=0.003", "lr=0.01"]
    assert all(p.stat().st_mtime_ns == t for p, t in stamps.items())  # finished trials were not rebuilt


def test_a_sweep_over_sub_configs_names_each_trial_by_digest(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    args = ["--version", "2026.07.16", "--max-concurrent", "1", "--sweep", "optimizer=[{type: adam}, {type: muon}]"]
    result = runner.invoke(experiment_main(_knob_trial), args)
    assert result.exit_code == 0, result.output
    names = {line.split()[-1] for line in result.output.splitlines() if "models/knobs/set-" in line}
    assert len(names) == 2


def test_a_sweep_needs_an_explicit_max_concurrent(runner):
    result = runner.invoke(experiment_main(_knob_trial), ["--version", "dev", "--sweep", "lr=[1e-3, 3e-3]"])
    assert result.exit_code != 0
    assert "--max-concurrent" in result.output


def test_an_override_the_experiment_ignores_is_rejected(runner):
    result = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--set", "lr=1e-3"])
    assert result.exit_code != 0
    assert "never applied" in result.output


def test_a_bad_override_fails_before_anything_runs(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    result = runner.invoke(experiment_main(_knob_trial), ["--version", "dev", "--set", "stepz=1", "--run"])
    assert result.exit_code != 0
    assert "did you mean steps" in result.output
    assert not list(tmp_path.rglob("done"))
```

Imports: `dataclasses`, `Artifact`, `read_record` from `poorin.execution.artifact`, `ArtifactStep` from `poorin.execution.lazy`, `resolve_version` from `poorin.execution.build_context`, `Overrides` from `poorin.experiment.overrides`, `AdamConfig`, `OptimizerConfig` from `levanter.optim.config`.

In `tests/execution/test_build_context.py`, rename `overrides=` to `by_name=` and `unused_overrides` to `unused_names`.

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_cli.py tests/execution/test_build_context.py -n 0 -q`
Expected: FAIL: `--set`, `--sweep`, and `--version-of` are unknown options.

- [ ] **Step 3: Rename the codex field**

In `build_context.py`: `VersionCodex.overrides` → `by_name`, `unused_overrides()` → `unused_names()`, the error prefix `override {name!r}` → `{name!r}`, and the docstrings ("per-name ``overrides``" → "per-name versions in ``by_name``", "``--override``" → "``--version-of``").

- [ ] **Step 4: Add the options and the trial loop**

In `cli.py`: import `from poorin.experiment.overrides import Override, OverrideError, Overrides` and `from poorin.experiment.sweep import grid`; add `DEFAULT_MAX_CONCURRENT = 8`. Rename `_override_callback` to `_version_of_callback` (messages: "is not NAME=VERSION", "duplicate --version-of for", "--version-of {name!r}: {e}"). Add callbacks:

```python
def _set_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> list[Override]:
    try:
        return [Override.parse(item) for item in value]
    except OverrideError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e


def _sweep_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> dict[str, list[str]]:
    """Parse repeated ``PATH=[V1, V2, ...]`` into ``{path: [value text, ...]}``.

    A list element that YAML reads as a string keeps its text, so ``1e-3`` is decoded later like a
    ``--set`` value; any other element (a number, a mapping) is written back as JSON, which YAML reads.
    """
    axes: dict[str, list[str]] = {}
    for item in value:
        path, sep, text = item.partition("=")
        values = parse_string(text) if sep else None
        if not isinstance(values, list) or not values:
            raise click.BadParameter(f"{item!r} is not PATH=[V1, V2, ...]", ctx=ctx, param=param)
        if path in axes:
            raise click.BadParameter(f"{path!r} is swept twice", ctx=ctx, param=param)
        axes[path] = [v if isinstance(v, str) else json.dumps(v) for v in values]
    return axes
```

(`from draccus.cfgparsing import parse_string`.) Replace the `--override` option with:

```python
    @click.option(
        "--version-of",
        "versions_of",
        multiple=True,
        metavar="NAME=VERSION",
        callback=_version_of_callback,
        help="The version of one artifact, by the name its builder passes. Repeatable.",
    )
    @click.option(
        "--set",
        "sets",
        multiple=True,
        metavar="PATH=VALUE",
        callback=_set_callback,
        help="Override one field of the experiment's config by dotted path; VALUE is YAML. Repeatable.",
    )
    @click.option(
        "--sweep",
        "sweeps",
        multiple=True,
        metavar="PATH=[V1, ...]",
        callback=_sweep_callback,
        help="One trial per value, as if --set PATH=Vi; several form a grid. Repeatable.",
    )
```

and change `--max-concurrent` to `default=None` with help "Max steps built concurrently (default 8; required when --sweep builds more than one trial)." In `wrapper` (parameters `versions_of: dict[str, str]`, `sets: list[Override]`, `sweeps: dict[str, list[str]]`, `max_concurrent: int | None`), replace the build and checks:

```python
        points = grid(**sweeps) if sweeps else [{}]
        if len(points) > 1 and max_concurrent is None:
            raise click.UsageError(
                f"--sweep builds {len(points)} trials; pass --max-concurrent (1 runs them one after another on one GPU)"
            )
        codex = VersionCodex(default=version, by_name=versions_of)
        try:
            trials = [
                Overrides((*sets, *(Override.parse(f"{path}={text}") for path, text in point.items())))
                for point in points
            ]
            with build_context(BuildContext(versions=codex)):
                handles = [h for trial in trials for h in _as_handles(fn(*args, overrides=trial, **kwargs))]
            states = [handle_state(h) for h in graph_handles(handles)]  # applies every override
        except OverrideError as e:
            raise click.UsageError(str(e)) from e
        unused = codex.unused_names()
        if unused:
            raise click.BadParameter(
                f"never applied to any artifact: {sorted(unused)}. The name is a typo, or that "
                "artifact hardcodes its version (an explicit version wins).",
                param_hint="--version-of",
            )
        unapplied = sorted({path for trial in trials for path in trial.unapplied()})
        if unapplied:
            raise click.BadParameter(
                f"never applied: {unapplied}. This experiment does not apply its overrides to its config.",
                param_hint="--set/--sweep",
            )
```

followed by the existing `--show-config`, plan, `--resume`, and run code, with `runner(handles, max_concurrent or DEFAULT_MAX_CONCURRENT, frozenset(accept_drift))`. Change `experiment_main`'s docstring: "``build`` takes an ``overrides`` keyword, which it applies to its config." Update the module docstring's option list (`--version-of`, `--set`, `--sweep`, `--show-config`, `--accept-drift`) and its example (`def main(device, overrides): return build(device=device, overrides=overrides)`).

- [ ] **Step 5: Run the CLI tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_cli.py tests/execution/test_build_context.py -n 0 -q`
Expected: pass.

- [ ] **Step 6: Adopt overrides in `train_lm`, grug, and the tutorial**

`train.py`: add `overrides: Overrides = Overrides()` after `env_vars` (import from `poorin.experiment.overrides`); wrap the inner config, `inner = overrides.apply(ctx, TrainLmConfig(...))`; name the step `user_namespaced_name(overrides.named(name), version)`; derive the tracker ID per trial:

```python
    if overrides.items and run_id is None:
        raise ValueError("train_lm needs a run_id when overrides are set, so each trial logs to its own tracker run")
    base_id = f"{run_id}-{version}" if run_id is not None and is_mutable_version(version) else run_id
    tracker_id = overrides.run_id(base_id) if base_id is not None else None
```

Add to `tests/experiment/test_train_lm.py`:

```python
def test_overrides_reach_the_inner_config_and_name_the_trial(tmp_path):
    step = train_lm(
        name="checkpoints/unit", model=_MODEL, optimizer=_OPTIMIZER, datasets={_corpus(): 1.0},
        batch_size=8, seq_len=128, num_train_steps=50, z_loss_weight=1e-4, evals=None,
        resources=_RESOURCES, run_id="unit", version=_V,
        overrides=Overrides((Override.parse("trainer.num_train_steps=7"),)),
    )
    assert step.name == "checkpoints/unit/trainer.num_train_steps=7"
    assert _assemble(step, str(tmp_path)).train_config.trainer.num_train_steps == 7
```

`launch.py`: `grug_base_trial(*, overrides: Overrides, version: str | None = None, finelog_url: str | None = None)`; `base_name = "grug/base-trial"`, `version = resolve_version(base_name, version)`, `run_id = overrides.run_id(f"grug-base-{version}")`, `build_config` returns `overrides.apply(ctx, GrugBaseLaunchConfig(...))`, and `name=user_namespaced_name(overrides.named(base_name), version)`. `main(finelog_url, overrides)` returns `grug_base_trial(finelog_url=finelog_url, overrides=overrides)`. Update Task 4's grug tests to pass `overrides=Overrides()`, and add:

```python
def test_a_set_reaches_grugs_config_and_names_the_trial():
    trial = grug_base_trial(version="2026.10.09", overrides=Overrides((Override.parse("steps=200"),)))
    assert trial.name == "grug/base-trial/steps=200"
    assert '"steps": 200' in trial.fingerprint_payload()
```

`train_tiny_model.py`: `build(*, device, data, overrides)` passes `overrides=overrides` to `train_lm`; `main(device, data, overrides)`. Add a docstring example line: `python -m experiments.tutorials.train_tiny_model --dataset wikitext --version dev --set optimizer.learning_rate=1e-3 --run`.

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_grug_base_launch.py tests/experiment -n 0 -q`
Expected: pass.

- [ ] **Step 7: Docs, architecture rows, ruling**

- `experiments/grug/README.md`: in the quickstart add `--set` and `--sweep` examples (the three commands from the spec's "Command-line surface", with `uv run --no-sync`); where it says to copy `base/` into a variant to change knobs, say that `--set` covers field changes and a variant is for code changes.
- `lib/poorin/README.md` line 7: the CLI options become `--version`, `--resume`, `--version-of`, `--set`, `--sweep`, `--show-config`, `--accept-drift`, `--run`.
- `docs/explanations/lazy-artifacts.md`: the main-block options list (`--override` → `--version-of`, add `--set`, `--sweep`, `--show-config`, `--accept-drift`), and a short "Overrides and sweeps" section: what `--set`/`--sweep` do, that overrides enter the fingerprint and the name, the digest rule, and the derived-value caveat.
- `lib/poorin/src/poorin/experiment/sweep.py` module docstring: add "For a grid over config fields from the command line, use ``--sweep`` (:mod:`poorin.experiment.cli`)."
- `docs/poorin/architecture.md`: add `| \`poorin.experiment.overrides\` | Typed dotted-path config overrides from --set and --sweep, decoded by draccus | R1 |` after `poorin.experiment.namespacing`; change the `poorin.experiment.cli` row's purpose to "Click options (`--version`, `--version-of`, `--set`, `--sweep`, `--run`) that plan or build an experiment graph"; change the `poorin.experiment.sweep` row's "Reached from" to `R1` (the CLI imports it now).
- D93: append "- **`train_lm` needs a `run_id` when overrides are set**, so each trial logs to its own tracker run; without one, Levanter names the run after the version, which every trial shares."

- [ ] **Step 8: Lint, full check, commit**

Run `uv run --no-sync python -m tools.lint --fix` and `uv run --no-sync python -m tools.prune_check --phase Overrides --step "Add --set and --sweep to experiments"`. Expected: exit 0.

```bash
git add -A lib/poorin experiments/grug experiments/tutorials tests docs .agents
git commit -m "[poorin] Add --set and --sweep to experiments" -m "<body: --set PATH=VALUE overrides any config field, decoded by draccus against its type; --sweep runs one trial per value; each override enters the trial's fingerprint and name, and its tracker run ID; --override is renamed --version-of; grug, the tutorial, and train_lm apply their overrides. Decisions D92 and D93.>"
```

---

### Task 7: GPU check and close-out

**Files:**
- Modify: `docs/poorin/decisions.md` (D92 status, Deferred rows), the spec's status line

- [ ] **Step 1: Run a two-trial grug sweep on the GPU**

```bash
WANDB_MODE=offline uv run --no-sync python experiments/grug/base/launch.py --version dev --sweep 'steps=[2, 3]' --max-concurrent 1 --run
```

Expected: exit 0; two checkpoints under `$POORIN_PREFIX/users/<you>/grug/base-trial/steps=2/` and `.../steps=3/` (default prefix `/tmp/poorin`), each with an `.artifact.json` whose `config.steps` is 2 and 3; two W&B offline run directories under `wandb/` with different run IDs (`grug-base-<version>-<digest>`). If the second training fails in the same process, stop and report the traceback; do not work around it in the sweep code.

- [ ] **Step 2: Print a sweep's config without running**

```bash
uv run --no-sync python experiments/grug/base/launch.py --version dev --sweep 'optimizer.learning_rate=[1e-3, 3e-3]' --show-config
```

Expected: two `mutable` trial lines and two YAML blocks whose `optimizer.learning_rate` differ, with `tracker: <tracker>` and `output_path: <output_path>`.

- [ ] **Step 3: Close the decision and record what was left out**

- D92: `**Status:** Adopted` with "Landed: <the four commit hashes>".
- The spec's status line: "approved design, 2026-10-08; implemented <date> (plan `plans/2026-10-08-experiment-overrides.md`)".
- Deferred table rows (source D92, resolution "Open"): writing the materialized config when a run starts; grug's log file in the run directory; pinning the Hugging Face revision of grug's training data; named presets for `--set`; adaptive search (Optuna) as a loop over `resolve`; a SLURM backend for Fray; forwarding `--set` to SkyRL runs as Hydra overrides (D68).

```bash
git add docs/poorin/decisions.md docs/poorin/specs/2026-10-08-experiment-overrides-design.md
git commit -m "[poorin] Close D92 and record what the overrides left out"
```
