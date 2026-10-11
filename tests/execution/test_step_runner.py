# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextvars
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import poorin.execution.step_runner as step_runner_module
import pytest
from fray.current_client import current_client, set_current_client
from fray.types import ResourceConfig
from poorin.execution.artifact import Artifact, read_artifact, write_artifact
from poorin.execution.lazy import OUT, apply, lower_all, run
from poorin.execution.remote import remote
from poorin.execution.step_runner import StepRunner, steps_to_build
from poorin.execution.step_spec import StepSpec
from poorin.execution.step_status import STATUS_FAILED, STATUS_SUCCESS, StatusFile, read_status
from pydantic import BaseModel

# A frozen calendar version for the toy steps these tests build.
VERSION = "2026.07.16"

# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------


class TokenizeMetadata(BaseModel):
    path: str
    num_tokens: int


class TrainMetadata(BaseModel):
    tokens_seen: int
    checkpoint_path: str


# ---------------------------------------------------------------------------
# Pipeline functions: download → tokenize → train
#
# Each function accepts artifact instances as inputs and returns an artifact
# describing its output.
# ---------------------------------------------------------------------------


def download_raw_data(output_path: str, source_url: str) -> Artifact:
    """Download raw data shards to output_path."""
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    for shard in range(3):
        with open(os.path.join(data_dir, f"shard-{shard}.jsonl"), "w") as f:
            for i in range(10):
                json.dump({"id": shard * 10 + i, "text": f"doc {shard * 10 + i}", "src": source_url}, f)
                f.write("\n")
    return Artifact(path=data_dir)


def tokenize_data(output_path: str, raw_data: Artifact, tokenizer: str) -> TokenizeMetadata:
    """Tokenize documents from the raw data artifact."""
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    total_tokens = 0
    out_docs = []
    for fname in sorted(os.listdir(raw_data.path)):
        with open(os.path.join(raw_data.path, fname)) as f:
            for line in f:
                doc = json.loads(line)
                tokens = doc["text"].split()
                total_tokens += len(tokens)
                out_docs.append({"id": doc["id"], "tokens": tokens, "tokenizer": tokenizer})

    with open(os.path.join(data_dir, "tokenized.jsonl"), "w") as f:
        for doc in out_docs:
            json.dump(doc, f)
            f.write("\n")

    return TokenizeMetadata(path=data_dir, num_tokens=total_tokens)


def train_on_tokenized_data(output_path: str, tokenized: TokenizeMetadata) -> TrainMetadata:
    """Train a model on the tokenized data artifact."""
    os.makedirs(output_path, exist_ok=True)
    total_tokens = 0
    for fname in sorted(os.listdir(tokenized.path)):
        with open(os.path.join(tokenized.path, fname)) as f:
            for line in f:
                doc = json.loads(line)
                total_tokens += len(doc["tokens"])

    ckpt_path = os.path.join(output_path, "ckpt")
    with open(ckpt_path, "w") as f:
        f.write(f"model trained on {total_tokens} tokens\n")

    return TrainMetadata(tokens_seen=total_tokens, checkpoint_path=ckpt_path)


# ---------------------------------------------------------------------------
# Building StepSpecs directly
# ---------------------------------------------------------------------------


def _spec(
    tmp_path: Path,
    name: str,
    fn: Callable[[str], Any],
    *,
    deps: list[StepSpec] | None = None,
    version: str = VERSION,
) -> StepSpec:
    """A step at ``tmp_path/name`` whose recipe is keyed by its name."""
    payload = json.dumps({"name": name})
    return StepSpec(
        name=name,
        version=version,
        output_path=(tmp_path / name).as_posix(),
        fingerprint=f"{name}-recipe",
        fingerprint_payload=payload,
        fn=fn,
        deps=deps or [],
    )


def _saving(build: Callable[[str], BaseModel]) -> Callable[[str], None]:
    """A step fn that saves what ``build`` returns as the output's record."""

    def fn(output_path: str) -> None:
        write_artifact(build(output_path), output_path)

    return fn


def _recording(name: str, executed: list[str]) -> Callable[[str], None]:
    """A step fn that appends ``name`` to ``executed`` when (and only when) it runs."""

    def fn(output_path: str) -> None:
        executed.append(name)
        os.makedirs(output_path, exist_ok=True)

    return fn


def _mark_built(step: StepSpec) -> None:
    StatusFile(step.output_path, "test").write_status(STATUS_SUCCESS)


# ---------------------------------------------------------------------------
# StepRunner tests: three-step pipeline
# ---------------------------------------------------------------------------


def _build_pipeline(tmp_path: Path) -> list[StepSpec]:
    """Build download → tokenize → train as StepSpecs.

    Each step saves its artifact as its record; inter-step data flows through ``read_artifact``,
    deferred to execution time.
    """
    source_url = "http://data.example.com/raw.tar"
    download_step = _spec(tmp_path, "download", _saving(lambda out: download_raw_data(out, source_url)))
    tokenize_step = _spec(
        tmp_path,
        "tokenize",
        _saving(lambda out: tokenize_data(out, read_artifact(download_step.output_path, Artifact), "word")),
        deps=[download_step],
    )
    train_step = _spec(
        tmp_path,
        "train",
        _saving(lambda out: train_on_tokenized_data(out, read_artifact(tokenize_step.output_path, TokenizeMetadata))),
        deps=[tokenize_step],
    )
    return [download_step, tokenize_step, train_step]


def test_runner_executes_pipeline(tmp_path: Path):
    """The runner should execute download → tokenize → train in order."""
    steps = _build_pipeline(tmp_path)
    StepRunner().run(steps)

    download_path, tokenize_path, train_path = (s.output_path for s in steps)

    # Download produced shards
    raw_artifact = read_artifact(download_path, Artifact)
    assert os.path.isdir(raw_artifact.path)
    assert len(os.listdir(raw_artifact.path)) == 3

    # Tokenize produced output with correct token count
    tokenize_artifact = read_artifact(tokenize_path, TokenizeMetadata)
    assert tokenize_artifact.num_tokens == 60  # 30 docs * 2 words each

    # Train produced a checkpoint
    train_artifact = read_artifact(train_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0
    assert os.path.exists(train_artifact.checkpoint_path)


def test_runner_skips_completed_steps(tmp_path: Path):
    """Running the same pipeline twice should skip already-succeeded steps."""
    steps = _build_pipeline(tmp_path)
    StepRunner().run(steps)

    tokenize_artifact_path = os.path.join(steps[1].output_path, ".artifact.json")
    mtime_before = os.path.getmtime(tokenize_artifact_path)

    StepRunner().run(steps)

    mtime_after = os.path.getmtime(tokenize_artifact_path)
    assert mtime_before == mtime_after, "Tokenize artifact should not have been rewritten"


def test_runner_respects_dependency_order(tmp_path: Path):
    """Steps should execute in dependency order even if given out of order."""
    steps = _build_pipeline(tmp_path)
    StepRunner().run(list(reversed(steps)))

    train_artifact = read_artifact(steps[2].output_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0


def test_runner_max_concurrent(tmp_path: Path):
    """max_concurrent=1 should still complete the full pipeline."""
    steps = _build_pipeline(tmp_path)
    StepRunner().run(steps, max_concurrent=1)

    train_artifact = read_artifact(steps[2].output_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0


def test_runner_walks_transitive_deps(tmp_path: Path):
    """Passing only terminal steps should cause the runner to walk and run transitive deps."""
    executed: list[str] = []
    dep = _spec(tmp_path, "dep", _recording("dep", executed))
    mid = _spec(tmp_path, "mid", _recording("mid", executed), deps=[dep])
    terminal = _spec(tmp_path, "terminal", _recording("terminal", executed), deps=[mid])

    StepRunner().run([terminal])

    assert executed == ["dep", "mid", "terminal"]


def test_runner_walks_transitive_deps_with_cache_hit(tmp_path: Path):
    """Deps already succeeded on disk must be recognized via cache-hit during the walk."""
    executed: list[str] = []
    dep = _spec(tmp_path, "dep", _recording("dep", executed))
    downstream = _spec(tmp_path, "downstream", _recording("downstream", executed), deps=[dep])

    # Prime the cache for ``dep`` only.
    StepRunner().run([dep])
    assert executed == ["dep"]

    # Pass only ``downstream``; the runner walks deps and cache-hits ``dep``.
    StepRunner().run([downstream])
    assert executed == ["dep", "downstream"]


def test_runner_reruns_a_step_that_failed_before(tmp_path: Path):
    attempts: list[str] = []

    def flaky(output_path: str) -> None:
        attempts.append(output_path)
        if len(attempts) == 1:
            raise ValueError("first attempt fails")
        os.makedirs(output_path, exist_ok=True)

    step = _spec(tmp_path, "flaky", flaky)
    with pytest.raises(RuntimeError):
        StepRunner().run([step])
    assert read_status(step.output_path) == STATUS_FAILED

    StepRunner().run([step])

    assert len(attempts) == 2
    assert read_status(step.output_path) == STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Pruning already-built subtrees
# ---------------------------------------------------------------------------


def _download_normalize_tokenize(tmp_path: Path, executed: list[str]) -> tuple[StepSpec, StepSpec, StepSpec]:
    download = _spec(tmp_path, "download", _recording("download", executed))
    normalize = _spec(tmp_path, "normalize", _recording("normalize", executed), deps=[download])
    tokenize = _spec(tmp_path, "tokenize", _recording("tokenize", executed), deps=[normalize])
    return download, normalize, tokenize


def test_runner_prunes_built_terminal_subtree(tmp_path: Path):
    """A cached terminal must not rebuild its unbuilt intermediate ancestors.

    Regression for #6783: ``download → normalize → tokenize`` with only ``tokenize``
    cached used to rebuild the unpinned ``normalize`` (and ``download``) anyway.
    """
    executed: list[str] = []
    _download, _normalize, tokenize = _download_normalize_tokenize(tmp_path, executed)

    # Prime ONLY the terminal as a cached SUCCESS; its ancestors stay unbuilt.
    _mark_built(tokenize)

    StepRunner().run([tokenize])

    # Terminal served from cache, ancestors pruned: nothing ran.
    assert executed == []


def test_runner_builds_unbuilt_ancestors(tmp_path: Path):
    """With nothing cached, the full chain runs in dependency order (no over-pruning)."""
    executed: list[str] = []
    _download, _normalize, tokenize = _download_normalize_tokenize(tmp_path, executed)

    StepRunner().run([tokenize])

    assert executed == ["download", "normalize", "tokenize"]


def test_runner_prune_keeps_unbuilt_sibling_branch(tmp_path: Path):
    """A shared dep feeding a cached branch and an unbuilt branch still runs once.

    Diamond ``a → {b, c}``, ``b → d``, ``c → d``: ``c`` is cached but its dep ``d``
    is not. Pruning ``c`` must not strand ``d`` — ``d`` is still needed by ``b``.
    """
    executed: list[str] = []
    d = _spec(tmp_path, "d", _recording("d", executed))
    b = _spec(tmp_path, "b", _recording("b", executed), deps=[d])
    c = _spec(tmp_path, "c", _recording("c", executed), deps=[d])
    a = _spec(tmp_path, "a", _recording("a", executed), deps=[b, c])

    # ``c`` is already built; its dep ``d`` is not.
    _mark_built(c)

    StepRunner().run([a])

    assert "c" not in executed  # served from cache, not rebuilt
    assert executed.count("d") == 1  # built once, for the unbuilt ``b`` branch
    assert sorted(executed) == ["a", "b", "d"]


def test_runner_does_not_prune_dev_node(tmp_path: Path):
    """A mutable (dev) artifact at SUCCESS always rebuilds, so its deps are not pruned."""
    executed: list[str] = []
    dep = _spec(tmp_path, "dep", _recording("dep", executed))
    dev = _spec(tmp_path, "dev", _recording("dev", executed), deps=[dep], version="dev")

    # Even with a SUCCESS status on disk, a dev version is rebuilt — and pulls its dep.
    _mark_built(dev)

    StepRunner().run([dev])

    assert executed == ["dep", "dev"]


def test_runner_prune_cache_vanished_fails(tmp_path: Path, monkeypatch):
    """If a pruned node's cached output is gone at confirm time, the run fails loudly
    rather than running the node with its inputs pruned away."""
    executed: list[str] = []
    dep = _spec(tmp_path, "dep", _recording("dep", executed))
    terminal = _spec(tmp_path, "terminal", _recording("terminal", executed), deps=[dep])

    # Treat ``terminal`` as built (pruning ``dep``) although nothing is on disk — the
    # state if its cached output vanished between scheduling and confirmation.
    monkeypatch.setattr(step_runner_module, "step_is_built", lambda s: s.output_path == terminal.output_path)

    with pytest.raises(RuntimeError, match=r"1 step\(s\) failed"):
        StepRunner().run([terminal])

    # The pruned node is not run (its inputs were dropped), and neither is its dep.
    assert executed == []


def test_runner_dedups_shared_deps(tmp_path: Path):
    """A dep shared by multiple terminals must be executed exactly once."""
    executed: list[str] = []
    dep = _spec(tmp_path, "shared_dep", _recording("shared_dep", executed))
    a = _spec(tmp_path, "a", _recording("a", executed), deps=[dep])
    b = _spec(tmp_path, "b", _recording("b", executed), deps=[dep])

    StepRunner().run([a, b])

    assert executed.count("shared_dep") == 1


def test_runner_preserves_underlying_step_exception(tmp_path: Path):
    """The top-level runner error should retain the original failing exception as a cause."""

    def failing_step(_output_path: str) -> None:
        raise ValueError("sentinel step failure")

    step = _spec(tmp_path, "failing_step", failing_step)

    with pytest.raises(RuntimeError, match=r"1 step\(s\) failed") as exc_info:
        StepRunner().run([step])

    step_failure = exc_info.value.__cause__
    assert isinstance(step_failure, RuntimeError)
    assert "Step failed: failing_step" in str(step_failure)
    assert isinstance(step_failure.__cause__, ValueError)
    assert "sentinel step failure" in str(step_failure.__cause__)


# ---------------------------------------------------------------------------
# Remote step functions
# ---------------------------------------------------------------------------


class _SubmitSpy:
    """Wraps a fray client and captures every ``submit`` call's request."""

    def __init__(self, inner):
        self._inner = inner
        self.requests = []

    def submit(self, request, adopt_existing: bool = True):
        self.requests.append(request)
        return self._inner.submit(request, adopt_existing=adopt_existing)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _call_remote_with_submit_spy(fn, fray_client) -> _SubmitSpy:
    spy = _SubmitSpy(fray_client)
    with set_current_client(spy):
        fn()
    return spy


def _assert_single_submit_extras(spy: _SubmitSpy, expected: list[str]) -> None:
    assert len(spy.requests) == 1
    assert spy.requests[0].environment.extras == expected


def test_runner_builds_a_remote_step_as_one_fray_job(tmp_path: Path, fray_client):
    """A ``remote`` step fn submits itself to the runner's Fray client and builds the output."""
    custom = ResourceConfig.with_cpu(cpu=2, ram="8g")

    @remote(resources=custom)
    def my_step(output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        Path(output_path, "done").write_text("built")

    step = _spec(tmp_path, "remote_step", my_step)
    spy = _SubmitSpy(fray_client)
    with set_current_client(spy):
        StepRunner().run([step])

    assert [request.resources for request in spy.requests] == [custom]
    assert Path(step.output_path, "done").read_text() == "built"
    assert read_status(step.output_path) == STATUS_SUCCESS


def test_remote_direct_call_uses_device_extra(fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    @remote(resources=resources)
    def my_step() -> None:
        return None

    _assert_single_submit_extras(_call_remote_with_submit_spy(my_step, fray_client), ["gpu"])


def test_remote_direct_call_dependency_groups_can_override_device_extra(fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    @remote(resources=resources, pip_dependency_groups=[])
    def my_step() -> None:
        return None

    _assert_single_submit_extras(_call_remote_with_submit_spy(my_step, fray_client), [])


# ---------------------------------------------------------------------------
# Context propagation to worker threads
# ---------------------------------------------------------------------------


def test_runner_propagates_context_vars(tmp_path):
    """StepRunner must propagate contextvars to worker threads.

    This ensures that fray's ``set_current_client`` is visible inside step
    functions dispatched by the thread pool, so ZephyrContext (and anything
    else that calls ``current_client()``) picks up the correct client.
    """

    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_var", default=None)
    observed: list[str | None] = []

    def capture_ctx(output_path: str) -> None:
        observed.append(test_var.get())

    step = _spec(tmp_path, "ctx_check", capture_ctx)

    test_var.set("from_parent")
    StepRunner().run([step])

    assert observed == ["from_parent"], f"Expected context var to propagate, got {observed}"


def test_runner_propagates_fray_client(tmp_path):
    """StepRunner explicitly propagates the fray client to worker threads.

    This tests the explicit client capture path (not just generic contextvars)
    to ensure current_client() returns the correct client inside step functions.
    """

    class FakeClient:
        """Marker client to verify propagation."""

    observed_clients: list[type] = []

    def check_client(output_path: str) -> None:
        observed_clients.append(type(current_client()))

    step = _spec(tmp_path, "fray_check", check_client)

    with set_current_client(FakeClient()):
        StepRunner().run([step])

    assert observed_clients == [FakeClient], f"Expected FakeClient in worker thread, got {observed_clients}"


# ---------------------------------------------------------------------------
# steps_to_build
# ---------------------------------------------------------------------------


def _write(out, **_inputs):
    os.makedirs(out, exist_ok=True)
    Path(out, "done").write_text("built")


def _graph():
    data = apply("data/toy", _write, version=VERSION, out=OUT)
    model = apply("models/toy", _write, version=VERSION, out=OUT, data=data)
    return data, model


def _names(steps):
    return [s.name for s in steps]


def test_steps_to_build_lists_the_unbuilt_graph_in_schedule_order():
    _, model = _graph()
    assert _names(steps_to_build(lower_all([model]))) == ["data/toy", "models/toy"]


def test_steps_to_build_leaves_out_built_steps_and_inputs_pruned_behind_them():
    data, model = _graph()
    run(model)
    shutil.rmtree(data.path())  # an input behind a built consumer is not rebuilt
    report = apply("reports/toy", _write, version="dev", out=OUT, model=model)
    assert _names(steps_to_build(lower_all([model]))) == []
    assert _names(steps_to_build(lower_all([report]))) == ["reports/toy"]
