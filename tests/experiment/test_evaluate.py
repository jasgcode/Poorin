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
        return EvalResult(
            run=run.label, step=step, suite=self.name, metrics={"eval/loss": self.scale * step}, examples=1
        )


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
