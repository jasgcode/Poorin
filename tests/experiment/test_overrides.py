# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed --set overrides: decoding, refusals, and the names they give a trial."""

import dataclasses
import typing

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


# Marked as grug's launch config is, which typing.get_type_hints then reports as having no fields.
@typing.no_type_check
@dataclasses.dataclass(frozen=True)
class Unchecked:
    steps: int


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


def test_a_config_marked_no_type_check_still_decodes_to_its_field_types():
    assert _set("steps=200").apply(FINGERPRINT, Unchecked(steps=50)).steps == 200


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
        ("trainer.steps={", r"trainer\.steps=\{"),
        ("optimizer=3", "optimizer=3"),
    ],
)
def test_a_bad_override_is_refused_with_its_path(item, message):
    with pytest.raises(OverrideError, match=message):
        _set(item).apply(FINGERPRINT, _config())


def test_a_path_set_twice_is_refused():
    with pytest.raises(OverrideError, match=r"trainer\.steps"):
        _set("trainer.steps=1", "trainer.steps=2")


@pytest.mark.parametrize(
    "items",
    [
        ("optimizer={type: muon}", "optimizer.learning_rate=1"),
        ("optimizer.learning_rate=1", "optimizer={type: muon}"),
    ],
)
def test_a_path_and_one_beneath_it_are_refused_in_either_order(items):
    # Applied in command-line order the result would depend on it, while the trial's name would not.
    with pytest.raises(OverrideError, match=r"optimizer and optimizer\.learning_rate"):
        _set(*items)


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
