# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The artifact record: full-record provenance, refused recipe drift, dev mutability, pins.

Drives the real lower -> StepRunner pipeline against a tmp prefix so the drift check is
exercised exactly as it fires in production (before a run starts), plus the manual
``read_artifact``/``write_artifact`` payload API.
"""

import dataclasses
import json
import os
import shutil

import pytest
from poorin.execution.artifact import (
    Artifact,
    DriftAcceptanceError,
    FingerprintMismatchError,
    StepRecordIdentity,
    read_artifact,
    read_record,
    write_artifact,
    write_step_record,
)
from poorin.execution.lazy import OUT, ArtifactStep, apply, lower, run
from poorin.execution.step_runner import StepRunner
from pydantic import BaseModel
from rigging.provenance import Provenance

# A frozen calendar version for the toy artifacts these tests build.
V = "2026.06.28"


class Toy(Artifact):
    payload: str


def _toy(version: str, payload: str) -> ArtifactStep[Toy]:
    """A toy value artifact whose recipe (and thus fingerprint) is keyed by ``payload``."""
    return ArtifactStep(
        name="datasets/toy",
        version=version,
        artifact_type=Toy,
        run=lambda config: Toy(payload=config["payload"]),
        build_config=lambda ctx: {"out": ctx.output_path, "payload": payload},
    )


# --- the full record -----------------------------------------------------------


def test_records_full_provenance_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    art = _toy(V, "a")
    run(art)

    record = read_record(f"{tmp_path}/datasets/toy/{V}")
    assert record is not None
    assert (record.name, record.version, record.fingerprint) == ("datasets/toy", V, art.fingerprint())
    # The unified record carries the full provenance, not just the payload.
    assert record.result_type.endswith(".Toy")
    assert record.provenance is not None
    assert record.provenance.command_line  # sys.argv of the launching process
    # result is the artifact's declared value fields only (``path`` is not a value field).
    assert record.result == {"payload": "a"}


def test_record_carries_fingerprint_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    art = _toy(V, "a")
    run(art)

    record = read_record(f"{tmp_path}/datasets/toy/{V}")
    assert record.fingerprint_payload == art.fingerprint_payload()


def test_same_recipe_rerun_is_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))
    # Identical recipe + version: a cache hit, no error.
    run(_toy(V, "a"))


def _make_dir(out):
    os.makedirs(out, exist_ok=True)


def test_run_records_the_provenance_it_is_given():
    handle = apply("data/given", _make_dir, version="2026.07.16", out=OUT)
    given = dataclasses.replace(Provenance.capture(), note="from the command line")
    run(handle, provenance=given)
    assert read_record(handle.path()).provenance.note == "from the command line"


# --- recipe drift --------------------------------------------------------------


def _consumer(dep: ArtifactStep, name: str = "checkpoints/consumer") -> ArtifactStep[Toy]:
    return ArtifactStep(
        name=name,
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


@pytest.mark.parametrize("drifted_first", [True, False])
def test_two_recipes_for_one_output_in_one_graph_are_refused(tmp_path, monkeypatch, drifted_first):
    # In either order, the handle with the recorded recipe must not hide the drifted one.
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_consumer(_toy(V, "a"), "checkpoints/one"), _consumer(_toy(V, "a"), "checkpoints/two"))
    drifted = _consumer(_toy(V, "b"), "checkpoints/one")
    unchanged = _consumer(_toy(V, "a"), "checkpoints/two")
    handles = (drifted, unchanged) if drifted_first else (unchanged, drifted)
    with pytest.raises(FingerprintMismatchError) as refused:
        run(*handles)
    message = str(refused.value)
    assert f"datasets/toy@{V}" in message
    assert _toy(V, "a").fingerprint() in message and _toy(V, "b").fingerprint() in message


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


def test_accepting_drift_of_an_artifact_two_handles_build_records_it_once(tmp_path, monkeypatch):
    # A sweep's trials each construct their own handle for a shared dataset.
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    old = _toy(V, "a")
    run(old)
    run(_toy(V, "b"), _toy(V, "b"), accept_drift={"datasets/toy"})
    assert read_record(f"{tmp_path}/datasets/toy/{V}").accepted_from == [old.fingerprint()]


def test_accepting_a_drifted_dependency_serves_its_built_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_consumer(_toy(V, "a")))
    run(_consumer(_toy(V, "b")), accept_drift={"datasets/toy"})
    assert read_record(f"{tmp_path}/datasets/toy/{V}").fingerprint == _toy(V, "b").fingerprint()


def test_accepting_drift_restamps_the_record_where_the_output_now_lives(tmp_path, monkeypatch):
    # The record names the prefix it was built under. After a copy to another prefix, accepting the
    # change must re-stamp the copy's record, or the runner refuses the copy again.
    built, copied = tmp_path / "built", tmp_path / "copied"
    monkeypatch.setenv("POORIN_PREFIX", str(built))
    run(_toy(V, "a"))
    shutil.copytree(built, copied)
    before = {path: path.read_bytes() for path in built.rglob("*") if path.is_file()}
    monkeypatch.setenv("POORIN_PREFIX", str(copied))
    (served,) = run(_toy(V, "b"), accept_drift={"datasets/toy"})
    assert served.payload == "a"
    assert read_record(f"{copied}/datasets/toy/{V}").fingerprint == _toy(V, "b").fingerprint()
    assert {path: path.read_bytes() for path in built.rglob("*") if path.is_file()} == before


def test_a_runner_called_directly_refuses_a_changed_recipe(tmp_path, monkeypatch):
    # Code that lowers handles and runs StepRunner itself skips run()'s whole-graph check, so the
    # runner's own check must refuse the output.
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))
    with pytest.raises(FingerprintMismatchError, match="payload: 'a' -> 'b'"):
        StepRunner().run([lower(_toy(V, "b"))])
    assert read_record(f"{tmp_path}/datasets/toy/{V}").result == {"payload": "a"}


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


def test_a_pin_to_an_output_built_from_another_recipe_is_served(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))
    pinned = dataclasses.replace(_toy(V, "b"), name="datasets/pinned", override_path=f"datasets/toy/{V}")
    (served,) = run(pinned)
    assert served.payload == "a"


def test_a_fixed_version_cannot_depend_on_a_dev_one():
    # Refused when the handle is built, so neither a plan nor a run can start from such a graph.
    with pytest.raises(ValueError, match="depends on mutable"):
        _consumer(_toy("dev", "a"))


def test_a_dev_version_rebuilds_on_every_run(tmp_path, monkeypatch):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    builds = []

    def build(config):
        builds.append(config["payload"])
        return Toy(payload=config["payload"])

    dev = dataclasses.replace(_toy("dev", "a"), run=build)
    run(dev)
    run(dev)  # a calendar version would be served from cache here
    assert builds == ["a", "a"]


# --- the manual typed-payload API ----------------------------------------------


class _Doc(BaseModel):
    title: str
    tokens: int


def test_write_then_read_artifact_round_trips_a_payload(tmp_path):
    out = (tmp_path / "step").as_posix()
    write_artifact(_Doc(title="x", tokens=7), out)

    loaded = read_artifact(out, _Doc)
    assert loaded == _Doc(title="x", tokens=7)


def test_read_artifact_requires_a_pydantic_schema(tmp_path):
    out = (tmp_path / "step").as_posix()
    write_artifact(_Doc(title="x", tokens=7), out)
    with pytest.raises(TypeError):
        read_artifact(out, dict)


def test_read_artifact_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_artifact((tmp_path / "nothing").as_posix(), _Doc)


# --- the legacy Ray executor .executor_info fallback ---------------------------

# A trimmed per-step ``.executor_info`` in the old ``ExecutorStepInfo`` schema, as written for the
# pinned llama3 Nemotron-CC caches: the tokenizer lives only under ``config``.
_EXECUTOR_INFO = json.dumps(
    {
        "name": "tokenized/nemotron_cc/hq_actual",
        "fn_name": "tokenize",
        "config": {"tokenizer": "meta-llama/Meta-Llama-3.1-8B", "tags": ["nemotron"]},
        "description": None,
        "override_output_path": None,
        "version": {},
        "dependencies": ["gs://bucket/raw/nemotron"],
        "output_path": "gs://bucket/tokenized/nemotron_cc/hq_actual-5af4cc",
    }
)


def test_read_record_recovers_config_from_executor_info(tmp_path):
    """A cache with only a legacy ``.executor_info`` resolves its config, not ``None``."""
    (tmp_path / ".executor_info").write_text(_EXECUTOR_INFO)

    record = read_record(tmp_path.as_posix())
    assert record is not None
    assert record.config == {"tokenizer": "meta-llama/Meta-Llama-3.1-8B", "tags": ["nemotron"]}
    assert record.name == "tokenized/nemotron_cc/hq_actual"
    assert record.deps == ["gs://bucket/raw/nemotron"]


def test_read_record_prefers_modern_record_over_executor_info(tmp_path):
    """``.executor_info`` is a last resort: a modern ``.artifact.json`` still wins."""
    (tmp_path / ".executor_info").write_text(_EXECUTOR_INFO)
    (tmp_path / ".artifact.json").write_text('{"name": "modern", "config": {"tokenizer": "gpt2"}}')

    record = read_record(tmp_path.as_posix())
    assert record.name == "modern"
    assert record.config == {"tokenizer": "gpt2"}


# --- the pre-#6649 bare-payload .artifact.json ---------------------------------

# Before #6649 the value model was serialized straight into ``.artifact.json`` with no record
# wrapper: every payload field sits at top level and there is no name/fingerprint/result_type/result.
_LEGACY_BARE_PAYLOAD = json.dumps(
    {"version": "v1", "output_dir": "gs://bucket/datakit/embed/cp/peps_52c03790", "counters": {"rows": 128}}
)


class _EmbedPayload(BaseModel):
    version: str
    output_dir: str
    counters: dict


def test_read_artifact_adopts_pre_6649_bare_payload(tmp_path):
    """A pre-#6649 bare ``.artifact.json`` loads: the whole document becomes the result payload."""
    (tmp_path / ".artifact.json").write_text(_LEGACY_BARE_PAYLOAD)

    loaded = read_artifact(tmp_path.as_posix(), _EmbedPayload)
    assert loaded == _EmbedPayload(
        version="v1", output_dir="gs://bucket/datakit/embed/cp/peps_52c03790", counters={"rows": 128}
    )


def test_read_record_leaves_a_genuine_data_record_untouched(tmp_path):
    """A modern record with identity but no ``result`` is not mistaken for a bare payload."""
    (tmp_path / ".artifact.json").write_text(
        '{"name": "datasets/x", "version": "v1", "fingerprint": "abc", "result": null}'
    )

    record = read_record(tmp_path.as_posix())
    assert record.name == "datasets/x"
    assert record.result is None


# --- the schedule-time StepRunner .executor_info stub --------------------------

# A schedule-time ``StepRunner`` stub: same filename, but ``executor_version == "step_runner"`` and
# a ``config`` that is the identity ``hash_attrs`` (no materialized tokenizer/format/tags).
_STEP_RUNNER_STUB = json.dumps(
    {
        "executor_version": "step_runner",
        "name": "paloma/4chan-llama3",
        "config": {"fingerprint": "449bd131", "version": "2026.06.28", "deps": []},
        "output_path": "gs://bucket/paloma/4chan-llama3/2026.06.28",
    }
)


def test_read_record_ignores_step_runner_stub(tmp_path):
    """A schedule-time ``StepRunner`` ``.executor_info`` stub is not served as a record — it
    carries no materialized config, so the output reads back as record-less rather than as a
    tokenizer-less cache (#6836)."""
    (tmp_path / ".executor_info").write_text(_STEP_RUNNER_STUB)

    assert read_record(tmp_path.as_posix()) is None


class _Meta(BaseModel):
    path: str
    n: int


def test_write_step_record_roundtrips_identity_and_payload(tmp_path):
    """The full step record carries name/deps/config/fingerprint_payload + the typed payload, and
    stamps best-effort provenance without failing when git is unavailable."""
    out = tmp_path.as_posix()
    write_step_record(
        StepRecordIdentity(
            name="child",
            deps=["parent_ab12cd34"],
            dep_paths=[f"{out}/normalize"],
            config={"tokenizer": "gpt2"},
            fingerprint_payload="fp-json",
        ),
        output_path=out,
        result=_Meta(path=out, n=7),
    )

    rec = read_record(out)
    assert rec is not None
    assert rec.name == "child"
    assert rec.deps == ["parent_ab12cd34"]
    assert rec.dep_paths == [f"{out}/normalize"]
    assert rec.config == {"tokenizer": "gpt2"}
    assert rec.fingerprint_payload == "fp-json"
    assert rec.result == {"path": out, "n": 7}
    # provenance is best-effort: git fields degrade to empty on a git-less bundle; never raises
    assert read_artifact(out, _Meta).n == 7


def test_write_step_record_tolerates_none_result(tmp_path):
    """A side-effect step (fn returns None) records identity + lineage with a null payload."""
    out = tmp_path.as_posix()
    identity = StepRecordIdentity(name="train", deps=[], dep_paths=[], config={"seed": 42})
    write_step_record(identity, output_path=out, result=None)

    rec = read_record(out)
    assert rec is not None
    assert rec.name == "train"
    assert rec.config == {"seed": 42}
    assert rec.result is None
