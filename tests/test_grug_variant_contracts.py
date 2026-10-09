# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for grug variants under experiments/grug/*.

These checks are intentionally variant-discovered: if a subdirectory contains
`model.py` and/or `train.py`, it is expected to satisfy the corresponding
lowering and training contracts.
"""

import contextlib
import contextvars
import dataclasses
import importlib
import json
import logging
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
import pytest
from fray.types import ResourceConfig
from jax._src import config as jax_config
from jax.sharding import use_abstract_mesh
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import _compact_grug_mesh_shape
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig


def _discover_grug_variants_with_file(filename: str) -> list[str]:
    grug_dir = Path(__file__).resolve().parents[1] / "experiments" / "grug"
    variants: list[str] = []
    found_any = False
    for child in sorted(grug_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if (child / filename).is_file():
            found_any = True
            if _variant_has_noverify(child):
                continue
            variants.append(child.name)
    if not variants and not found_any:
        raise AssertionError(f"No grug variants with {filename} found under {grug_dir}")
    return variants


def _variant_module_name(variant: str, module: str) -> str:
    return f"experiments.grug.{variant}.{module}"


def test_compact_grug_mesh_shape_allows_expert_axis_to_span_processes():
    assert _compact_grug_mesh_shape(
        process_count=32,
        local_device_count=4,
        expert_axis_size=16,
        replica_axis_size=4,
        model_axis_size=1,
    ) == (4, 2, 1, 16, 1)


def test_compact_grug_mesh_shape_keeps_expert_axis_at_size_one():
    """Standardized contract: compact_grug_mesh always carries the expert axis.

    The data-loader and model code reference "expert" unconditionally; we keep the axis at
    size 1 instead of dropping it so size-1 cases (e.g. the GPU canary) don't fall through
    a separate "axis absent" code path. See #6252 for the bug this contract prevents.
    """
    assert _compact_grug_mesh_shape(
        process_count=1,
        local_device_count=4,
        expert_axis_size=1,
        replica_axis_size=1,
        model_axis_size=1,
    ) == (1, 4, 1, 1, 1)


def test_compact_grug_mesh_axis_sizes():
    """Distinct context and expert widths detect swapped axis positions."""
    run_on_cpu_devices(
        device_count=8,
        script="""
        from levanter.grug.sharding import compact_grug_mesh

        mesh = compact_grug_mesh(replica_axis_size=1, context_axis_size=2, expert_axis_size=4)
        assert tuple(mesh.shape.items()) == (
            ("replica_dcn", 1),
            ("data", 1),
            ("context", 2),
            ("expert", 4),
            ("model", 1),
        ), mesh.shape
        """,
    )


def test_legacy_checkpoint_restore_with_unit_context_axis():
    """Adding a size-one context axis preserves checkpoint values and placement."""
    run_on_cpu_devices(
        device_count=8,
        script="""
        import tempfile

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.checkpoint import load_checkpoint, save_checkpoint
        from levanter.grug.sharding import compact_grug_mesh

        legacy_mesh = Mesh(
            np.asarray(jax.devices(), dtype=object).reshape(1, 4, 2, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)
        assert dict(mesh.shape) == {"replica_dcn": 1, "data": 4, "context": 1, "expert": 2, "model": 1}, mesh.shape

        spec = P(("replica_dcn", "data", "expert"), None)
        shape = (8, 4)
        legacy_sharding = NamedSharding(legacy_mesh, spec)
        sharding = NamedSharding(mesh, spec)
        assert sharding.devices_indices_map(shape) == legacy_sharding.devices_indices_map(shape)

        written = jax.device_put(jnp.arange(32, dtype=jnp.float32).reshape(shape), legacy_sharding)
        with tempfile.TemporaryDirectory() as tmpdir:
            with jax.set_mesh(legacy_mesh):
                save_checkpoint({"params": written}, step=6000, checkpoint_path=tmpdir)
            with jax.set_mesh(mesh):
                restored = load_checkpoint(
                    {"params": jax.ShapeDtypeStruct(shape, jnp.float32, sharding=sharding)},
                    checkpoint_path=tmpdir,
                    mesh=mesh,
                )["params"]

        assert restored.sharding == sharding, restored.sharding
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(written))
        """,
    )


def test_legacy_checkpoint_restore_with_context_sharding():
    """A legacy expert checkpoint restores with its target context partitioning."""
    run_on_cpu_devices(
        device_count=8,
        script="""
        import tempfile

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.checkpoint import load_checkpoint, save_checkpoint
        from levanter.grug.sharding import compact_grug_mesh

        legacy_mesh = Mesh(
            np.asarray(jax.devices(), dtype=object).reshape(1, 4, 2, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        mesh = compact_grug_mesh(expert_axis_size=2, context_axis_size=2, replica_axis_size=1)
        assert dict(mesh.shape) == {"replica_dcn": 1, "data": 2, "context": 2, "expert": 2, "model": 1}, mesh.shape

        # An expert stack [experts, fan_in, fan_out]: four shards on the target mesh, two on the
        # mesh that wrote it.
        shape = (4, 3, 2)
        written = jax.device_put(
            jnp.arange(24, dtype=jnp.float32).reshape(shape),
            NamedSharding(legacy_mesh, P("expert", None, None)),
        )
        sharding = NamedSharding(mesh, P(("expert", "context"), None, None))
        assert len({index for index in sharding.devices_indices_map(shape).values()}) == 4

        with tempfile.TemporaryDirectory() as tmpdir:
            with jax.set_mesh(legacy_mesh):
                save_checkpoint({"params": written}, step=6000, checkpoint_path=tmpdir)
            with jax.set_mesh(mesh):
                restored = load_checkpoint(
                    {"params": jax.ShapeDtypeStruct(shape, jnp.float32, sharding=sharding)},
                    checkpoint_path=tmpdir,
                    mesh=mesh,
                )["params"]

        assert restored.sharding == sharding, restored.sharding
        np.testing.assert_array_equal(np.asarray(restored), np.asarray(written))
        """,
    )


def _variant_has_noverify(variant_dir: Path) -> bool:
    train_file = variant_dir / "train.py"
    if not train_file.is_file():
        return False
    return "# GRUG NOVERIFY" in train_file.read_text(encoding="utf-8")


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _discover_grug_variants_with_model_and_train() -> list[str]:
    model_variants = set(_discover_grug_variants_with_file("model.py"))
    train_variants = set(_discover_grug_variants_with_file("train.py"))
    variants = sorted(model_variants & train_variants)
    if not variants and model_variants and train_variants:
        return []
    if not variants:
        raise AssertionError("No grug variants with both model.py and train.py found")
    return variants


def _small_model_config(model_config_cls, *, vocab_size: int, seq_len: int):
    base_kwargs = {
        "vocab_size": vocab_size,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 2,
        "max_seq_len": seq_len,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "shared_expert_intermediate_dim": 64,
    }
    field_names = {field.name for field in dataclasses.fields(model_config_cls)}
    kwargs = {k: v for k, v in base_kwargs.items() if k in field_names}
    return model_config_cls(**kwargs)


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_model_and_train(),
)
def test_grug_variant_one_step_contract_lowers_with_default_ctor(variant: str):
    train_module = importlib.import_module(_variant_module_name(variant, "train"))
    model_module = importlib.import_module(_variant_module_name(variant, "model"))
    model_config_cls = model_module.GrugModelConfig
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{_variant_module_name(variant, 'model')} must define debug_mesh_and_token_pspec")

    cfg = model_config_cls(vocab_size=1024)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((32, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((32, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, out_watch_shape = eqx.filter_eval_shape(one_step)

    assert out_state_shape.step.shape == ()
    assert "train/loss" in out_metrics_shape
    assert out_metrics_shape["train/loss"].shape == ()
    assert out_watch_shape is None


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_model_and_train(),
)
def test_grug_variant_initial_state_only_stores_ema_when_enabled(variant: str):
    train_module = importlib.import_module(_variant_module_name(variant, "train"))
    model_module = importlib.import_module(_variant_module_name(variant, "model"))
    model_config_cls = model_module.GrugModelConfig
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{_variant_module_name(variant, 'model')} must define debug_mesh_and_token_pspec")

    cfg = model_config_cls(vocab_size=1024)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    mesh, _ = mesh_fn(num_devices=4)

    def init_state_shape(*, ema_beta: float | None):
        def build():
            return initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=ema_beta)

        with _reset_abstract_mesh(), use_abstract_mesh(mesh):
            return eqx.filter_eval_shape(build)

    no_ema_state_shape = init_state_shape(ema_beta=None)
    assert no_ema_state_shape.ema_params is None

    with_ema_state_shape = init_state_shape(ema_beta=0.999)
    assert with_ema_state_shape.ema_params is not None


@contextlib.contextmanager
def _json_tracker_logger() -> Iterator[tuple[str, StringIO]]:
    """A fresh logger for ``JsonLoggerConfig``, and the stream its records land in."""
    logger_name = f"test_grug_json_tracker_{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield logger_name, stream
    finally:
        logger.removeHandler(handler)


def _json_tracker_records(stream: StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _tiny_grug_base_run_config(run_dir: Path, *, run_id: str, logger_name: str):
    """A one-step grug base run on CPU over in-memory data, tracked by a JSON logger."""
    train_module = importlib.import_module("experiments.grug.base.train")
    model_module = importlib.import_module("experiments.grug.base.model")

    vocab_size = 128
    seq_len = 32
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))
    eval_examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + 100) % vocab_size)]

    train_dataset = ListAsyncDataset(examples)
    eval_dataset = ListAsyncDataset(eval_examples)
    data_config = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"train": train_dataset, "validation": eval_dataset})},
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    trainer_config = TrainerConfig(
        id=run_id,
        num_train_steps=1,
        train_batch_size=max(1, len(jax.devices())),
        tracker=JsonLoggerConfig(logger_name=logger_name),
        require_accelerator=False,
        use_explicit_mesh_axes=True,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_dir=run_dir / "logs",
        checkpointer=CheckpointerConfig(base_path=str(run_dir / "checkpoints")),
    )

    return train_module.GrugRunConfig(
        model=_small_model_config(model_module.GrugModelConfig, vocab_size=vocab_size, seq_len=seq_len),
        data=data_config,
        resources=ResourceConfig.with_cpu(),
        trainer=train_module.GrugTrainerConfig(
            trainer=trainer_config,
            log_every=1,
            backward_flow=train_module.BackwardFlowConfig(interval=0),
        ),
        eval=train_module.GrugEvalConfig(
            eval_batch_size=1,
            steps_per_eval=1,
            max_eval_batches=1,
            eval_current=True,
            eval_ema=False,
            compute_bpb=False,
        ),
    )


def test_grug_base_run_emits_expected_metrics_with_json_tracker(tmp_path: Path):
    train_module = importlib.import_module("experiments.grug.base.train")
    with _json_tracker_logger() as (logger_name, stream):
        run_cfg = _tiny_grug_base_run_config(tmp_path / "base", run_id="test-grug-base-metrics", logger_name=logger_name)
        train_module.run_grug(run_cfg)

    records = _json_tracker_records(stream)
    finish_records = [record for record in records if record.get("event") == "finish"]
    assert len(finish_records) == 1
    summary = finish_records[0]["summary"]

    required_keys = [
        "train/loss",
        "global_step",
        "throughput/duration",
        "throughput/hook_time",
        "throughput/loading_time",
        "throughput/total_tokens",
        "throughput/examples_per_second",
        "throughput/tokens_per_second",
        "throughput/flops_per_example_analytic",
        "eval/loss",
        "eval/loading_time",
        "eval/total_time",
    ]
    for key in required_keys:
        assert key in summary


def test_grug_runs_started_together_each_log_to_their_own_tracker(tmp_path: Path):
    # A sweep's trials run as threads of one process, and Levanter's tracker belongs to the process.
    # Two runs that overlapped both logged into the tracker of the run that started last, and the
    # first to finish finished it, so the other run's tracker never finished.
    train_module = importlib.import_module("experiments.grug.base.train")
    run_ids = ["test-grug-together-a", "test-grug-together-b"]
    with contextlib.ExitStack() as stack:
        loggers = [stack.enter_context(_json_tracker_logger()) for _ in run_ids]
        configs = [
            _tiny_grug_base_run_config(tmp_path / run_id, run_id=run_id, logger_name=logger_name)
            for run_id, (logger_name, _) in zip(run_ids, loggers, strict=True)
        ]
        with ThreadPoolExecutor(max_workers=len(configs)) as pool:
            futures = [pool.submit(contextvars.copy_context().run, train_module.run_grug, cfg) for cfg in configs]
            for future in futures:
                future.result()

    for run_id, (_, stream) in zip(run_ids, loggers, strict=True):
        records = _json_tracker_records(stream)
        logged_configs = [r["hparams"] for r in records if r["event"] == "hparams" and "trainer" in r["hparams"]]
        assert [config["trainer"]["trainer"]["id"] for config in logged_configs] == [run_id]
        assert [r["event"] for r in records if r["event"] == "finish"] == ["finish"]
