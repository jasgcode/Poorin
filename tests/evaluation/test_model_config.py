# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model YAML validation and serving-option rendering."""

import textwrap

import pytest
from draccus.utils import ParsingError
from poorin.evaluation.model_config import (
    ServeBackend,
    ServeConfig,
    load_model_config,
    scan_model_configs,
    serve_config_vllm_args,
)

_CATALOG_YAML = textwrap.dedent(
    """
    name: qwen3-32b
    location: Qwen/Qwen3-32B
    apply_chat_template: true
    resource_hint:
      gpu:
        H100: 2
    serve:
      tensor_parallel_size: 2
      max_model_len: 32768
      tool_call_parser: hermes
      reasoning_parser: qwen3
      vllm_extra_args: ["--enable-prefix-caching"]
    generation:
      extra_gen_kwargs:
        skip_special_tokens: "false"
    agent:
      agent_kwargs:
        extra_body: '{"chat_template_kwargs":{"enable_thinking":true}}'
    """
)


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_load_model_config_round_trips_the_catalog_shape(tmp_path):
    config = load_model_config(_write(tmp_path, "Qwen/Qwen3-32B.yaml", _CATALOG_YAML))

    assert config.name == "qwen3-32b"
    assert config.serve.backend is ServeBackend.VLLM
    assert config.serve.tensor_parallel_size == 2
    assert config.resource_hint.gpu == {"H100": 2}
    assert config.serve.vllm_extra_args == ("--enable-prefix-caching",)
    assert dict(config.generation.extra_gen_kwargs) == {"skip_special_tokens": "false"}
    assert "enable_thinking" in config.agent.agent_kwargs["extra_body"]


def test_load_model_config_rejects_name_with_job_path_separator(tmp_path):
    body = "name: Qwen/Qwen3-8B\nlocation: Qwen/Qwen3-8B\n"

    with pytest.raises(ParsingError):
        load_model_config(_write(tmp_path, "model.yaml", body))


def test_load_rejects_unknown_field_at_load_time(tmp_path):
    # A typo in a serve field name is the exact class of bug the schema exists to catch: it must fail
    # at load, not surface as a dropped serve flag hours into a run. draccus wraps the field-level
    # decoding error, so the offending name rides on the exception chain rather than the top message.
    bad = "name: x\nlocation: org/x\nserve:\n  swp_space: 8\n"
    with pytest.raises(Exception) as excinfo:
        load_model_config(_write(tmp_path, "bad.yaml", bad))
    chain = " ".join(str(exc) for exc in (excinfo.value, excinfo.value.__cause__, excinfo.value.__context__) if exc)
    assert "swp_space" in chain


@pytest.mark.parametrize(
    ("field", "value"),
    [("swap_space_gb", "32"), ("trust_remote_code", "true")],
)
def test_load_rejects_removed_serve_knob(tmp_path, field, value):
    # swap_space_gb and trust_remote_code were removed from the schema: the CUDA vLLM fork rejects
    # --swap-space, and the native server already forces --trust-remote-code on for every model, so a
    # per-model knob only duplicated the flag. A catalog file that still carries either must fail at
    # load rather than lower into a flag the server rejects or warns on.
    body = f"name: x\nlocation: org/x\nserve:\n  {field}: {value}\n"
    with pytest.raises(Exception) as excinfo:
        load_model_config(_write(tmp_path, "removed.yaml", body))
    chain = " ".join(str(exc) for exc in (excinfo.value, excinfo.value.__cause__, excinfo.value.__context__) if exc)
    assert field in chain


def test_serve_config_vllm_args_renders_typed_knobs():
    serve = ServeConfig(
        data_parallel_size=2,
        limit_mm_per_prompt='{"image":0,"video":0}',
        reasoning_parser="qwen3",
        tool_call_parser="hermes",
        vllm_extra_args=("--enable-prefix-caching",),
    )
    args = serve_config_vllm_args(serve)

    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"image":0,"video":0}'
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-prefix-caching" in args
    # The removed knobs are never reintroduced by rendering.
    assert "--swap-space" not in args
    assert "--trust-remote-code" not in args


def test_serve_config_vllm_args_explicit_flag_wins_over_typed_knob():
    # An operator who hand-wrote --data-parallel-size in the escape hatch must not get a second one
    # from the typed knob; the typed knob fills a gap, it does not duplicate.
    serve = ServeConfig(data_parallel_size=2, vllm_extra_args=("--data-parallel-size", "4"))
    args = serve_config_vllm_args(serve)
    assert args.count("--data-parallel-size") == 1
    assert args[args.index("--data-parallel-size") + 1] == "4"


def _gib(memory: str) -> int:
    return int(memory.removesuffix("g"))


def test_scan_model_configs_keys_by_name_and_skips_underscored(tmp_path):
    _write(tmp_path, "Qwen/Qwen3-8B.yaml", "name: qwen3-8b\nlocation: Qwen/Qwen3-8B\n")
    _write(tmp_path, "org/other.yaml", "name: other\nlocation: org/other\n")
    _write(tmp_path, "_patterns.yaml", "name: ignored\nlocation: x\n")
    configs = scan_model_configs(tmp_path)
    assert set(configs) == {"qwen3-8b", "other"}


def test_scan_model_configs_rejects_duplicate_names(tmp_path):
    _write(tmp_path, "a/dup.yaml", "name: dup\nlocation: a/x\n")
    _write(tmp_path, "b/dup.yaml", "name: dup\nlocation: b/x\n")
    with pytest.raises(ValueError, match="duplicate model name"):
        scan_model_configs(tmp_path)


@pytest.mark.parametrize(
    "serve",
    [
        {"pipeline_parallel_size": 2},
        {"pipeline_parallel_size": 0},
        {"tensor_parallel_size": 8, "pipeline_parallel_size": 2, "backend": ServeBackend.LEVANTER},
        {"gpu_memory_utilization": 1.1},
        {"gpu_memory_utilization": 0.9, "vllm_extra_args": ("--gpu-memory-utilization=0.8",)},
    ],
)
def test_invalid_pipeline_or_memory_settings_fail_before_submission(serve):
    with pytest.raises(ValueError):
        ServeConfig(**serve)
