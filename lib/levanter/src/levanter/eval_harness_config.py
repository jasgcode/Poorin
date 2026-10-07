# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the LM Evaluation Harness, importable without loading the harness.

Training configs hold an ``LmEvalHarnessConfig``; ``levanter.eval_harness``, which loads the
inference engine, imports these classes from here.
"""

import dataclasses
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from tqdm_loggable.auto import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for one LM Evaluation Harness task.

    Fields left as ``None`` are omitted from the harness task dictionary so the
    harness can apply its own defaults.
    """

    task: str
    task_alias: str | None = None
    num_fewshot: int | None = None

    use_prompt: str | None = None
    description: str | None = None
    target_delimiter: str | None = None
    fewshot_delimiter: str | None = None
    doc_to_text: str | None = None
    doc_to_target: str | None = None
    doc_to_choice: str | None = None

    # Inline task-spec fields bypass registered-task override semantics when a
    # task definition is not present in lm-eval's registry.
    dataset_path: str | None = None
    dataset_name: str | None = None
    output_type: str | None = None
    test_split: str | None = None
    training_split: str | None = None
    validation_split: str | None = None
    fewshot_split: str | None = None
    metric_list: list[dict] | None = None
    tag: list[str] | None = None
    metadata: dict | None = None

    additional_stop_strings: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the task dictionary expected by LM Evaluation Harness."""
        return {key: value for key, value in dataclasses.asdict(self).items() if value is not None}


_TASK_DATASET_PATH_OVERRIDES = {
    "gsm8k": "openai/gsm8k",
    "gsm8k_cot": "openai/gsm8k",
}


def _task_config_with_dataset_path_override(task: TaskConfig) -> dict[str, object]:
    if task.dataset_path is None:
        dataset_path = _TASK_DATASET_PATH_OVERRIDES.get(task.task)
        if dataset_path is not None:
            task = dataclasses.replace(task, dataset_path=dataset_path)
    return task.to_dict()


def _call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 10,
    base_delay: float = 5.0,
    max_delay: float = 300.0,
) -> T:
    """
    Call a function with retry logic and exponential backoff.

    Handles HuggingFace rate limit errors (HTTP 429) by parsing the RateLimit header
    when available to determine appropriate wait time.

    Args:
        fn: The function to call (should take no arguments).
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        The result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt + 1 >= max_retries:
                break

            error_type = type(e).__name__
            error_msg = str(e)

            # Check if this is a rate limit error from HuggingFace
            wait_time = base_delay * (2**attempt)

            # Try to extract rate limit info from HuggingFace errors
            if "429" in error_msg or "Too Many Requests" in error_msg:
                # Try to parse RateLimit header if available
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    try:
                        rate_limit_header = e.response.headers.get("RateLimit", "")
                        if rate_limit_header:
                            # Format: "api|pages|resolvers;r=[remaining];t=[seconds until reset]"
                            rate_limit_wait = int(rate_limit_header.split(";")[-1].split("=")[-1])
                            wait_time = max(wait_time, rate_limit_wait + 10)
                    except Exception:
                        logger.debug("Failed to parse RateLimit header, using exponential backoff")

            # Add jitter to avoid thundering herd
            jitter = random.uniform(0, min(wait_time * 0.25, 30))
            wait_time = min(wait_time + jitter, max_delay)

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {error_type}: {error_msg}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed after {max_retries} attempts. Last error: {type(last_exception).__name__}: {last_exception}"
    ) from last_exception


@dataclass(frozen=True)
class SampleLoggingConfig:
    """Configuration for controlling how many sample generations we keep per benchmark."""

    log_all: bool = False
    """If True, keep every sample that the harness sees."""
    max_samples_per_benchmark: int | None = None
    """Optional cap on the number of samples to store for each benchmark."""

    def __post_init__(self):
        if self.max_samples_per_benchmark is not None and self.max_samples_per_benchmark < 0:
            raise ValueError("max_samples_per_benchmark must be non-negative or None")

    def should_log(self) -> bool:
        """Return True if any sample logging should occur."""
        if self.log_all:
            return True
        return self.max_samples_per_benchmark is not None and self.max_samples_per_benchmark > 0

    def allow_more(self, current_count: int) -> bool:
        """Return True if we should store another sample for the benchmark."""
        if not self.should_log():
            return False
        if self.log_all:
            return True
        assert self.max_samples_per_benchmark is not None
        return current_count < self.max_samples_per_benchmark


@dataclass(frozen=True)
class LmEvalHarnessConfig:
    """
    Configuration for running the LM Eval Harness.

    This class contains all the configuration options needed to run the LM Eval Harness
    on a Levanter model, including task specifications, generation parameters, and
    logging options.
    """

    task_spec: list[TaskConfig | str]
    max_examples: int | None = None
    max_length: int | None = None
    log_samples: bool = False
    bootstrap_iters: int = 0
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    confirm_run_unsafe_code: bool = True
    sample_logging: SampleLoggingConfig = dataclasses.field(default_factory=SampleLoggingConfig)
    generation_kwargs: dict = dataclasses.field(
        default_factory=lambda: {"max_gen_toks": 256, "temperature": 0.0, "n": 1, "seed": None}
    )
    """
    Default generation parameters for text generation tasks.

    Supported parameters:
    - max_gen_toks: Maximum number of tokens to generate (default: 256)
    - temperature: Sampling temperature, 0.0 for deterministic (default: 0.0)
    - n: Number of completions to generate per prompt (default: 1)
    - seed: Random seed for generation, None for random (default: None)

    These can be overridden on a per-request basis by the evaluation harness.
    """

    @property
    def max_gen_toks(self) -> int:
        """Backward compatibility property for max_gen_toks."""
        return self.generation_kwargs.get("max_gen_toks", 256)

    def to_task_spec(self) -> list[str | dict]:
        """
        Convert task specifications to a list of dictionaries or strings.

        Returns:
            List of task specifications, with TaskConfig objects converted to dictionaries
        """
        task_specs: list[str | dict] = []
        for task in self.task_spec:
            if isinstance(task, str):
                if task not in _TASK_DATASET_PATH_OVERRIDES:
                    task_specs.append(task)
                    continue
                task = TaskConfig(task=task)
            task_specs.append(_task_config_with_dataset_path_override(task))
        return task_specs

    def to_task_dict(self) -> dict:
        """
        Convert the task spec to a dictionary that the LM Eval Harness expects.

        This is a bit more complex than we'd like, because we want to run e.g. Hellaswag 0-shot and 10-shot in the same
        run, and LM Eval Harness doesn't seem to want to do that by default. So we need to do some hacky stuff to make
        it work.

        Uses retry logic with exponential backoff to handle HuggingFace rate limits when
        downloading evaluation datasets.
        """
        logger.info("Loading tasks...")
        import lm_eval.tasks as tasks  # noqa: PLC0415  # optional dep: lm_eval

        manager = tasks.TaskManager()
        # we need to do it this way b/c i can't figure out how to run e.g. hellaswag 0 shot and 10 shot in a single run
        this_tasks = {}
        for task in tqdm(self.to_task_spec()):
            try:
                if isinstance(task, str):
                    task_dict = _call_with_retry(lambda t=task: tasks.get_task_dict(t, manager))
                    this_tasks.update(task_dict)
                else:
                    our_name = task.get("task_alias", task["task"]) if isinstance(task, dict) else task
                    assert isinstance(our_name, str)
                    our_name = our_name.replace(" ", "_")
                    tasks_for_this_task_spec = self._get_task_and_rename(manager, our_name, task)
                    for k, v in tasks_for_this_task_spec.items():
                        if k in this_tasks:
                            raise ValueError(f"Task {k} already exists")
                        this_tasks[k] = v
            except Exception as e:
                logger.exception(f"Failed to load task {task}")
                raise ValueError(f"Failed to load task {task}") from e

        logger.info(f"Loaded {len(this_tasks)} tasks")
        return this_tasks

    def _get_task_and_rename(self, manager, our_name, task: dict | str):
        """
        Get a task from the task manager and rename it to our_name.
        LM Eval Harness doesn't seem to want to run multiple instances of the same task with different fewshot settings,
        (or other differences) so we need to hack around that.

        Uses retry logic with exponential backoff to handle HuggingFace rate limits.
        """
        import lm_eval.tasks as tasks  # noqa: PLC0415  # optional dep: lm_eval

        task_name = task if isinstance(task, str) else task["task"]

        task_dict = _call_with_retry(lambda: tasks.get_task_dict([task], manager))
        assert len(task_dict) == 1, f"Expected 1 task, got {len(task_dict)}"
        try:
            this_task = self._rename_tasks_for_eval_harness(task_dict, task_name, our_name)
        except AttributeError:
            logger.exception(f"Failed to rename task {task}: {task_dict}")
            raise ValueError(f"Failed to rename task {task}: {task_dict}")
        return this_task

    def _rename_tasks_for_eval_harness(self, this_task, lm_eval_task_name, our_name):
        from lm_eval.api.group import ConfigurableGroup  # noqa: PLC0415  # optional dep: lm_eval
        from lm_eval.api.task import ConfigurableTask  # noqa: PLC0415  # optional dep: lm_eval

        # hacky, but this allows us to run multiple instances of the same task with different fewshot settings
        if isinstance(this_task, dict):
            out = {}
            for k, v in this_task.items():
                v = self._rename_tasks_for_eval_harness(v, lm_eval_task_name, our_name)

                if isinstance(k, ConfigurableGroup):
                    k._config.group = self._replace_name_with_our_name(k.group, lm_eval_task_name, our_name)
                    out[k] = v
                elif isinstance(k, str):
                    k = self._replace_name_with_our_name(k, lm_eval_task_name, our_name)
                    if isinstance(v, dict):
                        subtask_list = self._get_child_tasks(v)
                        # ok so inexplicably, lm_eval_harness doesn't wrap the key in a ConfigurableGroup when you pass
                        # in a task dict (it seems like a mistake), so we need to do that here
                        # subtask is the name of all of the child tasks in v
                        group = ConfigurableGroup(config={"group": k, "task": subtask_list})
                        out[group] = v
                    else:
                        out[k] = v
                else:
                    raise ValueError(f"Unknown key type: {k}")

            return out

        elif isinstance(this_task, ConfigurableTask):
            this_task.config.task = self._replace_name_with_our_name(
                this_task.config.task, lm_eval_task_name, our_name
            )
            return this_task
        else:
            raise ValueError(f"Unknown task type: {this_task}")

    def _replace_name_with_our_name(self, lm_eval_name, lm_eval_prefix, our_name_prefix):
        if our_name_prefix.startswith(lm_eval_prefix):
            suffix = our_name_prefix[len(lm_eval_prefix) :]
            prefix = lm_eval_prefix
        else:
            suffix = ""
            prefix = our_name_prefix
        if lm_eval_prefix in lm_eval_name:
            lm_eval_name = lm_eval_name.replace(lm_eval_prefix, prefix) + suffix
        else:
            lm_eval_name = prefix + "_" + lm_eval_name + suffix
        return lm_eval_name

    def _get_child_tasks(self, task_group):
        from lm_eval.api.group import ConfigurableGroup  # noqa: PLC0415  # optional dep: lm_eval

        out = []
        for k, v in task_group.items():
            if isinstance(k, ConfigurableGroup):
                subtask_or_tasks = k.config.task
                if isinstance(subtask_or_tasks, str):
                    out.append(subtask_or_tasks)
                else:
                    out.extend(subtask_or_tasks)
            elif isinstance(k, str):
                out.append(k)
            else:
                raise ValueError(f"Unknown key type: {k}")

        return out
