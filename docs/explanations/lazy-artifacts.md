# Lazy artifacts

Marin's execution model is built around **lazy artifacts**: typed, identity-keyed handles
that describe what to build without doing any work at definition time. An experiment script
constructs `ArtifactStep[T]` handles, lowers them to a runnable step graph, and hands the
graph to `StepRunner`. Tokenization, training, and every other step execute exactly once and
are cached for future runs.

## Artifacts: `name@version` handles

A **handle** is an `ArtifactStep[T]` — a frozen object whose identity is its `name` and
`version`. Constructing one does not run anything. The type parameter `T` is the
materialized result type produced when the step runs:

- `ArtifactStep[TokenizedCache]` — a handle whose step produces a tokenized dataset
  (consumed by `train_lm` and other data assemblers).
- `ArtifactStep[LevanterCheckpoint]` — a handle whose step produces a Levanter model
  checkpoint (consumed by `train_lm`'s `init_from`).

The result types are `Artifact` subclasses that live with their producers:
`TokenizedCache` in `poorin.processing.tokenize.tokenize`, `LevanterCheckpoint` in
`poorin.training.training`. They describe the materialized result, not the handle itself.
Import the handle from `poorin.execution.lazy`:

```python
from poorin.execution.lazy import ArtifactStep              # the handle
from poorin.processing.tokenize.tokenize import TokenizedCache
from poorin.training.training import LevanterCheckpoint     # materialized result types
```

A handle's storage address is its explicit `{prefix}/{name}/{version}` path. There is no
content hash in the path. The `version` is a calendar version, `YYYY.MM.DD` (with an
optional `.N` suffix), or a `dev` string during development. Changing the name or version
creates a distinct artifact; changing how an existing `name@version` is built is refused
before anything runs (see [Recipe drift](#recipe-drift)).

In practice, experiment scripts rarely construct `ArtifactStep(...)` directly. Helpers such
as `tokenized`, `pretokenized`, and `train_lm` return the appropriate `ArtifactStep[T]` handle
from high-level arguments.

## Where pipeline code belongs

Construct graphs, bind steps, and choose artifact identities in `experiments/**`. Keep one-off
workflow code and experiment-specific tests beside the experiment, outside root `tests/**`
and `lib/**`. Put concrete behavior intended for reuse in `lib/**`, then call it from an
experiment step. New library code should not construct
`ArtifactStep` or `StepSpec` objects, compose a graph, or refer to specific artifact handles
or output locations. Library tests continue to cover reusable behavior.

`poorin.experiment.data` and `poorin.experiment.train` currently provide library builders that
construct handles. They remain supported APIs, but do not establish a location for new
pipeline builders. The `Artifact` and `ArtifactStep` types themselves remain in the
execution framework.

## Steps and StepContext

An `ArtifactStep` is flat — there is no separate recipe object. Its fields split cleanly
into identity (`name`, `version`, `artifact_type`) and build instructions:

```python
ArtifactStep(
    name="checkpoints/my-run",        # identity: the {name}/{version} address
    version="2026.06.28",             # identity: calendar version
    artifact_type=LevanterCheckpoint, # the materialized result type
    run=_train_job,                   # the step fn, or remote(fn, resources=…)
    build_config=build_config,        # assembles the config from the StepContext
    deps=(dataset,),                  # upstream handles this step reads
    runtime_args={"train_resources": resources},  # execution choices, excluded from identity
)
```

`build_config` is a pure function of a `StepContext`. The context is the dividing line
between what an artifact *is* and *where/how* it runs:

```python
@dataclass(frozen=True)
class StepContext:
    output_path: str    # this artifact's output path (excluded from fingerprint)
    prefix: str         # the live storage prefix (excluded)
    region: str | None  # the GCP region, resolved at run time (excluded)
    is_fingerprint: bool # True only while computing the fingerprint

    def artifact_path(self, dep: ArtifactStep) -> str: ...  # a dependency's output path
    def runtime_arg(self, key: str) -> Any: ...             # a step-declared runtime arg
```

Values written as literals in `build_config` — model architecture, hyperparameters, dep
versions — define the artifact and enter its fingerprint. Values pulled from `ctx` —
output paths, the prefix, the region, compute resources — are execution choices and are
excluded from the fingerprint.

A concrete example from `experiments/tutorials/train_tiny_model.py`:

```python
return train_lm(
    name=f"checkpoints/tiny-{data}-{device}",
    run_id=f"tiny-{data}-{device}",
    model=model,                 # literal → bears identity
    optimizer=AdamConfig(
        learning_rate=6e-4,      # literal → bears identity
        weight_decay=0.1,
    ),
    datasets={dataset(data): 1.0},  # literal → bears identity
    batch_size=batch_size,       # literal → bears identity
    ...
    resources=resources,         # runtime arg → excluded from fingerprint
)
```

`train_lm` constructs the `ArtifactStep` internally and passes `resources` as a
`runtime_arg`, so changing the hardware never re-fingerprints the checkpoint.

## Lowering and running

`run(*handles)` lowers the handle graph into the `StepSpec` graph that `StepRunner` executes and
builds it (independent steps in parallel) and **returns each handle's loaded, typed
artifact**, in argument order — so the driver gets the resolved values directly, no separate
read. `resolve(handle)` is `run` for a single handle, returning its one artifact.

```python
from poorin.execution.lazy import resolve

checkpoint = build()                      # returns ArtifactStep[LevanterCheckpoint]; nothing runs yet
result = resolve(checkpoint)              # materializes deps, runs training, returns LevanterCheckpoint
print(result.training_metrics().eval_loss)
```

Inside a step you never read a dependency this way — use `ctx.resolved(dep)`, which the runner
guarantees is materialized first. `run`/`resolve` are the driver-level entry points.

`StepRunner.run` applies two guards before executing each step:

1. **Cache check** — if the output records a different recipe, the run is refused (see
   [Recipe drift](#recipe-drift)). If the output path's `.executor_status` reads `SUCCESS`,
   skip. A `dev` version is mutable and always rebuilds. A step that failed earlier runs again.
2. **Lock** — prevents concurrent processes from building the same step twice.

The standard `main` block in a Poorin experiment wraps `build` in the shared command-line
options (`--version`, `--resume`, `--version-of`, `--set`, `--sweep`, `--show-config`,
`--accept-drift`, `--max-concurrent`, `--run`); without `--run` it prints the plan. `build` takes
an `overrides` keyword, which it applies to its config (see [Overrides and
sweeps](#overrides-and-sweeps)):

```python
from poorin.experiment.cli import experiment_main

if __name__ == "__main__":
    experiment_main(build)()
```

An experiment with options of its own stacks `@build_options` under its `@click.option`s, as
`experiments/grug/base/launch.py` does.

## Recipe drift

Each computed artifact records a **fingerprint** — a hash of its config, built with
dependency versions substituted in place of paths and placeholders for every value drawn
from `ctx`. The fingerprint captures hyperparameters and dep identities but is independent
of the storage prefix, region, or hardware.

An output records the fingerprint it was built from: in `.artifact.json` when the build
finished, or in `.artifact.started.json`, which the runner writes when a build starts. Before
anything runs, `run` checks every artifact in the graph against the fingerprint its output
records, including dependencies that the runner would skip behind an already-built consumer.
If any differ, the run is refused, and the error lists each changed artifact and its changed
config fields:

```text
datasets/toy@2026.06.28: the output records recipe 1d3f2a9c, but the recipe is now 8b04e6f1.
Changed config values:
  payload: 'a' -> 'b'
Change the version to build the new recipe, or pass --accept-drift datasets/toy to keep the output.
```

Two handles in one graph that build the same `name@version` from different recipes are refused
too, whether or not the output exists.

To build the new recipe, change the version:

```python
return train_lm(name="checkpoints/dclm_7b_1x", version="2026.07.01", ...)
```

To keep the existing output, pass `--accept-drift NAME` (or `run(..., accept_drift={"NAME"})`).
The output's record then carries the new fingerprint and keeps the old one in `accepted_from`,
so later runs are clean; a started build resumes under the new recipe. Naming an artifact
whose recipe did not change is an error.

A `dev` version string (e.g. `"dev"` or `"mylabel-dev"`) is never served from cache: it
always rebuilds. Its output still records its recipe, so rerunning the same `dev` version
with a changed recipe is refused. The command line gives each `--version dev` invocation a
fresh suffixed version, so only `--resume` reuses one.

Pinned artifacts (`override_path`) are not checked, because they point at existing data.

## Code provenance

The fingerprint covers an artifact's config. The code that ran it is outside the fingerprint, so
each output's record names the code: the commit, its subject, the branch, and any uncommitted
changes.

Before `--run` builds anything, the command line checks the working tree, untracked files
included. A calendar-versioned artifact builds only from committed code: if one would build from a
dirty tree, the run stops and lists the artifacts and the uncommitted paths. A run that builds only
`dev` artifacts saves the uncommitted changes as a snapshot commit under `refs/poorin/snapshots/`,
leaving the branch, index, and files alone, and the record names the snapshot.
`--allow-uncommitted` builds calendar artifacts from a snapshot as well. A run that builds nothing
saves no snapshot and is never refused. Outside a Git checkout the code cannot be recorded, so
calendar builds are refused there too unless `--allow-uncommitted` is passed.

The plan prints the code's one-line label and what `--run` would do:

```text
Prefix: /data/poorin
Code:   main@a0a406f + 3 uncommitted: llama.py, model.py, notes.md
        --run saves a snapshot
```

`--note TEXT` puts your own words in the label in place of the commit subject or file list. The
label also reaches W&B as the run config's `code` field.

`uv run --no-sync python -m poorin.experiment.runs list [FILTER]` lists the runs under the prefix,
newest first, one line each. `uv run --no-sync python -m poorin.experiment.runs show NAME@VERSION`
shows one run's code in full, with `git diff` and `git worktree add` commands for its snapshot.

Snapshot refs stay in the local repository: `git push origin main` does not send them, and
`git push origin 'refs/poorin/*:refs/poorin/*'` backs them up. A snapshot holds every untracked file
that `.gitignore` does not exclude, so check the working tree before pushing them anywhere public.

## Overrides and sweeps

`--set PATH=VALUE` overrides one field of the experiment's config by dotted path. The value is
YAML, decoded by draccus against the field's annotated type: `--set steps=200` gives an `int`, and
`--set optimizer='{type: muon, learning_rate: 0.02}'` replaces the optimizer with the class
registered as `muon`. Values are YAML 1.1, so a string such as `yes`, `on`, or `null` meant
literally must be quoted: `--set label="'yes'"`. `--sweep PATH=[V1, ...]` runs one trial per
value, as if each were passed to `--set`. Several `--sweep` options form their Cartesian product
(`poorin.experiment.cli.grid`), and each `--set` applies to every trial.

```bash
uv run --no-sync python experiments/grug/base/launch.py --version 2026.10.09 \
    --sweep 'optimizer.learning_rate=[1e-3, 3e-3, 1e-2]' --run
```

The experiment receives one trial's overrides as `overrides`
(`poorin.experiment.overrides.Overrides`) and uses them in three places, as `train_lm` and
`grug_base_trial` do:

- `overrides.apply(ctx, config)` inside `build_config`, so each override enters the fingerprint.
- `overrides.named(name)` for the artifact's name. The overrides become a suffix such as
  `grug/base-trial/optimizer.learning_rate=0.001,steps=200`, with numbers in one spelling, so
  `1e-3` and `0.001` name one path. When a value has a character outside `[A-Za-z0-9_.+-]` (a
  mapping, a quoted string, or a `,` or `=`), is longer than 40 characters, or the suffix would
  exceed 100 characters, the suffix is `set-<digest>`, a hash of the sorted overrides.
- `overrides.run_id(base)` for the tracker run ID, which appends the same digest, so each trial
  logs to its own run.

These errors are raised before anything runs: an unknown path, with the closest field names
suggested; a value that does not decode as the field's type; a path whose value comes from the
step context (the output path or a path built from it, a runtime argument such as the tracker, or
a dependency); one trial that sets a field twice, or a field and one beneath it; and an override
the experiment never applied.

`train_lm` sets some fields when the training job starts, after the overrides apply, so they are
set at run time and not overridable. A `--set` on one is accepted and enters the trial's name and
fingerprint, but the run never sees it: `hf_save_path`, `trainer.checkpointer.base_path`,
`trainer.checkpointer.temporary_base_path`, and `trainer.checkpointer.append_run_id_to_base_path`
(set from the output path), `data.auto_build_caches` (set from the pod config), and
`trainer.require_accelerator` when the job runs on CPU.

Trainings in one process run one at a time, at any `--max-concurrent`. Fray's local client runs
each job as a thread of the process, and Levanter's tracker and the W&B run belong to the process,
so a training that would start while another runs waits for it to finish
(`poorin.training.training.exclusive_accelerator`). Evaluations hold the same lock, so an
evaluation and a training, or two evaluations, also run one at a time. Other steps, such as tokenization, still run up
to `--max-concurrent` at once. Trainings also run one at a time in a sweep built in Python and passed to `run`, and in any
graph with several training steps.

With a calendar version, a rerun of a sweep builds only the trials that are missing, and a longer
value list builds only the new values. With `dev`, every invocation runs every trial at a fresh
version.

`--set` edits the finished config, so a value the experiment derives from another field in Python
does not follow it. The tutorial passes `seq_len=model.max_seq_len` to `train_lm`, so
`--set model.max_seq_len=512` leaves `train_seq_len` unchanged. `--show-config` prints each
trial's config without running anything.

## Pinning existing data

A handle with a pin (`pin=` on the data builders, `override_path` on `ArtifactStep`) points at
data that already exists instead of building it at `{prefix}/{name}/{version}`. A relative pin is
resolved against the prefix. A pin writes no record, and the drift check skips it, so pin only data
whose contents you trust not to change under the handle.

## Data builders

`poorin.experiment.data` provides concise builders that return `ArtifactStep[TokenizedCache]`
handles:

- `tokenized(name, *, tokenizer, version, source=…, paths=…)` — tokenize a HuggingFace dataset
  or a raw path glob into a Levanter cache. Provide exactly one of `source` or `paths`.
- `pretokenized(name, *, repo_id, tokenizer, version, revision=…)` — a pre-built Levanter cache hosted on
  HuggingFace (downloads rather than re-tokenizes).

For a custom single step (a filter, conversion, or format change), construct an
`ArtifactStep(...)` directly, or use `apply` for the common keyword-input shape:

```python
from poorin.execution.lazy import OUT, apply

# fn receives the resolved inputs as keyword arguments; OUT resolves to the output path,
# and any ArtifactStep input is resolved to its output path automatically.
shards = apply("data/sharded", shard_parquet, version="2026.06.28", source=raw, output_path=OUT)
```

## How this differs from the old content-addressed executor

The previous executor assigned each step an output path containing a hash of the step's
config and all its transitive dependencies. A hyperparameter change silently re-addressed
every downstream step, and the path gave no clue about what version of the data it held.

Lazy artifacts address data by an explicit `{name}/{version}`, so the storage path is
stable and human-readable. The fingerprint catches a config change under an existing
`name@version` and refuses the run until the author changes the version or accepts the
change. Bumping a version is an explicit author decision, not an automatic consequence of
any config change.
