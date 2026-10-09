# Typed overrides, sweeps, and fail-closed drift for experiments

Status: approved design, 2026-10-08; implemented 2026-10-09 (plan
`plans/2026-10-08-experiment-overrides.md`). Decision D92.

## TL;DR

Poorin's experiments are Python functions that return `ArtifactStep` handles, run through
`poorin.experiment.cli.build_options`. Changing a learning rate means editing code, sweeping means
writing a loop, and the plan an experiment prints without `--run` is one 4,000-character line. A
cached output is served when its recipe has changed: three kinds of change log a warning, and one
logs nothing.

This design brings Hydra's useful parts into that workflow, built on draccus (already a dependency,
and the parser behind Levanter's own configs) instead of Hydra:

- `--set PATH=VALUE` overrides any field of an experiment's config, typed by draccus.
- `--sweep 'PATH=[V1, V2, ...]'` runs one trial per value; several form a grid.
- The plan prints one line per artifact with its cache state, and `--show-config` prints each
  trial's identity config without running anything.
- A changed recipe under an existing output is an error, checked across the whole graph before
  anything runs, and `--accept-drift NAME` accepts a change on the record.

Each override enters the trial's fingerprint and its name, so an overridden run never lands on an
output built from another recipe. Hydra itself is out (D51 already chose the executor; OmegaConf
cannot represent several of Levanter's config types).

## Background

### How a run is identified today

An `ArtifactStep` (`lib/poorin/src/poorin/execution/lazy.py`) has a `name`, a `version`, and a
`build_config(ctx)` function. Its output lives at `{prefix}/{name}/{version}`. Its fingerprint is
an 8-hex MD5 of the canonical JSON of `build_config` run against a fingerprint context, in which
values pulled from the context (`ctx.output_path`, `ctx.runtime_arg(key)`) render as placeholders
such as `<output_path>`, and each dependency as `name@version`. Literals in `build_config` bear
identity; pulled values do not.

A version is a calendar version (`YYYY.MM.DD[.N]`, immutable) or a mutable `dev`/`<label>-dev`,
which `build_options` suffixes with a timestamp so each invocation writes to a fresh path.
`--resume V` reuses an earlier suffixed version. `--override NAME=VERSION` sets one artifact's
version.

### What the runner does with an existing output

A calendar-versioned step whose `.executor_status` reads `SUCCESS` is skipped, and its
dependencies are pruned from the schedule. Before serving it, `check_drift`
(`lib/poorin/src/poorin/execution/artifact.py`) compares fingerprints and, on a mismatch, logs a
field-level diff and serves the cached output anyway, unless the handle sets
`expected_fingerprint`, which no experiment does. A probe on toy steps (2026-10-08) confirmed four
cases where a stale output is served:

| Change under an existing calendar-versioned output | What is logged |
|---|---|
| A config value changed, same version | A drift warning |
| A dependency's version bumped | A drift warning; the new dependency version is not built either |
| A dependency's config changed, same dependency version | Nothing: the built consumer prunes the dependency, so its check never runs |
| A config class swapped for one with equal fields (`MuonConfig` for `GrugMuonConfig`) | Nothing: the fingerprint encodes a dataclass by its fields only |

Mutable versions skip the drift check and rebuild, so a `--resume` with a changed recipe continues
the old run's checkpoints under the new recipe without a word.

### Why draccus

draccus decodes a YAML value into any annotated field type, including unions and registry
subclasses selected by a `type:` key (`draccus.decode(OptimizerConfig, {"type": "muon", ...})`
returns a `MuonConfig`), and Levanter registers its codecs (`jmp.Policy`, `timedelta`, dtypes,
`PartitionSpec`) with it when `levanter.config` is imported. OmegaConf, which Hydra builds on,
accepts unions of primitive types only and knows nothing of draccus registries, so Levanter's
`OptimizerConfig` (`warmup: int | float`, `lr_schedule: LrSchedule | str`) would need an adapter.
Everything Poorin-specific (names, fingerprints, sweeps, the plan) is Poorin code under any choice.

## Command-line surface

`build_options` gains these options; everything else about it is unchanged.

```bash
# Override fields by dotted path; values are YAML, decoded against the field's type
python experiments/grug/base/launch.py --version dev \
    --set optimizer.learning_rate=1e-3 --set steps=200 --run

# Swap a sub-config by its registered name
python experiments/grug/base/launch.py --version dev \
    --set optimizer='{type: muon, learning_rate: 0.02}' --run

# Sweep: one trial per value; several --sweep options form a grid
python experiments/grug/base/launch.py --version 2026.10.09 \
    --sweep 'optimizer.learning_rate=[1e-3, 3e-3, 1e-2]' --max-concurrent 1 --run

# The plan plus each trial's identity config; runs nothing
python experiments/grug/base/launch.py --version dev --set steps=200 --show-config
```

(Run each with `uv run --no-sync`.)

| Option | Meaning |
|---|---|
| `--set PATH=VALUE` | Override one field of the experiment's config. Repeatable. |
| `--sweep PATH=[V1, ...]` | One trial per listed value, as if `--set PATH=Vi`. Repeatable; several form their Cartesian product, last axis fastest (`poorin.experiment.sweep.grid`). Combines with `--set`. |
| `--show-config` | Print the plan and each returned handle's identity config as YAML. An error with `--run`. |
| `--accept-drift NAME` | Accept a changed recipe for the artifact named `NAME` (the part before `@` in the plan) and re-stamp its record. Repeatable. |
| `--version-of NAME=VERSION` | The old `--override`, renamed so it is not confused with `--set`. Same behavior; no alias. |
| `--max-concurrent N` | Unchanged default of 8, except that a sweep with more than one trial requires it explicitly. Fray's `LocalClient` runs jobs as threads in one process, so the default would start up to 8 trainings on one GPU. |

Errors, all raised before any step runs:

- an unknown path, with the closest field names suggested (`difflib`);
- a value that does not decode as the field's type (`steps=2.5`);
- a path whose current value comes from the step context: a placeholder such as `<output_path>`
  or `<tracker>`, or a dependency reference. These are execution settings or graph structure, not
  recipe;
- the same path set twice in one trial;
- an override the experiment never applied (it ignores its `overrides` argument);
- an `--accept-drift` name that matches no drifted artifact;
- `--sweep` with more than one trial and no `--max-concurrent`.

### The plan

Without `--run`, the command prints the prefix and one line per artifact in the graph (deduplicated
by `name@version`, dependencies first): its cache state, its fingerprint, and `name@version`. Under
a drifted artifact it prints the field-level diff (`describe_drift`). With `--show-config` it then
prints, for each returned handle, its fingerprint payload as YAML, where `<output_path>` and the
other placeholders mark where execution settings begin.

```text
Run version: 20261009-101500-3f2a-dev (continue this run with --resume 20261009-101500-3f2a-dev)
Prefix: /tmp/poorin
  not built   8fcddfec  fineweb-edu-10M@2026.06.28
  mutable     41c0a9d2  users/johnnyas/grug/base-trial/steps=200@20261009-101500-3f2a-dev
```

## Overrides

A new module, `lib/poorin/src/poorin/experiment/overrides.py`, holds one trial's overrides:

```python
@dataclass(frozen=True)
class Override:
    path: tuple[str, ...]   # ("optimizer", "learning_rate")
    text: str               # the YAML value as typed: "1e-3"

    @classmethod
    def parse(cls, item: str) -> "Override": ...   # "PATH=VALUE"; rejects an empty path segment

    @property
    def label(self) -> str: ...   # "optimizer.learning_rate=0.001", numbers in one spelling


@dataclass(frozen=True)
class Overrides:
    """One trial's field overrides. ``apply`` runs inside ``build_config``, so the fingerprint sees them."""

    items: tuple[Override, ...] = ()

    def apply(self, ctx: StepContext, config: T) -> T: ...
    def named(self, name: str) -> str: ...
    def run_id(self, base: str) -> str: ...
    def unapplied(self) -> list[str]: ...
```

- `apply(ctx, config)` walks each path through the config's dataclass fields, decodes the value
  with `draccus.decode` against the field's annotated type (`typing.get_type_hints`), and rebuilds
  each level with `dataclasses.replace`. A value set on a registry-typed field replaces the whole
  node, so `optimizer={type: muon, ...}` swaps the class. Under the fingerprint context, a path
  whose current value is a context value is refused; `StepContext.is_context_value(value)` answers
  that, since `StepContext` is where placeholders and dependency references are defined. The
  fingerprint is always computed before a run (by the plan, the preflight check, and lowering), so
  a refused path fails before anything runs. The module imports `levanter.config` so Levanter's
  draccus codecs are registered.
- `named(name)` returns `name` with no overrides, else `name/<suffix>`. The suffix is the sorted
  labels joined by `,` when every normalized value matches `[A-Za-z0-9_.+-]{1,40}` and the whole
  suffix is at most 100 characters; otherwise `set-<digest>`. A value with `,` or `=` takes the
  digest form, because two different override sets could otherwise spell the same suffix. Numbers
  are normalized (`1e-3` and `0.001` both read `0.001`), so equal values reuse a path. Two spellings
  that decode equal but normalize differently (`1` and `1.0` for a float field) get two paths with
  the same fingerprint: a cache miss, never a stale hit.
- `run_id(base)` returns `base` with no overrides, else `{base}-{digest}`, so each trial has its
  own W&B run and log file. `digest` is `fingerprint_hash` of the sorted labels.
- `unapplied()` lists paths `apply` never consumed. Like `VersionCodex`'s unused-override check,
  it is instrumentation on a frozen instance.

`--set` edits the finished config, so a value an experiment derives from another field in Python
does not follow it: the tutorial sets `seq_len=model.max_seq_len`, so `--set model.max_seq_len=512`
leaves `train_seq_len` unchanged. `--show-config` shows this before a run.

### How an experiment uses them

`build_options` calls the experiment function with an `overrides: Overrides` keyword, once per
trial, all inside one `BuildContext`. `experiment_main(build)` does the same, so `build` takes
`overrides`. Grug's builder, changed lines only:

```python
def grug_base_trial(*, overrides: Overrides, finelog_url: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    base_name = "grug/base-trial"
    version = resolve_version(base_name, None)  # --version-of grug/base-trial=... reaches every trial

    def build_config(ctx: StepContext) -> GrugBaseLaunchConfig:
        return overrides.apply(ctx, GrugBaseLaunchConfig(...))  # the literals as today

    return ArtifactStep(
        name=user_namespaced_name(overrides.named(base_name), version),
        ...,
        runtime_args={
            "train_resources": _TRAIN_RESOURCES,
            "finelog_url": finelog_url,
            "run_id": overrides.run_id(f"grug-base-{version}"),
            "tracker": WandbConfig(project="poorin", tags=["grug", "template"], group="grug-base-trial"),
            "log_every": 1,
        },
    )
```

`train_lm` (`lib/poorin/src/poorin/experiment/train.py`) takes `overrides: Overrides = Overrides()`
and applies them to the inner `TrainLmConfig`, so its paths match Levanter's YAML
(`optimizer.learning_rate`, `trainer.num_train_steps`, `model.hidden_dim`). The tutorial passes its
`overrides` through.

### Tracker settings leave identity

The W&B config (project, tags, group, `save_code`), the run ID, and grug's `log_every` become
runtime arguments, as `resources` already is: `build_config` reads them through
`ctx.runtime_arg`, so they render as placeholders in the fingerprint, and changing a tag no longer
changes it. The W&B `replicate_path`, which is the step's output path, is set at run time: in
grug's `_trackers` from `launch.output_path`, and in `train_lm`'s `build_config` from
`ctx.output_path` (D94 records why it is not set in the run function). Overriding these settings
with `--set` is refused; they are edited in code.

## Identity and drift

### The config class enters the fingerprint

`_FingerprintEncoder` (`lib/poorin/src/poorin/execution/fingerprint.py`) encodes a dataclass as a
shallow dictionary of its fields plus a `__class__` key, so nested dataclasses reach the encoder
too and each level carries its class. The key is the draccus choice name for a `ChoiceRegistry`
subclass (`muon`, `grug_muon`), else the class's `__qualname__`. The module is left out so that
moving a class does not change fingerprints. The same canonical form is the record's human
`config`; consumers read that by key (`TokenizedCache` reads `tokenizer`, `format`, `tags`), so an
extra key is harmless.

Every existing fingerprint changes once. On 2026-10-08 the local prefix held three records;
`--accept-drift` absorbs them.

### Fingerprinting is strict

A value with no canonical form raises instead of falling back to a `vars()`-based encoding with a
warning. With drift fatal, an encoding that varies between processes would block runs, so the
fallback goes, and with it `set_strict` and `POORIN_FINGERPRINT_STRICT`. Both training roots
fingerprint under strict mode today, identically across processes.

`expected_fingerprint` is deleted. No experiment sets it, and the checks below cover what it
guarded.

### What an existing output records

At an artifact's path, its recorded recipe is the fingerprint in `.artifact.json` when the build
finished, else the fingerprint in the `.executor_info` stub the runner writes when a build starts.
The stub gains `fingerprint_payload`, so a started build can be diffed too.

A new module, `lib/poorin/src/poorin/execution/cache_state.py`, computes one handle's state for
the plan and the checks:

```python
class CacheState(StrEnum):
    PINNED = "pinned"                      # override_path: existing data, not checked
    MUTABLE = "mutable"                    # a dev version: always builds (a resumed one continues)
    NOT_BUILT = "not built"                # calendar version, nothing at the path
    STARTED = "started"                    # calendar version, a stub but no SUCCESS: the build restarts
    BUILT = "built"                        # calendar version, SUCCESS: served from cache
    DRIFTED = "built from another recipe"  # any version: the recorded fingerprint differs


@dataclass(frozen=True)
class HandleState:
    handle: ArtifactStep
    fingerprint: str
    state: CacheState
    recorded_fingerprint: str | None
    change: str  # describe_drift output; empty unless DRIFTED


def handle_state(handle: ArtifactStep) -> HandleState: ...
```

Pinned artifacts (`override_path`) are not checked, because they point at existing data. An
adopted artifact is checked: its record at the canonical address names its source, so re-adopting
from another source is refused.

`graph_handles` moves from `cli.py` to `lazy.py`, beside `run`.

### The check runs over the whole graph before a run

`lazy.run(*handles, max_concurrent, force_run_failed, accept_drift=())` computes `handle_state` for
every handle in the graph before starting the runner, including dependencies the runner would prune
behind a built consumer. Any `DRIFTED` handle not named in `accept_drift` raises
`FingerprintMismatchError`, listing every drifted artifact and its diff at once. This covers all
four cases in the background table, and a resumed dev run or a restarted calendar build whose
recipe changed, because their stubs record the recipe they started with. A fresh dev version has
nothing at its path, so it never drifts.

`check_drift` in the runner raises on a mismatch instead of warning, for code that lowers handles
and calls `StepRunner` directly. `--resume` on a returned handle with nothing at its path is an
error: there is nothing to resume, which usually means the `--set` flags differ from the original
run's.

### Accepting a change

For each name in `accept_drift`, `run` rewrites the recorded recipe before the runner starts:
`fingerprint` and `fingerprint_payload` in `.artifact.json` (or the fingerprint and payload in the
stub, for a started build), appending the old fingerprint to a new `ArtifactRecord.accepted_from`
list. The run then proceeds as if the recipe had not changed: a built output is served, and a
started build resumes under the new recipe.

## Sweeps

`build_options` expands `--sweep` with `poorin.experiment.sweep.grid` and calls the experiment once
per point with that point's `Overrides`, all inside one `BuildContext`. A dependency shared by the
trials is built at most once: the runner deduplicates steps by output path. With a calendar
version, a rerun after a crash runs only the trials that are missing, and widening the list runs
only the new values. With a `dev` version every invocation runs every trial at a fresh version, and
`--resume` resumes each. Trials run one after another with `--max-concurrent 1`.

`sweep.sweep(trial, **axes)` stays for sweeps written in Python; its docstring points at
`--sweep` for the command-line case.

## Build order

Four commits, each passing `tools.prune_check` and recording its decision:

1. **The plan and `--show-config`:** `cache_state`, the plan printer, `graph_handles` in `lazy.py`.
   No change to running.
2. **Fail-closed drift:** the class in the fingerprint, strict-only fingerprinting, the stub's
   payload, the whole-graph check, `--accept-drift`, the `--resume` check, and the deletion of
   `expected_fingerprint`. Absorbs the three local records with `--accept-drift`.
3. **Tracker settings out of identity**, in grug and `train_lm`.
4. **`--set` and `--sweep`:** the `overrides` module, the CLI options, the `--version-of` rename
   (with `VersionCodex.overrides` renamed `by_name`), and adoption in grug and the tutorial.

After commit 4, one GPU check: a two-trial grug sweep at a few steps with `--max-concurrent 1`,
confirming that two trainings run back to back in one process and log to two W&B runs.

## Tests

Behavior tests on toy steps under `tmp_path`, following `TESTING.md`:

- A `--set` run records the overridden value and lands at the override-named path; the same value
  spelled differently reuses that path; each refusal case in the error list is rejected before
  anything runs.
- A calendar-versioned `--sweep` run twice runs its trials once; widened by one value, it runs one
  trial.
- Each of the four stale cases is refused and names the changed field; a dependency the runner
  would prune is still checked.
- `--accept-drift` serves the output and records the old fingerprint in `accepted_from`.
- A resumed dev run with a changed recipe is refused; `--resume` with nothing at the path is
  refused.
- Two configs that differ only in class fingerprint differently; a registry subclass is keyed by
  its choice name.
- The plan reports each cache state.

## Documents

Updated in the commit that changes the behavior they describe: the "Advisory drift" section of
`docs/explanations/lazy-artifacts.md`, `.agents/skills/write-pipeline/references/artifact-binding.md`,
`experiments/grug/README.md` (the quickstart, and `--set` in place of copying a variant directory
for small changes), the `poorin.experiment.sweep` docstring, `lib/poorin/README.md`, and the module
rows of `docs/poorin/architecture.md`.

## Out of scope

Each gets a row in the Deferred table of `decisions.md`:

- writing the materialized config when a run starts (today only on success);
- grug's log file in the run's directory instead of `logs/` under the working directory;
- pinning the Hugging Face revision of grug's training data;
- named presets for `--set` values;
- adaptive search (Optuna), as a Python loop over `resolve` when a study needs it;
- a SLURM backend for Fray;
- forwarding `--set` to SkyRL runs as Hydra overrides (D68).
