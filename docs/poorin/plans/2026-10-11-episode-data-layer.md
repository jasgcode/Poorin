# Episode Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Poorin to Python 3.14, then add one data format for video world models, VLAs, and
WAMs: episode datasets built as artifacts and read as clips through Grain.

**Architecture:** A new package, `poorin.episodes`, holds the schema and per-field encodings
(`schema.py`, `encodings.py`), a helper that decodes any video file to frames (`video_files.py`),
the Zephyr build pipeline and the `EpisodeDataset` artifact type (`build.py`), and the Grain read
path (`read.py`). `poorin.experiment.data` gains an `episodes(...)` handle. Three example readers in
`experiments/datasets/episodes/` exercise the format and the throughput and smoke checks.

**Tech Stack:** Python 3.14, PyAV (H.264 through its bundled FFmpeg), ArrayRecord, msgpack, Grain,
Zephyr on Fray's local client, pandas/pyarrow for the index, JAX for placement, pytest.

**Spec:** `docs/poorin/specs/2026-10-11-episode-data-layer-design.md` (approved 2026-10-11).

## Global Constraints

- Work on the branch `feature/episode-data-layer`, cut from `main`. Never commit to `main`.
- Run every project command with `uv run --no-sync`. Never run a syncing `uv run`. After a
  dependency change, sync with `uv sync --locked --all-packages --extra=gpu --extra=lm_eval`.
- Unit tests run on CPU: `JAX_PLATFORMS=cpu uv run --no-sync pytest <file> -n 0`.
- Before each commit: `uv run --no-sync python -m tools.lint --fix`, then the full check
  `uv run --no-sync python -m tools.prune_check --phase <D110 or D111> --step "<the commit title>"`.
  The full check picks tests from the **uncommitted** change, so run it before committing and do not
  edit the tree while it runs. Commit the row it appends to `docs/poorin/prune-metrics.csv` with the
  task. A commit that changes only `pyproject.toml`/`uv.lock` (a dependency commit) runs
  `tools.lint` and `tools.affected_tests` instead of the full check.
- Each dependency change is its own commit. Check its `uv.lock` diff for packages whose versions
  moved without being asked (Task 1, Step 4 has the script).
- ty passes with warnings as errors: `uvx ty@0.0.84 check --error-on-warning` (`tools.lint` runs it).
- Lines are at most 121 characters.
- Commit titles: imperative, at most 72 characters, a `[scope]` prefix (`[poorin]`, `[levanter]`).
  The body says what changed and why. No `Co-Authored-By` trailer and no "Generated with" line.
- All imports at the top of the file. No `TYPE_CHECKING` guards. Top-level constants for magic
  strings and numbers. No `*_utils.py` modules. No `__all__` in `__init__.py`.
- Tests follow `TESTING.md`: behavior through public interfaces, real files in `tmp_path`, videos
  generated in the test, no downloads, no mocks of PyAV, ArrayRecord, Grain, or Zephyr.
- `poorin.episodes` imports nothing from `levanter`.
- File names inside a dataset are exactly `dataset.json`, `episodes.parquet`, and
  `shard-{n:05d}.array_record`.
- A record is a msgpack map from field name to bytes. Never pickle.
- Video streams: `yuv420p`, no B-frames (`bf=0`), a keyframe exactly every `keyframe_interval`
  frames (`g`, `keyint_min`, `sc_threshold=0`), frame `i` has `pts = i` in a time base of
  `1 / step_rate`.
- Every field of a dataset shares one step clock, `step_rate` steps per second.

## Review Focus

1. **A clip starting just after a keyframe, or at the last valid start.** Decoding seeks backward
   to a keyframe and must return exactly the requested steps. Pinned in Task 2 (starts 0, 1, 15,
   16, 17, and the last start) and Task 4 (clip at the last valid start of an episode).
2. **A non-integer step rate such as 29.97.** The frame index is recovered from `pts * time_base *
   step_rate`; rounding must land on the right frame. Pinned in Task 2.
3. **An episode shorter than one clip.** It contributes no clips and must not raise or produce a
   short clip. Pinned in Task 4.
4. **A reader that yields a field with the wrong shape, a missing field, or fields of unequal
   length.** The build fails and names the episode and the field. Pinned in Task 3.
5. **Two source units yielding the same episode id.** The build fails and names the id, instead of
   writing an index where one row shadows another. Pinned in Task 3.

---

### Task 0: Branch, spec, plan, and D110

**Files:**
- Create: `docs/poorin/specs/2026-10-11-episode-data-layer-design.md` (already written)
- Create: `docs/poorin/plans/2026-10-11-episode-data-layer.md` (this file)
- Modify: `docs/poorin/decisions.md` (append D110)

- [ ] **Step 1: Cut the branch**

```bash
git switch -c feature/episode-data-layer main
```

- [ ] **Step 2: Append D110 to `docs/poorin/decisions.md`**

Append after D109, in the file's format:

```markdown
### D110. An episode data layer, read through Grain

- **When:** 2026-10-11. **Status:** Adopted; implementing. Spec:
  `specs/2026-10-11-episode-data-layer-design.md`; plan: `plans/2026-10-11-episode-data-layer.md`.
- **Decision:** Sub-projects 3, 4, and 5 come next, in that order; 1b, 2, 6, 7, and 8 wait.
  Episode data (video, trajectories, latents) gets its own format: a schema, a parquet index, and
  ArrayRecord shards of msgpack records, one per episode. It reads through Grain. Text stays on
  Levanter's caches, and Levanter stays the training library. Every field of a dataset shares one
  step clock. How a field is stored is a per-field encoding (`raw` or `video` to start), not a
  property of the format.
- **Why:** The owner, 2026-10-11, has not chosen datasets and wants a foundation that does not fix
  the storage choice. Levanter's data stack assumes items that cost nothing to decode; Grain keeps
  its deterministic, resumable data order and adds decoding in worker processes. The owner's
  `open-dreamer` already uses Grain and ArrayRecord.
- **Alternatives:** Extend Levanter's cache and loader with field encodings; adopt LeRobot v3 as the
  on-disk format; a separate rate per field; CoinRun as an example (Procgen has no wheels above
  Python 3.10).
- **Cost if wrong:** Two data stacks persist. If Grain's worker processes cannot meet 1,000 clips
  per second, the read path changes encoding or moves decoding to the GPU.
```

- [ ] **Step 3: Commit**

```bash
git add docs/poorin/specs/2026-10-11-episode-data-layer-design.md \
  docs/poorin/plans/2026-10-11-episode-data-layer.md docs/poorin/decisions.md
git commit -m "[poorin] Add the episode data layer spec and plan (D110)" \
  -m "Sub-project 3 adds one data format for video, robot trajectories, and latents, read as clips through Grain. Text stays on Levanter's caches. D110 records the choice and the sub-project order the owner set."
```

---

### Task 1: Move to Python 3.14 (D111)

**Files:**
- Modify: `.python-version`
- Modify: `pyproject.toml` (`requires-python`, `[tool.ruff] target-version`, `[tool.ty.environment] python-version`)
- Modify: `lib/poorin/pyproject.toml`, `lib/fray/pyproject.toml`, `lib/finelog/pyproject.toml`,
  `lib/zephyr/pyproject.toml`, `lib/rigging/pyproject.toml`, `lib/levanter/pyproject.toml`,
  `lib/finestore/pyproject.toml`, `lib/dupekit/pyproject.toml` (`requires-python`; Levanter's
  `target-version`)
- Modify: `uv.lock`
- Modify: `AGENTS.md`, `.readthedocs.yaml`, `lib/levanter/.readthedocs.yaml`,
  `docs/tutorials/installation.md`, `docs/dev-guide/building-docs.md`,
  `.agents/skills/add-pallas-kernel/docs/kernel-sources.md`
- Modify: `docs/poorin/decisions.md` (append D111)

**Interfaces:**
- Produces: a `.venv` on CPython 3.14 that every later task uses.

- [ ] **Step 1: Install Python 3.14**

```bash
uv python install 3.14
uv python find 3.14
```

Expected: a path to a CPython 3.14 interpreter.

- [ ] **Step 2: Raise the version everywhere it is pinned**

- `.python-version`: `3.14`.
- Every `requires-python` listed under Files: `">=3.14"`. This removes the `<3.14` caps Marin set
  in poorin, fray, finelog, zephyr, and rigging (including the stray one under `[build-system]` in
  `lib/poorin/pyproject.toml`, which is deleted).
- `pyproject.toml`: `target-version = "py314"` under `[tool.ruff]`; `python-version = "3.14"` under
  `[tool.ty.environment]`.
- `lib/levanter/pyproject.toml`: `target-version = "py314"`.
- `.readthedocs.yaml` and `lib/levanter/.readthedocs.yaml`: `python: "3.14"`.
- `AGENTS.md`: `- Python >= 3.14.`
- `docs/tutorials/installation.md`: "Python 3.14 or higher" and `uv venv --python 3.14`.
- `docs/dev-guide/building-docs.md`: "Python 3.14 or higher".
- `.agents/skills/add-pallas-kernel/docs/kernel-sources.md`: `.venv/lib/python3.14/site-packages/...`.
- Leave `lib/dupekit/rust/Cargo.toml`'s `abi3-py312` alone: an abi3 wheel built for 3.12 loads on
  3.14. Leave `lib/levanter/infra/helpers/setup-tpu-vm*.sh` alone: they provision TPU VMs, which
  sub-project 2 owns.
- `lib/poorin/pyproject.toml`'s `backports-zstd ... python_version < '3.14'` marker stays; on 3.14
  the standard library provides `compression.zstd` and the marker drops the backport.

- [ ] **Step 3: Relock**

```bash
cp uv.lock /tmp/uv.lock.py312
uv lock
```

Expected: success. If a package has no 3.14 wheel and no sdist that builds, `uv lock` or the sync in
Step 5 fails naming it. Stop there and report the package to the owner; do not pin around it.

- [ ] **Step 4: Check which versions moved**

```bash
uv run --no-sync python - <<'EOF'
import tomllib
def versions(path):
    with open(path, "rb") as f:
        return {p["name"]: p["version"] for p in tomllib.load(f)["package"]}
old, new = versions("/tmp/uv.lock.py312"), versions("uv.lock")
for name in sorted(old.keys() | new.keys()):
    if old.get(name) != new.get(name):
        print(f"{name}: {old.get(name)} -> {new.get(name)}")
EOF
```

Expected: the backport packages that only served Python below 3.14 disappear (for example
`backports-zstd`). Any other moved version is listed in D111 with its reason; a move that has no
reason is reverted with `uv lock --upgrade-package NAME==OLD`.

- [ ] **Step 5: Build a fresh environment**

```bash
mv .venv /tmp/venv-py312-rollback
uv sync --locked --all-packages --extra=gpu --extra=lm_eval
uv run --no-sync python -c "import sys, jax; print(sys.version); print(jax.devices())"
```

Expected: `3.14.x` and `[CudaDevice(id=0)]`. Keep `/tmp/venv-py312-rollback` until the owner
approves the task; deleting it is the owner's call.

- [ ] **Step 6: Lint the whole tree and review the rewrite**

```bash
uv run --no-sync python -m tools.lint --fix --all-files
git diff --stat
```

Ruff's `py314` target can rewrite code, for example removing quotes from forward-reference
annotations (PEP 649 makes annotations lazy). Read the diff. Keep mechanical rewrites; revert and
report anything that changes behavior.

- [ ] **Step 7: Run every suite on CPU, then the GPU kernels**

```bash
uv run --no-sync python -m tools.affected_tests --all
uv run --no-sync pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -n 0
```

Expected: both pass. The second runs on the GPU (no `JAX_PLATFORMS=cpu`) and covers the CuTe DSL
kernels behind FlashAttention 4, the dependency most likely to break on a new Python.

- [ ] **Step 8: Append D111 to `docs/poorin/decisions.md`**

```markdown
### D111. Python 3.14

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** Poorin requires Python 3.14. Every `requires-python` is `>=3.14`; the `<3.14` caps
  Marin set on five libraries are gone. Ruff and ty target 3.14.
- **Why:** The owner asked, 2026-10-11. Every compiled dependency that matters publishes 3.14
  wheels (jaxlib and its CUDA plugin, tensorstore, torch, the CuTe DSL behind FlashAttention 4,
  grain, array-record, av). Lazy annotations (PEP 649) remove quoted forward references. Moving
  before sub-project 3 locks its new dependencies once.
- **Versions that moved:** <the Step 4 list, one line each with its reason>.
- **Alternatives:** Python 3.13, which the lock already resolved; staying on 3.12.
- **Cost if wrong:** A dependency added later may lack 3.14 wheels and need building from source.
```

Replace the "Versions that moved" line's placeholder with Step 4's actual output before committing.

- [ ] **Step 9: Full check and commit**

```bash
uv run --no-sync python -m tools.prune_check --phase D111 --step "Move to Python 3.14"
git add -A
git commit -m "[poorin] Move to Python 3.14 (D111)" \
  -m "Every requires-python is >=3.14, dropping the <3.14 caps inherited from Marin; ruff and ty target 3.14. The lock is regenerated; D111 lists each version that moved."
```

Expected: the full check passes and appends one metrics row.

---

### Task 2: Schema, encodings, and video files

**Files:**
- Modify: `lib/poorin/pyproject.toml`, `uv.lock` (add `av`; its own commit)
- Create: `lib/poorin/src/poorin/episodes/__init__.py`
- Create: `lib/poorin/src/poorin/episodes/schema.py`
- Create: `lib/poorin/src/poorin/episodes/encodings.py`
- Create: `lib/poorin/src/poorin/episodes/video_files.py`
- Test: `tests/episodes/__init__.py`, `tests/episodes/test_encodings.py`, `tests/episodes/test_video_files.py`

**Interfaces:**
- Produces:
  - `RawEncoding()`, `VideoEncoding(codec: str = "libx264", keyframe_interval: int = 16, crf: int = 18)`,
    `FieldEncoding = RawEncoding | VideoEncoding`.
  - `FieldKind` (`StrEnum`: `VIDEO = "video"`, `ARRAY = "array"`).
  - `FieldSpec(kind: FieldKind, dtype: str, shape: tuple[int, ...], encoding: FieldEncoding)`.
  - `DatasetSchema(step_rate: float, fields: Mapping[str, FieldSpec])` with
    `to_json() -> dict` and `DatasetSchema.from_json(data: dict) -> DatasetSchema`.
  - `Episode(id: str, fields: Mapping[str, np.ndarray], metadata: Mapping[str, Any])`.
  - `validate_episode(schema: DatasetSchema, episode: Episode) -> int` (returns the length; raises
    `ValueError` naming the episode and field).
  - `encode_field(spec: FieldSpec, step_rate: float, values: np.ndarray) -> bytes`.
  - `decode_field(spec: FieldSpec, step_rate: float, data: bytes, length: int, steps: range) -> np.ndarray`
    (returns `[len(steps), *spec.shape]`).
  - `load_video(path: str, *, size: tuple[int, int], step_rate: float) -> np.ndarray`
    (`uint8 [T, H, W, 3]`, resized on the shorter side and center-cropped, resampled to `step_rate`).

- [ ] **Step 1: Add PyAV in its own commit**

In `lib/poorin/pyproject.toml`'s `dependencies`, add `"av>=19.0.1"`. Then:

```bash
uv lock
uv sync --locked --all-packages --extra=gpu --extra=lm_eval
uv run --no-sync python -c "import av; assert 'libx264' in av.codecs_available; print(av.__version__)"
uv run --no-sync python -m tools.lint --fix
git add lib/poorin/pyproject.toml uv.lock
git commit -m "[poorin] Add PyAV for episode video" \
  -m "PyAV bundles its own FFmpeg with libx264, so episode datasets encode and decode H.264 without the system ffmpeg."
```

Check the lock diff with Task 1 Step 4's script against the previous lock; only `av` should be new.

- [ ] **Step 2: Write the failing encoding tests**

`tests/episodes/__init__.py` is empty. `tests/episodes/test_encodings.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from poorin.episodes.encodings import decode_field, encode_field
from poorin.episodes.schema import (
    DatasetSchema,
    Episode,
    FieldKind,
    FieldSpec,
    RawEncoding,
    VideoEncoding,
    validate_episode,
)

KEYFRAME_INTERVAL = 16


def moving_pattern(length: int, height: int = 64, width: int = 64) -> np.ndarray:
    """Frames whose content changes every step, so a frame from the wrong step is detectable."""
    t = np.arange(length)[:, None, None]
    y = np.arange(height)[None, :, None]
    x = np.arange(width)[None, None, :]
    channels = np.broadcast_arrays((x * 4 + t * 3) % 256, (y * 4 + t * 5) % 256, ((x + y) * 2) % 256)
    return np.stack(channels, axis=-1).astype(np.uint8)


def video_spec(height: int = 64, width: int = 64) -> FieldSpec:
    return FieldSpec(
        kind=FieldKind.VIDEO,
        dtype="uint8",
        shape=(height, width, 3),
        encoding=VideoEncoding(keyframe_interval=KEYFRAME_INTERVAL),
    )


def test_raw_field_round_trips_exactly():
    spec = FieldSpec(kind=FieldKind.ARRAY, dtype="float32", shape=(7,), encoding=RawEncoding())
    values = np.random.default_rng(0).normal(size=(40, 7)).astype(np.float32)
    data = encode_field(spec, 20.0, values)
    np.testing.assert_array_equal(decode_field(spec, 20.0, data, 40, range(0, 40)), values)
    np.testing.assert_array_equal(decode_field(spec, 20.0, data, 40, range(5, 30, 3)), values[5:30:3])


@pytest.mark.parametrize("step_rate", [20.0, 12.0, 29.97])
def test_video_clip_equals_the_same_steps_of_a_full_decode(step_rate):
    frames = moving_pattern(256)
    spec = video_spec()
    data = encode_field(spec, step_rate, frames)
    full = decode_field(spec, step_rate, data, 256, range(0, 256))
    for start in (0, 1, KEYFRAME_INTERVAL - 1, KEYFRAME_INTERVAL, KEYFRAME_INTERVAL + 1, 100, 240):
        clip = decode_field(spec, step_rate, data, 256, range(start, start + 16))
        np.testing.assert_array_equal(clip, full[start : start + 16])
    skipped = decode_field(spec, step_rate, data, 256, range(37, 83, 3))
    np.testing.assert_array_equal(skipped, full[37:83:3])


def test_video_decode_stays_close_to_the_original():
    frames = moving_pattern(64)
    spec = video_spec()
    decoded = decode_field(spec, 20.0, encode_field(spec, 20.0, frames), 64, range(0, 64))
    mse = np.mean((decoded.astype(np.float64) - frames) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    # yuv420p halves chroma resolution; this high-frequency color pattern measured 26.9 dB at crf 18.
    assert psnr > 24


def test_schema_round_trips_through_json():
    schema = DatasetSchema(
        step_rate=20.0,
        fields={"video": video_spec(), "action": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding())},
    )
    assert DatasetSchema.from_json(schema.to_json()) == schema


def test_an_episode_that_breaks_the_schema_names_the_episode_and_field():
    schema = DatasetSchema(
        step_rate=20.0,
        fields={"video": video_spec(), "action": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding())},
    )
    good = {"video": moving_pattern(10), "action": np.zeros((10, 7), np.float32)}
    assert validate_episode(schema, Episode(id="ok", fields=good, metadata={})) == 10

    wrong_shape = {**good, "action": np.zeros((10, 6), np.float32)}
    with pytest.raises(ValueError, match=r"episode 'e1'.*field 'action'"):
        validate_episode(schema, Episode(id="e1", fields=wrong_shape, metadata={}))
    missing = {"video": good["video"]}
    with pytest.raises(ValueError, match=r"episode 'e2'.*field 'action'"):
        validate_episode(schema, Episode(id="e2", fields=missing, metadata={}))
    unequal = {**good, "action": np.zeros((9, 7), np.float32)}
    with pytest.raises(ValueError, match=r"episode 'e3'.*field 'action'"):
        validate_episode(schema, Episode(id="e3", fields=unequal, metadata={}))
```

- [ ] **Step 3: Run them to see them fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes/test_encodings.py -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'poorin.episodes'`.

- [ ] **Step 4: Write `schema.py`**

`lib/poorin/src/poorin/episodes/__init__.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Episode datasets: per-step fields for video world models, VLAs, and WAMs, read as clips."""
```

`lib/poorin/src/poorin/episodes/schema.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The episode format's schema: fields, their encodings, and one step clock per dataset.

An episode is a set of named per-step arrays. A plain video is an episode with no action field.
Each field declares its per-step shape and dtype and how it is stored; ``dataset.json`` is
:meth:`DatasetSchema.to_json`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

RAW_ENCODING_NAME = "raw"
VIDEO_ENCODING_NAME = "video"


class FieldKind(StrEnum):
    VIDEO = "video"
    ARRAY = "array"


@dataclass(frozen=True)
class RawEncoding:
    """The array's bytes, C order: actions, proprioception, latents."""


@dataclass(frozen=True)
class VideoEncoding:
    """One compressed stream per episode with a keyframe every ``keyframe_interval`` steps, so a clip
    decodes from the nearest keyframe instead of the episode's start."""

    codec: str = "libx264"
    keyframe_interval: int = 16
    crf: int = 18


FieldEncoding = RawEncoding | VideoEncoding


@dataclass(frozen=True)
class FieldSpec:
    kind: FieldKind
    dtype: str
    shape: tuple[int, ...]
    """Per step; an episode of length T stores ``[T, *shape]``."""
    encoding: FieldEncoding

    def __post_init__(self) -> None:
        if isinstance(self.encoding, VideoEncoding):
            if self.dtype != "uint8" or len(self.shape) != 3 or self.shape[2] != 3:
                raise ValueError(f"a video-encoded field is uint8 [H, W, 3], not {self.dtype} {list(self.shape)}")
            if self.shape[0] % 2 or self.shape[1] % 2:
                raise ValueError(f"yuv420p video needs an even height and width, not {list(self.shape[:2])}")


@dataclass(frozen=True)
class DatasetSchema:
    step_rate: float
    """Steps per second, shared by every field."""
    fields: Mapping[str, FieldSpec]

    def to_json(self) -> dict:
        return {"step_rate": self.step_rate, "fields": {name: _field_json(spec) for name, spec in self.fields.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "DatasetSchema":
        return cls(
            step_rate=float(data["step_rate"]),
            fields={name: _field_from_json(spec) for name, spec in data["fields"].items()},
        )


@dataclass(frozen=True)
class Episode:
    id: str
    fields: Mapping[str, np.ndarray]
    """Field name to ``[length, *shape]``; every field has the same length."""
    metadata: Mapping[str, Any]
    """Per-episode values, stored in the index as JSON: a language instruction, an embodiment, a source."""


def validate_episode(schema: DatasetSchema, episode: Episode) -> int:
    """The episode's length, after checking it against ``schema``. Raises ``ValueError`` naming the
    episode and field on a missing or extra field, a wrong dtype or per-step shape, or unequal lengths."""
    if set(episode.fields) != set(schema.fields):
        field = sorted(set(episode.fields) ^ set(schema.fields))[0]
        raise ValueError(f"episode {episode.id!r}: field {field!r} is in only one of the episode and the schema")
    lengths = {name: len(values) for name, values in episode.fields.items()}
    length = next(iter(lengths.values()))
    for name, spec in schema.fields.items():
        values = episode.fields[name]
        if values.dtype != np.dtype(spec.dtype) or values.shape[1:] != spec.shape:
            raise ValueError(
                f"episode {episode.id!r}: field {name!r} is {values.dtype} {list(values.shape[1:])} per step, "
                f"the schema says {spec.dtype} {list(spec.shape)}"
            )
        if lengths[name] != length:
            raise ValueError(f"episode {episode.id!r}: field {name!r} has {lengths[name]} steps, others have {length}")
    return length


def _field_json(spec: FieldSpec) -> dict:
    match spec.encoding:
        case RawEncoding():
            encoding: dict = {"name": RAW_ENCODING_NAME}
        case VideoEncoding(codec=codec, keyframe_interval=interval, crf=crf):
            encoding = {"name": VIDEO_ENCODING_NAME, "codec": codec, "keyframe_interval": interval, "crf": crf}
    return {"kind": spec.kind.value, "dtype": spec.dtype, "shape": list(spec.shape), "encoding": encoding}


def _field_from_json(data: dict) -> FieldSpec:
    encoding = dict(data["encoding"])
    name = encoding.pop("name")
    if name == RAW_ENCODING_NAME:
        parsed: FieldEncoding = RawEncoding(**encoding)
    elif name == VIDEO_ENCODING_NAME:
        parsed = VideoEncoding(**encoding)
    else:
        raise ValueError(f"unknown field encoding {name!r}")
    return FieldSpec(kind=FieldKind(data["kind"]), dtype=data["dtype"], shape=tuple(data["shape"]), encoding=parsed)
```

- [ ] **Step 5: Write `encodings.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode a field's ``[length, *shape]`` array to bytes, and decode any steps of it back.

A new encoding is one class in :mod:`poorin.episodes.schema` and one branch in each function here.
"""

import io
from fractions import Fraction

import av
import numpy as np

from poorin.episodes.schema import FieldSpec, RawEncoding, VideoEncoding

VIDEO_CONTAINER = "mp4"
VIDEO_PIXEL_FORMAT = "yuv420p"
FRAME_FORMAT = "rgb24"
# Keeps a rate such as 29.97 exact as 30000/1001.
MAX_RATE_DENOMINATOR = 1001


def encode_field(spec: FieldSpec, step_rate: float, values: np.ndarray) -> bytes:
    match spec.encoding:
        case RawEncoding():
            return np.ascontiguousarray(values, dtype=spec.dtype).tobytes()
        case VideoEncoding() as encoding:
            return _encode_video(encoding, step_rate, values)


def decode_field(spec: FieldSpec, step_rate: float, data: bytes, length: int, steps: range) -> np.ndarray:
    """Steps ``steps`` of a field stored as ``data``, as ``[len(steps), *spec.shape]``."""
    match spec.encoding:
        case RawEncoding():
            return np.frombuffer(data, dtype=spec.dtype).reshape(length, *spec.shape)[steps.start : steps.stop : steps.step]
        case VideoEncoding():
            return _decode_video(step_rate, data, steps)


def _encode_video(encoding: VideoEncoding, step_rate: float, frames: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=VIDEO_CONTAINER) as container:
        stream = container.add_stream(encoding.codec, rate=Fraction(step_rate).limit_denominator(MAX_RATE_DENOMINATOR))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = VIDEO_PIXEL_FORMAT
        interval = str(encoding.keyframe_interval)
        # No B-frames and no scene-cut keyframes: frame i decodes at pts i and keyframes fall exactly
        # every keyframe_interval frames, which the seek in _decode_video relies on.
        stream.options = {"g": interval, "keyint_min": interval, "sc_threshold": "0", "bf": "0", "crf": str(encoding.crf)}
        for index, frame in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(frame, format=FRAME_FORMAT)
            video_frame.pts = index
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode())
    return buffer.getvalue()


def _decode_video(step_rate: float, data: bytes, steps: range) -> np.ndarray:
    wanted = set(steps)
    decoded: dict[int, np.ndarray] = {}
    with av.open(io.BytesIO(data), mode="r", format=VIDEO_CONTAINER) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        container.seek(int(steps.start / step_rate / time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            assert frame.pts is not None
            index = round(float(frame.pts * time_base) * step_rate)
            if index >= steps.stop:
                break
            if index in wanted:
                decoded[index] = frame.to_ndarray(format=FRAME_FORMAT)
    return np.stack([decoded[index] for index in steps])
```

If ty rejects the `match` without a default branch, add `case _: raise TypeError(...)`; ty treats
the union as exhaustive in 0.0.84, so expect it not to.

- [ ] **Step 6: Run the encoding tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes/test_encodings.py -n 0`
Expected: PASS.

- [ ] **Step 7: Write the failing `load_video` test**

`tests/episodes/test_video_files.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import av
import numpy as np
from poorin.episodes.video_files import load_video

SOURCE_RATE = 30
SOURCE_FRAMES = 60


def write_source_video(path, height: int, width: int) -> None:
    """A source video whose red channel encodes the frame number, so resampling is checkable."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=SOURCE_RATE)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"bf": "0", "crf": "0"}
        for index in range(SOURCE_FRAMES):
            frame = np.zeros((height, width, 3), np.uint8)
            frame[..., 0] = index * 4
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = index
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode())


def test_load_video_resizes_crops_and_resamples(tmp_path):
    path = tmp_path / "source.mp4"
    write_source_video(path, height=90, width=160)

    frames = load_video(str(path), size=(32, 32), step_rate=10.0)

    # 60 frames at 30 fps is 2 seconds, so 20 steps at 10 Hz.
    assert frames.shape == (20, 32, 32, 3)
    assert frames.dtype == np.uint8
    # Step i is the source frame nearest i / 10 s, which is frame 3 * i.
    red = frames[:, 16, 16, 0].astype(int)
    np.testing.assert_allclose(red, np.arange(20) * 3 * 4, atol=6)
```

- [ ] **Step 8: Run it to see it fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes/test_video_files.py -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'poorin.episodes.video_files'`.

- [ ] **Step 9: Write `video_files.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode a video file to frames at a dataset's size and step rate, for source readers."""

import av
import numpy as np

FRAME_FORMAT = "rgb24"


def load_video(path: str, *, size: tuple[int, int], step_rate: float) -> np.ndarray:
    """Every step of the video at ``path`` as ``uint8 [T, height, width, 3]``.

    Frames are scaled so the shorter side matches and then center-cropped, and resampled to
    ``step_rate`` by taking the source frame nearest each step's time.
    """
    height, width = size
    times: list[float] = []
    frames: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        scale = max(height / stream.height, width / stream.width)
        scaled_height, scaled_width = round(stream.height * scale), round(stream.width * scale)
        for frame in container.decode(stream):
            assert frame.time is not None
            times.append(frame.time)
            scaled = frame.reformat(width=scaled_width, height=scaled_height, format=FRAME_FORMAT).to_ndarray()
            top, left = (scaled_height - height) // 2, (scaled_width - width) // 2
            frames.append(scaled[top : top + height, left : left + width])
    source_times = np.asarray(times) - times[0]
    duration = source_times[-1] + (source_times[1] - source_times[0] if len(times) > 1 else 0.0)
    step_times = np.arange(int(round(duration * step_rate))) / step_rate
    nearest = np.abs(source_times[None, :] - step_times[:, None]).argmin(axis=1)
    return np.stack([frames[index] for index in nearest])
```

- [ ] **Step 10: Run the episode tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes -n 0`
Expected: PASS.

- [ ] **Step 11: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D110 --step "Add the episode schema and field encodings"
git add lib/poorin/src/poorin/episodes tests/episodes docs/poorin/prune-metrics.csv
git commit -m "[poorin] Add the episode schema and field encodings" \
  -m "An episode dataset declares its fields, one step rate, and how each field is stored. raw stores array bytes; video stores H.264 with a fixed keyframe interval and decodes a clip from the nearest keyframe. load_video turns any video file into frames at a dataset's size and rate for source readers."
```

---

### Task 3: Build an episode dataset as an artifact

**Files:**
- Modify: `lib/poorin/pyproject.toml`, `uv.lock` (add `array-record`; its own commit; declare
  `msgpack` directly in the same commit if `lib/poorin/pyproject.toml` does not already)
- Create: `lib/poorin/src/poorin/episodes/build.py`
- Modify: `lib/poorin/src/poorin/experiment/data.py` (add `episodes(...)`)
- Test: `tests/episodes/test_build.py`

**Interfaces:**
- Consumes: Task 2's `DatasetSchema`, `Episode`, `validate_episode`, `encode_field`.
- Produces:
  - Constants `SCHEMA_FILENAME = "dataset.json"`, `INDEX_FILENAME = "episodes.parquet"`,
    `SHARD_PATTERN = "shard-{:05d}.array_record"`.
  - `Reader = Callable[[str, Mapping[str, Any]], Iterator[Episode]]`; `load_reader(path: str) -> Reader`.
  - `EpisodeBuildConfig(output_path: str, sources: tuple[str, ...], reader: str,
    reader_options: Mapping[str, Any], schema: DatasetSchema, num_shards: int)`.
  - `build_episodes(config: EpisodeBuildConfig) -> None`.
  - `EpisodeDataset(Artifact)` with `.schema -> DatasetSchema`, `.index -> pandas.DataFrame`
    (columns `episode`, `shard`, `record`, `length`, `metadata`), and `.shard_paths -> list[str]`.
  - `poorin.experiment.data.episodes(name, *, sources, reader, schema, num_shards, reader_options=None,
    version=None, pin=None, resources=None) -> ArtifactStep[EpisodeDataset]`.

- [ ] **Step 1: Add ArrayRecord in its own commit**

Add `"array-record>=0.8.4"` (and `"msgpack>=1.0"` if `grep -n msgpack lib/poorin/pyproject.toml`
finds nothing) to `lib/poorin/pyproject.toml`'s `dependencies`. Then:

```bash
uv lock
uv sync --locked --all-packages --extra=gpu --extra=lm_eval
uv run --no-sync python -c "from array_record.python.array_record_module import ArrayRecordWriter; print('ok')"
uv run --no-sync python -m tools.lint --fix
git add lib/poorin/pyproject.toml uv.lock
git commit -m "[poorin] Add ArrayRecord for episode shards" \
  -m "ArrayRecord stores one msgpack record per episode with random access by index, which the clip reader needs."
```

- [ ] **Step 2: Write the failing build tests**

`tests/episodes/test_build.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

import msgpack
import numpy as np
import pytest
from array_record.python.array_record_module import ArrayRecordReader
from poorin.episodes.build import EpisodeDataset, EpisodeBuildConfig, build_episodes
from poorin.episodes.encodings import decode_field
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding
from poorin.execution.lazy import resolve
from poorin.experiment.data import episodes

STEP_RATE = 10.0
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={
        "video": FieldSpec(FieldKind.VIDEO, "uint8", (16, 16, 3), VideoEncoding(keyframe_interval=4)),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (2,), RawEncoding()),
    },
)
READER = "tests.episodes.test_build:synthetic_reader"


def synthetic_episode(episode_id: str, length: int) -> Episode:
    seed = int.from_bytes(episode_id.encode(), "little") % 2**32
    rng = np.random.default_rng(seed)
    video = np.broadcast_to(np.arange(length, dtype=np.uint8)[:, None, None, None] * 8, (length, 16, 16, 3)).copy()
    action = rng.normal(size=(length, 2)).astype(np.float32)
    return Episode(id=episode_id, fields={"video": video, "action": action}, metadata={"task": f"task {episode_id}"})


def synthetic_reader(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """Each source names its episodes as ``id:length,id:length``; ``options['broken']`` breaks one."""
    for item in source.split(","):
        episode_id, length = item.split(":")
        episode = synthetic_episode(episode_id, int(length))
        if options.get("broken") == episode_id:
            wrong_action = np.zeros((int(length), 3), np.float32)
            episode = Episode(id=episode_id, fields={**episode.fields, "action": wrong_action}, metadata={})
        yield episode


def error_messages(error: BaseException | None) -> list[str]:
    """The messages along an exception's cause chain: Zephyr may wrap a worker's error."""
    messages = []
    while error is not None:
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    return messages


def build(tmp_path, sources, **options) -> EpisodeDataset:
    config = EpisodeBuildConfig(
        output_path=str(tmp_path / "ds"), sources=tuple(sources), reader=READER,
        reader_options=options, schema=SCHEMA, num_shards=2,
    )
    build_episodes(config)
    return EpisodeDataset.raw_load(config.output_path)


def test_build_writes_schema_index_and_records(tmp_path):
    dataset = build(tmp_path, ["a:12,b:7", "c:20"])

    assert dataset.schema == SCHEMA
    assert json.loads((tmp_path / "ds" / "dataset.json").read_text()) == SCHEMA.to_json()
    index = dataset.index
    assert sorted(index["episode"]) == ["a", "b", "c"]
    assert dict(zip(index["episode"], index["length"], strict=True)) == {"a": 12, "b": 7, "c": 20}
    assert json.loads(index.set_index("episode").loc["c", "metadata"]) == {"task": "task c"}
    assert len(dataset.shard_paths) == 2

    row = index.set_index("episode").loc["c"]
    reader = ArrayRecordReader(dataset.shard_paths[int(row["shard"])])
    record = msgpack.unpackb(reader.read([int(row["record"])])[0])
    reader.close()
    expected = synthetic_episode("c", 20)
    np.testing.assert_array_equal(
        decode_field(SCHEMA.fields["action"], STEP_RATE, record["action"], 20, range(20)), expected.fields["action"]
    )


def test_an_episode_that_breaks_the_schema_fails_the_build(tmp_path):
    with pytest.raises(Exception) as raised:
        build(tmp_path, ["a:12,b:7"], broken="b")
    assert any(re.search(r"episode 'b'.*field 'action'", message) for message in error_messages(raised.value))


def test_a_duplicate_episode_id_fails_the_build(tmp_path):
    with pytest.raises(ValueError, match=r"episode 'a'.*more than once"):
        build(tmp_path, ["a:12", "a:5"])


def test_the_handle_builds_and_its_fingerprint_follows_the_schema(tmp_path):
    handle = episodes("synthetic", sources=["a:12", "b:9"], reader=READER, schema=SCHEMA, num_shards=1,
                      version="2026.10.11")
    smaller = DatasetSchema(
        step_rate=STEP_RATE,
        fields={**SCHEMA.fields, "video": FieldSpec(FieldKind.VIDEO, "uint8", (8, 8, 3), VideoEncoding(keyframe_interval=4))},
    )
    resized = episodes("synthetic", sources=["a:12", "b:9"], reader=READER, schema=smaller, num_shards=1,
                       version="2026.10.11")

    assert handle.fingerprint() != resized.fingerprint()
    dataset = resolve(handle)
    assert sorted(dataset.index["episode"]) == ["a", "b"]
```

`ArtifactStep.fingerprint()` is a method (`lib/poorin/src/poorin/execution/lazy.py:229`).

- [ ] **Step 3: Run them to see them fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes/test_build.py -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'poorin.episodes.build'`.

- [ ] **Step 4: Write `build.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an episode dataset: read source units into episodes, encode their fields, and write
ArrayRecord shards, ``episodes.parquet``, and ``dataset.json``.

The build is one Zephyr stage. Each shard reads its group of source units, so the large decoded
arrays never pass between stages.
"""

import importlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import msgpack
import numpy as np
import pandas as pd
from array_record.python.array_record_module import ArrayRecordWriter
from fray.types import ResourceConfig
from rigging.filesystem.factory import open_url
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from poorin.episodes.encodings import encode_field
from poorin.episodes.schema import DatasetSchema, Episode, validate_episode
from poorin.execution.artifact import Artifact

SCHEMA_FILENAME = "dataset.json"
INDEX_FILENAME = "episodes.parquet"
SHARD_PATTERN = "shard-{:05d}.array_record"
# One record per group: every record is read on its own, by index.
ARRAY_RECORD_OPTIONS = "group_size:1"
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="16g")

Reader = Callable[[str, Mapping[str, Any]], Iterator[Episode]]


@dataclass(frozen=True)
class EpisodeBuildConfig:
    output_path: str
    sources: tuple[str, ...]
    """Source units, handed to the reader one at a time, in this order."""
    reader: str
    """``module.path:function``, a :data:`Reader`. A string, so the fingerprint can hash it."""
    reader_options: Mapping[str, Any]
    schema: DatasetSchema
    num_shards: int


class EpisodeDataset(Artifact):
    """A built episode dataset. Reads ``dataset.json`` and ``episodes.parquet`` from its path, so a
    pinned dataset with no record reads the same way."""

    @cached_property
    def schema(self) -> DatasetSchema:
        with open_url(os.path.join(self.path, SCHEMA_FILENAME)) as f:
            return DatasetSchema.from_json(json.load(f))

    @cached_property
    def index(self) -> pd.DataFrame:
        with open_url(os.path.join(self.path, INDEX_FILENAME), "rb") as f:
            return pd.read_parquet(f)

    @property
    def shard_paths(self) -> list[str]:
        count = int(self.index["shard"].max()) + 1 if len(self.index) else 0
        return [os.path.join(self.path, SHARD_PATTERN.format(shard)) for shard in range(count)]


def load_reader(path: str) -> Reader:
    module, _, name = path.partition(":")
    if not name:
        raise ValueError(f"a reader is 'module.path:function', not {path!r}")
    return getattr(importlib.import_module(module), name)


def build_episodes(config: EpisodeBuildConfig) -> None:
    groups = [list(group) for group in np.array_split(np.asarray(config.sources, dtype=object), config.num_shards)]
    groups = [group for group in groups if group]
    os.makedirs(config.output_path, exist_ok=True)

    def write(units: Iterator[list[str]], shard: ShardInfo) -> Iterator[list[dict]]:
        yield _write_shard(config, shard.shard_idx, [unit for group in units for unit in group])

    ctx = ZephyrContext(name="episodes", max_workers=min(len(groups), os.cpu_count() or 1), resources=WORKER_RESOURCES)
    rows = [row for shard_rows in ctx.execute(Dataset.from_list(groups).map_shard(write)).results for row in shard_rows]
    _write_index(config, rows)


def _write_shard(config: EpisodeBuildConfig, shard: int, units: list[str]) -> list[dict]:
    reader = load_reader(config.reader)
    schema = config.schema
    rows: list[dict] = []
    writer = ArrayRecordWriter(os.path.join(config.output_path, SHARD_PATTERN.format(shard)), ARRAY_RECORD_OPTIONS)
    try:
        for unit in units:
            for episode in reader(unit, config.reader_options):
                length = validate_episode(schema, episode)
                record = {
                    name: encode_field(spec, schema.step_rate, episode.fields[name])
                    for name, spec in schema.fields.items()
                }
                writer.write(msgpack.packb(record))
                rows.append({
                    "episode": episode.id, "shard": shard, "record": len(rows), "length": length,
                    "metadata": json.dumps(dict(episode.metadata), sort_keys=True),
                })
    finally:
        writer.close()
    return rows


def _write_index(config: EpisodeBuildConfig, rows: list[dict]) -> None:
    index = pd.DataFrame(rows, columns=["episode", "shard", "record", "length", "metadata"])
    duplicated = index["episode"][index["episode"].duplicated()]
    if len(duplicated):
        raise ValueError(f"episode {duplicated.iloc[0]!r} was read more than once; episode ids must be unique")
    index = index.sort_values(["shard", "record"], ignore_index=True)
    with open_url(os.path.join(config.output_path, INDEX_FILENAME), "wb") as f:
        index.to_parquet(f, index=False)
    with open_url(os.path.join(config.output_path, SCHEMA_FILENAME), "w") as f:
        json.dump(config.schema.to_json(), f, indent=2, sort_keys=True)
```

Notes for the implementer:
- `ZephyrContext` and `Dataset.from_list(...)` follow `poorin/processing/tokenize/tokenize.py`'s
  `_run_split`: each list element becomes one shard. Confirm `ShardInfo`'s import path with
  `grep -n "class ShardInfo" lib/zephyr/src/zephyr/dataset.py`.
- `open_url` and its modes follow existing uses: `grep -rn "open_url(" lib/poorin/src | head`.
- ArrayRecord's writer takes a local path; the prefix is local (`POORIN_PREFIX` on the data drive).
  Object storage is out of scope (spec).

- [ ] **Step 5: Add the `episodes(...)` handle to `poorin/experiment/data.py`**

Add these imports at the top: `from typing import Any`,
`from poorin.episodes.build import EpisodeBuildConfig, EpisodeDataset, build_episodes`, and
`from poorin.episodes.schema import DatasetSchema`. Add a bullet for `episodes` to the module
docstring. Add after `pretokenized`:

```python
def episodes(
    name: str,
    *,
    sources: Sequence[str],
    reader: str,
    schema: DatasetSchema,
    num_shards: int,
    reader_options: Mapping[str, Any] | None = None,
    version: str | None = None,
    pin: str | None = None,
    resources: ResourceConfig | None = None,
) -> ArtifactStep[EpisodeDataset]:
    """An episode-dataset handle (:mod:`poorin.episodes`).

    ``reader`` is ``module.path:function``: it turns one entry of ``sources`` into episodes, and
    ``reader_options`` is passed to it. The sources, the reader's import path, its options, the
    schema, and ``num_shards`` are the recipe. ``pin`` references a dataset that already exists.
    ``version`` defers to the ambient build context when omitted.
    """
    version = resolve_version(name, version)
    options = dict(reader_options or {})

    def build_config(ctx: StepContext) -> EpisodeBuildConfig:
        return EpisodeBuildConfig(
            output_path=ctx.output_path,
            # A source unit is opaque to the framework (a path, a Hub file name, a clip id); the
            # reader resolves it.
            sources=tuple(sources),
            reader=reader,
            reader_options=options,
            schema=schema,
            num_shards=num_shards,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=EpisodeDataset,
        run=_on(build_episodes, resources),
        build_config=build_config,
        override_path=pin,
    )
```

- [ ] **Step 6: Run the build tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes -n 0`
Expected: PASS.

- [ ] **Step 7: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D110 --step "Build episode datasets as artifacts"
git add lib/poorin/src/poorin/episodes/build.py lib/poorin/src/poorin/experiment/data.py tests/episodes \
  docs/poorin/prune-metrics.csv
git commit -m "[poorin] Build episode datasets as artifacts" \
  -m "episodes(...) is a data handle like tokenized(...): a Zephyr stage per shard reads its source units through a reader named by import path, checks each episode against the schema, encodes its fields, and writes ArrayRecord shards. The build writes episodes.parquet and dataset.json and refuses duplicate episode ids."
```

---

### Task 4: Read clips through Grain

**Files:**
- Modify: `lib/poorin/pyproject.toml`, `uv.lock` (add `grain`; its own commit)
- Create: `lib/poorin/src/poorin/episodes/read.py`
- Modify: `tests/episodes/test_build.py` (use `read_record`, per Task 3 Step 2)
- Test: `tests/episodes/test_read.py`

**Interfaces:**
- Consumes: `EpisodeDataset` (`.schema`, `.index`, `.shard_paths`), `decode_field`.
- Produces:
  - `ClipSpec(length: int, stride: int, skip: int)`; `ClipSpec.span -> int` (`(length - 1) * skip + 1`).
  - `clip_starts(lengths: np.ndarray, clip: ClipSpec) -> tuple[np.ndarray, np.ndarray]`
    (episode rows, start steps).
  - `read_record(dataset: EpisodeDataset, shard: int, record: int) -> dict[str, bytes]`.
  - `clip_dataset(dataset: EpisodeDataset, *, clip: ClipSpec, fields: Sequence[str],
    metadata: Sequence[str] = ()) -> grain.MapDataset` (finite, unshuffled; items are
    `dict[str, np.ndarray]`).
  - `batch_iterator(clips: grain.MapDataset, *, batch_size: int, start_step: int, num_workers: int)
    -> grain.IterDataset`.
  - `to_mesh(batch: Mapping[str, np.ndarray], mesh: jax.sharding.Mesh, axes: tuple[str, ...])
    -> dict[str, jax.Array | np.ndarray]`.

- [ ] **Step 1: Add Grain in its own commit**

Add `"grain>=0.2.18"` to `lib/poorin/pyproject.toml`'s `dependencies`, then lock, sync, check, and
commit as in Task 3 Step 1 with title `[poorin] Add Grain for reading episode clips` and body
"Grain reads clips by index with a deterministic shuffle and decodes them in worker processes."

- [ ] **Step 2: Write the failing read tests**

`tests/episodes/test_read.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import grain
import jax
import numpy as np
from jax.sharding import Mesh
from poorin.episodes.build import EpisodeBuildConfig, EpisodeDataset, build_episodes
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset, clip_starts, to_mesh
from poorin.episodes.schema import DatasetSchema, Episode

from tests.episodes.test_build import READER, SCHEMA, synthetic_episode

VIDEO_ONLY = DatasetSchema(step_rate=10.0, fields={"video": SCHEMA.fields["video"]})
VIDEO_ONLY_READER = "tests.episodes.test_read:video_only_reader"


def video_only_reader(source, options):
    for item in source.split(","):
        episode_id, length = item.split(":")
        video = synthetic_episode(episode_id, int(length)).fields["video"]
        yield Episode(id=episode_id, fields={"video": video}, metadata={})


def build(tmp_path, name, sources, schema=SCHEMA, reader=READER) -> EpisodeDataset:
    config = EpisodeBuildConfig(output_path=str(tmp_path / name), sources=tuple(sources), reader=reader,
                                reader_options={}, schema=schema, num_shards=2)
    build_episodes(config)
    return EpisodeDataset.raw_load(config.output_path)


def test_clip_starts_stay_inside_episodes():
    clip = ClipSpec(length=4, stride=2, skip=2)  # spans 7 steps
    rows, starts = clip_starts(np.array([10, 3, 7]), clip)
    # Episode 0 (10 steps): starts 0 and 2 fit (2 + 7 = 9 <= 10). Episode 1 is shorter than a clip.
    # Episode 2 (7 steps): start 0 only.
    assert rows.tolist() == [0, 0, 2]
    assert starts.tolist() == [0, 2, 0]


def test_a_clip_holds_the_requested_steps_of_every_selected_field(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12,b:7", "c:20"])
    clips = clip_dataset(dataset, clip=ClipSpec(length=5, stride=1, skip=3), fields=["video", "action"],
                         metadata=["task"])
    index = dataset.index
    expected_clips = sum(max(0, (n - 13) // 1 + 1) for n in index["length"])  # span is 13
    assert len(clips) == expected_clips

    last = clips[len(clips) - 1]  # the last valid start of the last episode, "c" (20 steps): start 7
    episode = synthetic_episode("c", 20)
    np.testing.assert_array_equal(last["action"], episode.fields["action"][7:20:3])
    np.testing.assert_array_equal(last["video"][:, 0, 0, 0], episode.fields["video"][7:20:3, 0, 0, 0])
    assert str(last["task"]) == "task c"


def test_unselected_fields_are_not_returned(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12"])
    clip = clip_dataset(dataset, clip=ClipSpec(length=4, stride=4, skip=1), fields=["action"])[0]
    assert set(clip) == {"action"}


def test_resuming_at_a_batch_reads_what_an_uninterrupted_run_reads(tmp_path):
    dataset = build(tmp_path, "ds", ["a:30,b:25", "c:40"])
    clips = clip_dataset(dataset, clip=ClipSpec(length=4, stride=1, skip=1), fields=["action"])
    shuffled = clips.shuffle(seed=7).repeat(None)

    uninterrupted = shuffled.batch(8)
    resumed = batch_iterator(shuffled, batch_size=8, start_step=5, num_workers=2)
    iterator = iter(resumed)
    try:
        for step in range(5, 9):
            np.testing.assert_array_equal(next(iterator)["action"], uninterrupted[step]["action"])
    finally:
        iterator.close()

    other_seed = clips.shuffle(seed=8).repeat(None).batch(8)
    assert not np.array_equal(other_seed[0]["action"], uninterrupted[0]["action"])


def test_each_epoch_has_its_own_order(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12"])
    clips = clip_dataset(dataset, clip=ClipSpec(length=2, stride=1, skip=1), fields=["action"])
    shuffled = clips.shuffle(seed=0).repeat(2)
    first = [shuffled[i]["action"].tobytes() for i in range(len(clips))]
    second = [shuffled[len(clips) + i]["action"].tobytes() for i in range(len(clips))]
    assert sorted(first) == sorted(second)
    assert first != second


def test_a_video_only_dataset_mixes_with_one_that_has_actions(tmp_path):
    robot = build(tmp_path, "robot", ["a:12"])
    video = build(tmp_path, "video", ["v:12"], schema=VIDEO_ONLY, reader=VIDEO_ONLY_READER)
    clip = ClipSpec(length=4, stride=4, skip=1)
    video_clips = clip_dataset(video, clip=clip, fields=["video"])
    assert set(video_clips[0]) == {"video"}

    robot_frames = clip_dataset(robot, clip=clip, fields=["video"])
    mixed = grain.MapDataset.mix([robot_frames, video_clips], weights=[1, 1])
    assert {mixed[i]["video"].shape for i in range(4)} == {(4, 16, 16, 3)}


def test_to_mesh_shards_numeric_fields_and_passes_text_through():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    batch = {"action": np.zeros((4, 3, 2), np.float32), "task": np.array(["a", "b", "c", "d"])}
    placed = to_mesh(batch, mesh, ("data",))
    assert isinstance(placed["action"], jax.Array)
    assert placed["action"].sharding.spec == jax.sharding.PartitionSpec(("data",))
    np.testing.assert_array_equal(placed["task"], batch["task"])
```

- [ ] **Step 3: Run them to see them fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes/test_read.py -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'poorin.episodes.read'`.

- [ ] **Step 4: Write `read.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read an episode dataset as clips through Grain.

:func:`clip_dataset` is a finite, unshuffled ``grain.MapDataset`` of clips; a training script
shuffles, repeats, and batches it. Everything up to batching has random access, so batch N is a
pure function of (seed, N): :func:`batch_iterator` resumes at a step by slicing, and decodes in
worker processes. :func:`to_mesh` places a batch on the mesh.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import grain
import jax
import msgpack
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.encodings import decode_field
from poorin.episodes.schema import DatasetSchema

# Numeric dtype kinds placed on the mesh: bool, signed, unsigned, float. Text stays on the host.
DEVICE_DTYPE_KINDS = "biuf"


@dataclass(frozen=True)
class ClipSpec:
    length: int
    """Steps in a clip."""
    stride: int
    """Steps between the starts of consecutive clips in an episode."""
    skip: int
    """Steps between consecutive steps of a clip; 1 takes every step."""

    @property
    def span(self) -> int:
        return (self.length - 1) * self.skip + 1


def clip_starts(lengths: np.ndarray, clip: ClipSpec) -> tuple[np.ndarray, np.ndarray]:
    """Every clip as (episode row, start step). A clip never crosses its episode's end, and an
    episode shorter than one clip has none."""
    counts = np.maximum(0, (lengths - clip.span) // clip.stride + 1)
    rows = np.repeat(np.arange(len(lengths)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return rows, offsets * clip.stride


def read_record(dataset: EpisodeDataset, shard: int, record: int) -> dict[str, bytes]:
    reader = ArrayRecordReader(dataset.shard_paths[shard])
    try:
        return msgpack.unpackb(reader.read([record])[0])
    finally:
        reader.close()


class _ClipSource:
    """A Grain source of clip locations: (shard, record, episode length, start, episode row)."""

    def __init__(self, dataset: EpisodeDataset, clip: ClipSpec):
        index = dataset.index
        self._rows, self._starts = clip_starts(index["length"].to_numpy(), clip)
        self._shards = index["shard"].to_numpy()
        self._records = index["record"].to_numpy()
        self._lengths = index["length"].to_numpy()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> tuple[int, int, int, int, int]:
        row = int(self._rows[i])
        return int(self._shards[row]), int(self._records[row]), int(self._lengths[row]), int(self._starts[i]), row


class _ClipReader:
    """Decodes one clip. Picklable for Grain's worker processes: readers open lazily in each worker."""

    def __init__(self, dataset: EpisodeDataset, clip: ClipSpec, fields: Sequence[str], metadata: Sequence[str]):
        self._shard_paths = dataset.shard_paths
        self._schema: DatasetSchema = dataset.schema
        self._clip = clip
        self._fields = tuple(fields)
        self._metadata_keys = tuple(metadata)
        self._metadata = [json.loads(value) for value in dataset.index["metadata"]] if metadata else []
        self._readers: dict[int, ArrayRecordReader] = {}
        unknown = set(self._fields) - set(self._schema.fields)
        if unknown:
            raise ValueError(f"fields {sorted(unknown)} are not in the dataset; it has {sorted(self._schema.fields)}")

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_readers": {}}

    def __call__(self, location: tuple[int, int, int, int, int]) -> dict[str, np.ndarray]:
        shard, record, length, start, row = location
        if shard not in self._readers:
            self._readers[shard] = ArrayRecordReader(self._shard_paths[shard])
        data = msgpack.unpackb(self._readers[shard].read([record])[0])
        steps = range(start, start + self._clip.span, self._clip.skip)
        clip = {
            name: decode_field(self._schema.fields[name], self._schema.step_rate, data[name], length, steps)
            for name in self._fields
        }
        for key in self._metadata_keys:
            clip[key] = np.asarray(self._metadata[row][key])
        return clip


def clip_dataset(
    dataset: EpisodeDataset, *, clip: ClipSpec, fields: Sequence[str], metadata: Sequence[str] = ()
) -> grain.MapDataset:
    """Every clip of ``dataset``, in index order, as ``{field: [clip.length, *shape]}`` plus the
    requested per-episode ``metadata`` values. Only ``fields`` are decoded."""
    return grain.MapDataset.source(_ClipSource(dataset, clip)).map(_ClipReader(dataset, clip, fields, metadata))


def batch_iterator(clips: grain.MapDataset, *, batch_size: int, start_step: int, num_workers: int) -> grain.IterDataset:
    """Batches from ``start_step`` on, decoded in ``num_workers`` processes. ``clips`` is shuffled
    and repeated by the caller; resuming at a step reads the batches an uninterrupted run would."""
    batched = clips.batch(batch_size, drop_remainder=True)
    options = grain.MultiprocessingOptions(num_workers=num_workers)
    return batched[start_step:].to_iter_dataset().mp_prefetch(options)


def to_mesh(batch: Mapping[str, np.ndarray], mesh: Mesh, axes: tuple[str, ...]) -> dict[str, jax.Array | np.ndarray]:
    """Numeric fields as ``jax.Array``s sharded on their leading (batch) dimension over ``axes``;
    text fields stay numpy arrays on the host."""
    sharding = NamedSharding(mesh, PartitionSpec(axes))
    return {
        name: jax.make_array_from_process_local_data(sharding, values) if values.dtype.kind in DEVICE_DTYPE_KINDS else values
        for name, values in batch.items()
    }
```

Then in `tests/episodes/test_build.py`, replace the three-line direct ArrayRecord read with
`record = read_record(dataset, int(row["shard"]), int(row["record"]))`, import `read_record` at the
top, and drop the `array_record` and `msgpack` imports there.

Notes for the implementer:
- Verify `ArrayRecordReader`'s `read([index])` and `close()` against the installed package
  (`uv run --no-sync python -c "from array_record.python.array_record_module import ArrayRecordReader; help(ArrayRecordReader)"`).
  Use the documented method if the names differ; do not wrap it.
- A Grain iterator with worker processes must be closed (`iterator.close()`). An unclosed one logs
  `AttributeError` tracebacks at interpreter exit.

- [ ] **Step 5: Run the episode tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes -n 0`
Expected: PASS. If `test_resuming_at_a_batch_reads_what_an_uninterrupted_run_reads` exceeds the
60-second test timeout because of worker start-up, mark it `@pytest.mark.timeout(120)`.

- [ ] **Step 6: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D110 --step "Read episode clips through Grain"
git add lib/poorin/src/poorin/episodes/read.py tests/episodes docs/poorin/prune-metrics.csv
git commit -m "[poorin] Read episode clips through Grain" \
  -m "clip_dataset plans clips from the index alone and decodes only the selected fields and steps. Every layer before batching has random access, so batch_iterator resumes at a step by slicing and decodes in worker processes; to_mesh places a batch on the mesh."
```

---

### Task 5: Example datasets, throughput, and an end-to-end smoke run

**Files:**
- Modify: `lib/poorin/pyproject.toml`, `uv.lock` (add `h5py`; its own commit)
- Create: `experiments/datasets/episodes/__init__.py`
- Create: `experiments/datasets/episodes/ssv2.py`
- Create: `experiments/datasets/episodes/libero.py`
- Create: `experiments/datasets/episodes/vpt.py`
- Create: `experiments/datasets/episodes/throughput.py`
- Create: `experiments/datasets/episodes/smoke_train.py`
- Test: `tests/episodes/test_examples.py`

**Interfaces:**
- Consumes: `episodes(...)`, `load_video`, `clip_dataset`, `batch_iterator`, `to_mesh`, `ClipSpec`.
- Produces: example handles `ssv2_subset()`, `libero_spatial()`, `vpt_subset()`, each returning
  `ArtifactStep[EpisodeDataset]`; the two scripts.

These are examples. The library does not import them, and they are not supported datasets.

- [ ] **Step 1: Add h5py in its own commit**

Add `"h5py>=3.16"` to `lib/poorin/pyproject.toml`'s `dependencies` (the LIBERO example reads
HDF5), then lock, sync, check, and commit as in Task 3 Step 1 with title
`[poorin] Add h5py for the LIBERO example` and body "LIBERO ships its demonstrations as HDF5."

- [ ] **Step 2: Write `experiments/datasets/episodes/__init__.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Example episode datasets that test the format: a few hundred Something-Something v2 clips (video
without actions), one LIBERO suite (two cameras, proprioception, actions), and a few Minecraft VPT
episodes (long video with keyboard and mouse actions). Not supported datasets."""
```

- [ ] **Step 3: Write `ssv2.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Something-Something v2 clips as episodes without actions: video, and the label as metadata."""

import functools
import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, VideoEncoding
from poorin.episodes.video_files import load_video
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import episodes

SSV2_ROOT = "/media/johnnyas/D/research-storage/ssv2"
VIDEO_DIR = "20bn-something-something-v2"
LABELS = "train.json"
STEP_RATE = 12.0  # SSv2's frame rate
SIZE = (128, 128)
SUBSET_CLIPS = 500
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={"video": FieldSpec(FieldKind.VIDEO, "uint8", (*SIZE, 3), VideoEncoding(keyframe_interval=16))},
)


@functools.cache
def _labels(root: str) -> dict[str, str]:
    with open(os.path.join(root, LABELS)) as f:
        return {item["id"]: item["label"] for item in json.load(f)}


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One SSv2 clip id."""
    root = options["root"]
    video = load_video(os.path.join(root, VIDEO_DIR, f"{source}.webm"), size=SIZE, step_rate=STEP_RATE)
    yield Episode(id=source, fields={"video": video}, metadata={"label": _labels(root)[source]})


def ssv2_subset(root: str = SSV2_ROOT) -> ArtifactStep[EpisodeDataset]:
    ids = sorted(_labels(root))[:SUBSET_CLIPS]
    return episodes(
        "examples/ssv2-500", sources=ids, reader="experiments.datasets.episodes.ssv2:read",
        reader_options={"root": root}, schema=SCHEMA, num_shards=16, version="2026.10.11",
    )
```

- [ ] **Step 4: Write `libero.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-Spatial demonstrations as episodes: two cameras, joint and gripper state, actions, and the
instruction as metadata."""

import os
from collections.abc import Iterator, Mapping
from typing import Any

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import episodes

REPO_ID = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITE = "libero_spatial"
TASKS = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)
STEP_RATE = 20.0  # LIBERO's control rate
CAMERA = FieldSpec(FieldKind.VIDEO, "uint8", (128, 128, 3), VideoEncoding(keyframe_interval=16))
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={
        "agentview": CAMERA,
        "wrist": CAMERA,
        "joint_state": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding()),
        "gripper_state": FieldSpec(FieldKind.ARRAY, "float32", (2,), RawEncoding()),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding()),
    },
)


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One task's HDF5 file: every demonstration in it."""
    path = hf_hub_download(REPO_ID, f"{SUITE}/{source}_demo.hdf5", repo_type="dataset", revision=options["revision"])
    instruction = source.replace("_", " ")
    with h5py.File(path, "r") as f:
        for demo in sorted(f["data"], key=lambda name: int(name.split("_")[1])):
            group = f["data"][demo]
            obs = group["obs"]
            yield Episode(
                id=f"{source}/{demo}",
                fields={
                    # LIBERO renders upside down; rotate 180 degrees.
                    "agentview": np.ascontiguousarray(obs["agentview_rgb"][:, ::-1, ::-1]),
                    "wrist": np.ascontiguousarray(obs["eye_in_hand_rgb"][:, ::-1, ::-1]),
                    "joint_state": obs["joint_states"][:].astype(np.float32),
                    "gripper_state": obs["gripper_states"][:].astype(np.float32),
                    "action": group["actions"][:].astype(np.float32),
                },
                metadata={"instruction": instruction, "suite": SUITE},
            )


def libero_spatial() -> ArtifactStep[EpisodeDataset]:
    return episodes(
        "examples/libero-spatial", sources=list(TASKS), reader="experiments.datasets.episodes.libero:read",
        reader_options={"revision": REVISION}, schema=SCHEMA, num_shards=len(TASKS), version="2026.10.11",
    )
```

Before running the build, open one file and confirm the key names and shapes:

```bash
uv run --no-sync python -c "
from huggingface_hub import hf_hub_download; import h5py
from experiments.datasets.episodes.libero import REPO_ID, REVISION, SUITE, TASKS
p = hf_hub_download(REPO_ID, f'{SUITE}/{TASKS[0]}_demo.hdf5', repo_type='dataset', revision=REVISION)
f = h5py.File(p); g = f['data/demo_0']; print(list(g), {k: g['obs'][k].shape for k in g['obs']}, g['actions'].shape)"
```

If a key differs (for example `ee_states` present, `joint_states` named differently), change the
reader and the schema to match the file, and record the change in D110's acceptance notes.

- [ ] **Step 5: Write `vpt.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A few Minecraft VPT contractor episodes: video with keyboard and mouse actions."""

import json
import os
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding
from poorin.episodes.video_files import load_video
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import episodes

BASE_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
# A fixed snapshot of OpenAI's contractor-data index (71,868 episodes); the first episodes are the subset.
INDEX_URL = BASE_URL + "snapshots/all_6xx_Jun_29.json"
SUBSET_EPISODES = 10
STEP_RATE = 20.0  # Minecraft's tick rate in the recordings
SIZE = (128, 128)
KEYS = (
    "key.keyboard.w", "key.keyboard.a", "key.keyboard.s", "key.keyboard.d", "key.keyboard.space",
    "key.keyboard.left.shift", "key.keyboard.left.control", "key.keyboard.e", "key.keyboard.q",
    *(f"key.keyboard.{n}" for n in range(1, 10)),
)
MOUSE_BUTTONS = 3
# Keys pressed (multi-hot), mouse dx and dy, mouse buttons held (multi-hot).
ACTION_SIZE = len(KEYS) + 2 + MOUSE_BUTTONS
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={
        "video": FieldSpec(FieldKind.VIDEO, "uint8", (*SIZE, 3), VideoEncoding(keyframe_interval=16)),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (ACTION_SIZE,), RawEncoding()),
    },
)


def _action(step: dict) -> np.ndarray:
    action = np.zeros(ACTION_SIZE, np.float32)
    pressed = set(step["keyboard"]["keys"])
    action[: len(KEYS)] = [key in pressed for key in KEYS]
    action[len(KEYS)] = step["mouse"]["dx"]
    action[len(KEYS) + 1] = step["mouse"]["dy"]
    for button in step["mouse"]["buttons"]:
        if button < MOUSE_BUTTONS:
            action[len(KEYS) + 2 + button] = 1.0
    return action


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One episode's relative path without extension, e.g. ``data/6.10/Player129-...-171344``."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "episode.mp4")
        urllib.request.urlretrieve(options["base_url"] + source + ".mp4", video_path)
        video = load_video(video_path, size=SIZE, step_rate=STEP_RATE)
    with urllib.request.urlopen(options["base_url"] + source + ".jsonl") as response:
        steps = [json.loads(line) for line in response.read().decode().splitlines() if line]
    length = min(len(video), len(steps))
    actions = np.stack([_action(step) for step in steps[:length]])
    yield Episode(id=source, fields={"video": video[:length], "action": actions}, metadata={"source": source})


def vpt_subset() -> ArtifactStep[EpisodeDataset]:
    with urllib.request.urlopen(INDEX_URL) as response:
        relpaths = json.load(response)["relpaths"]
    sources = [path.removesuffix(".mp4") for path in relpaths[:SUBSET_EPISODES]]
    return episodes(
        "examples/vpt-10", sources=sources, reader="experiments.datasets.episodes.vpt:read",
        reader_options={"base_url": BASE_URL}, schema=SCHEMA, num_shards=SUBSET_EPISODES, version="2026.10.11",
    )
```

Each VPT episode is about 115 MB of mp4 and 6,000 steps; the subset downloads about 1.2 GB.

- [ ] **Step 6: Write `throughput.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clips per second from an episode dataset against the number of decoding worker processes.

    uv run --no-sync python -m experiments.datasets.episodes.throughput PATH --field video --workers 1 4 8 16 24
"""

import argparse
import time

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset

CLIP_LENGTH = 16
BATCH_SIZE = 32
WARMUP_BATCHES = 5
MEASURED_BATCHES = 50
TARGET_CLIPS_PER_SECOND = 1000


def clips_per_second(dataset: EpisodeDataset, field: str, workers: int) -> float:
    clips = clip_dataset(dataset, clip=ClipSpec(length=CLIP_LENGTH, stride=1, skip=1), fields=[field])
    iterator = iter(batch_iterator(clips.shuffle(seed=0).repeat(None), batch_size=BATCH_SIZE, start_step=0,
                                   num_workers=workers))
    try:
        for _ in range(WARMUP_BATCHES):
            next(iterator)
        start = time.perf_counter()
        for _ in range(MEASURED_BATCHES):
            next(iterator)
        return MEASURED_BATCHES * BATCH_SIZE / (time.perf_counter() - start)
    finally:
        iterator.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--field", required=True)
    parser.add_argument("--workers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    dataset = EpisodeDataset.raw_load(args.path)
    for workers in args.workers:
        rate = clips_per_second(dataset, args.field, workers)
        verdict = "meets" if rate >= TARGET_CLIPS_PER_SECOND else "misses"
        print(f"{workers:3d} workers  {rate:8.0f} clips/s  {verdict} {TARGET_CLIPS_PER_SECOND}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Write `smoke_train.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End to end: LIBERO clips through to_mesh into a toy JAX step for 200 steps, then a resume at
step 100 that must see the same batches.

    uv run --no-sync python -m experiments.datasets.episodes.smoke_train PATH
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset, to_mesh

STEPS = 200
RESUME_STEP = 100
BATCH_SIZE = 16
WORKERS = 8
SEED = 0
LEARNING_RATE = 1e-3


@jax.jit
def train_step(weights: jax.Array, frames: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Predict each step's action from the mean color of its frame: a toy that exercises placement."""

    def loss(w):
        features = frames.astype(jnp.float32).mean(axis=(2, 3)) / 255.0  # [B, T, 3]
        return jnp.mean((features @ w - actions) ** 2)

    value, grad = jax.value_and_grad(loss)(weights)
    return weights - LEARNING_RATE * grad, value


def run(dataset: EpisodeDataset, start_step: int, steps: int) -> list[np.ndarray]:
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    clips = clip_dataset(dataset, clip=ClipSpec(length=8, stride=4, skip=1), fields=["agentview", "action"])
    iterator = iter(batch_iterator(clips.shuffle(seed=SEED).repeat(None), batch_size=BATCH_SIZE,
                                   start_step=start_step, num_workers=WORKERS))
    weights = jnp.zeros((3, 7))
    seen: list[np.ndarray] = []
    try:
        for step in range(start_step, steps):
            batch = to_mesh(next(iterator), mesh, ("data",))
            weights, loss = train_step(weights, batch["agentview"], batch["action"])
            seen.append(np.asarray(batch["action"]))
            if step % 50 == 0:
                print(f"step {step}  loss {float(loss):.4f}")
    finally:
        iterator.close()
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    dataset = EpisodeDataset.raw_load(parser.parse_args().path)
    full = run(dataset, 0, STEPS)
    resumed = run(dataset, RESUME_STEP, STEPS)
    for offset, batch in enumerate(resumed):
        np.testing.assert_array_equal(batch, full[RESUME_STEP + offset])
    print(f"resumed at step {RESUME_STEP}: {len(resumed)} batches match the uninterrupted run")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Write the example test**

The examples need data and network, so the test checks only what runs without them: each module
imports, and its handle's recipe is well formed. `tests/episodes/test_examples.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datasets.episodes import libero, ssv2, vpt
from poorin.episodes.build import load_reader


def test_example_readers_resolve_by_import_path():
    for module in (libero, ssv2, vpt):
        assert load_reader(f"{module.__name__}:read") is module.read


def test_the_libero_handle_plans_without_downloading():
    handle = libero.libero_spatial()
    assert handle.name == "examples/libero-spatial"
    assert handle.path().endswith("examples/libero-spatial/2026.10.11")
```

`ssv2_subset()` reads the labels file and `vpt_subset()` fetches the index, so their handles are not
built in tests.

- [ ] **Step 9: Run the episode tests**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/episodes -n 0`
Expected: PASS.

- [ ] **Step 10: Build the three examples (acceptance 1)**

With `POORIN_PREFIX` on the data drive:

```bash
uv run --no-sync python -c "
from poorin.execution.lazy import run
from experiments.datasets.episodes.ssv2 import ssv2_subset
from experiments.datasets.episodes.libero import libero_spatial
from experiments.datasets.episodes.vpt import vpt_subset
for dataset in run(ssv2_subset(), libero_spatial(), vpt_subset()):
    print(dataset.path, len(dataset.index), int(dataset.index['length'].sum()))"
du -sh "$POORIN_PREFIX"/examples/*/2026.10.11
```

Record each dataset's episodes, steps, size on disk, and build time for D110.

- [ ] **Step 11: Measure throughput (acceptance 2)**

```bash
uv run --no-sync python -m experiments.datasets.episodes.throughput \
  "$POORIN_PREFIX/examples/ssv2-500/2026.10.11" --field video --workers 1 4 8 16 24
```

Record the curve for D110. A miss at 24 workers is recorded with the curve, not hidden; report it to
the owner with the levers from the spec (shorter keyframe interval, JPEG per frame, smaller size).

- [ ] **Step 12: Run the end-to-end smoke (acceptance 3)**

```bash
uv run --no-sync python -m experiments.datasets.episodes.smoke_train \
  "$POORIN_PREFIX/examples/libero-spatial/2026.10.11"
```

Expected: loss lines every 50 steps, then `resumed at step 100: 100 batches match the uninterrupted
run`, and no traceback at exit.

- [ ] **Step 13: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D110 --step "Add example episode datasets and checks"
git add experiments/datasets/episodes tests/episodes docs/poorin/prune-metrics.csv
git commit -m "[poorin] Add example episode datasets and checks" \
  -m "Small readers for Something-Something v2, LIBERO-Spatial, and Minecraft VPT test the format on video without actions, two cameras with proprioception, and long video with keyboard and mouse actions. throughput.py measures clips per second against decoding workers; smoke_train.py trains a toy step on LIBERO clips and checks a resume. They are examples, not supported datasets."
```

---

### Task 6: Documentation and the D110 record

**Files:**
- Create: `docs/explanations/episodes.md`
- Modify: `mkdocs.yml` (add the page next to `lazy-artifacts.md`; find it with `grep -n lazy-artifacts mkdocs.yml`)
- Modify: `docs/poorin/architecture.md` (the `poorin.episodes` modules)
- Modify: `lib/poorin/AGENTS.md` (a line for `poorin.episodes`)
- Modify: `docs/poorin/decisions.md` (D110: status and acceptance results)

- [ ] **Step 1: Write `docs/explanations/episodes.md`**

Sections, in this order, written for a reader who knows `lazy-artifacts.md`:

1. **Episodes.** An episode is named per-step fields on one step clock; a plain video has no action
   field. The three files of a dataset and what each holds (copy the spec's tables).
2. **Field encodings.** `RawEncoding` and `VideoEncoding`, what a keyframe interval trades
   (smaller files against decoding up to `keyframe_interval - 1` extra frames per clip), and how to
   add an encoding (one class in `schema.py`, one branch in each function of `encodings.py`).
3. **Building a dataset.** A reader is `read(source, options) -> Iterator[Episode]`, named by
   import path; `load_video` for video files; the `episodes(...)` handle with a complete example
   (the SSv2 example's `read` and `ssv2_subset`); what is in the fingerprint and why the reader is a
   string.
4. **Reading clips in training.** `ClipSpec`, `clip_dataset`, shuffling and repeating with Grain,
   `batch_iterator` with `start_step` for resuming, `to_mesh`; a complete loop (the smoke script's
   `run`). Close the iterator.
5. **Mixing datasets.** Map each to common fields, then `grain.MapDataset.mix`.
6. **Throughput.** The measured curve from Task 5 and the levers if a model needs more.

- [ ] **Step 2: Update `architecture.md` and `lib/poorin/AGENTS.md`**

Add the five `poorin.episodes` modules to `architecture.md`'s module table with their line counts
(`wc -l lib/poorin/src/poorin/episodes/*.py`) and verdict "Keep". In `lib/poorin/AGENTS.md`, add one
line: `poorin.episodes` — episode datasets for video, trajectories, and latents, read as clips
through Grain; see `docs/explanations/episodes.md`.

- [ ] **Step 3: Finish D110**

Change D110's status to `Adopted.` Add:

```markdown
- **Rulings while planning:** A field's encoding is a class (`RawEncoding`, `VideoEncoding`), not an
  enum member, following AGENTS.md's "separate classes over flags"; the spec's "enum plus a pair"
  becomes "a class plus a branch". A reader delivers frames at the schema's size and step rate
  (`load_video` does this for files), and the build checks and encodes them. `episodes(...)` takes
  `num_shards`. Source units are opaque strings the reader resolves. The example readers download
  their data; VPT's subset is the first 10 episodes of the `all_6xx_Jun_29` index.
- **Acceptance (2026-10-XX):** <each example's episodes, steps, size, build time>; throughput
  <clips/s at 1, 4, 8, 16, 24 workers> against 1,000; the smoke run resumed at step 100 with
  identical batches.
```

Fill the acceptance line from Task 5's recorded numbers and today's date.

- [ ] **Step 4: Build the docs, full check, commit**

```bash
uv run --no-sync mkdocs build --strict 2>&1 | tail -5
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D110 --step "Document episode datasets"
git add docs mkdocs.yml lib/poorin/AGENTS.md
git commit -m "[poorin] Document episode datasets (D110)" \
  -m "A new explanation covers the format, field encodings, building with a reader, and reading clips in training. The architecture table and lib/poorin/AGENTS.md list poorin.episodes; D110 records the rulings made while planning and the acceptance results."
```

If `mkdocs build --strict` fails on something this task did not touch, note it in the commit body
and continue; fix only what this task broke.

---

## After the plan

- Merge `feature/episode-data-layer` into `main` only when the owner asks.
- Delete `/tmp/venv-py312-rollback` only when the owner says so.
- Sub-project 4 (model templates) starts from `clip_dataset` and `to_mesh`.
