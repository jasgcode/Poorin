# Episode Model Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three flow-matching templates (WAM, world model, VLA) that train on episode datasets through one shared loop, launch through the experiment CLI, and log a held-out loss, parameter count, and FLOPs that sub-project 5 can sweep.

**Architecture:** `experiments/episode_models/` holds shared modules (data, flow matching, layers, FLOPs, training loop, launch) and one directory per family with `model.py` and `launch.py`. A family's config builds its model from data-derived sizes; its model returns per-stream losses at given flow times; the shared loop samples times, trains, evaluates on a fixed time grid, checkpoints, and resumes.

**Tech Stack:** JAX 0.11 (explicit mesh), Equinox, Optax via Levanter's `OptimizerConfig`, Grain clips from `poorin.episodes`, Levanter tracker and checkpointer, the Poorin experiment CLI.

**Spec:** `docs/poorin/specs/2026-10-11-episode-model-templates-design.md`, decision D113.

## Global Constraints

- Run every command with `uv run --no-sync`. Unit tests run on CPU (`JAX_PLATFORMS=cpu`), one file with `-n 0`.
- Python >= 3.13; all imports at the top of the file; no `*_utils.py`; no `__all__`.
- Grug style: `equinox.Module` with explicit `init`, raw `jax.Array` parameters, `jnp.einsum`; no `eqx.nn`.
- Flow path `x_t = t * x + (1 - t) * noise`, `t = 1` clean; target velocity `x - noise`; logit-normal times (mean 0, scale 1); eval grid `(i + 0.5) / 8`, `i < 8`.
- Trunk: `head_dim = 64`, MLP width `4 * hidden_dim`, ReLU, RMSNorm eps `1e-5`, init std `0.02`, rotary theta `10_000`.
- Frames cross to the device as `uint8`; models map them to `[-1, 1]` in their parameter dtype.
- Noise and times come from `jax.random.fold_in(noise_key, step)`; nothing random runs in Grain workers.
- Held-out split: BLAKE2b (8-byte digest, little-endian) of the episode id, modulo 100, below `eval_percent`.
- Action normalization: per-dimension 1st and 99th percentiles to `[-1, 1]`, then clip.
- Experiment tests live beside the experiment (`experiments/episode_models/tests/`); the library change's test lives in `tests/episodes/`.
- Every new module gets a row in `docs/poorin/architecture.md`; the package is a kept root in `tools/roots.toml`.
- Commits: `[poorin]` prefix, imperative, at most 72 characters, no attribution trailer.

## Review Focus

- A source without a field the family needs (`--dataset ssv2-500` for the VLA or WAM): the launcher refuses before building anything, naming the missing field.
- An episode dataset whose split leaves no held-out or no training episodes (tiny datasets, `eval_percent` 0): training refuses at start with the counts, not a `StopIteration` mid-run.
- Frames whose size is not a multiple of the patch size: `init` refuses with both sizes.
- An action dimension that never varies (LIBERO's gripper is near-binary; a constant dimension has equal percentiles): normalization yields finite values, never NaN.
- Resuming a run: the restored step continues the same batches and the same noise, so losses after the resume match an uninterrupted run.

Tests pinning each line: Task 9 (`test_a_family_refuses_a_source_without_its_fields`), Task 4 (`test_split_refuses_an_empty_side`), Task 5 (`test_init_refuses_frames_the_patch_does_not_divide`), Task 4 (`test_a_constant_action_dimension_normalizes_to_finite_values`), Task 8 (`test_a_resumed_run_reports_the_losses_of_an_uninterrupted_one`).

---

### Task 1: Spec, plan, and decision

**Files:**
- Create: `docs/poorin/specs/2026-10-11-episode-model-templates-design.md` (written)
- Create: `docs/poorin/plans/2026-10-11-episode-model-templates.md` (this file)
- Modify: `docs/poorin/decisions.md` (D113, written)

- [ ] **Step 1: Commit**

```bash
git add docs/poorin/specs/2026-10-11-episode-model-templates-design.md docs/poorin/plans/2026-10-11-episode-model-templates.md docs/poorin/decisions.md
git commit -m "[poorin] Add the episode model templates spec and plan (D113)"
```

### Task 2: Restrict clips to chosen episodes

**Files:**
- Modify: `lib/poorin/src/poorin/episodes/read.py` (`_ClipSource.__init__`, `clip_dataset`)
- Test: `tests/episodes/test_read.py`

**Interfaces:**
- Produces: `clip_dataset(dataset, *, clip, fields, metadata=(), rows: Sequence[int] | None = None) -> grain.MapDataset`. With `rows`, only clips of those episode rows (positions in `dataset.index`) are planned, in index order.

- [ ] **Step 1: Write the failing test**

```python
def test_rows_restrict_clips_to_those_episodes(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12,b:7", "c:20"])
    clip = ClipSpec(length=4, stride=4, skip=1)
    every = clip_dataset(dataset, clip=clip, fields=["action"], metadata=["task"])
    only_c = clip_dataset(dataset, clip=clip, fields=["action"], metadata=["task"], rows=[2])

    assert {str(every[i]["task"]) for i in range(len(every))} == {"task a", "task b", "task c"}
    assert len(only_c) == 5  # 20 steps, clips of 4 every 4 steps
    assert {str(only_c[i]["task"]) for i in range(len(only_c))} == {"task c"}
```

- [ ] **Step 2: Run it and see it fail** — `uv run --no-sync pytest tests/episodes/test_read.py -n 0 -k rows` → `TypeError: unexpected keyword argument 'rows'`.

- [ ] **Step 3: Implement**

```python
class _ClipSource:
    """A Grain source of clip locations: (shard, record, episode length, start, episode row)."""

    def __init__(self, dataset: EpisodeDataset, clip: ClipSpec, rows: Sequence[int] | None):
        index = dataset.index
        selected = np.arange(len(index)) if rows is None else np.asarray(rows, dtype=np.int64)
        lengths = index["length"].to_numpy()
        positions, self._starts = clip_starts(lengths[selected], clip)
        self._rows = selected[positions]
        self._shards = index["shard"].to_numpy()
        self._records = index["record"].to_numpy()
        self._lengths = lengths
```

`clip_dataset` gains `rows: Sequence[int] | None = None`, documented as "only the clips of these episode rows", and passes it to `_ClipSource`.

- [ ] **Step 4: Run** `uv run --no-sync pytest tests/episodes/test_read.py -n 0` → all pass.

- [ ] **Step 5: Commit** `[poorin] Let clip_dataset read a subset of episodes`

### Task 3: Flow matching and trunk layers

**Files:**
- Create: `experiments/episode_models/__init__.py` (docstring only), `experiments/episode_models/flow.py`, `experiments/episode_models/layers.py`, `experiments/episode_models/tests/__init__.py`
- Test: `experiments/episode_models/tests/test_flow.py`, `experiments/episode_models/tests/test_layers.py`

**Interfaces:**
- Produces (`flow.py`): `interpolate(data, noise, t) -> Array`; `velocity_target(data, noise) -> Array`; `sample_times(key, batch_size) -> Float[B]`; `eval_times() -> np.ndarray` (8 values); `euler_sample(velocity: Callable[[PyTree, Float[B]], PyTree], noise: PyTree, num_steps: int) -> PyTree`.
- Produces (`layers.py`): `TrunkConfig(hidden_dim, num_layers)` with `num_heads`, `intermediate_dim`; `Linear.init(in_dim, out_dim, *, key)` with `__call__(x)`; `Trunk.init(config, *, key)` with `__call__(x: [B,S,D], positions: Int[S], mask: Bool[S,S]) -> [B,S,D]`; `TimeEmbedding.init(hidden_dim, *, key)` with `__call__(t: [B]) -> [B,D]`; `rotary(x: [B,S,H,K], positions: Int[S])`; `block_mask(blocks: np.ndarray) -> np.ndarray[bool]`; `patchify(frames: [B,T,H,W,C], patch) -> [B,T,N,P]`; `unpatchify(patches, patch, height, width) -> [B,T,H,W,C]`; `to_unit(frames_uint8, dtype)`; `init_weight(key, shape)`.

- [ ] **Step 1: Write failing tests**

```python
# test_flow.py
def test_the_path_runs_from_noise_to_data():
    data, noise = jnp.full((2, 3), 5.0), jnp.full((2, 3), -1.0)
    np.testing.assert_allclose(interpolate(data, noise, jnp.zeros(2)), noise)
    np.testing.assert_allclose(interpolate(data, noise, jnp.ones(2)), data)
    # A step of dt along the target velocity moves the path by dt.
    t = jnp.array([0.25, 0.5])
    moved = interpolate(data, noise, t) + 0.25 * velocity_target(data, noise)
    np.testing.assert_allclose(moved, interpolate(data, noise, t + 0.25), rtol=1e-6)

def test_euler_with_the_true_velocity_reaches_the_data():
    data = {"a": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones((2, 1))}
    noise = jax.tree.map(jnp.zeros_like, data)
    out = euler_sample(lambda x, t: jax.tree.map(lambda d, n: d - n, data, noise), noise, num_steps=4)
    jax.tree.map(lambda o, d: np.testing.assert_allclose(o, d, rtol=1e-6), out, data)

def test_sampled_times_lie_inside_the_unit_interval_and_eval_times_are_fixed():
    t = sample_times(jax.random.key(0), 4096)
    assert float(t.min()) > 0 and float(t.max()) < 1
    assert abs(float(jnp.median(t)) - 0.5) < 0.03
    np.testing.assert_allclose(eval_times(), (np.arange(8) + 0.5) / 8)

# test_layers.py
def test_rotary_at_consecutive_positions_matches_grug():
    q = jax.random.normal(jax.random.key(0), (2, 5, 3, 64))
    expected, _ = apply_rotary_embedding(q, q, seq_len=5, head_dim=64, rope=RotaryConfig())
    np.testing.assert_allclose(rotary(q, jnp.arange(5)), expected, rtol=1e-5, atol=1e-5)

def test_tokens_that_share_a_position_get_the_same_rotation():
    x = jnp.ones((1, 3, 1, 64))
    out = rotary(x, jnp.array([2, 2, 3]))
    np.testing.assert_array_equal(out[:, 0], out[:, 1])
    assert not np.allclose(out[:, 1], out[:, 2])

def test_block_mask_sees_its_block_and_earlier_ones():
    mask = block_mask(np.array([0, 1, 1, 2]))
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, True],
    ]

def test_patchify_round_trips():
    frames = jax.random.normal(jax.random.key(1), (2, 3, 16, 24, 3))
    patches = patchify(frames, 8)
    assert patches.shape == (2, 3, 6, 8 * 8 * 3)
    np.testing.assert_array_equal(unpatchify(patches, 8, 16, 24), frames)

def test_a_trunk_token_is_unaffected_by_later_blocks():
    trunk = Trunk.init(TrunkConfig(hidden_dim=64, num_layers=2), key=jax.random.key(2))
    x = jax.random.normal(jax.random.key(3), (1, 4, 64))
    blocks = np.array([0, 1, 1, 2])
    out = trunk(x, jnp.asarray(blocks), jnp.asarray(block_mask(blocks)))
    changed = trunk(x.at[:, 3].set(7.0), jnp.asarray(blocks), jnp.asarray(block_mask(blocks)))
    np.testing.assert_allclose(out[:, :3], changed[:, :3], rtol=1e-5, atol=1e-5)
    assert not np.allclose(out[:, 3], changed[:, 3])

def test_hidden_dim_must_be_a_multiple_of_the_head_dim():
    with pytest.raises(ValueError, match="multiple of 64"):
        TrunkConfig(hidden_dim=100, num_layers=1)
```

- [ ] **Step 2: Run them and see them fail** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `flow.py`**

```python
"""Flow matching on the rectified-flow path, with t = 1 clean: x_t = t * x + (1 - t) * noise."""

LOGIT_NORMAL_SCALE = 1.0
EVAL_TIME_COUNT = 8


def interpolate(data: Array, noise: Array, t: Float[Array, "B"]) -> Array:
    t = t.reshape(t.shape + (1,) * (data.ndim - t.ndim)).astype(data.dtype)
    return t * data + (1 - t) * noise


def velocity_target(data: Array, noise: Array) -> Array:
    return data - noise


def sample_times(key: PRNGKeyArray, batch_size: int) -> Float[Array, "B"]:
    """Logit-normal flow times, which weight the middle of the path (SD3)."""
    return jax.nn.sigmoid(LOGIT_NORMAL_SCALE * jax.random.normal(key, (batch_size,)))


def eval_times() -> np.ndarray:
    """The fixed grid a held-out loss averages over, so it is deterministic for given parameters."""
    return (np.arange(EVAL_TIME_COUNT) + 0.5) / EVAL_TIME_COUNT


def euler_sample(velocity, noise, num_steps: int):
    """Integrate ``velocity(x, t)`` from ``t = 0`` (``noise``) to ``t = 1`` in equal Euler steps."""
    dt = 1.0 / num_steps
    batch_size = jax.tree.leaves(noise)[0].shape[0]

    def step(x, i):
        v = velocity(x, jnp.full((batch_size,), i * dt, jnp.float32))
        return jax.tree.map(lambda a, b: a + dt * b.astype(a.dtype), x, v), None

    out, _ = jax.lax.scan(step, noise, jnp.arange(num_steps))
    return out
```

- [ ] **Step 4: Implement `layers.py`** — grug base's block (`experiments/grug/base/model.py`) without sharding annotations, plus a rotary embedding that takes positions, the block mask, patching, and the time embedding:

```python
HEAD_DIM = 64
MLP_RATIO = 4
INIT_STD = 0.02
NORM_EPS = 1e-5
ROPE_THETA = 10_000.0
TIME_FEATURES = 256
TIME_SCALE = 1000.0  # flow times in [0, 1] stretched so sinusoidal features resolve small steps


@dataclass(frozen=True)
class TrunkConfig:
    hidden_dim: int
    num_layers: int

    def __post_init__(self) -> None:
        if self.hidden_dim % HEAD_DIM:
            raise ValueError(f"hidden_dim must be a multiple of {HEAD_DIM}, not {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, not {self.num_layers}")

    @property
    def num_heads(self) -> int:
        return self.hidden_dim // HEAD_DIM

    @property
    def intermediate_dim(self) -> int:
        return MLP_RATIO * self.hidden_dim


def init_weight(key, shape) -> Array:
    return INIT_STD * jax.random.normal(key, shape, jnp.float32)


class Linear(eqx.Module):
    weight: Array
    bias: Array

    @staticmethod
    def init(in_dim: int, out_dim: int, *, key) -> "Linear":
        return Linear(weight=init_weight(key, (in_dim, out_dim)), bias=jnp.zeros((out_dim,), jnp.float32))

    def __call__(self, x):
        return jnp.einsum("...i,io->...o", x, self.weight) + self.bias


class RMSNorm(eqx.Module):
    weight: Array

    @staticmethod
    def init(dim: int) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), jnp.float32))

    def __call__(self, x):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + NORM_EPS)
        return (x * self.weight.astype(jnp.float32)).astype(dtype)


def rotary(x: Float[Array, "B S H K"], positions: Int[Array, "S"]) -> Float[Array, "B S H K"]:
    """Rotary embedding at explicit positions, in grug's split-half convention."""
    half = x.shape[-1] // 2
    inv_freq = ROPE_THETA ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(angles)[None, :, None, :], jnp.sin(angles)[None, :, None, :]
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def block_mask(blocks: np.ndarray) -> np.ndarray:
    """``[S, S]``: query ``i`` sees key ``j`` when ``blocks[j] <= blocks[i]``."""
    return blocks[None, :] <= blocks[:, None]


class Attention(eqx.Module):
    w_q: Array
    w_k: Array
    w_v: Array
    w_o: Array
    num_heads: int = eqx.field(static=True)

    @staticmethod
    def init(config: TrunkConfig, *, key) -> "Attention":
        k_q, k_k, k_v, k_o = jax.random.split(key, 4)
        d = config.hidden_dim
        return Attention(
            w_q=init_weight(k_q, (d, d)), w_k=init_weight(k_k, (d, d)), w_v=init_weight(k_v, (d, d)),
            w_o=init_weight(k_o, (d, d)), num_heads=config.num_heads,
        )

    def __call__(self, x, positions, mask):
        batch, seq, _ = x.shape
        heads = lambda w: jnp.einsum("bsd,de->bse", x, w).reshape(batch, seq, self.num_heads, HEAD_DIM)
        q, k, v = rotary(heads(self.w_q), positions), rotary(heads(self.w_k), positions), heads(self.w_v)
        dense = jnp.broadcast_to(mask, (batch, seq, seq))
        out = attention(q, k, v, dense, implementation="reference")
        return jnp.einsum("bse,ed->bsd", out.reshape(batch, seq, -1), self.w_o)


class MLP(eqx.Module):
    up: Array
    down: Array

    @staticmethod
    def init(config: TrunkConfig, *, key) -> "MLP":
        k_up, k_down = jax.random.split(key)
        return MLP(
            up=init_weight(k_up, (config.hidden_dim, config.intermediate_dim)),
            down=init_weight(k_down, (config.intermediate_dim, config.hidden_dim)),
        )

    def __call__(self, x):
        return jnp.einsum("bsm,md->bsd", jax.nn.relu(jnp.einsum("bsd,dm->bsm", x, self.up)), self.down)


class Block(eqx.Module):
    attention_norm: RMSNorm
    attention: Attention
    mlp_norm: RMSNorm
    mlp: MLP

    @staticmethod
    def init(config: TrunkConfig, *, key) -> "Block":
        k_attention, k_mlp = jax.random.split(key)
        return Block(
            attention_norm=RMSNorm.init(config.hidden_dim), attention=Attention.init(config, key=k_attention),
            mlp_norm=RMSNorm.init(config.hidden_dim), mlp=MLP.init(config, key=k_mlp),
        )

    def __call__(self, x, positions, mask):
        x = x + self.attention(self.attention_norm(x), positions, mask)
        return x + self.mlp(self.mlp_norm(x))


class Trunk(eqx.Module):
    blocks: tuple[Block, ...]
    final_norm: RMSNorm

    @staticmethod
    def init(config: TrunkConfig, *, key) -> "Trunk":
        keys = jax.random.split(key, config.num_layers)
        return Trunk(blocks=tuple(Block.init(config, key=k) for k in keys), final_norm=RMSNorm.init(config.hidden_dim))

    def __call__(self, x, positions, mask):
        for block in self.blocks:
            x = block(x, positions, mask)
        return self.final_norm(x)


class TimeEmbedding(eqx.Module):
    """Flow time to a hidden-size vector: sinusoidal features, then a two-layer SiLU MLP."""

    w_in: Linear
    w_out: Linear

    @staticmethod
    def init(hidden_dim: int, *, key) -> "TimeEmbedding":
        k_in, k_out = jax.random.split(key)
        return TimeEmbedding(w_in=Linear.init(TIME_FEATURES, hidden_dim, key=k_in), w_out=Linear.init(hidden_dim, hidden_dim, key=k_out))

    def __call__(self, t: Float[Array, "B"]) -> Float[Array, "B D"]:
        half = TIME_FEATURES // 2
        freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = TIME_SCALE * t.astype(jnp.float32)[:, None] * freqs[None, :]
        features = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(self.w_in.weight.dtype)
        return self.w_out(jax.nn.silu(self.w_in(features)))


def patchify(frames: Float[Array, "B T H W C"], patch: int) -> Float[Array, "B T N P"]:
    return rearrange(frames, "b t (h p) (w q) c -> b t (h w) (p q c)", p=patch, q=patch)


def unpatchify(patches, patch: int, height: int, width: int):
    return rearrange(patches, "b t (h w) (p q c) -> b t (h p) (w q) c", p=patch, q=patch, h=height // patch, w=width // patch)


def to_unit(frames: UInt8[Array, "..."], dtype) -> Array:
    """``uint8`` pixels to ``[-1, 1]`` in ``dtype``."""
    return frames.astype(dtype) / 127.5 - 1
```

- [ ] **Step 5: Run** `JAX_PLATFORMS=cpu uv run --no-sync pytest experiments/episode_models/tests -n 0` → pass.

- [ ] **Step 6: Commit** `[poorin] Add flow matching and trunk layers for episode models`

### Task 4: Episode data and sources

**Files:**
- Create: `experiments/episode_models/data.py`, `experiments/episode_models/sources.py`, `experiments/episode_models/tests/tiny_dataset.py`
- Test: `experiments/episode_models/tests/test_data.py`

**Interfaces:**
- Consumes: `clip_dataset(..., rows=...)` (Task 2).
- Produces (`data.py`): `BATCH_AXES = ("replica_dcn", "data")`; `ClipConfig(history=4, horizon=8, stride=4, skip=1)` with `.spec -> ClipSpec`, `.length`; `EpisodeDataConfig(video_field, action_field, task_key, clip=ClipConfig(), eval_percent=5, num_workers=16)`; `EpisodeDims(frame_shape, action_dim, num_tasks)`; `EpisodeBatch(frames, actions, task)` (a registered pytree dataclass); `held_out(episode_id, eval_percent) -> bool`; `split_rows(index, eval_percent) -> tuple[np.ndarray, np.ndarray]`; `task_vocabulary(index, task_key) -> tuple[str, ...]`; `dataset_dims(dataset, config) -> EpisodeDims`; `ActionStats(low, high)`; `ACTION_STATS_FILENAME`; `compute_action_stats(dataset_path, field, output_path) -> None`; `read_action_stats(path) -> ActionStats`; `normalize_actions(actions, stats) -> np.ndarray`; `action_stats(dataset_handle, field) -> ArtifactStep`; `example_dataset(dataset, config, rows, stats, tasks) -> grain.MapDataset`; `train_batches(examples, *, batch_size, seed, start_step, num_workers) -> Iterator[dict]`; `eval_batches(examples, *, batch_size, count) -> list[dict]`; `to_batch(arrays, mesh) -> EpisodeBatch`.
- Produces (`sources.py`): `EpisodeSource(dataset: Callable[[], ArtifactStep[EpisodeDataset]], video_field, action_field, task_key)`; `SOURCES: dict[str, EpisodeSource]`.
- Produces (`tests/tiny_dataset.py`): `build_tiny(tmp_path, episodes: int = 10, length: int = 16) -> EpisodeDataset` with a 16x16 `video`, a 3-dim `action` whose third dimension is constant, and an `instruction` per episode from two tasks.

- [ ] **Step 1: Write failing tests**

```python
def test_the_split_is_stable_and_disjoint(tmp_path):
    dataset = build_tiny(tmp_path)
    train, held = split_rows(dataset.index, eval_percent=50)
    assert set(train).isdisjoint(held) and len(train) + len(held) == len(dataset.index)
    # The hash is BLAKE2b, not Python's salted hash, so another process agrees.
    script = "from experiments.episode_models.data import held_out; print(held_out('e3', 50))"
    other = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert other.stdout.strip() == str(held_out("e3", 50))

def test_split_refuses_an_empty_side(tmp_path):
    dataset = build_tiny(tmp_path)
    with pytest.raises(ValueError, match="no held-out episodes"):
        split_rows(dataset.index, eval_percent=0)

def test_normalization_maps_the_percentiles_to_the_unit_interval():
    stats = ActionStats(low=np.array([0.0, -2.0]), high=np.array([10.0, 2.0]))
    out = normalize_actions(np.array([[0.0, -2.0], [10.0, 2.0], [5.0, 0.0], [20.0, -9.0]], np.float32), stats)
    np.testing.assert_allclose(out, [[-1, -1], [1, 1], [0, 0], [1, -1]])

def test_a_constant_action_dimension_normalizes_to_finite_values(tmp_path):
    dataset = build_tiny(tmp_path)
    compute_action_stats(dataset.path, "action", str(tmp_path / "stats"))
    stats = read_action_stats(str(tmp_path / "stats"))
    assert stats.low[2] == stats.high[2]
    assert np.isfinite(normalize_actions(np.ones((4, 3), np.float32), stats)).all()

def test_examples_carry_frames_normalized_actions_and_task_indices(tmp_path):
    dataset = build_tiny(tmp_path)
    compute_action_stats(dataset.path, "action", str(tmp_path / "stats"))
    config = EpisodeDataConfig(video_field="video", action_field="action", task_key="instruction",
                               clip=ClipConfig(history=2, horizon=2, stride=4, skip=1), eval_percent=50, num_workers=0)
    tasks = task_vocabulary(dataset.index, "instruction")
    examples = example_dataset(dataset, config, np.arange(len(dataset.index)), read_action_stats(str(tmp_path / "stats")), tasks)
    first = examples[0]
    assert first["frames"].shape == (4, 16, 16, 3) and first["frames"].dtype == np.uint8
    assert first["actions"].shape == (4, 3) and np.abs(first["actions"]).max() <= 1
    assert tasks == ("pour", "stack") and first["task"] in (0, 1)
    assert dataset_dims(dataset, config) == EpisodeDims(frame_shape=(16, 16, 3), action_dim=3, num_tasks=2)

def test_eval_batches_refuse_a_split_smaller_than_a_batch(tmp_path):
    dataset = build_tiny(tmp_path, episodes=2)
    config = EpisodeDataConfig(video_field="video", action_field=None, task_key=None, clip=ClipConfig(history=2, horizon=2, stride=4, skip=1), num_workers=0)
    examples = example_dataset(dataset, config, np.array([0]), None, None)
    with pytest.raises(ValueError, match="fewer clips than one batch"):
        eval_batches(examples, batch_size=64, count=1)
```

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement.** `tests/tiny_dataset.py` defines `read(source, options)` yielding episodes `e<i>` of `length` steps, frames a ramp by step, actions `[sin(step + i), cos(step), 0.5]`, `instruction` `("pour", "stack")[i % 2]`, and `build_tiny` runs `build_episodes` into `tmp_path / "tiny"` and returns `EpisodeDataset.raw_load(...)`. `data.py` implements the interfaces above:

```python
def held_out(episode_id: str, eval_percent: int) -> bool:
    digest = hashlib.blake2b(episode_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100 < eval_percent


def split_rows(index: pd.DataFrame, eval_percent: int) -> tuple[np.ndarray, np.ndarray]:
    held = np.array([held_out(str(e), eval_percent) for e in index["episode"]], dtype=bool)
    train, evaluation = np.flatnonzero(~held), np.flatnonzero(held)
    if not len(evaluation):
        raise ValueError(f"no held-out episodes at eval_percent={eval_percent} among {len(index)}; raise it")
    if not len(train):
        raise ValueError(f"no training episodes at eval_percent={eval_percent} among {len(index)}; lower it")
    return train, evaluation


def normalize_actions(actions: np.ndarray, stats: ActionStats) -> np.ndarray:
    span = stats.high - stats.low
    scaled = np.where(span > 0, 2 * (actions - stats.low) / np.where(span > 0, span, 1) - 1, 0)
    return np.clip(scaled, -1, 1).astype(np.float32)


def action_stats(dataset: ArtifactStep[EpisodeDataset], field: str) -> ArtifactStep[Artifact]:
    """Percentiles of one action field, versioned with its dataset, so each dataset version computes them once."""
    return apply(f"episode-stats/{dataset.name}/{field}", compute_action_stats, version=dataset.version,
                 dataset_path=dataset, field=field, output_path=OUT)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeBatch:
    frames: Array            # uint8 [B, T, H, W, C]
    actions: Array | None    # float32 [B, T, A], normalized
    task: Array | None       # int32 [B]
```

`example_dataset` reads `[video_field]` plus `action_field` when set, and `task_key` as metadata when set, through `clip_dataset(..., rows=rows)`, and maps each clip with a picklable `_ToExample` that renames fields to `frames`, `actions`, `task`, normalizes actions, and maps the instruction to its index. `train_batches` shuffles with the seed, repeats, and calls `batch_iterator`; `eval_batches` batches in index order with `drop_remainder=True` and takes the first `count`; `to_batch` calls `to_mesh(arrays, mesh, BATCH_AXES)`.

`sources.py`:

```python
SOURCES = {
    "libero-spatial": EpisodeSource(libero_spatial, video_field="agentview", action_field="action", task_key="instruction"),
    "vpt-10": EpisodeSource(vpt_subset, video_field="video", action_field="action", task_key=None),
    "ssv2-500": EpisodeSource(ssv2_subset, video_field="video", action_field=None, task_key=None),
}
```

- [ ] **Step 4: Run** the tests → pass.

- [ ] **Step 5: Commit** `[poorin] Read episode datasets as model batches`

### Task 5: FLOPs and the WAM model

**Files:**
- Create: `experiments/episode_models/flops.py`, `experiments/episode_models/wam/__init__.py`, `experiments/episode_models/wam/model.py`
- Test: `experiments/episode_models/tests/test_models.py`

**Interfaces:**
- Produces (`flops.py`): `trunk_flops_per_token(trunk: TrunkConfig, seq_len: int) -> float`; `linear_flops(in_dim, out_dim) -> float`; `time_embedding_flops(hidden_dim) -> float`.
- Produces (`wam/model.py`): `WamConfig(hidden_dim=384, num_layers=6, patch_size=16, action_loss_weight=1.0)` with `streams = ("video", "action")`, `conditions_on_task = True`, `needs_actions = True`, `trunk`, `stream_weights`, `init(dims, clip, *, key) -> Wam`, `tokens_per_example(clip, dims)`, `flops_per_example(clip, dims)` (forward); `Wam.stream_losses(batch, key, times) -> dict[str, Array]`; `Wam.sample(batch, key, num_steps) -> dict[str, Array]` (`video`: `[B, horizon, H, W, C]` in `[-1, 1]`, `action`: `[B, horizon, A]`).

- [ ] **Step 1: Write failing tests** (parametrized over families in Task 6; WAM first)

```python
def test_the_analytic_flops_match_xla_within_ten_percent(family):
    config, dims, clip, batch = tiny_setup(family, hidden_dim=256, num_layers=2)
    model = config.init(dims, clip, key=jax.random.key(0))
    times = {s: jnp.full((batch.frames.shape[0],), 0.5) for s in config.streams}
    compiled = jax.jit(lambda m, b: m.stream_losses(b, jax.random.key(1), times)).lower(model, batch).compile()
    xla = compiled.cost_analysis()["flops"]
    analytic = config.flops_per_example(clip, dims) * batch.frames.shape[0]
    assert abs(analytic - xla) / xla < 0.10

def test_the_sampler_returns_the_future_of_each_stream(family):
    config, dims, clip, batch = tiny_setup(family)
    model = config.init(dims, clip, key=jax.random.key(0))
    out = model.sample(batch, jax.random.key(2), num_steps=3)
    again = model.sample(batch, jax.random.key(2), num_steps=3)
    assert set(out) == set(config.streams)
    if "video" in out:
        assert out["video"].shape == (2, clip.horizon, 16, 16, 3)
    if "action" in out:
        assert out["action"].shape == (2, clip.horizon, 3)
    jax.tree.map(np.testing.assert_array_equal, out, again)

def test_init_refuses_frames_the_patch_does_not_divide(family):
    config, dims, clip, _ = tiny_setup(family)
    with pytest.raises(ValueError, match="patch_size 5"):
        dataclasses.replace(config, patch_size=5).init(dims, clip, key=jax.random.key(0))

def test_a_future_frame_changes_no_history_prediction():  # WAM: block causality reaches the loss
    ...  # perturb the last future frame; the first future step's velocity prediction is unchanged
```

`tiny_setup(family, hidden_dim=64, num_layers=1)` builds the family's config with `patch_size=8`, `EpisodeDims((16, 16, 3), 3, 2)` (`action_dim=None, num_tasks=None` for the world model), `ClipConfig(history=2, horizon=2, stride=1, skip=1)`, and a random batch of 2.

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement `flops.py`**

```python
def trunk_flops_per_token(trunk: TrunkConfig, seq_len: int) -> float:
    return lm_flops_per_token(
        hidden_dim=trunk.hidden_dim, intermediate_dim=trunk.intermediate_dim, num_layers=trunk.num_layers,
        num_kv_heads=trunk.num_heads, num_heads=trunk.num_heads, seq_len=seq_len, vocab_size=0, glu=False,
    )

def linear_flops(in_dim: int, out_dim: int) -> float:
    return 2.0 * in_dim * out_dim

def time_embedding_flops(hidden_dim: int) -> float:
    return linear_flops(TIME_FEATURES, hidden_dim) + linear_flops(hidden_dim, hidden_dim)
```

- [ ] **Step 4: Implement `wam/model.py`.** Token layout per spec: per timestep `N` patch tokens then one action token; a leading task token when `num_tasks` is set; `position = block = timestep + 1`, task token 0. Future frames and actions are noised at their stream's time; history tokens carry the `t = 1` embedding. Heads run on future tokens only.

```python
@dataclass(frozen=True)
class WamConfig:
    hidden_dim: int = 384
    num_layers: int = 6
    patch_size: int = 16
    action_loss_weight: float = 1.0
    streams: ClassVar[tuple[str, ...]] = ("video", "action")
    conditions_on_task: ClassVar[bool] = True
    needs_actions: ClassVar[bool] = True

    @property
    def trunk(self) -> TrunkConfig:
        return TrunkConfig(self.hidden_dim, self.num_layers)

    @property
    def stream_weights(self) -> dict[str, float]:
        return {"video": 1.0, "action": self.action_loss_weight}

    def init(self, dims: EpisodeDims, clip: ClipConfig, *, key) -> "Wam": ...
    def tokens_per_example(self, clip, dims) -> int:
        return int(dims.num_tasks is not None) + clip.length * (num_patches(dims, self.patch_size) + 1)
    def flops_per_example(self, clip, dims) -> float:
        n, p, a, d = num_patches(dims, self.patch_size), patch_dim(dims, self.patch_size), dims.action_dim, self.hidden_dim
        seq = self.tokens_per_example(clip, dims)
        return (
            seq * trunk_flops_per_token(self.trunk, seq)
            + clip.length * (n * linear_flops(p, d) + linear_flops(a, d))
            + clip.horizon * (n * linear_flops(d, p) + linear_flops(d, a))
            + 3 * time_embedding_flops(d)  # the video time, the action time, and the clean time of history
        )
```

`num_patches(dims, patch)` and `patch_dim(dims, patch)` live in `layers.py` (move there in this task), and `num_patches` raises `ValueError(f"frames of {h}x{w} do not divide into patches of patch_size {patch}")`. `Wam` holds `patch_embed: Linear`, `patch_position: Array [N, D]`, `action_embed: Linear`, `task_embed: Array | None`, `video_time: TimeEmbedding`, `action_time: TimeEmbedding`, `trunk: Trunk`, `frame_head: Linear`, `action_head: Linear`, and static `patch_size`, `history`, `frame_shape`. `_predict(history_patches, history_actions, future_patches, future_actions, task, t_video, t_action) -> (v_frames, v_actions)` assembles the tokens, runs the trunk, and applies the heads; `stream_losses` builds noisy futures with `interpolate` and returns the mean squared error against `velocity_target` in float32; `sample` runs `euler_sample` on the pytree `(future_patches, future_actions)` and unpatchifies.

- [ ] **Step 5: Run** → pass (WAM cases).

- [ ] **Step 6: Commit** `[poorin] Add the WAM template's model and FLOP count`

### Task 6: World model and VLA

**Files:**
- Create: `experiments/episode_models/world_model/{__init__.py,model.py}`, `experiments/episode_models/vla/{__init__.py,model.py}`
- Test: `experiments/episode_models/tests/test_models.py` (the `family` fixture gains both)

**Interfaces:**
- Produces: `WorldModelConfig(hidden_dim=384, num_layers=6, patch_size=16)` (`streams = ("video",)`, `conditions_on_task = False`, `needs_actions = False`) and `WorldModel`; `VlaConfig(hidden_dim=384, num_layers=6, patch_size=16)` (`streams = ("action",)`, `conditions_on_task = True`, `needs_actions = True`) and `Vla`. Same methods as Task 5.

- [ ] **Step 1: Add the two families to the fixture and add:**

```python
def test_the_vlas_actions_see_every_history_frame_and_each_other():
    config, dims, clip, batch = tiny_setup("vla")
    model = config.init(dims, clip, key=jax.random.key(0))
    blocks = model.blocks(task=True)
    mask = block_mask(blocks)
    actions = np.flatnonzero(blocks == blocks.max())
    assert mask[np.ix_(actions, np.arange(len(blocks)))].all()
```

- [ ] **Step 2: Run and see the new cases fail.**

- [ ] **Step 3: Implement** both by copying `wam/model.py` and deleting the removed stream (world model: action tokens, task token, action head, action time; VLA: future frame tokens, frame head, video time, history actions; the VLA's actions get `block = history + 1` and `position = timestep + 1`). `Vla.blocks(task)` and `Vla.positions(task)` return the numpy layout the model uses, so the test reads the real layout. FLOP counts follow the same formula with the removed terms gone; the VLA embeds only history frames: `history * n * linear_flops(p, d)`.

- [ ] **Step 4: Run** → all families pass.

- [ ] **Step 5: Commit** `[poorin] Add the world model and VLA templates' models`

### Task 7: The shared training loop

**Files:**
- Create: `experiments/episode_models/train.py`
- Test: `experiments/episode_models/tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 3–6; `experiments.grug.checkpointing.restore_grug_state_from_checkpoint`; `experiments.grug.dispatch.dispatch_grug_training_run`.
- Produces: `EpisodeModelConfig` and `EpisodeModel` protocols (spec); `EpisodeTrainConfig(steps=2000, batch_size=32, seed=0, mp="params=float32,compute=bfloat16,output=bfloat16", optimizer=AdamConfig(learning_rate=1e-3, weight_decay=0.01, warmup=0.05, lr_schedule="cosine", min_lr_ratio=0.1), steps_per_eval=250, eval_batches=4, checkpoint_every=500, log_every=1)`; `EpisodeTrainingRun(model, dataset_path, action_stats_path, data, train, trainer: TrainerConfig, resources)`; `EpisodeTrainState(step, params, opt_state, ema_params)`; `run_episode_training(run) -> None`; `EVAL_SEED = 0`.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_trains_and_logs_the_sweep_contract(tmp_path, family):
    records = train_tiny(tmp_path, family, steps=3)
    summary = finish_summary(records)
    for key in ["train/loss", "eval/loss", "parameter_count", "throughput/flops_per_example_analytic", "throughput/total_tokens"]:
        assert key in summary
    for stream in STREAMS[family]:
        assert f"eval/{stream}_loss" in summary

def test_a_resumed_run_reports_the_losses_of_an_uninterrupted_one(tmp_path):
    uninterrupted = losses_by_step(train_tiny(tmp_path / "a", "wam", steps=6))
    train_tiny(tmp_path / "b", "wam", steps=4)
    resumed = losses_by_step(train_tiny(tmp_path / "b", "wam", steps=6))  # restores step 4
    assert sorted(resumed) == [5, 6]
    for step in (5, 6):
        np.testing.assert_allclose(resumed[step], uninterrupted[step], rtol=1e-6)

def test_the_held_out_loss_is_the_same_on_every_call(tmp_path):
    ...  # evaluate the same parameters twice through the loop's evaluate function; equal to the bit
```

`train_tiny` builds the tiny dataset and its stats, an `EpisodeTrainingRun` with the family's config at `hidden_dim=64, num_layers=1, patch_size=8`, `ClipConfig(history=2, horizon=2, stride=2, skip=1)`, `eval_percent=50`, `num_workers=0`, a constant learning rate, `steps_per_eval=steps`, a `JsonLoggerConfig` tracker on a fresh logger, `require_accelerator=False`, `DistributedConfig(initialize_jax_distributed=False)`, and a checkpointer under the run directory; it returns the JSON records.

- [ ] **Step 2: Run and see them fail.**

- [ ] **Step 3: Implement `train.py`** following grug base's `_run_grug_local`: initialize the trainer and log the run config; load the dataset, dims, stats, tasks, and split; build the train and eval examples; init the state under `jax.jit` on `compact_grug_mesh()`; restore; log `parameter_count` and the analytic FLOPs (3x forward); place the eval batches once; then loop: next batch → `to_batch` → `train_step(state, batch, fold_in(noise_key, step))` → callbacks (Levanter's performance stats with `tokens_per_example` and the training FLOPs, pbar, step info) → stream losses → eval every `steps_per_eval` and at the end → `checkpointer.on_step`. Force the final callbacks and checkpoint, wait, finish the tracker.

```python
def _make_train_step(optimizer, mp, streams, weights):
    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch, key):
        time_key, noise_key = jax.random.split(key)
        size = batch.frames.shape[0]
        times = {s: sample_times(jax.random.fold_in(time_key, i), size) for i, s in enumerate(streams)}

        def loss_fn(params):
            losses = mp.cast_to_compute(params).stream_losses(batch, noise_key, times)
            return sum(weights[s] * losses[s] for s in streams), losses

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        next_state = dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state)
        return next_state, {"train/loss": loss, **{f"train/{s}_loss": losses[s] for s in streams}}

    return train_step


def _make_evaluate(mp, streams, weights):
    grid = jnp.asarray(eval_times(), jnp.float32)

    @jax.jit
    def evaluate_batch(params, batch, key):
        model = mp.cast_to_compute(params)
        size = batch.frames.shape[0]
        losses = jax.lax.map(lambda t: model.stream_losses(batch, key, {s: jnp.full((size,), t) for s in streams}), grid)
        return {s: losses[s].mean() for s in streams}

    def evaluate(params, batches) -> dict[str, float]:
        per_batch = [evaluate_batch(params, b, jax.random.fold_in(jax.random.key(EVAL_SEED), i)) for i, b in enumerate(batches)]
        means = {s: float(np.mean([float(r[s]) for r in per_batch])) for s in streams}
        return {"eval/loss": sum(weights[s] * means[s] for s in streams), **{f"eval/{s}_loss": means[s] for s in streams}}

    return evaluate
```

`run_episode_training(run)` calls `dispatch_grug_training_run(run_id=run.trainer.id, config=run, local_entrypoint=_train_locally, resources=run.resources)`.

- [ ] **Step 4: Run** → pass.

- [ ] **Step 5: Commit** `[poorin] Train episode models through one shared loop`

### Task 8: Launchers

**Files:**
- Create: `experiments/episode_models/launch.py`, `experiments/episode_models/{wam,world_model,vla}/launch.py`
- Test: `experiments/episode_models/tests/test_launch.py`

**Interfaces:**
- Produces (`launch.py`): `TRAIN_RESOURCES`; `EpisodeRunConfig(output_path, run_id, resources, tracker, finelog_url, dataset_path, action_stats_path)`; `EpisodeLaunchConfig(model, data, train, run)`; `build_training_run(launch) -> EpisodeTrainingRun`; `run_episode_trial(launch) -> None`; `episode_trial(*, family: str, model: EpisodeModelConfig, dataset: str, train: EpisodeTrainConfig, overrides: Overrides, version: str | None, finelog_url: str | None) -> ArtifactStep[LevanterCheckpoint]`.
- Produces (each family): `<family>_trial(*, dataset, overrides, version=None, finelog_url=None)` and a click `main` with `--dataset` (choices `SOURCES`, family default) and `--finelog-url`, wrapped in `build_options`.

- [ ] **Step 1: Write failing tests**

```python
def test_a_family_refuses_a_source_without_its_fields():
    with pytest.raises(ValueError, match="ssv2-500 has no action field"):
        vla_trial(dataset="ssv2-500", overrides=Overrides(), version="2026.10.11")

def test_a_set_reaches_the_model_and_names_the_trial():
    trial = wam_trial(dataset="libero-spatial", overrides=Overrides((Override.parse("model.hidden_dim=512"),)), version="2026.10.11")
    assert trial.name == "episode/wam-libero-spatial/model.hidden_dim=512"
    assert '"hidden_dim": 512' in trial.fingerprint_payload()

def test_the_dataset_and_its_action_stats_are_dependencies():
    trial = wam_trial(dataset="libero-spatial", overrides=Overrides(), version="2026.10.11")
    assert [d.name for d in trial.deps] == ["examples/libero-spatial", "episode-stats/examples/libero-spatial/action"]

def test_the_world_model_reads_no_actions_even_from_a_source_with_them():
    trial = world_model_trial(dataset="vpt-10", overrides=Overrides(), version="2026.10.11")
    ...  # its fingerprint payload has "action_field": null and only the dataset as a dependency
```

The VPT and SSv2 handles read the network or a local SSv2 tree when built, so the launch tests that name them monkeypatch `SOURCES` with a stand-in handle of the same fields.

- [ ] **Step 2: Run and see them fail.**

- [ ] **Step 3: Implement.** `episode_trial` looks up the source, refuses a missing action field when `model.needs_actions`, builds the data config (`action_field` only when the family uses actions, `task_key` only when it conditions on tasks), adds `action_stats(handle, field)` as a dependency when actions are read, names the trial `episode/<family>-<dataset>` plus overrides, and its `build_config` returns `overrides.apply(ctx, EpisodeLaunchConfig(...))` with the run fields pulled from the context (`ctx.output_path`, `ctx.artifact_path(...)`, `ctx.runtime_arg(...)`). `build_training_run` maps the launch config onto a `TrainerConfig` like grug's `build_grug_run_config` (W&B with `replicate_path`, telemetry, checkpointer under the output path keeping every `checkpoint_every` steps, `require_accelerator=True`).

- [ ] **Step 4: Run** → pass. Then `uv run --no-sync python experiments/episode_models/wam/launch.py --version dev` prints the plan.

- [ ] **Step 5: Commit** `[poorin] Launch episode model trials through the experiment CLI`

### Task 9: Documentation and registration

**Files:**
- Create: `experiments/episode_models/README.md`
- Modify: `docs/poorin/architecture.md` (a row per module), `tools/roots.toml` (`"experiments.episode_models.*"` under the episode workbench roots), `AGENTS.md` (nothing unless a command changes)

- [ ] **Step 1: Write the README**: what the three templates are, the commands (plan, run, `--set`, `--sweep`, `--dataset`, `--resume`), the batch contract, how to add a family (copy a `model.py`, meet the protocol, add a `launch.py`), and the metrics a sweep reads.
- [ ] **Step 2: Add the architecture rows and the root**; run `uv run --no-sync pytest tests/tools/test_architecture.py -n 0`.
- [ ] **Step 3: Full check**: `uv run --no-sync python -m tools.prune_check --phase P9 --step "episode model templates"` (lint, ty, affected tests, both training roots, a metrics row).
- [ ] **Step 4: Commit** `[poorin] Document the episode model templates`

### Task 10: Acceptance on the RTX 5090

- [ ] **Step 1:** `world_model/launch.py --dataset ssv2-500`, `wam/launch.py --dataset libero-spatial`, `vla/launch.py --dataset libero-spatial`, each `--version dev --run`. Check that `eval/loss` falls. Interrupt one run after its first permanent checkpoint and continue it with `--resume VERSION`; it restores and finishes.
- [ ] **Step 2:** each family with `--sweep 'model.hidden_dim=[128, 256, 384]' --set train.steps=1000`; every trial logs `parameter_count`, `throughput/flops_per_example_analytic`, and `eval/loss`.
- [ ] **Step 3:** record the numbers under D113 "Acceptance", commit `[poorin] Record the episode templates' acceptance runs (D113)`.
