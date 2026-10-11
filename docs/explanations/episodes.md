# Episode datasets

Video world models, VLAs, and WAMs train on episodes: sequences of per-step frames, states, and
actions. `poorin.episodes` stores them as one format and reads them as clips through
[Grain](https://google-grain.readthedocs.io/). A plain video is an episode with no action field, so
one format covers video-only data, robot trajectories, and latents. Text stays on Levanter's
token caches (see [The Language Modeling Pipeline](lm-pipeline.md)).

An episode dataset is an artifact like any other: an `ArtifactStep[EpisodeDataset]` handle built
by a step and cached at `{prefix}/{name}/{version}` (see [Lazy artifacts](lazy-artifacts.md)).

## Episodes

An episode is a set of named per-step arrays that share one step clock. A dataset's schema declares
each field's kind, dtype, per-step shape, and encoding, and one `step_rate` in steps per second for
every field. A reader resamples a source whose streams run at different rates.

A dataset is one directory:

```text
{prefix}/{name}/{version}/
  dataset.json              # the schema
  episodes.parquet          # one row per episode
  shard-00000.array_record  # one msgpack record per episode
  shard-00001.array_record
  ...
```

`dataset.json` is `DatasetSchema.to_json()`:

```json
{
  "step_rate": 20.0,
  "fields": {
    "agentview": {"kind": "video", "dtype": "uint8", "shape": [128, 128, 3],
                  "encoding": {"name": "video", "codec": "libx264", "keyframe_interval": 16, "crf": 18}},
    "action": {"kind": "array", "dtype": "float32", "shape": [7], "encoding": {"name": "raw"}}
  }
}
```

A field's `shape` is per step; an episode of length T holds `[T, *shape]`. A dataset without actions
declares no action field, and code that needs one checks the schema.

`episodes.parquet` holds one row per episode:

| Column | Meaning |
|---|---|
| `episode` | Episode id, unique in the dataset |
| `shard`, `record` | Where the episode's record is |
| `length` | Number of steps |
| `metadata` | Per-episode values as JSON: a language instruction, an embodiment, a source path |

The clip index is planned from this table alone, without opening a shard. Each shard is an
ArrayRecord file, which reads any record by index. A record is one episode: a msgpack map from
field name to that field's encoded bytes.

## Field encodings

How a field is stored is a property of the field, not of the format:

| Encoding | Stores | Decoding steps `[s, s+T)` |
|---|---|---|
| `RawEncoding()` | The `[length, *shape]` array's bytes, C order | Slice |
| `VideoEncoding(codec, keyframe_interval, crf)` | One H.264 stream per episode, a keyframe every `keyframe_interval` steps | Seek to the keyframe at or before `s`, then decode forward |

A video field is `uint8 [H, W, 3]` with an even height and width. The encoder writes no B-frames and
no scene-cut keyframes, so frame `i` has timestamp `i` and keyframes fall exactly every
`keyframe_interval` frames. A clip therefore decodes at most `keyframe_interval - 1` frames it
does not return. A shorter interval makes clips cheaper to decode and files larger. `crf` sets the
quality: 18 is close to visually lossless, and H.264's 4:2:0 chroma halves color resolution.

Raw fields hold actions, proprioception, and latents. A latent dataset is an episode dataset with a
raw float field, read through the same path.

To add an encoding, such as a JPEG per frame or float16 latents, add one class in
`poorin.episodes.schema` (and its JSON form there) and one branch in each of `encode_field` and
`decode_field` in `poorin.episodes.encodings`.

## Building a dataset

A reader turns one source unit into episodes:

```python
def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]: ...
```

A source unit is a string the reader resolves: a file path, a Hub file name, a clip id. The reader
delivers each field at the schema's shape and step rate; `poorin.episodes.video_files.load_video`
decodes a video file at a given size (scaled on the shorter side, then center-cropped) and step
rate. The build checks every episode against the schema and fails, naming the episode and the field,
on a missing or extra field, a wrong dtype or per-step shape, or fields of unequal length. It also
refuses an episode id that appears twice.

`poorin.experiment.data.episodes(...)` returns the handle. The Something-Something v2 example
(`experiments/datasets/episodes/ssv2.py`) stores video clips without actions, with each clip's label
as metadata:

```python
SCHEMA = DatasetSchema(
    step_rate=12.0,
    fields={"video": FieldSpec(FieldKind.VIDEO, "uint8", (128, 128, 3), VideoEncoding(keyframe_interval=16))},
)


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One SSv2 clip id."""
    root = options["root"]
    video = load_video(os.path.join(root, VIDEO_DIR, f"{source}.webm"), size=(128, 128), step_rate=12.0)
    yield Episode(id=source, fields={"video": video}, metadata={"label": _labels(root)[source]})


def ssv2_subset(root: str = SSV2_ROOT) -> ArtifactStep[EpisodeDataset]:
    ids = sorted(_labels(root))[:SUBSET_CLIPS]
    return episodes(
        "examples/ssv2-500",
        sources=ids,
        reader="experiments.datasets.episodes.ssv2:read",
        reader_options={"root": root},
        schema=SCHEMA,
        num_shards=16,
        version="2026.10.11",
    )
```

The build is one Zephyr stage. The sources are split into `num_shards` groups, and each shard reads
its group, encodes every field, and writes one ArrayRecord file, so the decoded arrays never pass
between stages. The build then writes `episodes.parquet` and `dataset.json`.

The sources, the reader's import path, `reader_options`, the schema, and `num_shards` are the
recipe the fingerprint hashes. The reader is a `"module.path:function"` string because the
fingerprint encodes the config as JSON, which a function object is not. Changing the resolution or
the keyframe interval is a new recipe, and the drift check refuses it under an existing version. A
change inside a reader is a code change: the record names the commit, and the version is bumped
when the change affects results.

`experiments/datasets/episodes/` also has LIBERO-Spatial (two cameras, joint and gripper state,
actions) and a ten-episode Minecraft VPT subset (long video with keyboard and mouse actions). They
are examples that test the format, not supported datasets.

## Reading clips in training

A clip is `length` steps taken every `skip` steps; clip starts in an episode are `stride` steps
apart. A clip spans `(length - 1) * skip + 1` steps, never crosses its episode's end, and an
episode shorter than one clip has none.

`clip_dataset` returns every clip of a dataset as a finite, unshuffled `grain.MapDataset` of
`{field: [length, *shape]}` arrays, plus any requested metadata values. Only the selected fields are
decoded. The training script shuffles and repeats it; `shuffle` before `repeat` gives each epoch its
own order. `batch_iterator` batches it, starts at `start_step`, and decodes in worker processes.
`to_mesh` places each batch's numeric fields on the mesh, sharded on the batch dimension, and leaves
text fields as numpy arrays.

The end-to-end check (`experiments/datasets/episodes/smoke_train.py`) reads LIBERO clips into a toy
JAX step:

```python
mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
clips = clip_dataset(dataset, clip=ClipSpec(length=8, stride=4, skip=1), fields=["agentview", "action"])
iterator = iter(
    batch_iterator(clips.shuffle(seed=SEED).repeat(None), batch_size=16, start_step=start_step, num_workers=8)
)
try:
    for step in range(start_step, steps):
        batch = to_mesh(next(iterator), mesh, ("data",))
        weights, loss = train_step(weights, batch["agentview"], batch["action"])
finally:
    iterator.close()
```

Close the iterator: it owns the worker processes.

**Resuming.** Every layer before batching is a `MapDataset` with random access, so batch N is a
function of the seed and N alone. A run that resumes at step N passes `start_step=N` and reads the
batches an uninterrupted run would have read. No data-loader state goes in the checkpoint.

**Transforms.** Resizing, cropping, and normalizing are the caller's `.map`, or `.random_map` for
random crops; Grain seeds `random_map` per index, so they stay deterministic across resumes.

## Mixing datasets

Map each dataset's clips to the same fields, then mix them with Grain:

```python
def agentview_as_video(clip: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"video": clip["agentview"]}


robot = clip_dataset(libero, clip=clip, fields=["agentview"]).map(agentview_as_video)
video = clip_dataset(ssv2, clip=clip, fields=["video"])
mixed = grain.MapDataset.mix([robot, video], weights=[1, 1])
```

A video-only dataset has no action field. A model that trains on both conditions on whether a clip
has actions, or masks the action loss.

## Throughput

`experiments/datasets/episodes/throughput.py` measures clips per second against decoding workers.
On the RTX 5090 machine (32 cores), the SSv2 example at 128 px with 16-step clips and batches of 32
measured, on 2026-10-11:

| Workers | Clips/s |
|---|---|
| 1 | 760 |
| 4 | 1,470 |
| 8 | 1,632 |
| 16 | 2,219 |
| 24 | 2,838 |

The target is 1,000 clips per second. If a model needs more, the levers are a shorter keyframe
interval (less wasted decoding per clip), a smaller frame size, an encoding that decodes per frame
(JPEG), or decoding on the GPU.
