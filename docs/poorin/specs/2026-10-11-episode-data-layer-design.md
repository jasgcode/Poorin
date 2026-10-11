# Episode data layer

Status: approved design, 2026-10-11 (plan `plans/2026-10-11-episode-data-layer.md`). Sub-project 3.
Decision D110.

## TL;DR

Poorin's data path today serves language models only: Levanter tokenizes text into a TensorStore
cache and its loader feeds token windows to grug. Sub-project 3 adds one data format for video
world models, VLAs, and WAMs, where an episode is a set of named per-step fields and a plain video
is an episode with no action field.

- An episode dataset is an artifact directory: a schema (`dataset.json`), an index with one row per
  episode (`episodes.parquet`), and ArrayRecord shards with one msgpack record per episode.
- Each field declares how it is stored. The first two encodings are `raw` (array bytes: actions,
  proprioception, latents) and `video` (H.264 or AV1 with a short keyframe interval, so a clip
  decodes from the nearest keyframe, not from the episode's start).
- Datasets are built by an `episodes(...)` handle whose run is a Zephyr pipeline, like
  `tokenized(...)`. A source reader, one small function per source, is the only dataset-specific
  code.
- Training reads clips through a Grain `MapDataset`: a clip index planned from the parquet index,
  a deterministic shuffle, decoding in worker processes, and an adapter that puts batches on the
  mesh. The batch at step N is a pure function of (seed, N), so a resumed run reads what an
  uninterrupted one would have.
- Text stays on Levanter's caches. Levanter stays the training library: checkpoints, the tracker,
  optimizers, attention kernels.

## Background

### Today's data path

`poorin.experiment.data.tokenized()` returns an `ArtifactStep[TokenizedCache]`. Its run is a
Zephyr pipeline (`processing/tokenize/`) that writes a Levanter cache: a `TreeStore` with one
`JaggedArrayStore` per pytree leaf, each a flat TensorStore array plus offsets, zarr3 with blosc.
Grug builds a `MixtureDataset[GrugLmExample]` from `LmDataConfig` and reads it through Levanter's
`DataLoader`, which prefetches in one background thread and builds sharded global arrays.

Nothing in the repository reads images, video, or trajectories, and no media library is a
dependency.

### Why episodes do not go through Levanter's data stack

Levanter's data stack was built for language-model pretraining on TPU pods: many hosts, data in
object storage, and items that are small integer arrays costing nothing to decode. Its design
follows from that: data order is a pure function of the step (pseudo-random permutations,
block-deterministic mixtures), every dataset has random access so each host reads only its slice,
and reads are asynchronous to hide object-storage latency.

Episode data inverts the cost model. A video clip costs real CPU to decode, and one prefetch thread
will not keep a GPU fed. A clip is also a time window across several fields, cut inside episode
boundaries, which `TokenSeqDataset`'s stride over one flat field does not express. The store
itself is generic (any pytree, any dtype) but stores frames uncompressed apart from blosc: 256
frames of 128x128 RGB is 12 MB per episode.

Grain keeps the property worth keeping, data order as a pure function of (seed, step), through
its random-access `MapDataset`, and adds what video needs: decoding in worker processes. The
owner's `open-dreamer` already trains on Grain and ArrayRecord. The cost is a second data stack
next to Levanter's for text; moving text onto Grain is left for later (Out of scope).

### Workloads

The owner works on video world models, VLAs, and WAMs, and has not chosen datasets. The format
therefore does not fix how video is stored: compressed video, raw frames, or latents from a frozen
encoder each suit some workload, and each is a field encoding. Three small example datasets test
the design and are not supported datasets:

| Example | Source | Exercises |
|---|---|---|
| Something-Something v2 | `.webm` clips and `labels.json`, already on the data drive | Video without actions, a text label |
| LIBERO, one suite | HDF5 demos from the Hugging Face Hub at a pinned revision | Two cameras, proprioception, actions, a language instruction |
| Minecraft VPT, about 20 episodes | A pinned subset of OpenAI's contractor data (mp4 and jsonl) | Long video with keyboard and mouse actions |

CoinRun was considered and dropped: Procgen publishes no wheels for Python above 3.10, and Poorin
runs 3.12.

## The format

An episode dataset is one artifact directory:

```text
{prefix}/{name}/{version}/
  dataset.json              # the schema
  episodes.parquet          # one row per episode
  shard-00000.array_record  # one msgpack record per episode
  shard-00001.array_record
  ...
```

### `dataset.json`

```json
{
  "step_rate": 20.0,
  "fields": {
    "agentview": {"kind": "video", "dtype": "uint8", "shape": [128, 128, 3],
                  "encoding": {"name": "video", "codec": "h264", "keyframe_interval": 16, "crf": 18}},
    "joint_state": {"kind": "array", "dtype": "float32", "shape": [7],
                    "encoding": {"name": "raw"}},
    "action": {"kind": "array", "dtype": "float32", "shape": [7],
               "encoding": {"name": "raw"}}
  }
}
```

- `kind` is `video` or `array`. A field's `shape` is per step; an episode of length T holds
  `[T, *shape]`.
- Every field shares the dataset's step clock (`step_rate`, in steps per second), the way LeRobot
  resamples every stream to one rate. A reader resamples a source whose streams run at different
  rates.
- A dataset without actions declares no action field. Code that needs one checks the schema.

### `episodes.parquet`

| Column | Meaning |
|---|---|
| `episode` | Episode id, unique in the dataset |
| `shard`, `record` | Where the episode's record is |
| `length` | Number of steps |
| `metadata` | Per-episode values as JSON: a language instruction, an embodiment, a source path |

The clip index is planned from this table alone, without opening a shard.

### Records

One record is one episode: a msgpack map from field name to that field's encoded bytes. msgpack
replaces `open-dreamer`'s pickle, which executes code on load and is readable only from Python.

### Encodings

| Encoding | Stores | Decoding a clip of steps `[s, s+T)` |
|---|---|---|
| `raw` | The `[length, *shape]` array's bytes, C order | Slice |
| `video` | One H.264 or AV1 stream for the episode, constant rate, a keyframe every `keyframe_interval` steps | Seek to the keyframe at or before `s`, decode at most `keyframe_interval - 1` frames before `s`, then `T` frames |

An encoding is a member of an `Encoding` enum plus one encode and one decode function. A JPEG per
frame or a float16 latent encoding would each be one member and one pair.

## Building

`poorin.experiment.data.episodes(...)` returns an `ArtifactStep[EpisodeDataset]`:

```python
def episodes(
    name: str,
    *,
    sources: Sequence[str],          # source units: paths or globs, resolved against the prefix
    reader: str,                     # import path of the reader, "module.path:function"
    schema: DatasetSchema,           # fields, encodings, step rate
    reader_options: Mapping[str, Any] | None = None,  # passed to the reader; in the fingerprint
    version: str | None = None,
    pin: str | None = None,
    resources: ResourceConfig | None = None,
) -> ArtifactStep[EpisodeDataset]: ...
```

A reader is any function from one source unit to episodes:

```python
def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]: ...

@dataclass(frozen=True)
class Episode:
    id: str
    fields: Mapping[str, np.ndarray]   # field name -> [length, *shape], all the same length
    metadata: Mapping[str, Any]
```

The reader is named by import path, not passed as a function, because the fingerprint hashes the
config as JSON. Changing which reader a dataset uses changes its fingerprint. A change inside a
reader is a code change and follows the existing rule: the record names the commit, and the
version is bumped when a code change affects results.

The run is a Zephyr pipeline shaped like the tokenizer's:

1. List the source units.
2. `flat_map(read)`: the reader yields `Episode`s. The build checks each against the schema: every
   declared field present, the dtype and per-step shape match, all fields the same length.
3. `map(encode)`: each field is encoded as its `encoding` says. Video is always re-encoded, to the
   schema's shape, step rate, and keyframe interval, so every dataset meets the read path's
   assumptions whatever its source.
4. `map_shard(write)`: write one ArrayRecord shard and that shard's index rows.
5. Join the index rows into `episodes.parquet` and write `dataset.json`.

The schema, the sources, the reader's import path, and `reader_options` are the fingerprint.
Changing the resolution or the keyframe interval is a new recipe, and the drift check refuses it
under an existing calendar version.

## Reading

`poorin.episodes.read.clip_dataset(...)` returns a Grain `MapDataset`. A training script composes
it in four layers.

1. **Clip index.** From `episodes.parquet`, list every clip as an (episode, start step) pair, for a
   clip length `T`, a stride between starts, and a frame skip `k` (a clip spans `(T - 1) * k + 1`
   steps). Clips never cross an episode's end. The index is two numpy arrays in memory.
2. **Shuffle and read.** `MapDataset.source(clips).shuffle(seed).map(read_clip)`. `read_clip`
   reads one record from ArrayRecord, which has random access, and decodes only the steps the
   clip needs. It returns `{field: np.ndarray [T, *shape]}` and the episode's metadata. A caller
   may select a subset of fields; unselected fields are not decoded.
3. **Transforms.** Resizing, cropping, normalizing, and tokenizing text are the caller's `.map` or
   `.random_map` (Grain seeds `random_map` per index, so random crops stay deterministic). A
   mixture is Grain's `MapDataset.mix` over datasets first mapped to common fields.
4. **Batching, decoding, and placement.** `.batch(B)`, then `.to_iter_dataset()` and
   `.mp_prefetch(...)`, so decoding runs in worker processes. `poorin.episodes.read.to_mesh` turns
   each batch into `jax.Array`s sharded over the mesh's data axis, which is what grug's train step
   takes.

**Resuming.** Every layer before `.to_iter_dataset()` is a `MapDataset`, so batch N is a pure
function of (seed, N). A run resuming at step N slices the batched dataset from N. No iterator
state is saved in the checkpoint. Epochs are a `repeat` with a new shuffle per epoch; the plan
settles the exact Grain composition and tests it.

**Latents.** A latent dataset is an episode dataset with a `raw` float field. It reads through the
same path and decodes nothing.

## Code layout

A new package, `lib/poorin/src/poorin/episodes/`, of roughly 800 to 1,000 lines:

| Module | Holds |
|---|---|
| `schema.py` | `FieldSpec`, `DatasetSchema` (reads and writes `dataset.json`), `Episode` |
| `encodings.py` | The `Encoding` enum and each encoding's encode and decode functions |
| `build.py` | The Zephyr build pipeline and the `EpisodeDataset` artifact type |
| `read.py` | The clip index, `read_clip`, `clip_dataset`, `to_mesh` |

The package imports nothing from Levanter. `poorin.experiment.data` gains `episodes(...)` next to
`tokenized(...)` and `pretokenized(...)`.

The example readers and handles live in `experiments/datasets/episodes/` (`ssv2.py`,
`libero.py`, `vpt.py`). Each is a reader of tens of lines and a handle over a small subset. They
are examples: the library does not import them, and they are not maintained as supported
datasets.

New dependencies, one commit each: `grain`, `array-record`, `av` (PyAV, which bundles its own
FFmpeg libraries), and `h5py` for the LIBERO example. `msgpack` is already locked.

## Tests

Unit tests run on the CPU against tiny videos generated in the test; nothing is downloaded.

- **Encodings.** `raw` round-trips exactly. For `video`, a clip decoded from the middle of an
  episode equals, frame for frame, the same steps sliced from a full decode, at starts on and off
  keyframes; and decoded frames stay above a PSNR bar against the originals.
- **Build.** A reader defined in the test builds a dataset whose `dataset.json`, index rows, and
  decoded records match its episodes. An episode that breaks the schema fails the build and names
  the episode and field. Changing the resolution changes the fingerprint.
- **Clip index.** No clip crosses an episode's end; the number of clips matches the formula for
  each stride and frame skip; an episode shorter than one clip contributes none.
- **Determinism.** Batches from a dataset sliced at batch N equal batches N onward from an
  uninterrupted iteration; a different seed gives a different order.
- **Datasets without actions and mixtures.** A video-only dataset reads with no action field. Two
  datasets with different fields mix after each is mapped to common fields.
- **Placement.** `to_mesh` returns arrays sharded over the data axis of a one-device CPU mesh.

## Acceptance

Run on the workstation and recorded in the plan's final task and in D110:

1. **Build** the three examples as artifacts: about 500 SSv2 clips, one LIBERO suite, about 20 VPT
   episodes.
2. **Throughput.** Measure clips per second for 16-frame clips at 128 px from the SSv2 example,
   against the number of worker processes. Target: 1,000 clips per second (16,000 frames per
   second), an estimate of what a small video model on the RTX 5090 consumes. A miss is recorded
   with the measured curve; the levers are a shorter keyframe interval, a JPEG-per-frame encoding,
   and a smaller resolution.
3. **End to end.** A smoke loop feeds LIBERO clips through `to_mesh` into a toy JAX step on the GPU
   for 200 steps, then resumes at step 100 and sees the same batches.

## Documents

- `docs/explanations/episodes.md`: the format, building a dataset with a reader, and reading clips
  in training.
- `docs/poorin/architecture.md` and `lib/poorin/AGENTS.md`: the `episodes` package.
- `docs/poorin/decisions.md`: D110.

## D110, as it will be recorded

- **Decision:** Sub-projects 3, 4, and 5 come next, in that order; 1b, 2, 6, 7, and 8 wait.
  Episode data (video, trajectories, latents) gets its own format and reads through Grain and
  ArrayRecord; text stays on Levanter's caches, and Levanter stays the training library. Every
  field of a dataset shares one step clock. How a field is stored is a per-field encoding, not a
  property of the format.
- **Why:** The owner, 2026-10-11, has not chosen datasets and wants a foundation that does not fix
  the storage choice. Levanter's data stack assumes items that cost nothing to decode; Grain keeps
  its deterministic, resumable data order and adds decoding in worker processes.
- **Alternatives:** Extend Levanter's cache and loader with field encodings; adopt LeRobot v3 as
  the on-disk format; a separate rate per field.
- **Cost if wrong:** Two data stacks persist. If Grain's worker processes cannot meet the
  throughput target, the read path changes encoding or moves decoding to the GPU.

## Out of scope

- Moving text onto Grain.
- A separate rate per field.
- Supported, maintained readers for particular datasets; LeRobot import or export.
- CoinRun.
- Training encoders that produce latents (sub-project 4).
- Tuning reads from object storage.
