# poorin

Poorin's experiment framework: the layer that makes runs reproducible.

- `poorin.execution`: lazy, versioned artifacts (`name@version`), config fingerprints, the step
  runner that caches finished work, and step status records.
- `poorin.experiment`: the launcher CLI (`--version`, `--resume`, `--override`, `--run`), data
  mixtures, and training and sweep helpers.
- `poorin.processing` and `poorin.datakit`: download, normalize, tokenize, deduplicate, and
  decontaminate datasets, run on Zephyr.
- `poorin.training`: run IDs, output paths, and dispatch of Levanter's training mains.
- `poorin.evaluation`: the lm-eval harness integration, evaluation records, and statistics.
- `poorin.scaling_laws`: isoflop analysis, fits, and plots.
- `poorin.profiling`: ingest and summarize xprof traces (the `profile-training` skill).

## Poorin's lighter fork of Marin's core

This library is upstream Marin's `marin-core` package, renamed (D85) because Poorin removed most
of it. Against the `upstream-baseline` tag it has 23,602 lines of source instead of 65,588.

Removed, by subpackage (lines at `upstream-baseline`):

| Subpackage | Lines | What it was | Decision |
|---|---|---|---|
| `datakit` (most of it) | 15,267 | Marin's dataset catalog and crawl sources | P7 sweep, D48 |
| `inference` | 7,939 | vLLM serving and the OpenAI-compatible server | D10, P3 |
| `transform` | 5,889 | HTML, ar5iv, Wikipedia, and StackExchange converters | P7 sweep |
| `evaluation` (part) | 4,976 | Fleet launchers: Harbor, Evalchemy, the eval runner | D10, D69 |
| `processing` (part) | 1,433 | Classifiers and transforms no root used | P7 sweep |
| `testing` | 1,367 | Cluster test support | Cleanup, D75 |
| `rl` | 1,217 | The in-repo SkyRL integration | D6, D68 |
| `markdown`, `mcp`, `publish`, `export`, `web`, `schemas`, `core`, `cluster`, `rollouts` | 2,769 | Smaller leftovers of the above | P3, P7 |

The `MARIN_*` environment variables and helpers such as `poorin_prefix()` keep their names for now.
