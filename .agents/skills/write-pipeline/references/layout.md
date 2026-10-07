# Placement and current examples

Use these examples for the decision they illustrate, not as templates to copy whole.

| Need | Location | Current example |
| --- | --- | --- |
| Reusable tokenization behavior and its result type | `lib/poorin/src/poorin/processing/tokenize/` | `tokenize.py` provides the work function and `TokenizedCache`. |
| Dataset identity, source pins, and selection | `experiments/datasets/` | `nemotron.py` builds keyed handles; mixture weights belong to the consuming experiment. |
| Experiment-specific DAG and tests | Beside the experiment | None today. The nearest is `experiments/grug/base/launch.py`, which builds one training artifact; `tests/test_grug_variant_contracts.py` in the root `tests/` checks it with every other grug variant. |
| Custom analysis and training stages | `experiments/` for binding; `lib/` for reusable work | None today. |

When a second graph depends on data the first discovers, run it as a second pass; do not make every pipeline two-pass. Keep workflow decisions in `experiments/**`, even when they contain concrete work tied to one experiment; promote that work only when it has a real reuse case.

The lazy-artifact migration in [#6649](https://github.com/marin-community/marin/pull/6649) replaced import-time `ExecutorStep` globals with handles built at run time. [Dataset catalog work](https://github.com/marin-community/marin/issues/6777#issuecomment-4848352552) established `experiments/datasets/` as the home for dataset definitions. Current `lib/poorin/src/poorin/experiment/{data,train}.py` and older `StepSpec` factories in `lib/poorin/src/poorin/datakit/` still build workflow objects. Treat them as existing exceptions to migrate when touching their API, not as placement guidance for new code.
