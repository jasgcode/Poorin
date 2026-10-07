# Experiments

At the [infrastructure level](../explanations/lazy-artifacts.md), an experiment is a DAG
of typed artifact handles to be materialized. Conceptually, an **experiment** represents a
unit of inquiry with a particular hypothesis or goal. Each such experiment is captured by a
GitHub issue with the `experiments` tag
(e.g., [#72](https://github.com/marin-community/marin/issues/72)).

An experiment might involve testing whether one optimizer is better than another in a
controlled setting, or trying out different tokenizers or data quality filtering schemes.
Regardless, an experiment consists of a DAG of steps.

To promote the reproducibility of experiments, we record all experiments in the
`experiments/` directory. Each launcher there (e.g., `experiments/grug/base/launch.py`)
builds the artifact graph for one experiment.

Running each experiment produces provenance records for every artifact it builds.
