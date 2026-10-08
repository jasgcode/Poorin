# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IsoFLOP analysis recovers known optima and power laws from synthetic sweeps."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from levanter.optim.config import AdamConfig
from poorin.scaling_laws import (
    CandidateConfig,
    IsoFlopRecord,
    ScalingFit,
    fit_scaling_laws,
    predict_optimal_config,
    round_flops_to_bucket,
)
from poorin.scaling_laws.isoflop_analysis import robust_quad_logx


def _parabola_in_log10(tokens: np.ndarray, optimum: float, curvature: float = 0.5, floor: float = 2.0) -> np.ndarray:
    return curvature * (np.log10(tokens) - math.log10(optimum)) ** 2 + floor


def _vertex(a: float, b: float) -> float:
    return 10 ** (-b / (2 * a))


def test_robust_quad_logx_recovers_the_minimum_of_a_parabola_in_log_tokens():
    tokens = np.logspace(8, 10, 9)

    a, b, _ = robust_quad_logx(jnp.asarray(tokens), jnp.asarray(_parabola_in_log10(tokens, 1e9)))

    assert _vertex(a, b) == pytest.approx(1e9, rel=0.02)


def test_robust_quad_logx_is_pulled_less_by_an_outlier_than_least_squares():
    tokens = np.logspace(8, 10, 9)
    losses = _parabola_in_log10(tokens, 1e9)
    losses[1] += 50.0  # a diverged run

    a, b, _ = robust_quad_logx(jnp.asarray(tokens), jnp.asarray(losses))
    lstsq_a, lstsq_b, _ = np.polyfit(np.log10(tokens), losses, 2)

    robust_error = abs(math.log10(_vertex(a, b)) - 9)
    lstsq_error = abs(math.log10(_vertex(lstsq_a, lstsq_b)) - 9)
    assert robust_error < 0.5 * lstsq_error


def test_fit_scaling_laws_recovers_the_power_law_behind_isoflop_sweeps():
    alpha, coefficient = 0.5, 0.2
    records = []
    for flops in (1e18, 1e19, 1e20):
        optimum = coefficient * flops**alpha
        tokens = np.logspace(math.log10(optimum) - 1, math.log10(optimum) + 1, 7)
        for t, loss in zip(tokens, _parabola_in_log10(tokens, optimum), strict=True):
            records.append(IsoFlopRecord(tokens=t, metric=loss, flops=flops, params=flops / (6 * t), label="web"))

    result = fit_scaling_laws(records)

    assert [m.flops for m in result.minima_records] == [1e18, 1e19, 1e20]
    for minimum in result.minima_records:
        assert minimum.optimal_tokens == pytest.approx(coefficient * minimum.flops**alpha, rel=0.05)
    fit = result.scaling_fits["web"]
    assert fit.alpha == pytest.approx(alpha, abs=0.02)
    assert fit.A == pytest.approx(coefficient, rel=0.3)


@dataclass(frozen=True)
class _Model:
    params: int

    def flops_per_token(self, vocab_size: int, seq_len: int) -> float:
        return 6.0 * self.params

    def total_trainable_params(self, vocab_size: int) -> int:
        return self.params


@dataclass
class _FixedCandidates:
    tokens: tuple[float, ...]
    name: str = "fixed"
    vocab_size: int = 1000

    def estimate_memory_bytes(self, candidate: CandidateConfig) -> int:
        return 0

    def candidates_for_budget(self, budget: float, seq_len: int = 4096) -> Iterator[CandidateConfig]:
        for t in self.tokens:
            yield CandidateConfig(
                model_config=_Model(params=int(budget / (6 * t))),
                optimizer_config=AdamConfig(),
                batch_size=1,
                train_steps=int(t // seq_len),
                tokens=t,
                flops_budget=budget,
            )


def test_predict_optimal_config_picks_the_smallest_candidate_at_or_above_the_predicted_tokens():
    heuristic = _FixedCandidates(tokens=(1e9, 2e9, 4e9, 8e9))
    fits = {"web": ScalingFit(alpha=0.5, A=1.0)}  # 2.5e9 tokens at 6.25e18 FLOPs

    chosen = predict_optimal_config(fits, target_flops=6.25e18, label="web", heuristic=heuristic)

    assert chosen is not None
    assert chosen.tokens == 4e9


def test_predict_optimal_config_falls_back_to_the_largest_candidate():
    heuristic = _FixedCandidates(tokens=(1e9, 2e9, 4e9))
    fits = {"web": ScalingFit(alpha=0.5, A=1.0)}  # 1e10 tokens at 1e20 FLOPs, above every candidate

    chosen = predict_optimal_config(fits, target_flops=1e20, label="web", heuristic=heuristic)

    assert chosen is not None
    assert chosen.tokens == 4e9


def test_round_flops_to_bucket_groups_budgets_within_half_a_bucket():
    center = 1.1**400

    assert round_flops_to_bucket(center * 1.02) == pytest.approx(center)
    assert round_flops_to_bucket(center / 1.02) == pytest.approx(center)
    assert round_flops_to_bucket(center * 1.08) == pytest.approx(center * 1.1)
