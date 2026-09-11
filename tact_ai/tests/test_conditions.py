"""Tests for the eight experiment conditions and the generalization split."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.config import EnvConfig, ObjectConfig
from tact_ai.experiments.conditions import (
    ALL_CONDITIONS,
    HeldOutSampler,
    RegionSplit,
    build_condition,
    scope_env_cfg,
    scope_object_cfg,
)

EXPECTED_CONDITIONS = [
    "baseline",
    "information",
    "robustness",
    "generalization",
    "compute",
    "property_scope",
    "beta_sweep",
    "cem_budget",
]


def test_all_conditions_registered():
    assert tuple(ALL_CONDITIONS) == tuple(EXPECTED_CONDITIONS)


def test_build_condition_returns_matching_cfg():
    for name in EXPECTED_CONDITIONS:
        cfg = build_condition(name)
        assert cfg.condition == name
        assert cfg.methods
        assert cfg.seeds
        assert cfg.n_trials_per_seed >= 1


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        build_condition("does_not_exist")


def test_baseline_grid_composition():
    cfg = build_condition("baseline")
    assert set(cfg.methods) == {"random", "greedy", "cem", "info_aware"}
    assert cfg.seeds == [0, 1, 2]
    assert cfg.n_trials_per_seed == 8


def test_held_out_sampler_split():
    oc = ObjectConfig()
    expected = {tuple(float(v) for v in g) for g in oc.candidate_grid}
    s = HeldOutSampler(oc, 0.3, seed=0)
    train = set(s.train_candidates())
    held = set(s.held_out_candidates())
    assert train.isdisjoint(held)
    assert train | held == expected
    n_hold = int(np.round(0.3 * len(oc.candidate_grid)))
    n_hold = max(1, min(len(oc.candidate_grid) - 1, n_hold))
    assert len(held) == n_hold


def test_held_out_sampler_deterministic_and_seeded():
    oc = ObjectConfig()
    a = HeldOutSampler(oc, 0.3, seed=0)
    b = HeldOutSampler(oc, 0.3, seed=0)
    c = HeldOutSampler(oc, 0.3, seed=7)
    assert a.held_out_candidates() == b.held_out_candidates()
    assert a.train_candidates() == b.train_candidates()
    assert a.held_out_candidates() != c.held_out_candidates()


def test_scope_object_cfg_grids():
    narrow = scope_object_cfg("narrow")
    assert narrow.mu_grid == [0.45, 0.5, 0.55]
    wide = scope_object_cfg("wide")
    assert wide.radius_grid == [0.3, 1.0, 1.8]
    with pytest.raises(KeyError):
        scope_object_cfg("huge")


def test_scope_env_cfg_preserves_force_scale():
    base = EnvConfig()
    for scope in ["narrow", "medium", "wide"]:
        env = scope_env_cfg(scope)
        assert env.force_small == pytest.approx(base.force_small)
        assert env.force_large == pytest.approx(base.force_large)
        assert env.damping == pytest.approx(base.damping)


def test_region_split_partitions_by_radius():
    oc = ObjectConfig()
    expected = {tuple(float(v) for v in g) for g in oc.candidate_grid}
    s = RegionSplit(oc)
    train = set(s.train_candidates())
    held = set(s.held_out_candidates())
    assert train.isdisjoint(held)
    assert train | held == expected
    assert len(train) == len(held) == oc.n_candidates // 2
    assert all(abs(t[2] - min(oc.radius_grid)) < 1e-9 for t in train)
    assert all(abs(t[2] - min(oc.radius_grid)) >= 1e-9 for t in held)


def test_new_condition_grids():
    beta = build_condition("beta_sweep")
    assert beta.beta_grid == [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
    assert beta.methods == ["info_aware"]
    cem = build_condition("cem_budget")
    assert cem.cem_budget_grid and cem.methods == ["cem"]
    assert build_condition("generalization").region_split is True