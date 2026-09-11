"""Tests for the physics-based belief filter (tact_ai.model.belief_filter)."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.config import BeliefConfig, EnvConfig, ObjectConfig, TactileConfig
from tact_ai.model.belief_filter import BeliefFilter
from tact_ai.simulation.environment import ManipulationEnv
from tact_ai.simulation.objects import candidate_index

N_STEPS = 12
SPEC = [
    (0, False, 0), (0, False, 1),
    (1, True, 0), (2, True, 1),
    (0, True, 0), (2, False, 1),
    (1, False, 0), (2, True, 1),
    (0, False, 1), (1, True, 1),
    (2, True, 1), (0, True, 0),
]


def _filter(seed: int = 0, floor: float = 0.05) -> BeliefFilter:
    return BeliefFilter(
        ObjectConfig(), BeliefConfig(max_entropy_floor=floor), EnvConfig(), TactileConfig(), seed=seed
    )


def _run(seed: int, descript: str = "lambda", floor: float = 0.05):
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    env = ManipulationEnv(ec, tc, oc)
    obs = env.reset(seed=seed)
    filt = BeliefFilter(oc, BeliefConfig(max_entropy_floor=floor), ec, tc, seed=0)
    true_idx = candidate_index(env.props)
    for step in range(N_STEPS):
        dx, dy = obs["pose"][0], obs["pose"][1]
        perp = float(np.arctan2(-dx, dy))
        mode, big, side = SPEC[step]
        th = perp if side == 0 else perp + np.pi
        a = ec.encode_action(th, ec.push_modes[mode], ec.force_from_level(bool(big)))
        nxt, _, _, _ = env.step(a)
        filt.update_obs(
            nxt["tactile"], nxt["pose"] - obs["pose"], obs, nxt, a,
            rng=np.random.default_rng(1000 + step),
        )
        obs = nxt
    return filt, env, true_idx


def test_uniform_prior_entropy():
    filt = _filter()
    assert filt.n_candidates == 18
    assert filt.entropy() == pytest.approx(np.log(18), rel=1e-9)
    b = filt.belief_vector()
    assert b.shape == (18,)
    assert b.sum() == pytest.approx(1.0)
    assert np.allclose(b, 1.0 / 18.0)


def test_candidate_forward_model_matches_true_pose_delta():
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    env = ManipulationEnv(ec, tc, oc)
    obs = env.reset(seed=3)
    filt = BeliefFilter(oc, BeliefConfig(), ec, tc, seed=0)
    dx, dy = obs["pose"][0], obs["pose"][1]
    perp = float(np.arctan2(-dx, dy))
    a = ec.encode_action(perp, ec.push_modes[0], ec.force_from_level(True))
    nxt, _, _, _ = env.step(a)
    real_delta = np.asarray(nxt["pose"], np.float64) - np.asarray(obs["pose"], np.float64)
    t_pred, d_pred = filt.update_predicted(candidate_index(env.props), a, pose_xy=obs["pose"][:2])
    assert t_pred.shape == (12,)
    assert d_pred.shape == (3,)
    assert np.allclose(d_pred, real_delta, atol=1e-9)


def test_posterior_concentrates_on_true_candidate():
    for seed in range(4):
        filt, env, true_idx = _run(seed)
        p = filt.belief_vector()
        assert int(np.argmax(p)) == true_idx, f"seed {seed}: argmax {np.argmax(p)} != true {true_idx}"
        assert p[true_idx] > 0.5, f"seed {seed}: p_true {p[true_idx]:.3f} not > 0.5"
        assert filt.entropy() < np.log(18)


def test_info_gain_informative_positive_uninformative_zero():
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    env = ManipulationEnv(ec, tc, oc)
    obs = env.reset(seed=0)
    filt = BeliefFilter(oc, BeliefConfig(max_entropy_floor=1.0), ec, tc, seed=0)
    graze = ec.encode_action(0.0, ec.push_modes[2], ec.force_from_level(True))
    ig_uniform = filt.expected_info_gain(graze, obs)
    assert ig_uniform > 0.01
    filt, env2, true_idx = _run(0, floor=1.0)
    ig_after = filt.expected_info_gain(graze, obs)
    assert ig_after == 0.0
    assert ig_uniform > ig_after


def test_reset_restores_uniform():
    filt, _, _ = _run(0)
    assert filt.entropy() < 0.2
    filt.reset()
    assert filt.entropy() == pytest.approx(np.log(18), rel=1e-9)
    assert np.allclose(filt.belief_vector(), 1.0 / 18.0)


def test_degenerate_measurement_is_graceful():
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    env = ManipulationEnv(ec, tc, oc)
    obs = env.reset(seed=1)
    filt = BeliefFilter(oc, BeliefConfig(), ec, tc, seed=0)
    _ = env.step(0)
    garbage_tactile = np.ones(12, dtype=np.float32)
    garbage_delta = np.ones(3, dtype=np.float32)
    filt.update_obs(garbage_tactile, garbage_delta, obs, obs, 0)
    b = filt.belief_vector()
    assert np.all(np.isfinite(filt.log_post))
    assert b.shape == (18,)
    assert b.sum() == pytest.approx(1.0)


def test_update_predicted_reveals_radius_and_mu():
    oc = ObjectConfig()
    ec = EnvConfig()
    filt = _filter()
    lo = oc.candidate_grid.index((0.2, 1.5, 0.4))
    hi = oc.candidate_grid.index((0.8, 1.5, 1.0))
    press = ec.encode_action(0.0, ec.push_modes[0], ec.force_from_level(True))
    t_lo, _ = filt.update_predicted(lo, press, pose_xy=(1.0, 0.0))
    t_hi, _ = filt.update_predicted(hi, press, pose_xy=(1.0, 0.0))
    assert abs(float(t_lo[7]) - float(t_hi[7])) > 0.01
    graze = ec.encode_action(0.0, ec.push_modes[2], ec.force_from_level(True))
    t_lo_g, _ = filt.update_predicted(lo, graze, pose_xy=(1.0, 0.0))
    t_hi_g, _ = filt.update_predicted(hi, graze, pose_xy=(1.0, 0.0))
    assert abs(float(t_lo_g[3]) - float(t_hi_g[3])) > 0.2