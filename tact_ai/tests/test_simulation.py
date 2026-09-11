"""End-to-end tests for the manipulation environment (tact_ai.simulation.environment)."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.config import EnvConfig
from tact_ai.simulation.environment import ManipulationEnv


def _env(env_cfg: EnvConfig | None = None) -> ManipulationEnv:
    return ManipulationEnv(env_cfg=env_cfg)


def test_reset_obs_keys_and_shapes():
    env = _env()
    obs = env.reset(seed=123)
    assert set(obs.keys()) == {"tactile", "pose", "action"}
    assert obs["tactile"].shape == (12,)
    assert obs["tactile"].dtype == np.float32
    assert obs["pose"].shape == (3,)
    assert obs["pose"].dtype == np.float32
    assert obs["action"] == -1
    assert "true_props" not in obs and "tactile_info" not in obs
    assert obs["pose"][2] == pytest.approx(float(np.hypot(obs["pose"][0], obs["pose"][1])))
    start_dist = obs["pose"][2]
    assert env.env_cfg.start_radius_min <= start_dist <= env.env_cfg.start_radius_max
    assert env.props is not None


def test_success_reached_with_goal_close():
    cfg = EnvConfig(start_radius_min=0.6, start_radius_max=0.6, goal_tol=0.3, max_steps=12)
    env = ManipulationEnv(env_cfg=cfg)
    obs = env.reset(seed=7)
    gx, gy = cfg.goal
    done, info = False, {}
    for _ in range(cfg.max_steps):
        d = np.array([gx, gy], dtype=float) - obs["pose"][:2]
        theta_c = float(np.arctan2(-d[1], -d[0]))  # n_in points at the goal
        action = cfg.encode_action(theta_c, 0.0, cfg.force_from_level(True))
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done
    assert info["dist_to_goal"] < cfg.goal_tol
    assert info["n_steps"] <= cfg.max_steps


def test_encode_decode_roundtrip_all_actions():
    env = _env()
    cfg = env.env_cfg
    assert cfg.n_actions == 216
    assert cfg.n_modes == 3 and cfg.n_forces == 2
    for a in range(cfg.n_actions):
        theta_c, phi_deg, force = cfg.decode_action(a)
        b = cfg.encode_action(theta_c, phi_deg, force)
        assert b == a, f"round-trip failed for action {a}: got {b}"
    # spot check the layout of the first action and the last.
    theta_c, phi_deg, force = cfg.decode_action(6)
    assert theta_c == pytest.approx(2 * np.pi / 36)
    assert phi_deg == cfg.push_modes[0]
    assert force == cfg.force_small
    theta_c, phi_deg, force = cfg.decode_action(cfg.n_actions - 1)
    assert phi_deg == cfg.push_modes[-1]
    assert force == cfg.force_large


def test_outward_push_no_motion():
    env = _env()
    env.reset(seed=0)
    before = env.pose
    state = env.simulate_contact(0.7, 95.0, env.env_cfg.force_from_level(True))
    assert not state.in_contact
    assert not env.last_contact.in_contact
    assert env.pose == before
    assert np.allclose(env._tactile, 0.0)


def test_random_episode_finite_and_in_range():
    env = _env()
    obs = env.reset(seed=42)
    rng = np.random.default_rng(0)
    info = {}
    for _ in range(40):
        action = int(rng.integers(0, env.env_cfg.n_actions))
        obs, reward, done, info = env.step(action)
        assert np.all(np.isfinite(obs["tactile"]))
        assert np.all(np.isfinite(obs["pose"]))
        assert np.all(obs["tactile"] >= -1.0 - 1e-6) and np.all(obs["tactile"] <= 1.0 + 1e-6)
        assert obs["pose"][2] >= 0.0
        assert obs["action"] == action
        assert reward == pytest.approx(-info["dist_to_goal"])
        assert "true_props" not in obs
        assert set(info.keys()) >= {"contact", "slip", "measured_mu", "true_props",
                                    "dist_to_goal", "n_steps", "tactile_info"}
        if done:
            break
    assert info["n_steps"] == 40 or info["n_steps"] >= env.env_cfg.max_steps


def test_step_out_of_range_raises():
    env = _env()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(env.env_cfg.n_actions)


def test_true_props_are_candidate_grid_points():
    from tact_ai.simulation.objects import candidate_index
    env = _env()
    for seed in range(5):
        env.reset(seed=seed)
        idx = candidate_index(env.props)
        assert 0 <= idx < env.object_cfg.n_candidates
        assert env.props.mu in env.object_cfg.mu_grid
        assert env.props.mass in env.object_cfg.mass_grid
        assert env.props.radius in env.object_cfg.radius_grid