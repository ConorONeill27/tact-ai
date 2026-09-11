"""Tests for the deep-ensemble tactile world model (tact_ai.model.tactile_world_model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tact_ai.config import BeliefConfig, EnvConfig, ModelConfig, ObjectConfig, TactileConfig
from tact_ai.model.belief_filter import BeliefFilter
from tact_ai.model.features import D_OUT, Transition, build_input
from tact_ai.model.replay import ReplayBuffer
from tact_ai.model.tactile_world_model import TactileWorldModel, make_mlp
from tact_ai.model.uncertainty import bald_estimate, epistemic_std_all, mean_variance_ratio
from tact_ai.simulation.environment import ManipulationEnv

TINY = ModelConfig(
    n_ensemble=3,
    hidden=[32],
    activation="tanh",
    lr=5e-3,
    n_epochs_per_update=6,
    batch_size=256,
    predict_std=True,
)


def _action(i: int, obs: dict, env_cfg: EnvConfig, rng: np.random.Generator) -> int:
    """Mix of random actions and scripted contact angles that reveal h."""
    if i % 3 == 0:
        return int(rng.integers(0, env_cfg.n_actions))
    dx, dy = obs["pose"][0], obs["pose"][1]
    perp = float(np.arctan2(-dx, dy))
    mode = int(rng.integers(0, env_cfg.n_modes))
    big = bool(rng.integers(0, 2))
    ang = perp + float(rng.integers(0, 3)) * np.pi / 2.0
    return env_cfg.encode_action(ang, env_cfg.push_modes[mode], env_cfg.force_from_level(big))


def _make_buffer(n: int, base_seed: int = 0, ep_len: int = 15) -> tuple[ReplayBuffer, ManipulationEnv]:
    """Generate ``n`` transitions (with a live physics belief posterior)."""
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    env = ManipulationEnv(ec, tc, oc)
    filt = BeliefFilter(oc, BeliefConfig(), ec, tc, seed=0)
    buf = ReplayBuffer(capacity=n + 1)
    rng = np.random.default_rng(0)
    ep = 0
    obs = env.reset(seed=base_seed)
    for i in range(n):
        if i % ep_len == 0 and i > 0:
            ep += 1
            obs = env.reset(seed=base_seed + ep)
            filt.reset()
        a = _action(i, obs, ec, rng)
        nxt, _, _, _ = env.step(a)
        blame = filt.belief_vector().astype(np.float32)
        buf.push(Transition(obs, a, blame, nxt, False))
        filt.update_obs(nxt["tactile"], nxt["pose"] - obs["pose"], obs, nxt, a, rng=rng)
        obs = nxt
    return buf, env


def _arrays(buf: ReplayBuffer, env_cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    ts = buf.transitions()
    xs = np.stack([build_input(t.obs, t.action_id, t.belief, env_cfg) for t in ts])
    ys = np.zeros((len(ts), D_OUT), dtype=np.float32)
    for j, t in enumerate(ts):
        ys[j, :12] = t.next_obs["tactile"]
        ys[j, 12:] = np.asarray(t.next_obs["pose"], np.float32) - np.asarray(t.obs["pose"], np.float32)
    return xs, ys


def _mse(pred_mean: np.ndarray, ys: np.ndarray) -> float:
    return float(np.mean((pred_mean - ys) ** 2))


def test_make_mlp_output_dims():
    net = make_mlp(40, [32], 15, activation="tanh", predict_std=False, seed=0)
    x = np.zeros((2, 40), dtype=np.float32)
    out = net(torch.from_numpy(x))
    assert out.shape == (2, 15)
    net2 = make_mlp(40, [32], 15, activation="tanh", predict_std=True, seed=0)
    assert net2(torch.from_numpy(x)).shape == (2, 30)


def test_predict_shapes_cold_start_and_reset():
    buf, env = _make_buffer(120, base_seed=0)
    model = TactileWorldModel(TINY, env.object_cfg.n_candidates, env_cfg=env.env_cfg, seed=0)
    xs, ys = _arrays(buf, env.env_cfg)
    pred = model.predict(xs[:32])
    assert pred["mean"].shape == (32, 15)
    assert pred["member_means"].shape == (3, 32, 15)
    assert pred["member_stds"].shape == (3, 32, 15)
    assert pred["mean"].dtype == np.float32
    assert model.epistemic_std(xs[:32]).shape == (32, 15)
    assert model.pose_delta(xs[:32]).shape == (32, 3)
    assert model.next_tactile(xs[:32]).shape == (32, 12)
    cold = pred["mean"].copy()
    model.record_stats(xs, ys)
    model.reset_state()
    after = model.predict(xs[:32])["mean"]
    assert np.allclose(after, cold, atol=1e-5)


def test_loss_and_epistemic_std_decrease():
    train, env = _make_buffer(400, base_seed=0)
    eval_buf, _ = _make_buffer(80, base_seed=500)
    xe, ye = _arrays(eval_buf, env.env_cfg)
    model = TactileWorldModel(TINY, env.object_cfg.n_candidates, env_cfg=env.env_cfg, seed=0)
    before = _mse(model.predict(xe)["mean"], ye)
    epi_before = float(model.epistemic_std(xe).mean())
    for _ in range(12):
        model.update_from_buffer(train)
    after = _mse(model.predict(xe)["mean"], ye)
    epi_after = float(model.epistemic_std(xe).mean())
    assert after < before * 0.4
    assert epi_after < epi_before
    assert np.all(np.isfinite(model.predict(xe)["mean"]))


def test_same_seed_same_outputs_different_seed_differs():
    buf, env = _make_buffer(200, base_seed=0)
    xs, _ = _arrays(buf, env.env_cfg)
    a = TactileWorldModel(TINY, env.object_cfg.n_candidates, env_cfg=env.env_cfg, seed=7)
    b = TactileWorldModel(TINY, env.object_cfg.n_candidates, env_cfg=env.env_cfg, seed=7)
    c = TactileWorldModel(TINY, env.object_cfg.n_candidates, env_cfg=env.env_cfg, seed=8)
    for m in (a, b, c):
        for _ in range(8):
            m.update_from_buffer(buf)
    pa, pb, pc = a.predict(xs[:32])["mean"], b.predict(xs[:32])["mean"], c.predict(xs[:32])["mean"]
    assert np.allclose(pa, pb, atol=1e-6)
    assert not np.allclose(pa, pc, atol=1e-4)


def test_update_noop_on_empty_buffer():
    model = TactileWorldModel(TINY, 18, seed=0)
    assert model.update_from_buffer(ReplayBuffer(10)) is False
    assert model.update(sampler=None) is False


def test_uncertainty_helpers():
    buf, env = _make_buffer(120, base_seed=1)
    model = TactileWorldModel(TINY, env.object_cfg.n_candidates, env_cfg=env.env_cfg, seed=0)
    xs, _ = _arrays(buf, env.env_cfg)
    assert epistemic_std_all(model, xs).shape == (xs.shape[0], 15)
    bald = bald_estimate(model, xs)
    assert bald.shape == (xs.shape[0],)
    assert np.all(bald >= 0.0)
    ratio = mean_variance_ratio(model.pose_delta(xs), model.epistemic_std(xs)[:, 12:])
    assert ratio.shape == (xs.shape[0],)
    assert np.all(np.isfinite(ratio))