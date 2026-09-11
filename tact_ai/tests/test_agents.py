"""Tests for the four agents (tact_ai.agents) and the belief-filter hot path."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.agents import CEMAgent, GreedyAgent, InfoAwareAgent, RandomAgent
from tact_ai.config import AgentConfig, BeliefConfig, EnvConfig, ModelConfig, ObjectConfig, TactileConfig
from tact_ai.model.belief_filter import BeliefFilter
from tact_ai.model.features import D_OUT, D_TACTILE, Transition, build_input
from tact_ai.simulation.environment import ManipulationEnv

N_CANDIDATES = 18


def _cfgs(
    plan_step: int = 1,
    batch_size: int = 16,
    warmup: int = 0,
    beta: float = 0.0,
    update_every: int = 1,
) -> tuple[ObjectConfig, EnvConfig, TactileConfig, BeliefConfig, ModelConfig, AgentConfig]:
    oc = ObjectConfig()
    ec = EnvConfig(max_steps=12)
    tc = TactileConfig(sigma_noise=0.02)
    bc = BeliefConfig(n_obs_samples=8)
    mc = ModelConfig(
        n_ensemble=2, hidden=[16], n_epochs_per_update=4, batch_size=batch_size,
        buffer_capacity=1024, update_every_steps=update_every,
    )
    ac = AgentConfig(
        warmup_steps=warmup, plan_action_step=plan_step, beta=beta,
        beta_warmup_steps=0, cem_pop=16, cem_elite=6, cem_iters=3,
        cem_rollout_horizon=2, seed=0,
    )
    return oc, ec, tc, bc, mc, ac


def _uniform_belief() -> np.ndarray:
    return np.full(N_CANDIDATES, 1.0 / N_CANDIDATES, dtype=np.float64)


def _ang_diff(a: float, b: float) -> float:
    d = float(abs((float(a) - float(b)) % (2.0 * np.pi)))
    return min(d, float(2.0 * np.pi) - d)


def _run_steps(agent, env, n: int, seed: int = 1):
    obs = env.reset(seed=seed)
    agent.on_episode_start(obs)
    for _ in range(n):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.observe(obs, action, next_obs, reward, done)
        obs = next_obs
        if done:
            break
    return obs


# --------------------------------------------------------------------------- #
# (a) RandomAgent lifecycle                                                    #
# --------------------------------------------------------------------------- #


def test_random_agent_episode_and_belief_updates():
    oc, ec, tc, bc, mc, ac = _cfgs(warmup=0)
    env = ManipulationEnv(ec, tc, oc)
    agent = RandomAgent("random", ac, ec, oc, tc, bc, mc)

    obs = env.reset(seed=5)
    agent.on_episode_start(obs)
    assert agent.belief.entropy() == pytest.approx(np.log(N_CANDIDATES), rel=1e-9)

    entropy_after = None
    for step in range(10):
        action = agent.act(obs)
        assert isinstance(action, int) and 0 <= action < ec.n_actions
        next_obs, reward, done, _ = env.step(action)
        agent.observe(obs, action, next_obs, reward, done)
        obs = next_obs
        entropy_after = agent.belief.entropy()

    assert len(agent.buffer) == 10
    assert agent.model_evals == 0  # random never queries the model
    assert entropy_after is not None and entropy_after < np.log(N_CANDIDATES)


# --------------------------------------------------------------------------- #
# Offline training helper                                                     #
# --------------------------------------------------------------------------- #


def _train_offline(agent, env, env_cfg: EnvConfig, n_poses: int = 6, n_epochs: int = 80) -> None:
    """Fill the agent buffer with scripted single pushes and train the world model.

    Progress labels are analytic: a push at contact angle ``theta_c`` reduces the
    distance to the goal by ``peak * (1 + cos(gap)) / 2`` where ``gap`` is the
    angular separation between ``theta_c`` and the instantaneous goal direction
    (the direction of the pose vector). This gives the model a smooth directional
    target without depending on the (nearly flat) single-push physics landscape:
    the argmin of predicted ``ddist`` is then exactly the goal direction.
    One push is issued per (pose, action) from a fresh reset, so states never chain.
    """
    belief = _uniform_belief()
    peak = 0.3
    for pose_seed in range(n_poses):
        obs = env.reset(seed=pose_seed)
        dx, dy, dist = [float(v) for v in obs["pose"]]
        goal_theta = float(np.arctan2(dy, dx))
        for aid in range(env_cfg.n_actions):
            theta_c, _, _ = env_cfg.decode_action(aid)
            prog = peak * (1.0 + np.cos(_ang_diff(theta_c, goal_theta))) / 2.0
            next_obs = {
                "tactile": np.zeros_like(obs["tactile"]),
                "pose": np.array([dx, dy, dist - prog], dtype=np.float32),
                "action": aid,
            }
            agent.buffer.push(Transition(obs, aid, belief.copy(), next_obs, False))
    agent.model.update_from_buffer(agent.buffer, n_epochs=n_epochs)


# --------------------------------------------------------------------------- #
# (b) Greedy / info-aware first policy action is goal-ward                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [GreedyAgent, InfoAwareAgent], ids=["greedy", "info_aware"])
def test_first_policy_action_is_goal_ward(cls):
    oc, ec, tc = ObjectConfig(), EnvConfig(max_steps=12), TactileConfig(sigma_noise=0.02)
    bc = BeliefConfig(n_obs_samples=8)
    mc = ModelConfig(
        n_ensemble=3, hidden=[128, 128], n_epochs_per_update=4, batch_size=256,
        buffer_capacity=1 << 16, update_every_steps=1,
    )
    ac = AgentConfig(
        warmup_steps=0, plan_action_step=1, beta=0.0, beta_warmup_steps=0,
        cem_pop=16, cem_elite=6, cem_iters=3, cem_rollout_horizon=2, seed=0,
    )
    env = ManipulationEnv(ec, tc, oc)
    agent = cls("test", ac, ec, oc, tc, bc, mc)
    _train_offline(agent, env, ec, n_poses=48, n_epochs=300)

    # the trained model must reproduce the directional ddist signal, not a flat mean
    trs = list(agent.buffer.transitions())
    xs = np.stack([build_input(t.obs, t.action_id, t.belief, ec) for t in trs])
    ys = np.asarray([t.next_obs["pose"][2] - t.obs["pose"][2] for t in trs], dtype=np.float64)
    pred = agent.model.predict(xs)["mean"][:, D_TACTILE + 2]
    r2 = 1.0 - float(np.mean((pred - ys) ** 2)) / float(np.var(ys))
    assert r2 > 0.9, f"world model did not learn directional progress (R2={r2:.3f})"

    for eval_seed in (99, 105, 111):
        obs = env.reset(seed=eval_seed)
        agent.on_episode_start(obs)
        dx, dy = float(obs["pose"][0]), float(obs["pose"][1])
        goal_theta = float(np.arctan2(dy, dx))
        chosen = agent.select_action(obs)
        theta_c, _, _ = ec.decode_action(chosen)
        assert _ang_diff(theta_c, goal_theta) < np.pi / 3, (
            f"chosen theta_c={theta_c:.3f} vs goal direction {goal_theta:.3f} "
            f"(seed {eval_seed}, action {chosen}, diff={_ang_diff(theta_c, goal_theta):.3f})"
        )


# --------------------------------------------------------------------------- #
# (c) CEM accepts actions and counts evals                                    #
# --------------------------------------------------------------------------- #


def test_cem_returns_valid_action_and_counts_evals():
    oc, ec, tc, bc, mc, ac = _cfgs(plan_step=1, batch_size=64)
    env = ManipulationEnv(ec, tc, oc)
    agent = CEMAgent("cem", ac, ec, oc, tc, bc, mc)
    _train_offline(agent, env, ec, n_poses=3, n_epochs=20)

    obs = env.reset(seed=7)
    agent.on_episode_start(obs)
    evals_before = agent.model_evals
    action = agent.select_action(obs)
    assert isinstance(action, int) and 0 <= action < ec.n_actions
    expected_evals = ac.cem_pop * ac.cem_rollout_horizon * ac.cem_iters
    assert agent.model_evals - evals_before == expected_evals


def test_cem_works_with_cold_model():
    oc, ec, tc, bc, mc, ac = _cfgs(plan_step=4)
    env = ManipulationEnv(ec, tc, oc)
    agent = CEMAgent("cem", ac, ec, oc, tc, bc, mc)
    obs = env.reset(seed=3)
    agent.on_episode_start(obs)
    action = agent.select_action(obs)  # must not crash with an untrained model
    assert 0 <= action < ec.n_actions
    assert agent.model_evals == ac.cem_pop * ac.cem_rollout_horizon * ac.cem_iters


# --------------------------------------------------------------------------- #
# (d) set_model_training(False) blocks updates                                #
# --------------------------------------------------------------------------- #


def test_set_model_training_false_prevents_updates():
    oc, ec, tc, bc, mc, ac = _cfgs(update_every=1)
    env = ManipulationEnv(ec, tc, oc)
    agent = GreedyAgent("greedy", ac, ec, oc, tc, bc, mc)

    _run_steps(agent, env, 6, seed=1)  # model now has data and has been updated

    first = list(agent.buffer.transitions())[:6]
    xs = np.stack([build_input(t.obs, t.action_id, t.belief, ec) for t in first])
    ys = np.zeros((len(first), D_OUT), dtype=np.float32)
    for i, t in enumerate(first):
        ys[i, :D_TACTILE] = t.next_obs["tactile"]
        ys[i, D_TACTILE:] = np.asarray(t.next_obs["pose"]) - np.asarray(t.obs["pose"])

    def loss_on_snapshot():
        pred = agent.model.predict(xs)["mean"]
        return float(np.mean((pred - ys) ** 2))

    loss_before = loss_on_snapshot()
    agent.set_model_training(False)
    _run_steps(agent, env, 6, seed=2)
    loss_after = loss_on_snapshot()
    assert loss_after == loss_before  # no weight update happened


def test_considered_actions_layout():
    oc, ec, tc, bc, mc, ac = _cfgs(plan_step=4)
    ac4 = agent_copy(ac)
    ac4.plan_action_step = 4
    agent = GreedyAgent("greedy", ac4, ec, oc, tc, bc, mc)
    acts = agent.considered_actions()
    n_per_angle = ec.n_modes * ec.n_forces
    assert len(acts) == 9 * n_per_angle
    blocks = np.asarray(acts).reshape(-1, n_per_angle)  # contiguous within each angle block
    assert np.all(np.diff(blocks, axis=1) == 1)
    angles = blocks[:, 0] // n_per_angle
    assert np.array_equal(angles, np.arange(0, 36, 4))


# --------------------------------------------------------------------------- #
# Hot path: batch info gain == per-action info gain                           #
# --------------------------------------------------------------------------- #


def test_expected_info_gain_batch_matches_per_action():
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    bc = BeliefConfig(n_obs_samples=12)
    env = ManipulationEnv(ec, tc, oc)
    obs = env.reset(seed=3)
    filt = BeliefFilter(oc, bc, ec, tc, seed=0)
    actions = np.asarray([0, 1, 6, 30, 60, 120, 180, 215], dtype=int)

    rng_per = np.random.RandomState(123)
    per = np.array(
        [filt.expected_info_gain(int(a), obs, rng=rng_per) for a in actions]
    )
    rng_batch = np.random.RandomState(123)
    batch = filt.expected_info_gain_batch(actions, obs, rng=rng_batch)
    assert batch.shape == (len(actions),)
    assert np.allclose(batch, per, atol=1e-6)


def test_expected_info_gain_batch_zero_when_concentrated():
    oc, ec, tc = ObjectConfig(), EnvConfig(), TactileConfig()
    bc = BeliefConfig(n_obs_samples=12, max_entropy_floor=1.0)
    env = ManipulationEnv(ec, tc, oc)
    obs = env.reset(seed=0)
    filt = BeliefFilter(oc, bc, ec, tc, seed=0)
    # spike the posterior directly: entropy ~ 0 -> the estimator short-circuits
    filt.log_post = np.full(18, -50.0)
    filt.log_post[7] = 0.0
    assert filt.entropy() < 1.0
    actions = np.asarray([0, 7, 100, 210], dtype=int)
    assert np.all(filt.expected_info_gain_batch(actions, obs) == 0.0)


def agent_copy(agent_cfg: AgentConfig) -> AgentConfig:
    """Shallow copy of an AgentConfig for surgical edits in tests."""
    from dataclasses import replace

    return replace(agent_cfg)