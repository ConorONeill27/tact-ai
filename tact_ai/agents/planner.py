"""Abstract agent base shared by every planner/baseline.

Lifecycle (Phase 3 contract)
----------------------------
- One agent instance persists across trials (episodes) within one
  ``(method, seed)``: it owns ONE :class:`TactileWorldModel`, ONE
  :class:`ReplayBuffer` and ONE :class:`BeliefFilter`.
- The BELIEF is reset to uniform at the start of every episode
  (:meth:`on_episode_start`); the MODEL and BUFFER persist, so the robot learns
  across objects.
- A per-agent **warm-up countdown** (``AgentConfig.warmup_steps``) is consumed
  at the very start of the run, once, before the first policy decision. It is
  never reset between episodes; while positive the agent must act randomly.
- After every observed transition the transition is pushed to the buffer and the
  belief is updated; every ``ModelConfig.update_every_steps`` env-steps the world
  model is retrained from the buffer (respecting :meth:`set_model_training`).

Model-eval accounting
---------------------
``model_evals`` counts raw forward passes of the ensemble (batch size times
calls). Model scoring must batch all considered actions into a single
``predict`` call of shape ``(n_considered, 40)``; a per-decision accounting
summary is documented in each agent module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tact_ai.config import AgentConfig, BeliefConfig, EnvConfig, ModelConfig, ObjectConfig, TactileConfig
from tact_ai.model.belief_filter import BeliefFilter
from tact_ai.model.features import D_TACTILE, Transition, build_input
from tact_ai.model.replay import ReplayBuffer
from tact_ai.model.tactile_world_model import TactileWorldModel

DDIST_COL = D_TACTILE + 2  # pose-delta ddist column in the model output


class Agent(ABC):
    """Base class: owns the model, buffer, belief filter and warm-up counter."""

    def __init__(
        self,
        name: str,
        agent_cfg: AgentConfig,
        env_cfg: EnvConfig,
        object_cfg: ObjectConfig,
        tact_cfg: TactileConfig,
        belief_cfg: BeliefConfig,
        model_cfg: ModelConfig,
    ) -> None:
        """Build the persistent model/buffer/belief and the agent RNG.

        ``name`` is the registry key (random|greedy|cem|info_aware). The agent
        RNG is a deterministic ``numpy.RandomState`` seeded from
        ``agent_cfg.seed``; it is used for random actions *and* passed into the
        belief filter so all stochastic draws are reentrant.
        """
        self.name = name
        self.agent_cfg = agent_cfg
        self.env_cfg = env_cfg
        self.object_cfg = object_cfg
        self.tact_cfg = tact_cfg
        self.belief_cfg = belief_cfg
        self.model_cfg = model_cfg
        self.rng = np.random.RandomState(int(agent_cfg.seed))
        self.buffer = ReplayBuffer(model_cfg.buffer_capacity)
        self._model = TactileWorldModel(
            model_cfg, object_cfg.n_candidates, env_cfg, seed=int(agent_cfg.seed)
        )
        self.belief = BeliefFilter(
            object_cfg, belief_cfg, env_cfg, tact_cfg, seed=int(agent_cfg.seed)
        )
        self.model_training = True
        self.model_evals = 0
        self._n_env_steps = 0
        self._n_policy_decisions = 0
        self.warmup_remaining = int(agent_cfg.warmup_steps)
        self._beta = float(agent_cfg.beta)

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> TactileWorldModel:
        """The persistent deep-ensemble world model."""
        return self._model

    def set_model_training(self, flag: bool) -> None:
        """Enable/disable world-model updates (generalization eval turns off)."""
        self.model_training = bool(flag)

    def set_beta(self, beta: float) -> None:
        """Override the info-gain weight used by :class:`InfoAwareAgent`."""
        self._beta = float(beta)

    def _model_belief_feats(self) -> np.ndarray:
        """Belief-invariant feature block fed to the learned world model.

        Phase-4 finding: conditioning the deep-ensemble on the (concentrating)
        posterior produced near action-invariant predictions at decision time
        (a belief out-of-distribution collapse; the ensemble generalised badly
        to concentrated posteriors). Every model-based planner therefore feeds a
        fixed uniform block to the world model, keeping the learned dynamics
        belief-agnostic, while the physics belief filter still drives
        ``info_aware``'s information gain and the recorded belief metrics.
        Applies identically to all planners, so method comparisons stay fair.
        """
        n = int(self.object_cfg.n_candidates)
        return np.full(n, 1.0 / n, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def on_episode_start(self, obs: dict) -> None:
        """Called once per trial before the first action; resets the belief."""
        del obs
        self.belief.reset()

    # ------------------------------------------------------------------ #
    # Act / observe loop                                                  #
    # ------------------------------------------------------------------ #

    def act(self, obs: dict) -> int:
        """Pick an action id.

        While ``warmup_remaining > 0`` (a *global* one-time countdown) the agent
        acts randomly. Once the model has enough data (``len(buffer) >=
        model_cfg.batch_size``) the policy takes over, otherwise a random action
        is still used (dead-cold fallback).
        """
        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
            return int(self.rng.randint(0, self.env_cfg.n_actions))
        if len(self.buffer) < self.model_cfg.batch_size:
            return int(self.rng.randint(0, self.env_cfg.n_actions))
        self._n_policy_decisions += 1
        return self.select_action(obs)

    @abstractmethod
    def select_action(self, obs: dict) -> int:
        """Return a discrete action id for the current observation."""

    def observe(
        self, obs: dict, action_id: int, next_obs: dict, reward: float, done: bool
    ) -> None:
        """Store the transition, update the belief and schedule model updates.

        The transition's belief block is the belief-invariant block
        (:meth:`_model_belief_feats`), matching what every planner feeds the
        world model at decision time so training and evaluation agree; the true
        posterior is still maintained for the information-gain term and the
        recorded belief metrics.
        """
        del reward
        belief_before = self._model_belief_feats()
        self.buffer.push(Transition(obs, int(action_id), belief_before, next_obs, bool(done)))
        pose_delta = np.asarray(next_obs["pose"], dtype=np.float64) - np.asarray(
            obs["pose"], dtype=np.float64
        )
        self.belief.update_obs(
            next_obs["tactile"], pose_delta, obs, next_obs, int(action_id), rng=self.rng
        )
        self._n_env_steps += 1
        if self.model_training and self._n_env_steps % self.model_cfg.update_every_steps == 0:
            self.model.update_from_buffer(self.buffer)

    # ------------------------------------------------------------------ #
    # Shared scoring helpers                                              #
    # ------------------------------------------------------------------ #

    def considered_actions(self) -> np.ndarray:
        """Decimated action subset used by all model-based planners.

        Selects contact-angle indices ``0, plan_action_step, 2*step, ...``
        (default step 4 -> 9 angles) and keeps every push-mode/force
        combination for those angles, so ``9 * n_modes * n_forces = 54`` actions
        at defaults. ``RandomAgent`` ignores this subset.
        """
        step = int(max(1, self.agent_cfg.plan_action_step))
        n_per_angle = self.env_cfg.n_modes * self.env_cfg.n_forces
        out: list[int] = []
        for angle_idx in range(0, self.env_cfg.n_contact_angles, step):
            base = int(angle_idx) * n_per_angle
            out.extend(range(base, base + n_per_angle))
        return np.asarray(out, dtype=int)

    def _progress_columns(self, mean_raw: np.ndarray) -> np.ndarray:
        """Return ``-(predicted ddist)`` from a raw ``(B, D_OUT)`` prediction."""
        return -np.asarray(mean_raw, dtype=np.float64)[:, DDIST_COL]

    def _predict_progress(self, x_batch: np.ndarray) -> np.ndarray:
        """Ensemble-mean predicted task progress ``-(predicted ddist)``, (B,).

        Counts ``len(x_batch)`` model evals (one batched ensemble forward pass).
        """
        pred = self.model.predict(x_batch)
        self.model_evals += int(len(x_batch))
        return self._progress_columns(pred["mean"])

    def _action_feats(self, action_ids: np.ndarray) -> np.ndarray:
        """Precomputed ``(n_actions, 7)`` action encodings for a decision."""
        from tact_ai.model.features import encode_action

        return np.stack([encode_action(self.env_cfg, int(a)) for a in action_ids]).astype(
            np.float32
        )