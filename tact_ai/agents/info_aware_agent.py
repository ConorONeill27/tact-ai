"""Information-aware planner — the proposed method.

Every policy decision balances expected task progress against expected
information gain about the hidden object properties ``h = (mu, mass, radius)``.

Scoring formula
---------------
score(a) = predicted_progress(a) + beta * expected_info_gain(a)

- ``predicted_progress(a) = -(predicted ddist(a))`` from the learned ensemble
  mean (same as :class:`GreedyAgent`).
- ``expected_info_gain(a)`` is the expected reduction in posterior entropy over
  the candidate grid, computed with the *analytical* candidate forward models
  (mixture-of-Gaussians estimator, ``BeliefConfig.n_obs_samples`` draws) via the
  vectorised :meth:`belief_filter.BeliefFilter.expected_info_gain_batch`. The
  physics-based estimate costs no world-model evals and does not consume the
  agent's model budget.
- ``beta`` starts at ``AgentConfig.beta`` and, when ``beta_warmup_steps > 0``,
  ramps linearly from 0 up to ``beta`` over the first ``beta_warmup_steps``
  *policy decisions* of the run (mirrors the warm-up: a global, one-time ramp).

Ties are broken deterministically (first index in the considered array).

Model-eval accounting (per policy decision)
-------------------------------------------
``model_evals += len(considered_actions)`` = 54 at ``plan_action_step=4``
(one batched progress ``predict``; the info-gain term uses physics, not the
ensemble).
"""

from __future__ import annotations

import numpy as np

from tact_ai.agents.planner import Agent
from tact_ai.model.features import build_input


class InfoAwareAgent(Agent):
    """Choose actions that are simultaneously goal-ward and informative."""

    def select_action(self, obs: dict) -> int:
        """argmax over considered actions of ``progress + beta * IG``."""
        actions = self.considered_actions()
        belief = self._model_belief_feats()
        x = np.stack([build_input(obs, int(a), belief, self.env_cfg) for a in actions])
        progress = self._predict_progress(x)
        ig = self.belief.expected_info_gain_batch(actions, obs, rng=self.rng)
        score = progress + self._effective_beta() * ig
        best = int(np.argmax(score))  # ties -> first index
        return int(actions[best])

    def _effective_beta(self) -> float:
        """Apply the linear ``beta_warmup_steps`` ramp if configured."""
        warmup = int(self.agent_cfg.beta_warmup_steps)
        if warmup <= 0:
            return self._beta
        frac = min(1.0, self._n_policy_decisions / float(warmup))
        return self._beta * frac