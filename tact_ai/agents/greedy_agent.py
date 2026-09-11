"""Greedy (reward-only) planner baseline.

Scoring formula
---------------
score(a) = predicted_progress(a) = -(predicted ddist(a))

where ``ddist`` is the third pose-delta component predicted by the ensemble
mean. All considered actions are scored in ONE batched ``predict`` call; ties
are broken deterministically by the first (lowest) index in the considered
array.

Model-eval accounting (per policy decision)
-------------------------------------------
``model_evals += len(considered_actions)`` = 54 at ``plan_action_step=4``.
"""

from __future__ import annotations

import numpy as np

from tact_ai.agents.planner import Agent
from tact_ai.model.features import build_input


class GreedyAgent(Agent):
    """Picks the considered action with the largest predicted one-step progress."""

    def select_action(self, obs: dict) -> int:
        """Score every considered action with the model and take the argmax."""
        actions = self.considered_actions()
        belief = self._model_belief_feats()
        x = np.stack([build_input(obs, int(a), belief, self.env_cfg) for a in actions])
        progress = self._predict_progress(x)
        best = int(np.argmax(progress))  # ties -> first index
        return int(actions[best])