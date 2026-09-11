"""Uniform-random baseline agent.

Samples actions uniformly from the full action space. It still observes every
transition and updates its belief (so belief-entropy trajectories are
comparable across methods), but never calls the world model.
"""

from __future__ import annotations

from tact_ai.agents.planner import Agent


class RandomAgent(Agent):
    """Baseline: uniform random actions over ``env_cfg.n_actions``."""

    def select_action(self, obs: dict) -> int:
        """Return a uniformly random action id."""
        del obs
        return int(self.rng.randint(0, self.env_cfg.n_actions))