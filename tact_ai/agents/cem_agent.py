"""Cross-Entropy Method (CEM) MPC baseline.

CEM samples many fixed-horizon action sequences from a per-position categorical
over the considered action set, rolls every sequence out through the learned
world model in *batches* (state recycling: the features for step ``t+1`` are the
predicted next features of step ``t``; the belief is frozen at the current
posterior), and refits the categoricals toward the elite sequences. Only the
first action of the best sequence found (tracked across iterations) is executed.

Scoring formula
---------------
score(sequence) = sum over the horizon of predicted_progress = -sum(predicted ddist)

Model-eval accounting (per policy decision)
-------------------------------------------
``model_evals += cem_pop * cem_rollout_horizon * cem_iters``
(e.g. 120 * 3 * 5 = 1800 at defaults); there is exactly one ``predict`` call
per (iteration, horizon step), each over the whole population batch.
"""

from __future__ import annotations

import numpy as np

from tact_ai.agents.planner import Agent
from tact_ai.model.features import D_TACTILE


class CEMAgent(Agent):
    """Model-based MPC: CEM over the discrete considered-action distribution."""

    def select_action(self, obs: dict) -> int:
        """Sample CEM populations, roll out, refit, and return the first action."""
        cfg = self.agent_cfg
        actions = self.considered_actions()
        n_actions = len(actions)
        pop = int(cfg.cem_pop)
        each = int(cfg.cem_elite)
        iters = int(cfg.cem_iters)
        horizon = int(cfg.cem_rollout_horizon)
        # When cem_budget is set, scale the population so that
        # pop * iters * horizon ≈ budget (the target model evals per decision).
        budget = float(getattr(cfg, "cem_budget", 0.0))
        if budget > 0:
            from math import ceil
            pop = max(2 * each, int(ceil(budget / max(1, iters * horizon))))
        belief = self._model_belief_feats()
        n_belief = len(belief)
        action_feats = self._action_feats(actions)  # (n_actions, D_ACTION)

        # Per-position categoricals over the considered-action array (uniform init).
        dist = np.full((horizon, n_actions), 1.0 / n_actions, dtype=np.float64)

        obs_tact = np.asarray(obs["tactile"], dtype=np.float32)
        obs_pose = np.asarray(obs["pose"], dtype=np.float32)
        best_action = int(actions[0])
        best_score = -np.inf

        for _ in range(iters):
            seq = np.empty((pop, horizon), dtype=int)
            for h in range(horizon):
                seq[:, h] = self.rng.choice(n_actions, size=pop, p=dist[h])

            score = np.zeros(pop, dtype=np.float64)
            tact = np.broadcast_to(obs_tact, (pop, D_TACTILE))
            pose3 = np.broadcast_to(obs_pose, (pop, 3))
            for h in range(horizon):
                x = np.concatenate(
                    [
                        tact,
                        pose3,
                        action_feats[seq[:, h]],
                        np.broadcast_to(belief, (pop, n_belief)),
                    ],
                    axis=1,
                )
                pred = self.model.predict(x)
                self.model_evals += pop
                mean = pred["mean"].astype(np.float64)
                score += self._progress_columns(mean)
                if h + 1 < horizon:
                    # State recycling: use predicted next features as the next input.
                    xy = np.asarray(obs_pose[:2], dtype=np.float64) + mean[:, D_TACTILE:D_TACTILE + 2]
                    pose3 = np.stack(
                        [xy[:, 0], xy[:, 1], np.hypot(xy[:, 0], xy[:, 1])], axis=1
                    ).astype(np.float32)
                    tact = mean[:, :D_TACTILE].astype(np.float32)

            elites = np.argsort(score)[::-1][: min(each, pop)]
            for i in elites:
                if score[i] > best_score:
                    best_score = float(score[i])
                    best_action = int(actions[int(seq[i, 0])])

            # Refit per-position categorical means toward the elites (eps avoids collapse).
            eps = 1e-2
            elite_seq = seq[elites]
            for h in range(horizon):
                counts = np.bincount(elite_seq[:, h], minlength=n_actions).astype(np.float64)
                dist[h] = (counts + eps) / (counts.sum() + eps * n_actions)

        return int(best_action)