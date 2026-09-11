"""Feature encoding and the transition container for the tactile world model.

Dimension constants (cross-phase contract for the ensemble model):

- ``D_TACTILE = 12`` tactile features, ``D_POSE = 3`` pose features
  (``(dx, dy, dist_to_goal)``).
- ``D_ACTION = 7`` action features
  ``[cos(theta_c), sin(theta_c), mode_onehot(3), force_onehot(2)]``.
- ``D_BELIEF`` = the number of hidden-property candidates (18 for the default
  grid); the model input carries the current belief vector.
- ``D_IN  = 12 + 3 + 7 + 18 = 40`` model input.
- ``D_OUT = 12 tactiles + 3 pose deltas = 15`` model target
  (``[next_tactile, (ddx, ddy, ddist)]``).

Because the goal is fixed at the origin (``EnvConfig.goal == (0, 0)``), the
object's *absolute* position is exactly ``obs["pose"][:2]`` and pose deltas
computed from consecutive observations live in the same world frame used by the
physics engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D_TACTILE = 12
D_POSE = 3
D_ACTION = 7
D_BELIEF = 18
D_IN = D_TACTILE + D_POSE + D_ACTION + D_BELIEF
D_OUT = D_TACTILE + D_POSE


@dataclass
class Transition:
    """One (observation, action, belief) -> next-observation experience.

    ``obs`` / ``next_obs`` are observation dicts exactly as produced by
    :class:`tact_ai.simulation.environment.ManipulationEnv` (keys ``tactile``,
    ``pose``, ``action``). ``belief`` is the ``(n_candidates,)`` posterior
    probability vector the agent held when the transition was collected.
    """

    obs: dict
    action_id: int
    belief: np.ndarray
    next_obs: dict
    done: bool


def encode_action(env_cfg, action_id: int) -> np.ndarray:
    """Encode a discrete action id into the ``(7,)`` action feature vector.

    Layout: ``[cos(theta_c), sin(theta_c), mode_onehot(n_modes),
    force_onehot(n_forces)]``. ``n_modes`` is 3 and ``n_forces`` 2 for the
    default config, so the total length is 7.
    """
    theta_c, phi_deg, force = env_cfg.decode_action(int(action_id))
    n_modes = int(env_cfg.n_modes)
    n_forces = int(env_cfg.n_forces)
    if D_ACTION != 2 + n_modes + n_forces:
        raise ValueError(
            f"action feature dim {D_ACTION} inconsistent with "
            f"n_modes={n_modes}, n_forces={n_forces}"
        )
    v = np.zeros(D_ACTION, dtype=np.float32)
    v[0] = float(np.cos(theta_c))
    v[1] = float(np.sin(theta_c))
    mode_idx = min(range(n_modes), key=lambda i: abs(env_cfg.push_modes[i] - phi_deg))
    v[2 + mode_idx] = 1.0
    force_idx = min(
        range(n_forces),
        key=lambda i: abs(force - env_cfg.force_from_level(bool(env_cfg.force_levels[i]))),
    )
    v[2 + n_modes + force_idx] = 1.0
    return v


def encode_obs(obs: dict, belief: np.ndarray) -> np.ndarray:
    """Encode an observation dict into the ``(15,)`` next-features vector.

    Layout: ``[tactile(12), pose(3)]``. The ``belief`` argument is accepted for
    API symmetry but is *not* part of this encoding (beliefs enter only via
    :func:`build_input`).
    """
    x = np.concatenate([np.asarray(obs["tactile"]), np.asarray(obs["pose"])])
    return np.asarray(x, dtype=np.float32)


def build_input(obs: dict, action_id: int, belief: np.ndarray, env_cfg) -> np.ndarray:
    """Build the ``(40,)`` model input for one transition.

    Layout: ``[tactile(12), pose(3), action(7), belief(n_candidates)]``.
    """
    x = np.concatenate(
        [
            np.asarray(obs["tactile"]),
            np.asarray(obs["pose"]),
            encode_action(env_cfg, action_id),
            np.asarray(belief),
        ]
    )
    return np.asarray(x, dtype=np.float32)