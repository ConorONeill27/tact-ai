"""2D planar pushing environment: a disk pushed to a goal via discrete actions.

The agent controls a virtual fingertip: it picks a boundary contact angle
(``36`` angles, 10 deg steps), a push mode (press / push-normal / grazing, i.e.
force angle ``phi`` relative to the inward normal), and a force level (small /
large). The true object properties ``h = (mu, mass, radius)`` are sampled
uniformly over the candidate grid and are **hidden from the agent**: the only
information about them comes from the 12-d tactile reading. Pose (relative to
the goal) is available as a vision-like channel.

State::
    obs = {"tactile": np.float32 (12,), "pose": np.float32 (3,) = (dx, dy,
           dist_to_goal), "action": int}

``step()`` returns ``(obs, reward, done, info)`` with ``reward = -dist_to_goal``
and ``done`` on goal achievement (within ``goal_tol``) or the step budget.
``info`` carries contact/slip diagnostics, the true hidden properties (harness
use only -- never emitted in observations) and the raw tactile scalars.
"""

from __future__ import annotations

import numpy as np

from tact_ai.config import EnvConfig, TactileConfig, ObjectConfig
from tact_ai.sensing.tactile import TactileSensor, TACTILE_DIM
from tact_ai.simulation import physics
from tact_ai.simulation.objects import sample_object, ObjectProps


class ManipulationEnv:
    """Discrete-action planar push environment (quasi-static analytical engine)."""

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        tactile_cfg: TactileConfig | None = None,
        object_cfg: ObjectConfig | None = None,
    ) -> None:
        """Construct the environment; call :meth:`reset` before stepping."""
        self.env_cfg = env_cfg if env_cfg is not None else EnvConfig()
        self.tactile_cfg = tactile_cfg if tactile_cfg is not None else TactileConfig()
        self.object_cfg = object_cfg if object_cfg is not None else ObjectConfig()
        self.sensor = TactileSensor(self.tactile_cfg, self.env_cfg)
        self.seed: int | None = None
        self._reset_internal()

    # ------------------------------------------------------------------ #
    # Episode API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> dict:
        """Start a new episode; returns the initial observation.

        A seed reserves the full episode: the object's hidden properties, its
        start pose (random radius in ``[start_radius_min, start_radius_max]``
        at a random angle around the goal) and the tactile noise stream are all
        derived from it.
        """
        self.seed = seed
        self._env_rng = np.random.default_rng(seed)
        self.sensor.set_seed(seed)
        self._reset_internal()
        self.props = sample_object(self.object_cfg, self._env_rng)

        gx, gy = self.env_cfg.goal
        radius = float(
            self._env_rng.uniform(self.env_cfg.start_radius_min, self.env_cfg.start_radius_max)
        )
        angle = float(self._env_rng.uniform(0.0, 2.0 * np.pi))
        self.pose = physics.Pose(gx + radius * np.cos(angle), gy + radius * np.sin(angle), 0.0)
        return self._make_obs()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """Apply a discrete action and integrate one physics step."""
        if not (0 <= int(action) < self.env_cfg.n_actions):
            raise ValueError(f"action {action} outside [0, {self.env_cfg.n_actions})")
        theta_c, phi_deg, force = self.env_cfg.decode_action(int(action))
        self.simulate_contact(theta_c, phi_deg, force)
        self.last_action = int(action)
        self.n_steps += 1

        dist = self._dist_to_goal()
        reward = -dist
        done = bool(dist < self.env_cfg.goal_tol or self.n_steps >= self.env_cfg.max_steps)
        return self._make_obs(), reward, done, self._make_info(dist)

    def simulate_contact(self, theta_c_rad: float, phi_deg: float, force: float) -> physics.ContactState:
        """Apply one quasi-static push at an arbitrary contact configuration.

        Public so later phases can roll out candidate pushes through identical
        physics (analytical belief filtering) and so tests can exercise the
        no-contact branch directly. ``cos(phi_deg) <= 0`` (outward push) yields
        ``in_contact=False`` and no motion. Updates ``self.pose`` and the
        tactile reading but **does not** advance episode bookkeeping
        (``n_steps`` / ``last_action`` are handled by :meth:`step`).
        """
        state, new_pose = physics.step_disk(
            self.pose,
            r=self.props.radius,
            m=self.props.mass,
            mu=self.props.mu,
            theta_c=theta_c_rad,
            phi_deg=phi_deg,
            F=force,
            dt=self.env_cfg.dt,
            damping=self.env_cfg.damping,
            rot_damping=self.env_cfg.rot_damping,
        )
        self.pose = new_pose
        self.last_contact = state
        self._tactile = self.sensor.read(state, theta_c_rad, self.props.radius)
        self.tactile_info = self.sensor.tactile_info
        return state

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _reset_internal(self) -> None:
        self.props: ObjectProps | None = None
        self.pose = physics.Pose(0.0, 0.0, 0.0)
        self.n_steps = 0
        self.last_action = -1
        self.last_contact = physics.ContactState(False, 0.0, 0.0, False, None, 0.0)
        self._tactile = np.zeros(TACTILE_DIM, dtype=np.float32)
        self.tactile_info: dict[str, float] = {}

    def _dist_to_goal(self) -> float:
        gx, gy = self.env_cfg.goal
        return float(np.hypot(self.pose.x - gx, self.pose.y - gy))

    def _make_obs(self) -> dict:
        gx, gy = self.env_cfg.goal
        dx = self.pose.x - gx
        dy = self.pose.y - gy
        return {
            "tactile": self._tactile.astype(np.float32).copy(),
            "pose": np.array([dx, dy, np.hypot(dx, dy)], dtype=np.float32),
            "action": int(self.last_action),
        }

    def _make_info(self, dist: float) -> dict:
        cs = self.last_contact
        return {
            "contact": bool(cs.in_contact),
            "slip": bool(cs.slip),
            "measured_mu": float(cs.measured_mu) if cs.measured_mu is not None else 0.0,
            "true_props": dict(self.props.to_dict()) if self.props is not None else {},
            "dist_to_goal": dist,
            "n_steps": self.n_steps,
            "tactile_info": dict(self.tactile_info),
        }