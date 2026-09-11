"""Model-based Bayesian belief filter over hidden object properties.

The agent observes a disk pushed in a 2D planar environment; hidden properties
``h = (mu, mass, radius)`` are unknown but the discrete candidate grid
(``ObjectConfig.candidate_grid``, length 18) is known *a priori*. This filter
maintains a posterior ``P(h | observations)`` by running the *analytical*
physics forward model for every candidate ``h_c`` and scoring the observed
measurement with a Gaussian likelihood.

Measurement layout (15 dims, matching the ensemble world model):

- ``[0:12]`` the tactile reading after the push.
- ``[12:15]`` the pose delta ``next_obs["pose"] - obs["pose"] = (ddx, ddy,
  ddist)``.

Tactile dims are scored with sigma ``BeliefConfig.obs_noise``; pose-delta dims
use ``BeliefConfig.pose_obs_noise`` (defaults to half of ``obs_noise``) because
pose is a vision channel with roughly half the uncertainty of the tactile one.

The candidate forward model is *deterministic* (noiseless sensor, same
``physics.step_disk`` the environment uses), so ``update_obs`` is fully
reproducible given the pre-push absolute pose. Because the goal is fixed at the
origin, ``obs["pose"][:2] = (dx, dy)`` is the object's absolute position and is
fed directly to the forward model.
"""

from __future__ import annotations

import numpy as np

from tact_ai.config import BeliefConfig, EnvConfig, ObjectConfig, TactileConfig
from tact_ai.model.features import D_TACTILE, D_OUT
from tact_ai.sensing.tactile import TactileSensor

_NEGLIGIBLE_LL = -0.5 * D_OUT * 9.0  # ~ -67: worse than 3 sigma per dim in every dim


class BeliefFilter:
    """Physics-based Bayesian filter over the candidate grid.

    Public API (Phase 3 contract):

    - ``__init__(object_cfg, belief_cfg, env_cfg, tactile_cfg, seed=0)``
    - ``reset()``, ``belief_vector() -> (n_candidates,)``, ``entropy() -> float``
    - ``update_predicted(candidate_idx, action_id, pose_xy, rng=None) ->
      (tactile_pred(12), pose_delta_pred(3))``
    - ``update_obs(tactile_obs, pose_delta_obs, obs, next_obs, action_id,
      rng=None) -> None``
    - ``expected_info_gain(action_id, obs, rng=None) -> float``
    - attribute ``n_candidates``
    """

    def __init__(
        self,
        object_cfg: ObjectConfig,
        belief_cfg: BeliefConfig,
        env_cfg: EnvConfig,
        tactile_cfg: TactileConfig,
        seed: int = 0,
    ) -> None:
        """Store configs, build the noiseless forward sensor, start at the prior."""
        self.object_cfg = object_cfg
        self.belief_cfg = belief_cfg
        self.env_cfg = env_cfg
        self.tactile_cfg = tactile_cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.grid = list(object_cfg.candidate_grid)
        self.n_candidates = len(self.grid)
        pred_sensor_cfg = TactileConfig(
            sigma_noise=0.0, noise_clip=0.0, slip_reliability=1.0
        )
        self._pred_sensor = TactileSensor(pred_sensor_cfg, env_cfg)
        self.log_post = np.zeros(self.n_candidates, dtype=np.float64)
        self.reset()

    # ------------------------------------------------------------------ #
    # Posterior state                                                     #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Restore the uniform log-posterior (``log 1/n_candidates``)."""
        self.log_post = np.full(
            self.n_candidates, -np.log(self.n_candidates), dtype=np.float64
        )

    def _log_prob(self, log_w: np.ndarray | None = None) -> np.ndarray:
        """Normalise ``log_post`` (or ``log_w``) into log-probabilities."""
        log_w = self.log_post if log_w is None else log_w
        m = float(np.max(log_w))
        if not np.isfinite(m):
            return np.full(self.n_candidates, -np.log(self.n_candidates))
        z = log_w - m
        p = np.exp(z)
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            return np.full(self.n_candidates, -np.log(self.n_candidates))
        return z - np.log(s)

    def belief_vector(self) -> np.ndarray:
        """Current posterior probabilities, ``(n_candidates,)`` float64."""
        return np.exp(self._log_prob())

    def entropy(self) -> float:
        """Current posterior entropy in natural units."""
        lp = self._log_prob()
        p = np.exp(lp)
        p = np.clip(p, 1e-300, 1.0)
        return float(-np.sum(p * lp))

    # ------------------------------------------------------------------ #
    # Candidate forward model                                             #
    # ------------------------------------------------------------------ #

    def update_predicted(
        self,
        candidate_idx: int,
        action_id: int,
        pose_xy=(0.0, 0.0),
        rng=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the measurement a candidate object would produce.

        Runs the same analytical physics the environment uses with candidate
        properties ``(mu_c, mass_c, radius_c)`` starting from the *absolute*
        pose ``pose_xy`` (goal is the origin), then reads the noiseless tactile
        sensor. ``rng`` is accepted for API symmetry; the forward model is
        deterministic.

        Returns ``(tactile_pred(12,), pose_delta_pred(3,))`` where
        ``pose_delta_pred = (ddx, ddy, ddist)``.
        """
        t_all, d_all = self._predict_all(int(action_id), np.asarray(pose_xy))
        return t_all[int(candidate_idx)], d_all[int(candidate_idx)]

    # ------------------------------------------------------------------ #
    # Vectorised candidate forward model (the hot path)                   #
    # ------------------------------------------------------------------ #

    def _predict_all(
        self, action_id: int, pose_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the measurement of *every* candidate at once.

        ``action_id`` is a single discrete action; ``pose_xy`` the absolute
        pre-push position ``(2,)``. Returns ``(tactile(C, 12),
        pose_delta(C, 3))`` computed with whole-grid numpy arrays (no python
        loop over candidates). Numerically identical to looping
        :meth:`update_predicted` over the grid, so posterior updates and info
        gains are reproducible regardless of which entry point is used.
        """
        theta_c, phi_deg, force = self.env_cfg.decode_action(int(action_id))
        phi_deg = float(phi_deg)
        cos_phi = float(np.cos(np.deg2rad(phi_deg)))
        motion = cos_phi >= 1e-9 and force > 0.0
        c = self.n_candidates
        if not motion:
            return (
                np.zeros((c, D_TACTILE), dtype=np.float64),
                np.zeros((c, D_OUT - D_TACTILE), dtype=np.float64),
            )
        grid = np.asarray(self.grid, dtype=np.float64)  # (C, 3): mu, mass, radius
        mu = grid[:, 0]
        mass = grid[:, 1]
        radius = grid[:, 2]
        n_in = -np.array([np.cos(theta_c), np.sin(theta_c)], dtype=np.float64)
        t_hat = np.array([-np.sin(theta_c), np.cos(theta_c)], dtype=np.float64)
        phi = np.deg2rad(phi_deg)
        N = float(np.cos(phi) * force)
        T = float(np.sin(phi) * force)
        muN = mu * N
        slip = np.abs(T) > muN
        t_eff = np.where(slip, np.sign(T) * muN, T)
        net_force = N * n_in[None, :] + t_eff[:, None] * t_hat[None, :]
        velocity = net_force / (mass[:, None] * self.env_cfg.damping)
        displacement = velocity * self.env_cfg.dt
        tau = radius * t_eff
        omega = tau / (mass * radius * radius * self.env_cfg.rot_damping)
        d_theta = omega * self.env_cfg.dt
        px, py = float(pose_xy[0]), float(pose_xy[1])
        new_x = px + displacement[:, 0]
        new_y = py + displacement[:, 1]
        ddx = displacement[:, 0]
        ddy = displacement[:, 1]
        ddist = np.hypot(new_x, new_y) - np.hypot(px, py)
        pose_delta = np.stack([ddx, ddy, ddist], axis=1)
        tactile = self._tactile_all(theta_c, radius, N, t_eff, slip, T)
        return tactile, pose_delta

    def _tactile_all(
        self,
        theta_c: float,
        radius: np.ndarray,
        N: float,
        t_eff: np.ndarray,
        slip: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """Vectorised analogue of ``TactileSensor._build_features``.

        Replicates the scalar feature construction for all ``C`` candidates in
        one shot; round-trips through float32 so the result is bit-identical to
        the scalar path used by :meth:`update_predicted`.
        """
        cfg = self.env_cfg
        c = len(radius)
        p_norm = np.full(c, np.clip(N / cfg.max_force_n, 0.0, 1.0))
        s_norm = np.clip(t_eff / cfg.max_force_n, -1.0, 1.0)
        half = cfg.sensor_size / 2.0
        centers = np.linspace(-half, half, cfg.n_patches)
        width = 0.5 * np.maximum(radius, 1e-6)
        gauss = np.exp(-0.5 * (centers[None, :] / width[:, None]) ** 2)
        weights = p_norm[:, None] * gauss
        total = weights.sum(axis=1)
        total_s = np.where(total > 0.0, total, 1.0)
        base_cx = (weights * centers[None, :]).sum(axis=1) / total_s
        centered = weights * ((centers[None, :] - base_cx[:, None]) ** 2)
        rms = np.sqrt(centered.sum(axis=1) / total_s)
        peak = weights.max(axis=1)
        cx = base_cx + 0.25 * cfg.sensor_size * np.tanh(s_norm)
        cy = 0.3 * (radius / cfg.sensor_size) * p_norm
        near_slip = np.full(c, min(1.0, abs(T) / max(N, 1e-6)))
        mu = np.asarray(self.grid, dtype=np.float64)[:, 0]
        mm = np.where(slip, mu, 0.0)
        feat = np.stack(
            [
                p_norm,
                np.abs(s_norm),
                slip.astype(np.float64),
                mm,
                np.clip(cx / half, -1.0, 1.0),
                np.clip(cy / half, 0.0, 1.0),
                np.clip(peak, 0.0, 1.0),
                np.clip(rms / half, 0.0, 1.0),
                np.full(len(radius), np.cos(theta_c)),
                np.full(len(radius), np.sin(theta_c)),
                np.ones(len(radius)),
                np.clip(near_slip, 0.0, 1.0),
            ],
            axis=1,
        )
        return feat.astype(np.float32).astype(np.float64)

    # ------------------------------------------------------------------ #
    # Bayesian update                                                     #
    # ------------------------------------------------------------------ #

    def _likelihood_grid(
        self, measurement: np.ndarray, action_id: int, pose_xy, rng
    ) -> np.ndarray:
        """Per-candidate log-likelihood of ``measurement``, ``(n_candidates,)``."""
        del rng
        sigma = np.full(D_OUT, float(self.belief_cfg.obs_noise))
        sigma[D_TACTILE:] = float(self.belief_cfg.pose_obs_noise)
        t_pred, d_pred = self._predict_all(int(action_id), np.asarray(pose_xy))
        mean = np.concatenate([t_pred, d_pred], axis=1)
        resid = (np.asarray(measurement, dtype=np.float64)[None, :] - mean) / sigma[None, :]
        return -0.5 * np.sum(resid * resid, axis=1)

    def update_obs(
        self,
        tactile_obs: np.ndarray,
        pose_delta_obs: np.ndarray,
        obs: dict,
        next_obs: dict,
        action_id: int,
        rng=None,
    ) -> None:
        """Bayesian update from one real environment step.

        ``obs`` is the observation *before* the push (gives the absolute pose,
        ``obs["pose"][:2]``, because the goal is the origin); ``next_obs`` the
        observation after it; ``tactile_obs`` / ``pose_delta_obs`` are the
        environment's tactile reading and ``next_obs["pose"] - obs["pose"]``
        (the caller computes the delta or passes them as-is).

        Adds a Gaussian log-likelihood per candidate and renormalises. If no
        candidate plausibly explains the measurement (max likelihood below
        ``3 sigma`` in every dim) the posterior is left unchanged.
        """
        rng = self.rng if rng is None else rng
        measurement = np.concatenate(
            [np.asarray(tactile_obs, dtype=np.float64), np.asarray(pose_delta_obs, dtype=np.float64)]
        )
        pose_xy = np.asarray(obs["pose"])[:2].astype(np.float64)
        ll = self._likelihood_grid(measurement, action_id, pose_xy, rng)
        if float(np.max(ll)) < _NEGLIGIBLE_LL:
            return
        self.log_post = self._log_prob(self.log_post + ll)

    # ------------------------------------------------------------------ #
    # Expected information gain                                           #
    # ------------------------------------------------------------------ #

    def expected_info_gain(self, action_id: int, obs: dict, rng=None) -> float:
        """Expected reduction in posterior entropy (nat units) for an action.

        Samples ``BeliefConfig.n_obs_samples`` hypothetical next measurements:
        pick a candidate from the current posterior, generate its predicted
        measurement (tactile + on-policy pose delta) and add Gaussian noise at
        the likelihood sigmas. For each sample the would-be posterior is
        computed and its entropy averaged. Returns 0 when the current entropy is
        below ``BeliefConfig.max_entropy_floor`` (nothing left to learn).
        """
        rng = self.rng if rng is None else rng
        h0 = self.entropy()
        if h0 < self.belief_cfg.max_entropy_floor:
            return 0.0
        k = int(self.belief_cfg.n_obs_samples)
        p = self.belief_vector()
        base_log = self.log_post.copy()
        pose_xy = np.asarray(obs["pose"])[:2].astype(np.float64)
        sigma = np.full(D_OUT, float(self.belief_cfg.obs_noise))
        sigma[D_TACTILE:] = float(self.belief_cfg.pose_obs_noise)
        h_after = 0.0
        for _ in range(k):
            c = int(rng.choice(self.n_candidates, p=p))
            t_pred, d_pred = self.update_predicted(c, action_id, pose_xy=pose_xy, rng=rng)
            mean = np.concatenate([t_pred, d_pred])
            y = mean + rng.normal(0.0, sigma)
            ll = self._likelihood_grid(y, action_id, pose_xy, rng)
            new_log = self._log_prob(base_log + ll)
            p_new = np.exp(new_log)
            p_new = np.clip(p_new, 1e-300, 1.0)
            h_after += float(-np.sum(p_new * new_log))
        return float(max(h0 - h_after / k, 0.0))

    def expected_info_gain_batch(
        self, action_ids: np.ndarray, obs: dict, rng=None
    ) -> np.ndarray:
        """Expected info gain for many actions, ``(len(action_ids),)`` (nat units).

        Vectorised twin of :meth:`expected_info_gain`. For each action the
        candidate forward models for the *whole grid* are computed in one numpy
        pass (via :meth:`_predict_all`); a small python loop over actions is
        kept (54 at ``plan_action_step=4``) but there is no per-candidate loop
        inside the physics. The posterior update for each hypothetical sample
        uses exactly the same likelihood sigma/measurement model as
        :meth:`update_obs`, and the RNG draw sequence (``choice`` then
        ``normal`` per sample, per action, in ``action_ids`` order) matches
        calling :meth:`expected_info_gain` per action, so values agree to
        within floating-point round-off (unit-tested at atol 1e-6).

        Returns an all-zero vector when the current entropy is below
        ``BeliefConfig.max_entropy_floor`` (nothing left to learn).
        """
        rng = self.rng if rng is None else rng
        action_ids = np.asarray(action_ids, dtype=int).reshape(-1)
        h0 = self.entropy()
        out = np.zeros(len(action_ids), dtype=np.float64)
        if h0 < self.belief_cfg.max_entropy_floor:
            return out
        k = int(self.belief_cfg.n_obs_samples)
        p = self.belief_vector()
        base_log = self.log_post.copy()
        pose_xy = np.asarray(obs["pose"])[:2].astype(np.float64)
        sigma = np.full(D_OUT, float(self.belief_cfg.obs_noise))
        sigma[D_TACTILE:] = float(self.belief_cfg.pose_obs_noise)
        for aidx, action_id in enumerate(action_ids):
            t_all, d_all = self._predict_all(int(action_id), pose_xy)
            mean = np.concatenate([t_all, d_all], axis=1)
            h_after = 0.0
            for _ in range(k):
                c = int(rng.choice(self.n_candidates, p=p))
                y = mean[c] + rng.normal(0.0, sigma)
                resid = (y[None, :] - mean) / sigma[None, :]
                ll = -0.5 * np.sum(resid * resid, axis=1)
                new_log = self._log_prob(base_log + ll)
                p_new = np.exp(new_log)
                p_new = np.clip(p_new, 1e-300, 1.0)
                h_after += float(-np.sum(p_new * new_log))
            out[aidx] = float(max(h0 - h_after / k, 0.0))
        return out