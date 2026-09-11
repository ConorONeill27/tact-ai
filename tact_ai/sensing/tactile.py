"""Tactile sensor: maps a physics :class:`ContactState` to a 12-d feature vector.

The sensor models a small flat 2D fingertip pad pressed against the disk
surface. Its local frame is: x = tangential direction ``t_hat`` (positive =
counter-clockwise at the boundary), y = inward normal ``n_in`` (positive into
the disk). Normal contact pressure is spread over ``EnvConfig.n_patches``
patches with a width proportional to the disk radius, so the *pressure spread
carries a weak but real signature of the hidden radius*. The measured kinetic
friction (``measured_mu``) carries the hidden friction, and the normal/tangential
forces plus the pose response carry the mass influence.

The centre of pressure (centroid) shifts tangentially with the transmitted
shear force and the pressure peak/spread grow with normal force and radius.
Output is deterministic given the ContactState + ``theta_c`` + radius; Gaussian
noise is then added (sigma = ``TactileConfig.sigma_noise``, clipped to
``TactileConfig.noise_clip`` stds).
"""

from __future__ import annotations

import numpy as np

from tact_ai.config import TactileConfig, EnvConfig
from tact_ai.simulation.physics import ContactState

TACTILE_DIM = 12
"""Length of the tactile feature vector (contract for later phases)."""


class TactileSensor:
    """Maps ContactState -> 12-d tactile feature vector (with noise).

    Feature layout, in order (all float32 after :meth:`add_noise`):

    0. ``normal_force_norm``        N / max_force_n                         [0, 1]
    1. ``tangential_force_norm``    |T_eff| / max_force_n                   [0, 1]
    2. ``slip_flag``                1.0 when slip, else 0.0                  {0, 1}
    3. ``measured_mu``              kinetic friction observed during slip,
                                    else 0.0                                [0, 1]
    4. ``contact_centroid_x_norm``  CoP offset along tangent / (size/2)      [-1, 1]
    5. ``contact_centroid_y_norm``  CoP offset along inward normal / (size/2) [0, 1]
    6. ``peak_pressure_norm``       max pressure over the tactile patches    [0, 1]
    7. ``pressure_spread_norm``     RMS pressure spread along tangent /
                                    (size/2) (grows with disk radius)       [0, 1]
    8. ``contact_angle_cos``        cos(theta_c)                             [-1, 1]
    9. ``contact_angle_sin``        sin(theta_c)                             [-1, 1]
    10. ``in_contact_flag``         1.0 when contact, else 0.0               {0, 1}
    11. ``near_slip_proxy``         min(1, |T_applied|/N): grazing/slip saturation [0, 1]

    When there is no contact every feature is 0. Noise (sigma = config.sigma
    in normalised units, clipped to ``noise_clip`` stds) is added per channel;
    magnitude features are clipped back into [0, 1] and the cosine/sine pair
    into [-1, 1]. With probability ``(1 - slip_reliability)`` a slip event
    reports ``measured_mu = 0``.
    """

    def __init__(self, config: TactileConfig, env_cfg: EnvConfig, seed: int = 0) -> None:
        """Store configs and seed the internal noise RNG (call ``set_seed``
        every episode for full reproducibility)."""
        self.config = config
        self.env_cfg = env_cfg
        self.set_seed(seed)
        self._raw: np.ndarray = np.zeros(TACTILE_DIM, dtype=np.float32)
        self._reading: np.ndarray = np.zeros(TACTILE_DIM, dtype=np.float32)
        self.tactile_info: dict[str, float] = {}

    def set_seed(self, seed: int | None) -> None:
        """Re-seed the internal noise generator (determinism contract)."""
        self._rng = np.random.default_rng(seed)

    # -- public API ----------------------------------------------------------

    def read(self, state: ContactState, theta_c_rad: float = 0.0, radius: float = 0.5) -> np.ndarray:
        """Return the 12-d tactile vector for a contact state (float32).

        ``state`` comes from :func:`tact_ai.simulation.physics.step_disk`;
        ``theta_c_rad`` / ``radius`` are needed to place the pad geometry on
        the disk boundary and to scale the pressure spread.
        """
        if not state.in_contact:
            # No contact: a crisp all-zero reading (no sensor noise while idle).
            raw = np.zeros(TACTILE_DIM, dtype=np.float32)
            self._raw = raw.copy()
            self._reading = raw.copy()
            self.tactile_info = self._make_info(raw)
            return self._reading
        else:
            raw = self._build_features(state, theta_c_rad, radius)
        self._raw = raw.copy()
        self.tactile_info = self._make_info(self._raw)
        self._reading = self.add_noise(raw, self._rng)
        return self._reading

    def __call__(self, state: ContactState, theta_c_rad: float = 0.0, radius: float = 0.5) -> np.ndarray:
        """Alias for :meth:`read`."""
        return self.read(state, theta_c_rad=theta_c_rad, radius=radius)

    def add_noise(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add clipped Gaussian noise to a feature vector (used by ``read``).

        Noise magnitude is ``sigma_noise`` in normalised feature units,
        truncated to ``noise_clip`` standard deviations, then the rounded
        (noiseless) feature bounds are re-applied channel-wise.
        """
        y = np.asarray(x, dtype=np.float32).copy()
        if self.config.sigma_noise > 0.0:
            noise = rng.normal(0.0, self.config.sigma_noise, size=y.shape).astype(np.float32)
            clip = self.config.noise_clip * self.config.sigma_noise
            noise = np.clip(noise, -clip, clip)
            y = y + noise
        y[0:8] = np.clip(y[0:8], 0.0, 1.0)
        y[8:10] = np.clip(y[8:10], -1.0, 1.0)
        y[10:12] = np.clip(y[10:12], 0.0, 1.0)
        return y

    # -- internals ------------------------------------------------------------

    def _build_features(self, state: ContactState, theta_c_rad: float, radius: float) -> np.ndarray:
        cfg = self.env_cfg
        N = state.normal_force
        t_eff = state.tangential_force
        p_norm = float(np.clip(N / cfg.max_force_n, 0.0, 1.0))
        s_norm = float(np.clip(t_eff / cfg.max_force_n, -1.0, 1.0))

        # Pressure profile over the pad patches (width scales with disk radius).
        half = cfg.sensor_size / 2.0
        centers = np.linspace(-half, half, cfg.n_patches)
        width = 0.5 * max(radius, 1e-6)
        gauss = np.exp(-0.5 * (centers / width) ** 2)
        weights = p_norm * gauss
        total = weights.sum()
        if total > 0.0:
            base_cx = float((weights * centers).sum() / total)
            rms = float(np.sqrt((weights * (centers - base_cx) ** 2).sum() / total))
            peak = float(weights.max())
        else:
            base_cx, rms, peak = 0.0, 0.0, 0.0

        # CoP: shear shifts it tangentially (sign of transmitted shear).
        cx = base_cx + 0.25 * cfg.sensor_size * np.tanh(s_norm)
        # Pressing harder / against a larger disk drives the CoP inward.
        cy = 0.3 * (radius / cfg.sensor_size) * p_norm

        near_slip = min(1.0, abs(state.applied_tangential) / max(N, 1e-6))
        mm = state.measured_mu
        if mm is not None and self._rng.uniform() > self.config.slip_reliability:
            mm = 0.0

        return np.asarray(
            [
                p_norm,                                            # 0 normal_force
                abs(s_norm),                                       # 1 tangential
                1.0 if state.slip else 0.0,                        # 2 slip
                float(mm) if mm is not None else 0.0,              # 3 measured_mu
                float(np.clip(cx / half, -1.0, 1.0)),              # 4 centroid x
                float(np.clip(cy / half, 0.0, 1.0)),               # 5 centroid y
                float(np.clip(peak, 0.0, 1.0)),                    # 6 peak pressure
                float(np.clip(rms / half, 0.0, 1.0)),              # 7 spread
                float(np.cos(theta_c_rad)),                        # 8 angle cos
                float(np.sin(theta_c_rad)),                        # 9 angle sin
                1.0,                                               # 10 in contact
                float(np.clip(near_slip, 0.0, 1.0)),               # 11 near-slip
            ],
            dtype=np.float32,
        )

    def _make_info(self, raw: np.ndarray) -> dict[str, float]:
        """Raw scalar summary used for analysis/debugging (pre-noise)."""
        return {
            "normal_force": float(raw[0]),
            "tangential_force": float(raw[1]),
            "slip": float(raw[2]),
            "measured_mu": float(raw[3]),
            "centroid_x": float(raw[4]),
            "centroid_y": float(raw[5]),
            "peak_pressure": float(raw[6]),
            "spread": float(raw[7]),
            "contact_angle_cos": float(raw[8]),
            "contact_angle_sin": float(raw[9]),
            "in_contact": float(raw[10]),
            "near_slip": float(raw[11]),
        }