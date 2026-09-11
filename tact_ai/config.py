"""Central configuration for Tact AI (SciFest 2026 simulation build).

This module is the cross-module CONTRACT for the whole package. Implementations
must respect existing field names and defaults; new fields may be added where
needed but existing ones must not be renamed or removed.

All experiments must stay bounded so the full pipeline runs in well under ~30
minutes on a CPU-only machine.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Simulation                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ObjectConfig:
    """Distributions from which hidden object properties h are sampled.

    The environment samples h from these ranges. The discrete *candidate grid*
    used for belief inference is built from the grid tensors below; the sampled
    h should coincide with a grid point (clean identifiability for the belief
    update) unless a condition deliberately asks otherwise.
    """

    mu_grid: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    mass_grid: list[float] = field(default_factory=lambda: [0.8, 1.5, 3.0])
    radius_grid: list[float] = field(default_factory=lambda: [0.4, 1.0])

    @property
    def candidate_grid(self) -> list[tuple[float, float, float]]:
        """Cartesian grid of candidates -> [(mu, mass, radius), ...]."""
        out: list[tuple[float, float, float]] = []
        for mu in self.mu_grid:
            for m in self.mass_grid:
                for r in self.radius_grid:
                    out.append((float(mu), float(m), float(r)))
        return out

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_grid)


@dataclass
class EnvConfig:
    """Physical + task configuration of the 2D planar push environment."""

    # Physics
    dt: float = 0.08              # quasi-static integration step
    damping: float = 2.5          # linear velocity = force / (mass * damping)
    rot_damping: float = 3.0      # angular velocity = torque / (mass * r^2 * rot_damping)
    mu_static: float = 0.5        # default static friction (real value comes from object)
    mu_kinetic: float = 0.5       # kinetic friction (kept equal to static per object)
    force_small: float = 3.0
    force_large: float = 7.0
    max_force_n: float = 7.0      # normalisation scale for tactile force features
    contact_pressure_gain: float = 1.0

    # Tactile sensor geometry (relative to fingertip)
    sensor_size: float = 2.0      # width of the tactile array in world units
    n_patches: int = 4            # fake "patch" resolution for pressure centroid/spread

    # Action space
    n_contact_angles: int = 36     # 10deg steps around the boundary
    push_modes: tuple[float, ...] = (0.0, 40.0, 85.0)  # force angle (deg) vs inward normal
    force_levels: tuple[bool, ...] = (False, True)     # small / large
    action_phi_grid: tuple[int, float, bool] = field(init=False, default=None)  # placeholder

    # Task
    goal: tuple[float, float] = (0.0, 0.0)
    goal_tol: float = 0.12
    max_steps: int = 60
    start_radius_min: float = 0.8
    start_radius_max: float = 1.6

    def __post_init__(self) -> None:
        if self.action_phi_grid is None:
            self.action_phi_grid = (self.n_contact_angles, tuple(self.push_modes), tuple(self.force_levels))

    @property
    def n_modes(self) -> int:
        return len(self.push_modes)

    @property
    def n_forces(self) -> int:
        return len(self.force_levels)

    def force_from_level(self, level: bool) -> float:
        """Force magnitude for a discrete force level (False -> small, True -> large)."""
        return float(self.force_large if level else self.force_small)

    def decode_action(self, action_id: int) -> tuple[float, float, float]:
        """Decode a discrete action id -> (theta_c_rad, phi_deg, force)."""
        n_modes = self.n_modes
        n_forces = self.n_forces
        angle_idx = action_id // (n_modes * n_forces)
        mode = (action_id // n_forces) % n_modes
        force_idx = action_id % n_forces
        theta_c = 2.0 * math.pi * angle_idx / self.n_contact_angles
        phi_deg = float(self.push_modes[mode])
        force = self.force_from_level(bool(self.force_levels[force_idx]))
        return theta_c, phi_deg, force

    def encode_action(self, theta_c_rad: float, phi_deg: float, force: float) -> int:
        """Invert :meth:`decode_action` (contact angle to nearest grid point)."""
        angle_idx = int(round(theta_c_rad * self.n_contact_angles / (2.0 * math.pi))) % self.n_contact_angles
        mode = min(range(self.n_modes), key=lambda i: abs(self.push_modes[i] - phi_deg))
        force_idx = min(range(self.n_forces), key=lambda i: abs(self.force_from_level(bool(self.force_levels[i])) - force))
        return angle_idx * (self.n_modes * self.n_forces) + mode * self.n_forces + force_idx

    @property
    def n_actions(self) -> int:
        return self.n_contact_angles * len(self.push_modes) * len(self.force_levels)


@dataclass
class TactileConfig:
    """Noise model applied to sensor readings."""

    sigma_noise: float = 0.05     # std of additive Gaussian noise (normalised units)
    noise_clip: float = 2.0       # clip noise magnitude in std units
    slip_reliability: float = 1.0  # [0,1] how reliably slip events report measured mu


# --------------------------------------------------------------------------- #
# World model                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    """Deep-ensemble tactile world model."""

    n_ensemble: int = 5
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    lr: float = 1e-3
    n_epochs_per_update: int = 8
    batch_size: int = 256
    buffer_capacity: int = 4000
    update_every_steps: int = 10
    grad_clip: float = 1.0
    normalize_inputs: bool = True
    predict_std: bool = False      # MSE by default (Gaussian-NLL training biases pose-delta dims to a high-std minimum); set True for heteroscedastic uncertainty


@dataclass
class BeliefConfig:
    """Model-based filter over the candidate grid of hidden properties."""

    obs_noise: float = 0.10        # sigma of observation likelihood (isotropic)
    pose_obs_noise: float = 0.05   # sigma of pose-delta measurement dims in the filter likelihood
    n_obs_samples: int = 24        # samples for expected-info-gain estimation
    prior: str = "uniform"         # "uniform" or "uniform_margins"
    max_entropy_floor: float = 0.05  # below this entropy value treat info gain as ~0


# --------------------------------------------------------------------------- #
# Agents / planners                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class AgentConfig:
    name: str = "info_aware"       # random | greedy | cem | info_aware
    warmup_steps: int = 40         # random-exploration budget before planning starts
    horizon: int = 1               # model roll-out horizon used to score actions
    plan_action_step: int = 4      # contact-angle stride for the considered action set
                                   # (used by model-based planners; ignores other parameters)
    beta: float = 1.0              # info-gain weight for info_aware agent
    cem_pop: int = 120             # CEM: population size
    cem_elite: int = 20            # CEM: number of elites kept
    cem_iters: int = 5             # CEM: optimisation iterations per decision
    cem_rollout_horizon: int = 3   # CEM: length of sampled action sequences
    beta_warmup_steps: int = 0     # steps over which info_aware beta ramps in
    n_warmup_episodes: int = 0     # off-camera random episodes run before the logged
                                   # trials (model training ON); fills the replay buffer
                                   # so every logged trial starts with a planning-capable
                                   # world model instead of the first ~2 episodes being
                                   # pure random (buffer < batch_size by design)
    cem_budget: float = 0.0        # CEM: target model evals per policy decision; when > 0
                                   # the population is scaled to pop ~ budget/(iters*horizon)
                                   # (used by the cem_budget compute sweep)
    seed: int = 0


# --------------------------------------------------------------------------- #
# Experiments                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class ExperimentConfig:
    """One experiment condition run. See conditions.py for the six grids."""

    condition: str = "baseline"
    methods: list[str] = field(default_factory=lambda: ["random", "greedy", "cem", "info_aware"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_trials_per_seed: int = 10
    max_steps_override: int | None = None

    # generalization condition
    n_train_trials_per_seed: int | None = None  # fall back to n_trials_per_seed
    n_eval_trials_per_seed: int | None = None

    # robustness condition
    noise_grid: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.15, 0.35, 0.6])

    # generalization condition
    held_out_fraction: float = 0.3

    # property-scope condition
    scope_grid: list[str] = field(default_factory=lambda: ["narrow", "medium", "wide"])

    # beta-sweep condition (info_aware beta values, including 0 = reward-only anchor)
    beta_grid: list[float] = field(default_factory=list)

    # cem_budget condition (target CEM model evals per policy decision)
    cem_budget_grid: list[float] = field(default_factory=list)

    # generalization condition: use a deterministic systematic region split instead
    # of the random HeldOutSampler (train on small-radius cells, eval on large-radius)
    region_split: bool = False

    # compute: how many model evals are counted (see metrics.py)
    output_root: str = "results/raw"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #

def load_config_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


ALL_CONFIG_CLASSES: tuple[Any, ...] = (ObjectConfig, EnvConfig, TactileConfig, ModelConfig, BeliefConfig, AgentConfig, ExperimentConfig)