"""The six experiment conditions and their configuration grids.

Each builder returns an :class:`ExperimentConfig` with bounded runtime baked in
(documentation of the estimated wall-times lives in the builder docstrings).
``HeldOutSampler`` implements the deterministic train/held-out split for the
generalization condition; ``scope_object_cfg`` / ``scope_env_cfg`` build the
property-scope object/environment configs.
"""

from __future__ import annotations

import math

import numpy as np

from tact_ai.config import EnvConfig, ExperimentConfig, ObjectConfig


def build_condition(name: str) -> ExperimentConfig:
    """Return the ``ExperimentConfig`` for one of the condition names."""
    builders = {
        "baseline": _baseline,
        "information": _information,
        "robustness": _robustness,
        "generalization": _generalization,
        "compute": _compute,
        "property_scope": _property_scope,
        "beta_sweep": _beta_sweep,
        "cem_budget": _cem_budget,
    }
    if name not in builders:
        raise KeyError(f"unknown condition {name!r}; choose from {sorted(builders)}")
    return builders[name]()


ALL_CONDITIONS: tuple[str, ...] = tuple(
    ["baseline", "information", "robustness", "generalization", "compute",
     "property_scope", "beta_sweep", "cem_budget"]
)


def _baseline() -> ExperimentConfig:
    """Primary result: four methods, three seeds, eight trials each.

    Phase-4 final grid (shaved from 10 to 8 trials per seed to keep the full
    six-condition pipeline inside the ~35 min wall-clock budget). Estimated
    wall-time (full run, default 60-step episodes, CPU): dominated by CEM
    (~1800 model evals per policy decision once the buffer fills) and by the
    periodic world-model retrains; allow ~4-8 min total.
    """
    return ExperimentConfig(
        condition="baseline",
        methods=["random", "greedy", "cem", "info_aware"],
        seeds=[0, 1, 2],
        n_trials_per_seed=8,
    )


def _information() -> ExperimentConfig:
    """Marker condition: same data as ``baseline`` for the Phase-4 entropy/argmax analysis.

    Phase 4 reuses the baseline JSON files; this marker runs a light version so
    the pipeline output is self-contained. Runs in well under a minute.
    """
    return ExperimentConfig(
        condition="information",
        methods=["random", "greedy", "cem", "info_aware"],
        seeds=[0],
        n_trials_per_seed=6,
    )


def _robustness() -> ExperimentConfig:
    """Sweep sensor noise sigma; three methods, two seeds, five trials each.

    Five grid points -> 3 x 2 x 5 x 5 = 150 trials; allow ~4-6 min.
    """
    return ExperimentConfig(
        condition="robustness",
        methods=["greedy", "info_aware", "random"],
        seeds=[0, 1],
        n_trials_per_seed=5,
        noise_grid=[0.0, 0.05, 0.15, 0.35, 0.6],
    )


def _generalization() -> ExperimentConfig:
    """Train the world model on small-radius candidates, evaluate on large-radius.

    ``RegionSplit`` gives 9 train / 9 eval candidates (default grid). The
    belief filter always sees the full analytic grid; this condition tests
    whether the *learned world model* generalises to unseen geometry. 3
    methods x 3 seeds x (10 train + 10 eval) = 180 episodes; ~8-12 min.
    """
    return ExperimentConfig(
        condition="generalization",
        methods=["greedy", "cem", "info_aware"],
        seeds=[0, 1, 2],
        n_trials_per_seed=5,
        n_train_trials_per_seed=10,
        n_eval_trials_per_seed=10,
        held_out_fraction=0.3,
        region_split=True,
    )


def _compute() -> ExperimentConfig:
    """Marker condition: model-eval counts per completed task (Phase-4 analysis).

    A light self-contained run; Phase 4 computes ``model_evals / success`` from
    these files (and from ``baseline``). Runs in under a minute.
    """
    return ExperimentConfig(
        condition="compute",
        methods=["greedy", "cem", "info_aware"],
        seeds=[0],
        n_trials_per_seed=6,
    )


def _property_scope() -> ExperimentConfig:
    """Vary the width of the unknown-property distribution.

    For each scope in ``scope_grid`` the ObjectConfig grid and a force-scaled
    EnvConfig (force scales with ``sqrt(median mass)``) are built by
    :func:`scope_object_cfg` / :func:`scope_env_cfg`. With the grids below the
    median mass is 1.5 in every scope, so the force scale factor is identically
    1.0 and reachability differences come purely from the spread of the three
    grids. 2 methods x 2 seeds x 5 trials x 3 scopes = 60 episodes; ~2-4 min.
    """
    return ExperimentConfig(
        condition="property_scope",
        methods=["greedy", "info_aware"],
        seeds=[0, 1],
        n_trials_per_seed=5,
        scope_grid=["narrow", "medium", "wide"],
    )


def _beta_sweep() -> ExperimentConfig:
    """Sweep the information-gain weight beta for the info_aware planner.

    beta=0 yields a reward-only planner (within the info_aware class; ties
    are broken by progress alone); beta=1 is the default; high beta biases
    the planner toward exploratory touches.  beta_grid lists the sweep
    points; greedy is included as a reference method. 2 seeds x 20 trials
    per (beta, method); ~8-12 min.
    """
    return ExperimentConfig(
        condition="beta_sweep",
        methods=["info_aware"],
        seeds=[0, 1],
        n_trials_per_seed=20,
        beta_grid=[0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
    )


def _cem_budget() -> ExperimentConfig:
    """Sweep the CEM compute budget (model evals per policy decision).

    Larger budgets give CEM more population to search over action sequences.
    The budget is mapped to a population size as ``pop = round(budget /
    (iters * horizon))`` with iters and horizon from the base AgentConfig;
    pop is clamped to at least 2*elite. 2 seeds x 20 trials per budget;
    ~10-15 min total.
    """
    return ExperimentConfig(
        condition="cem_budget",
        methods=["cem"],
        seeds=[0, 1],
        n_trials_per_seed=20,
        cem_budget_grid=[1000.0, 3000.0, 5000.0, 10000.0, 25000.0, 50000.0],
    )


# --------------------------------------------------------------------------- #
# Generalization split                                                        #
# --------------------------------------------------------------------------- #

DEFAULT_MASS_MEDIAN = 1.5


class HeldOutSampler:
    """Deterministic split of an ObjectConfig grid into train and held-out sets.

    The split is seeded (default 0), so any two runs produce identical
    partitions. The held-out set is ``round(held_out_fraction * N)`` candidates;
    train and held-out are always disjoint and together cover the grid.
    """

    def __init__(self, object_cfg: ObjectConfig, held_out_fraction: float, seed: int = 0) -> None:
        """Build the partition from ``object_cfg.candidate_grid``."""
        self.object_cfg = object_cfg
        self.held_out_fraction = float(held_out_fraction)
        self.seed = int(seed)
        grid = list(object_cfg.candidate_grid)
        self.grid = [tuple(map(float, g)) for g in grid]
        n_hold = max(1, min(len(self.grid) - 1, int(round(self.held_out_fraction * len(self.grid)))))
        rng = np.random.default_rng(self.seed)
        idx = list(range(len(self.grid)))
        rng.shuffle(idx)
        self._held_out = [self.grid[i] for i in idx[:n_hold]]
        self._train = [self.grid[i] for i in idx[n_hold:]]

    def train_candidates(self) -> list[tuple[float, float, float]]:
        """Candidate triples used for training episodes."""
        return list(self._train)

    def held_out_candidates(self) -> list[tuple[float, float, float]]:
        """Candidate triples reserved for evaluation episodes."""
        return list(self._held_out)


class RegionSplit:
    """Deterministic systematic split: train on small-radius, eval on large-radius.

    With the default ``ObjectConfig`` (radius_grid=[0.4, 1.0]) this gives 9
    train / 9 eval candidates; the held-out set consists entirely of the
    unseen large-radius geometry that the world model must generalise to.
    """

    def __init__(self, object_cfg: ObjectConfig) -> None:
        self.grid = [tuple(map(float, g)) for g in object_cfg.candidate_grid]
        self.radius_grid = list(object_cfg.radius_grid)
        self._small_r = min(self.radius_grid)
        self._train = [t for t in self.grid if abs(t[2] - self._small_r) < 1e-9]
        self._held_out = [t for t in self.grid if abs(t[2] - self._small_r) >= 1e-9]

    def train_candidates(self) -> list[tuple[float, float, float]]:
        return list(self._train)

    def held_out_candidates(self) -> list[tuple[float, float, float]]:
        return list(self._held_out)


# --------------------------------------------------------------------------- #
# Property-scope grids                                                        #
# --------------------------------------------------------------------------- #

SCOPE_GRIDS: dict[str, dict[str, list[float]]] = {
    "narrow": {
        "mu_grid": [0.45, 0.5, 0.55],
        "mass_grid": [1.4, 1.5, 1.6],
        "radius_grid": [0.9, 1.0, 1.1],
    },
    "medium": {
        "mu_grid": [0.2, 0.5, 0.8],
        "mass_grid": [0.8, 1.5, 3.0],
        "radius_grid": [0.4, 1.0],
    },
    "wide": {
        "mu_grid": [0.1, 0.5, 0.9],
        "mass_grid": [0.6, 1.5, 2.4],
        "radius_grid": [0.3, 1.0, 1.8],
    },
}


def scope_object_cfg(scope: str) -> ObjectConfig:
    """ObjectConfig for a property-scope name (narrow|medium|wide)."""
    if scope not in SCOPE_GRIDS:
        raise KeyError(f"unknown scope {scope!r}; choose from {sorted(SCOPE_GRIDS)}")
    g = SCOPE_GRIDS[scope]
    return ObjectConfig(
        mu_grid=list(g["mu_grid"]),
        mass_grid=list(g["mass_grid"]),
        radius_grid=list(g["radius_grid"]),
    )


def scope_env_cfg(scope: str) -> EnvConfig:
    """EnvConfig for a scope with reachability-preserving force scaling.

    The force scale factor is ``sqrt(median_scope_mass / median_default_mass)``
    (larger objects are pushed with more force). For the configured grids every
    median mass equals 1.5, so the factor is 1.0; the mapping is kept so future
    grids with different medians still behave consistently.
    """
    g = SCOPE_GRIDS[scope]
    mass_median = float(np.median(np.asarray(g["mass_grid"], dtype=np.float64)))
    scale = math.sqrt(mass_median / DEFAULT_MASS_MEDIAN)
    base = EnvConfig(goal_tol=0.20)
    return EnvConfig(
        dt=base.dt,
        damping=base.damping,
        rot_damping=base.rot_damping,
        mu_static=base.mu_static,
        mu_kinetic=base.mu_kinetic,
        force_small=base.force_small * scale,
        force_large=base.force_large * scale,
        max_force_n=base.max_force_n,
        contact_pressure_gain=base.contact_pressure_gain,
        sensor_size=base.sensor_size,
        n_patches=base.n_patches,
        n_contact_angles=base.n_contact_angles,
        push_modes=base.push_modes,
        force_levels=base.force_levels,
        goal=base.goal,
        goal_tol=base.goal_tol,
        max_steps=base.max_steps,
        start_radius_min=base.start_radius_min,
        start_radius_max=base.start_radius_max,
    )