"""Hidden object-property sampling and the discrete candidate grid.

Objects are sampled uniformly over the Cartesian product ``mu_grid x
mass_grid x radius_grid`` from :class:`tact_ai.config.ObjectConfig`, so every
sampled object lies on a candidate-grid point (clean identifiability for the
belief over hidden properties ``h = (mu, mass, radius)`` used in later phases).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

from tact_ai.config import ObjectConfig

HIDDEN_PROP_NAMES: tuple[str, ...] = ("mu", "mass", "radius")
"""Names of the hidden physical properties (belief/hierarchy contract)."""


@dataclass(frozen=True)
class ObjectProps:
    """One sampled set of hidden physical properties ``(mu, mass, radius)``."""

    mu: float
    mass: float
    radius: float

    def to_dict(self) -> dict[str, float]:
        """Properties as a primitive dict (for result JSON / info dicts)."""
        return {"mu": self.mu, "mass": self.mass, "radius": self.radius}


def sample_object(object_cfg: ObjectConfig, rng: np.random.Generator) -> ObjectProps:
    """Sample hidden properties uniformly over the candidate grid.

    Parameters
    ----------
    object_cfg: config whose ``candidate_grid`` is your sample space.
    rng: ``numpy.random.Generator`` (deterministic given its seed).

    Returns
    -------
    ``ObjectProps`` whose values coincide exactly with a grid point.
    """
    grid = object_cfg.candidate_grid
    idx = int(rng.integers(0, len(grid)))
    mu, mass, radius = grid[idx]
    return ObjectProps(mu=mu, mass=mass, radius=radius)


def candidate_index(
    properties: ObjectProps, object_cfg: ObjectConfig | None = None
) -> int:
    """Index of ``properties`` in ``object_cfg.candidate_grid``.

    The index follows the grid ordering in
    ``ObjectConfig.candidate_grid`` (mu outer, then mass, then radius inner).
    Raises ``ValueError`` when the properties are not on the grid (matching
    within a tiny tolerance, so exact floats from ``sample_object`` pass).
    """
    cfg = object_cfg if object_cfg is not None else ObjectConfig()
    for i, (mu, mass, radius) in enumerate(cfg.candidate_grid):
        if (
            math.isclose(properties.mu, mu, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(properties.mass, mass, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(properties.radius, radius, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return i
    raise ValueError(f"properties {properties} not on the candidate grid")


def props_from_tuple(values: Union[tuple[float, float, float], ObjectProps]) -> ObjectProps:
    """Coerce a ``(mu, mass, radius)`` triple (or ObjectProps) to ObjectProps."""
    if isinstance(values, ObjectProps):
        return values
    return ObjectProps(mu=float(values[0]), mass=float(values[1]), radius=float(values[2]))