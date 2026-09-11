"""Experience replay for the online tactile world model.

Stores :class:`tact_ai.model.features.Transition` objects in a fixed-capacity
deque; the model bootstrap-samples from it with replacement per ensemble member.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tact_ai.model.features import Transition


class ReplayBuffer:
    """A fixed-capacity, numpy-backed ring buffer of transitions."""

    def __init__(self, capacity: int = 4000) -> None:
        """Create a buffer holding at most ``capacity`` transitions."""
        self.capacity = int(capacity)
        self._data: deque = deque(maxlen=self.capacity)

    def push(self, transition: Transition) -> None:
        """Append a transition, evicting the oldest when over capacity."""
        self._data.append(transition)

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return len(self._data)

    def sample(self, batch_size: int, rng=None) -> list[Transition]:
        """Uniformly sample ``batch_size`` transitions (with replacement).

        ``rng`` is a ``numpy.random.Generator``; defaults to a fresh seeded
        generator when None.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        n = len(self._data)
        if n == 0:
            return []
        idx = rng.integers(0, n, size=int(batch_size))
        return [self._data[int(i)] for i in idx]

    def transitions(self) -> list[Transition]:
        """All stored transitions in insertion order (newest last)."""
        return list(self._data)

    def clear(self) -> None:
        """Drop all stored transitions."""
        self._data.clear()