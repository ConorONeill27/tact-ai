"""Unit tests for feature encoding and transition containers (tact_ai.model.features)."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.config import EnvConfig
from tact_ai.model import features as F


def _obs() -> dict:
    return {
        "tactile": np.arange(12, dtype=np.float32) / 12.0,
        "pose": np.array([0.5, -0.2, 0.538516], dtype=np.float32),
        "action": 0,
    }


def test_dimension_constants():
    assert F.D_TACTILE == 12
    assert F.D_POSE == 3
    assert F.D_ACTION == 7
    assert F.D_BELIEF == 18
    assert F.D_IN == 12 + 3 + 7 + 18 == 40
    assert F.D_OUT == 12 + 3 == 15


def test_encode_action_layout():
    cfg = EnvConfig()
    assert cfg.n_actions == 216
    for a in range(cfg.n_actions):
        v = F.encode_action(cfg, a)
        assert v.shape == (7,)
        assert v.dtype == np.float32
        theta_c, phi_deg, force = cfg.decode_action(a)
        assert v[0] == pytest.approx(np.cos(theta_c))
        assert v[1] == pytest.approx(np.sin(theta_c))
        mode = (a // cfg.n_forces) % cfg.n_modes
        force_idx = a % cfg.n_forces
        assert v[2 + mode] == pytest.approx(1.0)
        assert v[2 + cfg.n_modes + force_idx] == pytest.approx(1.0)
        assert int(v[2:2 + cfg.n_modes].sum()) == 1
        assert int(v[2 + cfg.n_modes:].sum()) == 1


def test_encode_action_distinct_for_distinct_ids():
    cfg = EnvConfig()
    seen = {tuple(F.encode_action(cfg, a).tolist()) for a in range(cfg.n_actions)}
    assert len(seen) == cfg.n_actions


def test_encode_obs_layout_and_belief_ignored():
    obs = _obs()
    b1 = np.full(18, 1 / 18, dtype=np.float32)
    b2 = np.zeros(18, dtype=np.float32)
    x = F.encode_obs(obs, b1)
    y = F.encode_obs(obs, b2)
    assert x.shape == (15,)
    assert x.dtype == np.float32
    assert np.allclose(x, y)
    assert np.allclose(x[:12], obs["tactile"])
    assert np.allclose(x[12:], obs["pose"])


def test_build_input_layout():
    cfg = EnvConfig()
    obs = _obs()
    a = 123
    belief = np.linspace(0.0, 1.0, 18, dtype=np.float32)
    x = F.build_input(obs, a, belief, cfg)
    assert x.shape == (40,)
    assert x.dtype == np.float32
    assert np.allclose(x[:12], obs["tactile"])
    assert np.allclose(x[12:15], obs["pose"])
    assert np.allclose(x[15:22], F.encode_action(cfg, a))
    assert np.allclose(x[22:], belief)


def test_transition_fields():
    t = F.Transition(obs=_obs(), action_id=5, belief=np.ones(18, np.float32),
                     next_obs=_obs(), done=False)
    assert t.action_id == 5
    assert t.done is False
    assert t.belief.shape == (18,)