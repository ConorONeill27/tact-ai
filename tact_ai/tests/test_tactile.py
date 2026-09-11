"""Unit tests for the tactile sensor (tact_ai.sensing.tactile)."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.config import TactileConfig, EnvConfig
from tact_ai.sensing.tactile import TactileSensor, TACTILE_DIM
from tact_ai.simulation.physics import ContactState


def _sensor(sigma: float = 0.0, reliability: float = 1.0, seed: int = 0) -> TactileSensor:
    return TactileSensor(
        TactileConfig(sigma_noise=sigma, slip_reliability=reliability),
        EnvConfig(),
        seed=seed,
    )


def _contact(state: ContactState, theta_c: float = 0.5, radius: float = 0.5) -> np.ndarray:
    return _sensor().read(state, theta_c_rad=theta_c, radius=radius)


def test_dims_and_dtype():
    sensor = _sensor()
    assert TACTILE_DIM == 12
    x = sensor.read(
        ContactState(in_contact=True, normal_force=5.0, tangential_force=0.8,
                     slip=False, measured_mu=None, applied_tangential=1.5),
        theta_c_rad=0.5, radius=0.5,
    )
    assert x.shape == (12,)
    assert x.dtype == np.float32


def test_typical_contact_sane_ranges():
    sensor = _sensor()
    x = sensor.read(
        ContactState(in_contact=True, normal_force=5.0, tangential_force=0.8,
                     slip=False, measured_mu=None, applied_tangential=1.5),
        theta_c_rad=0.5, radius=0.5,
    )
    assert np.all(x >= -1.0 - 1e-6) and np.all(x <= 1.0 + 1e-6)
    assert x[0] > 0.0                 # normal force present
    assert x[1] > 0.0                 # tangential shear present
    assert x[2] == 0.0                # no slip
    assert x[3] == 0.0                # measured mu only on slip
    assert x[6] > 0.0                 # peak pressure
    assert x[7] > 0.0                 # spread > 0 for a finite-radius disk
    assert x[10] == 1.0               # in-contact flag

    zero = sensor.read(
        ContactState(in_contact=False, normal_force=0.0, tangential_force=0.0,
                     slip=False, measured_mu=None, applied_tangential=0.0),
        theta_c_rad=0.5, radius=0.5,
    )
    assert np.allclose(zero, 0.0)


def test_spread_grows_with_radius():
    sensor = _sensor()
    state = ContactState(in_contact=True, normal_force=5.0, tangential_force=0.0,
                         slip=False, measured_mu=None, applied_tangential=0.0)
    small = sensor.read(state, theta_c_rad=0.0, radius=0.4)[7]
    large = sensor.read(state, theta_c_rad=0.0, radius=1.0)[7]
    assert large > small                       # radius signature
    assert large <= 1.0 + 1e-6


def test_slip_reports_mu_at_high_reliability():
    sensor = _sensor(sigma=0.0, reliability=1.0)
    x = sensor.read(
        ContactState(in_contact=True, normal_force=1.0, tangential_force=0.5,
                     slip=True, measured_mu=0.5, applied_tangential=6.9),
        theta_c_rad=0.3, radius=1.0,
    )
    assert x[2] == pytest.approx(1.0)
    assert x[3] == pytest.approx(0.5)


def test_noise_increases_variance():
    state = ContactState(in_contact=True, normal_force=5.0, tangential_force=1.5,
                         slip=False, measured_mu=None, applied_tangential=2.0)
    low, high = [], []
    for seed in range(100):
        low.append(_sensor(sigma=0.0, seed=seed).read(state, 0.4, 0.8))
        high.append(_sensor(sigma=0.05, seed=seed).read(state, 0.4, 0.8))
    std_low = np.std(np.stack(low), axis=0)
    std_high = np.std(np.stack(high), axis=0)
    assert float(std_high[0]) > float(std_low[0]) + 1e-4
    assert float(std_high[0]) > 0.0


def test_noise_stays_within_clip_bounds():
    sensor = _sensor(sigma=0.15, reliability=1.0, seed=3)
    state = ContactState(in_contact=True, normal_force=0.3, tangential_force=0.1,
                         slip=False, measured_mu=None, applied_tangential=0.2)
    x = sensor.read(state, theta_c_rad=0.0, radius=0.5)
    assert np.all(x >= -1.0 - 1e-6) and np.all(x <= 1.0 + 1e-6)


def test_slip_reliability_can_zero_mu():
    sensor = _sensor(sigma=0.0, reliability=0.0, seed=0)
    state = ContactState(in_contact=True, normal_force=1.0, tangential_force=0.5,
                         slip=True, measured_mu=0.5, applied_tangential=6.9)
    x = sensor.read(state, theta_c_rad=0.3, radius=1.0)
    assert x[3] == pytest.approx(0.0)