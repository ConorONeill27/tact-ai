"""Unit tests for the quasi-static contact physics (tact_ai.simulation.physics)."""

from __future__ import annotations

import numpy as np
import pytest

from tact_ai.simulation.physics import (
    Pose,
    step_disk,
    deg2rad,
    torque_from_tangential,
    moment_of_inertia,
)

DT = 0.08
DAMP = 2.5
ROT = 3.0


def test_normal_push_no_slip():
    start = Pose(0.0, 0.0, 0.0)
    state, new = step_disk(
        start, r=1.0, m=1.0, mu=0.5, theta_c=0.0, phi_deg=0.0, F=5.0,
        dt=DT, damping=DAMP, rot_damping=ROT,
    )
    assert state.in_contact
    assert not state.slip
    assert state.normal_force == pytest.approx(5.0)
    assert state.tangential_force == pytest.approx(0.0)
    assert state.applied_tangential == pytest.approx(0.0)
    assert state.measured_mu is None
    # theta_c=0 pushes along the inward normal: -x direction.
    expected = 5.0 / (1.0 * DAMP) * DT
    assert new.x == pytest.approx(-expected)
    assert new.y == pytest.approx(0.0)
    assert new.theta == pytest.approx(0.0)


def test_grazing_slip_and_measured_mu():
    start = Pose(0.0, 0.0, 0.0)
    state, _ = step_disk(
        start, r=1.0, m=1.0, mu=0.5, theta_c=0.0, phi_deg=85.0, F=5.0,
        dt=DT, damping=DAMP, rot_damping=ROT,
    )
    assert state.in_contact
    assert state.slip
    N = 5.0 * np.cos(deg2rad(85.0))
    assert state.normal_force == pytest.approx(N)
    # kinetic cap: |T_eff| = mu * N
    assert state.tangential_force == pytest.approx(0.5 * N)
    assert abs(state.tangential_force) < abs(state.applied_tangential)
    assert state.measured_mu == pytest.approx(0.5)


def test_heavy_object_moves_less():
    kwargs = dict(r=1.0, mu=0.5, theta_c=0.5, phi_deg=20.0, F=7.0, dt=DT, damping=DAMP, rot_damping=ROT)
    _, light = step_disk(Pose(0.0, 0.0, 0.0), m=0.8, **kwargs)
    _, heavy = step_disk(Pose(0.0, 0.0, 0.0), m=3.0, **kwargs)
    d_light = float(np.hypot(light.x, light.y))
    d_heavy = float(np.hypot(heavy.x, heavy.y))
    assert d_light > d_heavy
    assert d_light > 0.0


def test_larger_radius_larger_torque():
    # Torque grows linearly with the lever arm for fixed tangential force.
    t_eff = 0.4
    assert torque_from_tangential(1.0, t_eff) == pytest.approx(0.4)
    assert torque_from_tangential(1.0, t_eff) > torque_from_tangential(0.4, t_eff)

    # And the angular displacement in step_disk follows the analytic formula.
    start = Pose(0.0, 0.0, 0.0)
    state, new = step_disk(
        start, r=1.0, m=1.0, mu=0.5, theta_c=0.0, phi_deg=40.0, F=5.0,
        dt=DT, damping=DAMP, rot_damping=ROT,
    )
    tau = torque_from_tangential(1.0, state.tangential_force)
    expected_theta = tau / (moment_of_inertia(1.0, 1.0) * ROT) * DT
    assert new.theta == pytest.approx(expected_theta)


def test_outward_push_no_contact():
    start = Pose(1.0, 2.0, 0.4)
    state, new = step_disk(
        start, r=1.0, m=1.0, mu=0.5, theta_c=1.0, phi_deg=95.0, F=5.0,
        dt=DT, damping=DAMP, rot_damping=ROT,
    )
    assert not state.in_contact
    assert state.normal_force == 0.0
    assert state.tangential_force == 0.0
    assert state.measured_mu is None
    assert new == start


def test_phi_90_is_no_contact():
    start = Pose(0.0, 0.0, 0.0)
    state, new = step_disk(
        start, r=1.0, m=1.0, mu=0.5, theta_c=0.3, phi_deg=90.0, F=5.0,
        dt=DT, damping=DAMP, rot_damping=ROT,
    )
    assert not state.in_contact
    assert new == start


def test_slip_caps_tangential_transmission():
    start = Pose(0.0, 0.0, 0.0)
    state, _ = step_disk(
        start, r=0.4, m=1.0, mu=0.2, theta_c=0.0, phi_deg=85.0, F=7.0,
        dt=DT, damping=DAMP, rot_damping=ROT,
    )
    assert state.slip
    assert state.tangential_force == pytest.approx(0.2 * state.normal_force)
    assert state.measured_mu == pytest.approx(0.2)