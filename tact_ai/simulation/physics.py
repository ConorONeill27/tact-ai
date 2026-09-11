"""Quasi-static 2D rigid-disk contact physics for the Tact AI push environment.

Pure functions only: no environment state, no randomness, numpy exclusively.
All angles are in radians unless the parameter name says otherwise (e.g.
``phi_deg``); see :func:`deg2rad` / :func:`rad2deg`.

Conventions
-----------
- A disk of radius ``r`` is centred at ``(x, y)`` with orientation ``theta``.
- The contact point on the boundary is selected by a boundary angle ``theta_c``:
  ``p = (x + r*cos(theta_c), y + r*sin(theta_c))``.
- The *outward* surface normal at the contact is ``n_hat = (cos(theta_c),
  sin(theta_c))``; the *inward* normal (from the surface toward the disk
  centre) is ``n_in = -n_hat``.
- The tangent is ``t_hat = (-sin(theta_c), cos(theta_c))`` (counter-clockwise).
- A fingertip applies a force of magnitude ``F`` tilted by ``phi_deg`` relative
  to ``n_in``. Positive ``phi`` bends the force toward ``+t_hat``.
- The normal force transmitted to the disk is ``N = F*cos(phi)`` (clamped at 0;
  ``cos(phi) <= 0`` means an outward push / pull, which produces no contact).
- The tangential demand is ``T = F*sin(phi)``. If ``|T| > mu*N`` the fingertip
  slips over the surface and only kinetic friction is transmitted with
  ``T_eff = sign(T)*mu*N``; otherwise static friction transmits the full demand
  ``T_eff = T``. Because friction opposes the surface's relative slipping (the
  disk lags behind the fingertip), the disk is dragged *along* ``+t_hat`` when
  the fingertip attempts to slide the surface in that direction.
- Net force on the disk: ``N*n_in + T_eff*t_hat``. In the overdamped
  quasi-static limit the disk translates along the net force direction (away
  from the pusher): ``v_lin = F_net/(m*damping)``. The tangential force also
  exerts a signed torque about the centre ``tau = r*T_eff`` (positive =
  counter-clockwise), giving an angular velocity
  ``omega = tau/(m*r^2*rot_damping)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Pose(NamedTuple):
    """Rigid-disk pose ``(x, y, theta)``; ``theta`` in radians."""

    x: float
    y: float
    theta: float

    @property
    def xy(self) -> np.ndarray:
        """Position as a length-2 float array."""
        return np.array([self.x, self.y], dtype=float)

    def as_array(self) -> np.ndarray:
        """Full pose as a length-3 float array ``(x, y, theta)``."""
        return np.array([self.x, self.y, self.theta], dtype=float)


@dataclass(frozen=True)
class ContactState:
    """Outcome of one fingertip contact on the disk.

    Attributes
    ----------
    in_contact: True when a contact force is actually transmitted.
    normal_force: contact normal force ``N`` (>= 0), acting along ``n_in``.
    tangential_force: *transmitted* tangential force ``T_eff`` along ``t_hat``
        (full demand ``T`` when static, ``sign(T)*mu*N`` when slipping).
    slip: True when the tangential demand exceeded the Coulomb limit.
    measured_mu: kinetic friction coefficient observed during slip,
        otherwise None.
    applied_tangential: raw tangential demand ``T = F*sin(phi)`` (signed),
        kept for the tactile sensor (near-slip / shear features).
    """

    in_contact: bool
    normal_force: float
    tangential_force: float
    slip: bool
    measured_mu: float | None
    applied_tangential: float = 0.0


# --------------------------------------------------------------------------- #
# Unit / geometry helpers                                                     #
# --------------------------------------------------------------------------- #

def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return float(deg) * math_pi_deg


math_pi_deg = float(np.pi) / 180.0


def rad2deg(rad: float) -> float:
    """Convert radians to degrees."""
    return float(rad) / math_pi_deg


def outward_normal(theta_c: float) -> np.ndarray:
    """Outward unit surface normal ``n_hat`` at boundary angle ``theta_c``."""
    return np.array([np.cos(theta_c), np.sin(theta_c)], dtype=float)


def inward_normal(theta_c: float) -> np.ndarray:
    """Inward unit surface normal ``n_in`` (surface -> disk centre)."""
    return -outward_normal(theta_c)


def tangent(theta_c: float) -> np.ndarray:
    """Unit counter-clockwise tangent ``t_hat`` at boundary angle ``theta_c``."""
    return np.array([-np.sin(theta_c), np.cos(theta_c)], dtype=float)


def contact_point(pose: Pose, r: float, theta_c: float) -> np.ndarray:
    """World position of the boundary contact point for angle ``theta_c``."""
    return pose.xy + r * outward_normal(theta_c)


def moment_of_inertia(m: float, r: float) -> float:
    """Rotational inertia used by the quasi-static model: ``m*r^2``."""
    return m * r * r


def torque_from_tangential(r: float, t_eff: float) -> float:
    """Signed torque about the centre from a tangential force ``t_eff``."""
    return r * t_eff


# --------------------------------------------------------------------------- #
# Core step                                                                    #
# --------------------------------------------------------------------------- #

def step_disk(
    pose: Pose,
    r: float,
    m: float,
    mu: float,
    theta_c: float,
    phi_deg: float,
    F: float,
    dt: float,
    damping: float,
    rot_damping: float,
) -> tuple[ContactState, Pose]:
    """Integrate one quasi-static contact push of a disk.

    Parameters
    ----------
    pose: current ``(x, y, theta)``.
    r: disk radius.
    m: disk mass.
    mu: friction coefficient (shared static/kinetic).
    theta_c: boundary contact angle (radians).
    phi_deg: force angle in degrees relative to the inward normal at contact.
    F: force magnitude (> 0).
    dt: integration time step.
    damping: translational drag coefficient (``v_lin = F_net/(m*damping)``).
    rot_damping: rotational drag coefficient (``omega = tau/(m*r^2*rot_damping)``).

    Returns
    -------
    ``(ContactState, new_pose)``. Any push with ``cos(phi)`` below a tiny
    threshold (outward / purely tangential push, ``phi >= 90`` deg) or
    ``F <= 0`` produces ``in_contact=False`` and an unchanged pose (no contact,
    no tactile, no motion).
    """
    phi = deg2rad(phi_deg)
    cos_phi = float(np.cos(phi))
    # cos_phi can be ~1e-17 for phi = exactly 90 deg; treat anything below a tiny
    # threshold as a non-penetrating (outward / purely tangential) push.
    if cos_phi < 1e-9 or F <= 0.0:
        return ContactState(False, 0.0, 0.0, False, None, 0.0), pose

    N = float(cos_phi * F)
    T = float(np.sin(phi) * F)
    muN = float(mu * N)
    slip = abs(T) > muN
    T_eff = T if not slip else float(np.sign(T) * muN)

    n_in = inward_normal(theta_c)
    t_hat = tangent(theta_c)
    net_force = N * n_in + T_eff * t_hat

    velocity = net_force / (m * damping)
    displacement = velocity * dt

    tau = r * T_eff
    omega = tau / (m * r * r * rot_damping)
    d_theta = omega * dt

    if not np.all(np.isfinite(displacement)) or not np.isfinite(d_theta):
        raise FloatingPointError("step_disk produced non-finite displacement")

    new_pose = Pose(
        float(pose.x + displacement[0]),
        float(pose.y + displacement[1]),
        float(pose.theta + d_theta),
    )
    state = ContactState(
        in_contact=True,
        normal_force=N,
        tangential_force=T_eff,
        slip=bool(slip),
        measured_mu=float(mu) if slip else None,
        applied_tangential=T,
    )
    return state, new_pose