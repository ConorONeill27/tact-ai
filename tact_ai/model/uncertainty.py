"""Uncertainty utilities built on the deep-ensemble tactile world model.

The primary epistemic signal is *ensemble disagreement*: the per-dimension
standard deviation of the member mean predictions. A simple BALD-style score and
a signal-to-noise heuristic are also provided for planner scoring.
"""

from __future__ import annotations

import numpy as np


def epistemic_std_all(model, x_raw: np.ndarray) -> np.ndarray:
    """Epistemic uncertainty of an ensemble model, ``(B, D_OUT)``.

    Thin wrapper over ``model.epistemic_std``; provided so planners can target
    one uncertainty entry point regardless of the model implementation.
    """
    return model.epistemic_std(x_raw)


def bald_estimate(model, x_raw: np.ndarray, n_draws: int = 16) -> np.ndarray:
    """BALD-style information score per input, ``(B,)`` (nat units).

    Closed-form Gaussian approximation: with per-member means ``mu_m`` and stds
    ``sigma_m``, the predictive mixture is approximated as Gaussian with
    variance ``Var_m(mu_m) + mean_m(sigma_m**2)``. The score is the difference
    of the predictive entropy and the mean member entropy. ``n_draws`` is
    accepted for API compatibility (the closed form is used directly).
    """
    del n_draws
    pred = model.predict(x_raw)
    member_means = pred["member_means"].astype(np.float64)
    member_stds = pred["member_stds"].astype(np.float64)
    var_means = member_means.var(axis=0, ddof=0)
    mean_var = member_stds.mean(axis=0) ** 2
    pred_var = var_means + mean_var
    pred_entropy = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(pred_var, 1e-12))
    mean_member_entropy = 0.5 * np.log(
        2.0 * np.pi * np.e * np.maximum(member_stds ** 2, 1e-12)
    ).mean(axis=0)
    per_dim = np.maximum(pred_entropy - mean_member_entropy, 0.0)
    return np.asarray(per_dim.sum(axis=-1), dtype=np.float32)


def mean_variance_ratio(means: np.ndarray, stds: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Mean-to-standard-deviation ratio of predicted quantities, ``(B,)``.

    ``means`` / ``stds`` are ``(B, D)``; the per-dimension ratio
    ``mean**2 / (std**2 + eps)`` is averaged over dims. High ratio = high
    signal-to-noise, low ratio = mostly-noise prediction.
    """
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    if means.ndim != 2 or stds.ndim != 2:
        means = means.reshape(1, -1)
        stds = stds.reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = means ** 2 / (stds ** 2 + eps)
    ratios = np.where(np.isfinite(ratios), ratios, 0.0)
    return np.asarray(ratios.mean(axis=1), dtype=np.float32).reshape(-1)