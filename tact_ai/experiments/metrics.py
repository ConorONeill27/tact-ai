"""Pure functions mapping :class:`TrialRecord` lists to aggregate metrics.

Everything here is side-effect free so Phase-4 analysis and the unit tests can
rely on identical numbers. ``aggregate`` returns a flat dict ready for JSON.
"""

from __future__ import annotations

from statistics import median

import numpy as np

from tact_ai.experiments.harness import TrialRecord

TRUE_CONFIDENCE = 0.99  # posterior mass at the true candidate defining "concentration"


def _pad_or_truncate(values: list, n: int, fill: float | None = None) -> list[float]:
    """Clip a trajectory to length ``n``, padding the tail with ``fill`` (or its last value)."""
    if n <= 0:
        return []
    out = [float(v) for v in values]
    if len(out) >= n:
        return out[:n]
    if fill is None:
        fill = out[-1] if out else 0.0
    return out + [float(fill)] * (n - len(out))


def aggregate(
    records: list[TrialRecord],
    max_steps: int | None = None,
    step_cutoffs: tuple[float, ...] | None = None,
) -> dict:
    """Aggregate per-trial metrics over ``records``.

    Parameters
    ----------
    records: any non-empty list of trial records.
    max_steps: episode budget used to right-pad trajectories and to scale the
        default success-rate cutoffs; when None the longest record length is
        used.
    step_cutoffs: fractions of ``max_steps`` at which success rate is evaluated
        (default ``(0.5, 1.0)``).

    Returns a dict with keys: ``success_rate``, ``mean_steps_success``,
    ``median_steps_success``, ``mean_steps_all``, ``success_rate_at`` (list,
    aligned with ``step_cutoffs``), ``mean_model_evals``, ``mean_final_entropy``,
    ``entropy_at_k`` (length ``max_steps + 1``; index 0 is the pre-episode
    prior), ``posterior_true_at_k`` and ``argmax_correct_rate_at_k`` (length
    ``max_steps``), and ``mean_first_concentration_step``.
    """
    if not records:
        raise ValueError("aggregate() requires at least one trial record")
    steps_all = [int(r.steps) for r in records]
    if max_steps is None:
        max_steps = max([int(r.steps) for r in records] + [1])
    max_steps = int(max_steps)
    cutoffs = list(step_cutoffs) if step_cutoffs is not None else [0.5, 1.0]

    successes = [r for r in records if r.success]
    success_steps = [int(r.steps) for r in successes]

    success_rate_at = []
    for frac in cutoffs:
        limit = frac * max_steps
        success_rate_at.append(
            float(np.mean([1.0 if r.success and r.steps <= limit else 0.0 for r in records]))
        )

    entropy_at_k = np.zeros(max_steps + 1, dtype=np.float64)
    for r in records:
        entropy_at_k += np.asarray(_pad_or_truncate(r.entropy_traj, max_steps + 1), dtype=np.float64)
    entropy_at_k /= len(records)

    post_true_at_k = np.zeros(max_steps, dtype=np.float64)
    for r in records:
        post_true_at_k += np.asarray(
            _pad_or_truncate(r.posterior_true_traj, max_steps), dtype=np.float64
        )
    post_true_at_k /= len(records)

    argmax_at_k = np.zeros(max_steps, dtype=np.float64)
    for r in records:
        argmax_at_k += np.asarray(
            _pad_or_truncate([1.0 if b else 0.0 for b in r.argmax_correct], max_steps),
            dtype=np.float64,
        )
    argmax_at_k /= len(records)

    first_concentration: list[float] = []
    for r in records:
        idx = next(
            (i for i, p in enumerate(r.posterior_true_traj) if p > TRUE_CONFIDENCE),
            None,
        )
        first_concentration.append(float(max_steps) if idx is None else float(idx))

    return {
        "n_trials": len(records),
        "success_rate": float(np.mean([1.0 if r.success else 0.0 for r in records])),
        "mean_steps_success": float(np.mean(success_steps)) if success_steps else 0.0,
        "median_steps_success": float(median(success_steps)) if success_steps else 0.0,
        "mean_steps_all": float(np.mean(steps_all)),
        "success_rate_at": [float(v) for v in success_rate_at],
        "mean_model_evals": float(np.mean([r.model_evals for r in records])),
        "mean_final_entropy": float(np.mean([r.final_entropy for r in records])),
        "entropy_at_k": [float(v) for v in entropy_at_k.tolist()],
        "posterior_true_at_k": [float(v) for v in post_true_at_k.tolist()],
        "argmax_correct_rate_at_k": [float(v) for v in argmax_at_k.tolist()],
        "mean_first_concentration_step": float(np.mean(first_concentration)),
    }


def as_table(results_dict: dict) -> list[dict]:
    """Flatten an aggregate dict into a small team table for reports.

    ``results_dict`` maps a row label (e.g. a method or ``(method, seed)`` key)
    to the dict returned by :func:`aggregate`. Returns a list of flat row dicts
    of length-1 scalar summaries.
    """
    rows: list[dict] = []
    for label, metrics in results_dict.items():
        row: dict = {
            "label": str(label),
            "success_rate": metrics.get("success_rate", 0.0),
            "mean_steps_all": metrics.get("mean_steps_all", 0.0),
            "mean_steps_success": metrics.get("mean_steps_success", 0.0),
            "mean_model_evals": metrics.get("mean_model_evals", 0.0),
            "mean_final_entropy": metrics.get("mean_final_entropy", 0.0),
            "mean_first_concentration_step": metrics.get(
                "mean_first_concentration_step", 0.0
            ),
        }
        rows.append(row)
    return rows