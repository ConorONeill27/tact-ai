"""Phase-4 result loading: JSON files under ``results/raw`` -> tidy TrialRecords.

Every per-condition result file written by :mod:`tact_ai.experiments.harness`
carries a stable payload (``condition``, ``method``, ``seed``, optional ``tag``
/``phase``, the config snapshot and a ``trials`` list of :class:`TrialRecord`
dicts). This module loads them into plain Python objects only (numpy arrays are
used internally for the trajectory metrics; there is no pandas dependency).

Public entry points
-------------------
- :func:`load_results` / :func:`load_all` / :func:`results_from_dir`:
  ``dict[(condition, method, seed)] -> TrialRecords`` (records from a
  ``(method, seed)`` are merged across tags, e.g. every sigma of robustness).
- :func:`load_condition`: same shape, restricted to one condition and keyed by
  ``(method, seed)``.
- :func:`load_files`: the per-file views (with tag/phase/sigma) used by the
  plots and report modules.
- :func:`aggregate_records` / :func:`condition_table`: aggregate helpers built
  on :mod:`tact_ai.experiments.metrics`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tact_ai.experiments.harness import TrialRecord
from tact_ai.experiments.metrics import aggregate, as_table

TrialRecords = list[TrialRecord]
"""Loose alias for a list of episode records belonging to one (method, seed)."""

_TRIAL_FIELDS = frozenset(TrialRecord.__dataclass_fields__)


# --------------------------------------------------------------------------- #
# File views                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class ResultFile:
    """One persisted result JSON: metadata snapshot plus its trial records."""

    condition: str
    method: str
    seed: int
    tag: str | None
    phase: str
    sigma_noise: float | None
    model_training: bool
    max_steps: int
    env_cfg: dict[str, Any] = field(default_factory=dict)
    model_cfg: dict[str, Any] = field(default_factory=dict)
    agent_cfg: dict[str, Any] = field(default_factory=dict)
    records: TrialRecords = field(default_factory=list)


def _trial_from_dict(d: dict) -> TrialRecord:
    """Rebuild a :class:`TrialRecord` from a JSON dict (unknown keys dropped)."""
    return TrialRecord(**{k: v for k, v in d.items() if k in _TRIAL_FIELDS})


def load_file(path: str | Path) -> ResultFile | None:
    """Load and validate one result file; returns ``None`` for non-result JSON."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("condition", "method", "seed", "trials"):
        if key not in payload:
            return None
    try:
        seed = int(payload["seed"])
        max_steps = int(payload.get("max_steps") or 0)
    except (TypeError, ValueError):
        return None
    trials_raw = payload.get("trials")
    if not isinstance(trials_raw, list):
        return None
    records = [_trial_from_dict(t) for t in trials_raw if isinstance(t, dict)]
    return ResultFile(
        condition=str(payload["condition"]),
        method=str(payload["method"]),
        seed=seed,
        tag=(str(payload["tag"]) if payload.get("tag") is not None else None),
        phase=str(payload.get("phase") or "train"),
        sigma_noise=payload.get("sigma_noise"),
        model_training=bool(payload.get("model_training", True)),
        max_steps=max_steps,
        env_cfg=dict(payload.get("env_cfg_used") or {}),
        model_cfg=dict(payload.get("model_cfg") or {}),
        agent_cfg=dict(payload.get("agent_cfg") or {}),
        records=records,
    )


def load_files(out_dir: str = "results/raw") -> list[ResultFile]:
    """Load every valid result file directly under ``out_dir`` (non-recursive)."""
    out = Path(out_dir)
    if not out.is_dir():
        return []
    files: list[ResultFile] = []
    for path in sorted(out.glob("*.json")):
        if path.name.startswith("_"):
            continue  # manifest / walltimes bookkeeping
        f = load_file(path)
        if f is not None:
            files.append(f)
    return files


# --------------------------------------------------------------------------- #
# Aggregated views                                                            #
# --------------------------------------------------------------------------- #


def load_all(out_dir: str = "results/raw") -> dict[tuple[str, str, int], TrialRecords]:
    """Tidy loader: ``{(condition, method, seed): TrialRecords}``.

    Records from tagged files (sigma / scope / train-eval) under one
    ``(condition, method, seed)`` key are merged in file order.
    """
    by_key: dict[tuple[str, str, int], TrialRecords] = {}
    for f in load_files(out_dir):
        key = (f.condition, f.method, f.seed)
        by_key.setdefault(key, []).extend(f.records)
    return by_key


def results_from_dir(out_dir: str = "results/raw") -> dict[tuple[str, str, int], TrialRecords]:
    """Alias for :func:`load_all` (kept for API symmetry)."""
    return load_all(out_dir)


def load_results(out_dir: str = "results/raw") -> dict[tuple[str, str, int], TrialRecords]:
    """Daemon-facing loader; identical to :func:`load_all`."""
    return load_all(out_dir)


def load_condition(
    out_dir: str, condition: str
) -> dict[tuple[str, int], TrialRecords]:
    """Restrict :func:`load_all` to one condition, keyed by ``(method, seed)``."""
    out: dict[tuple[str, int], TrialRecords] = {}
    for (cond, method, seed), records in load_all(out_dir).items():
        if cond == condition:
            out[(method, seed)] = records
    return out


# --------------------------------------------------------------------------- #
# Aggregation helpers                                                         #
# --------------------------------------------------------------------------- #


def aggregate_records(records: TrialRecords, max_steps: int | None = None) -> dict:
    """Aggregate one record group with :func:`metrics.aggregate`."""
    return aggregate(records, max_steps=max_steps)


def condition_table(
    by_method: dict[str, TrialRecords], max_steps: int | None = None
) -> list[dict]:
    """Flatten per-method aggregates to report rows (numpy-only content)."""
    return as_table({label: aggregate(recs, max_steps=max_steps) for label, recs in by_method.items()})


def default_max_steps(files: list[ResultFile]) -> int:
    """Largest episode budget seen across files (used to align trajectories)."""
    steps = [f.max_steps for f in files if f.max_steps]
    return int(max(steps)) if steps else 60