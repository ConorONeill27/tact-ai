"""Tests for the Phase-4 analysis pipeline (results, plots, report).

Uses small hand-written result-JSON fixtures (no experiment runs) so the suite
stays near-instant; exercises the exact payload schema written by the harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tact_ai.analysis import available
from tact_ai.analysis.plots import make_plots
from tact_ai.analysis.report import run_report
from tact_ai.analysis.results import (
    aggregate_records,
    condition_table,
    load_all,
    load_condition,
    load_files,
)


def _trial(seed: int, tidx: int, success: bool, steps: int, evals: int) -> dict:
    ent = list(np.linspace(2.8, 0.4 if success else 1.0, steps + 1))
    post = list(np.linspace(0.2, 0.98, steps))
    return {
        "seed": seed,
        "trial_idx": tidx,
        "true_props": {"mass": 1.5, "mu": 0.5, "radius": 1.0},
        "success": bool(success),
        "steps": steps,
        "final_dist": 0.05 if success else 0.9,
        "model_evals": evals,
        "entropy_traj": [float(v) for v in ent],
        "dist_traj": [float(v) for v in range(steps)],
        "posterior_true_traj": [float(v) for v in post],
        "argmax_correct": [True] * steps,
        "phase": "train",
        "final_entropy": float(ent[-1]),
    }


def _write(raw, condition: str, method: str, seed: int, tag: str | None, phase: str, trials, sigma: float = 0.05):
    fname = f"{condition}_{method}_{seed}" + (f"_{tag}" if tag else "") + ".json"
    payload = {
        "condition": condition,
        "method": method,
        "seed": seed,
        "phase": phase,
        "tag": tag,
        "sigma_noise": sigma,
        "object_subset": None,
        "model_training": True,
        "agent_cfg": {"name": method, "warmup_steps": 10, "seed": seed},
        "env_cfg_used": {"goal_tol": 0.2, "max_steps": 60},
        "model_cfg": {"n_ensemble": 3, "hidden": [16], "lr": 0.0005},
        "trials": trials,
    }
    (raw / fname).write_text(json.dumps(payload, indent=1), encoding="utf-8")


@pytest.fixture()
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for m in ("random", "greedy", "cem", "info_aware"):
        _write(raw, "baseline", m, 0, None, "train",
               [_trial(0, i, i % 3 == 0, 60, 54 * (i + 1)) for i in range(2)])
    for m in ("greedy", "info_aware"):
        _write(raw, "information", m, 0, None, "train", [_trial(0, 0, True, 6, 0)])
    for m in ("greedy", "info_aware"):
        for s in (0.0, 0.5):
            _write(raw, "robustness", m, 0, f"sigma{s}", "train",
                   [_trial(0, 0, s < 0.3, 10, 0)], sigma=s)
    for m in ("greedy", "cem", "info_aware"):
        _write(raw, "generalization", m, 0, "train", "train", [_trial(0, 0, True, 8, 0)])
        _write(raw, "generalization", m, 0, "eval", "eval", [_trial(0, 0, False, 5, 0)])
    for m in ("greedy", "cem", "info_aware"):
        _write(raw, "compute", m, 0, None, "train", [_trial(0, 0, True, 6, 1800)])
    for m in ("greedy", "info_aware"):
        for scope in ("narrow", "medium", "wide"):
            _write(raw, "property_scope", m, 0, scope, "train", [_trial(0, 0, scope == "narrow", 5, 0)])
    (raw / "_walltimes.json").write_text(
        json.dumps({"conditions": {"baseline": 10.0, "compute": 5.0}, "total": 15.0}),
        encoding="utf-8",
    )
    return raw


# --------------------------------------------------------------------------- #
# results.py                                                                  #
# --------------------------------------------------------------------------- #


def test_available_true():
    assert available() is True


def test_load_all_keys_and_merge(raw_dir):
    data = load_all(str(raw_dir))
    assert ("baseline", "greedy", 0) in data
    assert len(data[("baseline", "random", 0)]) == 2
    # tag-bearing robustness files merge under one (condition, method, seed) key
    assert len(data[("robustness", "greedy", 0)]) == 2
    # baseline 4 + information 2 + robustness 2 + generalization 3 + compute 3 + scope 2
    assert len(data) == 16


def test_load_files_returns_valid_result_files(raw_dir):
    files = load_files(str(raw_dir))
    names = {f.condition for f in files}
    assert names == {
        "baseline", "information", "robustness", "generalization", "compute", "property_scope",
    }
    all_records = sum(len(f.records) for f in files)
    assert all_records == 4 * 2 + 2 + 4 + 6 + 3 + 6


def test_load_condition_restriction(raw_dir):
    cond = load_condition(str(raw_dir), "baseline")
    assert set(cond) == {("random", 0), ("greedy", 0), ("cem", 0), ("info_aware", 0)}


def test_aggregate_records_matches_metrics(raw_dir):
    data = load_all(str(raw_dir))
    records = data[("baseline", "greedy", 0)]
    agg = aggregate_records(records, max_steps=60)
    assert agg["n_trials"] == 2
    assert agg["success_rate"] == pytest.approx(0.5)
    assert agg["mean_model_evals"] == pytest.approx(54 * 1.5)
    assert len(agg["entropy_at_k"]) == 61


def test_condition_table_rows(raw_dir):
    data = load_all(str(raw_dir))
    by_method = {m: data[("baseline", m, 0)] for m in ("random", "greedy", "cem", "info_aware")}
    table = condition_table(by_method, max_steps=60)
    labels = [row["label"] for row in table]
    assert labels == ["random", "greedy", "cem", "info_aware"]
    assert all("success_rate" in row and "mean_model_evals" in row for row in table)


# --------------------------------------------------------------------------- #
# plots.py                                                                    #
# --------------------------------------------------------------------------- #


def test_make_plots_renders_six_figures(raw_dir, tmp_path):
    figures = tmp_path / "figures"
    written = make_plots(str(raw_dir), str(figures))
    expect = {"fig_baseline", "fig_information", "fig_robustness",
              "fig_generalization", "fig_compute", "fig_property_scope",
              "fig_trial_index"}
    names = {Path(p).stem for p in written}
    assert names == expect
    assert len(written) == 7
    for p in written:
        assert Path(p).stat().st_size > 500  # a real raster, not an empty canvas


# --------------------------------------------------------------------------- #
# report.py                                                                   #
# --------------------------------------------------------------------------- #


def test_run_report_writes_markdown(raw_dir, tmp_path):
    report_path = str(tmp_path / "RESULTS.md")
    run_report(str(raw_dir), report_path)
    text = Path(report_path).read_text(encoding="utf-8")
    assert "Tact AI" in text and "SciFest" in text
    assert "## Baseline" in text
    assert "figure:" in text and "fig_baseline.png" in text
    assert "## Wall times" in text
    assert "baseline" in text
    assert "total" in text