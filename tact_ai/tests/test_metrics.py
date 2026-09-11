"""Tests for the side-effect-free metric aggregation (tact_ai.experiments.metrics)."""

from __future__ import annotations

import pytest

from tact_ai.experiments.harness import TrialRecord
from tact_ai.experiments.metrics import TRUE_CONFIDENCE, aggregate, as_table


def _rec(
    seed: int = 0,
    success: bool = True,
    steps: int = 10,
    evals: int = 0,
    entropy=(2.5, 1.5, 0.4),
    post=(0.3, 0.7, 0.95),
    argmax=(True, True, True),
    idx: int = 0,
) -> TrialRecord:
    return TrialRecord(
        seed=seed,
        trial_idx=idx,
        true_props={"mass": 1.5},
        success=success,
        steps=steps,
        final_dist=0.01 if success else 0.9,
        model_evals=evals,
        entropy_traj=[float(v) for v in entropy],
        dist_traj=[float(s) for s in range(steps)],
        posterior_true_traj=[float(v) for v in post],
        argmax_correct=[bool(v) for v in argmax],
        phase="train",
        final_entropy=float(entropy[-1]),
    )


def test_aggregate_basic_stats():
    records = [
        _rec(seed=0, steps=8, evals=10, entropy=(2.0, 1.0, 0.2)),
        _rec(seed=1, steps=12, evals=20, entropy=(2.0, 1.2, 0.5)),
    ]
    agg = aggregate(records)
    assert agg["n_trials"] == 2
    assert agg["success_rate"] == 1.0
    assert agg["mean_steps_success"] == 10.0
    assert agg["median_steps_success"] == 10.0
    assert agg["mean_steps_all"] == 10.0
    assert agg["mean_model_evals"] == 15.0
    assert agg["mean_final_entropy"] == pytest.approx(0.35)
    assert agg["success_rate_at"] == [0.0, 1.0]  # limits 6 and 12
    assert len(agg["entropy_at_k"]) == 13  # max record length 12 -> 13 (prior at index 0)
    assert agg["entropy_at_k"][0] == pytest.approx(2.0)


def test_aggregate_success_rate_cutoffs():
    records = [_rec(seed=0, steps=4), _rec(seed=1, steps=8), _rec(seed=2, steps=12)]
    agg = aggregate(records, max_steps=10, step_cutoffs=(0.5, 1.0))
    # limit 5 and 10: only the 4-step trial clears the first cutoff
    assert agg["success_rate_at"] == [1 / 3, 2 / 3]


def test_aggregate_uses_max_steps_for_padding():
    records = [_rec(seed=0, steps=4, entropy=(2.0, 1.0, 0.2), post=(0.2, 0.8, 1.0))]
    agg = aggregate(records, max_steps=5)
    assert len(agg["entropy_at_k"]) == 6  # max_steps + 1 (index 0 = prior)
    assert agg["entropy_at_k"][0] == pytest.approx(2.0)
    assert len(agg["posterior_true_at_k"]) == 5
    # short trajectory pads with its last value, so the tail is 1.0
    assert agg["posterior_true_at_k"][-1] == pytest.approx(1.0)
    assert len(agg["argmax_correct_rate_at_k"]) == 5
    assert agg["argmax_correct_rate_at_k"][-1] == pytest.approx(1.0)


def test_aggregate_empty_records_raises():
    with pytest.raises(ValueError):
        aggregate([])


def test_aggregate_no_successes():
    records = [_rec(success=False, steps=9), _rec(success=False, steps=3)]
    agg = aggregate(records, max_steps=9)
    assert agg["success_rate"] == 0.0
    assert agg["mean_steps_success"] == 0.0
    assert agg["median_steps_success"] == 0.0
    assert agg["success_rate_at"] == [0.0, 0.0]
    assert agg["mean_first_concentration_step"] == pytest.approx(9.0)  # never crosses -> max_steps


def test_aggregate_first_concentration_step():
    never = _rec(seed=0, steps=10, post=(0.2, 0.4, 0.8))  # never > TRUE_CONFIDENCE
    at2 = _rec(seed=1, steps=10, post=(0.2, 0.4, 0.995))
    agg = aggregate([never, at2], max_steps=10)
    assert TRUE_CONFIDENCE == pytest.approx(0.99)
    assert agg["mean_first_concentration_step"] == pytest.approx((10 + 2) / 2)


def test_as_table_flattens_scalars():
    a = _rec(seed=0, steps=8, evals=5)
    b = _rec(seed=1, steps=12, evals=7)
    table = as_table({"random": aggregate([a]), "greedy": aggregate([b])})
    assert [r["label"] for r in table] == ["random", "greedy"]
    assert table[0]["mean_model_evals"] == 5.0
    assert table[1]["mean_steps_all"] == 12.0
    assert set(table[0].keys()) >= {
        "label", "success_rate", "mean_steps_all", "mean_steps_success",
        "mean_model_evals", "mean_final_entropy", "mean_first_concentration_step",
    }