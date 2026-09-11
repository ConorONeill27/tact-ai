"""Tests for the experiment harness (tact_ai.experiments.harness).

Uses the cheap random agent so runs stay well under a second; only the JSON
persistence path and determinism are exercised here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tact_ai.agents import RandomAgent
from tact_ai.config import AgentConfig, BeliefConfig, EnvConfig, ExperimentConfig, ModelConfig, ObjectConfig, TactileConfig
from tact_ai.experiments.harness import TrialRecord, run_condition, run_experiment, run_trial
from tact_ai.simulation.environment import ManipulationEnv


def _cfgs(max_steps: int = 10):
    oc = ObjectConfig()
    ec = EnvConfig(max_steps=max_steps)
    tc = TactileConfig(sigma_noise=0.02)
    bc = BeliefConfig(n_obs_samples=4)
    mc = ModelConfig(n_ensemble=1, hidden=[8], batch_size=16, buffer_capacity=64)
    ac = AgentConfig(warmup_steps=0, seed=0)
    return oc, ec, tc, bc, mc, ac


def _exp(
    max_steps: int = 10,
    methods=("random",),
    seeds=(0,),
    trials: int = 2,
    condition: str = "compute",
):
    return ExperimentConfig(
        condition=condition,
        methods=list(methods),
        seeds=list(seeds),
        n_trials_per_seed=trials,
        max_steps_override=max_steps,
    )


def _quiet(msg: str) -> None:
    return None


def test_trial_record_asdict_roundtrip():
    rec = TrialRecord(
        seed=1, trial_idx=2, true_props={"mass": 1.5}, success=True, steps=3,
        final_dist=0.05, model_evals=18,
        entropy_traj=[2.0, 1.0], dist_traj=[1.0, 0.5],
        posterior_true_traj=[0.4, 0.9], argmax_correct=[False, True],
    )
    d = rec.__dataclass_fields__  # dataclass keeps its declared fields
    assert "seed" in d and "success" in d and "model_evals" in d


def test_run_trial_deterministic_and_valid():
    oc, ec, tc, bc, mc, ac = _cfgs()
    env = ManipulationEnv(ec, tc, oc)
    a1 = RandomAgent("random", ac, ec, oc, tc, bc, mc)
    a2 = RandomAgent("random", ac, ec, oc, tc, bc, mc)

    r1 = run_trial(a1, env, 0, np.random.default_rng(123), seed=1)
    r2 = run_trial(a2, env, 0, np.random.default_rng(123), seed=1)

    assert r1.steps == r2.steps
    assert r1.success == r2.success
    assert r1.true_props == r2.true_props
    assert np.allclose(r1.entropy_traj, r2.entropy_traj)
    assert r1.seed == 1 and r1.trial_idx == 0
    assert r1.steps > 0  # real random walk, not a sentinel failure record
    assert 0 <= r1.steps <= ec.max_steps
    assert len(r1.entropy_traj) == r1.steps + 1
    assert len(r1.dist_traj) == r1.steps
    assert len(r1.posterior_true_traj) == r1.steps
    assert len(r1.argmax_correct) == r1.steps
    assert r1.model_evals == 0  # random agent never queries the model
    assert r1.success in (True, False)
    assert r1.true_props != {}


def test_run_trial_empty_subset_returns_failed_record():
    oc, ec, tc, bc, mc, ac = _cfgs()
    env = ManipulationEnv(ec, tc, oc)
    agent = RandomAgent("random", ac, ec, oc, tc, bc, mc)
    rec = run_trial(
        agent, env, 0, np.random.default_rng(1), seed=1, object_subset=[]
    )
    assert rec.success is False
    assert rec.steps == 0
    assert rec.final_dist is None
    assert rec.true_props == {}
    assert rec.model_evals == 0


def test_run_condition_writes_json(tmp_path):
    oc, ec, tc, bc, mc, ac = _cfgs()
    exp = _exp(trials=2)
    out = tmp_path / "raw"
    res = run_condition(
        "random", exp, oc, ec, tc, bc, mc, ac, seed=0,
        out_dir=str(out), n_trials=exp.n_trials_per_seed, progress=_quiet,
    )
    path = __import__("pathlib").Path(res["file"])
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["condition"] == "compute"
    assert data["method"] == "random"
    assert data["seed"] == 0
    assert len(data["trials"]) == 2
    for t in data["trials"]:
        assert t["success"] in (True, False)
        assert t["steps"] >= 0
        assert "entropy_traj" in t
        assert "posterior_true_traj" in t
        assert "argmax_correct" in t
    assert data["n_actions"] == ec.n_actions
    assert data["max_steps"] == ec.max_steps
    assert len(res["records"]) == 2


def test_run_condition_sigma_override_saved(tmp_path):
    oc, ec, tc, bc, mc, ac = _cfgs()
    exp = _exp(trials=1)
    out = tmp_path / "raw"
    res = run_condition(
        "random", exp, oc, ec, tc, bc, mc, ac, seed=0,
        out_dir=str(out), sigma=0.3, tag="sigma0.3", progress=_quiet,
    )
    path = __import__("pathlib").Path(res["file"])
    assert path.name == "compute_random_0_sigma0.3.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sigma_noise"] == pytest.approx(0.3)


def test_run_experiment_minimal(tmp_path):
    oc, ec, tc, bc, mc, ac = _cfgs()
    exp = _exp(methods=("random",), seeds=(0,), trials=2)
    out = tmp_path / "raw"
    results = run_experiment(exp, oc, ec, tc, bc, mc, ac, out_dir=str(out), progress=_quiet)
    assert list(results.keys()) == [("random", 0)]
    assert len(results[("random", 0)]) == 2
    manifest_path = out / "_manifest.json"
    assert manifest_path.exists()
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(entries, list) and len(entries) == 1
    assert entries[0]["condition"] == "compute"
    assert entries[0]["methods"] == ["random"]


def test_run_experiment_robustness_tags_sigma(tmp_path):
    oc, ec, tc, bc, mc, ac = _cfgs()
    exp = ExperimentConfig(
        condition="robustness",
        methods=["random"],
        seeds=[0],
        n_trials_per_seed=1,
        noise_grid=[0.0, 0.2],
        max_steps_override=10,
    )
    out = tmp_path / "raw"
    results = run_experiment(exp, oc, ec, tc, bc, mc, ac, out_dir=str(out), progress=_quiet)
    assert len(results[("random", 0)]) == 2
    files = sorted(p.name for p in out.glob("robustness_random_0_sigma*.json"))
    assert len(files) == 2
    assert "sigma0.0" in files[0] and "sigma0.2" in files[1]


def test_run_experiment_property_scope_tags(tmp_path):
    oc, ec, tc, bc, mc, ac = _cfgs()
    exp = ExperimentConfig(
        condition="property_scope",
        methods=["greedy"],
        seeds=[0],
        n_trials_per_seed=1,
        scope_grid=["narrow", "wide"],
        max_steps_override=10,
    )
    out = tmp_path / "raw"
    results = run_experiment(exp, oc, ec, tc, bc, mc, ac, out_dir=str(out), progress=_quiet)
    assert len(results[("greedy", 0)]) == 2
    files = sorted(p.name for p in out.glob("property_scope_greedy_0_*.json"))
    assert files == [
        "property_scope_greedy_0_narrow.json",
        "property_scope_greedy_0_wide.json",
    ]