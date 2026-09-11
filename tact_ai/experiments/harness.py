"""Experiment harness: run trials, record results, save per-condition JSON.

One ``run_condition`` call runs every trial of one ``(method, seed)`` pair for an
``ExperimentConfig`` and writes a single JSON file
``out_dir/{condition}_{method}_{seed}[_{tag}].json`` with a stable schema (see
the module docstring of :func:`run_condition`). ``run_experiment`` orchestrates
the methods/seeds (and condition-specific axes such as noise sigma, scope, or the
train/eval phases of generalization) and returns a ``{("method", seed): [TrialRecord]}
`` mapping for the sweep runner / Phase-4 analysis.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tact_ai.agents import Agent
from tact_ai.agents.cem_agent import CEMAgent
from tact_ai.agents.greedy_agent import GreedyAgent
from tact_ai.agents.info_aware_agent import InfoAwareAgent
from tact_ai.agents.random_agent import RandomAgent
from tact_ai.config import (
    AgentConfig,
    BeliefConfig,
    EnvConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectConfig,
    TactileConfig,
)
from tact_ai.simulation.environment import ManipulationEnv
from tact_ai.simulation.objects import candidate_index, props_from_tuple

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "cem": CEMAgent,
    "info_aware": InfoAwareAgent,
}


@dataclass
class TrialRecord:
    """One episode: trajectories plus summary outcome and model-eval count."""

    seed: int
    trial_idx: int
    true_props: dict
    success: bool
    steps: int
    final_dist: float | None
    model_evals: int
    entropy_traj: list[float] = field(default_factory=list)
    dist_traj: list[float] = field(default_factory=list)
    posterior_true_traj: list[float] = field(default_factory=list)
    argmax_correct: list[bool] = field(default_factory=list)
    phase: str = "train"  # "train" | "eval" (generalization condition)
    final_entropy: float = 0.0
    """Final belief entropy after the last update of the episode."""


def _clone(cfg, **kw: Any) -> Any:
    """Shallow-deep copy a config dataclass and override ``kw`` fields.

    ``dataclasses.replace`` cannot be used here because ``EnvConfig`` has an
    ``init=False`` field; deep-copy-then-setattr works for every config.
    """
    out = copy.deepcopy(cfg)
    for key, value in kw.items():
        setattr(out, key, value)
    return out


def _failed_record(seed: int, trial_idx: int, phase: str = "train", evals_delta: int = 0) -> TrialRecord:
    """A sentinel record for a trial that could not start or crashed."""
    return TrialRecord(
        seed=seed,
        trial_idx=trial_idx,
        true_props={},
        success=False,
        steps=0,
        final_dist=None,
        model_evals=evals_delta,
        phase=phase,
        final_entropy=np.log(18.0),
    )


def run_trial(
    agent: Agent,
    env: ManipulationEnv,
    trial_idx: int,
    rng: np.random.Generator,
    seed: int,
    object_subset: list[tuple[float, float, float]] | None = None,
    max_steps: int | None = None,
    phase: str = "train",
) -> TrialRecord:
    """Run one episode of ``agent`` in ``env`` and record its trajectory.

    ``rng`` is a dedicated per-trial generator (derived from the condition seed
    and trial index by the caller); it draws the hidden object (when
    ``object_subset`` is given) and the env reset seed, so runs are
    deterministic given ``(seed, trial_idx)``.
    """
    max_steps = int(max_steps) if max_steps is not None else int(env.env_cfg.max_steps)
    evals_start = int(agent.model_evals)
    try:
        if object_subset is not None and len(object_subset) == 0:
            raise ValueError("empty object subset")
        true_props: dict | None = None
        if object_subset is not None:
            props = props_from_tuple(object_subset[int(rng.integers(0, len(object_subset)))])
            true_props = dict(props.to_dict())
        env_seed = int(rng.integers(0, 2**31 - 1))
        obs = env.reset(seed=env_seed)
        if object_subset is not None:
            env.props = props
            true_props = dict(props.to_dict())
        true_idx = candidate_index(env.props, env.object_cfg)

        agent.on_episode_start(obs)
        entropy_traj: list[float] = [float(agent.belief.entropy())]
        posterior_true_traj: list[float] = []
        argmax_correct: list[bool] = []
        dist_traj: list[float] = []
        steps, success, final_dist = 0, False, None

        for _ in range(max_steps):
            action = int(agent.act(obs))
            next_obs, reward, done, info = env.step(action)
            agent.observe(obs, action, next_obs, reward, done)
            steps = int(env.n_steps)
            final_dist = float(info["dist_to_goal"])
            dist_traj.append(final_dist)
            entropy_traj.append(float(agent.belief.entropy()))
            b = agent.belief.belief_vector()
            posterior_true_traj.append(float(b[true_idx]))
            argmax_correct.append(bool(int(np.argmax(b)) == true_idx))
            obs = next_obs
            if done:
                break
        success = bool(final_dist < env.env_cfg.goal_tol)
        return TrialRecord(
            seed=seed,
            trial_idx=trial_idx,
            true_props=dict(env.props.to_dict()) if true_props is None else true_props,
            success=success,
            steps=steps,
            final_dist=final_dist,
            model_evals=int(agent.model_evals) - evals_start,
            entropy_traj=entropy_traj,
            dist_traj=dist_traj,
            posterior_true_traj=posterior_true_traj,
            argmax_correct=argmax_correct,
            phase=phase,
            final_entropy=float(agent.belief.entropy()),
        )
    except Exception:
        return _failed_record(seed, trial_idx, phase, evals_delta=int(agent.model_evals) - evals_start)


def run_condition(
    method: str,
    exp_cfg: ExperimentConfig,
    object_cfg: ObjectConfig,
    env_cfg: EnvConfig,
    tact_cfg: TactileConfig,
    belief_cfg: BeliefConfig,
    model_cfg: ModelConfig,
    agent_cfg: AgentConfig,
    seed: int,
    out_dir: str = "results/raw",
    object_subset: list[tuple[float, float, float]] | None = None,
    model_training: bool = True,
    phase: str = "train",
    tag: str | None = None,
    sigma: float | None = None,
    n_trials: int | None = None,
    agent: Agent | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run every trial of one ``(method, seed)`` and write one JSON result file.

    Returns a dict with keys ``{"condition", "method", "seed", "tag", "phase",
    "file", "records": [TrialRecord ...], "object_subset", "model_training"}``.

    ``sigma`` overrides the tactile noise in the *environment sensor only* (the
    belief filter keeps its own likelihood noise). ``object_subset`` samples the
    hidden object from the given list instead of the full grid; ``agent`` allows
    reusing a persistent agent instance (train -> eval generalization phases);
    ``model_training`` is passed to ``agent.set_model_training``.
    """
    log = progress if progress is not None else lambda msg: print(msg, flush=True)

    if sigma is not None:
        tact_cfg = _clone(tact_cfg, sigma_noise=float(sigma))
    if exp_cfg.max_steps_override is not None:
        env_cfg = _clone(env_cfg, max_steps=int(exp_cfg.max_steps_override))
    agent_cfg = _clone(agent_cfg, seed=seed)
    n_trials = int(n_trials) if n_trials is not None else int(exp_cfg.n_trials_per_seed)

    env = ManipulationEnv(env_cfg, tact_cfg, object_cfg)
    if agent is None:
        agent = AGENT_REGISTRY[method](
            name=method,
            agent_cfg=agent_cfg,
            env_cfg=env_cfg,
            object_cfg=object_cfg,
            tact_cfg=tact_cfg,
            belief_cfg=belief_cfg,
            model_cfg=model_cfg,
        )
    agent.set_model_training(model_training)

    # Warmup: run off-camera episodes with random actions before the logged
    # trials.  Buffer + model updates are active; every logged trial then
    # starts with a planning-capable world model instead of spending ~128
    # transitions in random fallback mode.
    n_warmup = int(getattr(agent_cfg, "n_warmup_episodes", 0))
    if n_warmup > 0 and agent is not None:
        log(
            f"  [{exp_cfg.condition}/{method}/seed{seed}] warmup: "
            f"{n_warmup} episodes ({n_warmup * int(env_cfg.max_steps)} steps)"
        )
        for w in range(n_warmup):
            w_rng = np.random.default_rng(seed * 100003 + w * 7919 + 9999)
            run_trial(
                agent, env, trial_idx=-1, rng=w_rng, seed=seed,
                object_subset=object_subset,
                max_steps=env_cfg.max_steps, phase="warmup",
            )

    records: list[TrialRecord] = []
    t0 = time.time()
    for trial_idx in range(n_trials):
        trial_rng = np.random.default_rng(seed * 100003 + trial_idx * 7919 + 17)
        rec = run_trial(
            agent,
            env,
            trial_idx,
            trial_rng,
            seed=seed,
            object_subset=object_subset,
            max_steps=env_cfg.max_steps,
            phase=phase,
        )
        records.append(rec)
        if (trial_idx + 1) % max(1, n_trials // 2) == 0 or trial_idx + 1 == n_trials:
            log(
                f"  [{exp_cfg.condition}/{method}/seed{seed}{'/' + phase if phase != 'train' else ''}] "
                f"trial {trial_idx + 1}/{n_trials} "
                f"success={rec.success} steps={rec.steps} "
                f"({time.time() - t0:.1f}s elapsed)"
            )

    # --- JSON output ---------------------------------------------------- #
    out_path = Path(out_dir)
    fname = f"{exp_cfg.condition}_{method}_{seed}" + (f"_{tag}" if tag else "") + ".json"
    payload: dict[str, Any] = {
        "condition": exp_cfg.condition,
        "method": method,
        "seed": seed,
        "phase": phase,
        "tag": tag,
        "scope": None,
        "sigma_noise": float(sigma) if sigma is not None else float(tact_cfg.sigma_noise),
        "object_subset": [list(t) for t in object_subset] if object_subset is not None else None,
        "model_training": bool(model_training),
        "agent_cfg": asdict(agent_cfg),
        "env_cfg_used": asdict(env_cfg),
        "object_cfg": asdict(object_cfg),
        "tactile_cfg": asdict(tact_cfg),
        "belief_cfg": asdict(belief_cfg),
        "model_cfg": asdict(model_cfg),
        "experiment_cfg": asdict(exp_cfg),
        "n_actions": int(env_cfg.n_actions),
        "max_steps": int(env_cfg.max_steps),
        "trials": [asdict(r) for r in records],
    }
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / fname).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "condition": exp_cfg.condition,
        "method": method,
        "seed": seed,
        "phase": phase,
        "tag": tag,
        "file": str(out_path / fname),
        "records": records,
        "object_subset": object_subset,
        "model_training": bool(model_training),
    }


# --------------------------------------------------------------------------- #
# Condition orchestration                                                     #
# --------------------------------------------------------------------------- #


def run_experiment(
    exp_cfg: ExperimentConfig,
    object_cfg: ObjectConfig,
    env_cfg: EnvConfig,
    tact_cfg: TactileConfig,
    belief_cfg: BeliefConfig,
    model_cfg: ModelConfig,
    agent_cfg: AgentConfig,
    out_dir: str = "results/raw",
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run one full experiment condition and return ``{(method, seed): records}``.

    Handles the condition-specific axes:

    - ``baseline``/``information``/``compute``: methods x seeds.
    - ``robustness``: additionally sweeps ``exp_cfg.noise_grid`` (each point is a
      separate JSON file tagged ``sigma{sigma}``).
    - ``generalization``: train episodes on the train subset (model updates ON)
      then eval episodes on the held-out subset (updates OFF) with the *same*
      persistent agent, split via :class:`conditions.HeldOutSampler`.
    - ``property_scope``: additionally sweeps ``exp_cfg.scope_grid`` using scope
      object/env configs from :mod:`tact_ai.experiments.conditions`.
    """
    log = progress if progress is not None else lambda msg: print(msg, flush=True)
    log(f"[{exp_cfg.condition}] running methods={exp_cfg.methods} seeds={exp_cfg.seeds}")

    results: dict[tuple[str, int], list[TrialRecord]] = {}
    eps = list(exp_cfg.methods)

    if exp_cfg.condition == "robustness":
        for sigma in exp_cfg.noise_grid:
            for method in eps:
                for seed in exp_cfg.seeds:
                    res = run_condition(
                        method, exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg,
                        model_cfg, agent_cfg, seed,
                        out_dir=out_dir, sigma=float(sigma),
                        tag=f"sigma{sigma}", progress=log,
                    )
                    results.setdefault((method, seed), []).extend(res["records"])
        _append_manifest(exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg, model_cfg,
                         agent_cfg, out_dir, extra={"noise_grid": list(exp_cfg.noise_grid)})

    elif exp_cfg.condition == "generalization":
        from tact_ai.experiments.conditions import HeldOutSampler, RegionSplit

        if getattr(exp_cfg, "region_split", False):
            split = RegionSplit(object_cfg)
        else:
            split = HeldOutSampler(object_cfg, float(exp_cfg.held_out_fraction), seed=0)
        train_set = split.train_candidates()
        held_set = split.held_out_candidates()
        log(f"[{exp_cfg.condition}] split: {len(train_set)} train / {len(held_set)} held-out")
        n_train = int(exp_cfg.n_train_trials_per_seed or exp_cfg.n_trials_per_seed)
        n_eval = int(exp_cfg.n_eval_trials_per_seed or exp_cfg.n_trials_per_seed)
        for method in eps:
            for seed in exp_cfg.seeds:
                # One agent instance persists from train into eval (model/buffer
                # carry over; only model updates are switched off for eval).
                persistent = AGENT_REGISTRY[method](
                    name=method,
                    agent_cfg=_clone(agent_cfg, seed=seed),
                    env_cfg=_clone(env_cfg),
                    object_cfg=object_cfg,
                    tact_cfg=tact_cfg,
                    belief_cfg=belief_cfg,
                    model_cfg=model_cfg,
                )
                res_train = run_condition(
                    method, exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg,
                    model_cfg, agent_cfg, seed,
                    out_dir=out_dir,
                    object_subset=train_set, model_training=True,
                    phase="train", tag="train", n_trials=n_train,
                    agent=persistent, progress=log,
                )
                res_eval = run_condition(
                    method, exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg,
                    model_cfg, agent_cfg, seed,
                    out_dir=out_dir,
                    object_subset=held_set, model_training=False,
                    phase="eval", tag="eval", n_trials=n_eval,
                    agent=persistent, progress=log,
                )
                results.setdefault((method, seed), []).extend(res_train["records"])
                results.setdefault((method, seed), []).extend(res_eval["records"])
        _append_manifest(exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg, model_cfg,
                         agent_cfg, out_dir, extra={
                             "held_out_fraction": float(exp_cfg.held_out_fraction),
                             "n_train": len(train_set), "n_held_out": len(held_set),
                         })

    elif exp_cfg.condition == "property_scope":
        from tact_ai.experiments.conditions import scope_env_cfg as _scope_env_cfg
        from tact_ai.experiments.conditions import scope_object_cfg as _scope_object_cfg

        for scope in exp_cfg.scope_grid:
            scp_obj_cfg = _scope_object_cfg(scope)
            scp_env_cfg = _scope_env_cfg(scope)
            for method in eps:
                for seed in exp_cfg.seeds:
                    res = run_condition(
                        method, exp_cfg, scp_obj_cfg, scp_env_cfg, tact_cfg,
                        belief_cfg, model_cfg, agent_cfg, seed,
                        out_dir=out_dir, tag=scope, progress=log,
                    )
                    results.setdefault((method, seed), []).extend(res["records"])
        _append_manifest(exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg, model_cfg,
                         agent_cfg, out_dir, extra={"scope_grid": list(exp_cfg.scope_grid)})

    elif exp_cfg.condition == "beta_sweep":
        beta_grid = list(getattr(exp_cfg, "beta_grid", [1.0]))
        for beta in beta_grid:
            for method in eps:
                if method not in ("info_aware", "greedy"):
                    continue
                for seed in exp_cfg.seeds:
                    ac = _clone(agent_cfg, seed=seed, beta=float(beta))
                    res = run_condition(
                        method, exp_cfg, object_cfg, env_cfg, tact_cfg,
                        belief_cfg, model_cfg, ac, seed,
                        out_dir=out_dir, tag=f"beta{beta}", progress=log,
                    )
                    results.setdefault((method, seed), []).extend(res["records"])
        _append_manifest(exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg, model_cfg,
                         agent_cfg, out_dir, extra={"beta_grid": beta_grid})

    elif exp_cfg.condition == "cem_budget":
        budget_grid = list(getattr(exp_cfg, "cem_budget_grid", []))
        for budget in budget_grid:
            for method in eps:
                if method != "cem":
                    continue
                for seed in exp_cfg.seeds:
                    ac = _clone(agent_cfg, seed=seed, cem_budget=float(budget))
                    res = run_condition(
                        method, exp_cfg, object_cfg, env_cfg, tact_cfg,
                        belief_cfg, model_cfg, ac, seed,
                        out_dir=out_dir, tag=f"budget{int(budget)}", progress=log,
                    )
                    results.setdefault((method, seed), []).extend(res["records"])
        _append_manifest(exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg, model_cfg,
                         agent_cfg, out_dir, extra={"cem_budget_grid": budget_grid})

    else:  # baseline / information / compute
        for method in eps:
            for seed in exp_cfg.seeds:
                res = run_condition(
                    method, exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg,
                    model_cfg, agent_cfg, seed, out_dir=out_dir, progress=log,
                )
                results.setdefault((method, seed), []).extend(res["records"])
        _append_manifest(exp_cfg, object_cfg, env_cfg, tact_cfg, belief_cfg, model_cfg,
                         agent_cfg, out_dir)

    n_records = sum(len(r) for r in results.values())
    log(f"[{exp_cfg.condition}] finished: {len(results)} (method,seed) keys, "
        f"{n_records} trial records written under {out_dir}")
    return results


def _append_manifest(
    exp_cfg: ExperimentConfig,
    object_cfg: ObjectConfig,
    env_cfg: EnvConfig,
    tact_cfg: TactileConfig,
    belief_cfg: BeliefConfig,
    model_cfg: ModelConfig,
    agent_cfg: AgentConfig,
    out_dir: str,
    extra: dict | None = None,
) -> None:
    """Append one condition run to ``results/raw/_manifest.json`` (reproducibility)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "_manifest.json"
    entries: list[dict] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    entry: dict[str, Any] = {
        "condition": exp_cfg.condition,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methods": list(exp_cfg.methods),
        "seeds": list(exp_cfg.seeds),
        "n_trials_per_seed": int(exp_cfg.n_trials_per_seed),
        "max_steps_override": exp_cfg.max_steps_override,
        "object_cfg": asdict(object_cfg),
        "env_cfg": asdict(env_cfg),
        "tactile_cfg": asdict(tact_cfg),
        "belief_cfg": asdict(belief_cfg),
        "model_cfg": asdict(model_cfg),
        "agent_cfg": asdict(agent_cfg),
        "n_actions": int(env_cfg.n_actions),
    }
    if extra:
        entry.update(extra)
    entries.append(entry)
    manifest_path.write_text(
        json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8"
    )