"""Sweep runner: run every requested condition and report progress.

Thin orchestrator over :func:`harness.run_experiment`; calls
:func:`conditions.build_condition` for each name, writes the per-condition JSON
files and a global ``results/raw/_manifest.json``, and prints a progress line per
condition so unattended runs can be monitored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from tact_ai.config import (
    AgentConfig,
    BeliefConfig,
    EnvConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectConfig,
    TactileConfig,
)
from tact_ai.experiments.conditions import ALL_CONDITIONS, build_condition
from tact_ai.experiments.harness import run_experiment


def run_sweep(
    condition_names: list[str],
    object_cfg: ObjectConfig,
    env_cfg: EnvConfig,
    tact_cfg: TactileConfig,
    belief_cfg: BeliefConfig,
    model_cfg: ModelConfig,
    agent_cfg: AgentConfig,
    out_dir: str = "results/raw",
    smoke: bool = False,
    exp_override: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run conditions in order; returns ``{condition: {(method, seed): records}}``.

    With ``smoke=True`` every condition is shrunk via :func:`make_smoke`
    (n_trials 2, seeds [0], short episodes, smaller model/CEM) for fast checks.
    ``exp_override`` applies extra ``ExperimentConfig`` field values on top of
    each built condition (used by the CLI ``--n-trials`` / ``--seeds`` flags).
    """
    log = progress if progress is not None else lambda msg: print(msg, flush=True)
    t_start = time.time()
    all_results: dict[str, Any] = {}
    wall_times: dict[str, float] = {}
    for name in condition_names:
        exp_cfg = build_condition(name)
        if exp_override:
            for key, value in exp_override.items():
                setattr(exp_cfg, key, value)
        a_cfg, m_cfg = agent_cfg, model_cfg
        if smoke:
            exp_cfg, a_cfg, m_cfg = make_smoke(exp_cfg, agent_cfg, model_cfg)
            log(f"  (smoke) {name}: trials={exp_cfg.n_trials_per_seed} seeds={exp_cfg.seeds} "
                f"max_steps={exp_cfg.max_steps_override}")
        t0 = time.time()
        log(f"== condition {name!r} starting ==")
        results = run_experiment(
            exp_cfg,
            object_cfg,
            env_cfg,
            tact_cfg,
            belief_cfg,
            m_cfg,
            a_cfg,
            out_dir=out_dir,
            progress=log,
        )
        n = sum(len(v) for v in results.values())
        wall = time.time() - t0
        wall_times[name] = wall
        all_results[name] = results
        log(
            f"[{name}] done in {wall:.1f}s "
            f"({n} trial records, "
            f"total elapsed {(time.time() - t_start) / 60:.1f} min)"
        )
    _write_summary_manifest(all_results, out_dir)
    _append_walltimes(wall_times, out_dir)
    return all_results


def _append_walltimes(wall_times: dict[str, float], out_dir: str) -> None:
    """Merge per-condition wall-times into ``out_dir/_walltimes.json``.

    Merging (not overwriting) lets the six conditions be launched in separate
    CLI invocations without losing earlier conditions' timings. The file is
    one JSON object ``{"conditions": {name: seconds}, "stamp": iso, "total":
    sum}``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "_walltimes.json"
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    conds = dict(data.get("conditions", {}))
    conds.update({str(k): float(v) for k, v in wall_times.items()})
    payload = {
        "conditions": conds,
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": float(sum(conds.values())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary_manifest(all_results: dict, out_dir: str) -> None:
    """Rewrite the manifest with one entry per condition actually completed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "_manifest.json"
    if not manifest.exists():
        manifest.write_text("[]", encoding="utf-8")
        return
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        entries = []
    completed = {c: sum(len(v) for v in all_results[c].values()) for c in all_results}
    summary = {
        "summary": "sweep runner summary",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_conditions": completed,
        "total_records": int(sum(completed.values())),
        "entries": entries,
    }
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def default_configs() -> tuple[ObjectConfig, EnvConfig, TactileConfig, BeliefConfig, ModelConfig, AgentConfig]:
    """Canned default configs for the CLI / quick runs.

    These are the *tuned* experiment defaults (goal_tol 0.20, 5 x [192,192]
    model, lr 5e-4, 20 epochs/update, update every 3 steps, beta 1.0).  The
    ``n_warmup_episodes`` is set to 3 so every logged trial starts with a
    planning-capable world model (180 buffer transitions from off-camera
    random episodes).
    """
    return (
        ObjectConfig(),
        EnvConfig(goal_tol=0.20),
        TactileConfig(sigma_noise=0.05),
        BeliefConfig(n_obs_samples=24),
        ModelConfig(
            n_ensemble=5, hidden=[192, 192], lr=5e-4,
            n_epochs_per_update=20, batch_size=128,
            buffer_capacity=4000, update_every_steps=3,
        ),
        AgentConfig(warmup_steps=40, plan_action_step=4, beta=1.0, n_warmup_episodes=3),
    )


SMOKE_AGENT_CFG = dict(
    warmup_steps=8, plan_action_step=8, cem_pop=48, cem_iters=3,
    beta_warmup_steps=4, seed=0, n_warmup_episodes=0,
)
SMOKE_MODEL_CFG = dict(
    n_ensemble=3, hidden=[32], n_epochs_per_update=6, batch_size=16,
    buffer_capacity=512, update_every_steps=5,
)
SMOKE_MAX_STEPS = 30


def make_smoke(
    exp_cfg: ExperimentConfig,
    agent_cfg: AgentConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> tuple[ExperimentConfig, AgentConfig, ModelConfig]:
    """Shrink a condition and the agent/model configs for the ``--smoke`` run.

    ``n_trials`` becomes 2, seeds collapse to ``[0]``, episodes are capped at
    ``SMOKE_MAX_STEPS`` steps, the planning action stride doubles (fewer
    considered actions), and CEM population is cut. Used only by the CLI smoke
    mode / tests so the full sweep stays well under the acceptance time budget.
    """
    smoke_exp = ExperimentConfig(
        condition=exp_cfg.condition,
        methods=list(exp_cfg.methods),
        seeds=[0],
        n_trials_per_seed=2,
        max_steps_override=SMOKE_MAX_STEPS,
        noise_grid=list(exp_cfg.noise_grid),
        held_out_fraction=exp_cfg.held_out_fraction,
        n_train_trials_per_seed=2,
        n_eval_trials_per_seed=2,
        scope_grid=list(exp_cfg.scope_grid),
        beta_grid=list(getattr(exp_cfg, "beta_grid", [])),
        cem_budget_grid=list(getattr(exp_cfg, "cem_budget_grid", [])),
        region_split=getattr(exp_cfg, "region_split", False),
        output_root=exp_cfg.output_root,
    )
    base_agent = agent_cfg if agent_cfg is not None else AgentConfig()
    base_model = model_cfg if model_cfg is not None else ModelConfig()
    smoke_agent = AgentConfig(
        name=base_agent.name,
        warmup_steps=SMOKE_AGENT_CFG["warmup_steps"],
        horizon=base_agent.horizon,
        plan_action_step=SMOKE_AGENT_CFG["plan_action_step"],
        beta=base_agent.beta,
        cem_pop=SMOKE_AGENT_CFG["cem_pop"],
        cem_elite=max(4, int(SMOKE_AGENT_CFG["cem_pop"] // 8)),
        cem_iters=SMOKE_AGENT_CFG["cem_iters"],
        cem_rollout_horizon=base_agent.cem_rollout_horizon,
        beta_warmup_steps=SMOKE_AGENT_CFG["beta_warmup_steps"],
        n_warmup_episodes=SMOKE_AGENT_CFG["n_warmup_episodes"],
        seed=0,
    )
    smoke_model = ModelConfig(
        n_ensemble=SMOKE_MODEL_CFG["n_ensemble"],
        hidden=list(SMOKE_MODEL_CFG["hidden"]),
        activation=base_model.activation,
        lr=base_model.lr,
        n_epochs_per_update=SMOKE_MODEL_CFG["n_epochs_per_update"],
        batch_size=SMOKE_MODEL_CFG["batch_size"],
        buffer_capacity=SMOKE_MODEL_CFG["buffer_capacity"],
        update_every_steps=SMOKE_MODEL_CFG["update_every_steps"],
        grad_clip=base_model.grad_clip,
        normalize_inputs=base_model.normalize_inputs,
        predict_std=base_model.predict_std,
    )
    return smoke_exp, smoke_agent, smoke_model


def condition_names_from_args(names: list[str] | None) -> list[str]:
    """Validate the CLI condition list; empty means every built-in condition."""
    if names is None or not names:
        return list(ALL_CONDITIONS)
    for n in names:
        if n not in ALL_CONDITIONS:
            raise ValueError(f"unknown condition {n!r}; choose from {list(ALL_CONDITIONS)}")
    return list(names)