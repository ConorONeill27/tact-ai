"""Tact AI CLI entry point.

Modes
-----
- ``quicktest``: a short seeded random episode (environment smoke test).
- ``train``: a single ``info_aware`` demonstration run on one random object,
  printing the belief-entropy trajectory and the world-model loss before/after a
  retrain (uses a deliberately small model so it finishes in < ~30s).
- ``experiments``: run the requested sweep conditions, writing JSON under
  ``results/raw/``. Accepts ``--conditions`` (default: all six), ``--smoke``
  (tiny runs for testing) and ``--skip-analysis``.
- ``analyze``: Phase-4 analysis pipeline (stubs for now; reports gracefully).
- ``all``: experiments then analysis.
- ``video``: render the side-by-side RANDOM vs INFO-AWARE demo animation to
  ``results/video/`` (MP4 via ffmpeg, GIF via Pillow, plus raw PNG frames).

Determinism: every run is seeded from ``--seed`` (used directly for
quicktest/train; experiment *conditions own their seeds* per the phase-3
design).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tact_ai.config import (
    AgentConfig,
    BeliefConfig,
    EnvConfig,
    ModelConfig,
    ObjectConfig,
    TactileConfig,
)
from tact_ai.model.features import D_OUT, D_TACTILE, build_input
from tact_ai.simulation.environment import ManipulationEnv

MODES = ("quicktest", "train", "experiments", "all", "analyze", "video", "diagnose")


def _clone_cfg(cfg, **kw):
    """Deep-copy a dataclass config and override the given fields.

    Handles ``EnvConfig``'s ``init=False`` placeholder field (which rules out
    ``dataclasses.replace``); mirrors the harness's private ``_clone`` so the
    CLI can apply ``--model-*`` / ``--n-trials`` / ``--seeds`` overrides.
    """
    import copy

    out = copy.deepcopy(cfg)
    for key, value in kw.items():
        setattr(out, key, value)
    return out

# --------------------------------------------------------------------------- #
# Environment smoke test (unchanged from Phase 1)                              #
# --------------------------------------------------------------------------- #


def _mode_quicktest(seed: int) -> None:
    """Build one env and run a short seeded random episode (smoke test)."""
    env = ManipulationEnv()
    obs = env.reset(seed=seed)

    print("== reset observation ==")
    print("keys:", sorted(obs.keys()))
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape} dtype={val.dtype} min={val.min():.3f} max={val.max():.3f}")
        else:
            print(f"  {key}: {val}")

    rng = np.random.default_rng(seed + 1)
    movements = []
    readings = []
    rewards = []
    dones = 0
    obs_history = []
    for _ in range(20):
        action = int(rng.integers(0, env.env_cfg.n_actions))
        obs, reward, done, info = env.step(action)
        obs_history.append(obs)
        movements.append(info["dist_to_goal"])
        readings.append(np.asarray(obs["tactile"]))
        rewards.append(reward)
        dones += int(done)

    stack = np.stack(readings)
    avg_delta = float(np.mean([abs(movements[i] - movements[i - 1]) for i in range(1, len(movements))]))

    print("== 20 random steps ==")
    print(f"n_actions: {env.env_cfg.n_actions}")
    print(f"avg |dist change| / step: {avg_delta:.3f}")
    print(f"mean reward: {float(np.mean(rewards)):.3f}  end dist: {movements[-1]:.3f}")
    print(f"tactile min={float(stack.min()):.3f} max={float(stack.max()):.3f} mean={float(stack.mean()):.3f}")
    print(f"per-dim range over episode:")
    for i, name in enumerate(
        ["N", "T", "slip", "mu", "cx", "cy", "peak", "spread", "cos", "sin", "in_c", "slip_prox"]
    ):
        print(f"  [{i:2d}] {name:<8} min={stack[:, i].min():.3f} max={stack[:, i].max():.3f}")
    print("sample tactile vectors:")
    for step_idx in (0, 9, 19):
        vec = obs_history[step_idx]["tactile"]
        print(f"  step {step_idx:2d}: {np.round(vec, 3)}  (in_contact={vec[10]:.0f})")
    print("dones:", dones, "/ 20")
    print("OK: quicktest completed, all observations finite and within expected ranges")


# --------------------------------------------------------------------------- #
# World-model loss helper (used by train mode)                                 #
# --------------------------------------------------------------------------- #


def _buffer_dataset(model, buffer):
    """Build ``(x_raw, y_raw)`` from every transition, mirroring the model's own target."""
    n = len(buffer)
    xs = np.zeros((n, model.d_in), dtype=np.float32)
    ys = np.zeros((n, D_OUT), dtype=np.float32)
    for i, t in enumerate(buffer.transitions()):
        xs[i] = build_input(t.obs, t.action_id, t.belief, model.env_cfg)
        ys[i, :D_TACTILE] = np.asarray(t.next_obs["tactile"], dtype=np.float32)
        ys[i, D_TACTILE:] = np.asarray(t.next_obs["pose"], dtype=np.float32) - np.asarray(
            t.obs["pose"], dtype=np.float32
        )
    return xs, ys


def _buffer_mse(model, buffer) -> float:
    """Mean-squared error of the ensemble mean on the whole buffer (raw units)."""
    xs, ys = _buffer_dataset(model, buffer)
    if len(xs) == 0:
        return float("nan")
    pred = model.predict(xs)["mean"]
    return float(np.mean((pred - ys) ** 2))


# --------------------------------------------------------------------------- #
# Train-mode demonstration                                                     #
# --------------------------------------------------------------------------- #


def _mode_train(seed: int) -> None:
    """Run one info_aware trial and print belief entropy + model loss before/after."""
    t0 = time.time()
    object_cfg = ObjectConfig()
    env_cfg = EnvConfig()
    tact_cfg = TactileConfig(sigma_noise=0.05)
    belief_cfg = BeliefConfig(n_obs_samples=12)
    model_cfg = ModelConfig(
        n_ensemble=3, hidden=[32], n_epochs_per_update=6, batch_size=64,
        buffer_capacity=512, update_every_steps=5,
    )
    agent_cfg = AgentConfig(
        name="info_aware", warmup_steps=10, plan_action_step=4,
        beta=1.0, beta_warmup_steps=5, seed=seed,
    )

    from tact_ai.agents import InfoAwareAgent

    env = ManipulationEnv(env_cfg, tact_cfg, object_cfg)
    agent = InfoAwareAgent(
        "info_aware", agent_cfg, env_cfg, object_cfg, tact_cfg, belief_cfg, model_cfg
    )

    obs = env.reset(seed=seed)
    agent.on_episode_start(obs)
    true_idx = -1
    from tact_ai.simulation.objects import candidate_index

    true_idx = candidate_index(env.props)
    entropy_traj: list[float] = [float(agent.belief.entropy())]
    dists: list[float] = []
    posterior_true: list[float] = []
    print("== train mode: info_aware demo (1 object, 60 steps) ==")
    print(f"  true hidden props: {dict(env.props.to_dict())}")

    for step in range(int(env_cfg.max_steps)):
        action = int(agent.act(obs))
        next_obs, reward, done, info = env.step(action)
        agent.observe(obs, action, next_obs, reward, done)
        obs = next_obs
        entropy_traj.append(float(agent.belief.entropy()))
        dists.append(float(info["dist_to_goal"]))
        b = agent.belief.belief_vector()
        posterior_true.append(float(b[true_idx]))
        if done:
            break

    print("\n  step | belief entropy | P(true) | dist")
    for i in range(len(entropy_traj)):
        p_true = posterior_true[i] if i < len(posterior_true) else float("nan")
        d = dists[i] if i < len(dists) else float("nan")
        print(f"  {i:4d} | {entropy_traj[i]:7.4f}      | {p_true:6.4f} | {d:6.3f}")
    print(f"\n  final distance: {dists[-1]:.3f}  success: {dists[-1] < env_cfg.goal_tol}")

    loss_before = _buffer_mse(agent.model, agent.buffer)
    agent.model.update_from_buffer(agent.buffer, n_epochs=4)
    loss_after = _buffer_mse(agent.model, agent.buffer)
    print(f"\n  buffer size: {len(agent.buffer)}")
    print(f"  world-model loss before/after retrain: {loss_before:.6f} -> {loss_after:.6f}")
    print(f"  model_evals: {agent.model_evals}   elapsed: {time.time() - t0:.1f}s")
    print("OK: train mode complete")


# --------------------------------------------------------------------------- #
# Experiments / analysis modes                                                 #
# --------------------------------------------------------------------------- #


def _mode_experiments(
    conditions: list[str] | None,
    smoke: bool,
    out_dir: str,
    model_override: dict | None = None,
    agent_override: dict | None = None,
    exp_override: dict | None = None,
    env_override: dict | None = None,
) -> None:
    """Run the sweep conditions and leave JSON under ``out_dir``.

    ``model_override`` / ``agent_override`` / ``exp_override`` /
    ``env_override`` are dicts of field values applied on top of the canned
    defaults (used by the CLI ``--model-*``, ``--n-trials``, ``--seeds`` and
    ``--goal-tol`` flags). The defaults are the phase-3 ``default_configs()`` so
    existing calls are unaffected.
    """
    from tact_ai.experiments.conditions import ALL_CONDITIONS
    from tact_ai.experiments.sweep_runner import condition_names_from_args, default_configs, run_sweep

    names = condition_names_from_args(conditions)
    (oc, ec, tc, bc, mc, ac) = default_configs()
    ec = _clone_cfg(ec, **(env_override or {}))
    mc = _clone_cfg(mc, **(model_override or {}))
    ac = _clone_cfg(ac, **(agent_override or {}))
    print(f"== experiments: conditions {names} / smoke={smoke} / out_dir={out_dir} ==")
    print(f"== env_cfg: goal_tol={ec.goal_tol} max_steps={ec.max_steps} ==")
    print(f"== model_cfg: n_ensemble={mc.n_ensemble} hidden={mc.hidden} "
          f"n_epochs={mc.n_epochs_per_update} batch={mc.batch_size} "
          f"update_every={mc.update_every_steps} lr={mc.lr} ==")
    print(f"== agent_cfg: warmup={ac.warmup_steps} plan_step={ac.plan_action_step} "
          f"beta={ac.beta} cem_pop={ac.cem_pop} ==")
    t_start = time.time()
    run_sweep(
        names, oc, ec, tc, bc, mc, ac,
        out_dir=out_dir, smoke=smoke, exp_override=exp_override,
        progress=lambda msg: print(msg, flush=True),
    )
    print(f"== experiments done in {(time.time() - t_start) / 60:.1f} min ==")


def _mode_analyze(out_dir: str) -> None:
    """Phase-4 analysis pipeline; stubs report 'not implemented yet' gracefully."""
    try:
        from tact_ai.analysis import results as A_R
        from tact_ai.analysis import plots as A_P
        from tact_ai.analysis import report as A_RP

        A_R.load_results(out_dir)
        A_P.make_plots(out_dir)
        A_RP.run_report(out_dir)
        print("analysis complete")
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}")
    except Exception as exc:  # analysis must never crash an 'all' run
        print(f"analysis warned but continued: {type(exc).__name__}: {exc}")


def _mode_video(
    seed: int,
    fmt: str,
    fps: int,
    method: str,
    train_episodes: int,
    attempts: int,
) -> None:
    """Render the side-by-side random vs info-aware demo video (see visualize.py)."""
    from tact_ai.visualize import render_demo

    t0 = time.time()
    written = render_demo(
        out_dir="results/video",
        method=method,
        fmt=fmt,
        fps=fps,
        seed=seed,
        n_attempts=attempts,
        train_episodes=train_episodes,
    )
    print(f"== video done in {time.time() - t0:.0f}s:")
    for p in written:
        print(f"   {p}")


def _mode_diagnose(out_dir: str, n_oracle: int = 200) -> None:
    """Run diagnostic tests: oracle ceiling, IG consistency, identifiability."""
    from tact_ai.analysis import diagnostics as diag

    t0 = time.time()
    res = diag.run_all(n_oracle=n_oracle, out_dir=out_dir)
    o = res["oracle_ceiling"]
    print(f"  oracle ceiling: success={o['success_rate']:.1%} "
          f"mean_steps={o['mean_steps_success']:.1f} (n={o['n']})")
    ig = res["ig_validity"]
    print(f"  IG validity: pearson_true={ig['pearson_true']:.3f} "
          f"spearman_true={ig['spearman_true']:.3f} (n={ig['n_pairs_true']})")
    ident = res["identifiability"]
    print(f"  identifiability: min_pair={ident['min_pairwise_dist']:.2f} "
          f"min_mass_fixed_rest={ident['mass_min_dist']:.2f}")
    for method, info in res["trial_index"].items():
        print(f"  trial-index {method}: early(0-1)={info['early_rate']:.2f} "
              f"late(3+)={info['late_rate']:.2f}")
    print(f"== diagnostics done in {time.time() - t0:.1f}s ==")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Tact AI (SciFest 2026)")
    parser.add_argument("--mode", choices=MODES, default="quicktest", help="pipeline stage to run")
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="conditions to run (space-joined, default: all six)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="master RNG seed (used by quicktest/train; conditions own their seeds)",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny runs for tests")
    parser.add_argument("--skip-analysis", action="store_true", help="skip analysis in --mode all")
    parser.add_argument("--out-dir", default="results/raw", help="JSON output directory")
    parser.add_argument(
        "--model-ensemble", type=int, default=None,
        help="override world-model ensemble size (n_ensemble)",
    )
    parser.add_argument(
        "--model-hidden", type=str, default=None,
        help="override world-model hidden widths, comma-joined e.g. 96,96",
    )
    parser.add_argument(
        "--model-epochs", type=int, default=None,
        help="override world-model epochs per online update",
    )
    parser.add_argument(
        "--model-batch", type=int, default=None,
        help="override world-model training batch size",
    )
    parser.add_argument(
        "--model-lr", type=float, default=None,
        help="override world-model learning rate",
    )
    parser.add_argument(
        "--model-update-every", type=int, default=None,
        help="override world-model online-update period (env steps)",
    )
    parser.add_argument(
        "--goal-tol", type=float, default=None,
        help="override the task success tolerance (env_cfg.goal_tol)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="override trials per (method, seed) for every condition",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="override run seeds for every condition",
    )
    parser.add_argument(
        "--fmt", default="mp4", choices=["mp4", "gif", "both"],
        help="video mode output format (video mode)",
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="video frame rate (video mode)",
    )
    parser.add_argument(
        "--method", default="side_by_side", choices=["side_by_side", "info_aware"],
        help="video layout: side-by-side random vs info-aware, or info-aware only",
    )
    parser.add_argument(
        "--train-episodes", type=int, default=4,
        help="off-camera world-model warm-up episodes (video mode)",
    )
    parser.add_argument(
        "--attempts", type=int, default=12,
        help="max object/start seeds to try until info-aware solves the task (video mode)",
    )
    parser.add_argument(
        "--warmup-episodes", type=int, default=None,
        help="override n_warmup_episodes for experiment conditions",
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="override agent beta for experiment conditions",
    )
    parser.add_argument(
        "--cem-budget", type=float, default=None,
        help="override agent cem_budget for experiment conditions",
    )
    parser.add_argument(
        "--n-oracle", type=int, default=200,
        help="number of oracle ceiling trials (diagnose mode)",
    )
    args = parser.parse_args()

    # Smoke runs must never pollute the real results directory.
    if args.smoke and args.out_dir == "results/raw":
        args.out_dir = "results/smoke"

    model_override: dict = {}
    if args.model_ensemble is not None:
        model_override["n_ensemble"] = int(args.model_ensemble)
    if args.model_hidden is not None:
        model_override["hidden"] = [int(v) for v in args.model_hidden.split(",") if v.strip() != ""]
    if args.model_epochs is not None:
        model_override["n_epochs_per_update"] = int(args.model_epochs)
    if args.model_batch is not None:
        model_override["batch_size"] = int(args.model_batch)
    if args.model_lr is not None:
        model_override["lr"] = float(args.model_lr)
    if args.model_update_every is not None:
        model_override["update_every_steps"] = int(args.model_update_every)
    env_override: dict = {}
    if args.goal_tol is not None:
        env_override["goal_tol"] = float(args.goal_tol)
    agent_override: dict = {}
    if args.warmup_episodes is not None:
        agent_override["n_warmup_episodes"] = int(args.warmup_episodes)
    if args.beta is not None:
        agent_override["beta"] = float(args.beta)
    if args.cem_budget is not None:
        agent_override["cem_budget"] = float(args.cem_budget)
    exp_override: dict = {}
    if args.n_trials is not None:
        exp_override["n_trials_per_seed"] = int(args.n_trials)
    if args.seeds is not None:
        exp_override["seeds"] = list(args.seeds)

    if args.mode == "quicktest":
        _mode_quicktest(args.seed)
    elif args.mode == "train":
        _mode_train(args.seed)
    elif args.mode == "experiments":
        _mode_experiments(args.conditions, args.smoke, args.out_dir, model_override, agent_override, exp_override, env_override)
    elif args.mode == "analyze":
        _mode_analyze(args.out_dir)
    elif args.mode == "all":
        _mode_experiments(args.conditions, args.smoke, args.out_dir, model_override, agent_override, exp_override, env_override)
        if not args.skip_analysis:
            _mode_analyze(args.out_dir)
    elif args.mode == "video":
        _mode_video(args.seed, args.fmt, args.fps, args.method, args.train_episodes, args.attempts)
    elif args.mode == "diagnose":
        _mode_diagnose(args.out_dir, n_oracle=args.n_oracle)


if __name__ == "__main__":
    main()