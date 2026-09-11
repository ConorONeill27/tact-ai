"""Diagnostic experiments: oracle ceiling, IG validity, identifiability, trial-index.

These produce result JSON and PNG figures under ``results/diagnostics/`` and are
called from ``--mode diagnose``.  Every function is self-contained, writes its
own files, and returns a summary dict for the CLI.  Only numpy/matplotlib /
project internals are used (no external stats dependency; Pearson/Spearman are
implemented directly here).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tact_ai.config import BeliefConfig, EnvConfig, ObjectConfig, TactileConfig
from tact_ai.model.belief_filter import BeliefFilter
from tact_ai.model.features import D_OUT, D_TACTILE
from tact_ai.simulation import physics
from tact_ai.simulation.environment import ManipulationEnv
from tact_ai.simulation.objects import candidate_index, ObjectProps, props_from_tuple

_DIAG_DIR = Path("results/diagnostics")


def _save(fig, name: str) -> Path:
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    p = _DIAG_DIR / name
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _write_json(name: str, payload: dict) -> Path:
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    p = _DIAG_DIR / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Oracle ceiling                                                               #
# --------------------------------------------------------------------------- #


def oracle_ceiling(n_attempts: int = 200, goal_tol: float = 0.20) -> dict:
    """Greedy perfect-information planner using the analytical physics.

    At each step the oracle simulates all 216 actions with the *true* hidden
    properties and picks the one that minimises distance to goal.  This
    establishes the theoretical task ceiling given the action discretisation
    and 60-step episode budget.
    """
    env_cfg = EnvConfig(goal_tol=goal_tol)
    obj_cfg = ObjectConfig()
    tact_cfg = TactileConfig()
    successes, all_steps, succ_steps = 0, [], []
    for attempt in range(n_attempts):
        env = ManipulationEnv(env_cfg, tact_cfg, obj_cfg)
        obs = env.reset(seed=attempt)
        for _ in range(int(env_cfg.max_steps)):
            best_a, best_d = -1, float("inf")
            for a in range(int(env_cfg.n_actions)):
                tc, phi, F = env_cfg.decode_action(a)
                _, np_ = physics.step_disk(
                    env.pose, env.props.radius, env.props.mass, env.props.mu,
                    tc, phi, F, env_cfg.dt, env_cfg.damping, env_cfg.rot_damping,
                )
                d = np.hypot(np_.x, np_.y)
                if d < best_d:
                    best_d = d
                    best_a = a
            obs, _, done, info = env.step(best_a)
            if done:
                break
        ok = info["dist_to_goal"] < goal_tol
        successes += int(ok)
        all_steps.append(int(env.n_steps))
        if ok:
            succ_steps.append(int(env.n_steps))

    rate = successes / max(1, n_attempts)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["oracle"], [rate], color="#55A868")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(f"oracle ceiling (n={n_attempts}, perfect info)")
    for i, v in enumerate([rate]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_oracle_ceiling.png")

    result = {
        "success_rate": rate,
        "n": n_attempts,
        "mean_steps_all": float(np.mean(all_steps)) if all_steps else 0.0,
        "mean_steps_success": float(np.mean(succ_steps)) if succ_steps else 0.0,
    }
    _write_json("oracle_ceiling.json", result)
    return result


# --------------------------------------------------------------------------- #
# Trial-index analysis                                                         #
# --------------------------------------------------------------------------- #


def trial_index_analysis(out_dir: str = "results/raw") -> dict:
    """Success rate by trial index from existing baseline JSON files.

    Reveals the warm-up artifact: the first ~2 episodes are fully random
    because the replay buffer hasn't reached ``model_cfg.batch_size`` yet
    (128 transitions inside 60-step episodes).
    """
    from tact_ai.analysis.results import load_files

    files = load_files(out_dir)
    baseline = [f for f in files if f.condition == "baseline" and f.phase == "train"]
    by_method: dict[str, dict[int, list[bool]]] = {}
    for f in baseline:
        by_method.setdefault(f.method, {})
        for r in f.records:
            by_method[f.method].setdefault(int(r.trial_idx), []).append(bool(r.success))

    result: dict[str, dict] = {}
    for method, idx_map in by_method.items():
        idxs = sorted(idx_map)
        rates = {i: float(np.mean(idx_map[i])) for i in idxs}
        early = [rates[i] for i in idxs if i <= 1]
        late = [rates[i] for i in idxs if i >= 3]
        result[method] = {
            "rates_by_trial": {str(k): v for k, v in rates.items()},
            "early_rate": float(np.mean(early)) if early else 0.0,
            "late_rate": float(np.mean(late)) if late else 0.0,
        }
    _write_json("trial_index.json", result)
    return result


# --------------------------------------------------------------------------- #
# Identifiability                                                              #
# --------------------------------------------------------------------------- #


def identifiability() -> dict:
    """Pairwise identifiability of candidate pairs using a probe suite.

    For every pair of candidates we compute the total standardised measurement
    distance over a probe suite (several poses x representative actions),
    using the exact belief-filter likelihood metric (per-dim sigmas).  A small
    distance means the two hidden-property combinations produce nearly
    indistinguishable observations, i.e. are weakly identifiable.  Separately
    we report the minimum distance among pairs differing only in *mass*
    (identical mu and radius) - the identity test for each property axis.
    """
    obj_cfg = ObjectConfig()
    belief_cfg = BeliefConfig(n_obs_samples=24)
    env_cfg = EnvConfig(goal_tol=0.20)
    tact_cfg = TactileConfig(sigma_noise=0.05)
    bf = BeliefFilter(obj_cfg, belief_cfg, env_cfg, tact_cfg, seed=0)
    grid = list(bf.grid)
    n_c = bf.n_candidates
    C = bf.env_cfg
    sigma = np.full(D_OUT, belief_cfg.obs_noise)
    sigma[D_TACTILE:] = belief_cfg.pose_obs_noise

    probe_poses = [
        (1.2, 0.0), (-1.0, 0.6), (0.3, -1.3), (0.0, 1.0), (-0.6, -1.2),
    ]
    # Probe every contact angle at a subset of push modes and forces so that
    # both normal-load (mass/geometry) and tangential/slip (friction) channels
    # are exercised: 9 angles x 3 modes x 2 forces = 54 actions.
    n_per_angle = C.n_modes * C.n_forces
    probe_actions = []
    for angle_idx in range(0, C.n_contact_angles, 4):
        base = angle_idx * n_per_angle
        probe_actions.extend(range(base, base + n_per_angle))

    meas = []  # (C, n_probes, 15)
    for px, py in probe_poses:
        pose_xy = np.array([px, py], dtype=np.float64)
        for a in probe_actions:
            t_all, d_all = bf._predict_all(int(a), pose_xy)
            meas.append(np.concatenate([t_all, d_all], axis=1))
    meas = np.stack(meas, axis=1)  # (C, n_probes, 15)
    n_probes = meas.shape[1]
    flat = meas.reshape(n_c, -1)  # (C, n_probes*15)
    sigma_tiled = np.tile(sigma, n_probes)

    dist_mat = np.zeros((n_c, n_c), dtype=np.float64)
    mu_dists, mass_dists, all_dists = [], [], []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            d = float(np.sqrt(np.sum(((flat[i] - flat[j]) / sigma_tiled) ** 2)))
            dist_mat[i, j] = dist_mat[j, i] = d
            all_dists.append(d)
            if grid[i][2] != grid[j][2]:
                continue  # only compare within the same radius
            if grid[i][0] != grid[j][0]:
                mu_dists.append(d)
            elif grid[i][1] != grid[j][1]:
                mass_dists.append(d)

    min_all = float(np.min(all_dists)) if all_dists else 0.0
    min_mass = float(np.min(mass_dists)) if mass_dists else 0.0
    min_mu = float(np.min(mu_dists)) if mu_dists else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_mat, aspect="auto", cmap="viridis")
    ax.set_title("candidate pair distance (lower = harder to distinguish)")
    ax.set_xlabel("candidate index")
    ax.set_ylabel("candidate index")
    fig.colorbar(im, ax=ax, label="total standardised distance")
    fig.tight_layout()
    _save(fig, "fig_identifiability.png")

    result = {
        "min_pairwise_dist": min_all,
        "mass_min_dist": min_mass,
        "mu_min_dist": min_mu,
        "mean_mass_dist": float(np.mean(mass_dists)) if mass_dists else 0.0,
        "mean_mu_dist": float(np.mean(mu_dists)) if mu_dists else 0.0,
        "n_pairs": len(all_dists),
    }
    _write_json("identifiability.json", result)
    return result


# --------------------------------------------------------------------------- #
# Information-gain consistency                                                 #
# --------------------------------------------------------------------------- #


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xm * xm).sum() * (ym * ym).sum()))
    return float((xm * ym).sum() / denom) if denom > 0 else 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        r = np.empty_like(order)
        prev = prev_val = None
        n = 0
        idx = 0
        while idx < len(v):
            start = idx
            while idx + 1 < len(v) and v[order[idx + 1]] == v[order[idx]]:
                idx += 1
            avg = (start + idx) / 2.0
            for k in range(start, idx + 1):
                r[order[k]] = avg
            idx += 1
        return r + 1.0

    return _pearson(ranks(np.asarray(x, dtype=np.float64)), ranks(np.asarray(y, dtype=np.float64)))


def info_gain_consistency(n_episodes: int = 6, actions_per_state: int = 12) -> dict:
    """Cross-sectional validation: predicted IG vs realised entropy reduction.

    During genuine episodes the belief filter's predicted info gain for a
    suite of actions is compared with the entropy reduction actually produced
    (a) when the hypothetical measurement is generated for the *true*
    candidate, and (b) when it is generated for the candidate drawn from the
    current posterior (i.e. the estimator's own self-consistency).  The Pearson
    correlation of the true-candidate pair is the headline calibration check.
    """
    obj_cfg = ObjectConfig()
    belief_cfg = BeliefConfig(n_obs_samples=24)
    env_cfg = EnvConfig(goal_tol=0.20)
    tact_cfg = TactileConfig(sigma_noise=0.05)
    sigma = np.full(D_OUT, belief_cfg.obs_noise)
    sigma[D_TACTILE:] = belief_cfg.pose_obs_noise
    actions = np.arange(0, env_cfg.n_actions, 12, dtype=int)[:actions_per_state]
    rng = np.random.default_rng(7)

    pred_true, real_true, pred_post, real_post = [], [], [], []
    for ep in range(n_episodes):
        bf = BeliefFilter(obj_cfg, belief_cfg, env_cfg, tact_cfg, seed=100 + ep)
        env = ManipulationEnv(env_cfg, tact_cfg, obj_cfg)
        obs = env.reset(seed=2000 + ep)
        true_idx = candidate_index(env.props, obj_cfg)
        for _ in range(int(env_cfg.max_steps)):
            h0 = bf.entropy()
            if h0 > belief_cfg.max_entropy_floor + 0.02 and h0 < 2.8:
                obs_dict = {
                    "pose": np.array(
                        [obs["pose"][0], obs["pose"][1], float(np.hypot(obs["pose"][0], obs["pose"][1]))],
                        dtype=np.float32,
                    ),
                    "tactile": np.asarray(obs["tactile"], dtype=np.float32),
                    "action": int(obs["action"]),
                }
                pose_xy = np.asarray(obs["pose"])[:2].astype(np.float64)
                ig = bf.expected_info_gain_batch(actions, obs_dict, rng=rng)
                for aidx, a in enumerate(actions):
                    if a < 0 or a >= env_cfg.n_actions:
                        continue
                    predicted = float(ig[aidx])
                    if predicted <= 1e-9:
                        continue
                    # realised for the TRUE candidate
                    t_all, d_all = bf._predict_all(int(a), pose_xy)
                    mean = np.concatenate([t_all, d_all], axis=1)
                    y_true = mean[true_idx] + rng.normal(0.0, sigma)
                    ll = bf._likelihood_grid(y_true, int(a), pose_xy, rng)
                    bf_c = BeliefFilter(obj_cfg, belief_cfg, env_cfg, tact_cfg, seed=21)
                    bf_c.log_post = bf.log_post.copy()
                    bf_c.log_post = bf_c._log_prob(bf_c.log_post + ll)
                    pred_true.append(predicted)
                    real_true.append(float(max(h0 - bf_c.entropy(), 0.0)))
                    # realised for the posterior-sampled candidate
                    if belief_cfg.prior == "uniform":
                        c = int(rng.choice(bf.n_candidates, p=bf.belief_vector()))
                        y_post = mean[c] + rng.normal(0.0, sigma)
                        ll_p = bf._likelihood_grid(y_post, int(a), pose_xy, rng)
                        bf_d = BeliefFilter(obj_cfg, belief_cfg, env_cfg, tact_cfg, seed=22)
                        bf_d.log_post = bf.log_post.copy()
                        bf_d.log_post = bf_d._log_prob(bf_d.log_post + ll_p)
                        pred_post.append(predicted)
                        real_post.append(float(max(h0 - bf_d.entropy(), 0.0)))
            action = int(rng.integers(0, env_cfg.n_actions))
            next_obs, _, done, _ = env.step(action)
            bf.update_obs(
                next_obs["tactile"], next_obs["pose"] - obs["pose"],
                obs, next_obs, action, rng=rng,
            )
            obs = next_obs
            if done:
                break

    pr_true = _pearson(pred_true, real_true)
    sr_true = _spearman(pred_true, real_true)
    pr_post = _pearson(pred_post, real_post) if pred_post else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, (np_, nr), title in [
        (axes[0], (pred_true, real_true), "measurement from true candidate"),
        (axes[1], (pred_post, real_post), "measurement from posterior draw"),
    ]:
        if np_:
            ax.scatter(np_, nr, alpha=0.45, s=12, edgecolors="none")
        lim = max(float(np.max(pred_true)) if pred_true else 0.1,
                  float(np.max(real_true)) if real_true else 0.1, 0.1) * 1.15
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("predicted info gain (nat)")
        rf = _pearson(np_, nr) if np_ else 0.0
        ax.set_title(f"{title}\nr={rf:.3f}  (n={len(np_)})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    axes[0].set_ylabel("realised entropy reduction (nat)")
    fig.suptitle("information-gain calibration", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_ig_validity.png")

    result = {
        "pearson_true": pr_true,
        "spearman_true": sr_true,
        "pearson_posterior": pr_post,
        "n_pairs_true": len(pred_true),
        "n_pairs_posterior": len(pred_post),
        "mean_predicted_ig": float(np.mean(pred_true)) if pred_true else 0.0,
        "mean_realised_dh": float(np.mean(real_true)) if real_true else 0.0,
    }
    _write_json("ig_validity.json", result)
    return result


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


def run_all(n_oracle: int = 200, out_dir: str = "results/raw") -> dict:
    """Run every diagnostic and return their summaries (for ``--mode diagnose``)."""
    return {
        "oracle_ceiling": oracle_ceiling(n_attempts=n_oracle),
        "trial_index": trial_index_analysis(out_dir),
        "identifiability": identifiability(),
        "ig_validity": info_gain_consistency(),
    }