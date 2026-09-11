"""Phase-4 headless (Agg) figures: six condition PNGs under ``results/figures``.

Rendering is purely from the aggregated result files — no experiment logic is
re-run. Every figure is saved at ~1200x800 px (figsize 12x8, dpi 110) and the
module only touches matplotlib through the Agg backend so it runs unattended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from tact_ai.analysis.results import ResultFile, TrialRecords, aggregate_records, load_files  # noqa: E402

_K = TypeVar("_K")
T = TypeVar("T")


def _finalize(fig, path: Path) -> None:
    """Save one figure at the fixed size and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _group(files: list[ResultFile], keyfn: Callable[[ResultFile], _K]) -> dict[_K, list[ResultFile]]:
    """Bucket result files by an arbitrary key in stable file order."""
    out: dict[_K, list[ResultFile]] = {}
    for f in files:
        out.setdefault(keyfn(f), []).append(f)
    return out


def _records(files: list[ResultFile]) -> TrialRecords:
    merged: TrialRecords = []
    for f in files:
        merged.extend(f.records)
    return merged


def _success_rate(records: TrialRecords) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if r.success else 0.0 for r in records]))


def _mean(records: TrialRecords, attr: str) -> float:
    vals = [float(getattr(r, attr)) for r in records if getattr(r, attr, None) is not None]
    return float(np.mean(vals)) if vals else 0.0


def _wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for k/n successes; returns (rate, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(centre - half, 0.0), min(centre + half, 1.0))


def _success_ci(records: TrialRecords) -> tuple[float, float, float]:
    """(rate, wilson-lo, wilson-hi) for a record group."""
    n = len(records)
    k = sum(1 for r in records if r.success)
    return _wilson_ci(n, k)


# --------------------------------------------------------------------------- #
# Figure builders                                                             #
# --------------------------------------------------------------------------- #


def _fig_baseline(files: list[ResultFile], figures_dir: Path) -> Path | None:
    cond = [f for f in files if f.condition == "baseline" and f.phase == "train"]
    if not cond:
        return None
    by_method = _group(cond, keyfn=lambda f: f.method)
    methods = ["random", "greedy", "cem", "info_aware"]
    names = [m for m in methods if m in by_method]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    sr, sr_lo, sr_hi, mdist = [], [], [], []
    for m in names:
        recs = _records(by_method[m])
        rate, lo, hi = _success_ci(recs)
        sr.append(rate)
        sr_lo.append(rate - lo)
        sr_hi.append(hi - rate)
        mdist.append(_mean(recs, "final_dist"))
    ax = axes[0]
    ax.bar(
        names, sr, yerr=[sr_lo, sr_hi], capsize=4,
        color="#4C72B0", error_kw={"elinewidth": 1.2},
    )
    for i, rate in enumerate(sr):
        ax.text(i, sr_hi[i] + 0.02, f"{rate:.0%}", ha="center", fontsize=10)
    ax.set_ylabel("success rate"); ax.set_title("baseline: success rate by method")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    ax.bar(names, mdist, color="#DD8452")
    ax.set_ylabel("mean |final dist - goal|"); ax.set_title("baseline: final distance by method")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("baseline condition (methods x 3 seeds x 30 trials, 60-step episodes, "
                 "3 warm-up episodes)", fontsize=12)
    fig.tight_layout()
    path = figures_dir / "fig_baseline.png"
    _finalize(fig, path)
    return path


def _fig_information(files: list[ResultFile], figures_dir: Path, max_k: int) -> Path | None:
    cond = [f for f in files if f.condition in ("information", "baseline")][:1]
    info = [f for f in files if f.condition == "information"]
    base_files = info or cond
    if not base_files:
        return None
    by_method = _group(base_files, keyfn=lambda f: f.method)
    methods = ["random", "greedy", "cem", "info_aware"]
    names = [m for m in methods if m in by_method]
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    max_k = int(max_k)
    for m in names:
        agg = aggregate_records(_records(by_method[m]), max_steps=max_k)
        ent = np.asarray(agg["entropy_at_k"])
        post = np.asarray(agg["posterior_true_at_k"])
        arm = np.asarray(agg["argmax_correct_rate_at_k"])
        t_ent = np.arange(len(ent)); t = np.arange(len(post))
        axes[0].plot(t_ent, ent, label=m)
        axes[1].plot(t, post, label=m)
        axes[2].plot(t, arm, label=m)
    axes[0].set_title("mean belief entropy (nat)"); axes[0].set_xlabel("env step")
    axes[1].set_title("mean posterior mass at true candidate"); axes[1].set_xlabel("env step")
    axes[1].set_ylim(0, 1)
    axes[2].set_title("mean argmax-correct rate"); axes[2].set_xlabel("env step")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("information condition: belief-filter behaviour by method", fontsize=12)
    fig.tight_layout()
    path = figures_dir / "fig_information.png"
    _finalize(fig, path)
    return path


def _fig_robustness(files: list[ResultFile], figures_dir: Path) -> Path | None:
    cond = [f for f in files if f.condition == "robustness"]
    if not cond:
        return None
    by_sigma = _group(cond, keyfn=lambda f: float(f.sigma_noise or 0.0))
    sigmas = sorted(by_sigma)
    by_method = {m: [] for m in ["greedy", "info_aware", "random"]}
    for f in cond:
        if f.method in by_method:
            by_method[f.method].append(f)
    fig, ax = plt.subplots(figsize=(12, 8))
    for m, files_m in by_method.items():
        if not files_m:
            continue
        by_s = _group(files_m, keyfn=lambda f: float(f.sigma_noise or 0.0))
        xs = sorted(by_s)
        ys = [_success_rate(_records(by_s[s])) for s in xs]
        ax.plot(xs, ys, marker="o", label=m)
        ax.fill_between(xs, ys, alpha=0.08) if False else None
    ax.set_xlabel("sensor noise sigma"); ax.set_ylabel("success rate")
    ax.set_ylim(0, 1); ax.set_title("robustness condition: success vs sensor noise")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path = figures_dir / "fig_robustness.png"
    _finalize(fig, path)
    return path


def _fig_generalization(files: list[ResultFile], figures_dir: Path) -> Path | None:
    cond = [f for f in files if f.condition == "generalization"]
    if not cond:
        return None
    methods = ["greedy", "cem", "info_aware"]
    phases = ["train", "eval"]
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.35
    for i, m in enumerate(methods):
        vals = []
        for ph in phases:
            grp = _records([f for f in cond if f.method == m and f.phase == ph])
            vals.append(_success_rate(grp))
        x = np.arange(len(phases)) + i * width - len(methods) * width / 2
        ax.bar(x, vals, width=width, label=m)
    ax.set_xticks(np.arange(len(phases))); ax.set_xticklabels(phases)
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1)
    ax.set_title("generalization condition: train vs held-out success")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    path = figures_dir / "fig_generalization.png"
    _finalize(fig, path)
    return path


def _fig_compute(files: list[ResultFile], figures_dir: Path) -> Path | None:
    cond = [f for f in files if f.condition == "compute"]
    if not cond:
        return None
    by_method = _group(cond, keyfn=lambda f: f.method)
    methods = ["greedy", "cem", "info_aware"]
    names = [m for m in methods if m in by_method]
    fig, ax = plt.subplots(figsize=(12, 8))
    evals = [_mean(_records(by_method[m]), "model_evals") for m in names]
    ax.bar(names, evals, color="#55A868")
    ax.set_yscale("log"); ax.set_ylabel("mean model evals / episode (log)")
    ax.set_title("compute condition: world-model eval budget by method")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(evals):
        ax.text(i, v * 1.05, f"{v:.0f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = figures_dir / "fig_compute.png"
    _finalize(fig, path)
    return path


def _fig_property_scope(files: list[ResultFile], figures_dir: Path) -> Path | None:
    cond = [f for f in files if f.condition == "property_scope"]
    if not cond:
        return None
    scopes = ["narrow", "medium", "wide"]
    methods = ["greedy", "info_aware"]
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.3
    for i, m in enumerate(methods):
        vals = []
        for scope in scopes:
            vals.append(_success_rate(_records([f for f in cond if f.method == m and f.tag == scope])))
        x = np.arange(len(scopes)) + i * width - width / 2
        ax.bar(x, vals, width=width, label=m)
    ax.set_xticks(np.arange(len(scopes))); ax.set_xticklabels(scopes)
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1)
    ax.set_title("property_scope condition: success vs unknown-property spread")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    path = figures_dir / "fig_property_scope.png"
    _finalize(fig, path)
    return path


def _fig_trial_index(files: list[ResultFile], figures_dir: Path) -> Path | None:
    """Success rate by trial index using the expanded baseline records."""
    cond = [f for f in files if f.condition == "baseline" and f.phase == "train"]
    if not cond:
        return None
    methods = ["random", "greedy", "cem", "info_aware"]
    fig, ax = plt.subplots(figsize=(12, 8))
    for m in methods:
        recs = _records([f for f in cond if f.method == m])
        if not recs:
            continue
        by_idx: dict[int, TrialRecords] = {}
        for r in recs:
            by_idx.setdefault(int(r.trial_idx), []).append(r)
        xs = sorted(by_idx)
        label = [i if i < 9 else "9+" for i in xs]
        xs_disp = [i if i < 9 else 9 for i in xs]
        ys = [_success_rate(by_idx[i]) for i in xs]
        ax.plot(xs_disp, ys, marker="o", label=m, linewidth=1.8)
    ax.set_xlabel("trial index within seed"); ax.set_ylabel("success rate")
    ax.set_ylim(0, 1); ax.set_title("baseline: success rate by trial index (3 warm-up episodes precede trial 0)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path = figures_dir / "fig_trial_index.png"
    _finalize(fig, path)
    return path


def _tag_value(tag: str, prefix: str) -> float:
    """Parse ``beta0.5`` / ``budget5000`` style tags to their numeric value."""
    for key in (prefix,):
        if tag.startswith(key):
            try:
                return float(tag[len(key):])
            except (TypeError, ValueError):
                pass
    return float("nan")


def _fig_beta_sweep(files: list[ResultFile], figures_dir: Path) -> Path | None:
    """Success rate and identification speed vs the information-gain weight beta."""
    cond = [f for f in files if f.condition == "beta_sweep"]
    if not cond:
        return None
    by_tag = _group(cond, keyfn=lambda f: str(f.tag))
    betas = sorted({_tag_value(t, "beta") for t in by_tag})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    xs = betas
    rates, los, his, conc = [], [], [], []
    max_steps = max([f.max_steps for f in cond if f.max_steps] or [60])
    for b in xs:
        recs = _records(by_tag[f"beta{b}"])
        rate, lo, hi = _success_ci(recs)
        rates.append(rate); los.append(rate - lo); his.append(hi - rate)
        conc.append(aggregate_records(recs, max_steps=max_steps)["mean_first_concentration_step"])
    axes[0].errorbar(xs, rates, yerr=[los, his], marker="o", capsize=4, color="#4C72B0")
    axes[0].set_xlabel("information-gain weight $\\beta$"); axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1); axes[0].set_title("beta sweep: success rate")
    axes[0].grid(alpha=0.3)
    axes[1].plot(xs, conc, marker="o", color="#DD8452")
    axes[1].set_xlabel("information-gain weight $\\beta$")
    axes[1].set_ylabel("first concentration step")
    axes[1].set_title("beta sweep: identification speed ($P$ true > 0.99)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("beta_sweep condition (info-aware x 2 seeds x 20 trials)", fontsize=12)
    fig.tight_layout()
    path = figures_dir / "fig_beta_sweep.png"
    _finalize(fig, path)
    return path


def _fig_cem_budget(files: list[ResultFile], figures_dir: Path) -> Path | None:
    """Success rate vs the CEM per-decision forward-pass budget."""
    cond = [f for f in files if f.condition == "cem_budget"]
    if not cond:
        return None
    by_tag = _group(cond, keyfn=lambda f: str(f.tag))
    budgets = sorted({_tag_value(t, "budget") for t in by_tag})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    xs = budgets
    rates, los, his, evals = [], [], [], []
    for b in xs:
        recs = _records(by_tag[f"budget{int(b)}"])
        rate, lo, hi = _success_ci(recs)
        rates.append(rate); los.append(rate - lo); his.append(hi - rate)
        evals.append(_mean(recs, "model_evals"))
    axes[0].errorbar(xs, rates, yerr=[los, his], marker="o", capsize=4, color="#4C72B0")
    axes[0].set_xscale("log"); axes[0].set_xlabel("CEM forward-pass budget / decision")
    axes[0].set_ylabel("success rate"); axes[0].set_ylim(0, 1)
    axes[0].set_title("CEM budget: success rate")
    axes[0].grid(alpha=0.3)
    axes[1].plot(xs, evals, marker="o", color="#55A868")
    axes[1].set_xscale("log"); axes[1].set_xlabel("CEM forward-pass budget / decision")
    axes[1].set_ylabel("mean model evals / episode"); axes[1].set_yscale("log")
    axes[1].set_title("CEM budget: realised model evals")
    axes[1].grid(alpha=0.3)
    fig.suptitle("cem_budget condition (cem x 2 seeds x 20 trials)", fontsize=12)
    fig.tight_layout()
    path = figures_dir / "fig_cem_budget.png"
    _finalize(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def make_plots(out_dir: str = "results/raw", figures_dir: str = "results/figures") -> list[str]:
    """Render the six condition figures under ``figures_dir``; returns paths.

    Figures whose condition has no data on disk are skipped. Safe to call with
    missing results — empty conditions are silently omitted.
    """
    figures = Path(figures_dir)
    files = load_files(out_dir)
    max_k = max([f.max_steps for f in files if f.max_steps] or [60])
    builders = [
        _fig_baseline,
        lambda fs, fd: _fig_information(fs, fd, max_k),
        _fig_robustness,
        _fig_generalization,
        _fig_compute,
        _fig_property_scope,
        _fig_trial_index,
        _fig_beta_sweep,
        _fig_cem_budget,
    ]
    written: list[str] = []
    for build in builders:
        path = build(files, figures)
        if path is not None:
            written.append(str(path))
    return written