"""Render the Tact AI simulation to a video.

Builds a side-by-side animation of the disk-pushing task: on the left a RANDOM
policy, on the right the PROPOSED information-aware policy (the learned world
model + physics-based belief filter). Both act on the *same* object and start
pose (same env seed). The info-aware agent is pre-warmed off-camera across a few
training episodes (exactly as in the experiments); the world model then operates
believing the object's physical properties to be unknown.

Each frame shows:

- top-down scene: goal (green ring, dashed = success tolerance), disk object,
  the latest push (orange arrow at the contact point), and the motion trail;
- live belief bars: posterior probability over the 18 candidate
  (mu, mass, radius) cells, with the true cell highlighted;
- read-outs: method, step, distance-to-goal, belief entropy.

Output: MP4 (via ffmpeg) and/or animated GIF (via Pillow) plus the raw PNG
frames (handy for posters). Fully headless (matplotlib Agg).
"""

from __future__ import annotations

import copy
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tact_ai.config import (  # noqa: E402
    AgentConfig,
    BeliefConfig,
    EnvConfig,
    ModelConfig,
    ObjectConfig,
    TactileConfig,
)
from tact_ai.simulation.environment import ManipulationEnv  # noqa: E402
from tact_ai.simulation.objects import candidate_index, props_from_tuple  # noqa: E402


@dataclass
class FrameData:
    """Everything needed to draw one step of one method."""

    pos: np.ndarray          # absolute (x, y) of the disk centre
    radius: float            # true radius (audience knows, robot doesn't)
    dist: float              # distance to goal
    action: int              # action id taken this step
    entropy: float           # belief entropy after this step (nats)
    belief: np.ndarray       # posterior over candidates
    contact: bool            # tactile in-contact flag
    num_step: int
    done: bool


@dataclass
class EpisodeRecord:
    """Recorded run of one method on one object."""

    method: str
    frames: list[FrameData]
    true_props: dict
    start_pos: np.ndarray


def _run_on(*, agent, env, seed: int, max_steps: int) -> EpisodeRecord:
    """Run one episode capturing FrameData per step (freezing after done)."""
    obs = env.reset(seed=seed)
    agent.on_episode_start(obs)
    true_props = dict(env.props.to_dict())
    radius = float(env.props.radius)
    start_pos = np.asarray(obs["pose"][0:2], dtype=float)
    frames: list[FrameData] = []
    done = False
    for step in range(int(max_steps)):
        if not done:
            action = int(agent.act(obs))
            next_obs, reward, done, info = env.step(action)
            agent.observe(obs, action, next_obs, reward, done)
            obs = next_obs
        else:
            action = frames[-1].action  # freeze: reuse the last action
        pos = np.asarray(obs["pose"][0:2], dtype=float)
        frames.append(
            FrameData(
                pos=pos,
                radius=radius,
                dist=float(obs["pose"][2]),
                action=action,
                entropy=float(agent.belief.entropy()),
                belief=np.asarray(agent.belief.belief_vector(), dtype=float),
                contact=bool(obs["tactile"][10] > 0.5),
                num_step=step,
                done=done,
            )
        )
    return EpisodeRecord(method=agent.name, frames=frames, true_props=true_props, start_pos=start_pos)


# --------------------------------------------------------------------------- #
# Drawing                                                                      #
# --------------------------------------------------------------------------- #

def _draw_scene(ax, rec: EpisodeRecord, goal: tuple[float, float], goal_tol: float, env_cfg) -> None:
    props = rec.true_props
    ax.set_aspect("equal", adjustable="box")
    # goal
    ax.add_patch(plt.Circle(goal, goal_tol, color="#43a047", alpha=0.18, zorder=1))
    ax.plot(*goal, marker="+", color="#2e7d32", ms=10, mew=2, zorder=2)
    ax.add_patch(plt.Circle(goal, goal_tol * 2.2, fill=False, ls="--", color="#2e7d32", lw=1.0, zorder=2))
    # motion trail
    trail = np.stack([f.pos for f in rec.frames])
    ax.plot(trail[:, 0], trail[:, 1], alpha=0.35, color="#1565c0", lw=1.2, zorder=3)
    # last step render
    f = rec.frames[-1]
    ax.add_patch(plt.Circle(f.pos, f.radius, color="#bdbdbd", alpha=0.85, zorder=4))
    ax.add_patch(plt.Circle(f.pos, f.radius, fill=False, color="#424242", lw=1.2, zorder=5))
    ax.plot(*f.pos, marker="o", color="#424242", ms=4, zorder=5)
    if f.action >= 0 and f.contact:
        theta_c, phi_deg, force = env_cfg.decode_action(f.action)
        n_in = np.array([-math.cos(theta_c), -math.sin(theta_c)])
        t_hat = np.array([-math.sin(theta_c), math.cos(theta_c)])
        phi = math.radians(float(phi_deg))
        fdir = n_in * math.cos(phi) + t_hat * math.sin(phi)
        contact_pt = f.pos + f.radius * np.array([math.cos(theta_c), math.sin(theta_c)])
        scale = float(force) / env_cfg.max_force_n
        ax.annotate(
            "", xy=contact_pt + fdir * scale * 0.55, xytext=contact_pt,
            arrowprops=dict(arrowstyle="-|>", color="#e65100", lw=2.4),
            zorder=6,
        )
    sup = " (already reached goal)" if f.done else ""
    ax.set_title(
        f"{rec.method.upper().replace('_', ' ')}  |  step {f.num_step:2d}  |  "
        f"dist-to-goal {f.dist:.2f}  |  entropy {f.entropy:.2f} nats{sup}",
        fontsize=10,
    )
    ax.set_xlabel("x (world units)", fontsize=8)
    ax.set_ylabel("y (world units)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.annotate(
        f"true object (unknown to robot):  mu={props['mu']:.2f}  "
        f"mass={props['mass']:.1f}  radius={props['radius']:.1f}",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="#555555",
    )


def _draw_beliefs(ax, rec: EpisodeRecord, n_candidates: int) -> None:
    f = rec.frames[-1]
    true_idx = candidate_index(props_from_tuple(
        (rec.true_props["mu"], rec.true_props["mass"], rec.true_props["radius"])))
    cols = ["#90a4ae"] * n_candidates
    cols[true_idx] = "#2e7d32"
    ax.bar(np.arange(n_candidates), f.belief, color=cols, width=0.85)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, n_candidates - 0.4)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6)
    ax.set_xlabel("candidate cell (mu x mass x radius)", fontsize=7)
    ax.set_title("belief over object properties (green = true cell)", fontsize=8)


def _build_figure(recs: list[EpisodeRecord], goal, goal_tol, env_cfg, n_candidates: int):
    """Panel per recorded method: one scene + belief inset each."""
    n_panels = len(recs)
    fig, axs = plt.subplots(1, n_panels, figsize=(6.6 * n_panels, 5.4), dpi=110)
    if n_panels == 1:
        axs = [axs]

    # identical axis limits for both panels so the two agents are comparable
    all_pos = np.concatenate([np.stack([f.pos for f in r.frames]) for r in recs])
    bounds = np.array([
        min(all_pos[:, 0].min(), goal[0]) - 0.6,
        max(all_pos[:, 0].max(), goal[0]) + 0.6,
        min(all_pos[:, 1].min(), goal[1]) - 0.6,
        max(all_pos[:, 1].max(), goal[1]) + 0.6,
    ])
    for ax, rec in zip(axs, recs):
        _draw_scene(ax, rec, goal, goal_tol, env_cfg)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.30)
    for ax, rec in zip(axs, recs):
        ax_box = ax.get_position()
        bel_ax = fig.add_axes([ax_box.x0 + ax_box.width * 0.03, 0.045,
                               ax_box.width * 0.94, 0.085])
        _draw_beliefs(bel_ax, rec, n_candidates)
    fig.suptitle("Tact AI - SciFest 2026: what is the most valuable next touch?",
                 fontsize=12)
    return fig


# --------------------------------------------------------------------------- #
# Encoding                                                                     #
# --------------------------------------------------------------------------- #

def encode_mp4(frame_files: list[Path], out: Path, fps: int) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; use --format gif instead")
    pattern = (frame_files[0].parent / "frame_%04d.png").as_posix()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-crf", "18", "-preset", "medium", out.as_posix()],
        check=True,
    )
    return out


def encode_gif(frame_files: list[Path], out: Path, fps: int, scale: float = 0.6) -> Path:
    imgs = []
    for p in frame_files:
        im = Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE)
        if scale != 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        imgs.append(im)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=1000 / fps, loop=0, optimize=True)
    return out


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def render_demo(
    *,
    out_dir: str = "results/video",
    method: str = "side_by_side",
    fmt: str = "mp4",
    fps: int = 5,
    seed: int = 0,
    n_attempts: int = 12,
    train_episodes: int = 4,
    warmup_steps: int = 40,
    agent_cfg_override: dict | None = None,
    progress: Callable[[str], None] = print,
) -> list[Path]:
    """Render and encode the demo video; returns written file paths.

    ``method``: "side_by_side" (random vs info_aware) or "info_aware"
    (single-panel). ``fmt``: "mp4", "gif" or "both".
    """
    import matplotlib.image as mpimg
    from tact_ai.agents import InfoAwareAgent, RandomAgent

    object_cfg = ObjectConfig()
    env_cfg = EnvConfig(goal_tol=0.20)  # same task tolerance the experiments used
    tact_cfg = TactileConfig(sigma_noise=0.05)
    belief_cfg = BeliefConfig(n_obs_samples=12)
    model_cfg = ModelConfig(  # mirrors the experiment config (5x[192,192], lr 5e-4)
        n_ensemble=5, hidden=[192, 192], activation="tanh",
        lr=5e-4, n_epochs_per_update=20, batch_size=128,
        buffer_capacity=3000, update_every_steps=3,
    )
    agent_cfg = AgentConfig(
        name="info_aware", warmup_steps=warmup_steps, plan_action_step=4,
        beta=1.0, seed=seed,
        **(agent_cfg_override or {}),
    )
    rng = np.random.default_rng(seed)

    def make_info_agent(agent_seed: int) -> InfoAwareAgent:
        ac = copy.deepcopy(agent_cfg)
        ac.seed = agent_seed
        return InfoAwareAgent("info_aware", ac, env_cfg, object_cfg, tact_cfg, belief_cfg, model_cfg)

    # Off-camera training: the robot learns a world model across objects.
    info_agent = make_info_agent(seed)
    progress(f"[train] warming world model over {train_episodes} off-camera episodes...")
    for ep in range(train_episodes):
        env = ManipulationEnv(env_cfg, tact_cfg, object_cfg)
        _run_on(agent=info_agent, env=env, seed=int(rng.integers(0, 1_000_000)),
                max_steps=env_cfg.max_steps)
    progress(f"[train] done; buffer={len(info_agent.buffer)} model_evals={info_agent.model_evals}")

    # Try object+start seeds until the info-aware agent solves the task. The
    # agent keeps learning online across attempts (experience is experience);
    # the filmed episode is simply the first successful one.
    chosen_seed: int | None = None
    chosen_records: list[EpisodeRecord] = []
    for s in range(n_attempts):
        env_seed = int(rng.integers(0, 1_000_000))
        env = ManipulationEnv(env_cfg, tact_cfg, object_cfg)
        rec_info = _run_on(agent=info_agent, env=env, seed=env_seed,
                           max_steps=env_cfg.max_steps)
        success = rec_info.frames[-1].dist < env_cfg.goal_tol
        progress(
            f"[attempt {s + 1}/{n_attempts}] seed={env_seed} "
            f"final_dist={rec_info.frames[-1].dist:.2f} "
            f"entropy={rec_info.frames[-1].entropy:.2f} success={success}"
        )
        if success:
            chosen_seed = env_seed
            chosen_records = [rec_info]
            if method == "side_by_side":
                ac_r = copy.deepcopy(agent_cfg)
                ac_r.warmup_steps = warmup_steps
                rand_agent = RandomAgent("random", ac_r, env_cfg, object_cfg,
                                         tact_cfg, belief_cfg, model_cfg)
                env3 = ManipulationEnv(env_cfg, tact_cfg, object_cfg)
                rec_random = _run_on(agent=rand_agent, env=env3, seed=env_seed,
                                     max_steps=env_cfg.max_steps)
                chosen_records.insert(0, rec_random)
            break

    if chosen_seed is None:
        raise RuntimeError(
            f"no success found in {n_attempts} attempts for the current config; "
            "raise warmup_steps/train_episodes or relax the task"
        )

    # Build frames.
    recs = chosen_records
    n_candidates = object_cfg.n_candidates
    goal = tuple(env_cfg.goal)
    frame_dir = Path(out_dir) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_files: list[Path] = []
    n_steps = max(len(r.frames) for r in recs)
    with tempfile.TemporaryDirectory(dir=frame_dir) as tmp:
        tmpd = Path(tmp)
        for i in range(n_steps):
            one_steps = [
                EpisodeRecord(r.method, [r.frames[min(i, len(r.frames) - 1)]], r.true_props, r.start_pos)
                for r in recs
            ]
            fig = _build_figure(one_steps, goal, env_cfg.goal_tol, env_cfg, n_candidates)
            fp = tmpd / f"frame_{i:04d}.png"
            fig.savefig(fp)  # fixed canvas -> identical frame sizes for encoding
            plt.close(fig)
            frame_files.append(fp)
        if fmt in ("mp4", "both"):
            mp4 = Path(out_dir) / "tact_ai_demo.mp4"
            encode_mp4(sorted(frame_files), mp4, fps)
            progress(f"[encode] wrote {mp4} ({mp4.stat().st_size // 1024} KiB)")
        if fmt in ("gif", "both"):
            gif = Path(out_dir) / "tact_ai_demo.gif"
            encode_gif(sorted(frame_files), gif, fps)
            progress(f"[encode] wrote {gif} ({gif.stat().st_size // 1024} KiB)")
        # keep raw frames for posters (copied while the temp dir still exists)
        for i, src in enumerate(sorted(frame_files)):
            shutil.copy(src, frame_dir / f"frame_{i:04d}.png")
    out_paths: list[Path] = []
    if fmt in ("mp4", "both"):
        out_paths.append(Path(out_dir) / "tact_ai_demo.mp4")
    if fmt in ("gif", "both"):
        out_paths.append(Path(out_dir) / "tact_ai_demo.gif")
    out_paths = [p for p in out_paths if p.exists()]
    progress(f"[demo] selected seed {chosen_seed}; true props: {recs[-1].true_props}")
    return out_paths