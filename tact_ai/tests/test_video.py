"""Tests for the video/visualisation pipeline (matplotlib Agg + ffmpeg/Pillow).

These are deliberately cheap: they exercise figure drawing and encoding on
synthetic records rather than running a full (slow) demo simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tact_ai.config import EnvConfig, ObjectConfig
from tact_ai.visualize import EpisodeRecord, FrameData, _build_figure, encode_gif

_OBJECT = ObjectConfig()
_ENV = EnvConfig(goal_tol=0.20)


def _fake_record(name: str, t: int = 8) -> EpisodeRecord:
    frames: list[FrameData] = []
    for i in range(t):
        y = -0.5 + 0.5 * i / max(t - 1, 1)
        frames.append(
            FrameData(
                pos=np.array([0.4 + 0.3 * i / max(t - 1, 1), y], dtype=float),
                radius=0.4,
                dist=float(np.hypot(0.4 + 0.3 * i / max(t - 1, 1), y)),
                action=int(i % 21),
                entropy=2.0 - 1.8 * i / max(t - 1, 1),
                belief=np.zeros(_OBJECT.n_candidates),
                contact=bool(i % 2 == 0),
                num_step=i,
                done=i == t - 1,
            )
        )
    frames[1].belief[0] = 1.0
    return EpisodeRecord(method=name, frames=frames,
                         true_props={"mu": 0.8, "mass": 0.8, "radius": 0.4},
                         start_pos=np.array([0.4, -0.5]))


def test_build_figure_renders_panel(tmp_path):
    rec = _fake_record("info_aware")
    fig = _build_figure([rec, _fake_record("random")], (0.0, 0.0),
                        _ENV.goal_tol, _ENV, _OBJECT.n_candidates)
    out = tmp_path / "panel.png"
    fig.savefig(out, bbox_inches="tight")
    assert out.stat().st_size > 500


def test_encode_gif_writes_animation(tmp_path):
    from tact_ai.visualize import _build_figure

    frames = []
    for i in range(4):
        rec = _fake_record("info_aware", t=4)
        rec.frames = rec.frames[i:] + rec.frames[:i]  # rotate to vary frames
        fig = _build_figure([rec], (0.0, 0.0), _ENV.goal_tol, _ENV, _OBJECT.n_candidates)
        p = tmp_path / f"f{i}.png"
        fig.savefig(p)
        frames.append(p)
    out = tmp_path / "anim.gif"
    encode_gif(frames, out, fps=4)
    assert out.exists() and out.stat().st_size > 500
    assert len(frames) == 4