"""Deep-ensemble MLP tactile world model.

A dynamics model ``P(next_features | tactile, pose, action, belief)`` where the
next features are ``[next_tactile(12), pose_delta(3)]`` with ``pose_delta =
(ddx, ddy, ddist)``. The belief vector (posterior over the candidate grid of
hidden properties ``h``) is part of the input, so the learned model can use any
belief-correlated structure the (phase-2) physics filter does not.

Each ensemble member is a small MLP (:func:`make_mlp`). When ``predict_std`` is
True every member outputs a mean *and* a per-dimension standard deviation (via
softplus, floor 1e-3) trained with a Gaussian negative-log-likelihood; otherwise
members output means only and are trained with MSE. Input/output features are
standardised with running statistics accumulated during ``update``/``update_from
_buffer``; ``predict`` applies those statistics internally and reports RAW
units.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tact_ai.config import EnvConfig, ModelConfig
from tact_ai.model.features import D_ACTION, D_IN, D_OUT, D_POSE, D_TACTILE, build_input
from tact_ai.model.replay import ReplayBuffer

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def make_mlp(
    in_dim: int,
    hidden: list[int] | tuple[int, ...],
    out_dim: int,
    activation: str = "tanh",
    predict_std: bool = False,
    seed: int | None = None,
) -> nn.Sequential:
    """Build a small fully-connected MLP as an ``nn.Sequential``.

    Output width is ``out_dim`` (MSE mode) or ``2 * out_dim`` (``predict_std``
    mode: concatenated means then log-stds). Seeding the builder (via
    ``torch.manual_seed``) makes the resulting weights reproducible.
    """
    act = _ACTIVATIONS.get(activation)
    if act is None:
        raise ValueError(f"unsupported activation {activation!r} (use tanh or relu)")
    if seed is not None:
        torch.manual_seed(seed)
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(act())
        prev = int(h)
    final = int(out_dim) * 2 if predict_std else int(out_dim)
    layers.append(nn.Linear(prev, final))
    return nn.Sequential(*layers)


class TactileWorldModel:
    """Deep-ensemble world model with per-member bootstrap training.

    Public API (Phase 3 contract):

    - ``__init__(model_cfg, n_candidates, env_cfg=None, seed=0)``
    - ``update(batch_indices=None, sampler=None, n_epochs=None) -> bool``
    - ``update_from_buffer(buffer, n_epochs=None) -> bool``
    - ``predict(x_raw) -> {"mean", "member_means", "member_stds"}``
    - ``epistemic_std(x_raw) -> (B, D_OUT)``
    - ``pose_delta(x_raw) -> (B, 3)``, ``next_tactile(x_raw) -> (B, 12)``
    - ``reset_state()``, ``record_stats(x_raw, y_raw)``
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        n_candidates: int,
        env_cfg: EnvConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Build ``n_ensemble`` member MLPs on CPU with a deterministic seed."""
        self.model_cfg = model_cfg
        self.n_candidates = int(n_candidates)
        self.env_cfg = env_cfg if env_cfg is not None else EnvConfig()
        self.seed = int(seed)
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        # Input width = tactile + pose + action + belief(n_candidates). For the
        # default 18-candidate grid this equals features.D_IN (40).
        self.d_in = int(D_TACTILE + D_POSE + D_ACTION + self.n_candidates)
        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.members = nn.ModuleList(
            [
                make_mlp(
                    self.d_in,
                    list(model_cfg.hidden),
                    D_OUT,
                    model_cfg.activation,
                    bool(model_cfg.predict_std),
                    seed=self.seed + 1 + i,
                )
                for i in range(model_cfg.n_ensemble)
            ]
        )
        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=model_cfg.lr) for m in self.members
        ]
        self._stats_initialized = False
        self._in_mean = np.zeros(self.d_in, dtype=np.float32)
        self._in_std = np.ones(self.d_in, dtype=np.float32)
        self._out_mean = np.zeros(D_OUT, dtype=np.float32)
        self._out_std = np.ones(D_OUT, dtype=np.float32)
        self._n_stats = 0

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(
        self,
        batch_indices: np.ndarray | None = None,
        sampler=None,
        n_epochs: int | None = None,
    ) -> bool:
        """Train all members using data supplied by ``sampler``.

        ``sampler`` is a zero-argument callable returning ``(x_raw, y_raw)``
        numpy arrays of shape ``(N, D_IN)`` and ``(N, D_OUT)`` in raw units.
        Each member trains on its own bootstrap subset (resampled with
        replacement) of size ``ModelConfig.batch_size`` for ``n_epochs`` (or
        ``ModelConfig.n_epochs_per_update``) epochs. When ``batch_indices`` is
        given, only those rows of the sampled data are used. Returns True when
        any training happened; returns False (no-op) when ``sampler`` is None.
        """
        if sampler is None:
            return False
        x_raw, y_raw = sampler()
        x_raw = np.asarray(x_raw, dtype=np.float32)
        y_raw = np.asarray(y_raw, dtype=np.float32)
        if batch_indices is not None:
            x_raw = x_raw[batch_indices]
            y_raw = y_raw[batch_indices]
        if x_raw.shape[0] == 0:
            return False
        self.record_stats(x_raw, y_raw)
        self._train_on(x_raw, y_raw, n_epochs)
        return True

    def update_from_buffer(
        self, buffer: ReplayBuffer, n_epochs: int | None = None
    ) -> bool:
        """Train one update round from every transition in ``buffer``.

        Inputs are built from the stored transitions via
        :func:`tact_ai.model.features.build_input`; targets are
        ``[next_tactile, next_pose - pre_pose]`` (pose deltas in raw units).
        """
        n = len(buffer)
        if n == 0:
            return False
        xs = np.zeros((n, self.d_in), dtype=np.float32)
        ys = np.zeros((n, D_OUT), dtype=np.float32)
        for i, t in enumerate(buffer.transitions()):
            xs[i] = build_input(t.obs, t.action_id, t.belief, self.env_cfg)
            ys[i, :D_TACTILE] = np.asarray(t.next_obs["tactile"], dtype=np.float32)
            ys[i, D_TACTILE:] = np.asarray(
                t.next_obs["pose"], dtype=np.float32
            ) - np.asarray(t.obs["pose"], dtype=np.float32)
        return self.update(sampler=lambda: (xs, ys), n_epochs=n_epochs)

    def record_stats(self, x_raw: np.ndarray, y_raw: np.ndarray) -> None:
        """Accumulate running input/output mean and std over a raw batch."""
        x64 = np.asarray(x_raw, dtype=np.float64)
        y64 = np.asarray(y_raw, dtype=np.float64)
        n = len(x64)
        if n == 0:
            return
        if self._n_stats == 0:
            self._in_sum = np.zeros(self.d_in, dtype=np.float64)
            self._in_sumsq = np.zeros(self.d_in, dtype=np.float64)
            self._out_sum = np.zeros(D_OUT, dtype=np.float64)
            self._out_sumsq = np.zeros(D_OUT, dtype=np.float64)
        self._in_sum += x64.sum(axis=0)
        self._in_sumsq += np.square(x64).sum(axis=0)
        self._out_sum += y64.sum(axis=0)
        self._out_sumsq += np.square(y64).sum(axis=0)
        self._n_stats += n
        count = float(self._n_stats)
        self._in_mean = (self._in_sum / count).astype(np.float32)
        self._out_mean = (self._out_sum / count).astype(np.float32)
        in_var = np.maximum(
            self._in_sumsq / count - self._in_mean.astype(np.float64) ** 2, 0.0
        )
        out_var = np.maximum(
            self._out_sumsq / count - self._out_mean.astype(np.float64) ** 2, 0.0
        )
        self._in_std = self._sanitise_std(np.sqrt(in_var))
        self._out_std = self._sanitise_std(np.sqrt(out_var))
        self._stats_initialized = True

    @staticmethod
    def _sanitise_std(std: np.ndarray) -> np.ndarray:
        """Return a usable std array: constant dims unnormalised, others >= 1e-3."""
        out = np.asarray(std, dtype=np.float64)
        out = np.where(out < 1e-6, 1.0, out)
        out = np.where(np.isfinite(out) & (out < 1e-3), 1e-3, out)
        return out.astype(np.float32)

    def reset_state(self) -> None:
        """Forget all accumulated statistics (keeps network weights).

        Useful for generalisation experiments where the model is re-evaluated
        without any training data seen this run.
        """
        self._stats_initialized = False
        self._in_mean = np.zeros(self.d_in, dtype=np.float32)
        self._in_std = np.ones(self.d_in, dtype=np.float32)
        self._out_mean = np.zeros(D_OUT, dtype=np.float32)
        self._out_std = np.ones(D_OUT, dtype=np.float32)
        self._n_stats = 0

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(self, x_raw: np.ndarray) -> dict[str, np.ndarray]:
        """Predict next features for ``x_raw`` of shape ``(B, D_IN)``.

        Returns ``{"mean": (B, D_OUT), "member_means": (K, B, D_OUT),
        "member_stds": (K, B, D_OUT)}`` all in raw units.
        """
        x_raw = np.asarray(x_raw, dtype=np.float32)
        x = self._standardise_in(x_raw)
        xt = torch.from_numpy(x)
        k = len(self.members)
        b = len(x_raw)
        member_means = np.zeros((k, b, D_OUT), dtype=np.float32)
        member_stds = np.zeros((k, b, D_OUT), dtype=np.float32)
        for i, m in enumerate(self.members):
            m.eval()
            out = m(xt).numpy()
            if self.model_cfg.predict_std:
                mean_z = out[:, :D_OUT]
                log_std_z = out[:, D_OUT:]
                std_z = np.logaddexp(np.zeros_like(log_std_z), log_std_z) + 1e-3
            else:
                mean_z = out
                std_z = np.zeros((b, D_OUT), dtype=np.float32)
            member_means[i] = mean_z * self._out_std + self._out_mean
            member_stds[i] = std_z * self._out_std
        mean = member_means.mean(axis=0)
        return {
            "mean": np.asarray(mean, dtype=np.float32),
            "member_means": np.asarray(member_means, dtype=np.float32),
            "member_stds": np.asarray(member_stds, dtype=np.float32),
        }

    def epistemic_std(self, x_raw: np.ndarray) -> np.ndarray:
        """Predictive epistemic uncertainty: std over member means, ``(B, D_OUT)``."""
        pred = self.predict(x_raw)
        return np.std(pred["member_means"], axis=0, dtype=np.float32)

    def pose_delta(self, x_raw: np.ndarray) -> np.ndarray:
        """Predicted ``(ddx, ddy, ddist)`` pose deltas, ``(B, 3)`` raw units."""
        return self.predict(x_raw)["mean"][:, D_TACTILE:]

    def next_tactile(self, x_raw: np.ndarray) -> np.ndarray:
        """Predicted next tactile vector, ``(B, 12)`` raw units."""
        return self.predict(x_raw)["mean"][:, :D_TACTILE]

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _standardise_in(self, x_raw: np.ndarray) -> np.ndarray:
        return ((x_raw - self._in_mean) / self._in_std).astype(np.float32)

    def _standardise_out(self, y_raw: np.ndarray) -> np.ndarray:
        return ((y_raw - self._out_mean) / self._out_std).astype(np.float32)

    def _train_on(
        self, x_raw: np.ndarray, y_raw: np.ndarray, n_epochs: int | None
    ) -> None:
        cfg = self.model_cfg
        n_epochs = cfg.n_epochs_per_update if n_epochs is None else int(n_epochs)
        n = len(x_raw)
        bs = min(int(cfg.batch_size), n)
        x_std = self._standardise_in(x_raw)
        y_std = self._standardise_out(y_raw)
        xt = torch.from_numpy(x_std)
        yt = torch.from_numpy(y_std)
        for member, opt in zip(self.members, self.optimizers):
            member.train()
            opt.zero_grad(set_to_none=True)
            for _ in range(n_epochs):
                idx = self._rng.integers(0, n, size=bs)
                bx = xt[idx]
                by = yt[idx]
                out = member(bx)
                loss = self._loss(out, by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(member.parameters(), cfg.grad_clip)
                opt.step()

    def _loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.model_cfg.predict_std:
            mean, log_std = out.chunk(2, dim=-1)
            std = torch.nn.functional.softplus(log_std) + 1e-3
            nll = 0.5 * ((y - mean) / std) ** 2 + torch.log(std)
            return nll.mean() + 0.5 * np.log(2.0 * np.pi)
        return torch.nn.functional.mse_loss(out, y)