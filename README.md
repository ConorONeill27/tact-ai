# Tact AI — Intelligent Tactile Exploration (SciFest 2026)

A CPU-only, headless research prototype testing the hypothesis that a robot
facing an unfamiliar object benefits from explicitly valuing **information** —
reducing uncertainty about unknown physical properties — and not just immediate
task progress, when every physical interaction is costly.

The loop under study: `Touch -> Predict -> Decide -> Act -> Learn`. A robot must
push a disk to a goal using touch, the only channel that reveals the hidden
properties `h = (friction mu, mass, radius)`.

## Setup

```powershell
# nothing to install beyond the standard scientific stack
python -c "import numpy, torch, matplotlib"   # numpy 2.x, torch 2.x CPU, matplotlib 3.x
```

Fully deterministic when seeded (torch/numpy seeds set at every entry point).
No CUDA required — the world model is small enough to train in minutes on CPU.

## Run

All entry points go through one CLI:

```powershell
# tiny end-to-end sanity run
python -m tact_ai.run --mode quicktest

# an interactive run of one info-aware trial
python -m tact_ai.run --mode train --seed 0

# the full pipeline: experiments + analysis + report
python -m tact_ai.run --mode all

# experiments only, selected conditions
python -m tact_ai.run --mode experiments --conditions baseline information robustness generalization compute property_scope beta_sweep cem_budget

# diagnostics (oracle ceiling, info-gain validity, identifiability)
python -m tact_ai.run --mode diagnose --n-oracle 200

# regenerate figures + report from existing raw JSON (no reruns)
python -m tact_ai.run --mode analyze

# smokescreen runs (never touch results/raw)
python -m tact_ai.run --mode experiments --smoke

# render a demo animation (mp4/gif/both)
python -m tact_ai.run --mode video --fmt both
```

Key overrides: `--n-trials`, `--seeds`, `--warmup-episodes`, `--beta`,
`--cem-budget`, `--model-ensemble`, `--model-hidden`, `--model-lr`,
`--goal-tol`. Run `python -m tact_ai.run --help` for the full list.

Tests: `python -m pytest tact_ai/tests/ -q` (84 tests).

## Results (headline)

| method | success | 95% CI | mean model evals | first conc step |
| --- | --- | --- | --- | --- |
| random | 4.4% | [1.7, 10.9] | 0 | 6.4 |
| greedy | 72.2% | [62.2, 80.4] | ~1.6k | 11.9 |
| cem | 84.4% | [75.6, 90.5] | ~48.5k | 12.5 |
| info_aware | 87.8% | [79.4, 93.0] | ~1.25k | 1.97 |

- **info_aware identifies the object ~6x faster** (first concentration step 1.97
  vs 11.9/12.5) and converts that into the best success rate at
  **~39x fewer model evaluations than CEM**.
- Oracle ceiling (perfect-information greedy on the analytic physics): **100%**
  success in 8.5 mean steps — the task is solvable; observed failures are
  planner/model artifacts.
- Info-gain estimator validity: predicted vs realised entropy reduction
  r = 0.84 (Pearson), rho = 0.50 (Spearman), n = 414.
- Candidate grid identifiability: all 153 pairs separated by >= 17 sigma.
- Full tables and interpretation: `results/RESULTS.md`; figures:
  `results/figures/*.png`; demo video: `results/video/`.

## Architecture

| component | module | role |
| --- | --- | --- |
| environment | `tact_ai/simulation/` | 2D quasi-static planar push; 216 actions (36 contact angles x 3 modes x 2 forces); motion, slip and contact torques depend on `h`. |
| tactile sensing | `tact_ai/sensing/` | 12-dim tactile vector (normal/tangential force, slip + measured friction, centroid, peak/spread, contact angle, near-slip proxy) with configurable noise. |
| world model | `tact_ai/model/` | deep ensemble of MLPs predicting next tactile features + pose deltas; trained online from experience (MSE); provides fast internal roll-outs. |
| belief filter | `tact_ai/model/belief_filter.py` | physics-based Bayesian posterior over the 18 candidate `(mu, mass, radius)` cells, updated from tactile + motion observations; estimates expected information gain. |
| planners | `tact_ai/agents/` | random; greedy (reward-only); cem (Cross-Entropy Method MPC); info_aware (proposed: progress + beta * expected info gain). |

All planners share the environment, sensor, world model and belief filter; they
differ only in how they score candidate actions.

## Experiment conditions

| condition | purpose |
| --- | --- |
| baseline | main comparison of the four planners (3 seeds x 30 trials) |
| information | how quickly each method learns the object |
| robustness | sensor-noise sweep (sigma = 0..0.6) |
| generalization | region split: train small-radius cells, evaluate large-radius cells, updates OFF at eval |
| compute | world-model forward passes per episode |
| property_scope | uncertainty width narrow/medium/wide |
| beta_sweep | information-gain weight beta = 0, .25, .5, 1, 2, 5 |
| cem_budget | CEM forward-pass budget per decision |
| diagnostics | oracle ceiling, IG-validity, identifiability, trial-index post-hoc |

Every model-based planner runs 3 off-camera random warm-up episodes (model
updates ON) before logged trials so every trial starts with a planning-capable
world model.

## Outputs

- `results/raw/*.json` — per-trial/per-condition results, manifest and wall times
- `results/figures/*.png` — figures
- `results/diagnostics/*` — diagnostic results + figures
- `results/RESULTS.md` — final report
- `results/video/*` — demo animation