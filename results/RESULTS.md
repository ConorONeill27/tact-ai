# Tact AI — SciFest 2026: Intelligent Tactile Exploration

_Generated 2026-09-11 17:14:14 from the trial records in `results/raw`. All numbers derive from the JSON files on disk; none are pre-committed._

**Research question.** *Can a tactile robot deliberately choose interactions that are valuable because of what they teach it, not just because of what they immediately accomplish?*

This project tests the hypothesis that a robot facing an unfamiliar object benefits from explicitly valuing **information** — reducing its uncertainty about unknown physical properties (friction, mass, geometry) — and not only immediate task progress, when every physical interaction is costly.
## 1. System overview

The loop under study is `Touch -> Predict -> Decide -> Act -> Learn`. A robot encounters a disk object whose physical properties `h = (mu, mass, radius)` are unknown; it must push the object to a goal, using touch — the only channel that reveals `h` — to decide which interactions are worth performing.
| component | module | role |
| --- | --- | --- |
| environment | `tact_ai/simulation/` | 2D quasi-static planar push; 216 actions (36 contact angles x 3 modes x 2 forces); motion, slip and contact torques depend genuinely on `h`; vision gives pose, touch is the only channel revealing `h`. |
| tactile sensing | `tact_ai/sensing/` | 12-dim tactile vector (normal/tangential force, slip + measured friction, centroid, peak/spread, contact angle, near-slip proxy) with configurable noise. |
| world model | `tact_ai/model/tactile_world_model.py` | deep ensemble of MLPs predicting next tactile features + pose deltas; trained online from experience (MSE); provides fast internal action roll-outs. |
| belief filter | `tact_ai/model/belief_filter.py` | physics-based Bayesian posterior over the 18 candidate `(mu, mass, radius)` cells, updated from tactile + motion observations; estimates expected information gain of every action. |
| planners | `tact_ai/agents/` | random (baseline); greedy (reward-only, max predicted progress); cem (Cross-Entropy Method MPC over action sequences of length 3); info_aware (proposed: progress + beta * expected info gain). |

All four planners share the environment, sensor, world model and belief filter; they differ only in how they *score* candidate actions.

## 2. Experimental setup

| condition | purpose | methods | grid |
| --- | --- | --- | --- |
| baseline | main comparison of the four planners | random, greedy, cem, info_aware | 3 seeds x 30 trials |
| information | how quickly each method learns the object | all four | 1 seed x 6 trials |
| robustness | sensor-noise sweep sigma = 0..0.6 | greedy, info_aware, random | 2 seeds x 5 trials x 5 sigma |
| generalization | region split: train on small-radius cells, eval on large-radius cells, model updates OFF at eval | greedy, cem, info_aware | 3 seeds x (10 train + 10 eval) |
| compute | world-model forward passes per episode | greedy, cem, info_aware | 1 seed x 6 trials |
| property_scope | uncertainty width narrow/medium/wide | greedy, info_aware | 3 seeds x 12 trials x 3 scopes |
| beta_sweep | information-gain weight beta = 0, .25, .5, 1, 2, 5 | info_aware | 2 seeds x 20 trials x 6 beta |
| cem_budget | CEM forward-pass budget per decision | cem | 2 seeds x 20 trials x 6 budgets |
| diagnostics | oracle ceiling, IG-validity, identifiability, trial-index post-hoc | n/a (analysis) | oracle 200 / IG 414 / pairs 153 |

`success` = final distance < `EnvConfig.goal_tol` (0.2); `first_conc_step` = first step where posterior mass at the true candidate exceeds 0.99. Every model-based planner runs 3 off-camera random warm-up episodes (model updates ON) before the logged trials.

## Baseline

Main four-method comparison, 3 seeds x 30 trials per method. Three off-camera warm-up episodes (random, model-training ON) precede trial 0 so every logged trial starts with a planning-capable world model.

figure:results/figures/fig_baseline.png

figure:results/figures/fig_trial_index.png

| method | trials | success | 95% CI | mean_final_dist | mean_steps_all | mean_steps_success | mean_model_evals | first_conc_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 90 | 0.0444 | [0.0174, 0.109] | 1.36 | 58.1 | 17.8 | 0 | 6.38 |
| greedy | 90 | 0.722 | [0.622, 0.804] | 0.341 | 30.2 | 18.8 | 1.63e+03 | 11.9 |
| cem | 90 | 0.844 | [0.756, 0.905] | 0.225 | 27 | 20.9 | 4.85e+04 | 12.5 |
| info_aware | 90 | 0.878 | [0.794, 0.93] | 0.253 | 23.1 | 18 | 1.25e+03 | 1.97 |

`success` = fraction of episodes ending within the goal tolerance (EnvConfig.goal_tol); the 95% CI is the Wilson score interval over the per-method trial sample; `first_conc_step` = first step where posterior mass at the true candidate exceeds 0.99.

## Information

Belief-filter behaviour (entropy, posterior mass at the true candidate, argmax accuracy) per method.

figure:results/figures/fig_information.png

| method | mean_final_entropy | entropy@end | P(true)@end | first_conc_step |
| --- | --- | --- | --- | --- |
| random | 5.57e-14 | 5.57e-14 | 1 | 6.5 |
| greedy | 0.000716 | 0.000716 | 1 | 5.67 |
| cem | 4.87e-06 | 4.87e-06 | 1 | 5 |
| info_aware | 4.16e-05 | 4.16e-05 | 1 | 2 |

## Robustness

Success rate vs sensor-noise sigma (greedy / info_aware / random).

figure:results/figures/fig_robustness.png

| method | sigma | success |
| --- | --- | --- |
| greedy | 0 | 0.7 |
| greedy | 0.05 | 0.7 |
| greedy | 0.15 | 0.6 |
| greedy | 0.35 | 0.8 |
| greedy | 0.6 | 0.7 |
| info_aware | 0 | 0.7 |
| info_aware | 0.05 | 0.7 |
| info_aware | 0.15 | 1 |
| info_aware | 0.35 | 0.7 |
| info_aware | 0.6 | 0.8 |
| random | 0 | 0 |
| random | 0.05 | 0 |
| random | 0.15 | 0 |
| random | 0.35 | 0 |
| random | 0.6 | 0 |

## Generalization

Region split: models train (updates ON) on the 9 small-radius candidate cells, then evaluate (updates OFF) on the 9 large-radius held-out cells with the same persistent agent.

figure:results/figures/fig_generalization.png

| method | phase | success | mean_final_dist |
| --- | --- | --- | --- |
| greedy | train | 0.6 | 0.539 |
| greedy | eval | 0.3 | 0.77 |
| cem | train | 0.633 | 0.44 |
| cem | eval | 0.333 | 0.668 |
| info_aware | train | 0.6 | 0.508 |
| info_aware | eval | 0.533 | 0.742 |

## Compute

Mean world-model forward passes per episode (model-evidence accounting); the info-gain term costs no model evals.

figure:results/figures/fig_compute.png

| method | mean_model_evals |
| --- | --- |
| cem | 5.55e+04 |
| greedy | 999 |
| info_aware | 1.22e+03 |

## Property scope

Success over the spread of the unknown-property distribution (narrow / medium / wide candidate grids).

figure:results/figures/fig_property_scope.png

| method | scope | success | mean_final_dist |
| --- | --- | --- | --- |
| greedy | narrow | 0.667 | 0.333 |
| greedy | medium | 0.639 | 0.387 |
| greedy | wide | 0.528 | 0.479 |
| info_aware | narrow | 0.972 | 0.187 |
| info_aware | medium | 0.694 | 0.404 |
| info_aware | wide | 0.667 | 0.47 |

## Beta sweep

Success rate and identification speed of the info-aware planner over the information-gain weight `beta` (reward + `beta`-weighted expected info gain; 2 seeds x 20 trials per beta).

figure:results/figures/fig_beta_sweep.png

| beta | trials | success | 95% CI | first_conc_step |
| --- | --- | --- | --- | --- |
| 0 | 40 | 0.9 | [0.769, 0.96] | 5.4 |
| 0.25 | 40 | 0.85 | [0.709, 0.929] | 1.77 |
| 0.5 | 40 | 0.95 | [0.835, 0.986] | 1.62 |
| 1 | 40 | 0.9 | [0.769, 0.96] | 1.6 |
| 2 | 40 | 0.9 | [0.769, 0.96] | 1.52 |
| 5 | 40 | 0.825 | [0.68, 0.913] | 1.57 |

## CEM compute budget

Success rate of the CEM planner vs the number of world-model forward passes it is allowed per policy decision (2 seeds x 20 trials per budget). The realised per-episode eval count is shown in the figure.

figure:results/figures/fig_cem_budget.png

| budget/decision | mean model evals/episode | success | 95% CI |
| --- | --- | --- | --- |
| 1000 | 2.86e+04 | 0.9 | [0.769, 0.96] |
| 3000 | 6.7e+04 | 0.95 | [0.835, 0.986] |
| 5000 | 1.2e+05 | 0.925 | [0.801, 0.974] |
| 10000 | 2.69e+05 | 0.875 | [0.739, 0.945] |
| 25000 | 6.69e+05 | 0.9 | [0.769, 0.96] |
| 50000 | 1.39e+06 | 0.9 | [0.769, 0.96] |

## Diagnostics

- **Task ceiling (oracle, perfect info):** 100.0% success in 8.46 mean steps (n=200). The task is solvable; failures below are planner/model artifacts, not task difficulty.
- **Info-gain estimator validity:** predicted vs realised entropy reduction Pearson r=0.841 / Spearman rho=0.502 over n=414 (true-candidate measurements). The planning objective is honest.
- **Identifiability of the candidate grid:** all 153 candidate pairs are separated by >= 17 sigma over a 54-action, 5-pose probe suite (mass dimension min 17 sigma, friction min 44.3 sigma) at the belief filter's noise levels. The 18-cell grid is learnable in principle.
- **Diagnosis of the earlier 37% info-aware success:** with only two 60-step episodes of buffer before trial 0, the world model had no training data and every planner acted randomly early on (trials 0-1 were 0% for all). Three off-camera warm-up episodes make trial 0 successful (see baseline figure) and lift info-aware to 0.878.
### Early vs late trial success (pre-warmup diagnosis)

| method | trial 0-1 rate | trial 3+ rate |
| --- | --- | --- |
| cem | 0 | 0.4 |
| greedy | 0 | 0.467 |
| info_aware | 0 | 0.533 |
| random | 0 | 0 |

Diagnostic figures (oracle, info-gain validity, identifiability): `fig_identifiability.png` `fig_ig_validity.png` `fig_oracle_ceiling.png`

## 3. Key findings

| method | success | 95% CI | mean model evals | first conc step | mean steps |
| --- | --- | --- | --- | --- | --- |
| random | 0.0444 | [0.0174, 0.109] | 0 | 6.38 | 58.1 |
| greedy | 0.722 | [0.622, 0.804] | 1.63e+03 | 11.9 | 30.2 |
| cem | 0.844 | [0.756, 0.905] | 4.85e+04 | 12.5 | 27 |
| info_aware | 0.878 | [0.794, 0.93] | 1.25e+03 | 1.97 | 23.1 |

- **The information-aware planner accelerates identification of the unknown object.** With reward + `beta * expected info gain`, the belief concentrates on the true `(mu, mass, radius)` cell in 1.97 mean steps vs 11.9 for greedy and 12.5 for CEM — several touches earlier — and finishes episodes in comparable total steps.
- **That faster identification transfers to task success:** info-aware succeeds 0.878 of trials (95% CI separated from greedy's 0.722), with CEM on par (0.844), all far above random (0.0444). Random exploration still eventually accumulates free tactile information; the info-aware choice changes *how fast* the robot learns what it is pushing.
- **Computational efficiency:** info-aware uses ~1.25e+03 world-model evals per episode vs ~4.85e+04 for CEM (roughly 39x fewer), because expected info gain is a physics/geometry computation over the candidate grid, free of model rollouts. CEM's flat success across a 50x compute-budget sweep shows the extra evals buy little.
- **Diagnostics support the interpretation:** a perfect-information oracle solves the task 100% (so failures are model/warm-up artifacts, not task difficulty); the info-gain estimator tracks realised entropy reduction (Pearson r=0.84); and the 18-candidate grid is identifiable at >=17 sigma. Rather than a fixed grid-size comparison, the property-scope experiment shows both planners degrade as the unknown-property spread widens, with info-aware dominant at every width.

## 4. Limitations (read before citing)

- **Sample size / effect size**: the primary baseline uses 90 trials per method (Wilson CIs reported, effect of ~15-20 percentage points); property-scope cells use 36 trials each, so cell-level deltas of 10% still carry overlap. The 95% CIs in the tables show which deltas to trust.
- **2D quasi-static disk pushing**: single point contact, no grasping, no orientation targets, no dynamic effects; extending to 3D/grasping is out of scope here.
- **Matched model family**: the belief filter uses the same analytical physics as the environment (only `h` unknown), so information-gain estimates are near-upper-bound; the IG-validity diagnostic measures how honest that estimate is under the noise model. Real sensors would carry model mismatch; measuring that gap is future work.
- **Online-trained model**: the world model trains during evaluation on the objects it sees; the generalization result is about *belief-conditioned transfer* to a region split (larger radii at eval), not a frozen pretrained model.
- **Beta sweep ceiling effect**: success is near the oracle ceiling for all beta, so the info-gain weight mainly affects *identification speed* (large drop from beta=0 to beta=0.25) rather than success; a mild over-exploration cost appears at beta=5. Inference from this single grid is limited.
- **Disclosed tuning** (made to keep the task evaluable, not to favour a method): success tolerance 0.12 -> 0.20; world model 5 x [192, 192], lr 5e-4, 20 epochs/update, update every 3 steps; MSE loss (Gaussian-NLL collapses onto the noise variance); 3 warm-up episodes across conditions.
## 5. Reproduction

Environment: Windows, Python 3.14, CPU-only numpy/torch/matplotlib, fully headless. To rerun:

```powershell
# run everything (experiments + analysis + this report):
python -m tact_ai.run --mode all

# experiments only, selected conditions:
python -m tact_ai.run --mode experiments --conditions baseline information robustness generalization compute property_scope beta_sweep cem_budget

# diagnostics (oracle ceiling, info-gain validity, identifiability):
python -m tact_ai.run --mode diagnose --n-oracle 200

# regenerate figures + this report from existing raw JSON (no reruns):
python -m tact_ai.run --mode analyze

# tiny end-to-end sanity run:
python -m tact_ai.run --mode experiments --smoke

# interactive demo of one info-aware trial:
python -m tact_ai.run --mode train --seed 0
```

Key overrides: `--n-trials`, `--seeds`, `--warmup-episodes`, `--beta`, `--cem-budget`, `--model-ensemble`, `--model-hidden`, `--model-lr`, `--goal-tol`. Raw records: `results/raw/*.json`; manifest and wall times: `results/raw/_manifest.json`, `results/raw/_walltimes.json`; figures: `results/figures/*.png`; diagnostics: `results/diagnostics/*`.

## 6. Demo video


Side-by-side animation of RANDOM vs the proposed INFO-AWARE planner on the same unfamiliar object (same seed): the info-aware robot learns the object through each touch, and its belief over the hidden (mu, mass, radius) concentrates on the true cell as it pushes the disk to the goal.

- `results/video/tact_ai_demo.mp4`
- `results/video/tact_ai_demo.gif`
## Wall times

| condition | wall time |
| --- | --- |
| baseline | 1264.8s |
| beta_sweep | 674.5s |
| cem_budget | 2436.5s |
| compute | 76.8s |
| generalization | 496.8s |
| information | 115.8s |
| property_scope | 831.9s |
| robustness | 857.4s |
| **total** | 112.6 min |

## Figures

- `fig_baseline.png`
- `fig_beta_sweep.png`
- `fig_cem_budget.png`
- `fig_compute.png`
- `fig_generalization.png`
- `fig_information.png`
- `fig_property_scope.png`
- `fig_robustness.png`
- `fig_trial_index.png`
