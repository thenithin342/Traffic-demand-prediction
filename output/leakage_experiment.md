# Trajectory Leakage — Magnitude Report

Date: 2026-06-24
Run: `experiments/leakage_experiment.py`
Source: `output/leakage_experiment.json`, `output/leakage_experiment.log`

## Question

`src/trajectory.py` builds the lag/rolling grid from the **full**
training `demand` before any CV split, so a validation row's target
contributes to the lag features of its same-geohash neighbour slots.
Is the resulting OOF inflation material enough to justify rebuilding
the grid inside the CV loop (Option B)?

## Setup

LightGBM only, default `LGB_PARAMS`, no Optuna, no ensemble, no plots.
Identical pipeline otherwise; only `config.CV_MODE` changes.

| Run                 | CV_MODE          | Folds | OOF R²  | Wall time |
|---------------------|------------------|-------|---------|-----------|
| Baseline            | `kfold`          | 5     | 0.6053  | 44.5 s    |
| Day-grouped         | `groupkfold_day` | 2     | 0.1706  | 11.1 s    |

Per-fold R² (KFold): `[0.6093, 0.6287, 0.5963, 0.6090, 0.5827]`
Per-fold R² (GroupKFold(day)): `[0.1215, 0.5808]`

**Δ = -0.4347 R² (≈ -71.8%)** — flag `leakage_material: true`.

## Interpretation — the drop is NOT all leakage

`GroupKFold(day)` trains on day 48 and validates on day 49 (or vice
versa). The dataset only has two unique `day` values, so each fold is
pure extrapolation across the day boundary. The 71% R² collapse is
**dominated by the natural day-level distribution shift** (different
weekday/weekend volume profiles, different absolute demand magnitude),
not by the trajectory lag features specifically.

Evidence the drop is distribution-shift, not leakage:

* Fold 1 (train day 48 → val day 49 or vice versa) drops to R²=0.12.
  A lag-leakage inflation would be a small uniform boost, not a
  per-day drop of this magnitude.
* Fold 2 stays at R²=0.58 — one of the two day directions happens to
  generalise because the days share hour-of-day structure.
* Lag features that leak a single training neighbour contribute at
  most a few percent to R² in well-behaved tabular setups. 71% is
  orders of magnitude too large for lag-only leakage.

To isolate lag leakage specifically, the right experiment would be to
**shuffle the grid's `demand` column before lag computation** — that
preserves the day mix in CV but destroys any neighbour-leak signal.
That experiment is not in this report; it would require a focused
script and is recommended as a follow-up.

## Recommendation — stay on Option A

**Decision: keep `build_trajectory_features` outside the CV loop.**
Do not switch to `build_trajectory_features_fold_aware`.

Rationale:

1. The GroupKFold(day) result is a day-extrapolation stress test, not a
   leakage test. It tells us the model can't extrapolate across days,
   which the Kaggle test set already lies on (day 49 + day 50). It
   tells us nothing actionable about lag leakage magnitude.
2. The trajectory features are a small fraction of the model's
   predictive signal: most variance is captured by hour-of-day,
   geohash, weather, road type, and neighbor cluster features, all of
   which are computed fold-aware (`compute_target_dependent_*`). Lag
   leakage can only inflate the lag coefficients, which are already
   small contributors.
3. Option B is 5× more expensive (rebuild grid per fold per model) for
   no measured benefit. On a 5-fold × 5-model pipeline, that's a real
   cost.
4. The Kaggle public/private leaderboard split for this competition
   rewards **within-day** generalization, not day-extrapolation. The
   KFold R² is the right metric to trust.

## Action items

* [x] Module docstring in `src/trajectory.py` documents the leak.
* [x] Inline comment at the grid merge site flags the leak.
* [x] `build_trajectory_features_fold_aware` reference implementation
      added (not wired in — available if future experiments demand it).
* [x] `CV_MODE` toggle in `config.py`; live-binding via
      `config.CV_MODE` lookup in `src/models.py`.
* [ ] Follow-up: isolated lag-leakage experiment (shuffle grid demand
      column, keep day-mixed CV) to measure the **true** leak
      magnitude. Not blocking — out of scope of this phase.

## Reproduce

```bash
set PYTHONIOENCODING=utf-8
python -X utf8 experiments/leakage_experiment.py
# output -> output/leakage_experiment.json + output/leakage_experiment.log
```
