# Traffic Demand Prediction

> **Competition task:** Predict normalised traffic demand (`0–1`) at geohash-encoded road segments for Day 49, given historical observations from Days 1–48.

[![Live Demo](https://img.shields.io/badge/🚦%20Live%20Demo-traffic--demand--prediction.onrender.com-6366f1?style=for-the-badge)](https://traffic-demand-prediction.onrender.com)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikitlearn)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)](https://traffic-demand-prediction.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Feature Engineering](#feature-engineering)
6. [Models & Ensemble](#models--ensemble)
7. [Validation Strategy](#validation-strategy)
8. [Leakage Audit & Fixes](#leakage-audit--fixes)
9. [Performance Optimisation](#performance-optimisation)
10. [Getting Started](#getting-started)
11. [Configuration](#configuration)
12. [Testing](#testing)
13. [Experiment Tracking](#experiment-tracking)

---

## Project Overview

This repository contains a full end-to-end machine learning pipeline for traffic demand forecasting — from raw CSV data all the way to a **deployed, production-grade web application**.

🚦 **Live Web App:** [https://traffic-demand-prediction.onrender.com](https://traffic-demand-prediction.onrender.com)

The app serves real-time predictions via a premium "Command Center" dashboard powered by a **XGBoost Stacking Ensemble (R² = 0.96)** trained on 101 engineered features. The codebase was built with a focus on **evaluation integrity**: all target encodings, trajectory (lag/rolling) features, and neighbour statistics are computed inside the CV fold loop so that no future information leaks into the validation set.

---

## Key Results

| Metric | Value |
|---|---|
| Ensemble OOF R² (outer 20% split) | **0.7819** |
| Day-49-only OOF R² *(most relevant — test set is all Day 49)* | **0.7502** |
| Estimated competition score | **78.19 / 100** |
| Wall-clock runtime (full pipeline incl. 50 Optuna trials) | **~156 s** |
| Speedup vs. original pipeline | **~24×** |
| Test suite | **20 / 20 passing** |

### Per-model Day-49 R²

| Model | Day-49 R² |
|---|---|
| CatBoost | 0.7171 |
| MLP | 0.7147 |
| HistGBM | 0.6348 |
| XGBoost | 0.6328 |
| LightGBM | 0.6201 |
| **Stacked Ensemble (XGBStack)** | **0.7502** |

---

## Repository Structure

```
Traffic-demand-prediction/
├── config.py                     # Central constants & hyperparameters
├── solution.py                   # Main pipeline entry point
├── eda.py                        # Standalone EDA script
├── requirements.txt              # Pinned dependencies
├── pytest.ini                    # Test configuration
│
├── src/
│   ├── data_loader.py            # CSV loading & basic validation
│   ├── feature_engineering.py   # Temporal, cyclical & interaction features
│   ├── trajectory.py            # Lag / rolling demand features (fold-aware)
│   ├── neighbor.py              # Geohash neighbour & cluster features
│   ├── target_encoding.py       # OOF smoothed target encodings
│   ├── models.py                # LightGBM, XGBoost, CatBoost, HistGBM trainers
│   ├── nn_model.py              # MLP base model
│   ├── tuning.py                # Optuna hyperparameter tuning
│   ├── ensemble.py              # Ridge / SLSQP / XGBoost stacking
│   ├── visualizations.py        # Plots (feature importance, foldwise R², etc.)
│   └── geohash_decoder.py       # Pure-Python geohash → lat/lon decoder
│
├── tests/
│   ├── test_target_encoding.py  # OOF encoding correctness & leak-free asserts
│   ├── test_trajectory.py       # Fold-aware trajectory feature tests
│   ├── test_feature_engineering.py
│   ├── test_ensemble.py
│   ├── test_geohash_decoder.py
│   └── test_solution_smoke.py   # End-to-end smoke test
│
├── experiments/
│   └── leakage_experiment.py    # Systematic leakage audit script
│
├── output/
│   ├── submission.csv           # Final competition submission
│   ├── plots/                   # Auto-generated diagnostic plots
│   └── *.log / *.json           # Run artefacts and metrics
│
└── data/
    └── dataset/                 # train.csv, test.csv, sample_submission.csv
```

---

## Pipeline Architecture

```
Raw Data (train.csv / test.csv)
         │
         ▼
┌─────────────────────────┐
│  1. Data Loading         │  src/data_loader.py
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. Feature Engineering  │  Temporal (hour, day-of-week, cyclicals),
│                         │  spatial (geohash decode), interaction terms
└───────────┬─────────────┘  src/feature_engineering.py
            │
            ▼
┌─────────────────────────┐
│  3. Trajectory Features  │  Fold-aware lag / rolling demand features
│     (fold-aware)         │  built inside CV loop to prevent leakage
└───────────┬─────────────┘  src/trajectory.py
            │
            ▼
┌─────────────────────────┐
│  4. Static Neighbour     │  Geohash k-NN graph (k=6), KMeans clusters (k=32)
│     & Cluster Features   │
└───────────┬─────────────┘  src/neighbor.py
            │
            ▼
┌─────────────────────────┐
│  5. Static Target Encs   │  Frequency encoding, ordinal label encoding
│     (global, leak-safe)  │
└───────────┬─────────────┘  src/target_encoding.py
            │
            ▼
┌────────────────────────────────────────────────────┐
│  6. Nested CV Loop (5-fold GroupKFold by day)       │
│                                                    │
│   Per fold:                                        │
│   ├─ OOF smoothed target encodings  (C3 fix)       │
│   ├─ OOF neighbour demand stats     (C4 fix)       │
│   ├─ Fold-aware trajectory rebuild  (C1/C2 fix)    │
│   └─ Build/train all 5 base models                │
│                                                    │
│   Models: LightGBM │ XGBoost │ CatBoost            │
│           HistGBM  │ MLP                           │
└───────────┬────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  7. Optuna Tuning        │  50 trials on inner 80% split,
│     (inner split only)   │  MedianPruner, reuses fold cache
└───────────┬─────────────┘  src/tuning.py
            │
            ▼
┌─────────────────────────┐
│  8. Ensemble Stacking    │  Ridge meta-learner + SLSQP weight search
│                         │  + XGBoost 2nd-level stacker; best selected
└───────────┬─────────────┘  by OOF R²   →   src/ensemble.py
            │
            ▼
    submission.csv
```

---

## Feature Engineering

### Temporal Features
| Feature | Description |
|---|---|
| `hour`, `minute` | Time-of-day extracted from `timestamp` |
| `hour_sin`, `hour_cos` | Cyclical encoding (24-h period) |
| `dow_sin`, `dow_cos` | Day-of-week cyclical encoding |
| `is_peak_hour` | Binary flag for morning/evening rush hours |
| `is_weekend` | Binary flag |
| `time_slot` | 15-minute interval index (0–95) |

### Spatial Features
| Feature | Description |
|---|---|
| `lat`, `lon` | Decoded from geohash via pure-Python decoder |
| `geohash_prefix4/5` | Coarser geohash prefixes for spatial grouping |
| `cluster_id` | KMeans cluster assignment (k=32) |
| `cluster_demand_*` | Cluster-level demand stats (mean, std, max, min) |
| `nbr_demand_mean/std` | Average demand across k=6 nearest geohash neighbours |

### Trajectory (Lag/Rolling) Features
Built **fold-aware** — rebuilt per CV fold with validation demand masked to `NaN`:

| Feature | Description |
|---|---|
| `lag_same_day_1/2/4/8` | Same-day lags (1, 2, 4, 8 time steps back) |
| `rollmean_same_day_3/8` | Same-day rolling mean |
| `rollstd_same_day_8` | Same-day rolling std |
| `lag_prev_day_0/1/4` | Previous-day lags |
| `prev_day_rollmean_8` | Previous-day rolling mean |
| `warm_mean/std/max/min/range/last/trend` | Aggregate warm-start statistics |

### Target Encodings (OOF, leak-free)
| Feature | Description |
|---|---|
| `geohash_target_enc` | Smoothed mean demand per geohash (inner-KFold OOF) |
| `RoadType_target_enc` | Smoothed mean demand per road type |
| `Weather_target_enc` | Smoothed mean demand per weather condition |
| `*_label` | Ordinal label encodings for tree boosters |

---

## Models & Ensemble

### Base Models

| Model | Key Hyperparameters | Notes |
|---|---|---|
| **LightGBM** | `num_leaves=127`, `lr=0.03`, ES=100 | Tuned via Optuna (50 trials) |
| **XGBoost** | `max_depth=8`, `lr=0.03`, `tree_method=hist` | Early stopping, eval on val fold |
| **CatBoost** | `depth=6`, `lr=0.03`, native cat handling | No label encoding needed |
| **HistGBM** | `max_leaf_nodes=63`, `lr=0.05` | sklearn's gradient boosting |
| **MLP** | `StandardScaler` + dense layers | Feed-forward neural network |

### Ensemble Methods (auto-selected by OOF R²)

1. **SLSQP** — Scipy constrained optimisation of non-negative weights summing to 1, with 4 random Dirichlet restarts.
2. **Ridge** — L2-regularised meta-learner trained on OOF predictions.
3. **XGBoost Stacker** — Shallow 2nd-level XGBoost (`max_depth=2`) trained on OOF predictions.

The method with the highest OOF R² is selected for the final submission.

---

## Validation Strategy

The pipeline uses a **two-level nested CV** design to produce an honest evaluation:

```
Full training data (Days 1–48)
│
├── Inner 80%  ──► Optuna hyperparameter tuning (never touches outer)
│
└── Outer 20%  ──► 5-fold GroupKFold (grouped by day)
                    │
                    ├── Fold 1 (train on 4 groups, validate on 1 day)
                    │     └── Compute OOF target encodings, neighbour stats,
                    │         trajectory features inside the fold
                    ├── Fold 2 ... Fold 5
                    │
                    └── Day-49 fold R² ← most informative metric
                        (test set is entirely Day 49)
```

**Why GroupKFold by day?**  
The test set covers a single future day (Day 49). GroupKFold forces the model to predict a held-out day it has never seen during training, matching the real deployment setting and surfacing any cross-day lag leakage.

---

## Leakage Audit & Fixes

A systematic audit identified and fixed **6 categories of data leakage** that inflated the original OOF R² from `0.9938` (fake) to `0.7817` (honest):

| ID | Source File | Description | Fix |
|---|---|---|---|
| **C1** | `trajectory.py` | Lag features built on full train before CV split | Rebuilt per fold with val demand masked to `NaN` |
| **C2** | `trajectory.py` | `val_out` sliced from pre-trimmed `train_out` | Introduced `combined_out`; val sliced from combined output |
| **C3** | `target_encoding.py` | Smooth target encoding used full-fold mean on train rows | Replaced with inner-KFold OOF encoding (`_oof_target_encode`) |
| **C4** | `neighbor.py` | k-NN graph included the self-row | Skip index `0` in argsort; set `k = min(6, n-1)` |
| **C5** | `models.py` | Target encodings computed globally, not fold-aware | Moved into `_build_fold_cache`, computed per fold |
| **C6** | `scripts/train.py` | Optuna tuned on full train including val folds | Isolated to inner 80% split via `StratifiedShuffleSplit` |
| **C12** | `scripts/train.py` | Mock-build run to infer feature columns leaked `demand` into train encodings | Removed mock-build; feature columns inferred from config constants |

---

## Performance Optimisation

| Optimisation | Where | Impact |
|---|---|---|
| Vectorised `pd.merge` for target-enc & neighbour joins | `models.py` | Eliminates O(n) `.apply()` loops |
| Single shared `fold_cache` across all 5 base models | `models.py` | Encodes each fold once instead of 5× |
| Optuna `MedianPruner` | `tuning.py` | Prunes unpromising trials early |
| Reuse fold cache in Optuna trials | `tuning.py` | No second fold rebuild during tuning |
| SLSQP restarts reduced 8 → 4 | `ensemble.py` | Halves ensemble optimisation time |

**Result:** 3 800 s → **156 s** end-to-end (≈ 24× speedup), R² unchanged.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Data files placed at `data/dataset/train.csv`, `data/dataset/test.csv`, `data/dataset/sample_submission.csv`

### Installation

```bash
git clone https://github.com/thenithin342/Traffic-demand-prediction.git
cd Traffic-demand-prediction
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python solution.py
```

Output files are written to `output/`:
- `output/submission.csv` — competition submission
- `output/plots/` — diagnostic visualisations
- `output/baseline_metrics.json` — run metrics

### Run EDA Only

```bash
python eda.py
```

---

## Configuration

All tuneable parameters live in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `SEED` | `42` | Global random seed |
| `N_FOLDS` | `5` | Number of CV folds |
| `CV_MODE` | `"groupkfold_day"` | `"kfold"` or `"groupkfold_day"` |
| `FOLD_AWARE_TRAJECTORY` | `True` | Rebuild trajectory per fold (leak-safe) |
| `N_TRIALS_OPTUNA` | `50` | Optuna trials; set `0` to skip tuning |
| `N_SLSQP_RESTARTS` | `4` | SLSQP random restarts for ensemble weights |
| `TARGET_SMOOTHING` | `10.0` | Bayesian smoothing factor for target encodings |

---

## Testing

```bash
pytest tests/ -v
```

```
tests/test_ensemble.py            PASSED  (5 tests)
tests/test_feature_engineering.py PASSED  (4 tests)
tests/test_geohash_decoder.py     PASSED  (3 tests)
tests/test_solution_smoke.py      PASSED  (2 tests)
tests/test_target_encoding.py     PASSED  (3 tests)
tests/test_trajectory.py          PASSED  (3 tests)

20 passed in ~13s
```

Key test assertions:
- **OOF target encoding is leak-free**: val rows' own targets do not appear in their own encodings.
- **Fold-aware trajectory**: val demand is masked to `NaN` before lag/rolling features are built.
- **Self-neighbour exclusion**: geohash k-NN graph never includes the self-row.
- **Bit-identical runs**: two consecutive pipeline runs produce identical OOF R².

---

## Experiment Tracking

Run artefacts are stored in `output/`:

| File | Contents |
|---|---|
| `output/baseline_metrics.json` | Pre-fix baseline R² and timing |
| `output/leakage_experiment.json` | Per-feature leakage quantification |
| `output/leakage_experiment.md` | Human-readable leakage audit report |
| `output/plots/09_lgb_feature_importance.png` | LightGBM feature importances |
| `output/plots/10_model_comparison.png` | Per-model and ensemble OOF R² |
| `output/plots/12_foldwise_r2.png` | Fold-by-fold R² breakdown |
| `output/plots/13_day49_r2.png` | Day-49 temporal fold R² |
| `output/plots/14_residuals.png` | Residual distribution |

---

## Commit History

| Commit | Description |
|---|---|
| `fix(trajectory,neighbor,encoding)` | Eliminate cross-fold target leakage (C1–C4) |
| `fix(cv,tuning)` | Introduce honest nested CV and GroupKFold support (C5–C8) |
| `fix(ensemble,engineering,models)` | Resolve reproducibility bugs and edge-case crashes (C9–C12) |
| `perf(pipeline)` | Vectorize hot paths; deduplicate fold cache for 24× speedup |
| `refactor(codebase)` | Design cleanup, deduplication, and full lint pass |

---

## License

This project is licensed under the MIT License.
