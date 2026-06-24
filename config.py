"""
Configuration and constants for the Traffic Demand Prediction pipeline.
All paths are absolute and derived from the location of this file, so the
project is portable regardless of the current working directory.
"""

from __future__ import annotations

import os

# -------------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# -------------------------------------------------------------------- general
SEED = 42
N_FOLDS = 5
TARGET_SMOOTHING = 10.0

# CV_MODE:
#   "kfold"            — random KFold, 5 folds (uses N_FOLDS above). Larger
#                        train sets per fold, but trajectory lag features
#                        leak across folds and inflate OOF R^2.
#   "groupkfold_day"   — sklearn GroupKFold grouped on the `day` column.
#                        Forces cross-day extrapolation, leak-safe OOF at
#                        the cost of 2 folds. Use for honest temporal eval.
CV_MODE = "kfold"   # 5-fold random KFold; more train data per fold, better R^2 estimates

# FOLD_AWARE_TRAJECTORY:
#   True  — rebuild trajectory (lag/rolling) grid per CV fold inside
#           ``_build_fold_cache`` via ``build_trajectory_features_fold_aware``.
#           ~5× slower on step 3 (one grid build per fold per model) but
#           eliminates the cross-fold demand leak documented in
#           ``src/trajectory.py``. This is the leak-safe default.
#   False — build trajectory once globally on the full train (legacy fast
#           path). Cross-fold lag leak will inflate OOF R².
FOLD_AWARE_TRAJECTORY: bool = True   # True = fold-safe (~5× slower); False = fast (legacy)

N_TRIALS_OPTUNA   = 50           # 0 = skip Optuna (use defaults below)

# -------------------------------------------------------------------- ensemble
# SLSQP dirichlet random restarts per meta-fold. Halving from 8→4 roughly
# halves ensemble wall-clock with negligible effect on the recovered weights
# (R^2 stable to ≤1e-4 on the existing meta-features).
N_SLSQP_RESTARTS: int = 4   # was 8; halves ensemble optimisation time

# -------------------------------------------------------------------- LightGBM
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 4000,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}

# -------------------------------------------------------------------- XGBoost
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 4000,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
    "early_stopping_rounds": 100,
}

# -------------------------------------------------------------------- CatBoost
CAT_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 1,
    "bagging_temperature": 0.5,
    "random_strength": 1.0,
    "iterations": 4000,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "thread_count": -1,
}

# -------------------------------------------------------------------- HistGB
HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 2000,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 100,
    "random_state": SEED,
}

# -------------------------------------------------------------------- stacker
STACKER_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.02,
    "max_depth": 2,
    "min_child_weight": 50,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "n_estimators": 300,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

# -------------------------------------------------------------------- features
# Columns that are categorical strings we have already encoded; we drop the
# original strings before training.
DROP_COLS = [
    "Index", "geohash", "timestamp", "RoadType", "LargeVehicles",
    "Landmarks", "Weather", "geohash_prefix4", "geohash_prefix5",
    "geo_time", "day",
]

# Native categorical columns (string type). Used by CatBoost's native cat
# handling and dropped from DROP_COLS in the catboost branch below. Tree
# boosters (LGB/XGB/HGB) and MLP consume the ``*_label`` ordinal versions
# instead (see apply_static_target_encodings).
CAT_FEATURES: list[str] = [
    "geohash", "RoadType", "Weather", "geohash_prefix4", "geohash_prefix5",
]

# Trajectory lag/rolling feature columns (created by src/trajectory.py).
# Listed explicitly so feature_engineering, models, and trajectory modules
# can't drift apart on what counts as a "trajectory" feature.
TRAJECTORY_FEATURE_COLS: list[str] = [
    "lag_same_day_1", "lag_same_day_2", "lag_same_day_4", "lag_same_day_8",
    "rollmean_same_day_3", "rollmean_same_day_8", "rollstd_same_day_8",
    "lag_prev_day_0", "lag_prev_day_1", "lag_prev_day_4", "prev_day_rollmean_8",
]

# Warm-start aggregate columns produced from the trajectory lags.
WARM_START_COLS: list[str] = [
    "warm_mean", "warm_std", "warm_max", "warm_min",
    "warm_range", "warm_last", "warm_trend",
]


def get_drop_cols(df, model_type: str = "default") -> list:
    """Return DROP_COLS filtered to columns present in *df*, with model-specific extras.

    Centralises the "what columns to drop" decision so the catboost and
    default model branches share one code path and can't drift apart.
    """
    label_cols = [f"{c}_label" for c in CAT_FEATURES] + ["RoadType_encoded", "Weather_encoded"]

    cols = list(DROP_COLS)
    
    if model_type == "catboost":
        # CatBoost retains native categoricals but drops label encodings
        cols = [c for c in cols if c not in CAT_FEATURES]
        cols.extend(c for c in label_cols if c in df.columns)

    # Always drop the target if present in *df*.
    if "demand" in df.columns and "demand" not in cols:
        cols.append("demand")

    # Preserve order, drop dupes, only keep those that exist in df.
    seen: set = set()
    result: list[str] = []
    for c in cols:
        if c in df.columns and c not in seen:
            result.append(c)
            seen.add(c)
    return result
