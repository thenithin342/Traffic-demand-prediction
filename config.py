"""
Configuration and constants for the Traffic Demand Prediction pipeline.
"""

import os

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# ─── Model Settings ──────────────────────────────────────────
SEED = 42
N_FOLDS = 5

# ─── LightGBM ────────────────────────────────────────────────
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
    "n_estimators": 3000,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}

# ─── XGBoost ─────────────────────────────────────────────────
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
    "n_estimators": 3000,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}

# ─── CatBoost ────────────────────────────────────────────────
CAT_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "iterations": 3000,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "thread_count": -1,
}

# ─── Feature Lists ───────────────────────────────────────────
DROP_COLS = [
    "Index", "geohash", "timestamp", "RoadType", "LargeVehicles",
    "Landmarks", "Weather", "geohash_prefix4", "geohash_prefix5",
    "geo_time",
]

# ─── Target Encoding Smoothing ───────────────────────────────
TARGET_SMOOTHING = 10
