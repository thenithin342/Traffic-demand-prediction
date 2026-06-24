"""
Hyperparameter tuning using Optuna for LightGBM.
"""

from __future__ import annotations

import optuna
from optuna.pruners import MedianPruner
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

from config import SEED
from src.models.models import _build_fold_cache

def objective(trial: optuna.Trial, train: pd.DataFrame, fold_cache: dict[int, dict]) -> float:
    """Optuna objective function for LightGBM.

    Uses a pre-built fold cache (shared across all Optuna trials) to avoid
    recomputing target encodings + neighbor features on every trial.
    """

    # Define hyperparameter search space
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 4000,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    oof = np.zeros(len(train))
    y_global = train["demand"].values

    for fold, c in fold_cache.items():
        X_tr, y_tr, X_val, y_val, val_idx = (
            c["X_tr"], c["y_tr"], c["X_val"], c["y_val"], c["val_idx"],
        )

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        oof[val_idx] = model.predict(X_val)

    return r2_score(y_global, oof)

def tune_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    neighbor_map: dict,
    n_trials: int = 50,
    fold_cache: dict | None = None,
) -> dict:
    """Run Optuna study to find best LightGBM hyperparameters.

    Builds (or reuses) the fold feature cache once and shares it across all
    Optuna trials — avoids 50× recomputation of target encodings + neighbor
    features. Pass ``fold_cache`` to skip the cache rebuild when the caller
    already built one for the main training loop (P5).

    Adds a MedianPruner (P7) so bad hyperparameter regions stop early.
    """
    print(f"\n--- Starting Optuna Tuning for LightGBM ({n_trials} trials) ---")

    # We can turn off Optuna's verbose logging to keep output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Build fold cache ONCE — shared by all Optuna trials (P5: reuse if passed in)
    if fold_cache is None:
        cache = _build_fold_cache(train, test, neighbor_map, model_types=("default",))
        fold_cache = cache["default"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),  # P7
    )

    def wrapped_objective(trial):
        return objective(trial, train, fold_cache)

    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best OOF R2: {study.best_value:.6f}")
    print(f"  Best Params: {study.best_params}")

    return study.best_params
