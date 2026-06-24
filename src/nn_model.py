"""
Multi-Layer Perceptron (Neural Net) model for ensemble diversity.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from config import SEED

# MLP params
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.01,
    "batch_size": 256,
    "learning_rate_init": 0.001,
    "max_iter": 200,
    "tol": 1e-4,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": SEED,
}

def train_mlp(
    train: pd.DataFrame,
    fold_cache: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train MLPRegressor with CV using a pre-built fold cache; returns OOF/test preds and scores."""
    n_folds = len(fold_cache)
    n_test = next(iter(fold_cache.values()))["X_te"].shape[0]
    oof = np.zeros(len(train))
    preds = np.zeros(n_test)
    scores: list[float] = []

    y_global = train["demand"].values

    for fold, c in fold_cache.items():
        X_tr, y_tr, X_val, y_val, val_idx = (
            c["X_tr"], c["y_tr"], c["X_val"], c["y_val"], c["val_idx"],
        )
        print(f"  Fold {fold + 1}/{n_folds} ...", end=" ")

        # Build pipeline: Impute NaNs -> Standardize -> MLP
        # NNs require standardized features without missing values.
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**MLP_PARAMS))
        ])

        pipeline.fit(X_tr, y_tr)

        X_te = c["X_te"]
        oof[val_idx] = pipeline.predict(X_val)
        preds += pipeline.predict(X_te) / n_folds
        fold_r2 = r2_score(y_val, oof[val_idx])
        scores.append(fold_r2)
        print(f"R2 = {fold_r2:.6f}")

    print(f"  OOF R2 = {r2_score(y_global, oof):.6f}")
    return oof, preds, scores
