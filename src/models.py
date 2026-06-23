"""
Model training — LightGBM, XGBoost, and CatBoost with K-Fold CV.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from config import SEED, N_FOLDS, LGB_PARAMS, XGB_PARAMS, CAT_PARAMS


def train_lightgbm(
    X_train: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[float], object]:
    """Train LightGBM with K-Fold CV.

    Returns
    -------
    tuple
        ``(oof_preds, test_preds, fold_scores, last_model)``
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{N_FOLDS} ...", end=" ")
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        fold_r2 = r2_score(y_val, oof[val_idx])
        scores.append(fold_r2)
        print(f"R2 = {fold_r2:.6f}")

    print(f"  OOF R2 = {r2_score(y, oof):.6f}")
    return oof, preds, scores, model


def train_xgboost(
    X_train: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train XGBoost with K-Fold CV.

    Returns
    -------
    tuple
        ``(oof_preds, test_preds, fold_scores)``
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{N_FOLDS} ...", end=" ")
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        fold_r2 = r2_score(y_val, oof[val_idx])
        scores.append(fold_r2)
        print(f"R2 = {fold_r2:.6f}")

    print(f"  OOF R2 = {r2_score(y, oof):.6f}")
    return oof, preds, scores


def train_catboost(
    X_train: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train CatBoost with K-Fold CV.

    Returns
    -------
    tuple
        ``(oof_preds, test_preds, fold_scores)``
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{N_FOLDS} ...", end=" ")
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = cb.CatBoostRegressor(**CAT_PARAMS)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / N_FOLDS
        fold_r2 = r2_score(y_val, oof[val_idx])
        scores.append(fold_r2)
        print(f"R2 = {fold_r2:.6f}")

    print(f"  OOF R2 = {r2_score(y, oof):.6f}")
    return oof, preds, scores
