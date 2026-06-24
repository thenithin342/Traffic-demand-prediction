"""
Model training - LightGBM, XGBoost, CatBoost, and HistGradientBoosting.

Integrated with out-of-fold target encoding and neighbor features computed
inside the 5-fold CV loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from config import (
    SEED, N_FOLDS, LGB_PARAMS, XGB_PARAMS, CAT_PARAMS, HGB_PARAMS,
    CAT_FEATURES, get_drop_cols,
)
import config  # full module reference so CV_MODE lookups are live
from src.features.target_encoding import compute_target_dependent_encodings
from src.features.neighbor import compute_target_dependent_neighbor_features
from src.features.trajectory import build_trajectory_features_fold_aware as _traj_fold_aware


def _make_cv_splits(n_samples: int, groups: np.ndarray | None = None):
    """Return an iterable of (train_idx, val_idx) splits.

    Behaviour is controlled by ``config.CV_MODE``:

    * ``"kfold"`` (default): random KFold, no grouping. Production
      behaviour. Fold splits are independent of the target/feature
      structure, so OOF R² is a good proxy for held-out test R².
    * ``"groupkfold_day"``: ``sklearn.GroupKFold`` grouped on the
      ``day`` column. Used as a leakage-sanity check for
      ``src/trajectory.py`` — see the module docstring there.
      GroupKFold forces each fold to validate on rows whose day
      never appears in the corresponding training fold, which is a
      much harder extrapolation regime. If OOF R² only drops a few
      percent relative to KFold, the trajectory leakage is small and
      the production KFold setting is safe. Note: GroupKFold
      requires ``n_splits`` to divide the number of unique groups, so
      the effective number of folds is the number of unique ``day``
      values (currently 2 in this dataset).

    Parameters
    ----------
    n_samples : int
        Number of rows in the training set.
    groups : np.ndarray | None
        Group labels per row (e.g. ``train["day"].values``). Required
        when ``CV_MODE == "groupkfold_day"``.
    """
    if config.CV_MODE == "groupkfold_day":
        if groups is None:
            raise ValueError(
                "CV_MODE='groupkfold_day' requires passing `groups` "
                "(e.g. train['day'].values) to _make_cv_splits."
            )
        n_splits = min(N_FOLDS, len(np.unique(groups)))   # = 2 when only 2 unique days
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.zeros(n_samples), groups=groups)), f"GroupKFold(day, n_splits={n_splits})"

    # Default: random KFold.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(kf.split(np.zeros(n_samples))), "KFold"


def _prep_fold_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    neighbor_map: dict,
    model_type: str = "default",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Compute target-dependent features on the training fold, map to validation/test, and return X/y.

    NOTE: Heavy operation — target encodings + neighbor features recomputed per call.
    Use ``_build_fold_cache`` to compute once per fold and share across all base models.
    """
    # 1. Compute target dependent features using train fold only
    tr_feat, val_feat, te_feat = compute_target_dependent_encodings(train, val, test)
    tr_feat, val_feat, te_feat = compute_target_dependent_neighbor_features(tr_feat, val_feat, te_feat, neighbor_map)

    # 2. Extract targets
    y_tr = tr_feat["demand"].values
    y_val = val_feat["demand"].values

    # 3. Drop non-feature and target columns. CatBoost needs categorical
    # columns preserved as strings, so the cast happens before the drop
    # computation; ``get_drop_cols`` then unifies both model branches.
    if model_type == "catboost":
        for df in (tr_feat, val_feat, te_feat):
            for c in CAT_FEATURES:
                if c in df.columns:
                    df[c] = df[c].astype(str)

    drop_cols = get_drop_cols(tr_feat, model_type=model_type)

    X_tr = tr_feat.drop(columns=drop_cols)
    X_val = val_feat.drop(columns=drop_cols)

    # ``demand`` may be in drop_cols but is not in test.
    drop_cols_te = [c for c in drop_cols if c != "demand" and c in te_feat.columns]
    X_te = te_feat.drop(columns=drop_cols_te)

    # Ensure same columns are present and in the same order
    common_cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[common_cols]
    X_val = X_val[common_cols]
    X_te = X_te[common_cols]

    # Loud failure on silent column drift: if either side has columns the
    # other lacks, a feature was engineered asymmetrically and the model
    # would silently train/predict on different feature spaces.
    missing_in_test = set(X_tr.columns) - set(X_te.columns)
    missing_in_train = set(X_te.columns) - set(X_tr.columns)
    if missing_in_test or missing_in_train:
        raise ValueError(
            f"Column mismatch after feature engineering — "
            f"missing in test: {sorted(missing_in_test) or 'none'}, "
            f"missing in train: {sorted(missing_in_train) or 'none'}. "
            f"Investigate feature_engineering.py or DROP_COLS in config.py."
        )

    return X_tr, y_tr, X_val, y_val, X_te


def _build_fold_cache(
    train: pd.DataFrame,
    test: pd.DataFrame,
    neighbor_map: dict,
    model_types: tuple[str, ...] = ("default", "catboost"),
) -> dict[str, dict[int, dict]]:
    """Build a per-fold feature cache, computed once per fold, shared across base models.

    The expensive target-encoding + neighbor-feature computation is run **once per fold**
    for each requested ``model_type`` (CatBoost needs categorical columns preserved as strings
    and a different drop-cols set, hence the separate cache entry).

    Parameters
    ----------
    train, test, neighbor_map : passed through to ``_prep_fold_data``.
    model_types : which caches to build. Default builds both ``"default"`` (LGB/XGB/HGB/MLP)
        and ``"catboost"``.

    Returns
    -------
    dict
        ``{model_type: {fold_idx: {"X_tr": ..., "y_tr": ..., "X_val": ..., "y_val": ...,
        "X_te": ..., "val_idx": np.ndarray, "tr_idx": np.ndarray}}}``
    """
    # Pull day-level groups out of the train frame so GroupKFold works.
    groups = train["day"].values if "day" in train.columns and config.CV_MODE == "groupkfold_day" else None
    splits, mode = _make_cv_splits(len(train), groups=groups)
    print(f"  Building fold cache (CV mode: {mode}, folds={len(splits)}, model_types={model_types})")

    cache: dict[str, dict[int, dict]] = {mt: {} for mt in model_types}
    n_folds = len(splits)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        train_fold = train.iloc[tr_idx].copy()
        val_fold = train.iloc[val_idx].copy()

        # Conditionally rebuild trajectory per fold to eliminate cross-fold
        # lag/rolling leak (C1). Legacy path reuses the globally-built test
        # set; fold-aware path masks val_fold demand to NaN before grid build
        # and returns a fresh test set.
        if config.FOLD_AWARE_TRAJECTORY:
            train_fold, val_fold, test_for_fold = _traj_fold_aware(train_fold, val_fold, test)
        else:
            test_for_fold = test

        for mt in model_types:
            X_tr, y_tr, X_val, y_val, X_te = _prep_fold_data(
                train_fold, val_fold, test_for_fold, neighbor_map, model_type=mt,
            )
            cache[mt][fold] = {
                "X_tr": X_tr,
                "y_tr": y_tr,
                "X_val": X_val,
                "y_val": y_val,
                "X_te": X_te,
                "val_idx": val_idx,
                "tr_idx": tr_idx,
            }

        print(f"    Fold {fold + 1}/{n_folds} cached")

    return cache


def _run_cv(
    name: str,
    build_model_fn,
    fold_cache: dict[int, dict],
    y_global: np.ndarray,
    return_importances: bool = False,
) -> tuple:
    """Generic CV runner — eliminates boilerplate shared by all 4 train_* wrappers.

    Parameters
    ----------
    name : short label for logging (e.g. "LightGBM").
    build_model_fn : callable ``(X_tr, y_tr, X_val, y_val) -> fitted_model``.
        Model is responsible for any early-stopping wiring (via eval_set etc.).
    fold_cache : pre-built fold cache from ``_build_fold_cache``.
    y_global : full-train target vector, used for the OOF R² summary.
    return_importances : if True, accumulate and return feature importances.

    Returns
    -------
    tuple
        ``(oof, test_preds, scores)`` or ``(oof, test_preds, scores, importances)``.
    """
    n_folds = len(fold_cache)
    n_test = next(iter(fold_cache.values()))["X_te"].shape[0]
    oof = np.zeros(len(y_global))
    preds = np.zeros(n_test)
    scores: list[float] = []
    importances: np.ndarray | None = None

    for fold, c in fold_cache.items():
        X_tr, y_tr = c["X_tr"], c["y_tr"]
        X_val, y_val, X_te, val_idx = c["X_val"], c["y_val"], c["X_te"], c["val_idx"]
        print(f"  [{name}] Fold {fold + 1}/{n_folds} ...", end=" ")

        model = build_model_fn(X_tr, y_tr, X_val, y_val)
        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_te) / n_folds
        fold_r2 = r2_score(y_val, oof[val_idx])
        scores.append(fold_r2)

        if return_importances:
            if importances is None:
                importances = np.zeros(X_tr.shape[1], dtype=np.float64)
            importances += model.feature_importances_ / n_folds
        print(f"R2 = {fold_r2:.6f}")

    print(f"  [{name}] OOF R2 = {r2_score(y_global, oof):.6f}")
    return (oof, preds, scores, importances) if return_importances else (oof, preds, scores)


def _lgbm_build(X_tr, y_tr, X_val, y_val):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _xgb_build(X_tr, y_tr, X_val, y_val):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _cat_build(X_tr, y_tr, X_val, y_val):
    cat_cols = [c for c in CAT_FEATURES if c in X_tr.columns]
    model = cb.CatBoostRegressor(**CAT_PARAMS)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_cols)
    return model


def _hgb_build(X_tr, y_tr, X_val, y_val):
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(X_tr, y_tr)
    return model


def train_lightgbm(
    train: pd.DataFrame,
    fold_cache: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray]:
    """Train LightGBM with CV using a pre-built fold cache; returns OOF/test preds, scores, importances.

    ``fold_cache`` is the per-fold cache for model_type ``"default"``.
    Test predictions are averaged across folds.
    """
    return _run_cv(
        "LightGBM", _lgbm_build, fold_cache,
        train["demand"].values, return_importances=True,
    )


def train_xgboost(
    train: pd.DataFrame,
    fold_cache: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train XGBoost with CV using a pre-built fold cache (model_type='default')."""
    return _run_cv("XGBoost", _xgb_build, fold_cache, train["demand"].values)


def train_catboost(
    train: pd.DataFrame,
    fold_cache: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train CatBoost with CV using a pre-built fold cache (model_type='catboost')."""
    return _run_cv("CatBoost", _cat_build, fold_cache, train["demand"].values)


def train_histgbm(
    train: pd.DataFrame,
    fold_cache: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train HistGradientBoostingRegressor with CV using a pre-built fold cache (model_type='default')."""
    return _run_cv("HistGBM", _hgb_build, fold_cache, train["demand"].values)
