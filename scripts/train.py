"""
save_models.py — full-train refit + serialize all artifacts for inference.

Trains the 5-model ensemble (LightGBM, XGBoost, CatBoost, HistGBM, MLP) on the
FULL training set inside 5-fold CV (CV_MODE from config), then dumps:

- feature_state.pkl       (FeatureState for engineer_features on new data)
- neighbor_map.pkl        (geohash -> 6-nearest-neighbour list)
- geo_cluster_map.pkl     (geohash -> cluster id)
- inference_lookup.pkl    (9 tables: geo/time-slot/cluster demand stats,
                           target encodings, freq encoders)
- fold_models.pkl         (5 lists of 5 fold model objects)
- ensemble_meta.pkl       (weights, method, feature_cols, traj_means,
                           global_demand_mean)

Note: this script does NOT run Optuna tuning or the 80/20 holdout. It reuses
the static + per-fold feature pipeline from `solution.py` and re-implements
the training loops inline so the fitted model objects can be captured.

Reuses:
    src.data_loader.load_datasets
    src.feature_engineering.{engineer_features, FeatureState}
    src.target_encoding.apply_static_target_encodings
    src.neighbor.apply_static_neighbor_features
    src.trajectory.{build_trajectory_features, _add_time_slot}
    src.models.{_build_fold_cache, _lgbm_build, _xgb_build, _cat_build, _hgb_build}
    src.nn_model.MLP_PARAMS
    src.ensemble.build_ensemble
    config.{OUTPUT_DIR, TRAJECTORY_FEATURE_COLS, WARM_START_COLS,
            TARGET_SMOOTHING, get_drop_cols}
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import (
    OUTPUT_DIR,
    TARGET_SMOOTHING,
    TRAJECTORY_FEATURE_COLS,
    WARM_START_COLS,
)

from src.data.data_loader import load_datasets
from src.features.feature_engineering import FeatureState, engineer_features
from src.models.models import (
    _build_fold_cache,
    _cat_build,
    _hgb_build,
    _lgbm_build,
    _xgb_build,
)
from src.features.neighbor import apply_static_neighbor_features
from src.models.nn_model import MLP_PARAMS
from src.features.target_encoding import apply_static_target_encodings
from src.features.trajectory import _add_time_slot, build_trajectory_features


def _train_fold_models(
    fold_cache: dict,
    build_fn,
    name: str,
    y_global: np.ndarray,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Run a build_fn over each fold, return (model_list, oof, test_preds).

    `model_list` is a list of fitted estimators (one per fold). `oof` is the
    full out-of-fold prediction vector (val_idx-assembled). `test_preds` is
    the per-row mean of fold-wise `predict(X_te)` calls.
    """
    n_folds = len(fold_cache)
    first = next(iter(fold_cache.values()))
    oof = np.zeros(len(y_global))
    preds = np.zeros(first["X_te"].shape[0])
    models: list = []

    for fold, c in fold_cache.items():
        model = build_fn(c["X_tr"], c["y_tr"], c["X_val"], c["y_val"])
        oof[c["val_idx"]] = model.predict(c["X_val"])
        preds += model.predict(c["X_te"]) / n_folds
        models.append(model)
        fold_r2 = r2_score(c["y_val"], oof[c["val_idx"]])
        print(f"  [{name}] Fold {fold + 1}/{n_folds} R2 = {fold_r2:.6f}")

    print(f"  [{name}] OOF R2 = {r2_score(y_global, oof):.6f}")
    return models, oof, preds


def _train_mlp_fold_models(
    fold_cache: dict,
    y_global: np.ndarray,
) -> tuple[list, np.ndarray, np.ndarray]:
    """MLP variant: build a sklearn Pipeline per fold (imputer -> scaler -> MLP)."""
    from src.models.nn_model import MLP_PARAMS as _PARAMS  # late-bind for safety

    n_folds = len(fold_cache)
    first = next(iter(fold_cache.values()))
    oof = np.zeros(len(y_global))
    preds = np.zeros(first["X_te"].shape[0])
    pipelines: list = []

    for fold, c in fold_cache.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**_PARAMS)),
        ])
        pipeline.fit(c["X_tr"], c["y_tr"])
        oof[c["val_idx"]] = pipeline.predict(c["X_val"])
        preds += pipeline.predict(c["X_te"]) / n_folds
        pipelines.append(pipeline)
        fold_r2 = r2_score(c["y_val"], oof[c["val_idx"]])
        print(f"  [mlp] Fold {fold + 1}/{n_folds} R2 = {fold_r2:.6f}")

    print(f"  [mlp] OOF R2 = {r2_score(y_global, oof):.6f}")
    return pipelines, oof, preds


def _compute_inference_lookups(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Build the 9-table inference lookup bundle from the FULL train frame."""
    train = train.copy()

    # a. geo_stats
    geo_stats = train.groupby("geohash")["demand"].agg(
        ["mean", "std", "median", "min", "max"]
    )
    geo_stats["rank"] = geo_stats["mean"].rank()
    geo_stats["quantile"] = (
        pd.qcut(geo_stats["mean"], q=4, labels=False, duplicates="drop")
        .fillna(-1)
        .astype(int)
    )
    geo_stats.columns = [
        "geo_demand_mean", "geo_demand_std", "geo_demand_median",
        "geo_demand_min", "geo_demand_max",
        "geo_demand_rank", "geo_demand_quantile",
    ]

    # b. timeslot_stats
    timeslot_stats = (
        train.groupby("time_slot")["demand"]
        .agg(["mean", "std", "median"])
    )
    timeslot_stats.columns = [
        "timeslot_demand_mean", "timeslot_demand_std", "timeslot_demand_median",
    ]

    # c. weather_slot_stats
    weather_slot_stats = (
        train.groupby(["Weather", "time_slot"])["demand"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "weather_slot_mean", "std": "weather_slot_std"})
    )

    # d. cluster_stats
    cluster_stats = train.groupby("geo_cluster")["demand"].agg(
        ["mean", "std", "max", "min"]
    )
    cluster_stats.columns = [
        "cluster_demand_mean", "cluster_demand_std",
        "cluster_demand_max", "cluster_demand_min",
    ]

    # e. triple_stats
    triple_stats = (
        train.groupby(["Weather", "RoadType", "time_slot"])["demand"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "triple_demand_mean", "std": "triple_demand_std"})
    )

    # f. geo_weather_stats
    geo_weather_stats = (
        train.groupby(["geohash", "Weather"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "geo_weather_demand_mean"})
    )

    # g. geo_means
    geo_means = train.groupby("geohash")["demand"].mean().to_dict()

    # h. target_enc_tables (smoothed maps, same formula as _smooth_target_encode_fold)
    if "geohash_prefix4" not in train.columns:
        train["geohash_prefix4"] = train["geohash"].str[:4]
    if "geohash_prefix5" not in train.columns:
        train["geohash_prefix5"] = train["geohash"].str[:5]
    if "geo_time" not in train.columns:
        train["geo_time"] = (
            train["geohash"].astype(str) + "_" + train["time_slot"].astype(str)
        )

    global_mean = train["demand"].mean()
    target_enc_tables: dict[str, dict] = {}
    for col in (
        "geohash", "geohash_prefix4", "geohash_prefix5",
        "geo_time", "Weather", "RoadType",
    ):
        agg = train.groupby(col)["demand"].agg(["mean", "count"])
        agg["smooth"] = (
            (agg["mean"] * agg["count"] + TARGET_SMOOTHING * global_mean)
            / (agg["count"] + TARGET_SMOOTHING)
        )
        target_enc_tables[col] = agg["smooth"].to_dict()

    # i. static_enc (freq + ordinal encoder objects)
    static_enc: dict = {
        "geohash_freq":    train["geohash"].value_counts().to_dict(),
        "prefix4_density": train["geohash_prefix4"].value_counts().to_dict(),
    }
    for col in ("geohash", "geohash_prefix4", "geohash_prefix5"):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        combined = pd.concat([train[[col]], test[[col]]], ignore_index=True)
        oe.fit(combined)
        static_enc[f"{col}_oe"] = oe

    # j. cluster_centroids: per-cluster (lat, lon) mean — needed at inference
    # to compute ``cluster_dist`` (distance from each row to its cluster
    # centroid) without re-running KMeans.
    cluster_centroids = (
        train.groupby("geo_cluster")[["latitude", "longitude"]].mean()
    )
    if -1 in cluster_centroids.index and cluster_centroids.loc[-1].isna().any():
        # ``-1`` cluster only appears when an unknown geohash slips through;
        # fall back to global lat/lon mean.
        cluster_centroids.loc[-1] = [
            train["latitude"].mean(),
            train["longitude"].mean(),
        ]

    return {
        "geo_stats":          geo_stats,
        "timeslot_stats":     timeslot_stats,
        "weather_slot_stats": weather_slot_stats,
        "cluster_stats":      cluster_stats,
        "triple_stats":       triple_stats,
        "geo_weather_stats":  geo_weather_stats,
        "geo_means":          geo_means,
        "target_enc_tables":  target_enc_tables,
        "static_enc":         static_enc,
        "cluster_centroids":  cluster_centroids,
    }


def main() -> None:
    print("=" * 60)
    print("STEP 1: Loading data (FULL train, no split)...")
    print("=" * 60)
    train, test, _ = load_datasets()

    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering...")
    print("=" * 60)
    state = FeatureState()
    train = engineer_features(train, is_train=True, state=state)
    test = engineer_features(test, is_train=False, state=state)

    print("\n" + "=" * 60)
    print("STEP 3: Trajectory / warm-start...")
    print("=" * 60)
    if not config.FOLD_AWARE_TRAJECTORY:
        train, test = build_trajectory_features(train, test)
    else:
        train = _add_time_slot(train)
        test = _add_time_slot(test)

    # Compute trajectory means for inference-time fill.
    # When FOLD_AWARE_TRAJECTORY=True the fold-aware path (called inside
    # _build_fold_cache) only attaches lag/rolling columns to the
    # per-fold frames — the global ``train`` here still lacks them. Run a
    # one-shot non-fold-aware pass purely to recover column means; the
    # resulting frame is discarded.
    if config.FOLD_AWARE_TRAJECTORY:
        _train_global, _ = build_trajectory_features(train.copy(), test.copy())
        traj_means = {
            c: float(_train_global[c].mean())
            for c in (TRAJECTORY_FEATURE_COLS + WARM_START_COLS)
            if c in _train_global.columns
        }
        del _train_global
    else:
        traj_means = {
            c: float(train[c].mean())
            for c in (TRAJECTORY_FEATURE_COLS + WARM_START_COLS)
            if c in train.columns
        }
    print(
        f"  Trajectory means cached for {len(traj_means)} columns: "
        f"{list(traj_means.keys())}"
    )

    print("\n" + "=" * 60)
    print("STEP 4: Static neighbor / clustering...")
    print("=" * 60)
    train, test, neighbor_map = apply_static_neighbor_features(
        train, test, n_clusters=32,
    )

    print("\n" + "=" * 60)
    print("STEP 5: Static target encodings...")
    print("=" * 60)
    train, test = apply_static_target_encodings(train, test)

    # ----------------- Save artifact setup
    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ----------------- 6. Static artifacts
    print("\n" + "=" * 60)
    print("STEP 6: Saving static artifacts...")
    print("=" * 60)
    joblib.dump(state, os.path.join(models_dir, "feature_state.pkl"))
    joblib.dump(neighbor_map, os.path.join(models_dir, "neighbor_map.pkl"))
    geo_cluster_map = dict(zip(train["geohash"], train["geo_cluster"]))
    joblib.dump(geo_cluster_map, os.path.join(models_dir, "geo_cluster_map.pkl"))
    print(f"  feature_state.pkl, neighbor_map.pkl, geo_cluster_map.pkl written.")

    # ----------------- 7. Fold cache (5-fold CV over FULL train)
    print("\n" + "=" * 60)
    print("STEP 7: Building fold cache...")
    print("=" * 60)
    fold_cache_all = _build_fold_cache(
        train, test, neighbor_map,
        model_types=("default", "catboost"),
    )
    fold_cache_default = fold_cache_all["default"]
    fold_cache_cat = fold_cache_all["catboost"]

    # ----------------- 8. Train each base model, capture fold objects
    print("\n" + "=" * 60)
    print("STEP 8: Training base models...")
    print("=" * 60)
    y_global = train["demand"].values
    oof_arrays: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}

    print("\n--- LightGBM ---")
    fold_models_lgb, lgb_oof, lgb_preds = _train_fold_models(
        fold_cache_default, _lgbm_build, "lgb", y_global,
    )

    print("\n--- XGBoost ---")
    fold_models_xgb, xgb_oof, xgb_preds = _train_fold_models(
        fold_cache_default, _xgb_build, "xgb", y_global,
    )

    print("\n--- HistGBM ---")
    fold_models_hgb, hgb_oof, hgb_preds = _train_fold_models(
        fold_cache_default, _hgb_build, "hgb", y_global,
    )

    print("\n--- CatBoost ---")
    fold_models_cat, cat_oof, cat_preds = _train_fold_models(
        fold_cache_cat, _cat_build, "cat", y_global,
    )

    print("\n--- MLP ---")
    fold_models_mlp, mlp_oof, mlp_preds = _train_mlp_fold_models(
        fold_cache_default, y_global,
    )

    oof_arrays.update(lgb=lgb_oof, xgb=xgb_oof, cat=cat_oof, hgb=hgb_oof, mlp=mlp_oof)
    test_preds.update(lgb=lgb_preds, xgb=xgb_preds, cat=cat_preds, hgb=hgb_preds, mlp=mlp_preds)
    fold_models = {
        "lgb": fold_models_lgb,
        "xgb": fold_models_xgb,
        "cat": fold_models_cat,
        "hgb": fold_models_hgb,
        "mlp": fold_models_mlp,
    }

    # ----------------- 9. Ensemble weights
    print("\n" + "=" * 60)
    print("STEP 9: Building ensemble...")
    print("=" * 60)
    oof_list = [oof_arrays[k] for k in ("lgb", "xgb", "cat", "hgb", "mlp")]
    test_list = [test_preds[k] for k in ("lgb", "xgb", "cat", "hgb", "mlp")]
    names = ["LightGBM", "XGBoost", "CatBoost", "HistGBM", "MLP"]

    from src.models.ensemble import build_ensemble
    _, _, oof_r2, method, weights, stacker_models = build_ensemble(
        y_global, oof_list, test_list, names,
    )

    feature_cols = list(next(iter(fold_cache_default.values()))["X_tr"].columns)
    global_demand_mean = float(train["demand"].mean())

    ensemble_meta = {
        "weights":            np.asarray(weights),
        "method":             method,
        "feature_cols":       feature_cols,
        "traj_means":         traj_means,
        "global_demand_mean": global_demand_mean,
    }
    joblib.dump(ensemble_meta, os.path.join(models_dir, "ensemble_meta.pkl"))
    joblib.dump(stacker_models, os.path.join(models_dir, "xgb_stacker.pkl"))
    print(
        f"  Stacker models saved: "
        f"{'yes (' + str(len(stacker_models)) + ' boosters)' if stacker_models else 'no'} "
        f"(method={method})"
    )

    # ----------------- 10. Inference lookup tables
    print("\n" + "=" * 60)
    print("STEP 10: Computing inference lookup tables...")
    print("=" * 60)
    inference_lookup = _compute_inference_lookups(train, test)
    joblib.dump(inference_lookup, os.path.join(models_dir, "inference_lookup.pkl"))

    # ----------------- 11. Fold models + size summary
    print("\n" + "=" * 60)
    print("STEP 11: Saving fold models...")
    print("=" * 60)
    joblib.dump(fold_models, os.path.join(models_dir, "fold_models.pkl"))

    print("\n" + "=" * 60)
    print("SAVED ARTIFACTS")
    print("=" * 60)
    for fname in (
        "feature_state.pkl",
        "neighbor_map.pkl",
        "geo_cluster_map.pkl",
        "inference_lookup.pkl",
        "fold_models.pkl",
        "ensemble_meta.pkl",
        "xgb_stacker.pkl",
    ):
        path = os.path.join(models_dir, fname)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {fname:<25} {size_mb:>8.2f} MB")
    print(f"\nEnsemble OOF R² (full train refit, leak-prone): {oof_r2:.6f}")
    print(f"Selected method: {method}   weights: {weights}")
    print(f"Trajectory means cached for: {len(traj_means)} columns")
    print(f"Feature columns (final fold cache): {len(feature_cols)}")
    print("Done.")


if __name__ == "__main__":
    main()