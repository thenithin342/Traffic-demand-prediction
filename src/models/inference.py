"""
Inference module — load persisted artifacts and score new data with the
trained 5-model XGBStack ensemble.

Loads artifacts from ``output/models/`` (created by ``save_models.py``) and
exposes ``load_artifacts`` (cached), ``prepare_single_row``,
``prepare_batch_frame``, ``predict_single``, and ``predict_batch``.

Critical: the selected ensemble method is ``XGBStack``. The persisted
``xgb_stacker.pkl`` contains 5 XGBoost boosters trained on the base-model
OOF matrix. At inference we average the 5 stacker boosters, never the
``ensemble_meta["weights"]`` (those are equal priors for non-stack methods
and would be a meaningless linear blend for XGBStack).
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TRAJECTORY_FEATURE_COLS, WARM_START_COLS
from src.features.feature_engineering import engineer_features
from src.data.geohash_decoder import decode_geohash


def _ensure_cluster_centroids(lookup: dict, cluster_map: dict) -> None:
    """Inject ``cluster_centroids`` into *lookup* if missing.

    Older ``inference_lookup.pkl`` saves (predating the centroid addition in
    ``save_models.py``) lack this key. ``_apply_static_lookups`` reads it to
    compute ``cluster_dist``, so we recompute here from the
    ``geo_cluster_map`` (geohash -> cluster_id) plus a one-time geohash
    decode. Mutates *lookup* in place — no-op when the key already exists.
    """
    if "cluster_centroids" in lookup:
        return
    if not cluster_map:
        return
    decoded = pd.DataFrame(
        [
            (gh, *decode_geohash(gh), cid)
            for gh, cid in cluster_map.items()
        ],
        columns=["geohash", "latitude", "longitude", "geo_cluster"],
    )
    centroids = (
        decoded.groupby("geo_cluster")[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    # ``-1`` cluster (unknown geohash) may have all-NaN rows if no member.
    # Fall back to the global lat/lon mean so the lookup never returns NaN.
    global_lat = float(decoded["latitude"].mean())
    global_lon = float(decoded["longitude"].mean())
    centroids["latitude"] = centroids["latitude"].fillna(global_lat)
    centroids["longitude"] = centroids["longitude"].fillna(global_lon)
    lookup["cluster_centroids"] = centroids

# Module-level cache so ``load_artifacts()`` is a no-op after the first call.
_ARTIFACTS_CACHE: dict | None = None

# CatBoost was trained on native categorical strings (config.CAT_FEATURES).
# When we have to fill a missing column at inference, we use empty string
# for these and 0 for everything else.
_CAT_NATIVE_COLS = frozenset({
    "geohash", "RoadType", "Weather", "geohash_prefix4", "geohash_prefix5",
})


# ============================================================================
# Artifact loading
# ============================================================================
def load_artifacts(models_dir: str | None = None) -> dict:
    """Load all 7 persisted artifacts; cache the result for subsequent calls.

    Returns a dict with keys:
        feature_state, neighbor_map, geo_cluster_map, inference_lookup,
        fold_models, ensemble_meta, stacker_models.
    """
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is not None:
        return _ARTIFACTS_CACHE

    if models_dir is None:
        models_dir = os.path.join(OUTPUT_DIR, "models")

    feature_state = joblib.load(os.path.join(models_dir, "feature_state.pkl"))
    neighbor_map = joblib.load(os.path.join(models_dir, "neighbor_map.pkl"))
    geo_cluster_map = joblib.load(os.path.join(models_dir, "geo_cluster_map.pkl"))
    inference_lookup = joblib.load(os.path.join(models_dir, "inference_lookup.pkl"))
    # Older saves omit ``cluster_centroids``; recompute from cluster_map so
    # _apply_static_lookups doesn't KeyError at inference time.
    _ensure_cluster_centroids(inference_lookup, geo_cluster_map)
    fold_models = joblib.load(os.path.join(models_dir, "fold_models.pkl"))
    ensemble_meta = joblib.load(os.path.join(models_dir, "ensemble_meta.pkl"))

    stacker_models = None
    stacker_path = os.path.join(models_dir, "xgb_stacker.pkl")
    if os.path.exists(stacker_path):
        stacker_models = joblib.load(stacker_path)

    _ARTIFACTS_CACHE = {
        "feature_state": feature_state,
        "neighbor_map": neighbor_map,
        "geo_cluster_map": geo_cluster_map,
        "inference_lookup": inference_lookup,
        "fold_models": fold_models,
        "ensemble_meta": ensemble_meta,
        "stacker_models": stacker_models,
    }
    return _ARTIFACTS_CACHE


def reset_cache() -> None:
    """Clear the artifacts cache (used by tests / hot-reload)."""
    global _ARTIFACTS_CACHE
    _ARTIFACTS_CACHE = None


# ============================================================================
# Feature preparation
# ============================================================================
def _build_input_frame(records: list[dict]) -> pd.DataFrame:
    """Coerce a list of row dicts into a typed DataFrame with the right columns."""
    df = pd.DataFrame.from_records(records)
    # Enforce types matching the training pipeline.
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df["NumberofLanes"] = pd.to_numeric(df["NumberofLanes"], errors="coerce").astype("Int64")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    return df


def _decode_geohash_column(geohash_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Decode a geohash column to (latitude, longitude); cache unique lookups."""
    unique_ghs = geohash_series.unique()
    decoded = {gh: decode_geohash(gh) for gh in unique_ghs}
    lat = geohash_series.map(lambda x: decoded[x][0])
    lon = geohash_series.map(lambda x: decoded[x][1])
    return lat, lon


def _apply_static_lookups(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Attach every target-dependent and target-independent lookup column.

    Operates on a DataFrame already passed through ``engineer_features`` so
    the engineer step has produced ``latitude``/``longitude``/
    ``geohash_prefix4``/``geohash_prefix5``/``time_slot``.
    """
    df = df.copy()
    lookup = artifacts["inference_lookup"]
    cluster_map = artifacts["geo_cluster_map"]
    neighbor_map = artifacts["neighbor_map"]
    meta = artifacts["ensemble_meta"]
    global_demand_mean: float = meta["global_demand_mean"]

    # --- Cluster assignment from the saved map (engineer_features doesn't
    # know about clusters; apply_static_neighbor_features did it on full data).
    df["geo_cluster"] = df["geohash"].map(cluster_map).fillna(-1).astype(int)
    cluster_sizes = pd.Series(cluster_map).value_counts().to_dict()
    df["cluster_size"] = df["geo_cluster"].map(cluster_sizes).fillna(0).astype(int)

    # --- cluster_dist via saved centroids.
    centroids: pd.DataFrame = lookup["cluster_centroids"]
    if "centroid_lat" not in centroids.columns:
        # Reindex defensively in case the dataframe was saved with a
        # default integer index.
        centroids = centroids.reset_index()
    cent_lookup_lat = centroids.set_index("geo_cluster")["latitude"].to_dict()
    cent_lookup_lon = centroids.set_index("geo_cluster")["longitude"].to_dict()
    cent_lat = df["geo_cluster"].map(cent_lookup_lat).fillna(df["latitude"].mean())
    cent_lon = df["geo_cluster"].map(cent_lookup_lon).fillna(df["longitude"].mean())
    df["cluster_dist"] = np.hypot(df["latitude"] - cent_lat, df["longitude"] - cent_lon)

    # --- Static encodings (freq + ordinal labels from the saved encoders).
    static_enc = lookup["static_enc"]
    df["geohash_freq"] = df["geohash"].map(static_enc["geohash_freq"]).fillna(0)
    df["prefix4_density"] = (
        df["geohash_prefix4"].map(static_enc["prefix4_density"]).fillna(0)
    )
    for col in ("geohash", "geohash_prefix4", "geohash_prefix5"):
        oe = static_enc[f"{col}_oe"]
        df[f"{col}_label"] = (
            oe.transform(df[[col]]).astype(int).flatten()
        )

    # --- Target encodings (smoothed maps from full train).
    df["geo_time"] = (
        df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)
    )
    target_enc_tables = lookup["target_enc_tables"]
    for col in (
        "geohash", "geohash_prefix4", "geohash_prefix5",
        "geo_time", "Weather", "RoadType",
    ):
        mapping = target_enc_tables[col]
        df[f"{col}_target_enc"] = (
            df[col].map(mapping).fillna(global_demand_mean).astype(float)
        )

    # --- Geo demand stats (per-geohash mean/std/median/min/max/rank/quantile).
    geo_stats: pd.DataFrame = lookup["geo_stats"].copy()
    if geo_stats.index.name is None:
        # The index is named via reset_index in older saves; rebuild it.
        if "geohash" in geo_stats.columns:
            geo_stats = geo_stats.set_index("geohash")
        else:
            geo_stats.index.name = "geohash"
    geo_std_med = geo_stats["geo_demand_std"].median()
    for col in (
        "geo_demand_mean", "geo_demand_std", "geo_demand_median",
        "geo_demand_min", "geo_demand_max",
        "geo_demand_rank", "geo_demand_quantile",
    ):
        fallback = (
            geo_std_med if col == "geo_demand_std"
            else global_demand_mean if "mean" in col or "median" in col
            else 0.0
        )
        df[col] = df["geohash"].map(geo_stats[col]).fillna(fallback)

    # --- Time-slot demand stats.
    ts_stats: pd.DataFrame = lookup["timeslot_stats"].copy()
    if ts_stats.index.name is None:
        ts_stats.index.name = "time_slot"
    for col in (
        "timeslot_demand_mean", "timeslot_demand_std", "timeslot_demand_median",
    ):
        df[col] = df["time_slot"].map(ts_stats[col]).fillna(global_demand_mean)

    # --- (Weather, time_slot) stats via merge.
    ws: pd.DataFrame = lookup["weather_slot_stats"]
    ws_fallback_mean = ws["weather_slot_mean"].mean() if not ws.empty else global_demand_mean
    ws_fallback_std = ws["weather_slot_std"].mean() if not ws.empty else 0.0
    df = df.merge(ws, on=["Weather", "time_slot"], how="left")
    df["weather_slot_mean"] = df["weather_slot_mean"].fillna(ws_fallback_mean)
    df["weather_slot_std"] = df["weather_slot_std"].fillna(ws_fallback_std)

    # --- Cluster demand stats from the lookup.
    cluster_stats: pd.DataFrame = lookup["cluster_stats"].copy()
    if cluster_stats.index.name is None:
        if "geo_cluster" in cluster_stats.columns:
            cluster_stats = cluster_stats.set_index("geo_cluster")
        else:
            cluster_stats.index.name = "geo_cluster"
    for col in (
        "cluster_demand_mean", "cluster_demand_std",
        "cluster_demand_max", "cluster_demand_min",
    ):
        fallback = cluster_stats[col].median() if col in cluster_stats else global_demand_mean
        df[col] = df["geo_cluster"].map(cluster_stats[col]).fillna(fallback)

    # --- (Weather, RoadType, time_slot) triple stats via merge.
    triple: pd.DataFrame = lookup["triple_stats"]
    tri_fallback_mean = triple["triple_demand_mean"].mean() if not triple.empty else global_demand_mean
    tri_fallback_std = triple["triple_demand_std"].mean() if not triple.empty else 0.0
    df = df.merge(triple, on=["Weather", "RoadType", "time_slot"], how="left")
    df["triple_demand_mean"] = df["triple_demand_mean"].fillna(tri_fallback_mean)
    df["triple_demand_std"] = df["triple_demand_std"].fillna(tri_fallback_std)

    # --- (geohash, Weather) geo_weather_demand_mean via merge.
    gw: pd.DataFrame = lookup["geo_weather_stats"]
    df = df.merge(gw, on=["geohash", "Weather"], how="left")
    df["geo_weather_demand_mean"] = df["geo_weather_demand_mean"].fillna(global_demand_mean)

    # --- Neighbor demand stats (per row, 6 nearest neighbours).
    geo_means: dict = lookup["geo_means"]
    nbr_means = []
    nbr_maxs = []
    nbr_stds = []
    nbr_wts = []
    for gh in df["geohash"]:
        nbrs = neighbor_map.get(gh, [])
        if not nbrs:
            nbr_means.append(global_demand_mean)
            nbr_maxs.append(global_demand_mean)
            nbr_stds.append(0.0)
            nbr_wts.append(global_demand_mean)
            continue
        vals = np.array([geo_means.get(n, global_demand_mean) for n, _ in nbrs])
        dists = np.array([d for _, d in nbrs])
        weights = 1.0 / (dists + 1e-5)
        w_sum = weights.sum()
        nbr_means.append(float(vals.mean()))
        nbr_maxs.append(float(vals.max()))
        nbr_stds.append(float(vals.std()))
        nbr_wts.append(float((vals * weights).sum() / w_sum))
    df["neighbor_demand_mean"] = nbr_means
    df["neighbor_demand_max"] = nbr_maxs
    df["neighbor_demand_std"] = nbr_stds
    df["neighbor_weighted_mean"] = nbr_wts

    # --- Trajectory / warm-start columns: fill with cached means.
    traj_means: dict = meta.get("traj_means", {})
    for col in TRAJECTORY_FEATURE_COLS + WARM_START_COLS:
        if col not in df.columns:
            df[col] = traj_means.get(col, 0.0)

    # --- Engineered time-slot features used by some models.
    if "time_slot" in df.columns:
        df["slot_frac"] = df["time_slot"] / 96.0
        df["slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
        df["slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)

    return df


def _align_to_feature_cols(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Reindex to ``ensemble_meta["feature_cols"]``; fill missing cols with 0."""
    feature_cols: list[str] = artifacts["ensemble_meta"]["feature_cols"]
    # Add any column not in the expected list as 0 (shouldn't happen, but safe).
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df.reindex(columns=feature_cols, fill_value=0.0)


def prepare_single_row(row_dict: dict, artifacts: dict) -> pd.DataFrame:
    """Prepare a 1-row feature matrix aligned to the trained model columns."""
    df = _build_input_frame([row_dict])
    return _prepare_frame(df, artifacts)


def prepare_batch_frame(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Prepare an N-row feature matrix aligned to the trained model columns.

    Accepts a DataFrame already containing the 9 input columns. Runs the
    same vectorized feature pipeline as ``prepare_single_row`` (no Python
    per-row loops except for the neighbor stats, which are still < 1 ms
    for batches of a few thousand rows).
    """
    return _prepare_frame(df.copy(), artifacts)


def _prepare_frame(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Shared internal pipeline: engineer features -> static lookups -> align."""
    # 1. Engineer features (temporal/spatial/categorical/interaction).
    state = artifacts["feature_state"]
    df = engineer_features(df, is_train=False, state=state)

    # 2. Apply all static lookup columns (target encodings, demand stats,
    #    neighbor/cluster features, etc.).
    df = _apply_static_lookups(df, artifacts)

    # 3. Reindex to the model's expected feature column order.
    return _align_to_feature_cols(df, artifacts)


# ============================================================================
# Prediction
# ============================================================================
def _score_base_models(X: pd.DataFrame, fold_models: dict) -> dict[str, np.ndarray]:
    """Run all 5 base models on X; return dict {algo: 1d prediction array}.

    Each base model was trained on its own column ordering:
    - LGB/XGB/HGB/MLP use the default ``feature_cols`` order (already
      aligned by ``_align_to_feature_cols``).
    - CatBoost was trained on a different ordering (native categorical
      strings, ``*_label`` columns dropped), so we reindex to the model's
      own ``feature_names_`` list before predicting.
    """
    preds: dict[str, np.ndarray] = {}
    for algo, models in fold_models.items():
        algo_preds: list[np.ndarray] = []
        for m in models:
            X_in = X
            cat_names = getattr(m, "feature_names_", None)
            if cat_names and list(X_in.columns) != list(cat_names):
                # Reindex to the model's training column ordering. CatBoost
                # stores native categorical columns as strings; fill missing
                # values with empty string for those and 0 for the rest.
                cols = list(cat_names)
                missing = [c for c in cols if c not in X_in.columns]
                if missing:
                    extras = pd.DataFrame(
                        {c: ("" if c in _CAT_NATIVE_COLS else 0.0)
                         for c in missing},
                        index=X_in.index,
                    )
                    X_in = pd.concat([X_in, extras], axis=1)
                X_in = X_in[cols]
            algo_preds.append(np.asarray(m.predict(X_in), dtype=np.float64))
        preds[algo] = np.mean(algo_preds, axis=0)
    return preds


def _stack_predictions(
    base_preds: dict[str, np.ndarray],
    artifacts: dict,
) -> np.ndarray:
    """Combine per-algo base preds into a final demand array.

    XGBStack path: average 5 fitted stacker boosters.
    Fallback path: linear blend using ``ensemble_meta["weights"]``.
    """
    method = artifacts["ensemble_meta"].get("method", "EqualMean")
    order = ("lgb", "xgb", "cat", "hgb", "mlp")
    meta = np.column_stack([base_preds[k] for k in order])

    if method == "XGBStack":
        stacker_models = artifacts.get("stacker_models")
        if stacker_models:
            stacked = np.array(
                [m.predict(meta) for m in stacker_models], dtype=np.float64
            )
            return stacked.mean(axis=0)
        # Fallback: equal mean (should not happen if xgb_stacker.pkl exists).
        return meta.mean(axis=1)

    weights = np.asarray(artifacts["ensemble_meta"]["weights"], dtype=np.float64)
    if weights.shape[0] != meta.shape[1]:
        weights = np.ones(meta.shape[1]) / meta.shape[1]
    return meta @ weights


def predict_single(row_dict: dict, artifacts: dict) -> float:
    """Score a single row dict; return a clipped demand value in [0, 1]."""
    X = prepare_single_row(row_dict, artifacts)
    base_preds = _score_base_models(X, artifacts["fold_models"])
    final = _stack_predictions(base_preds, artifacts)
    return float(np.clip(final[0], 0.0, 1.0))


def predict_batch(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Score a DataFrame of rows; return a clipped demand array in [0, 1]."""
    if df.empty:
        return np.zeros(0, dtype=np.float64)
    X = prepare_batch_frame(df, artifacts)
    base_preds = _score_base_models(X, artifacts["fold_models"])
    final = _stack_predictions(base_preds, artifacts)
    return np.clip(final, 0.0, 1.0)


def predict_batch_from_records(records: list[dict], artifacts: dict) -> np.ndarray:
    """Convenience wrapper: take a list of row dicts, return demand array."""
    if not records:
        return np.zeros(0, dtype=np.float64)
    df = _build_input_frame(records)
    return predict_batch(df, artifacts)
