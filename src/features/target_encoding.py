"""
Target encoding, frequency encoding, and demand-statistic features.

Modified to separate static target-independent encodings (computed once globally)
from target-dependent encodings (computed dynamically inside each CV fold).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold as _KFold
from sklearn.preprocessing import OrdinalEncoder

from config import TARGET_SMOOTHING

# Module-level column lists — produced by ``compute_target_dependent_encodings``.
# ``solution.py`` imports these instead of running a mock-build to derive the
# feature column list (C12: mock-build leaked ``demand`` into train→train enc).
TARGET_DEPENDENT_COLS: list[str] = [
    "geohash_target_enc", "geohash_prefix4_target_enc", "geohash_prefix5_target_enc",
    "geo_time_target_enc", "Weather_target_enc", "RoadType_target_enc",
    "geo_demand_mean", "geo_demand_std", "geo_demand_median",
    "geo_demand_min", "geo_demand_max", "geo_demand_rank", "geo_demand_quantile",
    "timeslot_demand_mean", "timeslot_demand_std", "timeslot_demand_median",
    "weather_slot_mean", "weather_slot_std",
]


def apply_static_target_encodings(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply static, target-independent encodings (frequency and label encoding) globally."""
    train = train.copy()
    test = test.copy()

    # ------- Frequency encoding (leak-free: fit on train only, map to test)
    geo_freq = train["geohash"].value_counts().to_dict()
    train["geohash_freq"] = train["geohash"].map(geo_freq)
    test["geohash_freq"] = test["geohash"].map(geo_freq).fillna(0)

    # prefix4_density
    if "geohash_prefix4" not in train.columns:
        train["geohash_prefix4"] = train["geohash"].str[:4]
    if "geohash_prefix4" not in test.columns:
        test["geohash_prefix4"] = test["geohash"].str[:4]

    prefix4_freq = train["geohash_prefix4"].value_counts().to_dict()
    train["prefix4_density"] = train["geohash_prefix4"].map(prefix4_freq)
    test["prefix4_density"] = test["geohash_prefix4"].map(prefix4_freq).fillna(0)

    # ------- Label encoding for tree models
    for col in ("geohash", "geohash_prefix4", "geohash_prefix5"):
        if col not in train.columns:
            if col == "geohash_prefix4":
                train[col] = train["geohash"].str[:4]
                test[col] = test["geohash"].str[:4]
            elif col == "geohash_prefix5":
                train[col] = train["geohash"].str[:5]
                test[col] = test["geohash"].str[:5]

        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        combined = pd.concat([train[[col]], test[[col]]], ignore_index=True)
        oe.fit(combined)
        train[f"{col}_label"] = oe.transform(train[[col]]).astype(int).flatten()
        test[f"{col}_label"]  = oe.transform(test[[col]]).astype(int).flatten()

    return train, test


def _smooth_target_encode_fold(
    train_ref: pd.DataFrame,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    target: str = "demand",
    smoothing: float = TARGET_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bayesian-smoothed target encoding for *group_col* computed from *train_ref*."""
    global_mean = train_ref[target].mean()
    agg = train_ref.groupby(group_col)[target].agg(["mean", "count"])
    agg["smooth_mean"] = (
        agg["count"] * agg["mean"] + smoothing * global_mean
    ) / (agg["count"] + smoothing)

    col_name = f"{group_col}_target_enc"
    
    # Map to all dataframes
    train_fold[col_name] = train_fold[group_col].map(agg["smooth_mean"]).fillna(global_mean)
    val_fold[col_name] = val_fold[group_col].map(agg["smooth_mean"]).fillna(global_mean)
    test_df[col_name] = test_df[group_col].map(agg["smooth_mean"]).fillna(global_mean)

    return train_fold, val_fold, test_df


def _oof_target_encode(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    n_inner: int = 5,
    smoothing: float = TARGET_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Out-of-fold target encoding — training rows encoded without self-leak.

    Each training row's encoding is computed from inner KFold splits so that
    its own ``demand`` is excluded from its own group mean. Val and test rows
    use the full-train aggregate (no inner fold for held-out data).
    """
    global_mean = train_fold["demand"].mean()
    oof = np.full(len(train_fold), np.nan)
    # Cap inner splits at the train fold size (small test fixtures may have
    # fewer rows than n_inner).
    n_inner = max(2, min(n_inner, len(train_fold)))
    kf = _KFold(n_splits=n_inner, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(train_fold):
        agg = (train_fold.iloc[tr_idx]
               .groupby(group_col)["demand"]
               .agg(["mean", "count"]))
        agg["smooth"] = (
            (agg["mean"] * agg["count"] + global_mean * smoothing)
            / (agg["count"] + smoothing)
        )
        oof[va_idx] = (
            train_fold.iloc[va_idx][group_col]
            .map(agg["smooth"]).fillna(global_mean).values
        )

    full_agg = (train_fold.groupby(group_col)["demand"]
                .agg(["mean", "count"]))
    full_agg["smooth"] = (
        (full_agg["mean"] * full_agg["count"] + global_mean * smoothing)
        / (full_agg["count"] + smoothing)
    )

    col_name = f"{group_col}_target_enc"
    train_fold = train_fold.copy()
    train_fold[col_name] = oof

    val_fold = val_fold.copy()
    val_fold[col_name] = (
        val_fold[group_col].map(full_agg["smooth"]).fillna(global_mean).values
    )

    test_df = test_df.copy()
    test_df[col_name] = (
        test_df[group_col].map(full_agg["smooth"]).fillna(global_mean).values
    )
    return train_fold, val_fold, test_df


def compute_target_dependent_encodings(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    test_df: pd.DataFrame,
    smoothing: float = TARGET_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute and map target-dependent encodings & demand statistics using train_fold only."""
    train_fold = train_fold.copy()
    val_fold = val_fold.copy()
    test_df = test_df.copy()

    # Pre-generate geo_time if not present
    for df in (train_fold, val_fold, test_df):
        if "geo_time" not in df.columns:
            df["geo_time"] = df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)

    # ------- Target encoding (OOF on train to prevent self-leak)
    for col in ("geohash", "geohash_prefix4", "geohash_prefix5", "geo_time", "Weather", "RoadType"):
        train_fold, val_fold, test_df = _oof_target_encode(
            train_fold, val_fold, test_df, col, smoothing=smoothing
        )

    # ------- Geohash demand statistics
    geo_stats = train_fold.groupby("geohash")["demand"].agg(["mean", "std", "median", "min", "max"])
    geo_stats.columns = [
        "geo_demand_mean", "geo_demand_std", "geo_demand_median",
        "geo_demand_min", "geo_demand_max",
    ]
    
    # Add rank and quantile based on mean demand
    geo_stats["geo_demand_rank"] = geo_stats["geo_demand_mean"].rank()
    geo_stats["geo_demand_quantile"] = (
        pd.qcut(geo_stats["geo_demand_mean"], q=4, labels=False, duplicates="drop")
        .fillna(-1)
        .astype(int)
    )

    global_median_stats = geo_stats.median().to_dict()
    for col in geo_stats.columns:
        fallback = global_median_stats.get(col, 0.0)
        train_fold[col] = train_fold["geohash"].map(geo_stats[col]).fillna(fallback)
        val_fold[col] = val_fold["geohash"].map(geo_stats[col]).fillna(fallback)
        test_df[col] = test_df["geohash"].map(geo_stats[col]).fillna(fallback)

    # ------- Time-slot demand statistics
    time_stats = train_fold.groupby("time_slot")["demand"].agg(["mean", "std", "median"])
    time_stats.columns = ["timeslot_demand_mean", "timeslot_demand_std", "timeslot_demand_median"]
    global_median_time = time_stats.median().to_dict()
    for col in time_stats.columns:
        fallback = global_median_time.get(col, 0.0)
        train_fold[col] = train_fold["time_slot"].map(time_stats[col]).fillna(fallback)
        val_fold[col] = val_fold["time_slot"].map(time_stats[col]).fillna(fallback)
        test_df[col] = test_df["time_slot"].map(time_stats[col]).fillna(fallback)

    # ------- (Weather, time_slot) stats — vectorised via merge (P4)
    ws_stats = (
        train_fold.groupby(["Weather", "time_slot"])["demand"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "weather_slot_mean", "std": "weather_slot_std"})
    )

    global_ws_mean = ws_stats["weather_slot_mean"].mean() if not ws_stats.empty else 0.0
    global_ws_std = ws_stats["weather_slot_std"].mean() if not ws_stats.empty else 0.0

    ws_merge_cols = ["Weather", "time_slot", "weather_slot_mean", "weather_slot_std"]
    train_fold = train_fold.merge(ws_stats[ws_merge_cols],
                                  on=["Weather", "time_slot"], how="left")
    val_fold = val_fold.merge(ws_stats[ws_merge_cols],
                              on=["Weather", "time_slot"], how="left")
    test_df = test_df.merge(ws_stats[ws_merge_cols],
                            on=["Weather", "time_slot"], how="left")
    train_fold["weather_slot_mean"] = train_fold["weather_slot_mean"].fillna(global_ws_mean)
    train_fold["weather_slot_std"] = train_fold["weather_slot_std"].fillna(global_ws_std)
    val_fold["weather_slot_mean"] = val_fold["weather_slot_mean"].fillna(global_ws_mean)
    val_fold["weather_slot_std"] = val_fold["weather_slot_std"].fillna(global_ws_std)
    test_df["weather_slot_mean"] = test_df["weather_slot_mean"].fillna(global_ws_mean)
    test_df["weather_slot_std"] = test_df["weather_slot_std"].fillna(global_ws_std)

    return train_fold, val_fold, test_df
